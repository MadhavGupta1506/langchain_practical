import os
import asyncio
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnableConfig

# Load env
load_dotenv()
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")
hf_api_key = os.environ.get("HF_API_KEY")
hf_model = os.environ.get("HF_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")

if not pinecone_api_key:
    raise ValueError("Missing PINECONE_API_KEY in environment")
if not pinecone_index_name:
    raise ValueError("Missing PINECONE_INDEX_NAME in environment")
if not hf_api_key:
    raise ValueError("Missing HF_API_KEY in environment")

# Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
index_description = pc.describe_index(name=pinecone_index_name)

embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=hf_api_key,
    model=hf_model,
)
print(f"Embeddings model loaded: {hf_model}")

index_dimension = (
    index_description.dimension
    if hasattr(index_description, "dimension")
    else index_description.get("dimension")
)
embedding_dimension = len(embeddings.embed_query("dimension check"))

if index_dimension != embedding_dimension:
    raise ValueError(
        f"Embedding dimension mismatch: model outputs {embedding_dimension}, "
        f"but Pinecone index '{pinecone_index_name}' expects {index_dimension}. "
        "Set HF_EMBED_MODEL to a model with matching dimension or create a matching index."
    )
# Load PDF
file_path = "Dsa.pdf"
loader = PyPDFLoader(file_path)
documents = loader.load()
print(f"{len(documents)} pages loaded from PDF")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

print(len(texts), "chunks created")
# Create vector store (empty)
vector_store = PineconeVectorStore(
    index_name=pinecone_index_name,
    embedding=embeddings
)
# Async ingestion with concurrency
async def ingest():
    config = RunnableConfig(max_concurrency=5)
    await vector_store.aadd_documents(texts, config=config)
asyncio.run(ingest())
print("Ingestion complete")
