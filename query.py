from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

load_dotenv()
user=input("Enter your query: ")
embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=os.environ.get("HF_API_KEY"),
    model=os.environ.get("HF_EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2")
)

query_embedding = embeddings.embed_query(user)

pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pinecone_index_name = os.environ.get("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)
results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

print("Top 5 similar documents:")
print("\n".join([result["metadata"]["text"] for result in results["matches"]]))

