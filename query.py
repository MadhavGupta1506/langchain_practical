from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# Load env
load_dotenv()

# ENV variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
HF_API_KEY = os.environ.get("HF_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Initialize embeddings
embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=HF_API_KEY,
    model=os.environ.get(
        "HF_EMBED_MODEL",
        "sentence-transformers/all-mpnet-base-v2"
    )
)

# Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    timeout=5,
    max_retries=2,
)

# System prompt (ONLY behavior, no context)
messages = [
    {
        "role": "system",
        "content": """You are an experienced Data Structures and Algorithms (DSA) professor.

Use ONLY the provided context to answer the question.
If the answer is not present in the context, say:
"I could not find this in the provided material."

Explain clearly with:
- Intuition
- Step-by-step explanation
- Time and space complexity
- When to use
- Common mistakes

Tone: Patient, clear, and like a good classroom professor.
"""
    }
]

# Chat loop
while True:
    user = input("Enter your query (type 'exit' to quit): ")
    if user.lower() == "exit":
        break

    # Embed query
    query_embedding = embeddings.embed_query(user)

    # Retrieve context
    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    context = "\n\n---\n\n".join(
        [r["metadata"]["text"] for r in results["matches"]]
    )

    # Add user message WITH context
    messages.append({
        "role": "user",
        "content": f"Question:\n{user}\n\nContext:\n{context}"
    })

    # LLM response
    ai_msg = llm.invoke(messages)
    print("\n", ai_msg.content, "\n")

    # Save assistant reply
    messages.append({
        "role": "assistant",
        "content": ai_msg.content
    })