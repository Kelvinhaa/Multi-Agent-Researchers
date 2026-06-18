import os
from rag.embedder import embeddings
from dotenv import load_dotenv
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

def retrieve(query: str) -> list[dict]
    vector = embeddings.embed_query(query)