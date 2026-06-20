import os

from dotenv import load_dotenv
from pinecone import Pinecone

from rag.embedder import EMBED_MODEL, FIELD_MAP

load_dotenv(override=True)

CLOUD = "aws"
REGION = "us-east-1"

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


def ensure_index():
    """Create the integrated-inference index if it doesn't exist yet, then return it."""
    index_name = os.getenv("PINECONE_INDEX")
    if not pc.has_index(index_name):
        pc.create_index_for_model(
            name=index_name,
            cloud=CLOUD,
            region=REGION,
            embed={"model": EMBED_MODEL, "field_map": FIELD_MAP},
        )
    return pc.Index(index_name)


def get_index():
    """Get a handle to the index for read paths. Assumes it already exists."""
    return pc.Index(os.getenv("PINECONE_INDEX"))
