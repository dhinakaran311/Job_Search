"""
Chroma DB helper.

Provides:
- init_chroma_client(CHROMA_PATH)
- get_or_create_collection(collection_name, client=None)

Reads CHROMA_PATH and COLLECTION_NAME from environment variables if not passed.
"""
import os
from typing import Optional
import chromadb

def init_chroma_client(chroma_path: Optional[str] = None):
    """
    Initialize and return a persistent Chroma client.
    Creates the directory if missing.
    """
    if chroma_path is None:
        chroma_path = os.environ.get(
            "CHROMA_PATH",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db"))
        )
    os.makedirs(chroma_path, exist_ok=True)

    # New Chroma API (0.5.x)
    client = chromadb.PersistentClient(path=chroma_path)
    return client


def get_or_create_collection(collection_name: Optional[str] = None, client: Optional[object] = None):
    """
    Get or create a collection with HNSW settings suitable for cosine similarity.
    Returns the collection object.
    """
    if client is None:
        client = init_chroma_client()

    collection_name = collection_name or os.environ.get("COLLECTION_NAME", "jobs")
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    return collection
