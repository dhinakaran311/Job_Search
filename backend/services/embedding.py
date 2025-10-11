"""
Embedding service using sentence-transformers.

Provides:
- get_embedding_model(model_name=None) -> SentenceTransformer singleton
- embed_text(text) -> List[float]
- embed_texts(list_of_texts) -> List[List[float]]

Default model is sentence-transformers/all-MiniLM-L6-v2 (384 dims).
"""
import os
import warnings
from typing import List
from threading import Lock

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None

_MODEL_SINGLETON = {"model": None}
_MODEL_LOCK = Lock()

DEFAULT_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_embedding_model(model_name: str = None):
    """Return a singleton SentenceTransformer model instance."""
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers not installed. Run pip install sentence-transformers")

    model_name = model_name or DEFAULT_MODEL
    with _MODEL_LOCK:
        if _MODEL_SINGLETON["model"] is None:
            _MODEL_SINGLETON["model"] = SentenceTransformer(model_name)
        return _MODEL_SINGLETON["model"]


def embed_text(text: str, model_name: str = None) -> List[float]:
    """Embed a single text and return a 1-D float list. normalize_embeddings=True so cosine works."""
    model = get_embedding_model(model_name)
    if isinstance(text, str):
        embeddings = model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        return embeddings.tolist()
    else:
        raise ValueError("text must be a string")


def embed_texts(texts: List[str], model_name: str = None) -> List[List[float]]:
    """Embed a list of texts and return list of embedding lists."""
    model = get_embedding_model(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    # embeddings is numpy array shape (n, dim)
    return [vec.tolist() for vec in embeddings]
