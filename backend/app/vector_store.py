"""ChromaDB persistent client wrapper.

We bring our own embeddings (sentence-transformers) so we register Chroma with a
no-op embedding function and pass embeddings explicitly on add/query.
"""

from __future__ import annotations

from functools import lru_cache

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from app.config import settings


class _NoopEmbeddingFunction(EmbeddingFunction[Documents]):
    """Required by Chroma but never invoked because we always pass embeddings explicitly."""

    def __init__(self) -> None:
        pass

    def __call__(self, _input: Documents) -> Embeddings:
        raise RuntimeError(
            "Embeddings must be supplied explicitly; the Chroma collection's "
            "embedding function should not be invoked."
        )

    @staticmethod
    def name() -> str:
        return "noop"

    def get_config(self) -> dict:
        return {}

    @classmethod
    def build_from_config(cls, _config: dict) -> "_NoopEmbeddingFunction":
        return cls()


@lru_cache(maxsize=1)
def get_client() -> ClientAPI:
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(settings.chroma_persist_dir))


def get_collection(name: str | None = None) -> Collection:
    """Get-or-create the documents collection.

    Uses cosine distance because our sentence-transformers embeddings are L2-normalized,
    making cosine similarity equivalent to (and faster than) dot product semantics.
    """
    client = get_client()
    return client.get_or_create_collection(
        name=name or settings.chroma_collection,
        embedding_function=_NoopEmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(name: str | None = None) -> Collection:
    """Delete and recreate the collection (useful for tests / fresh starts)."""
    client = get_client()
    target = name or settings.chroma_collection
    try:
        client.delete_collection(target)
    except (ValueError, Exception):
        pass
    return get_collection(target)
