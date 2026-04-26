"""Semantic retrieval: embed a query and return the top-k most similar chunks."""

from __future__ import annotations

from app.config import settings
from app.embeddings import embed_texts
from app.schemas import ChunkMetadata, RetrievedChunk
from app.vector_store import get_collection


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Return the top-k chunks most similar to `query`, sorted by score (desc).

    Chroma is configured with cosine distance and we store L2-normalized vectors,
    so we convert distance -> similarity via `score = 1 - distance`. This gives
    a value in roughly [0, 1] where 1.0 is a perfect match.
    """
    if not query or not query.strip():
        return []

    k = top_k or settings.top_k
    collection = get_collection()
    if collection.count() == 0:
        return []

    query_vec = embed_texts([query])[0]
    raw = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    docs = raw.get("documents", [[]])[0]
    metas = raw.get("metadatas", [[]])[0]
    dists = raw.get("distances", [[]])[0]

    results: list[RetrievedChunk] = []
    for doc, meta, dist in zip(docs, metas, dists, strict=True):
        page_value = meta.get("page", -1)
        results.append(
            RetrievedChunk(
                text=doc,
                metadata=ChunkMetadata(
                    source=meta["source"],
                    page=None if page_value == -1 else int(page_value),
                    chunk_index=int(meta["chunk_index"]),
                ),
                score=float(1.0 - dist),
            )
        )

    return results
