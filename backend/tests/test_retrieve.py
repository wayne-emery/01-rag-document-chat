"""Tests for the retrieval pipeline.

These reuse the chroma isolation fixture from `test_ingest.py` (autouse, session-scoped).
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def populated_collection(tmp_path: Path):
    """Ingest a small, topic-diverse corpus and return the collection."""
    from app.ingest import ingest_file
    from app.vector_store import reset_collection

    reset_collection()

    docs = {
        "ml.txt": (
            "Gradient boosting builds an ensemble of decision trees sequentially. "
            "Each new tree fits the residual errors of the previous ensemble. "
            "XGBoost and LightGBM are popular high-performance implementations."
        ),
        "cooking.txt": (
            "Caramelizing onions requires low heat and patience over many minutes. "
            "Stir occasionally and add a pinch of salt to draw out moisture. "
            "The natural sugars brown gradually, producing a deep sweet flavor."
        ),
        "astronomy.txt": (
            "A neutron star forms when a massive star collapses at the end of its life. "
            "These objects are extraordinarily dense and rotate very rapidly. "
            "Pulsars are neutron stars that emit beams of electromagnetic radiation."
        ),
    }
    for name, text in docs.items():
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        ingest_file(path)

    from app.vector_store import get_collection

    return get_collection()


def test_retrieve_returns_top_k_results(populated_collection):
    from app.retrieve import retrieve

    results = retrieve("How do gradient boosted trees work?", top_k=3)

    assert 1 <= len(results) <= 3
    assert all(0.0 <= r.score <= 1.0 + 1e-6 for r in results)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "results must be sorted by score desc"


def test_retrieve_finds_topically_relevant_chunk(populated_collection):
    from app.retrieve import retrieve

    results = retrieve("how do you cook onions until brown and sweet?", top_k=1)
    assert results, "expected at least one result"
    top = results[0]

    assert top.metadata.source == "cooking.txt"
    assert "onion" in top.text.lower() or "caramel" in top.text.lower()


def test_retrieve_metadata_roundtrips(populated_collection):
    from app.retrieve import retrieve

    results = retrieve("astronomy and stars", top_k=2)
    assert results
    for r in results:
        assert r.metadata.source.endswith(".txt")
        assert r.metadata.page is None
        assert isinstance(r.metadata.chunk_index, int)
        assert r.metadata.chunk_index >= 0


def test_retrieve_empty_query_returns_empty():
    from app.retrieve import retrieve

    assert retrieve("") == []
    assert retrieve("   ") == []


def test_retrieve_handles_empty_collection():
    from app.retrieve import retrieve
    from app.vector_store import reset_collection

    reset_collection()
    assert retrieve("anything", top_k=5) == []
