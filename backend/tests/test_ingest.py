"""End-to-end tests for the ingestion pipeline.

We point Chroma at a temp directory per test session so we don't pollute the real
persistent store.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used-in-ingest")


@pytest.fixture(scope="session", autouse=True)
def _isolate_chroma(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Redirect Chroma persistence to a per-session temp dir before settings is imported."""
    tmp = tmp_path_factory.mktemp("chroma_test")
    os.environ["CHROMA_PERSIST_DIR"] = str(tmp)
    os.environ["CHROMA_COLLECTION"] = "test_documents"


@pytest.fixture()
def fresh_collection():
    from app.vector_store import reset_collection

    return reset_collection()


@pytest.fixture()
def sample_text_file(tmp_path: Path) -> Path:
    """A short text file with multiple sentences to exercise chunking."""
    content = (
        "Retrieval-augmented generation combines a retrieval step with a generative model. "
        "First, relevant documents are retrieved from a knowledge base using semantic search. "
        "Then, those documents are passed as context to a large language model. "
        "This grounds the model's response in real source material. "
        "It dramatically reduces hallucination and improves factual accuracy. "
        "ChromaDB is a popular open-source vector database designed for this use case. "
        "Sentence-transformers provides a wide selection of pre-trained embedding models. "
        "The all-MiniLM-L6-v2 model is small and fast while still producing useful embeddings. "
        "It outputs 384-dimensional vectors and runs comfortably on a laptop CPU. "
        "Anthropic's Claude family of models excels at instruction following and grounded answers."
    )
    p = tmp_path / "rag_intro.txt"
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture()
def sample_pdf_file(tmp_path: Path) -> Path:
    """A two-page PDF with distinct content per page."""
    pypdf = pytest.importorskip("pypdf")
    from pypdf import PdfWriter
    from pypdf.generic import (
        ArrayObject,
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
        NumberObject,
    )

    def make_page_stream(text: str) -> DecodedStreamObject:
        commands = (
            "BT /F1 12 Tf 72 720 Td "
            f"({text.replace('(', '\\(').replace(')', '\\)')}) Tj ET"
        ).encode("latin-1")
        s = DecodedStreamObject()
        s.set_data(commands)
        return s

    writer = PdfWriter()
    for content_text in [
        "Page one introduces vector databases and Chroma.",
        "Page two covers Anthropic Claude and prompt design.",
    ]:
        page = writer.add_blank_page(width=612, height=792)
        font = DictionaryObject()
        font[NameObject("/Type")] = NameObject("/Font")
        font[NameObject("/Subtype")] = NameObject("/Type1")
        font[NameObject("/BaseFont")] = NameObject("/Helvetica")
        font_ref = writer._add_object(font)

        resources = DictionaryObject()
        font_dict = DictionaryObject()
        font_dict[NameObject("/F1")] = font_ref
        resources[NameObject("/Font")] = font_dict
        page[NameObject("/Resources")] = resources
        page[NameObject("/MediaBox")] = ArrayObject(
            [NumberObject(0), NumberObject(0), NumberObject(612), NumberObject(792)]
        )

        stream = make_page_stream(content_text)
        page[NameObject("/Contents")] = writer._add_object(stream)

    out = tmp_path / "two_pages.pdf"
    with out.open("wb") as fh:
        writer.write(fh)
    return out


def test_ingest_text_file_creates_chunks(sample_text_file: Path, fresh_collection):
    from app.ingest import ingest_file

    result = ingest_file(sample_text_file)

    assert result.file_type == "txt"
    assert result.source == "rag_intro.txt"
    assert result.pages is None
    assert result.chunks_created >= 1
    assert result.total_tokens > 0

    assert fresh_collection.count() == result.chunks_created

    peek = fresh_collection.peek(limit=1)
    assert len(peek["embeddings"]) == 1
    assert len(peek["embeddings"][0]) == 384
    meta = peek["metadatas"][0]
    assert meta["source"] == "rag_intro.txt"
    assert meta["page"] == -1
    assert "chunk_index" in meta


def test_ingest_pdf_records_page_numbers(sample_pdf_file: Path, fresh_collection):
    from app.ingest import ingest_file

    result = ingest_file(sample_pdf_file)

    assert result.file_type == "pdf"
    assert result.pages == 2
    assert result.chunks_created >= 2

    all_data = fresh_collection.get(include=["metadatas", "documents"])
    pages_seen = {m["page"] for m in all_data["metadatas"]}
    assert pages_seen == {1, 2}

    by_page: dict[int, str] = {}
    for doc, meta in zip(all_data["documents"], all_data["metadatas"], strict=True):
        by_page[meta["page"]] = doc

    assert "Chroma" in by_page[1]
    assert "Claude" in by_page[2]


def test_chunking_respects_max_token_limit():
    from app.chunking import chunk_text
    from app.embeddings import get_embedder

    long_text = ". ".join([f"This is sentence number {i}" for i in range(200)]) + "."
    tokenizer = get_embedder().tokenizer

    chunks = chunk_text(long_text, tokenizer, chunk_size=64, chunk_overlap=10)

    assert len(chunks) >= 2
    for ch in chunks:
        assert ch.token_count <= 64, f"chunk exceeded budget: {ch.token_count}"
