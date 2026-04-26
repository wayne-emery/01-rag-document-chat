"""Document ingestion pipeline: parse -> chunk -> embed -> persist."""

from __future__ import annotations

import uuid
from pathlib import Path

from pypdf import PdfReader

from app.chunking import Chunk, chunk_text
from app.config import settings
from app.embeddings import embed_texts, get_embedder
from app.schemas import IngestResult
from app.vector_store import get_collection


def _extract_pages_from_pdf(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number_1_indexed, text), ...] for a PDF."""
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append((i, text))
    return pages


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _detect_file_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext in {".txt", ".md"}:
        return "txt"
    raise ValueError(f"Unsupported file type: {ext}")


def ingest_file(path: Path, source_name: str | None = None) -> IngestResult:
    """Ingest a single file into the vector store and return a summary.

    Args:
        path: Path to the PDF or text file on disk.
        source_name: Display name to record (defaults to the file name). Useful when
            ingesting an upload from a temp file but you want the original filename
            in citations.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    file_type = _detect_file_type(path)
    source = source_name or path.name

    tokenizer = get_embedder().tokenizer

    pieces: list[tuple[Chunk, int | None]] = []

    if file_type == "pdf":
        pages = _extract_pages_from_pdf(path)
        for page_num, page_text in pages:
            for ch in chunk_text(page_text, tokenizer, settings.chunk_size, settings.chunk_overlap):
                pieces.append((ch, page_num))
        page_count = len(pages)
    else:
        text = _read_text_file(path)
        for ch in chunk_text(text, tokenizer, settings.chunk_size, settings.chunk_overlap):
            pieces.append((ch, None))
        page_count = None

    if not pieces:
        return IngestResult(
            source=source,
            file_type=file_type,
            pages=page_count,
            chunks_created=0,
            total_tokens=0,
        )

    texts = [c.text for c, _ in pieces]
    embeddings = embed_texts(texts)

    collection = get_collection()
    ids = [str(uuid.uuid4()) for _ in pieces]
    metadatas = [
        {
            "source": source,
            "page": page if page is not None else -1,
            "chunk_index": i,
        }
        for i, (_, page) in enumerate(pieces)
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    return IngestResult(
        source=source,
        file_type=file_type,
        pages=page_count,
        chunks_created=len(pieces),
        total_tokens=sum(c.token_count for c, _ in pieces),
    )
