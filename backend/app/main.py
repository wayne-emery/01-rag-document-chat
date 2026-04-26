"""FastAPI entrypoint.

Endpoints:
  GET  /health          -> liveness check
  POST /upload          -> ingest a PDF or text/markdown file
  POST /chat            -> ask a question, get a grounded answer + sources
  GET  /documents       -> list ingested source filenames (handy for the UI)
  DELETE /documents     -> wipe the vector store (handy for demo / dev)

Static OpenAPI docs are served at /docs.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.generate import generate_answer
from app.ingest import ingest_file
from app.retrieve import retrieve
from app.schemas import ChatRequest, ChatResponse, IngestResult
from app.vector_store import get_collection, reset_collection

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    get_collection()
    try:
        yield
    finally:
        shutil.rmtree(Path(tempfile.gettempdir()) / "rag_uploads", ignore_errors=True)


app = FastAPI(
    title="RAG Document Chat",
    description="Upload documents and chat with them via retrieval-augmented generation.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str | int]:
    """Liveness check + a small sanity stat (chunks indexed)."""
    return {
        "status": "ok",
        "model": settings.claude_model,
        "embedding_model": settings.embedding_model,
        "chunks_indexed": get_collection().count(),
    }


@app.post("/upload", response_model=IngestResult)
async def upload(file: UploadFile = File(...)) -> IngestResult:
    """Accept a PDF/TXT/MD file, ingest it into the vector store, return a summary."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="filename is required")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type {ext!r}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        size = 0
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large; max is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                )
            tmp.write(chunk)
        tmp_path = Path(tmp.name)

    try:
        result = ingest_file(tmp_path, source_name=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=415, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest: {e}") from e
    finally:
        tmp_path.unlink(missing_ok=True)

    if result.chunks_created == 0:
        raise HTTPException(
            status_code=422,
            detail="No extractable text found in the uploaded file.",
        )
    return result


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Run retrieval + Claude generation against the ingested corpus."""
    if get_collection().count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet. POST a file to /upload first.",
        )

    chunks = retrieve(req.question, top_k=req.top_k)
    try:
        answer = generate_answer(req.question, chunks)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return ChatResponse(answer=answer, sources=chunks)


@app.get("/documents")
def list_documents() -> dict[str, list[dict[str, int | str]]]:
    """Return one entry per unique source file currently in the vector store."""
    collection = get_collection()
    if collection.count() == 0:
        return {"documents": []}

    data = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in data["metadatas"]:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    docs = [{"source": src, "chunks": n} for src, n in sorted(counts.items())]
    return {"documents": docs}


@app.delete("/documents")
def clear_documents() -> dict[str, str]:
    """Reset the vector store. Intended for development / demo resets."""
    reset_collection()
    return {"status": "cleared"}

