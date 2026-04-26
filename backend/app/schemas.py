"""Pydantic models for request/response payloads and internal data structures."""

from typing import Literal

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata stored alongside each chunk in the vector store."""

    source: str = Field(..., description="Original filename")
    page: int | None = Field(None, description="1-indexed page number (PDFs only)")
    chunk_index: int = Field(..., description="Sequential chunk index within the document")


class IngestResult(BaseModel):
    """Returned after ingesting a single file."""

    source: str
    file_type: Literal["pdf", "txt"]
    pages: int | None = None
    chunks_created: int
    total_tokens: int


class RetrievedChunk(BaseModel):
    """A chunk returned by semantic search, with its similarity score."""

    text: str
    metadata: ChunkMetadata
    score: float = Field(..., description="Similarity score (higher = more relevant)")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(None, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    sources: list[RetrievedChunk]
