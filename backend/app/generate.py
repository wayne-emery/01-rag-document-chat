"""LLM generation step: build a grounded prompt from retrieved chunks and ask Claude.

The prompt instructs the model to:
  1. Only answer from the provided context.
  2. Say "I don't know" when the context doesn't contain the answer.
  3. Cite sources using the [n] markers we attach to each chunk.

This pattern (numbered context blocks + explicit citation instructions) is a
standard, reliable way to keep RAG answers grounded and verifiable.
"""

from __future__ import annotations

from functools import lru_cache

from anthropic import Anthropic

from app.config import settings
from app.schemas import RetrievedChunk

SYSTEM_PROMPT = """You are a precise question-answering assistant.

You will be given numbered excerpts from one or more source documents, followed \
by a user question. Follow these rules strictly:

1. Answer ONLY using information from the provided excerpts. Do not use outside \
knowledge.
2. If the excerpts do not contain enough information to answer, reply exactly: \
"I don't have enough information in the provided documents to answer that."
3. When you state a fact, cite the supporting excerpt(s) using the bracket \
notation [1], [2], etc. that appears at the start of each excerpt.
4. Be concise. Prefer 1-3 short paragraphs.
5. Do not fabricate citation numbers. Only cite excerpts that were actually \
provided to you."""


@lru_cache(maxsize=1)
def get_anthropic_client() -> Anthropic:
    """Lazily construct the Anthropic SDK client (singleton per process)."""
    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to backend/.env "
            "(see .env.example for the format)."
        )
    return Anthropic(api_key=settings.anthropic_api_key)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as numbered excerpts the model can cite."""
    lines: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        page_part = f", p.{ch.metadata.page}" if ch.metadata.page is not None else ""
        header = f"[{i}] (source: {ch.metadata.source}{page_part})"
        lines.append(f"{header}\n{ch.text.strip()}")
    return "\n\n".join(lines)


def build_user_message(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return (
            f"No source excerpts were retrieved for this question.\n\n"
            f"Question: {question}"
        )
    return (
        f"Source excerpts:\n\n{format_context(chunks)}\n\n"
        f"---\n\nQuestion: {question}"
    )


def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    client: Anthropic | None = None,
    max_tokens: int = 700,
) -> str:
    """Call Claude with a grounded prompt and return the answer text.

    Args:
        question: The user's question.
        chunks: Retrieved context, in priority order. May be empty (the model
            will be told no excerpts were available and should refuse).
        client: Inject an Anthropic client for testing. Defaults to the singleton.
        max_tokens: Hard cap on the model's output. 700 is plenty for a few
            grounded paragraphs while keeping costs predictable.
    """
    if not question or not question.strip():
        raise ValueError("question must be non-empty")

    api = client or get_anthropic_client()
    user_message = build_user_message(question, chunks)

    response = api.messages.create(
        model=settings.claude_model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    return "".join(parts).strip()
