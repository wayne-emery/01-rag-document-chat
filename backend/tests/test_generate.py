"""Tests for the Claude generation step.

Includes both:
  - Mocked tests (always run, no network, no cost)
  - Live integration tests against the real Anthropic API
    (auto-skipped when ANTHROPIC_API_KEY is unset)
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.config import settings
from app.schemas import ChunkMetadata, RetrievedChunk

LIVE_API = bool(os.environ.get("ANTHROPIC_API_KEY") or settings.anthropic_api_key)
live_only = pytest.mark.skipif(
    not LIVE_API, reason="ANTHROPIC_API_KEY not set; skipping live API test"
)


def _chunk(text: str, source: str, page: int | None = None, idx: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        metadata=ChunkMetadata(source=source, page=page, chunk_index=idx),
        score=0.9,
    )


def _fake_client(answer_text: str) -> MagicMock:
    """Build a MagicMock that quacks like the Anthropic SDK client."""
    fake = MagicMock()
    msg = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = answer_text
    msg.content = [block]
    fake.messages.create.return_value = msg
    return fake


def test_format_context_numbers_and_labels_chunks():
    from app.generate import format_context

    chunks = [
        _chunk("Cosine similarity measures the angle between vectors.", "rag.pdf", page=3, idx=0),
        _chunk("Chroma is a local vector database.", "tools.txt", page=None, idx=2),
    ]
    rendered = format_context(chunks)

    assert "[1]" in rendered and "[2]" in rendered
    assert "rag.pdf" in rendered and "p.3" in rendered
    assert "tools.txt" in rendered
    assert "p." not in rendered.split("[2]")[1].split("\n")[0], (
        "page label should be omitted when page is None"
    )


def test_build_user_message_handles_empty_context():
    from app.generate import build_user_message

    msg = build_user_message("What is RAG?", [])
    assert "No source excerpts" in msg
    assert "What is RAG?" in msg


def test_generate_answer_passes_grounded_prompt_to_client():
    from app.generate import SYSTEM_PROMPT, generate_answer

    fake = _fake_client("RAG combines retrieval and generation [1].")
    chunks = [_chunk("RAG = Retrieval-Augmented Generation.", "intro.txt", idx=0)]

    answer = generate_answer("What does RAG stand for?", chunks, client=fake)

    assert answer == "RAG combines retrieval and generation [1]."
    fake.messages.create.assert_called_once()
    kwargs = fake.messages.create.call_args.kwargs
    assert kwargs["model"] == settings.claude_model
    assert kwargs["system"] == SYSTEM_PROMPT
    assert kwargs["max_tokens"] == 700
    user_content = kwargs["messages"][0]["content"]
    assert "RAG = Retrieval-Augmented Generation." in user_content
    assert "What does RAG stand for?" in user_content
    assert "[1]" in user_content


def test_generate_answer_rejects_empty_question():
    from app.generate import generate_answer

    with pytest.raises(ValueError):
        generate_answer("", [], client=_fake_client("ignored"))
    with pytest.raises(ValueError):
        generate_answer("   ", [], client=_fake_client("ignored"))


@live_only
def test_live_claude_answers_from_provided_context():
    """Real call to Claude. Verifies the model uses the context, not its priors."""
    from app.generate import generate_answer

    chunks = [
        _chunk(
            "The internal company codename for the 2027 quarterly retrospective is "
            "'Project Driftwood'. It involves cross-team analysis of velocity metrics.",
            source="internal_memo.txt",
            idx=0,
        )
    ]

    answer = generate_answer("What is the codename for the 2027 quarterly retrospective?", chunks)

    assert "driftwood" in answer.lower(), f"expected grounded answer, got: {answer!r}"
    assert "[1]" in answer, f"expected a citation marker, got: {answer!r}"


@live_only
def test_live_claude_refuses_when_context_lacks_answer():
    """Verify the model honors the 'I don't know' instruction when context is irrelevant."""
    from app.generate import generate_answer

    chunks = [
        _chunk(
            "The mitochondrion is the powerhouse of the cell, generating ATP via "
            "oxidative phosphorylation.",
            source="biology.txt",
            idx=0,
        )
    ]

    answer = generate_answer("Who won the 2024 Super Bowl?", chunks)

    lowered = answer.lower()
    refusal_signal = (
        "don't have enough" in lowered
        or "do not have enough" in lowered
        or "not contain" in lowered
        or "no information" in lowered
    )
    assert refusal_signal, f"expected refusal, got: {answer!r}"


@live_only
def test_live_end_to_end_retrieve_then_generate(tmp_path: Path):
    """Full pipeline smoke test: ingest -> retrieve -> generate, against real Claude."""
    from app.generate import generate_answer
    from app.ingest import ingest_file
    from app.retrieve import retrieve
    from app.vector_store import reset_collection

    reset_collection()

    doc = tmp_path / "facts.txt"
    doc.write_text(
        "The Eiffel Tower is 330 meters tall including its antennas. "
        "It was completed in 1889 for the World's Fair in Paris. "
        "Gustave Eiffel's company designed and built it.",
        encoding="utf-8",
    )
    ingest_file(doc)

    chunks = retrieve("How tall is the Eiffel Tower?", top_k=2)
    assert chunks, "retrieval returned nothing"

    answer = generate_answer("How tall is the Eiffel Tower?", chunks)
    assert "330" in answer, f"expected the height fact in the answer, got: {answer!r}"
