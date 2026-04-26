"""HTTP-level tests for the FastAPI surface."""

from __future__ import annotations

import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import settings

LIVE_API = bool(os.environ.get("ANTHROPIC_API_KEY") or settings.anthropic_api_key)
live_only = pytest.mark.skipif(
    not LIVE_API, reason="ANTHROPIC_API_KEY not set; skipping live API test"
)


@pytest.fixture()
def client():
    from app.main import app
    from app.vector_store import reset_collection

    reset_collection()
    with TestClient(app) as c:
        yield c
    reset_collection()


def _txt_upload(name: str, content: str) -> dict:
    return {"file": (name, BytesIO(content.encode("utf-8")), "text/plain")}


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["chunks_indexed"] == 0
    assert body["model"] == settings.claude_model


def test_upload_then_documents_lists_it(client: TestClient):
    payload = _txt_upload(
        "intro.txt",
        "RAG combines retrieval with generation. ChromaDB is a popular vector store.",
    )
    r = client.post("/upload", files=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "intro.txt"
    assert body["file_type"] == "txt"
    assert body["chunks_created"] >= 1

    r2 = client.get("/documents")
    assert r2.status_code == 200
    docs = r2.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["source"] == "intro.txt"
    assert docs[0]["chunks"] == body["chunks_created"]


def test_upload_rejects_unsupported_extension(client: TestClient):
    files = {"file": ("evil.exe", BytesIO(b"MZ"), "application/octet-stream")}
    r = client.post("/upload", files=files)
    assert r.status_code == 415


def test_upload_rejects_oversized_file(client: TestClient):
    big = b"a" * (11 * 1024 * 1024)
    files = {"file": ("big.txt", BytesIO(big), "text/plain")}
    r = client.post("/upload", files=files)
    assert r.status_code == 413


def test_upload_rejects_empty_text_file(client: TestClient):
    r = client.post("/upload", files=_txt_upload("blank.txt", ""))
    assert r.status_code == 422


def test_chat_requires_uploaded_documents(client: TestClient):
    r = client.post("/chat", json={"question": "Anything?"})
    assert r.status_code == 400
    assert "upload" in r.json()["detail"].lower()


def test_chat_validates_question(client: TestClient):
    client.post(
        "/upload",
        files=_txt_upload("seed.txt", "ChromaDB is a vector database."),
    )
    r = client.post("/chat", json={"question": ""})
    assert r.status_code == 422


def test_chat_with_mocked_claude_returns_answer_and_sources(client: TestClient):
    """Patches the Anthropic client so we can exercise the full /chat plumbing offline."""
    client.post(
        "/upload",
        files=_txt_upload(
            "facts.txt",
            "The Hubble Space Telescope was launched in 1990 by NASA. "
            "It orbits Earth at an altitude of about 540 kilometers.",
        ),
    )

    fake_client = MagicMock()
    fake_msg = MagicMock()
    fake_block = MagicMock()
    fake_block.type = "text"
    fake_block.text = "Hubble was launched in 1990 [1]."
    fake_msg.content = [fake_block]
    fake_client.messages.create.return_value = fake_msg

    with patch("app.main.generate_answer", wraps=lambda q, c: f"MOCKED({q})") as gen_mock:
        r = client.post("/chat", json={"question": "When was Hubble launched?", "top_k": 2})

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "MOCKED(When was Hubble launched?)"
    assert len(body["sources"]) >= 1
    assert all(s["metadata"]["source"] == "facts.txt" for s in body["sources"])
    gen_mock.assert_called_once()


def test_documents_delete_clears_collection(client: TestClient):
    client.post("/upload", files=_txt_upload("a.txt", "Some content here for indexing."))
    assert client.get("/documents").json()["documents"]
    r = client.delete("/documents")
    assert r.status_code == 200
    assert client.get("/documents").json()["documents"] == []


@live_only
def test_live_chat_full_pipeline(client: TestClient):
    """End-to-end: real upload -> real retrieve -> real Claude call via HTTP."""
    client.post(
        "/upload",
        files=_txt_upload(
            "rag_facts.txt",
            "The all-MiniLM-L6-v2 embedding model produces 384-dimensional vectors. "
            "It has a maximum input length of 256 tokens.",
        ),
    )

    r = client.post(
        "/chat",
        json={"question": "How many dimensions do the embedding vectors have?"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "384" in body["answer"], f"unexpected answer: {body['answer']!r}"
    assert body["sources"], "no sources returned"
    assert body["sources"][0]["metadata"]["source"] == "rag_facts.txt"
