# Backend &mdash; RAG Document Chat

FastAPI service for document ingestion, semantic retrieval, and grounded generation with Anthropic Claude.

> See the [project README](../README.md) for the full pitch and the [architecture doc](../docs/architecture.md) for design rationale.

## Setup

Requires [uv](https://github.com/astral-sh/uv) (it will install Python 3.11+ automatically if needed).

```bash
cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY
uv sync
uv run uvicorn app.main:app --reload
```

API will be available at <http://localhost:8000> with interactive docs at <http://localhost:8000/docs>.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Status, model name, indexed chunk count |
| `POST` | `/upload` | Multipart upload &rarr; chunk &rarr; embed &rarr; store |
| `POST` | `/chat` | `{question}` &rarr; `{answer, sources[]}` |
| `GET` | `/documents` | Per-source chunk counts |
| `DELETE` | `/documents` | Clear the entire collection |

## Layout

```
app/
├── main.py          # FastAPI app + routes + lifespan
├── ingest.py        # parse + chunk + embed + store
├── chunking.py      # sentence-aware token chunker
├── embeddings.py    # sentence-transformers singleton
├── vector_store.py  # ChromaDB persistent client
├── retrieve.py      # semantic search → scored chunks
├── generate.py      # grounded Claude prompting
├── schemas.py       # Pydantic request/response models
└── config.py        # pydantic-settings env loader
tests/               # 25 tests (4 hit the live Claude API and skip if no key)
```

## Tests

```bash
uv run pytest -v
```

Live tests skip automatically when `ANTHROPIC_API_KEY` is unset, so the suite stays green for fresh clones. To exercise everything:

```bash
ANTHROPIC_API_KEY=sk-... uv run pytest -v
```
