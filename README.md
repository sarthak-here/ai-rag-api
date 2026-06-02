# RAG Document Intelligence API

A **production-grade Retrieval-Augmented Generation (RAG) API** built with FastAPI, ChromaDB, and Anthropic Claude. Upload documents, ask natural-language questions, and receive grounded AI-generated answers with cited sources.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        HTTP Client                        │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                     FastAPI Application                   │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  API Layer  (routes, request/response schemas)      │ │
│  └──────────────────────┬──────────────────────────────┘ │
│  ┌───────────────────────▼────────────────────────────┐  │
│  │  Service Layer  (DocumentService · RAGService)     │  │
│  └──────────┬───────────────────────────┬────────────┘  │
│  ┌──────────▼──────────┐   ┌────────────▼────────────┐  │
│  │  SQLite (metadata)  │   │  ChromaDB (embeddings)  │  │
│  │  SQLAlchemy 2 async │   │  Cosine similarity ANN  │  │
│  └─────────────────────┘   └────────────┬────────────┘  │
│                                          │               │
│                             ┌────────────▼────────────┐  │
│                             │   Anthropic Claude API  │  │
│                             │   (answer generation)   │  │
│                             └─────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI 0.115 + Uvicorn |
| ORM | SQLAlchemy 2.0 async (aiosqlite) |
| Vector Store | ChromaDB (cosine similarity, HNSW index) |
| AI / LLM | Anthropic Claude (configurable model) |
| Validation | Pydantic v2 + pydantic-settings |
| Logging | structlog (JSON in prod, pretty-print in dev) |
| Linting | ruff (replaces black + isort + flake8) |
| Type checking | mypy strict mode |
| Testing | pytest + pytest-asyncio |
| Packaging | uv + hatchling |
| Containers | Docker (multi-stage) + docker-compose |
| CI | GitHub Actions |

## Project Structure

```
src/rag_api/
├── api/v1/endpoints/   ← thin route handlers (no business logic)
│   ├── documents.py    ← CRUD for documents
│   ├── queries.py      ← RAG ask + summarize
│   └── health.py       ← liveness probe
├── core/               ← cross-cutting concerns
│   ├── config.py       ← pydantic-settings (12-factor app)
│   ├── exceptions.py   ← typed HTTP-aware exceptions
│   └── logging.py      ← structlog setup
├── domain/
│   ├── schemas/        ← Pydantic request/response models
│   └── services/       ← business logic (DocumentService, RAGService)
└── infrastructure/
    ├── ai/             ← Anthropic client wrapper
    ├── db/             ← SQLAlchemy models + repository pattern
    └── vector_store/   ← ChromaDB wrapper
```

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An [Anthropic API key](https://console.anthropic.com/)

### Installation

```bash
# Clone the repo
git clone https://github.com/sarthak-here/ai-rag-api.git
cd ai-rag-api

# Create a virtual environment and install
uv venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

### Run

```bash
uvicorn rag_api.main:app --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

### Docker

```bash
docker compose up --build
```

## API Reference

### Documents

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/documents` | Upload & index a document |
| `GET` | `/api/v1/documents` | List all documents (paginated) |
| `GET` | `/api/v1/documents/{id}` | Fetch document detail |
| `DELETE` | `/api/v1/documents/{id}` | Delete document + embeddings |

### RAG Queries

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/queries/ask` | Ask a question (returns answer + sources) |
| `POST` | `/api/v1/queries/summarize` | AI summary of a document |

### Example

```bash
# 1. Index a document
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "FastAPI Best Practices",
    "content": "FastAPI is a modern, fast web framework for Python...",
    "source": "https://fastapi.tiangolo.com"
  }'

# 2. Ask a question
curl -X POST http://localhost:8000/api/v1/queries/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main advantages of FastAPI?", "top_k": 3}'
```

Response:
```json
{
  "question": "What are the main advantages of FastAPI?",
  "answer": "Based on the provided context, FastAPI's main advantages are...",
  "sources": [
    {
      "document_id": "abc-123",
      "chunk_id": "chunk-0",
      "content": "FastAPI is a modern, fast web framework...",
      "relevance_score": 0.923
    }
  ]
}
```

## Development

```bash
# Run tests with coverage
pytest

# Lint
ruff check src tests

# Format
ruff format src tests

# Type-check
mypy src

# Install pre-commit hooks
pre-commit install
```

## Design Decisions

**Repository pattern** — all database access goes through typed repository classes, making it trivial to swap SQLite for PostgreSQL without touching service logic.

**No ORM leakage into domain** — services and schemas use pure Python objects; ORM models never leave the infrastructure layer.

**Synchronous AI calls** — ChromaDB and Anthropic SDK are synchronous; they're called from sync route handlers or sync service methods, avoiding `asyncio.run_in_executor` complexity while keeping async database I/O.

**Structured logging** — every significant event carries a machine-parseable key=value payload so logs can be indexed by any observability stack (Datadog, Grafana Loki, CloudWatch).

**12-factor config** — all configuration comes from environment variables via pydantic-settings, making the app container-native.
