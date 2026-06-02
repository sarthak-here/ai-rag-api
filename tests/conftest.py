"""
Shared test fixtures.

Strategy:
  - Unit tests use pure in-memory fakes (no I/O).
  - Integration tests spin up a real FastAPI TestClient backed by a temp
    SQLite file, with VectorStore and AIClient replaced by MagicMock fakes
    via FastAPI's app.dependency_overrides — no external services needed.
"""

# Set required env vars BEFORE any project imports so pydantic-settings
# does not raise a validation error when loading Settings at import time.
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-tests")

from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from rag_api.infrastructure.db.base import Base
from rag_api.infrastructure.db.models import (  # noqa: F401 — ensures tables registered
    Chunk,
    Document,
)
from rag_api.infrastructure.db.repositories.document_repository import DocumentRepository

# ---------------------------------------------------------------------------
# In-memory async SQLite (unit tests only)
# ---------------------------------------------------------------------------
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def async_engine() -> AsyncGenerator[AsyncEngine, None]:
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def session(async_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    factory = async_sessionmaker(async_engine, expire_on_commit=False)
    async with factory() as s:
        yield s


@pytest_asyncio.fixture
async def document_repo(session: AsyncSession) -> DocumentRepository:
    return DocumentRepository(session)


# ---------------------------------------------------------------------------
# Integration: full app with dependency overrides
# ---------------------------------------------------------------------------
@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """
    Full TestClient with:
      - SQLite backed by a tmp file (isolated per test, no state leak)
      - VectorStore and AIClient replaced by fakes
    """
    from rag_api.api.deps import get_ai_client, get_vector_store
    from rag_api.core.config import get_settings
    from rag_api.main import create_app

    # Point the app at a fresh temp database
    db_path = tmp_path / "test_rag.db"
    os.environ["DATABASE_URL"] = f"sqlite+aiosqlite:///{db_path}"
    get_settings.cache_clear()

    app = create_app()

    # FastAPI-native dependency overrides (patch() doesn't work for DI)
    app.dependency_overrides[get_vector_store] = lambda: _fake_vector_store()
    app.dependency_overrides[get_ai_client] = lambda: _fake_ai_client()

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c

    app.dependency_overrides.clear()
    get_settings.cache_clear()
    os.environ.pop("DATABASE_URL", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fake_vector_store() -> MagicMock:
    vs = MagicMock()
    vs.add_chunks.return_value = None
    vs.delete_document.return_value = None
    vs.search.return_value = []
    vs.count.return_value = 0
    return vs


def _fake_ai_client() -> MagicMock:
    ai = MagicMock()
    ai.generate_answer.return_value = "Test answer from fake AI."
    ai.summarize = AsyncMock(return_value="Test summary from fake AI.")
    return ai
