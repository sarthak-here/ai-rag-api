"""Unit tests for RAGService."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from rag_api.core.exceptions import NotFoundError
from rag_api.domain.schemas.query import QueryRequest
from rag_api.domain.services.rag_service import RAGService
from rag_api.infrastructure.db.models import Document
from rag_api.infrastructure.vector_store.chromadb_store import SearchResult


def _make_service(
    repo: MagicMock | None = None,
    vector_store: MagicMock | None = None,
    ai_client: MagicMock | None = None,
) -> RAGService:
    return RAGService(
        repo=repo or MagicMock(),
        vector_store=vector_store or MagicMock(),
        ai_client=ai_client or MagicMock(),
    )


def _hit(doc_id: str = "doc-1", dist: float = 0.1) -> SearchResult:
    return SearchResult(
        chunk_id="chunk-1", document_id=doc_id, content="relevant text", distance=dist
    )


class TestRAGServiceQuery:
    def test_query_returns_answer_with_sources(self):
        vs = MagicMock()
        vs.search.return_value = [_hit()]
        ai = MagicMock()
        ai.generate_answer.return_value = "The answer."

        service = _make_service(vector_store=vs, ai_client=ai)
        result = service.query(QueryRequest(question="What is this?"))

        assert result.answer == "The answer."
        assert len(result.sources) == 1
        assert result.sources[0].relevance_score == pytest.approx(0.9)

    def test_query_filters_by_document_ids(self):
        vs = MagicMock()
        vs.search.return_value = [_hit("doc-1"), _hit("doc-2")]
        ai = MagicMock()
        ai.generate_answer.return_value = "Filtered answer."

        service = _make_service(vector_store=vs, ai_client=ai)
        result = service.query(QueryRequest(question="What?", document_ids=["doc-1"]))

        assert len(result.sources) == 1
        assert result.sources[0].document_id == "doc-1"

    def test_query_with_no_hits_returns_no_context_message(self):
        vs = MagicMock()
        vs.search.return_value = []
        ai = MagicMock()
        ai.generate_answer.return_value = "No relevant context found to answer this question."

        service = _make_service(vector_store=vs, ai_client=ai)
        result = service.query(QueryRequest(question="Unknown topic?"))

        assert "No relevant context" in result.answer


class TestRAGServiceSummarize:
    async def test_summarize_happy_path(self):
        repo = MagicMock()
        now = datetime.now(tz=timezone.utc)
        doc = Document(id="doc-1", title="T", content="Long text...", source=None)
        doc.created_at = now
        doc.updated_at = now
        doc.chunks = []
        repo.get = AsyncMock(return_value=doc)

        ai = MagicMock()
        ai.summarize.return_value = "A concise summary."

        service = _make_service(repo=repo, ai_client=ai)
        result = await service.summarize("doc-1")

        assert result.summary == "A concise summary."
        assert result.document_id == "doc-1"

    async def test_summarize_missing_doc_raises(self):
        repo = MagicMock()
        repo.get = AsyncMock(return_value=None)
        service = _make_service(repo=repo)

        with pytest.raises(NotFoundError):
            await service.summarize("missing")
