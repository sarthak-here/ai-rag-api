"""Unit tests for DocumentService — no I/O, all dependencies are fakes."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from rag_api.core.exceptions import EmptyDocumentError, NotFoundError
from rag_api.domain.schemas.document import DocumentCreate
from rag_api.domain.services.document_service import DocumentService, _chunk_text
from rag_api.infrastructure.db.models import Document


# ---------------------------------------------------------------------------
# _chunk_text helper
# ---------------------------------------------------------------------------
class TestChunkText:
    def test_single_chunk_when_content_fits(self):
        result = _chunk_text("hello world", chunk_size=10, overlap=2)
        assert result == ["hello world"]

    def test_multiple_chunks_with_overlap(self):
        words = " ".join([str(i) for i in range(20)])
        chunks = _chunk_text(words, chunk_size=5, overlap=2)
        assert len(chunks) > 1
        # Overlap: last words of chunk N appear at start of chunk N+1
        first_chunk_words = chunks[0].split()
        second_chunk_words = chunks[1].split()
        assert first_chunk_words[3:5] == second_chunk_words[:2]

    def test_empty_text_returns_empty_list(self):
        assert _chunk_text("", chunk_size=10, overlap=2) == []

    def test_no_overlap(self):
        words = " ".join([str(i) for i in range(10)])
        chunks = _chunk_text(words, chunk_size=5, overlap=0)
        assert len(chunks) == 2


# ---------------------------------------------------------------------------
# DocumentService
# ---------------------------------------------------------------------------
def _make_doc(doc_id: str = "doc-1") -> Document:
    now = datetime.now(tz=timezone.utc)
    doc = Document(id=doc_id, title="Test", content="hello world", source=None)
    doc.created_at = now
    doc.updated_at = now
    doc.chunks = []
    return doc


def _make_service(repo: MagicMock, vector_store: MagicMock) -> DocumentService:
    return DocumentService(
        repo=repo,
        vector_store=vector_store,
        chunk_size=50,
        chunk_overlap=10,
    )


class TestDocumentServiceCreate:
    async def test_create_happy_path(self):
        repo = MagicMock()
        doc = _make_doc()
        repo.add = AsyncMock(return_value=doc)
        repo.add_chunks = AsyncMock()
        vs = MagicMock()

        service = _make_service(repo, vs)
        result = await service.create(DocumentCreate(title="Test", content="hello world"))

        repo.add.assert_called_once()
        repo.add_chunks.assert_called_once()
        vs.add_chunks.assert_called_once()
        assert result.id == "doc-1"

    async def test_create_raises_on_empty_content(self):
        service = _make_service(MagicMock(), MagicMock())
        with pytest.raises(EmptyDocumentError):
            await service.create(DocumentCreate(title="T", content="   "))


class TestDocumentServiceGet:
    async def test_get_existing(self):
        repo = MagicMock()
        repo.get_with_chunks = AsyncMock(return_value=_make_doc())
        service = _make_service(repo, MagicMock())

        result = await service.get("doc-1")
        assert result.id == "doc-1"

    async def test_get_missing_raises_not_found(self):
        repo = MagicMock()
        repo.get_with_chunks = AsyncMock(return_value=None)
        service = _make_service(repo, MagicMock())

        with pytest.raises(NotFoundError):
            await service.get("missing")


class TestDocumentServiceDelete:
    async def test_delete_calls_vector_store(self):
        repo = MagicMock()
        repo.get = AsyncMock(return_value=_make_doc())
        repo.delete = AsyncMock()
        vs = MagicMock()
        service = _make_service(repo, vs)

        await service.delete("doc-1")
        vs.delete_document.assert_called_once_with("doc-1")
        repo.delete.assert_called_once()

    async def test_delete_missing_raises_not_found(self):
        repo = MagicMock()
        repo.get = AsyncMock(return_value=None)
        service = _make_service(repo, MagicMock())

        with pytest.raises(NotFoundError):
            await service.delete("missing")
