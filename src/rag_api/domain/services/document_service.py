from rag_api.core.exceptions import EmptyDocumentError, NotFoundError
from rag_api.core.logging import get_logger
from rag_api.domain.schemas.document import DocumentCreate, DocumentDetail, DocumentList, DocumentResponse
from rag_api.infrastructure.db.models import Chunk, Document
from rag_api.infrastructure.db.repositories.document_repository import DocumentRepository
from rag_api.infrastructure.vector_store.chromadb_store import VectorStore

logger = get_logger(__name__)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-boundary chunks."""
    words = text.split()
    if not words:
        return []
    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += step
    return chunks


class DocumentService:
    def __init__(
        self,
        repo: DocumentRepository,
        vector_store: VectorStore,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._repo = repo
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def create(self, payload: DocumentCreate) -> DocumentDetail:
        if not payload.content.strip():
            raise EmptyDocumentError

        doc = Document(
            title=payload.title,
            content=payload.content,
            source=payload.source,
        )
        doc = await self._repo.add(doc)

        raw_chunks = _chunk_text(payload.content, self._chunk_size, self._chunk_overlap)
        db_chunks = [
            Chunk(document_id=doc.id, content=text, chunk_index=i)
            for i, text in enumerate(raw_chunks)
        ]
        await self._repo.add_chunks(db_chunks)

        self._vector_store.add_chunks(
            chunk_ids=[c.id for c in db_chunks],
            contents=[c.content for c in db_chunks],
            document_id=doc.id,
        )

        logger.info("document_created", document_id=doc.id, chunks=len(db_chunks))
        return DocumentDetail(
            id=doc.id,
            title=doc.title,
            content=doc.content,
            source=doc.source,
            chunk_count=len(db_chunks),
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )

    async def get(self, document_id: str) -> DocumentDetail:
        doc = await self._repo.get_with_chunks(document_id)
        if doc is None:
            raise NotFoundError(f"Document '{document_id}' not found.")
        return DocumentDetail(
            id=doc.id,
            title=doc.title,
            content=doc.content,
            source=doc.source,
            chunk_count=len(doc.chunks),
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )

    async def list(self, skip: int, limit: int) -> DocumentList:
        docs = await self._repo.list(skip=skip, limit=limit)
        total = await self._repo.count()
        items = [
            DocumentResponse(
                id=d.id,
                title=d.title,
                source=d.source,
                chunk_count=len(d.chunks) if hasattr(d, "chunks") else 0,
                created_at=d.created_at,
                updated_at=d.updated_at,
            )
            for d in docs
        ]
        return DocumentList(items=items, total=total, skip=skip, limit=limit)

    async def delete(self, document_id: str) -> None:
        doc = await self._repo.get(document_id)
        if doc is None:
            raise NotFoundError(f"Document '{document_id}' not found.")
        self._vector_store.delete_document(document_id)
        await self._repo.delete(doc)
        logger.info("document_deleted", document_id=document_id)
