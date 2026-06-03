from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from rag_api.infrastructure.db.models import Chunk, Document
from rag_api.infrastructure.db.repositories.base import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    model = Document

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def list(self, *, skip: int = 0, limit: int = 100) -> list[Document]:
        result = await self._session.execute(
            select(Document).options(selectinload(Document.chunks)).offset(skip).limit(limit)
        )
        return list(result.scalars().all())

    async def get_with_chunks(self, document_id: str) -> Document | None:
        result = await self._session.execute(
            select(Document)
            .options(selectinload(Document.chunks))
            .where(Document.id == document_id)
        )
        return result.scalar_one_or_none()

    async def count(self) -> int:
        result = await self._session.execute(select(func.count(Document.id)))
        return result.scalar_one()

    async def add_chunks(self, chunks: list[Chunk]) -> None:
        self._session.add_all(chunks)
        await self._session.flush()

    async def delete_chunks_for_document(self, document_id: str) -> None:
        result = await self._session.execute(select(Chunk).where(Chunk.document_id == document_id))
        for chunk in result.scalars().all():
            await self._session.delete(chunk)
        await self._session.flush()
