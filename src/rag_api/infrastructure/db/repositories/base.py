from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_api.infrastructure.db.base import Base


class BaseRepository[ModelT: Base]:
    """Generic async repository providing common CRUD operations."""

    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, id: str) -> ModelT | None:
        return await self._session.get(self.model, id)

    async def list(self, *, skip: int = 0, limit: int = 100) -> list[ModelT]:
        result = await self._session.execute(select(self.model).offset(skip).limit(limit))
        return list(result.scalars().all())

    async def add(self, instance: ModelT) -> ModelT:
        self._session.add(instance)
        await self._session.flush()
        await self._session.refresh(instance)
        return instance

    async def delete(self, instance: ModelT) -> None:
        await self._session.delete(instance)
        await self._session.flush()
