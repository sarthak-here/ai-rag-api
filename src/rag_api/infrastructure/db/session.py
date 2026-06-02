from collections.abc import AsyncGenerator
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from rag_api.core.config import Settings
from rag_api.infrastructure.db.base import Base


def build_engine_and_session(settings: Settings) -> async_sessionmaker[AsyncSession]:
    # Ensure the data directory exists for SQLite
    if "sqlite" in settings.database_url:
        db_path = settings.database_url.split("///")[-1]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    engine = create_async_engine(
        settings.database_url,
        echo=settings.debug,
        pool_pre_ping=True,
    )
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def create_tables(settings: Settings) -> None:
    """Create all tables. Called once at application startup."""
    if "sqlite" in settings.database_url:
        db_path = settings.database_url.split("///")[-1]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    engine = create_async_engine(settings.database_url, echo=settings.debug)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine.dispose()


async def get_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
