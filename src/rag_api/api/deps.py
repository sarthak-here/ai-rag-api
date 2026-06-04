"""
FastAPI dependency injection providers.

All request-scoped dependencies are defined here so endpoints stay thin
and every dependency is independently testable.
"""

from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_api.core.config import Settings, get_settings
from rag_api.domain.services.document_service import DocumentService
from rag_api.domain.services.rag_service import RAGService
from rag_api.infrastructure.ai.client import AIClient
from rag_api.infrastructure.ai.ollama_client import OllamaClient
from rag_api.infrastructure.db.repositories.document_repository import DocumentRepository
from rag_api.infrastructure.db.session import get_session
from rag_api.infrastructure.vector_store.chromadb_store import VectorStore

# ── Singleton-style providers (resolved from app state) ──────────────────────


def get_session_factory(
    settings: Annotated[Settings, Depends(get_settings)],
) -> async_sessionmaker[AsyncSession]:
    from rag_api.infrastructure.db.session import build_engine_and_session

    return build_engine_and_session(settings)


async def db_session(
    factory: Annotated[async_sessionmaker[AsyncSession], Depends(get_session_factory)],
) -> AsyncGenerator[AsyncSession, None]:
    async for session in get_session(factory):
        yield session


def get_vector_store(
    settings: Annotated[Settings, Depends(get_settings)],
) -> VectorStore:
    return VectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection,
    )


def get_ai_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AIClient | OllamaClient:
    if settings.ai_provider == "ollama":
        return OllamaClient(settings)
    return AIClient(settings)


# ── Request-scoped service providers ─────────────────────────────────────────

SessionDep = Annotated[AsyncSession, Depends(db_session)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]
AIClientDep = Annotated[AIClient, Depends(get_ai_client)]


def get_document_service(
    session: SessionDep,
    vector_store: VectorStoreDep,
    settings: Annotated[Settings, Depends(get_settings)],
) -> DocumentService:
    repo = DocumentRepository(session)
    return DocumentService(
        repo=repo,
        vector_store=vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


def get_rag_service(
    session: SessionDep,
    vector_store: VectorStoreDep,
    ai_client: AIClientDep,
) -> RAGService:
    repo = DocumentRepository(session)
    return RAGService(repo=repo, vector_store=vector_store, ai_client=ai_client)


DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
