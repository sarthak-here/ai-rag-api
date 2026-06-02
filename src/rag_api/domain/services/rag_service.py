from rag_api.core.exceptions import NotFoundError
from rag_api.core.logging import get_logger
from rag_api.domain.schemas.query import (
    QueryRequest,
    QueryResponse,
    SourceChunk,
    SummarizeResponse,
)
from rag_api.infrastructure.ai.client import AIClient
from rag_api.infrastructure.db.repositories.document_repository import DocumentRepository
from rag_api.infrastructure.vector_store.chromadb_store import VectorStore

logger = get_logger(__name__)


class RAGService:
    def __init__(
        self,
        repo: DocumentRepository,
        vector_store: VectorStore,
        ai_client: AIClient,
    ) -> None:
        self._repo = repo
        self._vector_store = vector_store
        self._ai = ai_client

    def query(self, request: QueryRequest) -> QueryResponse:
        hits = self._vector_store.search(request.question, top_k=request.top_k)

        if request.document_ids:
            allowed = set(request.document_ids)
            hits = [h for h in hits if h.document_id in allowed]

        context_chunks = [h.content for h in hits]
        answer = self._ai.generate_answer(request.question, context_chunks)

        sources = [
            SourceChunk(
                document_id=h.document_id,
                chunk_id=h.chunk_id,
                content=h.content,
                relevance_score=round(1.0 - h.distance, 4),
            )
            for h in hits
        ]

        logger.info(
            "rag_query_complete",
            question=request.question[:80],
            sources_retrieved=len(sources),
        )
        return QueryResponse(question=request.question, answer=answer, sources=sources)

    async def summarize(self, document_id: str) -> SummarizeResponse:
        doc = await self._repo.get(document_id)
        if doc is None:
            raise NotFoundError(f"Document '{document_id}' not found.")
        summary = self._ai.summarize(doc.content)
        logger.info("document_summarized", document_id=document_id)
        return SummarizeResponse(document_id=document_id, summary=summary)
