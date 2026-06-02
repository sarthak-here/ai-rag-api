from fastapi import APIRouter

from rag_api.api.deps import RAGServiceDep
from rag_api.domain.schemas.query import (
    QueryRequest,
    QueryResponse,
    SummarizeRequest,
    SummarizeResponse,
)

router = APIRouter(prefix="/queries", tags=["RAG"])


@router.post("/ask", response_model=QueryResponse)
def ask(
    request: QueryRequest,
    service: RAGServiceDep,
) -> QueryResponse:
    """
    Ask a natural-language question against the indexed document corpus.

    Retrieves the most semantically relevant chunks and generates a grounded
    answer using Claude. Optionally scope to specific document IDs.
    """
    return service.query(request)


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(
    request: SummarizeRequest,
    service: RAGServiceDep,
) -> SummarizeResponse:
    """Generate a concise AI summary of an indexed document."""
    return await service.summarize(request.document_id)
