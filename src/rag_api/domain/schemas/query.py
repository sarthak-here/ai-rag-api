from pydantic import Field

from rag_api.domain.schemas.common import AppBaseModel


class QueryRequest(AppBaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_ids: list[str] | None = Field(
        default=None,
        description="Optional list of document IDs to scope the search. Searches all if omitted.",
    )


class SourceChunk(AppBaseModel):
    document_id: str
    chunk_id: str
    content: str
    relevance_score: float


class QueryResponse(AppBaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]


class SummarizeRequest(AppBaseModel):
    document_id: str


class SummarizeResponse(AppBaseModel):
    document_id: str
    summary: str
