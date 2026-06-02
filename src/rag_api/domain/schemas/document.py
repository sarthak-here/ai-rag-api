from datetime import datetime

from pydantic import Field, field_validator

from rag_api.domain.schemas.common import AppBaseModel

_MAX_CONTENT_BYTES = 5 * 1024 * 1024  # 5 MB


class DocumentCreate(AppBaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    source: str | None = Field(default=None, max_length=500)

    @field_validator("content")
    @classmethod
    def content_not_too_large(cls, v: str) -> str:
        if len(v.encode()) > _MAX_CONTENT_BYTES:
            msg = "Document content exceeds 5 MB limit."
            raise ValueError(msg)
        return v


class DocumentResponse(AppBaseModel):
    id: str
    title: str
    source: str | None
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class DocumentDetail(DocumentResponse):
    content: str


class DocumentList(AppBaseModel):
    items: list[DocumentResponse]
    total: int
    skip: int
    limit: int
