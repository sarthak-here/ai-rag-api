from fastapi import APIRouter, Query, status

from rag_api.api.deps import DocumentServiceDep
from rag_api.domain.schemas.document import DocumentCreate, DocumentDetail, DocumentList

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("", response_model=DocumentDetail, status_code=status.HTTP_201_CREATED)
async def create_document(
    payload: DocumentCreate,
    service: DocumentServiceDep,
) -> DocumentDetail:
    """
    Upload a document, chunk it, and index it for semantic search.

    - **title**: Human-readable name
    - **content**: Full text body (max 5 MB)
    - **source**: Optional provenance URL or file path
    """
    return await service.create(payload)


@router.get("", response_model=DocumentList)
async def list_documents(
    service: DocumentServiceDep,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> DocumentList:
    """Paginated list of all indexed documents."""
    return await service.list(skip=skip, limit=limit)


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(
    document_id: str,
    service: DocumentServiceDep,
) -> DocumentDetail:
    """Fetch a single document by ID including its full content."""
    return await service.get(document_id)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    service: DocumentServiceDep,
) -> None:
    """Delete a document and remove all its indexed chunks."""
    await service.delete(document_id)
