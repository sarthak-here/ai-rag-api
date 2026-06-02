from typing import Annotated

from fastapi import APIRouter, Depends

from rag_api.core.config import Settings, get_settings
from rag_api.domain.schemas.common import AppBaseModel

router = APIRouter()


class HealthResponse(AppBaseModel):
    status: str
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    """Liveness probe — returns 200 when the application is running."""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.environment,
    )
