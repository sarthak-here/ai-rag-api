from fastapi import APIRouter

from rag_api.api.v1.endpoints import documents, health, queries

router = APIRouter()

router.include_router(health.router)
router.include_router(documents.router)
router.include_router(queries.router)
