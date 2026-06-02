from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag_api.api.v1.router import router as v1_router
from rag_api.core.config import Settings, get_settings
from rag_api.core.exceptions import AppError
from rag_api.core.logging import configure_logging
from rag_api.infrastructure.db.session import create_tables

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings: Settings = get_settings()
    configure_logging(debug=settings.debug)
    await create_tables(settings)
    logger.info(
        "application_started",
        name=settings.app_name,
        version=settings.app_version,
        env=settings.environment,
    )
    yield
    logger.info("application_stopped")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-grade RAG API: upload documents, ask questions, get AI-powered answers."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if not settings.is_production else [],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ────────────────────────────────────────────────────
    @app.exception_handler(AppError)
    async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
        logger.warning("app_error", detail=exc.detail, status_code=exc.status_code)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(v1_router, prefix=settings.api_v1_prefix)

    return app


app = create_app()
