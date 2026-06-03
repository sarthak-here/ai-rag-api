from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "RAG Document Intelligence API"
    app_version: str = "1.0.0"
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    api_v1_prefix: str = "/api/v1"

    # ── Database ─────────────────────────────────────────────────────────────
    database_url: str = Field(default="sqlite+aiosqlite:///./data/rag.db")

    # ── Vector Store ─────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field(default="./data/chroma")
    chroma_collection: str = Field(default="documents")

    # ── Anthropic ────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(...)
    anthropic_model: str = Field(default="claude-3-5-haiku-20241022")

    # ── RAG Parameters ───────────────────────────────────────────────────────
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=64, ge=0)
    retrieval_top_k: int = Field(default=5, gt=0)
    max_context_tokens: int = Field(default=8_192, gt=0)

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info: object) -> int:
        # info.data is populated for fields validated after the ones they depend on
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
