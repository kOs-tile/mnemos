"""
MNEMOS configuration — Pydantic Settings backed by environment variables / .env file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Neo4j ────────────────────────────────────────────────────────────────
    neo4j_uri: str = Field(default="bolt://localhost:7687", description="Neo4j Bolt URI")
    neo4j_user: str = Field(default="neo4j", description="Neo4j username")
    neo4j_password: str = Field(default="mnemos_secret", description="Neo4j password")
    neo4j_max_connection_pool_size: int = Field(default=50)
    neo4j_connection_timeout: float = Field(default=30.0)

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection: str = Field(default="mnemos_memories")
    qdrant_grpc_port: int = Field(default=6334)
    qdrant_prefer_grpc: bool = Field(default=False)

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379")
    redis_query_cache_ttl: int = Field(default=300, description="Cache TTL in seconds")

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "deepseek"] = Field(default="openai")
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o-mini")
    deepseek_api_key: str = Field(default="")
    deepseek_base_url: str = Field(default="https://api.deepseek.com")
    deepseek_model: str = Field(default="deepseek-chat")

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_dim: int = Field(default=384)

    # ── Decay Engine ─────────────────────────────────────────────────────────
    decay_interval_hours: float = Field(
        default=6.0,
        description="How often to run the Ebbinghaus decay sweep (hours)",
    )
    decay_threshold: float = Field(
        default=0.05,
        description="Edge weight below this is archived",
        ge=0.0,
        le=1.0,
    )
    decay_stability_base: float = Field(
        default=1.0,
        description="Base stability factor S in R(t)=e^(-t/S). Higher = slower forgetting.",
        gt=0.0,
    )
    decay_resurrection_boost: float = Field(
        default=2.0,
        description="Stability multiplier when archived memory is reinforced",
        gt=1.0,
    )

    # ── Contradiction Resolution ──────────────────────────────────────────────
    contradiction_similarity_threshold: float = Field(
        default=0.85,
        description="Cosine similarity above which two facts are considered potentially conflicting",
        ge=0.0,
        le=1.0,
    )

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    mnemos_api_secret: str = Field(default="")

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_key_present(cls, v: str) -> str:
        # Validation is deferred — keys are checked at runtime, not import time.
        return v

    @property
    def llm_api_key(self) -> str:
        if self.llm_provider == "deepseek":
            return self.deepseek_api_key
        return self.openai_api_key

    @property
    def llm_model(self) -> str:
        if self.llm_provider == "deepseek":
            return self.deepseek_model
        return self.openai_model

    @property
    def llm_base_url(self) -> str | None:
        if self.llm_provider == "deepseek":
            return self.deepseek_base_url
        return None  # OpenAI uses its default base URL


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the global Settings singleton (cached after first call)."""
    return Settings()
