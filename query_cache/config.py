"""
Configuration for the SQL pattern cache.

All values are read from environment variables (the same ``env`` file the rest
of the app uses) with sensible defaults, so the cache "just works" alongside
``app-t16.py`` without extra wiring.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _get(name: str, default: str) -> str:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


@dataclass(frozen=True)
class CacheConfig:
    """Runtime configuration for the pattern cache."""

    # ── Database (re-uses the app's existing DB_* vars) ──────────────────────
    db_name: str = "KB"
    db_user: str = "postgres"
    db_password: str = ""
    db_host: str = "localhost"
    db_port: int = 5432

    # ── Ollama embeddings ────────────────────────────────────────────────────
    # llama3.2 is the *chat* model; embeddings should use a dedicated embedding
    # model. nomic-embed-text (768 dims) is the common default and is small/fast.
    # Pull it once on the host with:  ollama pull nomic-embed-text
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"

    # ── Matching behaviour ─────────────────────────────────────────────────────
    # Cosine similarity in [0, 1]. A match at or above this is served from cache.
    # 0.82 is a good starting point for nomic-embed-text; tune with --benchmark.
    match_threshold: float = 0.82
    # How many nearest patterns to retrieve before applying the threshold.
    top_k: int = 5
    # Table that stores the patterns + embeddings.
    table_name: str = "sql_query_patterns"
    # Request timeout (seconds) for Ollama embedding calls.
    embed_timeout: float = 30.0

    @property
    def dsn(self) -> str:
        """libpq connection string (psycopg2 / psycopg3 compatible)."""
        return (
            f"host={self.db_host} port={self.db_port} dbname={self.db_name} "
            f"user={self.db_user} password={self.db_password}"
        )


def load_config() -> CacheConfig:
    """Build a :class:`CacheConfig` from the process environment."""
    return CacheConfig(
        db_name=_get("DB_NAME", "KB"),
        db_user=_get("DB_USER", "postgres"),
        db_password=_get("DB_PASSWORD", ""),
        db_host=_get("DB_HOST", "localhost"),
        db_port=int(_get("DB_PORT", "5432")),
        ollama_base_url=_get("OLLAMA_BASE_URL", "http://localhost:11434"),
        embed_model=_get("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        match_threshold=float(_get("PATTERN_MATCH_THRESHOLD", "0.82")),
        top_k=int(_get("PATTERN_TOP_K", "5")),
        table_name=_get("PATTERN_TABLE", "sql_query_patterns"),
        embed_timeout=float(_get("EMBED_TIMEOUT", "30")),
    )
