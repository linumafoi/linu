"""
The pgvector-backed pattern store and the cache orchestrator.

``SQLPatternCache`` ties together an :class:`~query_cache.embedding.Embedder`
and a :class:`VectorStore`:

* :meth:`SQLPatternCache.init_schema` - create the pgvector extension, table
  and index sized to the embedder's dimension.
* :meth:`SQLPatternCache.seed`        - embed every example phrasing and upsert.
* :meth:`SQLPatternCache.match`       - embed an incoming question and return
  the closest pattern if its cosine similarity clears the threshold.

Two stores are provided: :class:`PgVectorStore` (production) and
:class:`InMemoryVectorStore` (offline tests / CI without Postgres).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from .config import CacheConfig
from .embedding import Embedder, cosine_similarity
from .sql_patterns import PATTERNS, SQLPattern

try:  # psycopg2 is only required for the production store.
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover - exercised on hosts without the driver
    psycopg2 = None  # type: ignore[assignment]


# ── Data carried back from a match ──────────────────────────────────────────
@dataclass
class PatternMatch:
    intent: str
    category: str
    example: str
    sql_template: str
    params: List[dict]
    multi_row: bool
    score: float


def _vector_literal(vec: Sequence[float]) -> str:
    """Render a vector as a pgvector literal: ``[0.1,0.2,...]``."""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


# ── Storage backends ─────────────────────────────────────────────────────────
class VectorStore(Protocol):
    def init_schema(self, dimension: int) -> None: ...
    def clear(self) -> None: ...
    def upsert(self, rows: Sequence[dict]) -> int: ...
    def search(self, query_vec: Sequence[float], top_k: int) -> List[Tuple[dict, float]]: ...
    def count(self) -> int: ...


class PgVectorStore:
    """Stores pattern embeddings in Postgres using the ``vector`` extension."""

    def __init__(self, config: CacheConfig) -> None:
        if psycopg2 is None:
            raise RuntimeError(
                "psycopg2 is required for PgVectorStore. "
                "Install it with: pip install psycopg2-binary"
            )
        self.cfg = config
        self.table = config.table_name

    # Connection per operation keeps the helper dependency-free; for per-request
    # use in FastAPI, pass a pooled connection factory instead (see README).
    def _connect(self):
        return psycopg2.connect(self.cfg.dsn)

    def init_schema(self, dimension: int) -> None:
        ddl = f"""
        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE IF NOT EXISTS {self.table} (
            id            BIGSERIAL PRIMARY KEY,
            intent        TEXT        NOT NULL,
            category      TEXT        NOT NULL,
            example       TEXT        NOT NULL,
            sql_template  TEXT        NOT NULL,
            params        JSONB       NOT NULL DEFAULT '[]'::jsonb,
            multi_row     BOOLEAN     NOT NULL DEFAULT FALSE,
            embedding     VECTOR({dimension}) NOT NULL,
            created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (intent, example)
        );

        -- HNSW (pgvector >= 0.5.0) gives fast approximate cosine search.
        -- For older pgvector use ivfflat instead (see README).
        CREATE INDEX IF NOT EXISTS {self.table}_embedding_idx
            ON {self.table} USING hnsw (embedding vector_cosine_ops);

        CREATE INDEX IF NOT EXISTS {self.table}_intent_idx
            ON {self.table} (intent);
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(ddl)
            conn.commit()

    def clear(self) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"TRUNCATE {self.table} RESTART IDENTITY;")
            conn.commit()

    def upsert(self, rows: Sequence[dict]) -> int:
        sql = f"""
        INSERT INTO {self.table}
            (intent, category, example, sql_template, params, multi_row, embedding)
        VALUES
            (%(intent)s, %(category)s, %(example)s, %(sql_template)s,
             %(params)s::jsonb, %(multi_row)s, %(embedding)s::vector)
        ON CONFLICT (intent, example) DO UPDATE SET
            category     = EXCLUDED.category,
            sql_template = EXCLUDED.sql_template,
            params       = EXCLUDED.params,
            multi_row    = EXCLUDED.multi_row,
            embedding    = EXCLUDED.embedding;
        """
        prepared = [
            {
                "intent": r["intent"],
                "category": r["category"],
                "example": r["example"],
                "sql_template": r["sql_template"],
                "params": json.dumps(r["params"]),
                "multi_row": r["multi_row"],
                "embedding": _vector_literal(r["embedding"]),
            }
            for r in rows
        ]
        with self._connect() as conn, conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, prepared, page_size=100)
            conn.commit()
        return len(prepared)

    def search(self, query_vec: Sequence[float], top_k: int) -> List[Tuple[dict, float]]:
        # `<=>` is cosine distance; similarity = 1 - distance for unit vectors.
        sql = f"""
        SELECT intent, category, example, sql_template, params, multi_row,
               1 - (embedding <=> %(q)s::vector) AS similarity
        FROM {self.table}
        ORDER BY embedding <=> %(q)s::vector
        LIMIT %(k)s;
        """
        with self._connect() as conn:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(sql, {"q": _vector_literal(query_vec), "k": top_k})
            out: List[Tuple[dict, float]] = []
            for row in cur.fetchall():
                sim = float(row.pop("similarity"))
                # params come back as parsed JSON already with RealDictCursor.
                if isinstance(row.get("params"), str):
                    row["params"] = json.loads(row["params"])
                out.append((dict(row), sim))
            return out

    def count(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table};")
            return int(cur.fetchone()[0])


class InMemoryVectorStore:
    """Pure-Python store for offline self-tests (no Postgres required)."""

    def __init__(self) -> None:
        self._rows: List[dict] = []
        self.dimension = 0

    def init_schema(self, dimension: int) -> None:
        self.dimension = dimension

    def clear(self) -> None:
        self._rows.clear()

    def upsert(self, rows: Sequence[dict]) -> int:
        index: Dict[Tuple[str, str], int] = {
            (r["intent"], r["example"]): i for i, r in enumerate(self._rows)
        }
        for r in rows:
            key = (r["intent"], r["example"])
            if key in index:
                self._rows[index[key]] = dict(r)
            else:
                index[key] = len(self._rows)
                self._rows.append(dict(r))
        return len(rows)

    def search(self, query_vec: Sequence[float], top_k: int) -> List[Tuple[dict, float]]:
        scored = [
            (row, cosine_similarity(query_vec, row["embedding"])) for row in self._rows
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._rows)


# ── Cache orchestrator ────────────────────────────────────────────────────────
class SQLPatternCache:
    def __init__(
        self,
        config: CacheConfig,
        embedder: Embedder,
        store: Optional[VectorStore] = None,
    ) -> None:
        self.cfg = config
        self.embedder = embedder
        self.store: VectorStore = store if store is not None else PgVectorStore(config)

    def init_schema(self) -> None:
        self.store.init_schema(self.embedder.dimension)

    def seed(self, patterns: Sequence[SQLPattern] = PATTERNS, rebuild: bool = False) -> int:
        """Embed every example of every pattern and upsert it into the store."""
        if rebuild:
            self.store.clear()
        rows: List[dict] = []
        for pat in patterns:
            params_json = [asdict(p) for p in pat.params]
            for example in pat.examples:
                rows.append(
                    {
                        "intent": pat.intent,
                        "category": pat.category,
                        "example": example,
                        "sql_template": pat.sql,
                        "params": params_json,
                        "multi_row": pat.multi_row,
                        "embedding": self.embedder.embed(example),
                    }
                )
        return self.store.upsert(rows)

    def add_pattern(self, pattern: SQLPattern) -> int:
        """Insert/refresh a single pattern (used by runtime auto-learning)."""
        return self.seed([pattern], rebuild=False)

    def match(self, query: str, threshold: Optional[float] = None) -> Optional[PatternMatch]:
        """Return the best pattern for ``query`` if it clears the threshold."""
        thr = self.cfg.match_threshold if threshold is None else threshold
        qvec = self.embedder.embed(query)
        hits = self.store.search(qvec, self.cfg.top_k)
        if not hits:
            return None
        row, score = hits[0]
        if score < thr:
            return None
        return PatternMatch(
            intent=row["intent"],
            category=row["category"],
            example=row["example"],
            sql_template=row["sql_template"],
            params=row.get("params") or [],
            multi_row=bool(row.get("multi_row")),
            score=float(score),
        )

    def count(self) -> int:
        return self.store.count()
