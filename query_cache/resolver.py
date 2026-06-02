"""
The query resolver - the piece ``app-t16.py`` calls on the hot path.

Flow
----
1. Embed the question once and look for a cached pattern (fast path).
2. If a pattern matches *and* all its required parameters can be filled,
   execute the parameterized SQL and return a ``data_result`` - no LLM.
3. Otherwise return a ``cache_miss`` result so the caller runs its existing
   LLM NL->SQL path. If a ``llm_fallback`` callable is supplied the resolver
   invokes it directly and (optionally) learns the result back into the cache.

Everything user-derived is bound as a SQL parameter; only single read-only
``SELECT`` statements are ever executed.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from .config import CacheConfig
from .param_extraction import extract_params
from .pattern_cache import PatternMatch, SQLPatternCache
from .sql_patterns import ParamSpec

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover
    psycopg2 = None  # type: ignore[assignment]


_READ_ONLY_RE = re.compile(r"^\s*(with|select)\b", re.IGNORECASE)
_FORBIDDEN_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|copy)\b",
    re.IGNORECASE,
)


def is_read_only(sql: str) -> bool:
    """Defense-in-depth: only allow a single read-only SELECT/CTE statement."""
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:  # no statement chaining
        return False
    if not _READ_ONLY_RE.match(stripped):
        return False
    if _FORBIDDEN_RE.search(stripped):
        return False
    return True


# ── DB execution backends ─────────────────────────────────────────────────────
class SqlExecutor(Protocol):
    def execute(self, sql: str, params: Dict[str, object]) -> Tuple[List[str], List[dict]]:
        ...


class PsycopgExecutor:
    """Executes read-only SELECTs and returns (columns, list-of-dict rows)."""

    def __init__(self, config: CacheConfig, connection=None) -> None:
        if psycopg2 is None:
            raise RuntimeError("psycopg2 is required. pip install psycopg2-binary")
        self.cfg = config
        self._conn = connection  # pass a pooled connection for per-request use

    def _connection(self):
        return self._conn if self._conn is not None else psycopg2.connect(self.cfg.dsn)

    def execute(self, sql: str, params: Dict[str, object]) -> Tuple[List[str], List[dict]]:
        conn = self._connection()
        owns = self._conn is None
        try:
            # Statement-level read-only guard in the DB itself.
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("SET TRANSACTION READ ONLY;")
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
            cols = list(rows[0].keys()) if rows else []
            conn.commit()
            return cols, rows
        finally:
            if owns:
                conn.close()


# ── Result ─────────────────────────────────────────────────────────────────────
@dataclass
class ResolverResult:
    type: str                       # "data_result" | "cache_miss"
    source: str                     # "cache" | "llm" | "none"
    intent: Optional[str] = None
    sql: Optional[str] = None
    params: Dict[str, object] = field(default_factory=dict)
    columns: List[str] = field(default_factory=list)
    rows: List[dict] = field(default_factory=list)
    row_count: int = 0
    summary: str = ""
    display_mode: str = "answer"
    score: float = 0.0
    elapsed_ms: float = 0.0
    missing_params: List[str] = field(default_factory=list)

    def to_response(self) -> dict:
        """Shape the result the way the frontend's renderBotResponse expects."""
        return {
            "type": self.type,
            "source": self.source,
            "intent": self.intent,
            "sql": self.sql,
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "summary": self.summary,
            "display_mode": self.display_mode,
            "score": round(self.score, 4),
            "cache_latency_ms": round(self.elapsed_ms, 2),
        }


def _summarize(match: PatternMatch, columns: List[str], rows: List[dict]) -> Tuple[str, str]:
    """Produce (summary, display_mode) for the chat bubble."""
    if not rows:
        return "No matching records were found.", "answer"
    if match.multi_row or len(rows) > 1:
        return f"Found {len(rows)} record(s).", "table"
    # Single row -> short prose, "col: value" pairs.
    row = rows[0]
    parts = [f"{c.replace('_', ' ')}: {row[c]}" for c in columns]
    return " | ".join(parts), "answer"


class QueryResolver:
    def __init__(
        self,
        cache: SQLPatternCache,
        config: CacheConfig,
        executor: Optional[SqlExecutor] = None,
        llm_fallback: Optional[Callable[[str, dict], dict]] = None,
        auto_learn: bool = False,
    ) -> None:
        self.cache = cache
        self.cfg = config
        self.executor = executor
        self.llm_fallback = llm_fallback
        self.auto_learn = auto_learn

    def resolve(self, query: str, context: Optional[Dict[str, object]] = None) -> ResolverResult:
        context = context or {}
        started = time.perf_counter()

        match = self.cache.match(query)
        if match is None:
            return self._miss(query, context, started, reason="no_pattern_match")

        # Reconstruct typed specs from what was stored alongside the pattern.
        specs = [ParamSpec(**p) for p in match.params]
        params, missing = extract_params(specs, query, context)
        if missing:
            # We matched an intent but cannot safely fill the query -> let the
            # LLM path (which can ask for clarification) take over.
            res = self._miss(query, context, started, reason="missing_params")
            res.intent = match.intent
            res.missing_params = missing
            res.score = match.score
            return res

        if not is_read_only(match.sql_template):  # pragma: no cover - guard
            return self._miss(query, context, started, reason="non_read_only")

        if self.executor is None:
            # No DB wired (e.g. offline test): return the resolved plan only.
            elapsed = (time.perf_counter() - started) * 1000
            return ResolverResult(
                type="data_result", source="cache", intent=match.intent,
                sql=match.sql_template, params=params, score=match.score,
                summary="(resolved from cache; executor not configured)",
                elapsed_ms=elapsed,
            )

        columns, rows = self.executor.execute(match.sql_template, params)
        summary, display_mode = _summarize(match, columns, rows)
        elapsed = (time.perf_counter() - started) * 1000
        return ResolverResult(
            type="data_result", source="cache", intent=match.intent,
            sql=match.sql_template, params=params, columns=columns, rows=rows,
            row_count=len(rows), summary=summary, display_mode=display_mode,
            score=match.score, elapsed_ms=elapsed,
        )

    def _miss(self, query: str, context: dict, started: float, reason: str) -> ResolverResult:
        if self.llm_fallback is not None:
            data = self.llm_fallback(query, context)
            elapsed = (time.perf_counter() - started) * 1000
            # Optionally learn this question against the LLM-produced SQL.
            if self.auto_learn:
                self._learn(query, data)
            return ResolverResult(
                type=data.get("type", "data_result"), source="llm",
                sql=data.get("sql"), columns=data.get("columns", []),
                rows=data.get("rows", []), row_count=data.get("row_count", 0),
                summary=data.get("summary", ""),
                display_mode=data.get("display_mode", "answer"),
                elapsed_ms=elapsed,
            )
        elapsed = (time.perf_counter() - started) * 1000
        return ResolverResult(
            type="cache_miss", source="none", summary=reason, elapsed_ms=elapsed,
        )

    def _learn(self, query: str, data: dict) -> None:
        """Hook for runtime auto-learning of new (question -> sql) pairs.

        Kept conservative: only learns verified, read-only SQL. Wire this up to
        persist into the same pgvector table via ``cache.add_pattern`` once you
        have validated the LLM output against your schema.
        """
        sql = (data or {}).get("sql")
        if not sql or not is_read_only(sql):
            return
        # Intentionally a no-op by default; see README "Auto-learning" section
        # for the recommended validation before persisting LLM-authored SQL.
        return
