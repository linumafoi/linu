"""
query_cache
===========

A pgvector-backed *semantic SQL pattern cache* for the HRSuits AI support agent.

Purpose
-------
The chat resolver in ``app-t16.py`` answers data questions by asking an LLM
(Ollama / llama3.2) to translate natural language into SQL on every request.
That LLM round-trip is the dominant source of latency.

This package removes the LLM from the hot path for *common* questions:

1.  A curated library of parameterized, read-only SQL patterns is defined
    (:mod:`query_cache.sql_patterns`).
2.  Every natural-language example phrasing for each pattern is embedded and
    stored in a pgvector table (:class:`query_cache.pattern_cache.SQLPatternCache`).
3.  At request time the incoming question is embedded once and matched against
    the stored vectors. On a hit (cosine similarity >= threshold) the cached
    SQL is filled with extracted parameters and executed directly - no LLM call.
4.  On a miss the resolver falls back to the existing LLM NL->SQL path and can
    optionally *learn* the new (question, sql) pair back into the cache.

Public surface
--------------
>>> from query_cache import SQLPatternCache, resolve_query, PATTERNS
"""

from .config import CacheConfig, load_config
from .sql_patterns import PATTERNS, SQLPattern
from .embedding import Embedder, OllamaEmbedder
from .pattern_cache import SQLPatternCache, PatternMatch
from .resolver import QueryResolver, ResolverResult

__all__ = [
    "CacheConfig",
    "load_config",
    "PATTERNS",
    "SQLPattern",
    "Embedder",
    "OllamaEmbedder",
    "SQLPatternCache",
    "PatternMatch",
    "QueryResolver",
    "ResolverResult",
]

__version__ = "1.0.0"
