#!/usr/bin/env python3
"""
Bootstrap / maintenance CLI for the SQL pattern cache.

Usage
-----
    # Create pgvector schema + embed & store all patterns
    python setup_pattern_cache.py init

    # Wipe and re-embed everything (after editing sql_patterns.py)
    python setup_pattern_cache.py init --rebuild

    # Try a question against the cache and print the match + resolved SQL
    python setup_pattern_cache.py test "what is my net salary" --employee HS1001

    # Quick recall check over every built-in example phrasing
    python setup_pattern_cache.py benchmark

Environment is read from the same vars as app-t16.py (DB_*, OLLAMA_*). Set
OLLAMA_EMBED_MODEL to the embedding model you pulled (default nomic-embed-text).

If Ollama is unreachable the CLI falls back to the offline HashingEmbedder and
an in-memory store, so you can still exercise the matching logic (pass
--offline to force this).
"""

from __future__ import annotations

import argparse
import sys

from query_cache.config import load_config
from query_cache.embedding import HashingEmbedder, OllamaEmbedder
from query_cache.pattern_cache import InMemoryVectorStore, SQLPatternCache
from query_cache.param_extraction import extract_params
from query_cache.resolver import QueryResolver
from query_cache.sql_patterns import PATTERNS, ParamSpec


def _build_cache(offline: bool) -> SQLPatternCache:
    cfg = load_config()
    if offline:
        print("• Using OFFLINE embedder (HashingEmbedder) + in-memory store.")
        embedder = HashingEmbedder(dimension=256)
        return SQLPatternCache(cfg, embedder, store=InMemoryVectorStore())

    embedder = OllamaEmbedder(cfg.ollama_base_url, cfg.embed_model, cfg.embed_timeout)
    try:
        _ = embedder.dimension  # probe Ollama now for a clear early error
    except Exception as exc:  # noqa: BLE001
        print(f"! Ollama embeddings unavailable: {exc}")
        print("  Falling back to the offline embedder. Use real Ollama for production.")
        return SQLPatternCache(cfg, HashingEmbedder(dimension=256), store=InMemoryVectorStore())

    print(f"• Embedder: Ollama '{cfg.embed_model}' ({embedder.dimension} dims) at {cfg.ollama_base_url}")
    return SQLPatternCache(cfg, embedder)


def cmd_init(args: argparse.Namespace) -> int:
    cache = _build_cache(args.offline)
    print(f"• Creating schema (table '{cache.cfg.table_name}', dim {cache.embedder.dimension}) ...")
    cache.init_schema()
    n = cache.seed(PATTERNS, rebuild=args.rebuild)
    print(f"✓ Seeded {n} example embeddings across {len(PATTERNS)} patterns.")
    print(f"✓ Total rows in cache: {cache.count()}")
    return 0


def cmd_test(args: argparse.Namespace) -> int:
    cache = _build_cache(args.offline)
    if cache.count() == 0:  # offline store starts empty
        cache.init_schema()
        cache.seed(PATTERNS)

    context = {"employee_id": args.employee} if args.employee else {}
    resolver = QueryResolver(cache, cache.cfg, executor=None)
    result = resolver.resolve(args.query, context)

    print(f"\nQuery     : {args.query}")
    print(f"Type      : {result.type}  (source={result.source})")
    if result.intent:
        print(f"Intent    : {result.intent}  (similarity={result.score:.3f})")
    if result.missing_params:
        print(f"Missing   : {result.missing_params}")
    if result.sql:
        print(f"SQL       : {result.sql}")
        print(f"Params    : {result.params}")
    print(f"Latency   : {result.elapsed_ms:.2f} ms (cache lookup only)")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    cache = _build_cache(args.offline)
    if cache.count() == 0:
        cache.init_schema()
        cache.seed(PATTERNS)

    total = 0
    hits = 0
    correct = 0
    for pat in PATTERNS:
        for example in pat.examples:
            total += 1
            m = cache.match(example)
            if m is not None:
                hits += 1
                if m.intent == pat.intent:
                    correct += 1
    print(f"\nExamples tested : {total}")
    print(f"Above threshold : {hits}  ({100*hits/total:.1f}%)  [threshold={cache.cfg.match_threshold}]")
    print(f"Correct intent  : {correct}  ({100*correct/total:.1f}%)")
    if args.offline:
        print("(offline embedder is for pipeline validation only; real recall uses Ollama)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SQL pattern cache bootstrap CLI")
    parser.add_argument("--offline", action="store_true",
                        help="Force the offline embedder + in-memory store.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create schema and seed patterns.")
    p_init.add_argument("--rebuild", action="store_true", help="Truncate before seeding.")
    p_init.set_defaults(func=cmd_init)

    p_test = sub.add_parser("test", help="Resolve a single question.")
    p_test.add_argument("query")
    p_test.add_argument("--employee", help="Employee id to use as session context.")
    p_test.set_defaults(func=cmd_test)

    p_bench = sub.add_parser("benchmark", help="Recall check over built-in examples.")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
