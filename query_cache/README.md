# SQL Pattern Cache (`query_cache`)

A **pgvector-backed semantic cache** that removes the LLM from the hot path for
common HR data questions, cutting `/api/chat` latency from an LLM round-trip
(hundreds of ms to seconds) down to a single embedding + vector lookup.

## How it works

```
            ┌── question ──┐
user ──▶ /api/chat         │
            │   1. embed question (Ollama: nomic-embed-text)
            │   2. cosine search in pgvector  ──▶  HIT  (similarity ≥ threshold)
            │                                      │   3. fill params, run read-only SELECT
            │                                      └─▶ data_result   ⚡ no LLM
            │
            └──────────────────────────────────▶  MISS
                                                   └─▶ existing LLM NL→SQL path
                                                       (optionally learn the result back)
```

* **Patterns** = parameterized, read-only `SELECT` templates + several natural
  language phrasings each (`query_cache/sql_patterns.py`). 18 patterns / 89
  phrasings shipped, covering payroll, statutory deductions, leave, attendance,
  profile and tickets.
* **Embeddings** = Ollama `nomic-embed-text` (768-dim) via HTTP. No extra Python
  client; an offline `HashingEmbedder` exists purely for tests.
* **Store** = a `sql_query_patterns` table with a `vector` column and an HNSW
  cosine index; lookups use the `<=>` operator.

## One-time setup

```bash
pip install -r ../requirements-patterns.txt      # psycopg2-binary
ollama pull nomic-embed-text                      # embedding model
# pgvector must be installed on the KB database (see requirements-patterns.txt)

python ../setup_pattern_cache.py init             # create schema + embed patterns
python ../setup_pattern_cache.py init --rebuild   # after editing sql_patterns.py
```

Verify matching without a database (uses the offline embedder):

```bash
python ../setup_pattern_cache.py --offline test "what is my net salary" --employee HS1001
python ../setup_pattern_cache.py --offline benchmark
```

## Wiring into `app-t16.py`

Build the resolver once at startup and call it first inside the data branch of
`/api/chat`. On a miss, run your existing LLM NL→SQL code.

```python
# ── startup (once) ──────────────────────────────────────────────────────────
from query_cache import load_config, SQLPatternCache
from query_cache.embedding import OllamaEmbedder
from query_cache.resolver import QueryResolver, PsycopgExecutor

_cfg      = load_config()
_embedder = OllamaEmbedder(_cfg.ollama_base_url, _cfg.embed_model, _cfg.embed_timeout)
_cache    = SQLPatternCache(_cfg, _embedder)          # PgVectorStore by default
_executor = PsycopgExecutor(_cfg)                     # pass a pooled conn for prod

def _llm_nl2sql(query: str, ctx: dict) -> dict:
    """Adapter around the EXISTING llama3.2 NL→SQL path; returns a data_result dict."""
    ...  # your current code that calls Ollama, runs the SQL, shapes the result
    return {"type": "data_result", "sql": sql, "columns": cols,
            "rows": rows, "row_count": len(rows), "summary": summary}

resolver = QueryResolver(_cache, _cfg, executor=_executor,
                         llm_fallback=_llm_nl2sql, auto_learn=False)

# ── inside the /api/chat handler, in the "this is a data question" branch ─────
context = {"employee_id": body.get("employee_id")}   # session-scoped, trusted
result  = resolver.resolve(body["query"], context)

if result.type == "data_result":
    return result.to_response()        # ⚡ served from cache (or llm_fallback)
# else: result.type == "cache_miss" → run your existing flow if no llm_fallback set
```

`to_response()` returns the exact shape the frontend's `renderBotResponse`
expects (`type`, `sql`, `columns`, `rows`, `row_count`, `summary`,
`display_mode`) plus `source`, `intent`, `score` and `cache_latency_ms` for
observability.

## Tuning

| Env var                   | Default            | Purpose                                   |
|---------------------------|--------------------|-------------------------------------------|
| `OLLAMA_EMBED_MODEL`      | `nomic-embed-text` | Embedding model (must be `ollama pull`ed) |
| `PATTERN_MATCH_THRESHOLD` | `0.82`             | Min cosine similarity to serve from cache |
| `PATTERN_TOP_K`           | `5`                | Neighbours fetched before thresholding    |
| `PATTERN_TABLE`           | `sql_query_patterns` | Table name                              |

* **Too many misses?** Lower the threshold or add more `examples` to a pattern.
* **Wrong pattern served?** Raise the threshold or make phrasings more distinct.

## Safety

* Templates are single read-only `SELECT`s; `resolver.is_read_only()` re-checks,
  and `PsycopgExecutor` runs each query in a `SET TRANSACTION READ ONLY` block.
* All values (employee id, month, ticket id, …) are bound as SQL parameters —
  never string-formatted — so the cache cannot introduce SQL injection.
* If required params can't be extracted, the pattern is skipped and the LLM path
  (which can ask the user to clarify) takes over.

## Older pgvector (no HNSW, < 0.5.0)

Replace the index in `pattern_cache.py` with:

```sql
CREATE INDEX ... USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
```
