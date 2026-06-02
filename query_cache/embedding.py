"""
Embedding backends.

The cache only needs one operation - turn text into a fixed-length vector - so
the interface is deliberately tiny. Two implementations are provided:

* :class:`OllamaEmbedder`  - production backend, calls the local Ollama server's
  ``/api/embeddings`` endpoint (stdlib ``urllib`` only, no extra dependency).
* :class:`HashingEmbedder` - deterministic, offline, dependency-free fallback
  used for local self-tests when Ollama is not reachable. It is **not** suitable
  for production matching quality, only for exercising the pipeline.

Both share an in-process LRU cache so repeated phrasings (and the user's own
re-asked questions) are embedded at most once.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import urllib.error
import urllib.request
from typing import List, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Anything that can map text to a unit-comparable float vector."""

    dimension: int

    def embed(self, text: str) -> List[float]:
        ...

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        ...


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class OllamaEmbedder:
    """Embeddings via a local Ollama server.

    Parameters
    ----------
    base_url:
        e.g. ``http://localhost:11434``.
    model:
        an *embedding* model that has been pulled, e.g. ``nomic-embed-text``.
    timeout:
        per-request timeout in seconds.
    normalize:
        L2-normalize vectors so cosine distance and inner product agree
        (recommended; keeps thresholds stable across models).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        normalize: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.normalize = normalize
        self._dimension = 0
        self._cache: "dict[str, List[float]]" = {}

    @property
    def dimension(self) -> int:
        """Vector size, discovered lazily on the first successful embed."""
        if self._dimension == 0:
            self.embed("dimension probe")
        return self._dimension

    def _call(self, text: str) -> List[float]:
        payload = json.dumps({"model": self.model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:  # pragma: no cover - network dependent
            raise RuntimeError(
                f"Ollama embeddings request failed ({self.base_url}, model={self.model}): {exc}"
            ) from exc

        vec = data.get("embedding")
        if not isinstance(vec, list) or not vec:
            raise RuntimeError(
                f"Ollama returned no embedding for model '{self.model}'. "
                "Did you run `ollama pull <embed-model>`?"
            )
        vec = [float(x) for x in vec]
        return _l2_normalize(vec) if self.normalize else vec

    def embed(self, text: str) -> List[float]:
        key = text.strip().lower()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        vec = self._call(key)
        self._dimension = len(vec)
        self._cache[key] = vec
        return vec

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        # Ollama's embeddings endpoint is single-input; loop with caching.
        return [self.embed(t) for t in texts]


class HashingEmbedder:
    """Deterministic offline embedder for tests (no server required).

    Produces a stable pseudo-embedding from token hashes. Semantically related
    phrases that share tokens land near each other, which is enough to validate
    the storage / matching / parameter-filling pipeline end to end.
    """

    def __init__(self, dimension: int = 256, normalize: bool = True) -> None:
        self._dimension = dimension
        self.normalize = normalize

    @property
    def dimension(self) -> int:
        return self._dimension

    def _token_vec(self, token: str) -> List[float]:
        # Expand a token into `dimension` deterministic floats in [-1, 1].
        out: List[float] = []
        counter = 0
        while len(out) < self._dimension:
            h = hashlib.blake2b(f"{token}:{counter}".encode("utf-8"), digest_size=32).digest()
            for i in range(0, len(h), 4):
                if len(out) >= self._dimension:
                    break
                (val,) = struct.unpack("<I", h[i : i + 4])
                out.append((val / 0xFFFFFFFF) * 2.0 - 1.0)
            counter += 1
        return out

    def embed(self, text: str) -> List[float]:
        tokens = [t for t in text.lower().replace("?", " ").split() if t]
        acc = [0.0] * self._dimension
        if not tokens:
            return acc
        for tok in tokens:
            tv = self._token_vec(tok)
            for i in range(self._dimension):
                acc[i] += tv[i]
        acc = [x / len(tokens) for x in acc]
        return _l2_normalize(acc) if self.normalize else acc

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in [-1, 1] (used by the offline matcher / tests)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)
