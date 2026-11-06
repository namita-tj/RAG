"""Retrievers shared by every method, so differences come from ingestion, not from search.

* BM25: keyword search (Okapi BM25, k1=1.5, b=0.75), runs anywhere.
* Dense: a sentence-transformers embedding model (GPU recommended, optional dependency).
* Hybrid: equal-weight sum of min-max normalised BM25 and dense scores. Strich et al. (2026)
  found this convex combination beat reciprocal rank fusion on text-and-table documents.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol

from tablerag.normalize import clean
from tablerag.types import Chunk

_TOKEN = re.compile(r"\d+(?:\.\d+)?|[a-z]+|[^\sa-z\d]")
_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "for",
        "to",
        "is",
        "are",
        "what",
        "which",
        "how",
        "does",
        "do",
        "and",
        "or",
        "with",
        "by",
        "at",
        "from",
        "be",
        "this",
        "that",
    ]
)


def tokenize(text: str) -> list[str]:
    toks = _TOKEN.findall(clean(text).lower())
    return [t for t in toks if t.isalnum() or "." in t if t not in _STOP]


class Retriever(Protocol):
    def search(self, query: str, k: int, doc_id: str | None = None) -> list[tuple[Chunk, float]]: ...


class BM25:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c.text)) for c in self.chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        out = []
        for d, length in zip(self.docs, self.lengths, strict=True):
            s = 0.0
            for t in q:
                f = d.get(t, 0)
                if f:
                    s += (
                        self.idf[t]
                        * f
                        * (self.k1 + 1)
                        / (f + self.k1 * (1 - self.b + self.b * length / (self.avg or 1)))
                    )
            out.append(s)
        return out

    def search(self, query: str, k: int, doc_id: str | None = None) -> list[tuple[Chunk, float]]:
        return _top(self.chunks, self.scores(query), k, doc_id)


class Dense:
    """Embedding search. ``model`` is a sentence-transformers model name or an object with encode()."""

    def __init__(self, chunks: Sequence[Chunk], model: Any = "BAAI/bge-small-en-v1.5", query_prefix: str | None = None):
        if isinstance(model, str):
            from sentence_transformers import SentenceTransformer

            if query_prefix is None and "bge" in model:
                query_prefix = "Represent this sentence for searching relevant passages: "
            model = SentenceTransformer(model)
        self.model, self.prefix = model, query_prefix or ""
        self.chunks = list(chunks)
        self.vectors = self.model.encode([c.text for c in self.chunks], normalize_embeddings=True, batch_size=64)

    def scores(self, query: str) -> list[float]:
        q = self.model.encode([self.prefix + query], normalize_embeddings=True)[0]
        return [
            float(sum(a * b for a, b in zip(v, q, strict=True))) if not hasattr(v, "dot") else float(v.dot(q))
            for v in self.vectors
        ]

    def search(self, query: str, k: int, doc_id: str | None = None) -> list[tuple[Chunk, float]]:
        return _top(self.chunks, self.scores(query), k, doc_id)


class Hybrid:
    def __init__(self, sparse: BM25, dense: Dense, weight: float = 0.5):
        self.sparse, self.dense, self.weight = sparse, dense, weight
        self.chunks = sparse.chunks

    def scores(self, query: str) -> list[float]:
        a, b = _minmax(self.sparse.scores(query)), _minmax(self.dense.scores(query))
        return [self.weight * x + (1 - self.weight) * y for x, y in zip(a, b, strict=True)]

    def search(self, query: str, k: int, doc_id: str | None = None) -> list[tuple[Chunk, float]]:
        return _top(self.chunks, self.scores(query), k, doc_id)


def _minmax(xs: list[float]) -> list[float]:
    if not xs:
        return xs
    lo, hi = min(xs), max(xs)
    return [0.0] * len(xs) if hi == lo else [(x - lo) / (hi - lo) for x in xs]


def _top(chunks: list[Chunk], scores: list[float], k: int, doc_id: str | None) -> list[tuple[Chunk, float]]:
    pairs = [(c, s) for c, s in zip(chunks, scores, strict=True) if doc_id is None or c.doc_id == doc_id]
    pairs.sort(key=lambda p: -p[1])
    return pairs[:k]
