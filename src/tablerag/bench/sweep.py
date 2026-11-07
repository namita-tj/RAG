"""Tuning the retrieval settings without fooling ourselves: tune on one half, report on the other.

The documents of each benchmark are split in half by document (no table is in both halves):

* **tune** (dev half): every embedding model x every BM25 weight, plus the reranker on the
  best weight of each model. Only retrieval is scored, so no reader model is needed.
* **final** (test half, touched once): the chosen settings against today's defaults.

    python -m tablerag.bench.sweep --suite specqa t2 --out results/sweep            # both stages
    python -m tablerag.bench.sweep --suite specqa --models BAAI/bge-small-en-v1.5 --no-rerank

Retrieval is scored exactly like the main benchmark (``found@N``: the evidence is inside the first
N characters retrieved), plus MRR (1 / rank of the first relevant chunk). Each embedding model
encodes the chunks once and all weights reuse those scores, so the weight sweep is almost free.

The choice: the highest mean found@1000 over the suites on the dev half; ties go to the smaller
model and to no reranker. Budgets and reader models (stage 2) are future work.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tablerag.bench.metrics import BUDGETS, chars_until, chunk_has_row, rank_of, retrieval_scores
from tablerag.bench.specqa import ROW_TYPES, build_specqa
from tablerag.ingest.structured import ingest_structured
from tablerag.query import Reranker, normalize_query
from tablerag.retrieve import BM25, _minmax
from tablerag.types import Chunk

DEFAULT = {"model": "BAAI/bge-small-en-v1.5", "weight": 0.5, "rerank": None}
MODELS = [
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-base",
]
WEIGHTS = [0.0, 0.3, 0.5, 0.7, 1.0]  # share of BM25: 0 = embeddings only, 1 = keywords only
RERANKER = "BAAI/bge-reranker-base"
# query / passage prefixes the models were trained with
PREFIXES = {
    "bge-small-en": ("Represent this sentence for searching relevant passages: ", ""),
    "bge-base-en": ("Represent this sentence for searching relevant passages: ", ""),
    "e5": ("query: ", "passage: "),
}
SIZES = {"small": 0, "base": 1, "m3": 2}  # tie-break: prefer the smaller model


def prefixes(model: str) -> tuple[str, str]:
    for key, value in PREFIXES.items():
        if key in model:
            return value
    return "", ""


def size_rank(model: str) -> int:
    return next((v for k, v in SIZES.items() if k in model), 1)


def split_docs(doc_ids: Sequence[str], seed: int = 0) -> tuple[set[str], set[str]]:
    """(dev, test): a fixed random half of the documents each, by document so no table is in both."""
    ids = sorted(set(doc_ids))
    random.Random(seed).shuffle(ids)
    half = len(ids) // 2
    return set(ids[:half]), set(ids[half:])


@dataclass
class Suite:
    name: str
    chunks: list[Chunk]
    questions: list[dict]  # only the questions retrieval is scored on
    scoped: bool  # SpecQA: search within the question's document; T²-RAGBench: the whole collection
    bm25: list[list[float]] = field(default_factory=list)  # per question, per chunk

    def relevant(self, q: dict) -> Callable[[Chunk], bool]:
        if self.scoped:
            return lambda c, ev=q["evidence"]: chunk_has_row(c, ev)
        return lambda c, d=q["doc_id"]: c.doc_id == d


def load_suite(name: str, data: Path, t2_contexts: int) -> Suite:
    if name == "specqa":
        pdfs, questions = build_specqa(data / "specqa")
        questions = [q for q in questions if q["type"] in ROW_TYPES]
    else:
        from tablerag.bench.t2rag import build_t2, load_t2

        pdfs, questions = build_t2(load_t2(t2_contexts), data / "t2")
    chunks = [c for p in pdfs for c in ingest_structured(p).chunks]
    return Suite(name, chunks, questions, scoped=name == "specqa")


def with_bm25(suite: Suite) -> Suite:
    bm25 = BM25(suite.chunks)
    suite.bm25 = [bm25.scores(normalize_query(q["question"])) for q in suite.questions]
    return suite


def dense_scores(suite: Suite, encoder: Any, model: str) -> list[list[float]]:
    """Cosine similarity of every question with every chunk; the chunks are encoded once."""
    import numpy as np

    q_prefix, p_prefix = prefixes(model)
    docs = np.asarray(
        encoder.encode([p_prefix + c.text for c in suite.chunks], normalize_embeddings=True, batch_size=64)
    )
    queries = np.asarray(
        encoder.encode(
            [q_prefix + normalize_query(q["question"]) for q in suite.questions],
            normalize_embeddings=True,
            batch_size=64,
        )
    )
    return (queries @ docs.T).tolist()


class Ranked:
    """A retriever over precomputed scores, so a reranker can sit on top of it."""

    def __init__(self, chunks: list[Chunk], scores: dict[str, list[float]]):
        self.chunks, self.by_query = chunks, scores

    def search(self, query: str, k: int, doc_id: str | None = None) -> list[tuple[Chunk, float]]:
        pairs = [
            (c, s)
            for c, s in zip(self.chunks, self.by_query[query], strict=True)
            if doc_id is None or c.doc_id == doc_id
        ]
        pairs.sort(key=lambda p: -p[1])
        return pairs[:k]


def evaluate(
    suite: Suite, dense: list[list[float]], weight: float, docs: set[str], reranker: Any = None
) -> dict[str, Any]:
    """found@N and MRR for one setting, on the questions of the given documents."""
    picked = [i for i, q in enumerate(suite.questions) if q["doc_id"] in docs]
    scores = {}
    for i in picked:
        a, b = _minmax(suite.bm25[i]), _minmax(dense[i])
        scores[normalize_query(suite.questions[i]["question"])] = [
            weight * x + (1 - weight) * y for x, y in zip(a, b, strict=True)
        ]
    retriever: Any = Ranked(suite.chunks, scores)
    if reranker is not None:
        retriever = Reranker(retriever, reranker, candidates=30)
    costs, reciprocal = [], []
    for i in picked:
        q = suite.questions[i]
        hits = [
            c for c, _ in retriever.search(normalize_query(q["question"]), 60, q["doc_id"] if suite.scoped else None)
        ]
        costs.append(chars_until(hits, suite.relevant(q)))
        rank = rank_of(hits, suite.relevant(q))
        reciprocal.append(1 / rank if rank else 0.0)
    out = retrieval_scores(costs)
    out["mrr"] = sum(reciprocal) / (len(reciprocal) or 1)
    out["questions"] = len(picked)
    return out


def tune(
    suites: list[Suite],
    encoders: Callable[[str], Any],
    models: list[str],
    weights: list[float],
    reranker: Any,
    dev: dict[str, set[str]],
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for model in models:
        t0 = time.perf_counter()
        encoder = encoders(model)
        dense = {s.name: dense_scores(s, encoder, model) for s in suites}
        for w in weights:
            rows.append(
                {
                    "model": model,
                    "weight": w,
                    "rerank": None,
                    **{s.name: evaluate(s, dense[s.name], w, dev[s.name]) for s in suites},
                }
            )
        if reranker is not None:
            best = max((r for r in rows if r["model"] == model), key=lambda r: _score(r, suites))
            rows.append(
                {
                    "model": model,
                    "weight": best["weight"],
                    "rerank": RERANKER,
                    **{s.name: evaluate(s, dense[s.name], best["weight"], dev[s.name], reranker) for s in suites},
                }
            )
        print(f"[tune] {model}: {time.perf_counter() - t0:.0f}s")
    chosen = max(rows, key=lambda r: (_score(r, suites), -size_rank(r["model"]), r["rerank"] is None))
    return rows, {k: chosen[k] for k in ("model", "weight", "rerank")}


def _score(row: dict, suites: list[Suite]) -> float:
    return sum(row[s.name]["found@1000"] for s in suites) / len(suites)


def final(
    suites: list[Suite],
    encoders: Callable[[str], Any],
    configs: dict[str, dict],
    reranker: Any,
    test: dict[str, set[str]],
) -> dict[str, dict]:
    out = {}
    for label, cfg in configs.items():
        encoder = encoders(cfg["model"])
        out[label] = {
            **cfg,
            **{
                s.name: evaluate(
                    s,
                    dense_scores(s, encoder, cfg["model"]),
                    cfg["weight"],
                    test[s.name],
                    reranker if cfg["rerank"] else None,
                )
                for s in suites
            },
        }
    return out


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def to_markdown(rows: list[dict], chosen: dict, results: dict[str, dict] | None, suites: list[str]) -> str:
    cols = [f"{s} @{b}" for s in suites for b in BUDGETS[:2]] + [f"{s} MRR" for s in suites]

    def cells(r: dict) -> str:
        vals = [pct(r[s][f"found@{b}"]) for s in suites for b in BUDGETS[:2]] + [f"{r[s]['mrr']:.3f}" for s in suites]
        return " | ".join(vals)

    lines = [
        "# Retrieval sweep",
        "",
        "Tuned on the dev half of the documents, reported on the test half. `@N`: evidence inside the first N characters.",
        "BM25 weight: 0 = embeddings only, 1 = keywords only.",
        "",
        "## Dev half: every setting",
        "",
        "| Embedding model | BM25 weight | Reranker | " + " | ".join(cols) + " |",
        "| --- | ---: | --- |" + " ---: |" * len(cols),
    ]
    lines += [f"| `{r['model']}` | {r['weight']} | {'yes' if r['rerank'] else '-'} | {cells(r)} |" for r in rows]
    lines += [
        "",
        f"Chosen: `{chosen['model']}`, BM25 weight {chosen['weight']}, reranker {'on' if chosen['rerank'] else 'off'} "
        "(highest mean found@1000 over the suites).",
    ]
    if results:
        lines += [
            "",
            "## Test half: chosen against today's defaults",
            "",
            "| Setting | Embedding model | BM25 weight | Reranker | " + " | ".join(cols) + " |",
            "| --- | --- | ---: | --- |" + " ---: |" * len(cols),
        ]
        lines += [
            f"| {label} | `{r['model']}` | {r['weight']} | {'yes' if r['rerank'] else '-'} | {cells(r)} |"
            for label, r in results.items()
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", nargs="+", default=["specqa", "t2"], choices=["specqa", "t2"])
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--weights", nargs="+", type=float, default=WEIGHTS)
    p.add_argument("--no-rerank", action="store_true", help="skip the reranker")
    p.add_argument("--t2-contexts", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", type=Path, default=Path("benchmarks/data"))
    p.add_argument("--out", type=Path, default=Path("results/sweep"))
    args = p.parse_args(argv)

    from sentence_transformers import CrossEncoder, SentenceTransformer

    cache: dict[str, Any] = {}

    def encoders(name: str) -> Any:
        if name not in cache:
            cache.clear()  # one embedding model in GPU memory at a time
            cache[name] = SentenceTransformer(name)
        return cache[name]

    suites = [with_bm25(load_suite(s, args.data, args.t2_contexts)) for s in args.suite]
    halves = {s.name: split_docs([q["doc_id"] for q in s.questions], args.seed) for s in suites}
    dev = {k: v[0] for k, v in halves.items()}
    test = {k: v[1] for k, v in halves.items()}
    for s in suites:
        print(f"{s.name}: {len(s.chunks)} chunks, {len(dev[s.name])} dev / {len(test[s.name])} test documents")
    reranker = None if args.no_rerank else CrossEncoder(RERANKER)
    rows, chosen = tune(suites, encoders, args.models, args.weights, reranker, dev)
    results = final(suites, encoders, {"defaults": DEFAULT, "chosen": chosen}, reranker, test)
    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "config": {"seed": args.seed, "split": {k: {"dev": sorted(dev[k]), "test": sorted(test[k])} for k in dev}},
        "dev": rows,
        "chosen": chosen,
        "test": results,
    }
    (args.out / "sweep.json").write_text(json.dumps(report, indent=2))
    md = to_markdown(rows, chosen, results, [s.name for s in suites])
    (args.out / "summary.md").write_text(md)
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
