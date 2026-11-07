"""The retrieval sweep: split by document, scores like the main benchmark, picks on dev only."""

import pytest

from tablerag.bench.metrics import chars_until, retrieval_scores
from tablerag.bench.specqa import ROW_TYPES, build_specqa
from tablerag.bench.sweep import Suite, evaluate, final, split_docs, to_markdown, tune, with_bm25
from tablerag.ingest.structured import ingest_structured
from tablerag.query import normalize_query
from tablerag.retrieve import BM25
from test_gpu_paths import HashEncoder


@pytest.fixture(scope="module")
def specqa(tmp_path_factory):
    pdfs, questions = build_specqa(tmp_path_factory.mktemp("specqa"))
    chunks = [c for p in pdfs for c in ingest_structured(p).chunks]
    qs = [q for q in questions if q["type"] in ROW_TYPES]
    return with_bm25(Suite("specqa", chunks, qs, scoped=True))


def test_split_is_by_document_fixed_and_disjoint():
    ids = [f"d{i}" for i in range(10)] * 3
    dev, test = split_docs(ids)
    assert dev.isdisjoint(test) and dev | test == set(ids) and len(dev) == 5
    assert split_docs(ids) == (dev, test) and split_docs(ids, seed=1) != (dev, test)


def test_bm25_only_matches_the_main_benchmark(specqa):
    docs = {q["doc_id"] for q in specqa.questions}
    dense = [[0.0] * len(specqa.chunks) for _ in specqa.questions]  # ignored at weight 1
    swept = evaluate(specqa, dense, 1.0, docs)
    bm25 = BM25(specqa.chunks)
    costs = [
        chars_until([c for c, _ in bm25.search(normalize_query(q["question"]), 60, q["doc_id"])], specqa.relevant(q))
        for q in specqa.questions
    ]
    expected = retrieval_scores(costs)
    assert {k: swept[k] for k in expected} == expected
    assert 0 < swept["mrr"] <= 1


class Overlap:
    def predict(self, pairs):
        return [len(set(q.lower().split()) & set(t.lower().split())) for q, t in pairs]


def test_tune_then_final(specqa):
    dev, test = split_docs([q["doc_id"] for q in specqa.questions])
    rows, chosen = tune(
        [specqa], lambda _: HashEncoder(), ["hash-a", "hash-b"], [0.0, 0.5, 1.0], Overlap(), {"specqa": dev}
    )
    assert len(rows) == 2 * 3 + 2  # every weight per model, plus the reranker on each model's best weight
    assert chosen in [{k: r[k] for k in ("model", "weight", "rerank")} for r in rows]
    assert all(r["specqa"]["questions"] == sum(q["doc_id"] in dev for q in specqa.questions) for r in rows)
    results = final(
        [specqa],
        lambda _: HashEncoder(),
        {"defaults": {"model": "hash-a", "weight": 0.5, "rerank": None}, "chosen": chosen},
        Overlap(),
        {"specqa": test},
    )
    assert results["chosen"]["specqa"]["questions"] == sum(q["doc_id"] in test for q in specqa.questions)
    md = to_markdown(rows, chosen, results, ["specqa"])
    assert "## Dev half" in md and "## Test half" in md and "Chosen:" in md
