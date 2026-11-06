"""The code that normally runs on a GPU (embedding search, the reader inside the benchmark),
exercised with small stand-in models so a Colab run cannot fail on a typo."""

import argparse
import hashlib

import numpy as np

from tablerag.bench import run
from tablerag.bench.specqa import build_specqa
from tablerag.retrieve import BM25, Dense, Hybrid
from tablerag.types import Chunk


class HashEncoder:
    """Bag-of-words vectors from hashed tokens: crude, deterministic, no download."""

    def encode(self, texts, normalize_embeddings=True, batch_size=64):
        out = np.zeros((len(texts), 64))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class FirstRowReader:
    """Answers with the last number of the first table row in the context and cites it."""

    def generate(self, prompts):
        import re

        out = []
        for p in prompts:
            m = re.search(r"\[([\w.\-]+:t\d+:r\d+)\][^\n]*?(\d+)\s*$", p, re.MULTILINE)
            out.append(f"{m.group(2)} | {m.group(1)}" if m else "NOT FOUND")
        return out


def test_dense_and_hybrid_rank_the_matching_chunk_first():
    chunks = [Chunk("a", "d", "pressure of stage five"), Chunk("b", "d", "run-in time in hours")]
    for r in (Dense(chunks, HashEncoder()), Hybrid(BM25(chunks), Dense(chunks, HashEncoder()))):
        assert r.search("stage five pressure", 1)[0][0].chunk_id == "a"


def test_benchmark_loop_with_a_reader(tmp_path):
    pdfs, questions = build_specqa(tmp_path / "specqa")
    keep = {"datasheet_7", "datasheet_mm_31"}
    pdfs = [p for p in pdfs if p.stem in keep]
    questions = [q for q in questions if q["doc_id"] in keep]
    args = argparse.Namespace(retriever="bm25", k=30, context_chars=2000)
    block = run.run_suite("specqa", pdfs, questions, ["naive", "ours"], args, FirstRowReader(), None)
    answers = block.pop("_answers")
    assert set(answers) == {"naive", "ours", "ours_noverify", "ours_ingest_only"}
    ours = block["ours"]["answers"]
    assert ours["routes"].get("structured", 0) > 0
    assert abs(ours["accuracy"] + ours["wrong_rate"] + ours["refusal_rate"] - 1) < 1e-9
    # verification can only turn wrong answers into refusals, never add wrong ones
    assert ours["wrong_rate"] <= block["ours_noverify"]["answers"]["wrong_rate"]
    md = run.to_markdown(
        {"config": {"retriever": "bm25", "reader": "fake", "context_chars": 2000}, "suites": {"specqa": block}}
    )
    assert "Correct answers by question type" in md
