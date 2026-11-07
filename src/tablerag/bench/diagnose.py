"""Which exact operations answer which questions, and are they right? No language model needed.

The exact path answers some questions without the reader. On a new benchmark it can fire on
questions it wasn't built for (a "total" in a financial report is rarely the sum of a column).
This runs only retrieval (BM25, as a stand-in for the evidence the reader would see), the query
clean-up and the exact path, and reports per operation how often it answered and how often it
was right. Runs on a CPU in a few minutes.

    python -m tablerag.bench.diagnose --suite t2 --out results/diagnose
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tablerag.answer import within_budget
from tablerag.bench.metrics import number_match, spec_correct
from tablerag.ingest.structured import ingest_structured
from tablerag.query import normalize_query
from tablerag.retrieve import BM25
from tablerag.structured_ops import try_structured


def diagnose(suite: str, pdfs: list[Path], questions: list[dict], k: int = 30, budget: int = 2000) -> dict:
    results = [ingest_structured(p) for p in pdfs]
    chunks = [c for r in results for c in r.chunks]
    tables = [t for r in results for t in r.tables]
    by_doc: dict[str, list] = {}
    for t in tables:
        by_doc.setdefault(t.doc_id, []).append(t)
    retriever = BM25(chunks)
    scoped = suite == "specqa"
    correct = spec_correct if scoped else (lambda q, a: number_match(q["answer"], a))
    rows, ops, right = [], Counter(), Counter()
    changed = 0
    for q in questions:
        question = normalize_query(q["question"])
        changed += question != q["question"]
        if scoped:
            candidates = by_doc.get(q["doc_id"], [])
        else:
            hits = [c for c, _ in retriever.search(question, k)]
            seen = {c.meta.get("table_id") for c in within_budget(hits, budget)}
            candidates = [t for t in tables if t.table_id in seen]
        op = try_structured(question, candidates)
        if op is None:
            continue
        ok = bool(correct(q, op.answer))
        ops[op.operation] += 1
        right[op.operation] += ok
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "normalized": question,
                "gold": q["answer"],
                "answer": op.answer,
                "operation": op.operation,
                "explanation": op.explanation,
                "correct": ok,
            }
        )
    return {
        "questions": len(questions),
        "query_changed_by_cleanup": changed,
        "operations": {o: {"answered": n, "correct": right[o], "wrong": n - right[o]} for o, n in ops.most_common()},
        "answers": rows,
    }


def to_markdown(suite: str, report: dict) -> str:
    lines = [
        f"## {suite}",
        "",
        f"{report['questions']} questions; the query clean-up changed {report['query_changed_by_cleanup']} of them.",
        "",
        "| Operation | Answered | Correct | Wrong |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines += [f"| {o} | {v['answered']} | {v['correct']} | {v['wrong']} |" for o, v in report["operations"].items()]
    wrong = [r for r in report["answers"] if not r["correct"]][:15]
    if wrong:
        lines += [
            "",
            "Wrong answers (first 15):",
            "",
            "| Operation | Question | Answer | Gold |",
            "| --- | --- | --- | --- |",
        ]
        lines += [
            f"| {r['operation']} | {r['question'][:110]} | {r['answer'][:40]} | {r['gold'][:30]} |" for r in wrong
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", nargs="+", default=["specqa", "t2"], choices=["specqa", "t2"])
    p.add_argument("--t2-contexts", type=int, default=100)
    p.add_argument("--data", type=Path, default=Path("benchmarks/data"))
    p.add_argument("--out", type=Path, default=Path("results/diagnose"))
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    md = ["# Exact path diagnosis", ""]
    for suite in args.suite:
        if suite == "specqa":
            from tablerag.bench.specqa import build_specqa

            pdfs, questions = build_specqa(args.data / "specqa")
        else:
            from tablerag.bench.t2rag import build_t2, load_t2

            pdfs, questions = build_t2(load_t2(args.t2_contexts), args.data / "t2")
        report = diagnose(suite, pdfs, questions)
        (args.out / f"{suite}.json").write_text(json.dumps(report, indent=2))
        md.append(to_markdown(suite, report))
    text = "\n".join(md)
    (args.out / "summary.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
