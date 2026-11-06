"""Runs the benchmark: every ingestion method, the same retriever and reader, the same scoring.

    python -m tablerag.bench.run --suite specqa                       # CPU: ingestion + BM25 retrieval
    python -m tablerag.bench.run --suite specqa t2 --retriever hybrid --reader Qwen/Qwen2.5-3B-Instruct \\
        --methods naive markdown camelot docling ours                 # full run on a GPU (Colab)

Writes results.json, summary.md and answers.jsonl into --out.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

from tablerag.answer import HFReader, Reader, answer_questions
from tablerag.bench.metrics import (
    BUDGETS,
    answer_scores,
    chars_until,
    chunk_has_row,
    ingestion_scores,
    number_match,
    retrieval_scores,
    spec_correct,
)
from tablerag.bench.specqa import ROW_TYPES
from tablerag.ingest.baselines import ingest_markdown, ingest_naive
from tablerag.ingest.structured import ingest_structured
from tablerag.retrieve import BM25, Dense, Hybrid
from tablerag.types import IngestResult

LABELS = {
    "naive": "Fixed-size text chunks (LangChain style)",
    "markdown": "Per-page markdown tables (pdfplumber)",
    "camelot": "Camelot tables + text chunks",
    "docling": "Docling + HybridChunker",
    "ours": "Ours: structure-aware rows + exact ops + verification",
    "ours_noverify": "Ours without verification",
    "ours_ingest_only": "Ours: ingestion only (plain reader)",
    "ablate_no_context": "Ours without row context",
    "ablate_no_stitch": "Ours without cross-page stitching",
    "ablate_rows_only": "Ours without table and profile chunks",
}


def ingester(name: str):  # noqa: ANN201
    if name == "naive":
        return ingest_naive
    if name == "markdown":
        return ingest_markdown
    if name == "camelot":
        from tablerag.ingest.camelot_ingest import ingest_camelot

        return ingest_camelot
    if name == "docling":
        from tablerag.ingest.docling_ingest import ingest_docling

        return ingest_docling
    if name == "ours":
        return ingest_structured
    if name == "ablate_no_context":
        return lambda p: ingest_structured(p, context=False)
    if name == "ablate_no_stitch":
        return lambda p: ingest_structured(p, stitch=False)
    if name == "ablate_rows_only":
        return lambda p: ingest_structured(p, table_chunks=False, profiles=False)
    raise ValueError(f"unknown method {name!r}")


def ingest_all(name: str, pdfs: list[Path]) -> tuple[list[IngestResult], dict]:
    fn = ingester(name)
    results = [fn(p) for p in pdfs]
    errors = [r.error for r in results if r.error]
    pages = sum(r.pages for r in results) or 1
    return results, {
        "seconds_per_page": round(sum(r.seconds for r in results) / pages, 4),
        "errors": len(errors),
        "first_error": errors[0] if errors else None,
    }


def make_retriever(kind: str, chunks, embed_model: Any):  # noqa: ANN001, ANN201
    bm25 = BM25(chunks)
    if kind == "bm25":
        return bm25
    dense = Dense(chunks, embed_model)
    return dense if kind == "dense" else Hybrid(bm25, dense)


def run_suite(
    suite: str,
    pdfs: list[Path],
    questions: list[dict],
    methods: list[str],
    args: argparse.Namespace,
    reader: Reader | None,
    embed_model: Any,
) -> dict:
    scoped = suite == "specqa"
    out: dict[str, Any] = {}
    gold_rows = [(q["doc_id"], q["evidence"]) for q in questions if q.get("evidence")]
    for method in methods:
        t0 = time.perf_counter()
        results, ing = ingest_all(method, pdfs)
        chunks = [c for r in results for c in r.chunks]
        tables = [t for r in results for t in r.tables]
        entry: dict[str, Any] = {"ingestion": ing}
        if not chunks:
            entry["skipped"] = ing["first_error"] or "no chunks"
            out[method] = entry
            print(f"[{suite}] {method}: skipped ({entry['skipped']})")
            continue
        if suite == "specqa":
            entry["ingestion"].update(ingestion_scores(chunks, gold_rows))
        else:
            entry["ingestion"].update({"chunks": len(chunks)})
        retriever = make_retriever(args.retriever, chunks, embed_model)

        # retrieval
        ranks, cross = [], []
        for q in questions:
            if suite == "specqa" and q["type"] not in ROW_TYPES:
                continue
            hits = [c for c, _ in retriever.search(q["question"], 60, q["doc_id"] if scoped else None)]
            if suite == "specqa":
                ranks.append(chars_until(hits, lambda c, ev=q["evidence"]: chunk_has_row(c, ev)))
                cross.append(q["cross_page"])
            else:
                ranks.append(chars_until(hits, lambda c, d=q["doc_id"]: c.doc_id == d))
        entry["retrieval"] = retrieval_scores(ranks)
        if suite == "specqa":
            entry["retrieval_cross_page"] = retrieval_scores([r for r, x in zip(ranks, cross, strict=True) if x])
            entry["retrieval_same_page"] = retrieval_scores([r for r, x in zip(ranks, cross, strict=True) if not x])

        # answers
        correct = spec_correct if suite == "specqa" else (lambda q, a: number_match(q["answer"], a))
        is_ours = method == "ours"
        modes = (
            [("ours", "ours"), ("ours_noverify", "ours_unverified"), ("ours_ingest_only", "plain")]
            if is_ours
            else [(method, "plain")]
        )
        if reader is None and is_ours:
            # without a reader only the exact structured path can answer; score it on the questions it takes
            from tablerag.structured_ops import try_structured

            by_doc: dict[str, list] = {}
            for t in tables:
                by_doc.setdefault(t.doc_id, []).append(t)
            taken = right = 0
            for q in questions:
                op = try_structured(q["question"], by_doc.get(q["doc_id"], []))
                if op is not None:
                    taken += 1
                    right += correct(q, op.answer)
            entry["structured_path"] = {
                "questions_taken": taken,
                "of": len(questions),
                "correct": right,
                "precision": right / taken if taken else 0.0,
            }
        if reader is not None and not method.startswith("ablate"):
            for label, mode in modes:
                answers = answer_questions(
                    questions,
                    retriever,
                    reader,
                    k=args.k,
                    tables=tables,
                    mode=mode,
                    scoped=scoped,
                    context_chars=args.context_chars,
                )
                key = label
                scores = answer_scores(questions, answers, correct)
                by_type: dict[str, list] = {}
                for q, a in zip(questions, answers, strict=True):
                    by_type.setdefault(q["type"], []).append((q, a))
                scores["by_type"] = {
                    t: answer_scores([q for q, _ in qa], [a for _, a in qa], correct) for t, qa in by_type.items()
                }
                if suite == "specqa":
                    xp = [(q, a) for q, a in zip(questions, answers, strict=True) if q.get("cross_page")]
                    scores["cross_page"] = answer_scores([q for q, _ in xp], [a for _, a in xp], correct)
                routes: dict[str, int] = {}
                for a in answers:
                    routes[a.route] = routes.get(a.route, 0) + 1
                scores["routes"] = routes
                out.setdefault("_answers", {})[key] = [
                    {
                        "id": q["id"],
                        "type": q["type"],
                        "question": q["question"],
                        "gold": q["answer"],
                        "answer": a.text,
                        "route": a.route,
                        "rows": a.rows,
                        "correct": correct(q, a.text),
                    }
                    for q, a in zip(questions, answers, strict=True)
                ]
                if key == method:
                    entry["answers"] = scores
                else:
                    out[key] = {"answers": scores}
        entry["seconds"] = round(time.perf_counter() - t0, 1)
        out[method] = entry
        print(f"[{suite}] {method}: {json.dumps({k: v for k, v in entry.items() if k != 'answers'})[:400]}")
    return out


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def to_markdown(results: dict) -> str:
    lines = [
        "# Table RAG benchmark",
        "",
        f"Retriever: {results['config']['retriever']}, reader: "
        f"{results['config']['reader'] or 'none (retrieval only)'}, the reader sees "
        f"{results['config']['context_chars']} characters of retrieved text.",
        "",
    ]
    for suite, block in results["suites"].items():
        meth = {k: v for k, v in block.items() if not k.startswith("_")}
        lines += [
            f"## {'SpecQA (synthetic datasheets, exact answers)' if suite == 'specqa' else 'T²-RAGBench (FinQA + TAT-DQA, rendered to PDF)'}",
            "",
        ]
        ing = [(m, v["ingestion"]) for m, v in meth.items() if "ingestion" in v and "skipped" not in v]
        if suite == "specqa" and ing:
            lines += [
                "**Ingestion: does each table row survive in one chunk?**",
                "",
                "| Method | Rows intact | Rows intact + section named | Chunks | Mean chunk chars | Sec/page |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
            lines += [
                f"| {LABELS.get(m, m)} | {pct(v['row_intact'])} | {pct(v['row_with_context'])} | {v['chunks']} | "
                f"{v['mean_chunk_chars']} | {v['seconds_per_page']} |"
                for m, v in ing
            ]
            lines.append("")
        ret = [(m, v) for m, v in meth.items() if "retrieval" in v]
        if ret:
            what = "the chunk holding the evidence row" if suite == "specqa" else "a chunk of the right document"
            lines += [
                f"**Retrieval: is {what} within the first N characters retrieved?**",
                "",
                "| Method | "
                + " | ".join(f"{b} chars" for b in BUDGETS)
                + " | Median chars needed |"
                + (" 2000 chars, cross-page rows |" if suite == "specqa" else ""),
                "| --- |" + " ---: |" * (len(BUDGETS) + 1) + (" ---: |" if suite == "specqa" else ""),
            ]
            for m, v in ret:
                r = v["retrieval"]
                extra = f" {pct(v['retrieval_cross_page']['found@2000'])} |" if suite == "specqa" else ""
                lines.append(
                    f"| {LABELS.get(m, m)} | "
                    + " | ".join(pct(r[f"found@{b}"]) for b in BUDGETS)
                    + f" | {r['median_chars'] if r['median_chars'] is not None else '-'} |{extra}"
                )
            lines.append("")
        ans = [(m, v["answers"]) for m, v in meth.items() if "answers" in v]
        if ans:
            lines += [
                "**Answers**",
                "",
                "| Method | Correct | Wrong | Refused | Precision when answered |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
            lines += [
                f"| {LABELS.get(m, m)} | {pct(a['accuracy'])} | {pct(a['wrong_rate'])} | {pct(a['refusal_rate'])} | "
                f"{pct(a['precision'])} |"
                for m, a in ans
            ]
            types = sorted({t for _, a in ans for t in a.get("by_type", {})})
            if types:
                lines += [
                    "",
                    "Correct answers by question type:",
                    "",
                    "| Method | " + " | ".join(types) + " |",
                    "| --- |" + " ---: |" * len(types),
                ]
                lines += [
                    f"| {LABELS.get(m, m)} | "
                    + " | ".join(pct(a["by_type"].get(t, {}).get("accuracy")) for t in types)
                    + " |"
                    for m, a in ans
                ]
            lines.append("")
        sp = meth.get("ours", {}).get("structured_path")
        if sp:
            lines += [
                f"Exact structured path (no language model): took {sp['questions_taken']} of {sp['of']} questions, "
                f"{sp['correct']} correct ({pct(sp['precision'])}).",
                "",
            ]
        skipped = [(m, v["skipped"]) for m, v in meth.items() if "skipped" in v]
        lines += [f"Skipped {LABELS.get(m, m)}: {why}" for m, why in skipped]
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", nargs="+", default=["specqa"], choices=["specqa", "t2"])
    p.add_argument(
        "--methods",
        nargs="+",
        default=["naive", "markdown", "camelot", "ours", "ablate_no_context", "ablate_no_stitch", "ablate_rows_only"],
    )
    p.add_argument("--retriever", default="bm25", choices=["bm25", "dense", "hybrid"])
    p.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--reader", default=None, help="Hugging Face model id, e.g. Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--k", type=int, default=30, help="chunks retrieved before the character budget is applied")
    p.add_argument("--context-chars", type=int, default=2000, help="characters of retrieved text the reader sees")
    p.add_argument("--t2-contexts", type=int, default=150, help="contexts per T²-RAGBench subset")
    p.add_argument("--data", type=Path, default=Path("benchmarks/data"))
    p.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = p.parse_args(argv)

    from tablerag.bench.specqa import build_specqa

    reader = HFReader(args.reader) if args.reader else None
    embed_model: Any = args.embed_model
    if args.retriever != "bm25":
        from sentence_transformers import SentenceTransformer

        embed_model = SentenceTransformer(args.embed_model)  # load once, share across methods
    results: dict[str, Any] = {
        "config": {
            "retriever": args.retriever,
            "embed_model": args.embed_model if args.retriever != "bm25" else None,
            "reader": args.reader,
            "k": args.k,
            "context_chars": args.context_chars,
            "methods": args.methods,
            "python": platform.python_version(),
        },
        "suites": {},
    }
    answers: dict[str, Any] = {}
    for suite in args.suite:
        if suite == "specqa":
            pdfs, questions = build_specqa(args.data / "specqa")
        else:
            from tablerag.bench.t2rag import build_t2, load_t2

            pdfs, questions = build_t2(load_t2(args.t2_contexts), args.data / "t2")
        print(f"{suite}: {len(pdfs)} PDFs, {len(questions)} questions")
        block = run_suite(suite, pdfs, questions, args.methods, args, reader, embed_model)
        answers[suite] = block.pop("_answers", {})
        results["suites"][suite] = block
        results["suites"][suite]["_questions"] = len(questions)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    (args.out / "summary.md").write_text(to_markdown(results))
    with (args.out / "answers.jsonl").open("w") as f:
        for suite, by_method in answers.items():
            for method, rows in by_method.items():
                for row in rows:
                    f.write(json.dumps({"suite": suite, "method": method, **row}) + "\n")
    print("\n" + to_markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
