"""Command line: index a folder of PDFs once, then ask questions against the saved index.

    tablerag index ./pdfs --db index.db                  # keyword search only, runs anywhere
    tablerag index ./pdfs --db index.db --embed-model BAAI/bge-small-en-v1.5   # adds embeddings
    tablerag ask "Maximum pressure of Stage 4 of the PX-100?" --db index.db
    tablerag ask "..." --db index.db --retriever hybrid --reader Qwen/Qwen2.5-3B-Instruct
    tablerag info --db index.db

Every answer comes with where it came from (document, page, row), or "NOT FOUND". Without a
reader model, the exact table operations still answer counts, ranges and maximums; for other
questions the command shows the best evidence instead of an answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tablerag.answer import NOT_FOUND, Answer, answer_questions, within_budget
from tablerag.ingest.structured import ingest_structured
from tablerag.query import Reranker, normalize_query
from tablerag.retrieve import BM25, Dense, Hybrid, prefixes
from tablerag.store import Store
from tablerag.structured_ops import try_structured
from tablerag.types import Chunk, StoredRow, StoredTable


def pdfs_in(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(x for x in p.rglob("*") if x.suffix.lower() == ".pdf")
        elif p.suffix.lower() == ".pdf":
            out.append(p)
        else:
            raise SystemExit(f"not a PDF or folder: {p}")
    return out


def load_encoder(name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit('embeddings need the optional packages: pip install -e ".[gpu]"') from None
    return SentenceTransformer(name)


def cmd_index(args: argparse.Namespace) -> int:
    files = pdfs_in(args.paths)
    if not files:
        raise SystemExit("no PDFs found")
    with Store(args.db) as store:
        embed, model = None, None
        if args.embed_model:
            encoder = load_encoder(args.embed_model)
            model = args.embed_model

            passage_prefix = prefixes(model)[1]

            def embed(texts: list[str]) -> Any:
                return encoder.encode([passage_prefix + t for t in texts], normalize_embeddings=True, batch_size=64)

        report = store.index(files, ingest_structured, embed, model)
        counts = store.counts()
    print(
        f"added {len(report.added)}, replaced {len(report.replaced)}, unchanged {len(report.unchanged)}, "
        f"failed {len(report.failed)}"
    )
    for doc, err in report.failed:
        print(f"  failed {doc}: {err}")
    print(", ".join(f"{v} {k}" for k, v in counts.items()))
    return 1 if report.failed else 0


def build_retriever(store: Store, chunks: list[Chunk], kind: str, rerank: str | None = None) -> Any:
    base = _base_retriever(store, chunks, kind)
    if rerank is None:
        return base
    try:
        return Reranker(base, rerank)
    except ImportError:
        raise SystemExit('the reranker needs the optional packages: pip install -e ".[gpu]"') from None


def _base_retriever(store: Store, chunks: list[Chunk], kind: str) -> Any:
    bm25 = BM25(chunks)
    if kind == "bm25":
        return bm25
    models = store.models()
    if not models:
        raise SystemExit("this index has no embeddings: index again with --embed-model, or use --retriever bm25")
    vectors = store.vectors(chunks, models[0])
    if vectors is None:
        raise SystemExit(f"some chunks have no {models[0]} embedding: index again with --embed-model {models[0]}")
    dense = Dense(chunks, load_encoder(models[0]), vectors=vectors, model_name=models[0])
    return dense if kind == "dense" else Hybrid(bm25, dense)


def source_of(row_ids: list[str], evidence: list[Chunk], rows: dict[str, StoredRow], tables: dict[str, StoredTable]):
    out = []
    for rid in row_ids:
        r = rows.get(rid)
        if r is None:
            continue
        t = tables[r.table_id]
        out.append(
            {
                "document": t.doc_id,
                "page": r.page,
                "table": t.title,
                "section": t.context,
                "row": rid,
                "cells": r.cells,
            }
        )
    if not out and evidence:
        c = evidence[0]
        out.append({"document": c.doc_id, "pages": c.pages, "chunk": c.chunk_id, "text": c.text[:300]})
    return out


def cmd_ask(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        chunks, tables = store.load()
        if not chunks:
            raise SystemExit(f"{args.db} is empty: run tablerag index first")
        retriever = build_retriever(store, chunks, args.retriever, args.rerank)
    rows = {r.row_id: r for t in tables for r in t.rows}
    by_id = {t.table_id: t for t in tables}
    q = {"question": args.question, "doc_id": args.doc}
    scoped = args.doc is not None
    if args.reader:
        from tablerag.answer import HFReader

        answer = answer_questions(
            [q], retriever, HFReader(args.reader), k=args.k, tables=tables, mode="ours", scoped=scoped
        )[0]
    else:
        question = normalize_query(args.question)
        hits = [c for c, _ in retriever.search(question, args.k, args.doc)]
        evidence = within_budget(hits, 2000)
        if scoped:
            candidates = [t for t in tables if t.doc_id == args.doc]
        else:
            seen = {c.meta.get("table_id") for c in evidence}
            candidates = [t for t in tables if t.table_id in seen]
        op = try_structured(question, candidates)
        if op is not None:
            answer = Answer(op.answer, "structured", evidence, [r.row_id for r in op.rows], op.explanation)
        else:
            answer = Answer(NOT_FOUND, "evidence only (no reader)", evidence)
    result = {
        "question": args.question,
        "answer": answer.text,
        "how": answer.route,
        "source": source_of(answer.rows, answer.evidence, rows, by_id),
    }
    if answer.route == "structured" and answer.raw:
        result["explanation"] = answer.raw
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_answer(result)
    return 0


def print_answer(result: dict) -> None:
    if result["how"].startswith("evidence only"):
        print("No reader model given (--reader), so this is the best evidence, not an answer:")
    else:
        print(f"Answer: {result['answer']}   ({result['how']})")
    if result.get("explanation"):
        print(f"Because: {result['explanation']}")
    for s in result["source"]:
        if "row" in s:
            print(f"Source: {s['document']}, page {s['page']}, {s['table'] or 'table'} ({s['section']})")
            print("  " + "; ".join(f"{k}: {v}" for k, v in s["cells"].items() if v))
        else:
            print(f"Best evidence: {s['document']}, pages {s['pages']}")
            print("  " + s["text"].replace("\n", " "))


def cmd_info(args: argparse.Namespace) -> int:
    with Store(args.db) as store:
        print(", ".join(f"{v} {k}" for k, v in store.counts().items()))
        print("embedding models:", ", ".join(store.models()) or "none")
        for doc, path, pages in store.documents():
            print(f"  {doc}: {pages} pages ({path})")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="tablerag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    i = sub.add_parser("index", help="add or update PDFs in the index")
    i.add_argument("paths", nargs="+", help="PDF files or folders")
    i.add_argument("--db", default="index.db")
    i.add_argument("--embed-model", default=None, help="also store embeddings, e.g. BAAI/bge-small-en-v1.5")
    i.set_defaults(func=cmd_index)
    a = sub.add_parser("ask", help="answer a question from the index")
    a.add_argument("question")
    a.add_argument("--db", default="index.db")
    a.add_argument("--doc", default=None, help="only search this document (its file name without .pdf)")
    a.add_argument("--retriever", default="bm25", choices=["bm25", "dense", "hybrid"])
    a.add_argument(
        "--rerank", default=None, help="cross-encoder that re-orders the top 30, e.g. BAAI/bge-reranker-base"
    )
    a.add_argument("--reader", default=None, help="Hugging Face model id, e.g. Qwen/Qwen2.5-3B-Instruct")
    a.add_argument("--k", type=int, default=30)
    a.add_argument("--json", action="store_true", help="print the answer as JSON")
    a.set_defaults(func=cmd_ask)
    n = sub.add_parser("info", help="what is in the index")
    n.add_argument("--db", default="index.db")
    n.set_defaults(func=cmd_info)
    args = p.parse_args(argv)
    if args.command != "index" and not Path(args.db).exists():
        raise SystemExit(f"{args.db} does not exist: run tablerag index first")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
