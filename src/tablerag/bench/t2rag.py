"""T²-RAGBench (Strich et al., 2026): public text-and-table questions from financial reports.

The benchmark (Hugging Face ``G4KMU/t2-ragbench``, CC-BY-4.0) gives each context as
markdown: report text plus a table, with numerical questions from FinQA and TAT-DQA. To test
ingestion from PDF, which is the step this project is about, each context is rendered back
into a PDF page (title, paragraphs, a ruled table that may break across pages with its header
repeated). Every method then has to read the same PDFs.

Note: the rendered tables are ruled, so this does not test borderless tables; those need a
layout model such as Docling's TableFormer.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SUBSETS = ("FinQA", "TAT-DQA")


def parse_markdown_context(context: str) -> list[tuple[str, object]]:
    """[('text', str) | ('table', [[cell, ...], ...])] in reading order."""
    blocks: list[tuple[str, object]] = []
    table: list[list[str]] = []
    para: list[str] = []

    def flush_para():
        if para:
            blocks.append(("text", " ".join(para)))
            para.clear()

    for line in context.splitlines():
        s = line.strip()
        if s.startswith("|"):
            flush_para()
            if re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", s):
                continue
            table.append([c.strip() for c in s.strip("|").split("|")])
            continue
        if table:
            blocks.append(("table", table))
            table = []
        if s:
            para.append(s)
        else:
            flush_para()
    if table:
        blocks.append(("table", table))
    flush_para()
    return blocks


def render_pdf(path: Path, title: str, context: str) -> None:
    styles = getSampleStyleSheet()
    cell = styles["BodyText"].clone("cell", fontSize=7, leading=8)
    story = [Paragraph(escape(title), styles["Heading2"]), Spacer(1, 6)]
    for kind, content in parse_markdown_context(context):
        if kind == "text":
            story += [Paragraph(escape(str(content)), styles["BodyText"]), Spacer(1, 4)]
            continue
        rows = content  # type: ignore[assignment]
        width = max(len(r) for r in rows)
        data = [[Paragraph(escape(c), cell) for c in r + [""] * (width - len(r))] for r in rows]
        t = Table(data, repeatRows=1, colWidths=[480 / width] * width)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.black), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story += [t, Spacer(1, 6)]
    path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(path), pagesize=A4, title=title).build(story)


def load_t2(n_contexts: int = 150, seed: int = 0, split: str = "test") -> list[dict]:
    """Rows of the benchmark for ``n_contexts`` random contexts per subset (needs ``datasets``)."""
    from datasets import load_dataset

    rng = random.Random(seed)
    out = []
    for subset in SUBSETS:
        ds = load_dataset("G4KMU/t2-ragbench", subset, split=split)
        by_ctx: dict[str, list[dict]] = {}
        for row in ds:
            by_ctx.setdefault(str(row["context_id"]), []).append(row)
        ids = sorted(by_ctx)
        for cid in rng.sample(ids, min(n_contexts, len(ids))):
            for row in by_ctx[cid]:
                out.append({**row, "subset": subset})
    return out


def build_t2(rows: list[dict], out_dir: Path) -> tuple[list[Path], list[dict]]:
    """Renders one PDF per context and returns (pdfs, questions)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs, questions, seen = [], [], {}
    for row in rows:
        key = f"{row['subset']}|{row['context_id']}"
        if key not in seen:
            doc_id = f"t2_{row['subset'].replace('-', '').lower()}_{len(seen)}"
            seen[key] = doc_id
            title = " ".join(str(x) for x in (row.get("company_name"), row.get("report_year")) if x) or "Annual report"
            pdf = out_dir / f"{doc_id}.pdf"
            render_pdf(pdf, title, row["context"])
            pdfs.append(pdf)
        questions.append(
            {
                "id": str(row["id"]),
                "doc_id": seen[key],
                "type": row["subset"],
                "question": row["question"],
                "answer": str(row["program_answer"]),
                "original_answer": str(row.get("original_answer", "")),
            }
        )
    (out_dir / "questions.jsonl").write_text("\n".join(json.dumps(q) for q in questions) + "\n")
    return pdfs, questions
