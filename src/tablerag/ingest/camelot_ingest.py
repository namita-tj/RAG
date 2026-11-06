"""Camelot as a baseline: a widely used table extractor for ruled PDF tables (lattice mode).

Each table Camelot finds becomes a markdown chunk; the text around it is chunked like the
naive baseline. Camelot has no notion of tables continuing across pages.
"""

from __future__ import annotations

import time
from pathlib import Path

import pdfplumber

from tablerag.ingest.baselines import CHUNK_OVERLAP, CHUNK_SIZE, MAX_TABLE_CHARS, _split_markdown_table
from tablerag.ingest.common import split_text, to_markdown
from tablerag.types import Chunk, IngestResult


def ingest_camelot(pdf_path: str | Path, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> IngestResult:
    start = time.perf_counter()
    path = Path(pdf_path)
    doc_id = path.stem
    try:
        import camelot

        found = camelot.read_pdf(str(path), pages="all", flavor="lattice")
    except Exception as exc:  # noqa: BLE001
        return IngestResult(doc_id, [], seconds=time.perf_counter() - start, error=f"{type(exc).__name__}: {exc}")
    by_page: dict[int, list] = {}
    for t in found:
        by_page.setdefault(int(t.page), []).append(t)
    chunks: list[Chunk] = []
    with pdfplumber.open(path) as pdf:
        n_pages = len(pdf.pages)
        for page in pdf.pages:
            boxes = []
            for t in by_page.get(page.page_number, []):
                x1, y1, x2, y2 = t._bbox  # PDF coordinates, origin bottom-left
                boxes.append((x1, page.height - y2, x2, page.height - y1))
                rows = [[str(c).strip() for c in r] for r in t.df.values.tolist()]
                for piece in _split_markdown_table(to_markdown(rows), MAX_TABLE_CHARS):
                    chunks.append(Chunk(f"{doc_id}#c{len(chunks)}", doc_id, piece.strip(), [page.page_number], "table"))

            def outside(obj, boxes=boxes):  # noqa: ANN001
                if obj.get("object_type") != "char":
                    return True
                x, y = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
                return not any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes)

            text = page.filter(outside).extract_text() or ""
            for s, e in split_text(text, size, overlap):
                chunks.append(Chunk(f"{doc_id}#c{len(chunks)}", doc_id, text[s:e].strip(), [page.page_number]))
    return IngestResult(doc_id, chunks, seconds=time.perf_counter() - start, pages=n_pages)
