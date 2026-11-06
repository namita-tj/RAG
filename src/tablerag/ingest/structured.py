"""OUR METHOD: structure-aware table ingestion that never loses a cell.

What usually goes wrong when RAG pipelines ingest tables, and what this module does instead:

1. A table that continues on the next page becomes two unrelated pieces, and the second
   piece often has no header.  -> Continuation tables are stitched back together, and every
   row keeps the page it was printed on.
2. Chunking cuts through rows and separates values from their column names.
   -> Every row is stored as a record: each value next to its column header.
3. A row out of context is ambiguous ("Stage 5" exists in every section).
   -> Every row carries its context: document title, section heading and table caption.
4. Numbers are only matched as text ("which stage covers 1450 rpm?" has no 1450 in it).
   -> Ranges and numbers are parsed into exact values (normalize.py), and questions about
   ranges, counts and extremes are answered by computing on the stored rows instead of
   asking a language model (structured_ops.py).
5. An answer cannot be checked.  -> Each chunk lists the row ids it came from, so an answer
   can be traced to one cell on one page, or refused (answer.py).

The index has three granularities: one chunk per row, one per small table (for questions
that need several rows) and one profile per table (columns, row count, value ranges).
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from tablerag.ingest.baselines import CHUNK_OVERLAP, CHUNK_SIZE, MAX_TABLE_CHARS
from tablerag.ingest.common import split_text, to_markdown
from tablerag.normalize import CHECK_MARKS, clean, parse_number, parse_range, unit_in_header
from tablerag.pdf import RawTable, continues, read_pdf, remove_furniture
from tablerag.types import Chunk, IngestResult, StoredRow, StoredTable

_HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*\.?|[A-Z]\.\d+(?:\.\d+)*|Appendix\s+[A-Z])\s+[A-Z][^\n]{1,100}$")


def heading_in(text: str) -> str:
    """The last numbered heading in a block of text ("4.2 PX-200 - Standard model"), else ''.
    Table-of-contents lines (dot leaders) and table captions are not headings."""
    found = ""
    for line in text.splitlines():
        line = clean(line)
        if _HEADING_RE.match(line) and ".." not in line and not line.lower().startswith("table"):
            found = line
    return found


def _headers(raw: list[str], width: int) -> list[str]:
    """Usable, unique column names; empty or repeated headers get a position-based name."""
    out, seen = [], set()
    for i in range(width):
        h = clean(raw[i]) if i < len(raw) else ""
        h = h or f"Column {i + 1}"
        while h.lower() in seen:
            h = f"{h} ({i + 1})"
        seen.add(h.lower())
        out.append(h)
    return out


def _fmt(x: float) -> str:
    return str(int(x)) if x == int(x) else repr(x)


def derived_fields(header: str, value: str) -> dict[str, str]:
    """Extra, exact readings of a cell: '900 - 1500' under 'Speed [rpm]' also gives
    'Speed min [rpm]: 900' and 'Speed max [rpm]: 1500'; a check mark also reads 'yes'."""
    out: dict[str, str] = {}
    rng = parse_range(value)
    if rng:
        unit = rng[2] or unit_in_header(header)
        name = re.sub(r"\s*[\[(].*?[\])]", "", header).strip() or header
        name = re.sub(r"\s*range$", "", name, flags=re.IGNORECASE) or name
        suffix = f" [{unit}]" if unit else ""
        out[f"{name} min{suffix}"] = _fmt(rng[0])
        out[f"{name} max{suffix}"] = _fmt(rng[1])
    if clean(value).lower() in CHECK_MARKS and clean(value).lower() not in {"yes", "y"}:
        out[header] = "yes"
    return out


def row_text(prefix: str, row: StoredRow) -> str:
    cells = dict(row.cells)
    for h, v in row.cells.items():
        cells.update(derived_fields(h, v))
    body = "; ".join(f"{h}: {v}" for h, v in cells.items() if v)
    return f"{prefix}\n[{row.row_id}] {body}"


def profile_text(prefix: str, table: StoredTable) -> str:
    """A short, exact description of the whole table (no language model involved)."""
    parts = [prefix, f"Table with {table.row_count} rows and columns: {', '.join(table.headers)}."]
    labels = [r.label for r in table.rows if r.label]
    if labels:
        parts.append(f"Rows: {', '.join(labels[:40])}{' ...' if len(labels) > 40 else ''}.")
    for h in table.headers[1:]:
        vals = [r.cells.get(h, "") for r in table.rows]
        filled = [v for v in vals if v]
        nums = [n for n in (parse_number(v) for v in filled) if n is not None]
        rngs = [r for r in (parse_range(v) for v in filled) if r]
        if rngs and len(rngs) == len(filled):
            parts.append(f"{h} spans {_fmt(min(r[0] for r in rngs))} to {_fmt(max(r[1] for r in rngs))}.")
        elif nums and len(nums) == len(filled):
            parts.append(f"{h} ranges from {_fmt(min(nums))} to {_fmt(max(nums))}.")
        else:
            distinct = sorted({clean(v) for v in filled})
            if 0 < len(distinct) <= 8:
                parts.append(f"{h} values: {', '.join(distinct)}.")
    if len(table.pages) > 1:
        parts.append(f"Continues over pages {table.pages[0]}-{table.pages[-1]}.")
    return " ".join(parts)


class _Building:
    """A table being assembled from one or more pages."""

    def __init__(self, raw: RawTable, context: str):
        self.header = raw.rows[0]
        self.rows = [(r, raw.page) for r in raw.rows[1:]]
        self.caption, self.context, self.pages, self.width = raw.caption, context, [raw.page], raw.width

    def extend(self, raw: RawTable) -> None:
        same_header = [c.lower() for c in raw.rows[0]] == [c.lower() for c in self.header]
        self.rows += [(r, raw.page) for r in (raw.rows[1:] if same_header else raw.rows)]
        self.pages.append(raw.page)


def ingest_structured(
    pdf_path: str | Path,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    table_chunks: bool = True,
    profiles: bool = True,
    context: bool = True,
    stitch: bool = True,
) -> IngestResult:
    """Our method. The keyword switches turn single ideas off for the ablation study."""
    start = time.perf_counter()
    path = Path(pdf_path)
    doc_id = path.stem
    pages, title = read_pdf(path)
    pages = remove_furniture(pages)

    # 1) follow headings through the text in reading order, so a table on a continuation
    #    page still belongs to the section that started before it; stitch continuations
    building: list[_Building] = []
    heading = ""
    last_table: RawTable | None = None
    ends_page = False  # the last table seen was the last thing on its page
    new_heading = False  # a section heading was printed since the last table
    for page in pages:
        tables_on_page = page.tables()
        for kind, block in page.blocks:
            if kind == "text":
                found = heading_in(block)
                if found:
                    heading, new_heading = found, True
                continue
            first = block is tables_on_page[0]
            if stitch and last_table is not None and continues(last_table, block, first, ends_page, new_heading):
                building[-1].extend(block)
            else:
                building.append(_Building(block, heading))
            last_table, new_heading = block, False
        if page.blocks:
            ends_page = page.blocks[-1][0] == "table"

    # 2) rows as records with context
    stored: list[StoredTable] = []
    for t_idx, b in enumerate(x for x in building if x.rows):
        width = max(b.width, *(len(r) for r, _ in b.rows))
        headers = _headers(b.header, width)
        table_id = f"{doc_id}:t{t_idx + 1}"
        rows = []
        for r_idx, (cells, page_no) in enumerate(b.rows):
            values = [clean(c) for c in cells] + [""] * (width - len(cells))
            label = next((v for v in values if v), "")
            rows.append(
                StoredRow(f"{table_id}:r{r_idx + 1}", table_id, dict(zip(headers, values, strict=True)), page_no, label)
            )
        stored.append(StoredTable(table_id, doc_id, b.caption, b.context, headers, rows, sorted(set(b.pages))))

    # 3) chunks
    chunks: list[Chunk] = []
    for t in stored:
        prefix = " > ".join(p for p in (title, t.context, t.title) if p) if context else t.title
        meta = {"table_id": t.table_id}
        for r in t.rows:
            chunks.append(Chunk(r.row_id, doc_id, row_text(prefix, r), [r.page], "row", [r.row_id], meta))
        if table_chunks:
            md = to_markdown([list(r.cells.values()) for r in t.rows], t.headers)
            if len(md) <= MAX_TABLE_CHARS:
                chunks.append(
                    Chunk(
                        f"{t.table_id}:table",
                        doc_id,
                        f"{prefix}\n{md}",
                        t.pages,
                        "table",
                        [r.row_id for r in t.rows],
                        meta,
                    )
                )
        if profiles:
            chunks.append(Chunk(f"{t.table_id}:profile", doc_id, profile_text(prefix, t), t.pages, "profile", [], meta))

    # 4) the remaining text, without page furniture, with the document title
    for page in pages:
        text = page.text()
        for i, (s, e) in enumerate(split_text(text, size, overlap)):
            body = text[s:e].strip()
            if len(body) > 20:
                chunks.append(
                    Chunk(
                        f"{doc_id}#p{page.number}.{i}", doc_id, f"{title}\n{body}" if context else body, [page.number]
                    )
                )
    return IngestResult(doc_id, chunks, stored, time.perf_counter() - start, len(pages))
