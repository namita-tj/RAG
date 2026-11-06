"""Reading a PDF page by page: text blocks and ruled tables in reading order, plus the
generic clean-up every method shares (running headers and footers, repeated letterhead
tables, empty rows and columns) and the detection of tables that continue on the next page.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber

from tablerag.normalize import clean

TABLE_SETTINGS = {"vertical_strategy": "lines", "horizontal_strategy": "lines", "snap_tolerance": 3}
CAPTION_RE = re.compile(r"^(Table\s+[\w.\-]+)\s*[:\-.]?\s*(.*)$", re.IGNORECASE)
CAPTION_SEARCH_PT = 60  # how far above a table a caption may be
REPEATED_SHARE = 0.5  # a line or table on at least half of the pages (and >= 3) is page furniture
_DIGITS = re.compile(r"\d+")


@dataclass
class RawTable:
    page: int
    top: float
    rows: list[list[str]]  # first row is the header as printed
    caption: str = ""

    @property
    def width(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    def signature(self) -> str:
        return _DIGITS.sub("#", "|".join("|".join(r) for r in self.rows).lower())


@dataclass
class Page:
    number: int
    blocks: list[tuple[str, Any]] = field(default_factory=list)  # ("text", str) | ("table", RawTable)

    def text(self) -> str:
        return "\n".join(b for k, b in self.blocks if k == "text")

    def tables(self) -> list[RawTable]:
        return [b for k, b in self.blocks if k == "table"]


def clean_cell(cell: Any) -> str:
    return clean("" if cell is None else str(cell))


def _caption(lines: list[dict[str, Any]], top: float) -> str:
    above = sorted(
        (ln for ln in lines if ln["bottom"] <= top + 2 and top - ln["bottom"] < CAPTION_SEARCH_PT),
        key=lambda ln: -ln["bottom"],
    )
    for ln in above:
        m = CAPTION_RE.match(ln["text"].strip())
        if m:
            return clean(f"{m.group(1)}: {m.group(2)}" if m.group(2) else m.group(1))
    return ""


def _lines(page: Any) -> list[dict[str, Any]]:
    words = sorted(page.extract_words(x_tolerance=3, y_tolerance=3), key=lambda w: (round(w["top"]), w["x0"]))
    lines: list[dict[str, Any]] = []
    for w in words:
        if lines and abs(w["top"] - lines[-1]["top"]) < 5:
            lines[-1]["text"] += " " + w["text"]
            lines[-1]["bottom"] = max(lines[-1]["bottom"], w["bottom"])
        else:
            lines.append({"text": w["text"], "top": w["top"], "bottom": w["bottom"]})
    return lines


def read_page(page: Any) -> Page:
    """Text between tables and the tables themselves, top to bottom."""
    found = [t for t in page.find_tables(table_settings=TABLE_SETTINGS) if t.extract()]
    boxes = [t.bbox for t in found]

    def outside(obj: dict[str, Any]) -> bool:
        if obj.get("object_type") != "char":
            return True
        x, y = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
        return not any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes)

    items: list[tuple[float, str, Any]] = []
    text_page = page.filter(outside)
    cuts = [0.0, *sorted(b[1] for b in boxes), float(page.height)]
    for top, bottom in zip(cuts, cuts[1:], strict=False):
        if bottom - top >= 1:
            text = (text_page.crop((0, top, page.width, min(bottom, page.height))).extract_text() or "").strip()
            if text:
                items.append((top, "text", text))
    lines = _lines(page)
    for t in found:
        rows = [[clean_cell(c) for c in r] for r in t.extract()]
        rows = [r for r in rows if any(r)]
        if rows:
            items.append(
                (t.bbox[1] + 0.5, "table", RawTable(page.page_number, t.bbox[1], rows, _caption(lines, t.bbox[1])))
            )
    return Page(page.page_number, [(k, v) for _, k, v in sorted(items, key=lambda x: x[0])])


def read_pdf(path: str | Path) -> tuple[list[Page], str]:
    """All pages and the document title (PDF metadata, else the first line)."""
    with pdfplumber.open(path) as pdf:
        pages = [read_page(p) for p in pdf.pages]
        title = clean((pdf.metadata or {}).get("Title", "") or "")
    if not title:
        first = next((p.text() for p in pages if p.text().strip()), "")
        title = clean(first.splitlines()[0]) if first else Path(path).stem
    return pages, title


# ── page furniture ────────────────────────────────────────────


def _norm_line(line: str) -> str:
    return _DIGITS.sub("#", clean(line).lower())


def remove_furniture(pages: list[Page]) -> list[Page]:
    """Drops running headers/footers ("Page 3 of 9") and tables repeated on many pages
    (letterheads), which otherwise end up in every chunk."""
    if len(pages) < 3:
        return pages
    need = max(3, REPEATED_SHARE * len(pages))
    line_counts: Counter[str] = Counter()
    table_counts: Counter[str] = Counter()
    for p in pages:
        line_counts.update({_norm_line(ln) for ln in p.text().splitlines() if ln.strip()})
        table_counts.update({t.signature() for t in p.tables()})
    lines = {ln for ln, n in line_counts.items() if n >= need}
    tables = {s for s, n in table_counts.items() if n >= need}
    out = []
    for p in pages:
        blocks: list[tuple[str, Any]] = []
        for kind, b in p.blocks:
            if kind == "text":
                kept = "\n".join(ln for ln in b.splitlines() if ln.strip() and _norm_line(ln) not in lines)
                if kept:
                    blocks.append(("text", kept))
            elif b.signature() not in tables:
                blocks.append(("table", b))
        out.append(Page(p.number, blocks))
    return out


# ── tables across pages ───────────────────────────────────────


def _same_row(a: list[str], b: list[str]) -> bool:
    return [c.lower() for c in a] == [c.lower() for c in b] and any(a)


def _label(caption: str) -> str:
    m = CAPTION_RE.match(caption)
    return m.group(1).lower() if m else ""


def continues(
    prev: RawTable, curr: RawTable, curr_is_first: bool, prev_is_last: bool, new_heading: bool = False
) -> bool:
    """``curr`` continues ``prev`` when all of these hold:

    * ``curr`` is the first table of the next page and ``prev`` the last of its page;
    * no new section heading was printed between them, and ``curr`` has no caption of its own
      ("Table 4.2-1" after "Table 4.1-1" is a new table, even with the same columns);
    * same number of columns, and ``curr`` repeats the header row or is captioned
      "(continued)" / "cont." (then its first row is already data).
    """
    if not (curr_is_first and prev_is_last) or new_heading:
        return False
    if curr.page != prev.page + 1 or curr.width != prev.width:
        return False
    marked = bool(re.search(r"continued|\bcont\.", curr.caption, re.IGNORECASE))
    if curr.caption and not marked and _label(curr.caption) != _label(prev.caption):
        return False
    return marked or _same_row(prev.rows[0], curr.rows[0])
