"""Helpers shared by the ingestion methods: a recursive text splitter and markdown tables."""

from __future__ import annotations

from collections.abc import Iterable

SEPARATORS = ("\n\n", "\n", " ")


def split_text(text: str, size: int = 1000, overlap: int = 200) -> list[tuple[int, int]]:
    """Character spans of chunks of at most ``size`` characters with ``overlap`` shared
    characters, cut at the largest separator that fits (paragraph, line, word), the way
    LangChain's RecursiveCharacterTextSplitter does. Returns (start, end) offsets."""
    if size <= overlap:
        raise ValueError("size must be larger than overlap")
    spans: list[tuple[int, int]] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            for sep in SEPARATORS:
                cut = text.rfind(sep, start + overlap + 1, end)
                if cut != -1:
                    end = cut + len(sep)
                    break
        if text[start:end].strip():
            spans.append((start, end))
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        # start the next chunk at a word boundary inside the overlap
        space = text.find(" ", nxt, end)
        start = space + 1 if space != -1 else nxt
    return spans


def pages_for_span(start: int, end: int, page_offsets: list[tuple[int, int]]) -> list[int]:
    """Pages whose text overlaps [start, end); ``page_offsets`` is a list of (page, offset)."""
    pages = []
    for i, (page, off) in enumerate(page_offsets):
        nxt = page_offsets[i + 1][1] if i + 1 < len(page_offsets) else float("inf")
        if off < end and nxt > start:
            pages.append(page)
    return pages


def md_cell(value: str) -> str:
    return (value or "").replace("|", "\\|").replace("\n", " ").strip()


def to_markdown(rows: Iterable[list[str]], header: list[str] | None = None) -> str:
    rows = [list(r) for r in rows]
    if header is None:
        if not rows:
            return ""
        header, rows = rows[0], rows[1:]
    width = max([len(header), *(len(r) for r in rows)]) if rows else len(header)
    header = header + [""] * (width - len(header))
    lines = ["| " + " | ".join(md_cell(c) for c in header) + " |", "|" + "---|" * width]
    for r in rows:
        r = r + [""] * (width - len(r))
        lines.append("| " + " | ".join(md_cell(c) for c in r) + " |")
    return "\n".join(lines)
