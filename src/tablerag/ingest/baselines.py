"""The two common ways RAG pipelines ingest PDFs today, used as baselines.

* ``naive``    the page text (tables included, as plain text) cut into 1000-character
               chunks with 200 characters of overlap. This is what LangChain's
               PyPDFLoader + RecursiveCharacterTextSplitter does, and what the first
               version of this repository did.
* ``markdown`` text plus every table turned into a markdown table on its own page,
               the output of layout-aware parsers. A table stays in one chunk if it
               fits, otherwise it is cut between rows. Tables are not joined across
               pages and the header is not repeated in later pieces.
"""

from __future__ import annotations

import time
from pathlib import Path

import pdfplumber

from tablerag.ingest.common import pages_for_span, split_text, to_markdown
from tablerag.pdf import read_page
from tablerag.types import Chunk, IngestResult

CHUNK_SIZE, CHUNK_OVERLAP = 1000, 200
MAX_TABLE_CHARS = 2000


def ingest_naive(pdf_path: str | Path, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> IngestResult:
    start = time.perf_counter()
    path = Path(pdf_path)
    doc_id = path.stem
    parts, offsets, pos = [], [], 0
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = (page.extract_text() or "") + "\n\n"
            offsets.append((page.page_number, pos))
            parts.append(text)
            pos += len(text)
        n_pages = len(pdf.pages)
    full = "".join(parts)
    chunks = [
        Chunk(f"{doc_id}#n{i}", doc_id, full[s:e].strip(), pages_for_span(s, e, offsets))
        for i, (s, e) in enumerate(split_text(full, size, overlap))
    ]
    return IngestResult(doc_id, chunks, seconds=time.perf_counter() - start, pages=n_pages)


def _split_markdown_table(md: str, limit: int) -> list[str]:
    if len(md) <= limit:
        return [md]
    pieces, current = [], ""
    for line in md.splitlines():
        if current and len(current) + len(line) + 1 > limit:
            pieces.append(current)
            current = ""
        current += line + "\n"
    return [*pieces, current] if current else pieces


def ingest_markdown(
    pdf_path: str | Path, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP, max_table_chars: int = MAX_TABLE_CHARS
) -> IngestResult:
    start = time.perf_counter()
    path = Path(pdf_path)
    doc_id = path.stem
    chunks: list[Chunk] = []
    with pdfplumber.open(path) as pdf:
        n_pages = len(pdf.pages)
        buffer, buffer_pages = "", []

        def flush() -> None:
            nonlocal buffer, buffer_pages
            for s, e in split_text(buffer, size, overlap):
                chunks.append(Chunk(f"{doc_id}#m{len(chunks)}", doc_id, buffer[s:e].strip(), sorted(set(buffer_pages))))
            buffer, buffer_pages = "", []

        for page in pdf.pages:
            for kind, block in read_page(page).blocks:
                if kind == "table":
                    flush()
                    for piece in _split_markdown_table(to_markdown(block.rows), max_table_chars):
                        chunks.append(
                            Chunk(f"{doc_id}#m{len(chunks)}", doc_id, piece.strip(), [page.page_number], "table")
                        )
                else:
                    buffer += block + "\n\n"
                    buffer_pages.append(page.page_number)
        flush()
    return IngestResult(doc_id, chunks, seconds=time.perf_counter() - start, pages=n_pages)
