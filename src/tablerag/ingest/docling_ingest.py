"""Docling as a baseline: IBM's open-source document converter with the TableFormer table model.

Docling's own recommended RAG setup is used: DocumentConverter, then HybridChunker, then
``contextualize`` (which prefixes each chunk with its headings). Needs ``pip install docling``
and downloads its models on first use, so this runs on Colab, not in CI.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from tablerag.types import Chunk, IngestResult

_converter: Any = None
_chunker: Any = None


def _setup() -> tuple[Any, Any]:
    global _converter, _chunker
    if _converter is None:
        from docling.chunking import HybridChunker
        from docling.document_converter import DocumentConverter

        _converter = DocumentConverter()
        _chunker = HybridChunker()
    return _converter, _chunker


def _pages(chunk: Any) -> list[int]:
    pages = set()
    for item in getattr(chunk.meta, "doc_items", []) or []:
        for prov in getattr(item, "prov", []) or []:
            pages.add(int(prov.page_no))
    return sorted(pages)


def ingest_docling(pdf_path: str | Path) -> IngestResult:
    start = time.perf_counter()
    path = Path(pdf_path)
    doc_id = path.stem
    try:
        converter, chunker = _setup()
        doc = converter.convert(str(path)).document
        chunks = []
        for i, ch in enumerate(chunker.chunk(dl_doc=doc)):
            kinds = {str(getattr(it, "label", "")) for it in getattr(ch.meta, "doc_items", []) or []}
            kind = "table" if any("table" in k.lower() for k in kinds) else "text"
            chunks.append(Chunk(f"{doc_id}#d{i}", doc_id, chunker.contextualize(chunk=ch), _pages(ch), kind))
        return IngestResult(doc_id, chunks, seconds=time.perf_counter() - start, pages=doc.num_pages())
    except Exception as exc:  # noqa: BLE001 - a failing baseline is reported, not fatal
        return IngestResult(doc_id, [], seconds=time.perf_counter() - start, error=f"{type(exc).__name__}: {exc}")
