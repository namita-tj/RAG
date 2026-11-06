"""The data every ingestion method produces, so all methods can share one retriever, reader and scorer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """One retrievable piece of text.

    ``kind`` is "text", "table", "row" or "profile". ``row_ids`` lists the stored rows
    this chunk was built from (only the structure-aware method fills it), so an answer
    can be traced back to an exact cell.
    """

    chunk_id: str
    doc_id: str
    text: str
    pages: list[int] = field(default_factory=list)
    kind: str = "text"
    row_ids: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class StoredRow:
    """A table row kept exactly as printed, with where it came from."""

    row_id: str
    table_id: str
    cells: dict[str, str]  # column header -> raw cell text
    page: int
    label: str  # first column, the row's name ("Stage 3", "Revenue")

    def values(self) -> list[str]:
        return [v for v in self.cells.values() if v]


@dataclass
class StoredTable:
    table_id: str
    doc_id: str
    title: str
    context: str  # section path, e.g. "10.1 RI-2 Immunity to Radiated Fields (ALSE)"
    headers: list[str]
    rows: list[StoredRow]
    pages: list[int]

    @property
    def row_count(self) -> int:
        return len(self.rows)


@dataclass
class IngestResult:
    """What an ingestion method returns for one document."""

    doc_id: str
    chunks: list[Chunk]
    tables: list[StoredTable] = field(default_factory=list)
    seconds: float = 0.0
    pages: int = 0
    error: str | None = None
