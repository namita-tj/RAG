"""A persistent index in one SQLite file: tables and rows, chunks and embeddings.

One relational database is the single source of truth, holding three views of the same data:

* the tables, their columns, rows and cells: what the exact operations and the evidence
  check read. Every cell keeps its text exactly as printed; parsed numbers and ranges
  (also converted to the base unit of their quantity) sit next to it for comparisons;
* the chunks and which rows each one was built from: what keyword search ranks;
* one embedding per chunk and model: what semantic search compares.

Retrieval does not change. ``load()`` returns the same Chunk and StoredTable objects that
ingestion produced, in the same order, so BM25 (our tokenizer and parameters) and the dense
index are rebuilt from them exactly as in memory. Dense search is brute force over all
vectors: exact, deterministic, and fast enough for this scale. ``tests/test_store.py``
checks that a stored and reloaded index gives the same answers as the in-memory one.

Re-indexing is incremental: a document whose file content has not changed is skipped, a
changed one is replaced in a single transaction, so the index is never half-updated.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from array import array
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tablerag.normalize import parse_number, parse_range, to_base, unit_in_header
from tablerag.types import Chunk, IngestResult, StoredRow, StoredTable

# Bump when ingestion or chunking changes, so an old index is rebuilt instead of mixed with new data.
PIPELINE_VERSION = "1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY, path TEXT NOT NULL, sha256 TEXT NOT NULL,
    pages INTEGER NOT NULL, indexed_at REAL NOT NULL, seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS tables (
    table_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents ON DELETE CASCADE,
    title TEXT NOT NULL, context TEXT NOT NULL, pages TEXT NOT NULL, seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS columns (
    table_id TEXT NOT NULL REFERENCES tables ON DELETE CASCADE, position INTEGER NOT NULL,
    name TEXT NOT NULL, unit TEXT NOT NULL, PRIMARY KEY (table_id, position)
);
CREATE TABLE IF NOT EXISTS rows (
    row_id TEXT PRIMARY KEY, table_id TEXT NOT NULL REFERENCES tables ON DELETE CASCADE,
    page INTEGER NOT NULL, label TEXT NOT NULL, seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS cells (
    row_id TEXT NOT NULL REFERENCES rows ON DELETE CASCADE, position INTEGER NOT NULL,
    raw TEXT NOT NULL,                -- exactly as printed
    num REAL, lo REAL, hi REAL,       -- a single number, or a range "lo - hi"
    quantity TEXT, base_lo REAL, base_hi REAL,   -- e.g. 'speed', converted to the base unit
    PRIMARY KEY (row_id, position)
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents ON DELETE CASCADE,
    kind TEXT NOT NULL, text TEXT NOT NULL, pages TEXT NOT NULL, meta TEXT NOT NULL, seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunk_rows (
    chunk_id TEXT NOT NULL REFERENCES chunks ON DELETE CASCADE, position INTEGER NOT NULL,
    row_id TEXT NOT NULL, PRIMARY KEY (chunk_id, position)
);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT NOT NULL REFERENCES chunks ON DELETE CASCADE, model TEXT NOT NULL,
    dim INTEGER NOT NULL, vector BLOB NOT NULL, PRIMARY KEY (chunk_id, model)
);
CREATE INDEX IF NOT EXISTS cells_range ON cells (quantity, base_lo, base_hi);
CREATE INDEX IF NOT EXISTS rows_table ON rows (table_id);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks (doc_id);
"""


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_cell(header: str, raw: str) -> tuple[Any, ...]:
    """(num, lo, hi, quantity, base_lo, base_hi) for one cell; None where it does not apply.
    ``quantity`` is what the unit measures ("speed", "pressure"); base_lo/base_hi are in its base unit."""
    num = parse_number(raw)
    rng = parse_range(raw)
    lo, hi = (rng[0], rng[1]) if rng else (num, num) if num is not None else (None, None)
    unit = (rng[2] if rng and rng[2] else "") or unit_in_header(header)
    quantity = base_lo = base_hi = None
    if lo is not None and unit:
        a, b = to_base(lo, unit), to_base(hi, unit)
        if a and b:
            quantity, base_lo, base_hi = a[0], a[1], b[1]
    return num, lo, hi, quantity, base_lo, base_hi


@dataclass
class IndexReport:
    added: list[str]
    replaced: list[str]
    unchanged: list[str]
    failed: list[tuple[str, str]]


class Store:
    """The index file. Use as a context manager, or call close()."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)
        version = self.get_meta("pipeline_version")
        if version is None:
            self.set_meta("pipeline_version", PIPELINE_VERSION)
        elif version != PIPELINE_VERSION:
            raise ValueError(
                f"{self.path} was built with pipeline version {version}, this code is version {PIPELINE_VERSION}. "
                "Delete the file and index again."
            )

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.db.close()

    # meta ---------------------------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    # writing ------------------------------------------------------------------------------
    def stored_hash(self, doc_id: str) -> str | None:
        row = self.db.execute("SELECT sha256 FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return row[0] if row else None

    def add(
        self,
        result: IngestResult,
        path: str | Path,
        sha256: str,
        vectors: Iterable[Iterable[float]] | None = None,
        model: str | None = None,
    ) -> None:
        """Writes one ingested document (replacing an older version) in a single transaction."""
        if vectors is not None and not model:
            raise ValueError("embeddings need the name of the model that made them")
        with self.db:
            self.db.execute("DELETE FROM documents WHERE doc_id = ?", (result.doc_id,))
            seq = self.db.execute("SELECT COALESCE(MAX(seq), -1) + 1 FROM documents").fetchone()[0]
            self.db.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                (result.doc_id, str(path), sha256, result.pages, time.time(), seq),
            )
            for t_seq, t in enumerate(result.tables):
                self.db.execute(
                    "INSERT INTO tables VALUES (?, ?, ?, ?, ?, ?)",
                    (t.table_id, t.doc_id, t.title, t.context, json.dumps(t.pages), t_seq),
                )
                self.db.executemany(
                    "INSERT INTO columns VALUES (?, ?, ?, ?)",
                    [(t.table_id, i, h, unit_in_header(h)) for i, h in enumerate(t.headers)],
                )
                for r_seq, r in enumerate(t.rows):
                    self.db.execute(
                        "INSERT INTO rows VALUES (?, ?, ?, ?, ?)", (r.row_id, t.table_id, r.page, r.label, r_seq)
                    )
                    self.db.executemany(
                        "INSERT INTO cells VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            (r.row_id, i, r.cells.get(h, ""), *parse_cell(h, r.cells.get(h, "")))
                            for i, h in enumerate(t.headers)
                        ],
                    )
            for c_seq, c in enumerate(result.chunks):
                self.db.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (c.chunk_id, c.doc_id, c.kind, c.text, json.dumps(c.pages), json.dumps(c.meta), c_seq),
                )
                self.db.executemany(
                    "INSERT INTO chunk_rows VALUES (?, ?, ?)", [(c.chunk_id, i, rid) for i, rid in enumerate(c.row_ids)]
                )
            if vectors is not None:
                rows = []
                for c, v in zip(result.chunks, vectors, strict=True):
                    vec = array("f", (float(x) for x in v))
                    rows.append((c.chunk_id, model, len(vec), vec.tobytes()))
                self.db.executemany("INSERT INTO embeddings VALUES (?, ?, ?, ?)", rows)

    def index(
        self,
        paths: Iterable[str | Path],
        ingest: Callable[[Path], IngestResult],
        embed: Callable[[list[str]], Iterable[Iterable[float]]] | None = None,
        model: str | None = None,
    ) -> IndexReport:
        """Ingests the given PDFs, skipping the ones whose content is already indexed."""
        report = IndexReport([], [], [], [])
        for p in map(Path, paths):
            digest = file_hash(p)
            old = self.stored_hash(p.stem)
            if old == digest:
                report.unchanged.append(p.stem)
                continue
            result = ingest(p)
            if result.error:
                report.failed.append((p.stem, result.error))
                continue
            vectors = embed([c.text for c in result.chunks]) if embed and result.chunks else None
            self.add(result, p, digest, vectors, model)
            (report.added if old is None else report.replaced).append(result.doc_id)
        return report

    # reading ------------------------------------------------------------------------------
    def documents(self) -> list[tuple[str, str, int]]:
        return self.db.execute("SELECT doc_id, path, pages FROM documents ORDER BY seq").fetchall()

    def counts(self) -> dict[str, int]:
        names = ["documents", "tables", "rows", "cells", "chunks", "embeddings"]
        return {n: self.db.execute(f"SELECT COUNT(*) FROM {n}").fetchone()[0] for n in names}  # noqa: S608

    def models(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT DISTINCT model FROM embeddings ORDER BY model")]

    def load(self) -> tuple[list[Chunk], list[StoredTable]]:
        """Everything, as the objects ingestion produced, in ingestion order."""
        row_ids: dict[str, list[str]] = {}
        for chunk_id, row_id in self.db.execute("SELECT chunk_id, row_id FROM chunk_rows ORDER BY chunk_id, position"):
            row_ids.setdefault(chunk_id, []).append(row_id)
        chunks = [
            Chunk(cid, doc, text, json.loads(pages), kind, row_ids.get(cid, []), json.loads(meta))
            for cid, doc, kind, text, pages, meta in self.db.execute(
                "SELECT c.chunk_id, c.doc_id, c.kind, c.text, c.pages, c.meta FROM chunks c "
                "JOIN documents d USING (doc_id) ORDER BY d.seq, c.seq"
            )
        ]
        headers: dict[str, list[str]] = {}
        for table_id, name in self.db.execute("SELECT table_id, name FROM columns ORDER BY table_id, position"):
            headers.setdefault(table_id, []).append(name)
        cells: dict[str, list[str]] = {}
        for row_id, raw in self.db.execute("SELECT row_id, raw FROM cells ORDER BY row_id, position"):
            cells.setdefault(row_id, []).append(raw)
        rows: dict[str, list[StoredRow]] = {}
        for row_id, table_id, page, label in self.db.execute(
            "SELECT row_id, table_id, page, label FROM rows ORDER BY table_id, seq"
        ):
            hs = headers.get(table_id, [])
            rows.setdefault(table_id, []).append(
                StoredRow(row_id, table_id, dict(zip(hs, cells.get(row_id, []), strict=True)), page, label)
            )
        tables = [
            StoredTable(tid, doc, title, context, headers.get(tid, []), rows.get(tid, []), json.loads(pages))
            for tid, doc, title, context, pages in self.db.execute(
                "SELECT t.table_id, t.doc_id, t.title, t.context, t.pages FROM tables t "
                "JOIN documents d USING (doc_id) ORDER BY d.seq, t.seq"
            )
        ]
        return chunks, tables

    def vectors(self, chunks: list[Chunk], model: str) -> list[array] | None:
        """The stored embeddings of these chunks for one model, in the same order; None if any is missing."""
        found = {
            cid: array("f", blob)
            for cid, blob in self.db.execute("SELECT chunk_id, vector FROM embeddings WHERE model = ?", (model,))
        }
        if not chunks or any(c.chunk_id not in found for c in chunks):
            return None
        return [found[c.chunk_id] for c in chunks]
