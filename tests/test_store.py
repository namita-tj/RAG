"""The SQLite index: a stored and reloaded index must behave exactly like the in-memory one."""

import json
import shutil
import sqlite3

import numpy as np
import pytest

from tablerag import cli
from tablerag.ingest.structured import ingest_structured
from tablerag.retrieve import BM25, Dense
from tablerag.store import PIPELINE_VERSION, Store, file_hash, parse_cell
from tablerag.structured_ops import try_structured
from tablerag.types import IngestResult
from test_gpu_paths import HashEncoder

QUESTIONS = [
    "Which stage of the PX-100 covers 1450 rpm?",
    "What is the maximum pressure of Stage 4 of the PX-100 in bar?",
    "How many stages does the PX-100 have?",
    "What is the highest maximum pressure of the PX-200?",
    "Speed range of Stage 2 of the PX-300",
]


@pytest.fixture()
def indexed(tmp_path, datasheet, minmax_sheet):
    pdfs = [datasheet[0], minmax_sheet[0]]
    results = [ingest_structured(p) for p in pdfs]
    store = Store(tmp_path / "index.db")
    for p, r in zip(pdfs, results, strict=True):
        store.add(r, p, file_hash(p))
    yield store, results, pdfs
    store.close()


def test_round_trip_gives_the_same_objects(indexed):
    store, results, _ = indexed
    chunks, tables = store.load()
    assert chunks == [c for r in results for c in r.chunks]
    assert tables == [t for r in results for t in r.tables]


def test_retrieval_and_exact_answers_are_identical_after_reloading(indexed):
    store, results, _ = indexed
    mem_chunks = [c for r in results for c in r.chunks]
    mem_tables = [t for r in results for t in r.tables]
    chunks, tables = store.load()
    before, after = BM25(mem_chunks), BM25(chunks)
    for q in QUESTIONS:
        assert [(c.chunk_id, s) for c, s in before.search(q, 30)] == [(c.chunk_id, s) for c, s in after.search(q, 30)]
        a, b = try_structured(q, mem_tables), try_structured(q, tables)
        assert (a and (a.answer, [r.row_id for r in a.rows])) == (b and (b.answer, [r.row_id for r in b.rows]))


def test_stored_embeddings_give_the_same_dense_ranking(tmp_path, datasheet):
    path = datasheet[0]
    result = ingest_structured(path)
    enc = HashEncoder()
    vectors = enc.encode([c.text for c in result.chunks]).astype(np.float32)
    with Store(tmp_path / "e.db") as store:
        store.add(result, path, file_hash(path), vectors, "hash-64")
        chunks, _ = store.load()
        loaded = store.vectors(chunks, "hash-64")
        assert store.vectors(chunks, "other-model") is None
    before = Dense(result.chunks, enc, vectors=vectors)
    after = Dense(chunks, enc, vectors=loaded)
    for q in QUESTIONS:
        assert [c.chunk_id for c, _ in before.search(q, 30)] == [c.chunk_id for c, _ in after.search(q, 30)]
    with pytest.raises(ValueError):
        Dense(chunks, enc, vectors=loaded[:-1])


def test_incremental_indexing(tmp_path, datasheet, minmax_sheet):
    folder = tmp_path / "pdfs"
    folder.mkdir()
    doc = folder / "sheet.pdf"
    shutil.copy(datasheet[0], doc)
    with Store(tmp_path / "i.db") as store:
        assert store.index([doc], ingest_structured).added == ["sheet"]
        assert store.index([doc], ingest_structured).unchanged == ["sheet"]
        before = store.counts()
        shutil.copy(minmax_sheet[0], doc)  # same name, new content
        assert store.index([doc], ingest_structured).replaced == ["sheet"]
        after = store.counts()
        assert after["documents"] == 1 and after["rows"] != before["rows"]
        # nothing of the old version is left behind
        chunks, tables = store.load()
        assert chunks == ingest_structured(doc).chunks and len(tables) == len(ingest_structured(doc).tables)


def test_cells_are_queryable_with_sql(indexed):
    store, results, _ = indexed
    # the row whose speed range contains 1450 rpm, found with one SQL query
    found = store.db.execute(
        "SELECT r.label FROM cells c JOIN rows r USING (row_id) "
        "WHERE c.quantity = 'speed' AND c.base_lo <= 1450 AND c.base_hi >= 1450 AND r.table_id = ?",
        (results[0].tables[0].table_id,),
    ).fetchall()
    expected = try_structured("Which stage of the PX-100 covers 1450 rpm?", results[0].tables[:1])
    assert expected is not None and [f[0] for f in found] == [expected.answer]
    assert parse_cell("Speed [rpm]", "900 - 1500")[:3] == (None, 900.0, 1500.0)
    assert parse_cell("Max. pressure [bar]", "8")[0] == 8.0
    assert parse_cell("Notes", "see text") == (None, None, None, None, None, None)


def test_old_index_versions_are_refused(tmp_path):
    db = tmp_path / "old.db"
    Store(db).close()
    con = sqlite3.connect(db)
    con.execute("UPDATE meta SET value = 'old' WHERE key = 'pipeline_version'")
    con.commit()
    con.close()
    with pytest.raises(ValueError, match="pipeline version"):
        Store(db)
    with Store(tmp_path / "new.db") as store, pytest.raises(ValueError, match="model"):
        store.add(IngestResult("d", []), "d.pdf", "hash", vectors=[[0.0]], model=None)
    assert PIPELINE_VERSION


def test_cli_index_ask_info(tmp_path, datasheet, capsys):
    db = str(tmp_path / "cli.db")
    folder = tmp_path / "docs"
    folder.mkdir()
    shutil.copy(datasheet[0], folder / "sheet.pdf")
    assert cli.main(["index", str(folder), "--db", db]) == 0
    assert "added 1" in capsys.readouterr().out
    assert cli.main(["ask", "Which stage of the PX-100 covers 1450 rpm?", "--db", db, "--doc", "sheet", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["how"] == "structured" and out["source"][0]["document"] == "sheet" and out["source"][0]["page"] >= 1
    assert cli.main(["ask", "What is the maximum pressure of Stage 4 in bar?", "--db", db]) == 0
    assert "best evidence" in capsys.readouterr().out.lower()
    assert cli.main(["info", "--db", db]) == 0
    assert "1 documents" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(["ask", "x", "--db", db, "--retriever", "hybrid"])  # no embeddings in this index
    with pytest.raises(SystemExit):
        cli.main(["ask", "x", "--db", str(tmp_path / "missing.db")])
