"""Metrics, the public-benchmark renderer and a small end-to-end run."""

import json

from tablerag.answer import NOT_FOUND, Answer
from tablerag.bench import run
from tablerag.bench.metrics import answer_scores, chars_until, chunk_has_row, number_match, spec_correct
from tablerag.bench.t2rag import parse_markdown_context, render_pdf
from tablerag.ingest.structured import ingest_structured
from tablerag.types import Chunk

EV = {"label": "Stage 5", "numbers": [900.0, 1200.0, 8.0], "section": "PX-200"}


def test_evidence_matching_is_strict():
    assert chunk_has_row(Chunk("a", "d", "Stage 5 | 900 - 1200 | 8"), EV)
    assert not chunk_has_row(Chunk("a", "d", "Stage 15 | 900 - 1200 | 8"), EV)
    assert not chunk_has_row(Chunk("a", "d", "Stage 5 | 900 - 1200"), EV)


def test_chars_until_counts_only_what_must_be_read():
    hits = [Chunk("a", "d", "x" * 500), Chunk("b", "d", "Stage 5 | 900 - 1200 | 8 " + "y" * 1000)]
    used = chars_until(hits, lambda c: chunk_has_row(c, EV))
    assert 500 < used <= 502 + 200
    assert chars_until(hits[:1], lambda c: chunk_has_row(c, EV)) is None


def test_answer_scoring():
    assert spec_correct({"type": "range", "answer": "900 - 1200"}, "900-1200 rpm")
    assert not spec_correct({"type": "range", "answer": "900 - 1200"}, "900 - 1300")
    assert spec_correct({"type": "value", "answer": "8"}, "8 bar") and not spec_correct(
        {"type": "value", "answer": "8"}, "80"
    )
    assert spec_correct({"type": "flag", "answer": "yes"}, "Yes, it is")
    assert number_match("0.25", "25%") and number_match("1234", "1,234.4") and not number_match("12", "13")
    qs = [{"type": "value", "answer": "8"}] * 3
    s = answer_scores(qs, [Answer("8", "r"), Answer("9", "r"), Answer(NOT_FOUND, "refused")], spec_correct)
    assert (s["accuracy"], s["wrong_rate"], s["refusal_rate"], s["precision"]) == (1 / 3, 1 / 3, 1 / 3, 0.5)


def test_rendered_public_context_round_trips_exactly(tmp_path):
    context = (
        "Revenue grew in 2019.\n\n| Item | 2019 | 2018 |\n|---|---|---|\n| Revenue | 1,234 | 1,100 |\n"
        "| Net income | (56) | 40 |\n\nAll figures in millions."
    )
    assert [k for k, _ in parse_markdown_context(context)] == ["text", "table", "text"]
    pdf = tmp_path / "ctx.pdf"
    render_pdf(pdf, "Example Corp 2019", context)
    table = ingest_structured(pdf).tables[0]
    assert table.headers == ["Item", "2019", "2018"]
    assert [list(r.cells.values()) for r in table.rows] == [["Revenue", "1,234", "1,100"], ["Net income", "(56)", "40"]]


def test_benchmark_runs_end_to_end(tmp_path):
    assert (
        run.main(
            ["--methods", "naive", "markdown", "ours", "--data", str(tmp_path / "data"), "--out", str(tmp_path / "out")]
        )
        == 0
    )
    res = json.loads((tmp_path / "out" / "results.json").read_text())
    spec = res["suites"]["specqa"]
    assert spec["ours"]["ingestion"]["row_with_context"] == 1.0
    sp = spec["ours"]["structured_path"]
    assert sp["questions_taken"] > 0 and sp["correct"] == sp["questions_taken"]  # never wrong
    assert spec["ours"]["retrieval"]["found@2000"] > spec["naive"]["retrieval"]["found@2000"]
    assert "SpecQA" in (tmp_path / "out" / "summary.md").read_text()
