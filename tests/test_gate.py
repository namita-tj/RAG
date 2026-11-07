"""The benchmark gate: passes on equal numbers, fails on a drop, never fails on a baseline method."""

import copy
import json

from tablerag.bench.gate import compare, main


def run(found=0.92, precision=1.0, naive=0.70, errors=0):
    ours = {
        "ingestion": {"row_intact": 1.0, "row_with_context": 1.0, "errors": errors},
        "retrieval": {"found@1000": 0.77, "found@2000": found, "found@4000": 1.0},
        "retrieval_cross_page": {"found@2000": 0.88},
        "structured_path": {"questions_taken": 192, "of": 445, "correct": 192, "precision": precision},
    }
    base = {"ingestion": {"errors": 0}, "retrieval": {"found@2000": naive}}
    return {"suites": {"specqa": {"naive": base, "ours": ours, "_questions": 445}}}


def test_equal_numbers_pass():
    rows, problems = compare(run(), run())
    assert problems == [] and all(r.status == "ok" for r in rows)


def test_a_drop_in_our_method_fails():
    _, problems = compare(run(), run(found=0.90))
    assert problems == ["specqa/ours: Evidence within 2000 chars dropped from 92.0% to 90.0%"]
    _, problems = compare(run(), run(precision=0.99))
    assert len(problems) == 1 and "Exact path precision" in problems[0]


def test_tolerance_and_improvements():
    assert compare(run(), run(found=0.915), tolerance=0.01)[1] == []
    rows, problems = compare(run(), run(found=0.95))
    assert problems == [] and any(r.status == "better" for r in rows)


def test_baselines_and_failures():
    rows, problems = compare(run(), run(naive=0.50))
    assert problems == [] and any(r.status == "changed" for r in rows)  # shown, not gated
    _, problems = compare(run(), run(errors=2))
    assert problems == ["specqa/ours: 2 documents failed to ingest"]
    missing = copy.deepcopy(run())
    del missing["suites"]["specqa"]["ours"]
    assert compare(run(), missing)[1] == ["specqa: ours is in the baseline but was not run"]


def test_main_writes_the_summary_and_sets_the_exit_code(tmp_path):
    base, cur, summary = tmp_path / "base.json", tmp_path / "cur.json", tmp_path / "summary.md"
    base.write_text(json.dumps(run()))
    cur.write_text(json.dumps(run()))
    assert main(["--baseline", str(base), "--current", str(cur), "--summary", str(summary)]) == 0
    assert "Passed" in summary.read_text()
    cur.write_text(json.dumps(run(found=0.5)))
    assert main(["--baseline", str(base), "--current", str(cur), "--summary", str(summary)]) == 1
    assert "Failed" in summary.read_text()
