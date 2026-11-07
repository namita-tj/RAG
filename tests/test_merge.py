"""Merging runs: a rerun replaces a method, and every method is rescored the same way."""

import json

from tablerag.bench.merge import merge


def write_run(path, methods, answers):
    path.mkdir()
    suites = {"specqa": {m: {"answers": {"accuracy": -1}} for m in methods} | {"_questions": 2}}
    config = {"retriever": "bm25", "reader": "r", "context_chars": 2000, "methods": methods}
    (path / "results.json").write_text(json.dumps({"config": config, "suites": suites}))
    rows = [
        {"suite": "specqa", "method": m, "id": i, "type": "value", "answer": a, "route": "reader", "correct": c}
        for m, i, a, c in answers
    ]
    (path / "answers.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


def test_merge_replaces_and_rescores(tmp_path):
    write_run(
        tmp_path / "a",
        ["naive", "ours"],
        [
            ("naive", "q1", "8", True),
            ("naive", "q2", "NOT_FOUND", False),
            ("ours", "q1", "NOT FOUND", False),
            ("ours", "q2", "NOT FOUND", False),
        ],
    )
    write_run(tmp_path / "b", ["ours"], [("ours", "q1", "8", True), ("ours", "q2", "5", False)])
    results, flat = merge([tmp_path / "a", tmp_path / "b"], {"specqa": {"q2"}})
    naive, ours = results["suites"]["specqa"]["naive"]["answers"], results["suites"]["specqa"]["ours"]["answers"]
    assert naive["refusal_rate"] == 0.5 and naive["wrong_rate"] == 0.0  # "NOT_FOUND" is a refusal
    assert ours["accuracy"] == 0.5 and ours["wrong_rate"] == 0.5  # the rerun replaced the old answers
    assert ours["cross_page"]["n"] == 1
    assert len(flat) == 4 and results["config"]["methods"] == ["naive", "ours"]
