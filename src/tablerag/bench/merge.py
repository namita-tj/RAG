"""Merges benchmark runs and rescores every answer with the current scoring rules.

Used when only some methods were rerun (for example ``--methods ours`` after a change to
the answer step): methods in later runs replace the same methods in earlier runs. All
answer scores are recomputed from answers.jsonl, so every method is scored the same way.

    python -m tablerag.bench.merge results/gpu_full results/gpu_ours --out results/gpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tablerag.answer import is_not_found
from tablerag.bench.run import to_markdown

ORDER = [
    "naive",
    "markdown",
    "camelot",
    "docling",
    "ours_ingest_only",
    "ours_noverify",
    "ours_cite",
    "ours",
    "ablate_no_context",
    "ablate_no_stitch",
    "ablate_rows_only",
]


def scores(rows: list[dict]) -> dict:
    n = len(rows) or 1
    refused = sum(is_not_found(r["answer"]) for r in rows)
    right = sum(bool(r["correct"]) and not is_not_found(r["answer"]) for r in rows)
    wrong = len(rows) - refused - right
    return {
        "accuracy": right / n,
        "wrong_rate": wrong / n,
        "refusal_rate": refused / n,
        "precision": right / (right + wrong) if right + wrong else 0.0,
        "n": len(rows),
    }


def rescore(rows: list[dict], cross_page: set[str] | None) -> dict:
    out = scores(rows)
    by_type: dict[str, list[dict]] = {}
    routes: dict[str, int] = {}
    for r in rows:
        by_type.setdefault(r["type"], []).append(r)
        routes[r["route"]] = routes.get(r["route"], 0) + 1
    out["by_type"] = {t: scores(rs) for t, rs in by_type.items()}
    if cross_page is not None:
        out["cross_page"] = scores([r for r in rows if r["id"] in cross_page])
    out["routes"] = routes
    return out


def merge(runs: list[Path], cross_page: dict[str, set[str]] | None = None) -> tuple[dict, list[dict]]:
    results: dict[str, Any] = {"config": {}, "suites": {}}
    answers: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        res = json.loads((run / "results.json").read_text())
        methods = list(results["config"].get("methods", []))
        methods += [m for m in res["config"].get("methods", []) if m not in methods]
        results["config"] = {**res["config"], "methods": methods}
        for suite, block in res["suites"].items():
            results["suites"].setdefault(suite, {}).update(block)
        run_answers: dict[tuple[str, str], list[dict]] = {}
        for line in (run / "answers.jsonl").read_text().splitlines():
            row = json.loads(line)
            run_answers.setdefault((row["suite"], row["method"]), []).append(row)
        answers.update(run_answers)  # a rerun method replaces its earlier answers
    for (suite, method), rows in answers.items():
        block = results["suites"].get(suite, {})
        if method in block:
            cp = (cross_page or {}).get(suite)
            block[method]["answers"] = rescore(rows, cp)
    for suite, block in results["suites"].items():
        rank = {m: i for i, m in enumerate(ORDER)}
        keys = sorted((k for k in block if not k.startswith("_")), key=lambda k: rank.get(k, len(ORDER)))
        results["suites"][suite] = {
            **{k: block[k] for k in keys},
            **{k: v for k, v in block.items() if k.startswith("_")},
        }
    flat = [row for (_, _), rows in answers.items() for row in rows]
    return results, flat


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path, help="result folders, oldest first")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("benchmarks/data"), help="to mark SpecQA cross-page questions")
    args = p.parse_args(argv)
    cross_page = None
    try:
        from tablerag.bench.specqa import build_specqa

        _, questions = build_specqa(args.data / "specqa")
        cross_page = {"specqa": {q["id"] for q in questions if q.get("cross_page")}}
    except Exception as exc:  # noqa: BLE001 - the merge still works without the cross-page split
        print(f"cross-page split skipped: {exc}")
    results, flat = merge(args.runs, cross_page)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    (args.out / "summary.md").write_text(to_markdown(results))
    with (args.out / "answers.jsonl").open("w") as f:
        for row in flat:
            f.write(json.dumps(row) + "\n")
    print(to_markdown(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
