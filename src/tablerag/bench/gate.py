"""Benchmark gate: compares a benchmark run with the committed baseline and fails on a regression.

CI runs the CPU benchmark on every push and pull request, then this script. It checks the
metrics that define the method (rows kept whole and in context, retrieval within the character
budget, the exact path never being wrong) and fails if any of them got worse. It also writes a
comparison table, which GitHub shows on the run's summary page.

    python -m tablerag.bench.gate --baseline results/cpu/results.json --current out/results.json

The CPU benchmark is deterministic, so the default tolerance is zero: any drop fails. An
intended change of the numbers is made by updating ``results/cpu/`` in the same pull request,
which makes the change visible in the diff:

    python -m tablerag.bench.run --out results/cpu
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from tablerag.bench.run import LABELS

# (section, metric, label). All are "higher is better" and gated for our method.
GATED = [
    ("ingestion", "row_intact", "Rows kept whole"),
    ("ingestion", "row_with_context", "Rows kept whole with their section"),
    ("retrieval", "found@1000", "Evidence within 1000 chars"),
    ("retrieval", "found@2000", "Evidence within 2000 chars"),
    ("retrieval", "found@4000", "Evidence within 4000 chars"),
    ("retrieval_cross_page", "found@2000", "Continuation-page rows within 2000 chars"),
    ("structured_path", "precision", "Exact path precision"),
    ("structured_path", "coverage", "Questions the exact path answers"),
]
# shown for the baselines too, but not gated: a change there means the benchmark itself changed
SHOWN = [("retrieval", "found@2000", "Evidence within 2000 chars")]
GATED_METHOD = "ours"


@dataclass
class Row:
    suite: str
    method: str
    label: str
    baseline: float | None
    current: float | None
    gated: bool
    tolerance: float = 0.0

    @property
    def status(self) -> str:
        if self.current is None:
            return "missing" if self.gated else "-"
        if self.baseline is None:
            return "new"
        if self.current < self.baseline - self.tolerance:
            return "regression" if self.gated else "changed"
        if self.current > self.baseline + self.tolerance:
            return "better" if self.gated else "changed"
        return "ok"


def metric(entry: dict, section: str, name: str) -> float | None:
    block = entry.get(section)
    if not isinstance(block, dict):
        return None
    if section == "structured_path" and name == "coverage":
        return block["questions_taken"] / block["of"] if block.get("of") else None
    value = block.get(name)
    return float(value) if isinstance(value, int | float) else None


def compare(baseline: dict, current: dict, tolerance: float = 0.0) -> tuple[list[Row], list[str]]:
    rows: list[Row] = []
    problems: list[str] = []
    for suite, block in current["suites"].items():
        base_block = baseline.get("suites", {}).get(suite, {})
        for method, entry in block.items():
            if method.startswith("_"):
                continue
            errors = entry.get("ingestion", {}).get("errors", 0)
            if errors:
                problems.append(f"{suite}/{method}: {errors} documents failed to ingest")
            base = base_block.get(method)
            if base is None:
                continue
            checks = GATED if method == GATED_METHOD else SHOWN
            for section, name, label in checks:
                rows.append(
                    Row(
                        suite,
                        method,
                        label,
                        metric(base, section, name),
                        metric(entry, section, name),
                        method == GATED_METHOD,
                        tolerance,
                    )
                )
        if GATED_METHOD in base_block and GATED_METHOD not in block:
            problems.append(f"{suite}: {GATED_METHOD} is in the baseline but was not run")
    for r in rows:
        if r.status in {"regression", "missing"}:
            problems.append(
                f"{r.suite}/{r.method}: {r.label} {'is missing' if r.current is None else 'dropped'}"
                + ("" if r.current is None or r.baseline is None else f" from {pct(r.baseline)} to {pct(r.current)}")
            )
    return rows, problems


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def change(r: Row) -> str:
    if r.current is None or r.baseline is None:
        return "-"
    d = 100 * (r.current - r.baseline)
    return "0.0" if abs(d) < 0.05 else f"{d:+.1f}"


ICONS = {"ok": "✅ ok", "better": "⬆️ better", "regression": "❌ regression", "missing": "❌ missing"}


def to_markdown(rows: list[Row], problems: list[str]) -> str:
    lines = ["## Benchmark gate", ""]
    if problems:
        lines += ["**Failed.** These got worse compared with `results/cpu/results.json`:", ""]
        lines += [f"- {p}" for p in problems]
    else:
        lines.append("**Passed.** Nothing got worse compared with `results/cpu/results.json`.")
    if any(r.status == "better" for r in rows):
        lines += [
            "",
            "Some numbers improved. If that's intended, update the baseline in this pull request: "
            "`python -m tablerag.bench.run --out results/cpu`.",
        ]
    lines += ["", "| Suite | Method | Metric | Baseline | This run | Change (points) | Status |"]
    lines += ["| --- | --- | --- | ---: | ---: | ---: | --- |"]
    for r in rows:
        status = ICONS.get(r.status, r.status) if r.gated else ("info" if r.status in {"ok", "-"} else r.status)
        lines.append(
            f"| {r.suite} | {LABELS.get(r.method, r.method)} | {r.label} | {pct(r.baseline)} | {pct(r.current)} | "
            f"{change(r)} | {status} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", type=Path, default=Path("results/cpu/results.json"))
    p.add_argument("--current", type=Path, required=True)
    p.add_argument("--tolerance", type=float, default=0.0, help="allowed drop, as a fraction (0.01 = 1 point)")
    p.add_argument(
        "--summary", type=Path, default=None, help="file to append the table to (default: $GITHUB_STEP_SUMMARY)"
    )
    args = p.parse_args(argv)
    rows, problems = compare(
        json.loads(args.baseline.read_text()), json.loads(args.current.read_text()), args.tolerance
    )
    report = to_markdown(rows, problems)
    print(report)
    summary = args.summary or (
        Path(os.environ["GITHUB_STEP_SUMMARY"]) if os.environ.get("GITHUB_STEP_SUMMARY") else None
    )
    if summary:
        with summary.open("a", encoding="utf-8") as f:
            f.write(report)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
