"""SpecQA: questions about synthetic datasheets whose every answer is known exactly.

The documents come from synthetic_docs.py. Questions are generated from the same objects
the PDFs are drawn from, so the ground truth is exact. Question types:

  value     What is the maximum pressure for Stage 5 of the PX-200?      one cell
  range     What is the speed range of Stage 5 of the PX-300, in rpm?   two numbers in one row
  flag      Is continuous duty allowed for Stage 2 of the PX-100?       a flag (text or ✓)
  covers    Which stage of the PX-200 covers 1450 rpm?                   a range lookup
  count     How many stages does the PX-400 define?                     the whole table
  max       What is the highest speed of the PX-500, in rpm?            the whole table
  run_in    What is the continuous run-in time of the PX-100, in hours? a second, small table

Each type has several phrasings, picked at random, so the exact operations in
structured_ops.py are not tuned to one sentence pattern. Row questions are marked
``cross_page`` when the row is printed on a continuation page.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from tablerag.bench.synthetic_docs import Section, Stage, build_doc, check_font, make_sections

RANGE_SEEDS = (7, 11, 23, 41, 53, 67)
MIN_MAX_SEEDS = (31, 89)
ROW_TYPES = {"value", "range", "flag", "covers"}


def _evidence(sec: Section, s: Stage) -> dict:
    return {"label": s.name, "numbers": [float(s.min_rpm), float(s.max_rpm), float(s.pressure)], "section": sec.code}


PHRASES = {
    "value": [
        "What is the maximum pressure in bar for {stage} of {model}?",
        "Up to what pressure (bar) may {stage} of {model} be used?",
        "{model}: maximum pressure of {stage} in bar?",
    ],
    "range": [
        "What is the speed range of {stage} of {model}, in rpm?",
        "Between which speeds (rpm) does {stage} of {model} run?",
        "{model}: speed range of {stage} in rpm?",
    ],
    "covers": [
        "Which stage of {model} covers {value} rpm?",
        "Which operating stage of {model} includes a speed of {value} rpm?",
        "{model} running at {value} rpm: which stage is that?",
    ],
    "flag": ["Is {duty} duty allowed for {stage} of {model}?", "Can {stage} of {model} be used for {duty} duty?"],
    "count": [
        "How many stages does {model} define?",
        "What is the number of operating stages listed for {model}?",
        "Count the stages of {model}.",
    ],
    "max": [
        "What is the highest speed of {model}, in rpm?",
        "What is the maximum speed in rpm that {model} reaches?",
        "Top speed of {model} in rpm?",
    ],
    "run_in": [
        "What is the continuous run-in time of {model}, in hours?",
        "How many hours of continuous run-in does {model} need?",
    ],
}


def questions_for(doc_id: str, sections: list[Section], rows_per_page: int, rng: random.Random) -> list[dict]:
    qs: list[dict] = []

    def add(qtype: str, question: str, answer: str, sec: Section, stage: Stage | None = None, idx: int = -1) -> None:
        qs.append(
            {
                "id": f"{doc_id}:{len(qs)}",
                "doc_id": doc_id,
                "type": qtype,
                "question": question,
                "answer": answer,
                "section": sec.code,
                "cross_page": idx >= rows_per_page,
                "page": sec.stage_pages[idx] if idx >= 0 else sec.start_page,
                "evidence": _evidence(sec, stage) if stage else None,
            }
        )

    def say(qtype: str, **kw: object) -> str:
        text = rng.choice(PHRASES[qtype]).format(**kw)
        return text[0].upper() + text[1:]

    for sec in sections:
        model = f"the {sec.code} ({sec.name})"
        n = len(sec.stages)
        first = [i for i in range(n) if i < rows_per_page]
        later = [i for i in range(n) if i >= rows_per_page]
        picks = rng.sample(first, min(2, len(first))) + rng.sample(later, min(2, len(later)))
        for j, i in enumerate(picks):
            s = sec.stages[i]
            if j % 2 == 0:
                add("value", say("value", stage=s.name, model=model), s.pressure, sec, s, i)
            else:
                add("range", say("range", stage=s.name, model=model), f"{s.min_rpm} - {s.max_rpm}", sec, s, i)
        for i in rng.sample(range(n), 2):
            s = sec.stages[i]
            value = (s.min_rpm + s.max_rpm) // 2
            if s.min_rpm < value < s.max_rpm:
                add("covers", say("covers", model=model, value=value), s.name, sec, s, i)
        i = rng.randrange(n)
        s = sec.stages[i]
        duty = rng.choice(["continuous", "intermittent"])
        allowed = s.continuous if duty == "continuous" else s.intermittent
        add("flag", say("flag", duty=duty, stage=s.name, model=model), "yes" if allowed else "no", sec, s, i)
        add("count", say("count", model=model), str(n), sec)
        add("max", say("max", model=model), str(max(s.max_rpm for s in sec.stages)), sec)
        if sec.run_in:
            add("run_in", say("run_in", model=model), sec.run_in[0], sec)
    return qs


def build_specqa(out_dir: Path, seed: int = 0) -> tuple[list[Path], list[dict]]:
    """Writes the PDFs and questions.jsonl into ``out_dir``; returns (pdf paths, questions)."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    plans = [(f"datasheet_{s}", make_sections(s), 5 + i % 3) for i, s in enumerate(RANGE_SEEDS)]
    if check_font() is not None:
        plans += [(f"datasheet_mm_{s}", make_sections(s, "min_max"), 5 + i) for i, s in enumerate(MIN_MAX_SEEDS)]
    pdfs, questions = [], []
    for i, (doc_id, sections, rpp) in enumerate(plans):
        pdf = out_dir / f"{doc_id}.pdf"
        build_doc(pdf, sections, rows_per_page=rpp, doc_id=f"DS-{1000 + i}")
        pdfs.append(pdf)
        questions += questions_for(doc_id, sections, rpp, rng)
    (out_dir / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in questions))
    return pdfs, questions
