"""How the methods are scored.

Ingestion (did the table survive?)
  row_intact          share of table rows whose values all sit together in one chunk
  row_with_context    ... in a chunk that also names the row's section (so it is unambiguous)

Retrieval (did the right row reach the reader?)
  found@1000/2000/4000  the evidence is inside the first N characters of retrieved text.
                      A budget in characters, not a number of chunks, keeps the comparison
                      fair: five 1000-character chunks hold far more than five rows. Evidence
                      is the whole row in one chunk (SpecQA) or text from the right document
                      (T²-RAGBench).

Answers (the part that matters when a number must be right)
  accuracy            correct answers / all questions
  wrong_rate          wrong answers / all questions: the dangerous case
  refusal_rate        "not found" / all questions: safe, but unhelpful
  precision           correct / answered: how far you can trust an answer that is given
SpecQA answers are compared exactly (numbers as numbers). T²-RAGBench uses its own Number
Match: relative tolerance 1e-2 and scale shifts such as 0.25 vs 25 (%).
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict
from collections.abc import Iterable

from tablerag.answer import NOT_FOUND, Answer, is_not_found
from tablerag.normalize import norm_key, numbers_in, same_value
from tablerag.types import Chunk


def _has_label(text: str, label: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(label)}(?![\d])", text, re.IGNORECASE) is not None


def chunk_has_row(chunk: Chunk, evidence: dict) -> bool:
    """True if one chunk holds the row name ("Stage 5") and every number of the evidence row."""
    if not _has_label(chunk.text, evidence["label"]):
        return False
    nums = set(numbers_in(chunk.text))
    return all(n in nums for n in evidence["numbers"])


def chunk_names_section(chunk: Chunk, evidence: dict) -> bool:
    return evidence["section"].lower() in chunk.text.lower()


def ingestion_scores(chunks: list[Chunk], rows: Iterable[tuple[str, dict]]) -> dict:
    """``rows``: (doc_id, evidence) for every ground-truth row."""
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_doc[c.doc_id].append(c)
    intact = ctx = n = 0
    for doc_id, ev in rows:
        n += 1
        holders = [c for c in by_doc[doc_id] if chunk_has_row(c, ev)]
        intact += bool(holders)
        ctx += any(chunk_names_section(c, ev) for c in holders)
    sizes = [len(c.text) for c in chunks] or [0]
    return {
        "rows": n,
        "row_intact": intact / n if n else 0.0,
        "row_with_context": ctx / n if n else 0.0,
        "chunks": len(chunks),
        "mean_chunk_chars": round(statistics.mean(sizes), 1),
    }


def rank_of(hits: list[Chunk], is_relevant) -> int | None:  # noqa: ANN001
    for i, c in enumerate(hits, 1):
        if is_relevant(c):
            return i
    return None


BUDGETS = (1000, 2000, 4000)


def chars_until(hits: list[Chunk], is_relevant) -> int | None:  # noqa: ANN001
    """Characters of retrieved text read before the evidence is complete (None if never)."""
    used = 0
    for c in hits:
        if is_relevant(c):
            # count only as much of this chunk as is needed to reach the end of the evidence
            for end in range(min(len(c.text), 200), len(c.text) + 200, 200):
                if is_relevant(Chunk(c.chunk_id, c.doc_id, c.text[:end])):
                    return used + min(end, len(c.text))
        used += len(c.text) + 2
    return None


def retrieval_scores(costs: list[int | None], budgets: tuple[int, ...] = BUDGETS) -> dict:
    n = len(costs) or 1
    out = {f"found@{b}": sum(1 for c in costs if c is not None and c <= b) / n for b in budgets}
    found = sorted(c for c in costs if c is not None)
    out["median_chars"] = found[len(found) // 2] if found else None
    return out


def spec_correct(question: dict, answer: str) -> bool:
    gold = question["answer"]
    if question["type"] == "range":
        want = numbers_in(gold)
        got = numbers_in(answer)
        return len(got) >= 2 and got[:2] == want
    if question["type"] == "flag":
        return norm_key(answer).startswith(gold)
    if question["type"] == "covers":
        return norm_key(answer) == norm_key(gold)
    got = numbers_in(answer)
    if numbers_in(gold) and got:
        return len(got) == 1 and same_value(str(got[0]), gold)
    return same_value(answer, gold)


def number_match(gold: str | float, answer: str, rel_tol: float = 1e-2) -> bool:
    """T²-RAGBench style: the first number of the answer matches within 1%, allowing scale shifts."""
    try:
        g = float(gold)
    except (TypeError, ValueError):
        nums = numbers_in(str(gold))
        if not nums:
            return norm_key(str(gold)) == norm_key(answer)
        g = nums[0]
    got = numbers_in(answer.replace("%", ""))
    if not got:
        return False
    p = got[0]
    for scale in (1, 100, 0.01, 1000, 0.001, 1e6, 1e-6):
        if abs(p * scale - g) <= rel_tol * max(abs(g), 1e-9):
            return True
    return False


def answer_scores(questions: list[dict], answers: list[Answer], correct) -> dict:  # noqa: ANN001
    n = len(questions) or 1
    right = wrong = refused = 0
    for q, a in zip(questions, answers, strict=True):
        if a.text == NOT_FOUND or is_not_found(a.text):
            refused += 1
        elif correct(q, a.text):
            right += 1
        else:
            wrong += 1
    answered = right + wrong
    return {
        "accuracy": right / n,
        "wrong_rate": wrong / n,
        "refusal_rate": refused / n,
        "precision": right / answered if answered else 0.0,
        "n": len(questions),
    }
