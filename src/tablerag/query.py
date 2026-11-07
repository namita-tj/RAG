"""Query handling before retrieval: a deterministic clean-up, and an optional reranker.

Clean-up (always on). Users write numbers and units in many ways; the documents print them
one way. ``normalize_query`` rewrites only a number followed by a unit, so nothing else in
the question changes:

    "1,450 revolutions per minute" -> "1450 rpm"      "5 bars" -> "5 bar"
    "80 degrees Celsius"           -> "80 °C"         "12 percent" -> "12 %"

Reranker (optional). Hybrid search scores the query and each chunk separately. A
cross-encoder reads the query and a chunk together, which ranks near-identical rows
("Stage 4" vs "Stage 14") better. ``Reranker`` re-orders the top candidates of any retriever
with such a model, for example ``BAAI/bge-reranker-base``; it needs the [gpu] extra.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from tablerag.normalize import clean
from tablerag.types import Chunk

# spelled-out unit -> the symbol datasheets print. Only applied right after a number.
UNIT_WORDS = {
    r"revolutions? per minute|rotations? per minute|rev/min|r/min": "rpm",
    r"degrees? celsius|deg\.? ?c": "°C",
    r"per ?cent|percent": "%",
    r"kilowatts?": "kW",
    r"watts?": "W",
    r"kilovolts?": "kV",
    r"millivolts?": "mV",
    r"volts?": "V",
    r"milliamps?|milliamperes?": "mA",
    r"amps?|amperes?": "A",
    r"millimet(?:er|re)s?": "mm",
    r"centimet(?:er|re)s?": "cm",
    r"kilomet(?:er|re)s?": "km",
    r"met(?:er|re)s?": "m",
    r"kilopascals?": "kPa",
    r"megapascals?": "MPa",
    r"pascals?": "Pa",
    r"millibars?": "mbar",
    r"bars": "bar",
    r"kilograms?|kilos?": "kg",
    r"grams?": "g",
    r"hours?|hrs?": "h",
    r"minutes?|mins": "min",
    r"seconds?|secs?": "s",
    r"kilohertz": "kHz",
    r"megahertz": "MHz",
    r"gigahertz": "GHz",
    r"hertz": "Hz",
    r"litres? per minute|liters? per minute": "l/min",
}
_NUMBER = r"(\d+(?:[.,]\d+)?)"
_UNIT_RES = [
    (re.compile(rf"{_NUMBER}\s*(?:{words})(?![\w/])", re.IGNORECASE), symbol) for words, symbol in UNIT_WORDS.items()
]
_THOUSANDS = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+)(?=\s*[^\d\s,.]|\s*$)")


def normalize_query(question: str) -> str:
    """The question with numbers and units written the way documents print them."""
    q = clean(question)
    q = _THOUSANDS.sub(lambda m: m.group(1).replace(",", ""), q)  # "1,450 rpm" -> "1450 rpm"; "1,5 bar" stays
    for pattern, symbol in _UNIT_RES:
        q = pattern.sub(lambda m, s=symbol: f"{m.group(1)} {s}", q)
    return q


class Reranker:
    """Re-orders the top ``candidates`` of a retriever with a cross-encoder.

    ``model`` is a sentence-transformers CrossEncoder name or any object with
    ``predict(list[tuple[str, str]]) -> scores``. Ties keep the retriever's order."""

    def __init__(self, base: Any, model: Any = "BAAI/bge-reranker-base", candidates: int = 30):
        if isinstance(model, str):
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model)
        self.base, self.model, self.candidates = base, model, candidates
        self.chunks: Sequence[Chunk] = getattr(base, "chunks", [])

    def search(self, query: str, k: int, doc_id: str | None = None) -> list[tuple[Chunk, float]]:
        hits = self.base.search(query, max(k, self.candidates), doc_id)
        if not hits:
            return []
        scores = [float(s) for s in self.model.predict([(query, c.text) for c, _ in hits])]
        order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))
        return [(hits[i][0], scores[i]) for i in order[:k]]
