"""Turning printed cell text into exact numbers and ranges, without guessing.

Every parser returns None rather than a best effort when the text is not clearly
a number or a range: in a specification a wrong number is worse than no number.
"""

from __future__ import annotations

import re
import unicodedata

_DASHES = "‐‑‒–—−"  # hyphens, en/em dash, minus sign
_NUM = r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?"
_NUMBER_RE = re.compile(rf"(?<![\w.]){_NUM}(?![\w])")
_RANGE_RE = re.compile(rf"^\s*({_NUM})\s*(?:-|to|\.\.|…)\s*({_NUM})\s*([A-Za-zµ/%°1]*)\s*$")
# unit -> (quantity, factor to the base unit). Only units listed here are compared or converted.
UNITS: dict[str, tuple[str, float]] = {
    "hz": ("frequency", 1.0),
    "khz": ("frequency", 1e3),
    "mhz": ("frequency", 1e6),
    "ghz": ("frequency", 1e9),
    "rpm": ("speed", 1.0),
    "1/min": ("speed", 1.0),
    "mm": ("length", 1e-3),
    "cm": ("length", 1e-2),
    "m": ("length", 1.0),
    "km": ("length", 1e3),
    "pa": ("pressure", 1.0),
    "kpa": ("pressure", 1e3),
    "mpa": ("pressure", 1e6),
    "mbar": ("pressure", 100.0),
    "bar": ("pressure", 1e5),
    "mv": ("voltage", 1e-3),
    "v": ("voltage", 1.0),
    "kv": ("voltage", 1e3),
    "ma": ("current", 1e-3),
    "a": ("current", 1.0),
    "w": ("power", 1.0),
    "kw": ("power", 1e3),
    "g": ("mass", 1e-3),
    "kg": ("mass", 1.0),
    "s": ("time", 1.0),
    "min": ("time", 60.0),
    "h": ("time", 3600.0),
    "°c": ("temperature", 1.0),
    "l/min": ("flow", 1.0),
    "%": ("ratio", 1.0),
}
CHECK_MARKS = {"✓", "✔", "☑", "x", "yes", "y"}


def clean(text: str) -> str:
    """Unicode-normalised text with one kind of dash and single spaces."""
    text = unicodedata.normalize("NFKC", text or "")
    for d in _DASHES:
        text = text.replace(d, "-")
    return re.sub(r"\s+", " ", text).strip()


def norm_key(text: str) -> str:
    """Lower-case comparison key: "Stage  3" == "stage 3"."""
    return clean(text).lower()


def parse_number(text: str) -> float | None:
    """'1,5' -> 1.5, '200' -> 200.0, '12 V/m' -> 12.0; None for '200 - 400', 'n/a' or two numbers."""
    t = clean(text)
    found = _NUMBER_RE.findall(t)
    if len(found) != 1 or _RANGE_RE.match(t):
        return None
    raw = found[0]
    # "1,234" is a thousands separator, "1,5" a decimal comma
    raw = raw.replace(",", "") if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+", raw) else raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def parse_range(text: str) -> tuple[float, float, str] | None:
    """'900 - 1500', '900–1500 rpm', '0.15 to 30' -> (low, high, unit). None if not a range."""
    m = _RANGE_RE.match(clean(text))
    if not m:
        return None
    lo, hi = (float(x.replace(",", ".")) for x in m.group(1, 2))
    return (lo, hi, m.group(3)) if lo <= hi else None


def numbers_in(text: str) -> list[float]:
    """All plain numbers in a sentence (used to read numbers out of questions and answers)."""
    out = []
    for raw in _NUMBER_RE.findall(clean(text)):
        try:
            out.append(
                float(raw.replace(",", "") if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+", raw) else raw.replace(",", "."))
            )
        except ValueError:
            continue
    return out


def to_base(value: float, unit: str) -> tuple[str, float] | None:
    """(quantity, value in the base unit): 1.5 kHz -> ('frequency', 1500.0); None for unknown units."""
    entry = UNITS.get(clean(unit).lower())
    return (entry[0], value * entry[1]) if entry else None


def unit_in_header(header: str) -> str:
    """'Speed [rpm]' -> 'rpm', 'Range (kHz)' -> 'kHz', 'Level' -> ''."""
    m = re.search(r"[\[(]\s*([A-Za-zµ/%°1]+)\s*[\])]", header or "")
    return m.group(1) if m else ""


def same_value(a: str, b: str, rel_tol: float = 0.0) -> bool:
    """Exact comparison by default: numbers compare as numbers ('200' == '200.0'),
    everything else as normalised text. ``rel_tol`` is only used by the public
    financial benchmark, whose answers are computed and rounded."""
    na, nb = parse_number(a), parse_number(b)
    if na is not None and nb is not None:
        if rel_tol == 0.0:
            return na == nb
        return abs(na - nb) <= rel_tol * max(abs(na), abs(nb), 1e-12)
    return norm_key(a) == norm_key(b)
