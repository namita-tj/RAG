"""Exact answers computed from the stored rows, for the questions language models get wrong.

Retrieval plus a language model is weak at table questions that need the whole table:

* range lookups   "Which stage covers 1450 rpm?"   (1450 is printed nowhere, only "1200 - 1600")
* counting        "How many stages are defined?"  (needs every row, top-k gives a few)
* filtered counts "How many stages allow intermittent duty?", "... have a pressure above 5 bar?"
* extremes        "What is the highest speed?", "Which stage has the highest pressure?"
* totals, means   "What is the total power of all stages?", "What is the average pressure?"

For these, this module finds the table the question is about and computes the answer from
the stored rows, returning the rows it used so the answer can be checked. When the question
does not clearly match one table and one operation it returns None, and the normal
retrieval + reader path answers instead. It never guesses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tablerag.normalize import CHECK_MARKS, UNITS, clean, norm_key, parse_number, parse_range, to_base, unit_in_header
from tablerag.retrieve import tokenize
from tablerag.types import StoredRow, StoredTable

_UNIT_ALT = "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))
_QUANTITY = re.compile(rf"(\d+(?:[.,]\d+)?)\s*({_UNIT_ALT})(?![\w/])", re.IGNORECASE)
_COUNT = re.compile(r"\b(?:how many|number of|count(?: the)?)\s+((?:[a-z\-]+\s+){0,2}[a-z\-]+)", re.IGNORECASE)
_MAX = re.compile(r"\b(highest|maximum|largest|max|upper|top|peak)\b", re.IGNORECASE)
_MIN = re.compile(r"\b(lowest|minimum|smallest|min)\b", re.IGNORECASE)
_CODE = re.compile(r"\b[A-Z]{1,5}-\d+(?:\.\d+)?\b")
_WHICH = re.compile(r"\bwhich\b", re.IGNORECASE)
_SUM = re.compile(r"\b(total|sum|combined)\b", re.IGNORECASE)
_MEAN = re.compile(r"\b(average|mean)\b", re.IGNORECASE)
_COMPARE = re.compile(
    r"\b(above|over|more than|greater than|higher than|exceeding|at least|below|under|less than|lower than|at most)"
    rf"\s+(\d+(?:[.,]\d+)?)\s*(?:({_UNIT_ALT})(?![\w/]))?",
    re.IGNORECASE,
)
# words of a calculation the reader should write as a formula, not an operation over a whole column:
# "the percentage of the total debt", "the difference in cumulative total return", "the average of the high and low"
_CALCULATION = re.compile(
    r"\b(percent(age)?|difference|change[sd]?|ratio|growth|grew|increase[sd]?|decrease[sd]?|compared?|"
    r"cumulative|return|proportion|share of|fraction|rate)\b|%",
    re.IGNORECASE,
)
_STRICT = {"above", "over", "more than", "greater than", "higher than", "exceeding", "below", "under", "less than"}
_LOWER = {"below", "under", "less than", "lower than", "at most"}
# words a count question may contain without being a condition ("How many stages does X define?")
_FILLER = {
    "many",
    "number",
    "count",
    "define",
    "defined",
    "defines",
    "list",
    "listed",
    "there",
    "exist",
    "exists",
    "has",
    "have",
    "contain",
    "contains",
    "include",
    "includes",
    "specified",
    "given",
    "shown",
    "available",
    "operating",
    "total",
    "all",
    "be",
    "can",
    "allow",
    "allows",
    "allowed",
    "support",
    "supports",
    "duty",
    "mode",
    "modes",
    "use",
    "used",
    "run",
    "runs",
}


@dataclass
class OpResult:
    answer: str
    operation: str
    rows: list[StoredRow] = field(default_factory=list)
    table_id: str = ""
    explanation: str = ""


def link_tables(query: str, tables: list[StoredTable]) -> list[StoredTable]:
    """The tables a question is about: an exact section code wins ("RI-2"), otherwise the
    best word overlap with section, caption and column names. Ties are all returned."""
    codes = {norm_key(c) for c in _CODE.findall(clean(query))}
    if codes:
        hits = [t for t in tables if any(c in norm_key(t.context).split() for c in codes)]
        return hits  # a named section that is not in the document: nothing to compute on
    q = set(tokenize(query))

    def score(t: StoredTable) -> int:
        return len(q & set(tokenize(f"{t.context} {t.title} {' '.join(t.headers)}")))

    best = max((score(t) for t in tables), default=0)
    return [t for t in tables if best > 0 and score(t) == best]


def _main_table(query: str, tables: list[StoredTable]) -> StoredTable | None:
    """Among linked tables, the one whose column names match the question best; None on a tie."""
    if not tables:
        return None
    q = set(tokenize(query))
    scored = sorted(
        ((len(q & set(tokenize(" ".join(t.headers)))), t.row_count, t) for t in tables),
        key=lambda x: (x[0], x[1]),
        reverse=True,
    )
    if len(scored) > 1 and scored[0][:2] == scored[1][:2]:
        return None
    return scored[0][2]


def _row_interval(row: StoredRow) -> tuple[str, float, float] | None:
    """(quantity, low, high) in base units, from a range cell ("900 - 1500" under "Speed [rpm]")
    or from a pair of columns ("Min. speed [rpm]" / "Max. speed [rpm]")."""
    low: dict[str, float] = {}
    high: dict[str, float] = {}
    for h, v in row.cells.items():
        rng = parse_range(v)
        if rng:
            lo, hi = to_base(rng[0], rng[2] or unit_in_header(h)), to_base(rng[1], rng[2] or unit_in_header(h))
            if lo and hi:
                return lo[0], lo[1], hi[1]
        n = parse_number(v)
        base = to_base(n, unit_in_header(h)) if n is not None else None
        if base is None:
            continue
        key = norm_key(h)
        if re.match(r"(min|start|from|lower)\b", key):
            low[base[0]] = base[1]
        elif re.match(r"(max|stop|end|to|upper)\b", key):
            high[base[0]] = base[1]
    for quantity in low.keys() & high.keys():
        return quantity, low[quantity], high[quantity]
    return None


def _column_asked(query: str, table: StoredTable, exclude: set[str]) -> str | None:
    q = set(tokenize(query))
    best, best_score = None, 0
    for h in table.headers[1:]:
        if h in exclude:
            continue
        s = len(q & set(tokenize(h)))
        if s > best_score:
            best, best_score = h, s
    return best


def range_lookup(query: str, tables: list[StoredTable]) -> OpResult | None:
    # "above 1000 rpm" and "how many ... at 1000 rpm" are comparisons and counts, not lookups
    if _COMPARE.search(clean(query)) or _COUNT.search(clean(query)):
        return None
    m = _QUANTITY.search(clean(query))
    if not m:
        return None
    asked = to_base(float(m.group(1).replace(",", ".")), m.group(2))
    if asked is None:
        return None
    hits = []
    for t in tables:
        for r in t.rows:
            iv = _row_interval(r)
            if iv and iv[0] == asked[0] and iv[1] <= asked[1] <= iv[2]:
                hits.append((t, r))
    if len(hits) != 1:  # none, or a boundary value shared by two rows: do not choose
        return None
    t, r = hits[0]
    # the columns that made the interval are not "the column asked for"
    quantity = asked[0]
    range_cols = {
        h
        for h, v in r.cells.items()
        if parse_range(v) or ((to_base(1.0, unit_in_header(h)) or ("", 0.0))[0] == quantity)
    }
    rest = _QUANTITY.sub(" ", query)
    col = _column_asked(rest, t, range_cols | {t.headers[0]})
    answer = r.label if re.search(r"\bwhich\b", query, re.IGNORECASE) or col is None else r.cells[col]
    return OpResult(
        answer, "range_lookup", [r], t.table_id, f"{m.group(0)} lies in {r.label} of {t.context or t.table_id}"
    )


def _singular(word: str) -> str:
    w = word.lower()
    return w[:-3] + "y" if w.endswith("ies") else w[:-1] if w.endswith("s") and not w.endswith("ss") else w


def _row_noun(t: StoredTable) -> set[str]:
    """What the rows of a table are called: the first column header and the label stem
    ("Stage" for "Stage 1", "Stage 2", ...)."""
    nouns = {_singular(w) for w in tokenize(t.headers[0]) if w.isalpha()}
    stems = {re.sub(r"[\d\s.\-]+$", "", r.label).strip().lower() for r in t.rows if r.label}
    nouns |= {_singular(s) for s in stems if s and len(stems) == 1}
    return nouns | {"row", "entry", "item", "line"}


def count_rows(query: str, tables: list[StoredTable]) -> OpResult | None:
    """Only when the counted thing is what the rows are ("how many stages"), not a value
    ("how many hours")."""
    m = _COUNT.search(clean(query))
    if not m:
        return None
    t = _main_table(query, tables)
    if t is None:
        return None
    words = []
    for w in m.group(1).lower().split():
        if w in {"of", "does", "do", "is", "are", "in", "for", "listed", "the", "that", "has", "have"}:
            break
        words.append(w)
    head = _singular(words[-1]) if words else ""  # "operating stages" -> "stage", "hours of ..." -> "hour"
    if head not in _row_noun(t):
        return None
    cond = _condition(query, t, words)
    if cond is False:
        return None
    if cond is None:
        return OpResult(
            str(t.row_count), "count", list(t.rows), t.table_id, f"{t.row_count} rows in {t.context or t.table_id}"
        )
    rows, why = cond
    return OpResult(str(len(rows)), "count_where", rows, t.table_id, f"{len(rows)} rows where {why}")


def extreme(query: str, tables: list[StoredTable]) -> OpResult | None:
    is_max, is_min = bool(_MAX.search(query)), bool(_MIN.search(query))
    if is_max == is_min:
        return None
    t = _main_table(query, tables)
    if t is None:
        return None
    q = set(tokenize(query)) | ({"max", "maximum"} if is_max else {"min", "minimum"})
    candidates: list[tuple[int, str, list[tuple[float, StoredRow]]]] = []
    for h in t.headers[1:]:
        vals: list[tuple[float, StoredRow]] = []
        for r in t.rows:
            v = r.cells.get(h, "")
            rng = parse_range(v)
            n = parse_number(v)
            if rng:
                vals.append((rng[1] if is_max else rng[0], r))
            elif n is not None:
                vals.append((n, r))
        if vals:
            name = h
            if parse_range(t.rows[0].cells.get(h, "")):
                name = f"{'max' if is_max else 'min'} {h}"
            candidates.append((len(q & set(tokenize(name))), h, vals))
    if not candidates:
        return None
    candidates.sort(key=lambda c: -c[0])
    if candidates[0][0] == 0 or (len(candidates) > 1 and candidates[0][0] == candidates[1][0]):
        return None  # cannot tell which column is meant
    _, h, vals = candidates[0]
    best = max(vals, key=lambda x: x[0]) if is_max else min(vals, key=lambda x: x[0])
    value = best[0]
    name = "max" if is_max else "min"
    if _WHICH.search(query):  # "Which stage has the highest pressure?": the row(s), all of them on a tie
        rows = [r for v, r in vals if v == value]
        return OpResult(
            ", ".join(r.label for r in rows), f"arg{name}", rows, t.table_id, f"{name} of {h} is {_fmt(value)}"
        )
    answer = str(int(value)) if value == int(value) else repr(value)
    return OpResult(answer, name, [best[1]], t.table_id, f"{name} of {h}")


def _fmt(x: float) -> str:
    """An exact result as text: 12 not 12.0, at most six decimals, no float noise."""
    if x == int(x):
        return str(int(x))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def _numbers(t: StoredTable, h: str) -> list[tuple[float, StoredRow]] | None:
    """The single numbers of a column, or None if it holds ranges or text (then it is not summable)."""
    out = []
    for r in t.rows:
        v = r.cells.get(h, "")
        if not clean(v):
            continue
        n = parse_number(v)
        if n is None or parse_range(v):
            return None
        out.append((n, r))
    return out if len(out) >= 2 else None


_SYNONYMS = {"maximum": "max", "highest": "max", "upper": "max", "minimum": "min", "lowest": "min", "lower": "min"}


def _words(text: str) -> set[str]:
    """Tokens with "maximum"/"highest" read as "max" and "minimum"/"lowest" as "min"."""
    return {_SYNONYMS.get(w, w) for w in tokenize(text)}


def _pick_column(query: str, t: StoredTable, columns: list[str]) -> str | None:
    """The one column whose name matches the question best; None if none or a tie."""
    q = _words(query)
    scored = sorted(((len(q & _words(h)), h) for h in columns), reverse=True)
    if not scored or scored[0][0] == 0 or (len(scored) > 1 and scored[0][0] == scored[1][0]):
        return None
    return scored[0][1]


def aggregate(query: str, tables: list[StoredTable]) -> OpResult | None:
    """Total or mean of one numeric column over all rows ("total power of all stages", "average pressure
    across the stages"). The question has to name what the rows are; "total revenue in 2019" names no rows
    and is not a sum over a column."""
    is_sum, is_mean = bool(_SUM.search(query)), bool(_MEAN.search(query))
    if is_sum == is_mean:
        return None
    t = _main_table(query, tables)
    if t is None:
        return None
    if not {_singular(w) for w in tokenize(query) if w.isalpha()} & (_row_noun(t) - {"row", "entry", "item", "line"}):
        return None
    numeric = [h for h in t.headers[1:] if _numbers(t, h)]
    h = _pick_column(query, t, numeric)
    if h is None:
        return None
    vals = _numbers(t, h) or []
    total = sum(v for v, _ in vals)
    value = total if is_sum else total / len(vals)
    name = "sum" if is_sum else "mean"
    return OpResult(
        _fmt(value),
        name,
        [r for _, r in vals],
        t.table_id,
        f"{name} of {h} over {len(vals)} rows of {t.context or t.table_id}",
    )


def _compare(op: str, a: float, b: float) -> bool:
    if op in _LOWER:
        return a < b if op in _STRICT else a <= b
    return a > b if op in _STRICT else a >= b


def _condition(query: str, t: StoredTable, head_words: list[str]) -> tuple[list[StoredRow], str] | None | bool:
    """The rows a count question is restricted to. None: no condition (count every row).
    False: there is a condition, but it is not clear which column it refers to (do not guess)."""
    m = _COMPARE.search(clean(query))
    if m:
        op, number, unit = m.group(1).lower(), float(m.group(2).replace(",", ".")), m.group(3)
        rest = query[: m.start()] + " " + query[m.end() :]
        columns = [h for h in t.headers[1:] if _numbers(t, h)]
        if unit:
            asked = to_base(number, unit)
            if asked is None:
                return False
            columns = [h for h in columns if (to_base(1.0, unit_in_header(h)) or ("", 0))[0] == asked[0]]
            h = columns[0] if len(columns) == 1 else _pick_column(rest, t, columns)
            if h is None:
                return False
            factor = (to_base(1.0, unit_in_header(h)) or ("", 1.0))[1]
            rows = [r for v, r in _numbers(t, h) or [] if _compare(op, v * factor, asked[1])]
        else:
            h = _pick_column(rest, t, columns)
            if h is None:
                return False
            rows = [r for v, r in _numbers(t, h) or [] if _compare(op, v, number)]
        return rows, f"{h} {op} {m.group(2)}{' ' + unit if unit else ''}"
    known = set(tokenize(f"{t.context} {t.title}")) | {_singular(w) for w in head_words} | set(head_words)
    words = {w for w in tokenize(query) if w.isalnum() and not w.isdigit()}
    cond = {w for w in words if w not in known and _singular(w) not in _row_noun(t) and w not in _FILLER}
    found: list[tuple[str, list[StoredRow]]] = []
    for h in t.headers[1:]:
        cells = [norm_key(r.cells.get(h, "")) for r in t.rows]
        if all(c in CHECK_MARKS or c in {"", "-", "no", "n"} for c in cells):  # a check-mark column
            if (cond | (words - known)) & set(tokenize(h)):
                found.append((h, [r for r, c in zip(t.rows, cells, strict=True) if c in CHECK_MARKS]))
            continue
        tokens = [set(tokenize(c)) for c in cells]
        informative = {
            w for w in (cond | (words & _FILLER)) if any(w in x for x in tokens) and not all(w in x for x in tokens)
        }
        if informative:
            found.append((h, [r for r, x in zip(t.rows, tokens, strict=True) if informative <= x]))
    if len(found) == 1:
        h, rows = found[0]
        return rows, f"{h} matches the question"
    if found or cond:
        return False  # several columns could be meant, or a condition no column holds
    return None


def names_a_row(query: str, tables: list[StoredTable]) -> bool:
    """True if the question names a specific row ("... for Stage 1 of ..."): then it is a
    lookup, not a computation, even if it says "maximum" ("maximum pressure" is a column)."""
    q = clean(query)
    return any(
        r.label and re.search(rf"(?<![\w]){re.escape(r.label)}(?![\w])", q, re.IGNORECASE)
        for t in tables
        for r in t.rows
    )


def try_structured(query: str, tables: list[StoredTable]) -> OpResult | None:
    """Answers with the one operation the question asks for, or None.

    The kind of question decides the operation: a count question is only answered by
    counting, a total or mean only by aggregating. If that operation cannot answer cleanly,
    no other operation gets a turn, so "how many stages have a maximum speed above X" can
    never come back as the maximum speed."""
    linked = link_tables(query, tables)
    if not linked or names_a_row(query, linked):
        return None
    q = clean(query)
    if _CALCULATION.search(q):
        return None  # a calculation: the reader writes the formula and the code checks it
    if _COUNT.search(q):
        ops = [count_rows]
    elif _COMPARE.search(q):
        return None  # a comparison outside a count ("is it above 5 bar?") is for the reader
    elif _SUM.search(q) or _MEAN.search(q):
        ops = [aggregate]
    else:
        ops = [range_lookup, extreme]
    for op in ops:
        res = op(query, linked)
        if res is not None:
            return res
    return None
