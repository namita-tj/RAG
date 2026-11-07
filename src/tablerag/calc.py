"""Checkable calculations: the language model writes the formula, the code computes it.

A small reader is bad at arithmetic, and a calculated answer ("the change was 1.6%") is
printed nowhere, so the evidence check cannot confirm it. Instead the model is asked for the
calculation itself, for example ``(5829 - 5735) / 5735 * 100``. This module

* evaluates it safely: only numbers, + - * /, parentheses and a leading minus are allowed,
  so nothing the model writes can run as code;
* checks that every number in it is printed in the retrieved evidence (plus the constants
  1 and 100, for "times 100" and "minus 1" in percentages and ratios).

So a calculated answer is either traceable to printed numbers, or refused.
"""

from __future__ import annotations

import ast
import re

from tablerag.normalize import clean, numbers_in

CONSTANTS = {1.0, 100.0}
MAX_LENGTH = 200

FORMULA_PROMPT = """The answer to the question has to be calculated from numbers in the context below. Write the calculation using only numbers copied exactly from the context, the operators + - * / and parentheses. Reply with only the calculation, for example: (5829 - 5735) / 5735 * 100. If the context does not contain the numbers needed, reply {nf}.

Context:
{context}

Question: {question}
Calculation:"""

_THOUSANDS = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})+)(?![\d,])")
_ALLOWED = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.USub,
    ast.UAdd,
)


def normalise_formula(text: str) -> str:
    """The calculation part of a reply: first line, no "=" result, no currency or percent signs,
    thousands separators removed ("1,234" -> "1234")."""
    line = clean(text.strip().splitlines()[0] if text.strip() else "")
    line = re.sub(r"^(calculation|formula|answer)\s*[:\-]\s*", "", line, flags=re.IGNORECASE)
    line = line.split("=")[0].strip().strip("`").strip()
    line = _THOUSANDS.sub(lambda m: m.group(1).replace(",", ""), line)
    line = re.sub(r"[$€£%]", "", line).replace("×", "*").replace("÷", "/")
    return line.strip()


def safe_eval(expression: str) -> float | None:
    """The value of an arithmetic expression, or None if it is anything else (names, calls,
    powers, division by zero, too long)."""
    if not expression or len(expression) > MAX_LENGTH:
        return None
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        return None
    if not all(isinstance(n, _ALLOWED) for n in ast.walk(tree)):
        return None
    if any(isinstance(n, ast.Constant) and not isinstance(n.value, int | float) for n in ast.walk(tree)):
        return None
    try:
        value = eval(compile(tree, "<formula>", "eval"), {"__builtins__": {}}, {})  # noqa: S307 - checked above
    except (ZeroDivisionError, OverflowError, ValueError):
        return None
    return float(value)


def operands(expression: str) -> list[float]:
    tree = ast.parse(expression, mode="eval")
    return [float(n.value) for n in ast.walk(tree) if isinstance(n, ast.Constant)]


def check_formula(reply: str, evidence_text: str) -> tuple[float, str, list[float]] | None:
    """(value, formula, operands) if the reply is a safe calculation whose numbers are all
    printed in the evidence; otherwise None."""
    formula = normalise_formula(reply)
    value = safe_eval(formula)
    if value is None:
        return None
    used = operands(formula)
    printed = set(numbers_in(evidence_text))
    if not used or not all(n in printed or n in CONSTANTS for n in used):
        return None
    if all(n in CONSTANTS for n in used):  # "100 / 1" is not a calculation from the document
        return None
    return value, formula, used


def fmt(value: float) -> str:
    """A computed result as text: integers without ".0", four decimals for values from 1 up,
    six significant digits below that (0.000123 stays 0.000123)."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    if abs(value) >= 1:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:.6g}"
