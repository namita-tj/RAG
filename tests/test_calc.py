"""Checkable calculations: safe evaluation, operands must be printed, and the answer route."""

from tablerag.answer import NOT_FOUND, answer_questions
from tablerag.calc import check_formula, fmt, normalise_formula, safe_eval
from tablerag.retrieve import BM25
from tablerag.types import Chunk

EVIDENCE = "Revenue was 5,829 million in 2019 and 5,735 million in 2018. Costs were 1200."


def test_safe_eval_allows_only_arithmetic():
    assert safe_eval("(5829 - 5735) / 5735 * 100") == (5829 - 5735) / 5735 * 100
    assert safe_eval("-3 + 4") == 1.0
    for bad in ["__import__('os')", "2 ** 999999", "abs(-1)", "x + 1", "1 / 0", "'a' + 'b'", "", "1" * 300]:
        assert safe_eval(bad) is None


def test_normalising_replies():
    assert normalise_formula("Calculation: (5,829 - 5,735) / 5,735 = 0.0164") == "(5829 - 5735) / 5735"
    assert normalise_formula("$1,200 * 2%") == "1200 * 2"
    assert normalise_formula("120 × 3\nbecause ...") == "120 * 3"


def test_operands_must_be_printed_in_the_evidence():
    value, formula, used = check_formula("(5829 - 5735) / 5735 * 100", EVIDENCE)
    assert abs(value - 1.639) < 1e-3 and sorted(used) == [100.0, 5735.0, 5735.0, 5829.0]
    assert check_formula("(5900 - 5735) / 5735", EVIDENCE) is None  # 5900 is printed nowhere
    assert check_formula("100 / 1", EVIDENCE) is None  # only constants: not from the document
    assert check_formula(NOT_FOUND, EVIDENCE) is None


def test_fmt():
    assert fmt(12.0) == "12" and fmt(1.6390061) == "1.639" and fmt(0.000123456) == "0.000123456"


class TwoStep:
    """First reply: a wrong number in its head; second reply: the calculation."""

    def __init__(self, first, formula):
        self.first, self.formula, self.prompts = first, formula, []

    def generate(self, prompts):
        self.prompts += prompts
        return [self.formula if "Calculation:" in p else self.first for p in prompts]


def test_calculated_answers_are_computed_and_traced():
    chunks = [Chunk("p", "d", EVIDENCE)]
    q = [{"question": "By how many percent did revenue grow from 2018 to 2019?", "doc_id": "d"}]
    ours = answer_questions(q, BM25(chunks), TwoStep("1.7", "(5829 - 5735) / 5735 * 100"), mode="ours")
    assert ours[0].route == "computed" and ours[0].text == "1.6391" and ours[0].raw == "(5829 - 5735) / 5735 * 100"
    # without the calculation step the unprinted 1.7 is refused
    nocalc = answer_questions(q, BM25(chunks), TwoStep("1.7", "(5829 - 5735) / 5735 * 100"), mode="ours_nocalc")
    assert nocalc[0].text == NOT_FOUND
    # a formula with an invented number stays refused
    bad = answer_questions(q, BM25(chunks), TwoStep("1.7", "(5900 - 5735) / 5735 * 100"), mode="ours")
    assert bad[0].text == NOT_FOUND and bad[0].route == "refused"
    # an answer that is printed is verified directly, no second call
    reader = TwoStep("1200", "unused")
    direct = answer_questions([{"question": "What were the costs?", "doc_id": "d"}], BM25(chunks), reader, mode="ours")
    assert direct[0].route == "verified" and not any("Calculation:" in p for p in reader.prompts)
