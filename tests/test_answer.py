"""Reader replies, citation checks and refusals, with a stand-in reader."""

from tablerag.answer import NOT_FOUND, answer_questions, build_context, parse_reply, verify, within_budget
from tablerag.retrieve import BM25
from tablerag.types import Chunk, StoredRow, StoredTable

ROW1 = StoredRow(
    "d:t1:r1", "d:t1", {"Stage": "Stage 1", "Speed [rpm]": "300 - 400", "Max. pressure [bar]": "6"}, 3, "Stage 1"
)
ROW2 = StoredRow(
    "d:t1:r2", "d:t1", {"Stage": "Stage 2", "Speed [rpm]": "400 - 550", "Max. pressure [bar]": "8"}, 3, "Stage 2"
)
TABLE = StoredTable("d:t1", "d", "Table 1: Operating points", "4.1 PX-100 - Pump", list(ROW1.cells), [ROW1, ROW2], [3])
CHUNKS = [
    Chunk(
        r.row_id,
        "d",
        f"4.1 PX-100 - Pump\n[{r.row_id}] " + "; ".join(f"{k}: {v}" for k, v in r.cells.items()),
        [3],
        "row",
        [r.row_id],
        {"table_id": "d:t1"},
    )
    for r in (ROW1, ROW2)
]
ROWS = {r.row_id: r for r in (ROW1, ROW2)}


class Echo:
    """Returns fixed replies and records the prompts it was given."""

    def __init__(self, reply: str):
        self.reply, self.prompts = reply, []

    def generate(self, prompts):
        self.prompts += prompts
        return [self.reply] * len(prompts)


def test_parse_reply_variants():
    assert parse_reply("8 | d:t1:r2") == ("8", "d:t1:r2")
    assert parse_reply("Answer: 8 [d:t1:r2]") == ("8", "d:t1:r2")
    assert parse_reply("8.\nbecause ...") == ("8", None)


def test_verify_accepts_only_supported_answers():
    assert verify("8", "d:t1:r2", CHUNKS, ROWS) == "d:t1:r2"
    assert verify("8", "d:t1:r1", CHUNKS, ROWS) == "d:t1:r2"  # wrong citation, but one row has it
    assert verify("12", "d:t1:r2", CHUNKS, ROWS) is None  # printed nowhere
    assert verify("6", None, CHUNKS[:1], ROWS) == "d:t1:r1"
    assert verify("400 - 550", "d:t1:r2", CHUNKS, ROWS) == "d:t1:r2"
    assert verify("yes", None, CHUNKS, ROWS) is None  # a yes/no answer needs a valid citation


def test_modes():
    retriever = BM25(CHUNKS)
    q = [{"question": "Maximum pressure of Stage 2 of the PX-100?", "doc_id": "d"}]
    plain = answer_questions(q, retriever, Echo("12"), tables=[TABLE], mode="plain")
    assert plain[0].text == "12" and plain[0].route == "reader"
    ours = answer_questions(q, retriever, Echo("12 | d:t1:r2"), tables=[TABLE], mode="ours")
    assert ours[0].text == NOT_FOUND and ours[0].route == "refused"
    unverified = answer_questions(q, retriever, Echo("12 | d:t1:r2"), tables=[TABLE], mode="ours_unverified")
    assert unverified[0].text == "12" and unverified[0].route == "unverified"
    good = answer_questions(q, retriever, Echo("8 | d:t1:r2"), tables=[TABLE], mode="ours")
    assert good[0].text == "8" and good[0].rows == ["d:t1:r2"] and good[0].route == "verified"


def test_structured_questions_skip_the_reader():
    reader = Echo("should not be used")
    res = answer_questions(
        [{"question": "Which stage of the PX-100 covers 480 rpm?", "doc_id": "d"}],
        BM25(CHUNKS),
        reader,
        tables=[TABLE],
        mode="ours",
    )
    assert res[0].text == "Stage 2" and res[0].route == "structured" and reader.prompts == []


def test_context_budget():
    many = [Chunk(str(i), "d", "x" * 600) for i in range(10)]
    assert len(within_budget(many, 2000)) == 4
    assert len(build_context(many, 2000)) < 2100
