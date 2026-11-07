"""Query clean-up and the optional reranker."""

from tablerag.answer import answer_questions
from tablerag.query import Reranker, normalize_query
from tablerag.retrieve import BM25
from tablerag.structured_ops import try_structured
from tablerag.types import Chunk


def test_numbers_and_units_are_written_like_the_documents():
    assert normalize_query("Which stage covers 1,450 revolutions per minute?") == "Which stage covers 1450 rpm?"
    assert normalize_query("Pressure above 5 bars and 80 degrees Celsius") == "Pressure above 5 bar and 80 °C"
    assert normalize_query("A 12 percent rise over 3 hours") == "A 12 % rise over 3 h"
    assert normalize_query("2.5 kilowatts or 400 volts") == "2.5 kW or 400 V"


def test_everything_else_stays_the_same():
    for q in [
        "How many hours of continuous run-in does the PX-100 need?",  # no number before "hours"
        "What is the maximum pressure in bar for Stage 4 of the PX-100?",
        "Is 1,5 bar allowed?",  # a decimal comma, not a thousands separator
        "What is the minimum speed?",
    ]:
        assert normalize_query(q) == q


def test_spelled_out_units_reach_the_exact_path(datasheet):
    from tablerag.ingest.structured import ingest_structured

    tables = ingest_structured(datasheet[0]).tables
    plain = try_structured("Which stage of the PX-100 covers 1450 rpm?", tables)
    spelled = try_structured(normalize_query("Which stage of the PX-100 covers 1,450 revolutions per minute?"), tables)
    assert plain is not None and spelled is not None and spelled.answer == plain.answer


class OverlapModel:
    """Stand-in cross-encoder: scores a pair by shared words, ignoring the retriever's order."""

    def predict(self, pairs):
        return [len(set(q.lower().split()) & set(t.lower().split())) for q, t in pairs]


def test_reranker_reorders_candidates_and_keeps_ties_stable():
    chunks = [
        Chunk("a", "d", "stage four speed pressure"),
        Chunk("b", "d", "stage fourteen pressure"),
        Chunk("c", "e", "stage four speed pressure bar"),
    ]
    base = BM25(chunks)
    ranked = Reranker(base, OverlapModel(), candidates=3)
    top = [c.chunk_id for c, _ in ranked.search("stage four speed pressure bar", 3)]
    assert top[0] == "c"
    assert [c.chunk_id for c, _ in ranked.search("stage four", 2, doc_id="d")] == ["a", "b"]  # filter still applies
    assert Reranker(BM25([]), OverlapModel()).search("x", 3) == []

    # it plugs into the answer step like any retriever
    class Echo:
        def generate(self, prompts):
            return ["four"] * len(prompts)

    res = answer_questions([{"question": "stage four", "doc_id": "d"}], ranked, Echo(), mode="plain")
    assert res[0].text == "four" and res[0].evidence[0].chunk_id == "a"
