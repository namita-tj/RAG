"""The exact path must answer correctly or not at all."""

from tablerag.structured_ops import link_tables, try_structured
from tablerag.types import StoredRow, StoredTable


def _table(
    code: str, rows: list[list[str]], headers=("Stage", "Speed [rpm]", "Max. pressure [bar]"), title="Operating points"
):
    tid = f"d:{code}"
    stored = [StoredRow(f"{tid}:r{i}", tid, dict(zip(headers, r, strict=True)), 3, r[0]) for i, r in enumerate(rows, 1)]
    return StoredTable(tid, "d", f"Table 1: {title}", f"4.1 {code} - Pump", list(headers), stored, [3])


A = _table("PX-100", [["Stage 1", "300 - 400", "6"], ["Stage 2", "400 - 550", "8"], ["Stage 3", "550 - 700", "6"]])
B = _table("PX-200", [["Stage 1", "450 - 600", "8"], ["Stage 2", "600 - 800", "10"]])
RUN_IN = _table(
    "PX-100", [["8", "2"]], headers=("Run-in continuous [h]", "Run-in intermittent [h]"), title="Run-in time"
)
TABLES = [A, B, RUN_IN]


def test_range_lookup_finds_the_one_row_containing_the_value():
    res = try_structured("Which stage of the PX-100 covers 480 rpm?", TABLES)
    assert res and res.answer == "Stage 2" and res.rows[0].row_id == "d:PX-100:r2"


def test_range_lookup_returns_the_asked_column():
    res = try_structured("What maximum pressure applies to the PX-200 at 700 rpm?", TABLES)
    assert res and res.answer == "10"


def test_range_lookup_refuses_boundaries_and_values_outside_every_row():
    assert try_structured("Which stage of the PX-100 covers 400 rpm?", TABLES) is None  # in Stage 1 and Stage 2
    assert try_structured("Which stage of the PX-100 covers 9000 rpm?", TABLES) is None
    assert try_structured("Which stage of the PX-100 covers 480 bar?", TABLES) is None  # wrong quantity


def test_min_max_columns_form_an_interval():
    t = _table(
        "PX-300",
        [["Stage 1", "100", "200"], ["Stage 2", "200", "350"]],
        headers=("Stage", "Min. speed [rpm]", "Max. speed [rpm]"),
    )
    res = try_structured("Which stage of the PX-300 includes a speed of 250 rpm?", [t])
    assert res and res.answer == "Stage 2"


def test_counting_only_counts_rows():
    res = try_structured("How many stages does the PX-100 define?", TABLES)
    assert res and res.answer == "3"
    assert try_structured("How many hours of continuous run-in does the PX-100 need?", TABLES) is None


def test_extremes_and_named_rows():
    res = try_structured("What is the highest speed of the PX-100, in rpm?", TABLES)
    assert res and res.answer == "700"
    res = try_structured("Top speed of the PX-200 in rpm?", TABLES)
    assert res and res.answer == "800"
    # "maximum pressure" of a named row is a lookup, not a computation
    assert try_structured("What is the maximum pressure for Stage 2 of the PX-100?", TABLES) is None


def test_a_section_code_that_is_not_in_the_document_links_nothing():
    assert link_tables("How many stages does the PX-900 define?", TABLES) == []
    assert try_structured("How many stages does the PX-900 define?", TABLES) is None
