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


# ── filtered counts, totals, means and "which ... highest" ──────────────────────────────


def _px100(datasheet):
    from tablerag.ingest.structured import ingest_structured

    path, sections = datasheet
    sec = next(s for s in sections if s.code == "PX-100")
    return ingest_structured(path).tables, sec


def test_filtered_counts(datasheet):
    tables, sec = _px100(datasheet)
    q = "How many stages of the PX-100 allow intermittent duty?"
    assert try_structured(q, tables).answer == str(sum("Intermittent" in s.duty for s in sec.stages))
    q = "How many stages of the PX-100 have a maximum pressure above 5 bar?"
    res = try_structured(q, tables)
    assert res.operation == "count_where" and res.answer == str(sum(float(s.pressure) > 5 for s in sec.stages))
    # the same threshold in another unit of the same quantity
    q = "How many stages of the PX-100 have a maximum pressure of at least 600 kPa?"
    assert try_structured(q, tables).answer == str(sum(float(s.pressure) >= 6 for s in sec.stages))
    # the plain count still works
    assert try_structured("How many stages does the PX-100 define?", tables).answer == str(len(sec.stages))


def test_conditions_it_cannot_place_are_refused(datasheet):
    tables, _ = _px100(datasheet)
    assert try_structured("How many stages of the PX-100 are painted blue?", tables) is None
    # speed is a range column: "above 1000 rpm" could mean the lower or the upper end
    assert try_structured("How many stages of the PX-100 have a speed above 1000 rpm?", tables) is None


def test_totals_and_means(datasheet):
    tables, sec = _px100(datasheet)
    pressures = [float(s.pressure) for s in sec.stages]
    res = try_structured("What is the average maximum pressure of the PX-100 stages?", tables)
    assert res.operation == "mean" and abs(float(res.answer) - sum(pressures) / len(pressures)) < 1e-6
    res = try_structured("What is the total maximum pressure over all stages of the PX-100?", tables)
    assert res.operation == "sum" and float(res.answer) == sum(pressures)
    # a range column has no single numbers to add up
    assert try_structured("What is the average speed of the PX-100?", tables) is None


def test_which_row_has_the_extreme(datasheet):
    tables, sec = _px100(datasheet)
    top = max(float(s.pressure) for s in sec.stages)
    res = try_structured("Which stage of the PX-100 has the highest maximum pressure?", tables)
    assert res.operation == "argmax"
    assert res.answer.split(", ") == [s.name for s in sec.stages if float(s.pressure) == top]  # every tied row
    # without "which" it is still the value
    assert try_structured("What is the highest speed of the PX-100, in rpm?", tables).answer == str(
        max(s.max_rpm for s in sec.stages)
    )


def test_a_count_is_never_answered_by_another_operation(minmax_sheet):
    from tablerag.ingest.structured import ingest_structured

    path, sections = minmax_sheet
    tables, sec = ingest_structured(path).tables, sections[0]
    q = f"How many stages of the {sec.code} have a maximum speed above 2000 rpm?"
    res = try_structured(q, tables)
    assert res is not None and res.operation == "count_where"
    assert res.answer == str(sum(s.max_rpm > 2000 for s in sec.stages))
    q = f"How many stages of the {sec.code} allow intermittent duty?"  # check-mark columns
    assert try_structured(q, tables).answer == str(sum(s.intermittent for s in sec.stages))
    # a comparison outside a count goes to the reader
    assert try_structured(f"Is the maximum pressure of the {sec.code} above 5 bar?", tables) is None


def test_financial_wording_is_not_a_column_operation():
    """From the diagnosis on T²-RAGBench: "total", "average" and "highest" in financial questions
    almost never mean an operation over a whole column."""
    from tablerag.types import StoredRow, StoredTable

    headers = ["Item", "2019", "2018"]
    rows = [
        StoredRow(f"r:t1:r{i}", "r:t1", dict(zip(headers, cells, strict=True)), 1, cells[0])
        for i, cells in enumerate(
            [
                ["Revenue", "5,829", "5,735"],
                ["Cost of sales", "3,100", "3,050"],
                ["Total recourse debt", "4,612", "4,500"],
            ],
            1,
        )
    ]
    table = StoredTable("r:t1", "r", "Consolidated results", "Annual report", headers, rows, [1])
    for q in [
        "What percentage of the total recourse debt was due in 2019?",
        "What was the difference in the cumulative total return between 2018 and 2019?",
        "What was the total revenue in 2019?",
        "What was the average of the high and low values in 2019?",
        "What is the percentage change from the highest value in 2019?",
    ]:
        assert try_structured(q, [table]) is None, q
