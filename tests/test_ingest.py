"""Ingestion on synthetic datasheets: tables must come back complete, in context and on the right page."""

from tablerag.ingest.baselines import ingest_markdown, ingest_naive
from tablerag.ingest.common import split_text, to_markdown
from tablerag.ingest.structured import derived_fields, heading_in, ingest_structured


def _main_tables(result):
    return [t for t in result.tables if t.title.endswith("Operating points")]


def _check_sheet(path, sections):
    result = ingest_structured(path)
    tables = _main_tables(result)
    assert len(tables) == len(sections)  # stitched: one table per model, not one per page
    for t, sec in zip(tables, sections, strict=True):
        assert t.context == f"{sec.number} {sec.code} - {sec.name}"
        assert [r.label for r in t.rows] == [s.name for s in sec.stages]
        assert [r.page for r in t.rows] == sec.stage_pages
        assert t.pages == sorted(set(sec.stage_pages))
    return result


def test_structured_restores_every_row_with_header_context_and_page(datasheet):
    path, sections = datasheet
    result = _check_sheet(path, sections)
    t = _main_tables(result)[0]
    s = sections[0].stages[0]
    assert t.rows[0].cells == {
        "Stage": s.name,
        "Speed [rpm]": f"{s.min_rpm} - {s.max_rpm}",
        "Duty": s.duty,
        "Max. pressure [bar]": s.pressure,
    }


def test_structured_handles_continuations_without_a_header(minmax_sheet):
    path, sections = minmax_sheet
    result = _check_sheet(path, sections)
    row = _main_tables(result)[0].rows[-1]  # printed on a page without a header row
    assert set(row.cells) == {
        "Stage",
        "Min. speed [rpm]",
        "Max. speed [rpm]",
        "Continuous",
        "Intermittent",
        "Max. pressure [bar]",
    }


def test_page_furniture_is_removed(datasheet):
    result = ingest_structured(datasheet[0])
    assert not any("Example Industries" in c.text or "Page 3 of" in c.text for c in result.chunks)
    assert not any("Document ID" in h for t in result.tables for h in t.headers)


def test_row_chunks_carry_context_ids_and_exact_values(datasheet):
    result = ingest_structured(datasheet[0])
    row = next(c for c in result.chunks if c.kind == "row")
    assert row.text.startswith("PX product family technical datasheet > 4.1 PX-100 - Compact pump > Table 4.1-1")
    assert f"[{row.row_ids[0]}]" in row.text and "Speed min [rpm]" in row.text
    kinds = {c.kind for c in result.chunks}
    assert {"row", "table", "profile", "text"} <= kinds


def test_ablation_switches(datasheet):
    plain = ingest_structured(datasheet[0], context=False, table_chunks=False, profiles=False)
    assert {c.kind for c in plain.chunks} == {"row", "text"}
    assert not any("PX-100 - Compact" in c.text for c in plain.chunks if c.kind == "row")
    unstitched = ingest_structured(datasheet[0], stitch=False)
    assert len(unstitched.tables) > len(ingest_structured(datasheet[0]).tables)


def test_baselines_produce_chunks_with_pages(datasheet):
    path, sections = datasheet
    for fn in (ingest_naive, ingest_markdown):
        r = fn(path)
        assert r.chunks and all(c.pages for c in r.chunks)
        assert r.pages == sections[-1].stage_pages[-1] + 1  # + the appendix page


def test_helpers():
    assert heading_in("Contents\n4.2 PX-200 - Standard pump\nThe PX-200 may ...") == "4.2 PX-200 - Standard pump"
    assert heading_in("4.1 PX-100 - Compact pump .............. 3") == ""
    assert derived_fields("Speed [rpm]", "900 - 1500") == {"Speed min [rpm]": "900", "Speed max [rpm]": "1500"}
    assert derived_fields("Continuous", "✓") == {"Continuous": "yes"}
    text = "word " * 500
    spans = split_text(text, 100, 20)
    assert all(e - s <= 100 for s, e in spans) and spans[-1][1] == len(text)
    assert to_markdown([["a", "b|c"], ["1", "2"]]).splitlines()[0] == "| a | b\\|c |"
