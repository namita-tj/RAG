from pathlib import Path

import pytest

from tablerag.bench.synthetic_docs import build_doc, make_sections


@pytest.fixture(scope="session")
def datasheet(tmp_path_factory) -> tuple[Path, list]:
    """One synthetic datasheet (range layout, 5 rows per page) and the sections it was drawn from."""
    path = tmp_path_factory.mktemp("docs") / "sheet.pdf"
    sections = build_doc(path, make_sections(7), rows_per_page=5)
    return path, sections


@pytest.fixture(scope="session")
def minmax_sheet(tmp_path_factory) -> tuple[Path, list]:
    """The min/max layout: continuation pages without a header row, check marks."""
    path = tmp_path_factory.mktemp("docs") / "sheet_mm.pdf"
    sections = build_doc(path, make_sections(31, "min_max"), rows_per_page=4)
    return path, sections
