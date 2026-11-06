"""Synthetic technical datasheets with known ground truth.

Each PDF describes a fictional product family (invented company, invented models) and has
what makes real technical documents hard to ingest:

* a title page, a table of contents with dot leaders and an appendix;
* a letterhead table and a "Page n of N" footer repeated on every page;
* one section per model with an operating-point table that continues over several pages,
  either repeating its header ("range" layout: "900 - 1500" in one cell) or without a
  header and captioned "(continued)" ("min_max" layout: separate columns, ✓ marks);
* the same row names ("Stage 3") in every section, so a row out of context is ambiguous;
* small extra tables (run-in times, operating modes) next to the main one.

All content is invented. Run ``python -m tablerag.bench.synthetic_docs <dir>`` to look at one.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

CHECK = "\u2713"
_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/DejaVuSans.ttf",
    "C:/Windows/Fonts/DejaVuSans.ttf",
]

MODELS = [  # code, name, first stage speed [rpm], pressure levels [bar], has run-in table, has modes table
    ("PX-100", "Compact pump", 300, ["4", "6"], True, False),
    ("PX-200", "Standard pump", 450, ["6", "8"], True, True),
    ("PX-300", "Standard pump, high pressure", 450, ["10", "12"], True, False),
    ("PX-400", "Heavy-duty pump", 600, ["12", "16"], True, False),
    ("PX-500", "Heavy-duty pump, extended range", 250, ["8", "10"], False, False),
    ("PX-600", "Low-noise pump", 350, ["5", "7"], False, False),
]
DUTIES = ["Continuous, Intermittent", "Continuous", "Intermittent", "Continuous, Intermittent"]


def check_font() -> str | None:
    if "DejaVuSans" in pdfmetrics.getRegisteredFontNames():
        return "DejaVuSans"
    for f in _FONTS:
        if Path(f).is_file():
            pdfmetrics.registerFont(TTFont("DejaVuSans", f))
            return "DejaVuSans"
    return None


@dataclass
class Stage:
    name: str
    min_rpm: int
    max_rpm: int
    duty: str
    pressure: str

    @property
    def continuous(self) -> bool:
        return "Continuous" in self.duty

    @property
    def intermittent(self) -> bool:
        return "Intermittent" in self.duty


@dataclass
class Section:
    number: str
    code: str
    name: str
    stages: list[Stage]
    run_in: tuple[str, str] | None = None  # hours (continuous, intermittent)
    modes_table: bool = False
    layout: str = "range"  # "range" or "min_max"
    start_page: int = 0
    stage_pages: list[int] = field(default_factory=list)


def make_sections(seed: int, layout: str = "range") -> list[Section]:
    rng = random.Random(seed)
    sections = []
    for i, (code, name, first, levels, run_in, modes) in enumerate(MODELS):
        lo, stages = first, []
        for s in range(rng.randint(6, 14)):
            hi = int(lo * rng.uniform(1.15, 1.45))
            stages.append(Stage(f"Stage {s + 1}", lo, hi, DUTIES[s % 4], levels[s % len(levels)]))
            lo = hi
        times = (str(rng.choice([2, 4, 8])), str(rng.choice([1, 2]))) if run_in else None
        sections.append(Section(f"4.{i + 1}", code, name, stages, times, modes, layout))
    return sections


def _grid(data: list[list[str]], widths: list[int], header: bool = True, font: str | None = None) -> Table:
    t = Table(data, colWidths=widths)
    style = [("GRID", (0, 0), (-1, -1), 0.5, colors.black), ("FONTSIZE", (0, 0), (-1, -1), 9)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey))
    if font:
        style.append(("FONTNAME", (0, 0), (-1, -1), font))
    t.setStyle(TableStyle(style))
    return t


def _letterhead(doc_id: str) -> Table:
    return _grid(
        [
            ["Example Industries (fictional)", f"Document ID: {doc_id}", "Rev. 3"],
            ["Datasheet", "Released", "2026-01-01"],
        ],
        [180, 170, 110],
        header=False,
    )


def build_doc(path: Path, sections: list[Section], rows_per_page: int = 6, doc_id: str = "DS-0001") -> list[Section]:
    """Writes the PDF; fills in the page of every stage and returns the sections."""
    styles = getSampleStyleSheet()
    body, h2 = styles["BodyText"], styles["Heading2"]
    font = check_font()
    page = 3
    for sec in sections:
        sec.start_page = page
        sec.stage_pages = [page + i // rows_per_page for i in range(len(sec.stages))]
        page = sec.stage_pages[-1] + 1
    total = page

    def footer(canvas, doc):  # noqa: ANN001
        canvas.setFont("Helvetica", 8)
        canvas.drawString(40, 20, f"Page {doc.page} of {total}")

    story: list = [
        _letterhead(doc_id),
        Spacer(1, 40),
        Paragraph("PX product family: technical datasheet (synthetic example)", styles["Title"]),
        Paragraph("Invented content for automated testing. No real product is described.", body),
        PageBreak(),
        _letterhead(doc_id),
        Spacer(1, 12),
        Paragraph("Contents", h2),
    ]
    for sec in sections:
        story.append(Paragraph(f"{sec.number} {sec.code} - {sec.name} {'.' * 30} {sec.start_page}", body))
    story += [Paragraph(f"Appendix A Measurement conditions {'.' * 30} {total}", body), PageBreak()]

    for sec in sections:
        story += [
            _letterhead(doc_id),
            Spacer(1, 12),
            Paragraph(f"{sec.number} {sec.code} - {sec.name}", h2),
            Paragraph(f"The {sec.code} may be operated at the stages listed below.", body),
            Paragraph(f"Table {sec.number}-1: Operating points", body),
        ]
        if sec.layout == "min_max":
            if font is None:
                raise RuntimeError("the min_max layout needs DejaVuSans for the check mark")
            header = [
                "Stage",
                "Min. speed [rpm]",
                "Max. speed [rpm]",
                "Continuous",
                "Intermittent",
                "Max. pressure [bar]",
            ]
            rows = [
                [
                    s.name,
                    str(s.min_rpm),
                    str(s.max_rpm),
                    CHECK if s.continuous else "",
                    CHECK if s.intermittent else "",
                    s.pressure,
                ]
                for s in sec.stages
            ]
            widths = [70, 90, 90, 70, 70, 100]
        else:
            header = ["Stage", "Speed [rpm]", "Duty", "Max. pressure [bar]"]
            rows = [
                [s.name, f"{s.min_rpm}{' – ' if i % 3 == 0 else ' - '}{s.max_rpm}", s.duty, s.pressure]
                for i, s in enumerate(sec.stages)
            ]
            widths = [90, 120, 150, 110]
        parts = [rows[i : i + rows_per_page] for i in range(0, len(rows), rows_per_page)]
        for p_idx, part in enumerate(parts):
            if p_idx:
                story += [
                    PageBreak(),
                    _letterhead(doc_id),
                    Spacer(1, 12),
                    Paragraph(f"Table {sec.number}-1 (continued)", body),
                ]
            no_header = p_idx > 0 and sec.layout == "min_max"
            story.append(_grid(part if no_header else [header, *part], widths, header=not no_header, font=font))
        if sec.run_in:
            story += [
                Spacer(1, 10),
                Paragraph(f"Table {sec.number}-2: Run-in time", body),
                _grid([["Run-in continuous [h]", "Run-in intermittent [h]"], list(sec.run_in)], [150, 150]),
            ]
        if sec.modes_table:
            story += [
                Spacer(1, 10),
                Paragraph(f"Table {sec.number}-3: Operating modes", body),
                _grid(
                    [["Mode", "Description", "Monitoring"], ["M1", "Standby", "Current"], ["M3", "Full load", "Bus"]],
                    [100, 160, 120],
                ),
            ]
        story.append(PageBreak())
    story += [
        _letterhead(doc_id),
        Spacer(1, 12),
        Paragraph("Appendix A Measurement conditions", h2),
        Paragraph("Informative only.", body),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(path), pagesize=A4, title="PX product family technical datasheet").build(
        story, onFirstPage=footer, onLaterPages=footer
    )
    return sections


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "example")
    print(build_doc(out / "example_datasheet.pdf", make_sections(7))[0])
