"""The exact-path diagnosis on SpecQA: every operation that answers is right there."""

from tablerag.bench.diagnose import diagnose, to_markdown
from tablerag.bench.specqa import build_specqa


def test_diagnose_specqa(tmp_path):
    pdfs, questions = build_specqa(tmp_path / "specqa")
    report = diagnose("specqa", pdfs, questions)
    assert report["questions"] == len(questions) and report["query_changed_by_cleanup"] == 0
    assert report["operations"] and all(v["wrong"] == 0 for v in report["operations"].values())
    md = to_markdown("specqa", report)
    assert "| range_lookup |" in md
