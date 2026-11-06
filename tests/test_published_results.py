"""The published numbers can be regenerated from the committed raw results."""

import json
import runpy
from pathlib import Path

from autodidact.eval.compare import compare
from autodidact.eval.report import load

ROOT = Path(__file__).resolve().parents[1]
PACK = {"restock_one", "restock_category", "fulfil_or_backorder"}


def test_headline_numbers_match_the_raw_results():
    before, after = load(ROOT / "results/docs-only"), load(ROOT / "results/learned")
    result = compare(before, after, groups={"pack": PACK})
    pack, overall = result["groups"]["pack"], result["overall"]
    assert (pack["passed_before"], pack["passed_after"], pack["attempts"]) == (15, 22, 24)
    assert (len(pack["fixed"]), len(pack["broken"])) == (7, 0)
    assert round(pack["p_value"], 3) == 0.016
    assert (overall["passed_before"], overall["passed_after"], overall["attempts"]) == (56, 61, 72)
    committed = json.loads((ROOT / "results/comparison.json").read_text())
    assert committed["overall"]["passed_after"] == overall["passed_after"]


def test_figures_and_site_data_rebuild(tmp_path, monkeypatch):
    for script in ("make_figures.py", "build_site.py"):
        runpy.run_path(str(ROOT / "scripts" / script), run_name="__main__")
    assert "22/24" in (ROOT / "docs/figures/headline.svg").read_text()
    data = (ROOT / "docs/data.js").read_text(encoding="utf-8")
    assert data.startswith("window.AUTODIDACT = ")
    payload = json.loads(data[len("window.AUTODIDACT = ") : -2])
    assert len(payload["attempts"]) == 72 and payload["summary"]["pack"]["fixed"] == 7


def test_no_secrets_in_published_files():
    for path in [*ROOT.glob("results/**/*"), *ROOT.glob("docs/**/*")]:
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            assert "AIza" not in text and "gsk_" not in text, path
