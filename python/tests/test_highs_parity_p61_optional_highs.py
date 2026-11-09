"""highs-parity P6.1: one module reaches HiGHS, and a missing highspy is a named error."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

ROOT = Path(__file__).resolve().parents[1] / "discopt"


def test_import_highspy_appears_only_in_the_optional_module():
    hits = []
    for p in ROOT.rglob("*.py"):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if re.match(r"\s*(import highspy|from highspy import)", line):
                hits.append(f"{p.relative_to(ROOT)}:{i}")
    assert len(hits) == 1 and hits[0].startswith("_optional_highs.py:"), hits


def test_missing_highspy_is_a_named_module_not_found_error(monkeypatch):
    from discopt import _optional_highs as oh
    from discopt.solvers import lp_milp_highs, milp_highs

    monkeypatch.setitem(sys.modules, "highspy", None)
    assert oh.highs_available() is False
    with pytest.raises(oh.MissingHighsError) as ei:
        oh.require_highspy("the test feature")
    err = ei.value
    assert isinstance(err, ModuleNotFoundError) and err.name == "highspy"
    assert "the test feature" in str(err) and "discopt[highs]" in str(err)
    # The historical names are the same class, so existing handlers still catch it.
    assert lp_milp_highs.HighsUnavailable is oh.MissingHighsError
    assert milp_highs.HighsBackendUnavailable is oh.MissingHighsError
    with pytest.raises(lp_milp_highs.HighsUnavailable):
        lp_milp_highs.require_highspy()


def test_available_when_installed():
    pytest.importorskip("highspy")
    from discopt import _optional_highs as oh

    assert oh.highs_available() is True
    assert oh.require_highspy("x").__name__ == "highspy"
