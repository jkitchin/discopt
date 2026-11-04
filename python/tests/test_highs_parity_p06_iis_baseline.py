"""highs-parity P0.6: the MILP IIS baseline (role R5) is pinned element for element.

``python/tests/data/highs_parity/iis/`` holds the 53 linear models the IIS test files hand to
``compute_iis`` (exported with ``Model.to_mps`` by ``discopt_benchmarks/scripts/
iis_parity_panel.py capture``) and ``expected.json``, the IIS ``compute_iis(...,
include_integrality=True)`` returned for each on HiGHS -- reproduced identically by two
fresh-process runs per model. P3.2 replaces that engine with a Rust handle and its bar is
"identical IIS element sets on every fixture", so a change to any set here is a change to the
baseline, never a passing test. ``n_solves`` is recorded, not pinned: another engine may
take a different number of probes to the same set.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DIR = _ROOT / "python" / "tests" / "data" / "highs_parity" / "iis"
_SCRIPTS = _ROOT / "discopt_benchmarks" / "scripts"
_EXPECTED = json.loads((_DIR / "expected.json").read_text())


@pytest.fixture(scope="module")
def panel():
    pytest.importorskip("highspy")
    sys.path.insert(0, str(_SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            "iis_parity_panel_p06", _SCRIPTS / "iis_parity_panel.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod
    finally:
        sys.path.remove(str(_SCRIPTS))


def test_baseline_has_at_least_20_models_and_lists_every_file():
    assert len(_EXPECTED["models"]) >= 20
    assert set(_EXPECTED["models"]) == {p.stem for p in _DIR.glob("*.mps")}
    # Integrality is in play: some models have integers and some IIS keep them.
    assert sum(m["n_int"] > 0 for m in _EXPECTED["models"].values()) >= 20
    assert any(m["integrality"] for m in _EXPECTED["models"].values())


@pytest.mark.parametrize("sha", sorted(_EXPECTED["models"]))
def test_iis_matches_the_baseline(panel, sha):
    want = _EXPECTED["models"][sha]
    got = panel._worker(os.fspath(_DIR / f"{sha}.mps"), 30.0)
    assert (got["n"], got["m"], got["n_int"]) == (want["n"], want["m"], want["n_int"])
    assert got["constraints"] == want["constraints"]
    assert got["bounds"] == want["bounds"]
    assert got["integrality"] == want["integrality"]
    assert got["proven_irreducible"] == want["proven_irreducible"]
    assert got["complete"] == want["complete"]
