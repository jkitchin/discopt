"""highs-parity P0.5: the R3 / R4 / R6 LP fixtures load and HiGHS reproduces them.

``python/tests/data/highs_parity/`` holds LPs captured from real solves by
``discopt_benchmarks/scripts/capture_highs_parity_fixtures.py``: the NLP-BB root LPs with the
GMI cuts HiGHS's basis produced (R3), the slack-form convex node LPs whose row dual feeds the
node certificate (R4), and the verified LPs the MINLP path hands ``lp_milp_highs`` (R6: the
#1537 arbitration, the #1611 linear-node rescue, the #1869 free-column certificate). They are
the inputs the P1 ports are judged on, so this pins that each one loads with consistent shapes
and that plain HiGHS still reproduces the stored objective to 1e-9 (relative). A fixture HiGHS
no longer reproduces is a moved baseline, not a passing test.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_DIR = _ROOT / "python" / "tests" / "data" / "highs_parity"
_SCRIPT = _ROOT / "discopt_benchmarks" / "scripts" / "capture_highs_parity_fixtures.py"
_MANIFEST = json.loads((_DIR / "manifest.json").read_text())["fixtures"]


@pytest.fixture(scope="module")
def cap():
    pytest.importorskip("highspy")
    spec = importlib.util.spec_from_file_location("capture_hp_p05", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_manifest_meets_the_p05_minimums():
    roles = [f["role"] for f in _MANIFEST]
    assert roles.count("R4") >= 50
    assert sum(1 for f in _MANIFEST if f["role"] == "R3" and f["gmi_cuts"] > 0) >= 20
    assert roles.count("R6") >= 1
    # Every R6 call site the capture reached is represented, verdict-only LPs included.
    callers = {f["caller"] for f in _MANIFEST if f["role"] == "R6"}
    assert callers == {"1537_arbitration", "1611_rescue", "1869_free_column"}
    assert {f["file"] for f in _MANIFEST} == {p.name for p in _DIR.glob("*.npz")}
    assert sum(p.stat().st_size for p in _DIR.glob("*.npz")) <= 5 * 1024 * 1024


@pytest.mark.parametrize("entry", _MANIFEST, ids=[f["file"] for f in _MANIFEST])
def test_fixture_loads_and_highs_reproduces_it(cap, entry):
    fx = cap.load_fixture(_DIR / entry["file"])
    rf = fx["rf"]
    n, m = len(rf["c"]), rf["A"].shape[0]
    assert (n, m) == (entry["n"], entry["m"])
    assert rf["A"].shape == (m, n)
    for k in ("row_lo", "row_hi"):
        assert rf[k].shape == (m,)
    for k in ("col_lo", "col_hi"):
        assert rf[k].shape == (n,)
    if "x" in fx:
        assert fx["x"].shape == (n,)
    if entry["role"] == "R4":
        assert fx["y"].shape == (m,)
    if entry["role"] == "R3":
        assert fx["is_int"].shape == (n,)
        assert fx["gmi_a"].shape == (entry["gmi_cuts"], n)
        assert fx["gmi_b"].shape == (entry["gmi_cuts"],)
    status, obj = cap.highs_resolve(rf)
    want = str(fx["expected_status"])
    if want:  # a verdict (the #1537 arbitration's infeasible LPs): HiGHS must agree
        assert status == want, (status, want)
        return
    assert status == "Optimal"
    stored = float(fx["objective"])
    assert abs(obj - stored) <= 1e-9 * max(1.0, abs(stored)), (obj, stored)


_HARD = json.loads((_DIR / "manifest.json").read_text())["hard"]


@pytest.mark.parametrize("entry", _HARD, ids=[f["file"] for f in _HARD])
def test_hard_fixture_route_point_is_feasible(cap, entry):
    """The LPs plain HiGHS (presolve off) cannot reproduce: the verified route answered
    them, HiGHS alone returns ``Unknown``/``Infeasible`` or another objective. Captured on
    nvs16's #1869 free-column certificate LPs (columns boxed at 1.6e9). The route's
    point is held to the row-scaled tolerance the route's own verification uses."""
    fx = cap.load_fixture(_DIR / entry["file"])
    rf = fx["rf"]
    n, m = len(rf["c"]), rf["A"].shape[0]
    assert (n, m) == (entry["n"], entry["m"])
    assert str(fx["plain_highs_status"]) == entry["plain_highs_status"]
    x = fx["x"]
    assert x.shape == (n,) and np.all(np.isfinite(x))
    act = rf["A"] @ x
    scale = 1.0 + abs(rf["A"]) @ np.abs(x)
    viol = np.maximum(rf["row_lo"] - act, act - rf["row_hi"]) / scale
    assert float(np.max(viol, initial=0.0)) <= 1e-6
    assert np.all(x >= rf["col_lo"] - 1e-6 * (1.0 + np.abs(rf["col_lo"])))
    assert np.all(x <= rf["col_hi"] + 1e-6 * (1.0 + np.abs(rf["col_hi"])))
