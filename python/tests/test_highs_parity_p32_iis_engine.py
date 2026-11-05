"""highs-parity P3.2: the MILP IIS engine behind a solver backend; Rust by default.

``_iis_milp._Engine``'s solver calls moved behind ``discopt._iis_backend``: HiGHS (the
pre-P3.2 code, unchanged -- P0.6's 53-fixture baseline still passes element for element
with ``iis_engine="highs"``) and the in-house MILP driver through a ``MilpHandle``. On two
fixtures the driver could not certify an infeasibility over free columns
(``FarkasRejectOpen``): the probe stayed undecided and the set, still proven infeasible,
was not irreducible. ``DISCOPT_FARKAS_EXACT_OPEN`` (default ON since its §5 panel)
certifies those exactly. ``"highs"`` stays the IIS default: on the broader IIS suite the
Rust driver is still worse on big-M integrality slack and pigeonhole (plan P3.2).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("highspy")

_ROOT = Path(__file__).resolve().parents[2]
_DIR = _ROOT / "python" / "tests" / "data" / "highs_parity" / "iis"
_EXPECTED = json.loads((_DIR / "expected.json").read_text())["models"]
#: Fixtures whose IIS needs the exact open-column Farkas check: with
#: ``DISCOPT_FARKAS_EXACT_OPEN=0`` the Rust set is a strict superset of the baseline.
EXACT_FARKAS_FIXTURES = {"1ec8e0a9f8f1c72c", "97083706a8966d15"}


@pytest.fixture(scope="module")
def panel():
    sys.path.insert(0, str(_ROOT / "discopt_benchmarks" / "scripts"))
    try:
        import iis_parity_panel

        yield iis_parity_panel
    finally:
        sys.path.remove(str(_ROOT / "discopt_benchmarks" / "scripts"))


def _sig(r):
    return {k: r[k] for k in ("constraints", "bounds", "integrality")}


def test_unknown_engine_is_refused():
    import discopt.modeling as dm
    from discopt.infeasibility import compute_iis

    m = dm.Model("inf")
    x = m.integer("x", lb=0, ub=3)
    m.subject_to(2 * x == 3, name="odd")
    m.minimize(0 * x)
    with pytest.raises(ValueError, match="iis_engine"):
        compute_iis(m, include_integrality=True, iis_engine="hihgs")


# e5580a1c8b69a9ae: a free-column LP probe the driver cannot certify; the backend's
# LP ladder decides it. 6e890b49289ca18d: an ordinary MILP fixture.
@pytest.mark.parametrize("sha", ["e5580a1c8b69a9ae", "6e890b49289ca18d"])
def test_rust_engine_returns_the_baseline_iis(panel, sha):
    got = panel._worker(os.fspath(_DIR / f"{sha}.mps"), 30.0, "rust")
    assert _sig(got) == _sig(_EXPECTED[sha])
    assert got["proven_irreducible"] and got["complete"]


def _worker_subprocess(sha: str, flag: str):
    """The panel worker in a fresh process (the Rust flag is read once per process)."""
    import subprocess

    script = _ROOT / "discopt_benchmarks" / "scripts" / "iis_parity_panel.py"
    env = dict(os.environ)
    env.pop("DISCOPT_FARKAS_EXACT_OPEN", None)
    if flag:
        env["DISCOPT_FARKAS_EXACT_OPEN"] = flag
    p = subprocess.run(
        [sys.executable, str(script), "_worker", str(_DIR / f"{sha}.mps"), "30", "rust"],
        env=env, capture_output=True, text=True, timeout=240,
    )  # fmt: skip
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    assert p.returncode == 0 and len(lines) == 1, p.stderr[-2000:]
    return json.loads(lines[0][len("RESULT ") :])


@pytest.mark.parametrize("sha", sorted(EXACT_FARKAS_FIXTURES))
def test_exact_farkas_returns_the_baseline_iis(sha):
    """Default environment (``DISCOPT_FARKAS_EXACT_OPEN`` ON since P3.2): the Rust engine
    returns P0.6's baseline IIS, proven irreducible (a strict superset with ``=0``)."""
    got = _worker_subprocess(sha, "")
    assert _sig(got) == _sig(_EXPECTED[sha])
    assert got["proven_irreducible"] and got["complete"]


@pytest.mark.parametrize("sha", sorted(EXACT_FARKAS_FIXTURES))
def test_opt_out_reproduces_the_gap_and_reports_it(sha):
    """``DISCOPT_FARKAS_EXACT_OPEN=0``: the probe stays undecided, the set is a strict
    superset of the baseline IIS, still proven infeasible, and NOT claimed irreducible."""
    got = _worker_subprocess(sha, "0")
    want = _sig(_EXPECTED[sha])
    for k in want:
        assert {json.dumps(e) for e in want[k]} <= {json.dumps(e) for e in got[k]}
    assert _sig(got) != want
    assert got["proven_irreducible"] is False
