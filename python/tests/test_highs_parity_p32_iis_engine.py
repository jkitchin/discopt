"""highs-parity P3.2: the MILP IIS engine behind a solver backend; Rust as an opt-in.

``_iis_milp._Engine``'s solver calls moved behind ``discopt._iis_backend``: HiGHS (the
pre-P3.2 code, unchanged -- P0.6's 53-fixture baseline still passes element for element)
and the in-house MILP driver through a ``MilpHandle``. On P0.6's fixtures the Rust
engine returns the baseline IIS on 51/53; on the other two its driver cannot certify an
infeasibility over free columns (``FarkasRejectOpen`` in the frozen simplex), so a probe
stays undecided, its element is kept, and the set -- still proven infeasible -- is
reported ``proven_irreducible=False``. Until that is decided ``"highs"`` stays the default.
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
#: Fixtures where the Rust engine's set is a strict superset of the baseline IIS
#: (measured 2026-11-05; integer probes the driver leaves undecided).
KNOWN_GAP = {"1ec8e0a9f8f1c72c", "97083706a8966d15"}


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


@pytest.mark.parametrize("sha", sorted(KNOWN_GAP))
def test_known_gap_is_reported_not_hidden(panel, sha):
    """The two fixtures still differing: the set is a strict superset of the baseline
    IIS, still proven infeasible, and NOT claimed irreducible. This test turns red when
    the gap closes (then move the fixture out of ``KNOWN_GAP``)."""
    got = panel._worker(os.fspath(_DIR / f"{sha}.mps"), 30.0, "rust")
    want = _sig(_EXPECTED[sha])
    for k in want:
        assert {json.dumps(e) for e in want[k]} <= {json.dumps(e) for e in got[k]}
    assert _sig(got) != want
    assert got["proven_irreducible"] is False


@pytest.mark.parametrize("sha", sorted(KNOWN_GAP))
def test_exact_farkas_closes_the_gap(sha):
    """With ``DISCOPT_FARKAS_EXACT_OPEN=1`` (read once per process, so a subprocess)
    the driver certifies the free-column infeasibilities exactly and the Rust engine
    returns P0.6's baseline IIS, proven irreducible."""
    import subprocess

    script = _ROOT / "discopt_benchmarks" / "scripts" / "iis_parity_panel.py"
    env = dict(os.environ, DISCOPT_FARKAS_EXACT_OPEN="1")
    p = subprocess.run(
        [sys.executable, str(script), "_worker", str(_DIR / f"{sha}.mps"), "30", "rust"],
        env=env, capture_output=True, text=True, timeout=240,
    )  # fmt: skip
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
    assert p.returncode == 0 and len(lines) == 1, p.stderr[-2000:]
    got = json.loads(lines[0][len("RESULT ") :])
    assert _sig(got) == _sig(_EXPECTED[sha])
    assert got["proven_irreducible"] and got["complete"]
