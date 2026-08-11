"""#956: the envelope outward-rounding guard — what it must do, and what it must not.

The guard relaxes every envelope rhs and auxiliary bound outward by an ulp-scaled
slack so the exactly-feasible point ``(x, f(x))`` can never fall outside the
envelope that was built to contain it. Two properties are load-bearing and are
pinned here:

* **it only ever loosens** — a larger ``<=`` rhs and a wider aux box admit more
  points, so the guard can never cut a feasible point nor lift a bound above the
  true optimum; and
* **off, it is the exact identity** — ``DISCOPT_ENVELOPE_OUTWARD_ROUND=0`` must
  restore the pre-#956 relaxation *byte* for byte, not merely value for value.

The second is the one that bit. ``rhs + 0.0`` is not the identity: ``-0.0 + 0.0``
is ``+0.0``, so a no-op add clears the sign bit of every negative-zero rhs. The LP
cannot tell (``-0.0 == 0.0``), but ``relaxation_fingerprint`` hashes the matrix
bytes, and the claim baseline gates on that hash. Measured before the fix, with the
guard OFF: 34 of the 66 vendored corpus instances' fingerprints moved away from the
committed baseline (22 of ``4stufen``'s 247 ``b_ub`` entries flip, 1 of ``alan``'s
10), which is what took ``test_961_optimal_requires_bound`` red on a PR whose whole
claim was that the opt-out changes nothing.

``outward_rounding_enabled()`` caches its env read in a module global (all three
engines must agree row for row, so it may not be re-read mid-build). These tests
therefore set that global directly and restore it, rather than the environment.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax import outward_rounding as orr
from discopt._relax.outward_rounding import (
    envelope_1d_slack,
    envelope_product_slack,
    outward_slack,
    relaxed_rhs,
    widen,
)


@pytest.fixture
def guard(monkeypatch):
    """Set the cached guard state explicitly; ``None`` restores the env-derived one."""

    def _set(on: bool):
        monkeypatch.setattr(orr, "_ENABLED", on, raising=False)

    return _set


def _bits(x: float) -> bytes:
    """The f64's bytes — what the fingerprint hashes, and what ``==`` cannot see."""
    return np.float64(x).tobytes()


# --- off, the guard is the exact identity ------------------------------------


def test_relaxed_rhs_preserves_negative_zero_when_the_slack_is_zero(guard):
    guard(False)
    checked = 0
    for rhs in (-0.0, 0.0, -1.5, 3.25, -1e18, 1e-18):
        s = outward_slack(abs(rhs))
        assert s == 0.0, "guard is off; the slack must be identically zero"
        got = relaxed_rhs(rhs, s)
        assert _bits(got) == _bits(rhs), f"rhs {rhs!r} changed bytes: {got!r}"
        checked += 1
    assert checked == 6, f"only {checked} rhs values compared"
    # The naive form is what this replaced -- pin that it really does differ, so a
    # revert cannot pass this test (CLAUDE.md §6).
    assert _bits(-0.0 + 0.0) != _bits(-0.0)


def test_widen_is_the_exact_identity_when_the_guard_is_off(guard):
    guard(False)
    checked = 0
    for lo, hi in ((-0.0, -0.0), (0.0, 0.0), (-2.5, 7.0), (-1e20, 1e20), (-0.0, 4.0)):
        got_lo, got_hi = widen(lo, hi)
        assert _bits(got_lo) == _bits(lo), f"lo {lo!r} changed bytes: {got_lo!r}"
        assert _bits(got_hi) == _bits(hi), f"hi {hi!r} changed bytes: {got_hi!r}"
        checked += 2
    assert checked == 10, f"only {checked} bounds compared"


def test_the_cold_build_keeps_its_negative_zeros_with_the_guard_off(guard):
    """End-to-end: the property the claim-baseline fingerprint actually gates on.

    ``4stufen`` is a gate probe, not a special case — it is simply the smallest
    vendored instance whose emitters produce negative-zero right-hand sides. Before
    the fix this count was 0 and its fingerprint no longer matched the baseline.
    """
    from pathlib import Path

    from discopt._relax.discretization import DiscretizationState
    from discopt._relax.milp_relaxation import build_milp_relaxation
    from discopt._relax.term_classifier import classify_nonlinear_terms
    from discopt.modeling.core import from_nl

    guard(False)
    nl = Path(__file__).parent / "data" / "minlplib_nl" / "4stufen.nl"
    model = from_nl(str(nl))
    relax, _info = build_milp_relaxation(
        model, classify_nonlinear_terms(model), DiscretizationState()
    )
    b = np.asarray(relax._b_ub, dtype=np.float64)
    assert b.size, "no rhs vector was built -- test is dead"
    n_neg_zero = int(np.sum((b == 0.0) & np.signbit(b)))
    assert n_neg_zero > 0, (
        "the guard-off cold build produced no negative-zero rhs; either the emitters "
        "changed or the guard is silently normalizing them (#956)"
    )


# --- on, the guard only ever loosens -----------------------------------------


def test_the_guard_only_ever_loosens(guard):
    guard(True)
    checked = 0
    for rhs in (-3.0, -0.0, 0.0, 12.5, 1e12):
        s = envelope_1d_slack(2.0, -4.0, 6.0, 1.0, 9.0, 0.5, rhs)
        assert s > 0.0
        assert relaxed_rhs(rhs, s) > rhs, f"rhs {rhs!r} did not loosen"
        checked += 1
    for lo, hi in ((-5.0, 15.0), (0.0, 9.0), (-2e6, -0.1)):
        wlo, whi = widen(lo, hi)
        assert wlo < lo and whi > hi, f"[{lo}, {hi}] did not widen"
        checked += 1
    assert checked == 8, f"only {checked} loosenings compared"


def test_an_unbounded_box_degrades_to_the_absolute_floor_not_a_1e20_guard(guard):
    """The infinity sentinel is ``1e20``; scaling a guard by it would relax by 3.6e5."""
    guard(True)
    floor = outward_slack(0.0)
    checked = 0
    for args in (
        (1.0, -1e20, 5.0, 0.0, 25.0, 0.0, 1.0),
        (1.0, -5.0, 1e20, 0.0, 25.0, 0.0, 1.0),
    ):
        assert envelope_1d_slack(*args) == floor
        checked += 1
    assert envelope_product_slack(1.0, 1.0, -1e20, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0) == floor
    checked += 1
    assert checked == 3, f"only {checked} unbounded cases compared"
    assert floor < 1e-14, f"absolute floor {floor} is not an ulp-scale guard"


def test_a_nonfinite_magnitude_contributes_zero_never_a_nan(guard):
    guard(True)
    checked = 0
    for v in (float("inf"), float("-inf"), float("nan")):
        assert orr.bounded_mag(v) == 0.0
        assert not orr.box_finite(v)
        checked += 1
    assert checked == 3
    s = envelope_1d_slack(float("nan"), -1.0, 1.0, 0.0, 1.0, 0.0, 0.5)
    assert math.isfinite(s) and s > 0.0, f"a NaN slope produced {s}"


# --- the invariant the guard exists to hold ----------------------------------

#: Boxes and points where the UNGUARDED cold build demonstrably cuts ``(x, f(x))``
#: out of its own envelope. Found by sweeping boxes over ten orders of magnitude
#: and scoring both corners and the interior; the recorded ``min_violation`` is the
#: residual the guard-off build produces there, which
#: :func:`test_the_guard_off_build_really_does_violate` pins as a negative control.
#: Small boxes are useless here — the defect is an ulp of the ROW's magnitude, so it
#: only becomes a resolvable absolute residual once that magnitude is large.
_VIOLATING_CASES = {
    "bilinear": {
        "x": (57699999.99999999, 314159000.0),
        "y": (61800000.0, 271800000.0),
        "at": "lo",
        "min_violation": 1.0,
    },
    "square": {
        "x": (-31830988.618379068, 127323954.47351627),
        "at": "hi",
        "min_violation": 0.5,
    },
    "cubic": {
        "x": (-318309.8861837907, 1273239.5447351628),
        "at": "lo",
        "min_violation": 64.0,
    },
}


def _build_case(expr_name):
    """The model, the on-graph lifted point, and the aux values, for one case."""
    import discopt.modeling as dm

    spec = _VIOLATING_CASES[expr_name]
    end = 0 if spec["at"] == "lo" else 1
    m = dm.Model(f"outward_{expr_name}")
    xlo, xhi = spec["x"]
    x = m.continuous("x", lb=xlo, ub=xhi)
    xv = spec["x"][end]
    if expr_name == "bilinear":
        ylo, yhi = spec["y"]
        y = m.continuous("y", lb=ylo, ub=yhi)
        m.minimize(x * y)
        yv = spec["y"][end]
        point, aux = {"x": xv, "y": yv}, [xv * yv]
    elif expr_name == "square":
        m.minimize(x * x)
        point, aux = {"x": xv}, [xv * xv]
    else:
        m.minimize(x * x * x)
        # The association order is load-bearing: the DAG evaluates ``x*x*x`` as
        # ``(x*x)*x``, and ``x**3`` is a DIFFERENT f64 — one ulp away. Scoring with
        # ``**3`` reports a spurious violation of a row that is in fact exactly
        # tight, which is a defect in the probe, not in the relaxation.
        point, aux = {"x": xv}, [(xv * xv) * xv]
    m.subject_to(x >= x.lb)
    return m, point, aux


def _worst_envelope_residual(m, point, aux, expr_name):
    """``max(A z - b)`` at the exactly-on-graph lifted point. Positive means cut."""
    from discopt._relax.discretization import DiscretizationState
    from discopt._relax.milp_relaxation import build_milp_relaxation
    from discopt._relax.term_classifier import classify_nonlinear_terms

    relax, _info = build_milp_relaxation(m, classify_nonlinear_terms(m), DiscretizationState())
    A = relax._A_ub
    A = np.asarray(A.todense()) if sp.issparse(A) else np.asarray(A)
    b = np.asarray(relax._b_ub, dtype=np.float64)
    assert A.ndim == 2 and A.shape[0] > 0, "no envelope rows were emitted -- test is dead"

    names = [v.name for v in m._variables]
    n_orig = len(names)
    assert A.shape[1] == n_orig + len(aux), (
        f"{expr_name}: expected {len(aux)} auxiliary column(s), got {A.shape[1] - n_orig} "
        "-- the lifting changed and this point is no longer on the graph"
    )
    z = np.zeros(A.shape[1], dtype=np.float64)
    for j, nm in enumerate(names):
        z[j] = point[nm]
    for k, val in enumerate(aux):
        z[n_orig + k] = val

    resid = A @ z - b
    assert resid.size, "no rows evaluated -- test is dead"
    return float(np.max(resid)), int(resid.size)


@pytest.mark.parametrize("expr_name", sorted(_VIOLATING_CASES))
def test_the_exactly_feasible_point_satisfies_every_emitted_row(guard, expr_name):
    """``(x, f(x))`` must satisfy every envelope row of the cold build, exactly.

    This is the #956 defect itself: unguarded, the closed forms cut their own graph
    by ~1 ulp of the row's magnitude, which on a large box is an absolute residual
    no LP tolerance absorbs. The assertion is at zero tolerance.
    """
    guard(True)
    m, point, aux = _build_case(expr_name)
    worst, n_rows = _worst_envelope_residual(m, point, aux, expr_name)
    assert worst <= 0.0, (
        f"{expr_name}: the exactly-feasible point violates its own envelope by "
        f"{worst:g} over {n_rows} rows (#956)"
    )


@pytest.mark.parametrize("expr_name", sorted(_VIOLATING_CASES))
def test_the_guard_off_build_really_does_violate(guard, expr_name):
    """Negative control: without the guard, each case above IS cut. (CLAUDE.md §6.)

    Without this, the test above would pass identically with the guard on and off —
    it would assert nothing about the guard. That is not hypothetical: the first
    version of this file used boxes around 5e3, where all three cases are exactly
    tight unguarded, and it passed in both arms.
    """
    guard(False)
    m, point, aux = _build_case(expr_name)
    worst, n_rows = _worst_envelope_residual(m, point, aux, expr_name)
    floor = _VIOLATING_CASES[expr_name]["min_violation"]
    assert worst >= floor, (
        f"{expr_name}: guard-off residual is {worst:g} over {n_rows} rows, below the "
        f"recorded {floor:g} -- this case no longer discriminates, so the positive "
        "test above is a tautology. Re-sweep for a box that still bites."
    )
