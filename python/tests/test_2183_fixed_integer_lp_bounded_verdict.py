"""#2183: the #1509 fixed-integer LP check declined a sound MILP certificate.

The #1509 refutation check solves the NS-safe LP over the HiGHS incumbent's own
integer assignment and holds the reported MILP bound against its point. With
highspy >= 1.13, on GDPlib ``med_term_purchasing`` (big-M) that LP returns a verified
feasible point whose NS gap does not close (0.00153 against a yardstick of 7.8e-6).
``solve_lp_std`` then reports ``feasible``, and the check called itself
"inconclusive" and declined the certificate.

But the check needs a falsifier, not LP optimality. The LP optimum lies in
``[ns, obj_fx]``. When ``ns`` already sits within the refutation yardstick of the
reported bound, no point on that integer assignment can refute it. Measured:
``ns = -6797.541`` against the HiGHS bound ``-6798.199`` (minimisation terms). That is
a rigorous verdict. The fix (``_fixed_integer_lp_cannot_refute``) issues it, and keeps
the merely-feasible point out of the #1612 / #1733 / #2151 paths that assume the LP
optimum.

``data/highs_2183_medterm_bigm_milp.npz`` is the standard form the route receives for
``med_term_purchasing`` big-M (the vendored Pyomo example in
``discopt_benchmarks/data/pyomo_gdp_examples``), captured with
``conv.model.solve(gdp_method="big-m")``. Its oracle is GDPlib's 6797.539701513444
(maximisation).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from discopt.solvers import lp_milp_highs
from discopt.solvers.lp_milp_highs import (
    CERT_ABS,
    CERT_REL,
    HighsOutcome,
    StdForm,
    solve_milp_std,
)

pytest.importorskip("highspy")

DATA = Path(__file__).parent / "data" / "highs_2183_medterm_bigm_milp.npz"
#: GDPlib's reference optimum of med_term_purchasing (maximisation), as a minimisation.
MEDTERM_OPT = -6797.539701513444


def _medterm() -> tuple[StdForm, dict]:
    with np.load(DATA) as d:
        A = sp.csc_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
        sf = StdForm.from_arrays(
            d["c"], A, d["b"], d["xl"], d["xu"], float(d["obj_const"]), d["int_idx"]
        )
        n_struct = int(d["n_struct"])
        kw = {
            "gap_tolerance": float(d["gap_tolerance"]),
            "n_struct": None if n_struct < 0 else n_struct,
        }
    return sf, kw


@pytest.mark.slow
@pytest.mark.correctness
def test_medterm_bigm_milp_is_certified():
    """The issue's instance, end to end. The time limit is a backstop: the solve
    measured 6.0 s on a dev machine.

    Platform-dependent: whether HiGHS's fixed-integer LP comes back ``feasible`` (NS gap
    unclosed) or ``optimal`` depends on the build. On Linux with highspy 1.13 through
    1.15.1 it is ``feasible``, and this test fails on the pre-#2183 code. On macOS with
    highspy 1.15.1 it is ``optimal``, the old arm settles the check, and the test also
    passes on the old code there. The fast monkeypatched tests below are the
    platform-independent regression guard. This one asserts that SOME arm settled the
    check, and names the arm in its failure messages (CLAUDE.md §6)."""
    sf, kw = _medterm()
    assert (sf.n, sf.m, sf.int_idx.size) == (1206, 924, 216)
    out = solve_milp_std(sf, time_limit=240.0, max_nodes=100_000, **kw)
    assert out.stats.get("milp/fixed_int_check_ran") == 1.0
    assert not out.stats.get("milp/fixed_int_check_inconclusive"), out.message
    bounded = bool(out.stats.get("milp/fixed_int_check_bounded"))
    optimal_arm = "milp/fixed_int_objective" in out.stats
    # The probe fired: exactly one arm gave the check its verdict.
    assert bounded != optimal_arm, (bounded, optimal_arm, out.stats)
    arm = "bounded (#2183)" if bounded else "optimal-LP (pre-#2183 arm)"
    assert out.status == "optimal" and out.gap_certified, (arm, out.message)
    assert out.objective == pytest.approx(MEDTERM_OPT, rel=1e-9)
    # Certificate invariant against the oracle (min sense): bound <= objective and
    # bound never above the true optimum.
    assert out.bound is not None
    assert out.bound <= out.objective
    assert out.bound <= MEDTERM_OPT + 1e-6 * (1.0 + abs(MEDTERM_OPT))
    if bounded:
        # The new arm settled the check: its NS bound is on the safe side of the bound,
        # and the merely-feasible LP point never posed as the LP optimum.
        ns = out.stats["milp/fixed_int_ns_bound"]
        assert ns >= out.bound - (CERT_ABS + CERT_REL * abs(out.bound))
        assert "milp/fixed_int_objective" not in out.stats
        assert out.refuted_by == ""


def _knapsack_sf() -> StdForm:
    """max x0 + x1 s.t. x0 + x1 <= 10, x0, x1 integer -- HiGHS returns kOptimal at -10."""
    return StdForm.from_arrays(
        c=np.array([-1.0, -1.0]),
        A=np.array([[1.0, 1.0]]),
        b=np.array([10.0]),
        xl=np.array([0.0, 0.0]),
        xu=np.array([1e20, 1e20]),
        int_idx=np.array([0, 1]),
    )


def _feasible_only(ns: float, calls: list):
    """``_fixed_integer_lp`` whose LP reaches the true point but whose NS gap 'does not
    close': status ``feasible`` with NS bound ``ns``, as ``solve_lp_std`` reports it."""
    real = lp_milp_highs._fixed_integer_lp

    def fake(sf, x, time_limit):
        fx = real(sf, x, time_limit)
        assert fx.status == "optimal", fx.message
        calls.append(fx)
        return HighsOutcome(
            "feasible",
            x=fx.x,
            objective=fx.objective,
            bound=ns,
            row_dual=fx.row_dual,
            col_dual=fx.col_dual,
            highs_status=fx.highs_status,
            message=f"NS safe bound {ns:.12g} is {fx.objective - ns:.3g} below the objective",
        )

    return fake


def test_feasible_lp_with_safe_ns_bound_keeps_the_certificate(monkeypatch):
    """NS bound on the safe side of the reported bound (here equal to it): the check
    has its verdict, so the certificate stands. The merely-feasible point stays out of
    the paths that assume an LP optimum."""
    calls: list = []
    monkeypatch.setattr(lp_milp_highs, "_fixed_integer_lp", _feasible_only(-10.0, calls))
    out = solve_milp_std(_knapsack_sf(), time_limit=30.0, gap_tolerance=1e-4, max_nodes=1000)
    assert calls, "the fixed-integer LP check never ran, so this asserts nothing"
    assert out.stats.get("milp/fixed_int_check_bounded") == 1.0
    assert not out.stats.get("milp/fixed_int_check_inconclusive")
    assert "milp/fixed_int_objective" not in out.stats
    assert "milp/incumbent_rederived" not in out.stats
    assert not out.stats.get("milp/certificate_refuted")
    assert out.status == "optimal" and out.gap_certified
    assert out.objective == -10.0 and out.bound == -10.0


def test_feasible_lp_with_ns_bound_below_the_yardstick_still_declines(monkeypatch):
    """NS bound further below the reported bound than the yardstick: some point on
    this integer assignment might refute the bound, and the LP did not say which. That
    is still "inconclusive", and the certificate is still declined (#1320)."""
    calls: list = []
    thr = CERT_ABS + CERT_REL * 10.0
    monkeypatch.setattr(lp_milp_highs, "_fixed_integer_lp", _feasible_only(-10.0 - 2 * thr, calls))
    out = solve_milp_std(_knapsack_sf(), time_limit=30.0, gap_tolerance=1e-4, max_nodes=1000)
    assert calls, "the fixed-integer LP check never ran, so this asserts nothing"
    assert out.stats.get("milp/fixed_int_check_inconclusive") == 1.0
    assert not out.stats.get("milp/fixed_int_check_bounded")
    assert out.status == "feasible" and not out.gap_certified
    assert out.labels.get("milp/certificate") == "declined"


def test_feasible_lp_with_unverified_point_still_declines(monkeypatch):
    """A safe NS bound alone is not enough. The arm also requires a verified point, so
    an LP point that fails the incumbent verifier stays "inconclusive"."""
    calls: list = []
    fake = _feasible_only(-10.0, calls)

    def bad_point(sf, x, time_limit):
        fx = fake(sf, x, time_limit)
        fx.x = np.array([20.0, 20.0])  # violates x0 + x1 <= 10
        return fx

    monkeypatch.setattr(lp_milp_highs, "_fixed_integer_lp", bad_point)
    out = solve_milp_std(_knapsack_sf(), time_limit=30.0, gap_tolerance=1e-4, max_nodes=1000)
    assert calls, "the fixed-integer LP check never ran, so this asserts nothing"
    assert out.stats.get("milp/fixed_int_check_inconclusive") == 1.0
    assert out.status == "feasible" and not out.gap_certified


def test_bound_cannot_refute_helper_edges():
    """The predicate itself: only ``feasible`` LPs with a finite NS bound inside the
    yardstick and a verified point qualify, and it returns that verified point."""
    sf = _knapsack_sf()
    x = np.array([10.0, 0.0])
    f = lp_milp_highs._fixed_integer_lp_cannot_refute
    ok = HighsOutcome("feasible", x=x, objective=-10.0, bound=-10.0)
    got = f(sf, ok, -10.0)
    assert got is not None
    assert got[1] == -10.0 and np.array_equal(got[0], x)
    assert f(sf, ok, -10.0 + 0.5 * CERT_ABS) is not None  # inside the yardstick
    assert None is f(sf, ok, -9.0)  # bound above ns by more than the yardstick
    assert None is f(sf, HighsOutcome("optimal", x=x, objective=-10.0, bound=-10.0), -10.0)
    assert None is f(sf, HighsOutcome("feasible", x=x, objective=-10.0, bound=None), -10.0)
    assert None is f(sf, HighsOutcome("feasible", x=x, objective=-10.0, bound=-np.inf), -10.0)
    assert None is f(sf, HighsOutcome("feasible", x=None, objective=-10.0, bound=-10.0), -10.0)
