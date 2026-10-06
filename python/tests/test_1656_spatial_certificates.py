"""#1656: two spatial branch-and-bound certificates that were lost on textbook models.

1. The native Rust spatial kernel stopped on a gap its own test called closed and
   ``_refuse_unclosed_published_pair`` (#1536) then withdrew the certificate, on the
   ``(x*y - 1)**2`` least squares over ``[-100, 100]**2``. Two causes, both fixed:
   the kernel judged the RELATIVE gap on its constant-free internal objective
   (-1.10 internally, 0.0274 published, offset 1.13), and it accepted a
   McCormick-tight point at ``c'x``, which sat 7.5e-6 below the objective at that
   point because each lifted term's 1e-6 slack composed through ``x**2 * y**2``.
2. The Python spatial tree (``root_fixpoint=False``) on Haverly ended ``exhausted``
   at bound 562.5 over an incumbent of 400. A child box the incumbent cutoff had
   moved 2.9e-8 past the optimum got a Neumaier-Shcherbina "safe" bound of
   +1.7e53 from an FBBT box that had diverged (column lower bound 6.9e23 over an
   upper bound of 0); the tree read it as a failed node and fathomed it without
   proof, keeping the parent's -562.5 as a floor. FBBT now never moves a bound
   past the column's opposite bound, so it cannot diverge.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt import SolverTuning
from discopt._relax.mccormick_lp import BOUND_PROVENANCE, reset_bound_provenance
from discopt.solvers.milp_simplex import (
    _fbbt_eq_bounds,
    _safe_lp_lower_bound_sharp,
    _safe_lp_lower_bound_std,
)


def _lsq():
    m = dm.Model("lsq")
    x = m.continuous("x", lb=-100, ub=100)
    y = m.continuous("y", lb=-100, ub=100)
    m.minimize((x * y - 1) ** 2 + 0.01 * (x - 2) ** 2 + 0.01 * (y - 3) ** 2)
    return m


def test_native_kernel_certifies_the_published_pair():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _lsq().solve(time_limit=60)
    assert r.algorithm_route.startswith("native-spatial"), r.algorithm_route
    assert r.status == "optimal", (r.status, r.objective, r.bound)
    assert r.gap_certified
    assert r.termination == "gap"
    # The published pair itself closes the requested gap (rel 1e-4).
    assert r.objective - r.bound <= 1e-4 * abs(r.objective)
    assert r.bound <= r.objective
    assert r.objective == pytest.approx(0.0273541381, rel=1e-6)
    # The certificate was not withdrawn after the fact.
    assert "certificate/published_pair_refused" not in (r.solver_stats or {})
    # Probe fired (CLAUDE.md §6): the kernel priced accepted LP points, and on
    # this model at least one lifted ``c'x`` was optimistic.
    stats = r.solver_stats or {}
    assert stats["tree/incumbent_value_calls"] >= 1
    assert stats["tree/incumbent_value_raised"] >= 1


def test_kernel_declines_a_lift_looser_than_the_model():
    """MINLPLib ``prob10``: the kernel's lift is strictly looser than the model
    (``c'x = 2.345`` at a McCormick-tight point whose objective is 3.446). Priced
    honestly the kernel could never close (100 000 nodes to ``node_limit``); it
    must decline at that point, exactly as #789 declined its final incumbent
    before, and leave the model to the Python tree, which certifies 3.4455."""
    import pathlib

    from discopt.modeling.core import from_nl

    path = pathlib.Path(__file__).parent / "data" / "minlplib" / "prob10.nl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = from_nl(str(path)).solve(time_limit=30)
    assert r.status == "optimal", (r.status, r.objective, r.bound, r.algorithm_route)
    assert r.gap_certified
    assert r.algorithm_route.startswith("spatial-bb"), r.algorithm_route
    assert r.objective == pytest.approx(3.44550379, rel=1e-6)
    assert r.node_count < 1000


def _haverly():
    m = dm.Model("haverly")
    fA, fB, fC, x, y, cX, cY = (
        m.continuous(n, lb=0, ub=1000) for n in ["fA", "fB", "fC", "x", "y", "cX", "cY"]
    )
    p = m.continuous("p", lb=0, ub=5)
    m.maximize(9 * (x + cX) + 15 * (y + cY) - 6 * fA - 16 * fB - 10 * fC)
    m.subject_to(fA + fB == x + y)
    m.subject_to(3 * fA + fB == p * x + p * y)
    m.subject_to(fC == cX + cY)
    m.subject_to(x + cX <= 100)
    m.subject_to(y + cY <= 200)
    m.subject_to(p * x + 2 * cX <= 2.5 * (x + cX))
    m.subject_to(p * y + 2 * cY <= 1.5 * (y + cY))
    return m


@pytest.mark.parametrize(
    "kw",
    [{"presolve": False}, {"tuning": SolverTuning(root_fixpoint=False)}],
    ids=["presolve_off", "root_fixpoint_off"],
)
def test_python_tree_does_not_drop_an_empty_child(kw):
    reset_bound_provenance()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _haverly().solve(**kw)
    # Every node bound came from a certificate, none from a trusted vertex.
    tally = dict(BOUND_PROVENANCE)
    assert tally.get("ns_safe_bound", 0) > 0, tally
    assert tally.get("trusted_vertex", 0) == 0 and tally.get("trusted_backend", 0) == 0, tally
    assert r.algorithm_route.startswith("spatial-bb"), r.algorithm_route
    assert r.status == "optimal", (r.status, r.objective, r.bound, r.termination)
    assert r.objective == pytest.approx(400.0, rel=1e-6)
    assert r.bound == pytest.approx(400.0, rel=1e-4)
    assert r.bound >= r.objective - 1e-6 * 400  # maximize: bound is an upper bound


# ── FBBT cannot cross, at the unit ─────────────────────────────────────────


def _inconsistent_chain(n=6, drift=1e-3):
    """``x_k - 10 x_{k+1} = 0`` chained, ``x_0 = 1 + drift``, every ``x_k`` in
    ``[0, 10**-k]``: inconsistent by ``drift``, and each row multiplies the
    overshoot by 10, which is how the uncapped propagation diverged."""
    rows = []
    for k in range(n - 1):
        r = np.zeros(n)
        r[k], r[k + 1] = 1.0, -10.0
        rows.append(r)
    r0 = np.zeros(n)
    r0[0] = 1.0
    rows.append(r0)
    a = sp.csc_matrix(np.array(rows))
    b = np.r_[np.zeros(n - 1), 1.0 + drift]
    lb = np.zeros(n)
    ub = np.array([10.0**-k for k in range(n)])
    return a, b, lb, ub


def test_fbbt_never_crosses_on_an_inconsistent_system():
    a, b, lb, ub = _inconsistent_chain()
    lo, hi = _fbbt_eq_bounds(a, b, lb, ub, rounds=10)
    assert np.all(lo <= hi), (lo, hi)
    # Never outside the incoming box on a finite side.
    assert np.all(lo >= lb) and np.all(hi <= ub)


def test_fbbt_still_tightens_a_consistent_system():
    a = sp.csc_matrix(np.array([[1.0, 1.0]]))
    lo, hi = _fbbt_eq_bounds(a, np.array([1.5]), np.array([-np.inf, 0.0]), np.array([1.0, 1.0]))
    assert lo[0] == pytest.approx(0.5)
    assert hi[1] == pytest.approx(1.0)


def _feedback_pair(d=0.5):
    """``x0 - 1000 x1 = d``, ``x1 - 1000 x0 = d``, ``x0 <= 1`` (open below),
    ``x1 in [0, 1]``: empty, and each FBBT round multiplies the overshoot by 1000
    through the two rows -- the feedback that took a Haverly column to 6.9e23."""
    a = sp.csc_matrix(np.array([[1.0, -1000.0], [-1000.0, 1.0]]))
    b = np.array([d, d])
    lb = np.array([-np.inf, 0.0])
    ub = np.array([1.0, 1.0])
    return a, b, lb, ub


@pytest.mark.parametrize("fn", [_safe_lp_lower_bound_std, _safe_lp_lower_bound_sharp])
def test_ns_bound_stays_inside_the_box_on_an_inconsistent_system(fn):
    """``min x0 + x1``: column 0's reduced cost selects its open lower side, so the
    evaluation consults FBBT. Uncapped, FBBT returned ``x0 >= ~5e5`` over ``x0 <=
    1`` and the "bound" was ~5e5; it cannot exceed what the box allows."""
    a, b, lb, ub = _feedback_pair()
    lo, hi = _fbbt_eq_bounds(a, b, lb, ub)
    assert np.all(lo <= hi), (lo, hi)
    g = fn(np.zeros(2), np.ones(2), a, b, lb, ub)
    assert g is not None
    assert g <= float(ub.sum()) + 1e-9
