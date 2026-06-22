"""Regression for issue #286: a nonconvex MINLP must never be declared globally
``unbounded`` because the *linear projection* of its relaxation is unbounded.

carton7 (=opt= 191.73) was reported ``unbounded`` after one node: it reached the
Rust pure-MILP simplex engine, whose ``extract_lp_data`` silently drops nonlinear
terms — including the very constraints that bound its infinite-upper-bound
continuous variables — so the dropped-down LP was genuinely unbounded. Two guards
now prevent this: the integer-bilinear reformulation is adopted (and routed to the
MILP engine) only when the *reformulated* model is a genuinely pure MILP, and
``_solve_milp_simplex`` itself defers on any model carrying nonlinear terms.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import time  # noqa: E402

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402

pytestmark = pytest.mark.unit


def _nonlinear_with_inf_bound():
    """min -x  s.t.  x**2 <= 100 (the only thing bounding x, ub=inf) and an
    integer-bilinear constraint k*j >= 6 that triggers the reformulation path.
    True optimum: x = 10 -> objective -10. The linear projection (dropping x**2)
    is unbounded below in -x."""
    m = dm.Model("i286")
    k = m.integer("k", lb=1, ub=5)
    j = m.integer("j", lb=1, ub=5)
    x = m.continuous("x", lb=0, ub=1e20)
    m.minimize(-x)
    m.subject_to(k * j >= 6)
    m.subject_to(x * x <= 100)
    return m


def test_no_false_unbounded():
    """End-to-end: never returns a global ``unbounded`` verdict on a problem with
    a finite optimum (the dropped nonlinear constraint bounds it)."""
    r = _nonlinear_with_inf_bound().solve(time_limit=20, gap_tolerance=1e-4)
    assert r.status != "unbounded"
    assert r.objective is None or r.objective <= 1e-6  # never reports a +inf-style win
    if r.objective is not None and r.status == "optimal":
        assert r.objective == pytest.approx(-10.0, abs=1e-2)


def test_milp_simplex_defers_on_nonlinear():
    """The Rust pure-MILP simplex engine must return None (defer) for any model
    carrying nonlinear terms — its extract_lp_data projection is lossy there."""
    from discopt.solver import _solve_milp_simplex

    m = _nonlinear_with_inf_bound()
    out = _solve_milp_simplex(m, 10.0, 1e-4, 1000, time.perf_counter())
    assert out is None


def test_pure_milp_still_uses_simplex():
    """Guard must be a no-op for a genuinely linear MILP (it still solves)."""
    from discopt.solver import _solve_milp_simplex

    m = dm.Model("milp")
    x = m.integer("x", lb=0, ub=10)
    y = m.integer("y", lb=0, ub=10)
    m.minimize(-x - 2 * y)
    m.subject_to(x + y <= 8)
    m.subject_to(3 * x + y <= 15)
    out = _solve_milp_simplex(m, 10.0, 1e-4, 100000, time.perf_counter())
    assert out is not None and out.status == "optimal"
    assert out.objective == pytest.approx(-16.0, abs=1e-6)
