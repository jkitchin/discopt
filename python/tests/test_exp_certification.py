"""Regression locks for the convex ``exp`` envelope (issue #140, bucket 3).

The MILP relaxer lifts a convex ``exp(g)`` term to an auxiliary column with a
tangent underestimator and a secant overestimator, so a model whose only
nonlinearity is ``exp`` of an affine argument certifies with a *sound* dual
bound (``bound <= objective``). This already works on ``main`` for both
standalone ``exp`` and ``exp(g) * y`` products (the univariate aux participates
in the bilinear McCormick decomposition); these tests pin that behavior so it
cannot silently regress.

Scope note (issue #140): ``ex1222`` is already covered by
``test_monomial_lp_bound.py``; ``st_e04`` (``exp`` of a composite
division argument) certifies but had no guard, so it is added here. ``st_e36``
is *not* an ``exp`` problem — its blocker is a high-degree polynomial product
constraint (tracked under bucket #2). ``exp`` of a *nonlinear, multivariable*
argument (e.g. ``exp(x*z)``) still drops to a feasibility objective and is a
documented follow-up, not covered here.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import math
from pathlib import Path

import discopt.modeling as dm
import pytest

_DATA = Path(__file__).parent / "data" / "minlplib"


@pytest.mark.correctness
def test_exp_standalone_certifies_sound_bound():
    """``min exp(x)`` over a box certifies at the convex optimum with a sound bound."""
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize(dm.exp(x))
    r = m.solve(time_limit=15, gap_tolerance=1e-5)

    assert r.objective is not None and r.bound is not None
    assert math.isclose(r.objective, 1.0, abs_tol=1e-4)  # exp(0) = 1
    assert r.bound <= r.objective + 1e-6, "dual bound must not exceed the optimum"
    assert r.bound <= 1.0 + 1e-4, "dual bound must not exceed the true optimum 1.0"


@pytest.mark.correctness
def test_exp_affine_times_var_product_certifies():
    """``exp(g) * y`` with affine ``g``: the univariate aux feeds the bilinear envelope."""
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=1.0, ub=2.0)
    # min exp(2x+1)*y -> x=0, y=1 -> e^1 = 2.718281828...
    m.minimize(dm.exp(2 * x + 1) * y)
    r = m.solve(time_limit=15, gap_tolerance=1e-5)

    assert r.objective is not None and r.bound is not None
    assert math.isclose(r.objective, math.e, rel_tol=1e-4)
    assert r.bound <= r.objective + 1e-6, "dual bound must not exceed the optimum"
    assert r.bound <= math.e + 1e-4, "dual bound must not exceed the true optimum e"


@pytest.mark.correctness
def test_exp_in_constraint_certifies_sound_bound():
    """``exp`` of an affine argument inside an inequality certifies soundly."""
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=2.0)
    # min -x  s.t.  exp(x) <= 5  ->  x* = ln 5 = 1.6094..., objective = -ln 5
    m.subject_to(dm.exp(x) <= 5.0)
    m.minimize(-x)
    r = m.solve(time_limit=15, gap_tolerance=1e-5)

    assert r.objective is not None and r.bound is not None
    assert math.isclose(r.objective, -math.log(5.0), abs_tol=1e-3)
    # Minimization: a valid dual bound never exceeds the optimum.
    assert r.bound <= r.objective + 1e-4, "dual bound must not exceed the optimum"


@pytest.mark.correctness
def test_st_e04_certifies_with_sound_bound():
    """st_e04 (``exp`` of a composite division argument) certifies soundly.

    Previously unguarded. discopt's dual bound equals the global optimum
    (~5194.866) and the within-gap incumbent sits just above it; the gap closes
    at the 1e-4 tolerance. The invariant we lock is soundness + certification,
    not the exact incumbent value.
    """
    nl = _DATA / "st_e04.nl"
    assert nl.exists(), f"missing {nl}"
    r = dm.from_nl(str(nl)).solve(time_limit=30, gap_tolerance=1e-4)

    assert r.status == "optimal"
    assert r.objective is not None and r.bound is not None
    # Soundness: the dual bound must not exceed the known global optimum.
    assert r.bound <= 5194.866 + 1e-2, f"unsound dual bound {r.bound} > global optimum"
    # The incumbent sits at or above the dual bound.
    assert r.objective >= r.bound - 1e-6
    assert r.gap_certified, "st_e04 must certify optimality at gap_tolerance=1e-4"
