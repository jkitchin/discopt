"""Regression test for the issue #120 follow-up: monomial / fractional-power
nonconvex models must get a *valid* LP spatial dual bound, not the unsound
"nlp"/alphaBB fallback.

The spatial LP relaxer (:class:`MccormickLPRelaxer`) was engaged only when the
model had a bilinear/trilinear/multilinear product (its ``has_bilinear`` gate).
Nonconvex models whose only nonlinearity is a univariate power — ``x**n``
(monomial) or ``x**p`` (fractional power) — were therefore routed to the
McCormick ``"nlp"`` objective bound, which is not a valid dual bound for
nonconvex models, and fell back to alphaBB with *no* certifying bound.

:func:`build_milp_relaxation` emits valid outer-approximation cuts for those
term types too, so the LP optimum is a rigorous lower bound. The gate now keys
on :attr:`MccormickLPRelaxer.has_relaxable_nonlinearity`, so these models
certify optimality with a sound bound.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import discopt.modeling as dm
import pytest

_DATA = Path(__file__).parent / "data" / "minlplib"

# (instance, global optimum) for monomial/fractional-power-only models that
# previously produced no dual bound and now certify. Optima are the proven
# values (gap closed: bound == objective) and agree with MINLPLib references.
_CERTIFY_CASES = [
    ("prob06", 1.177124),
    ("st_e13", 2.000000),
    ("ex1221", 7.667180),
    ("ex1222", 1.076543),
    ("ex1225", 31.000000),
    ("st_e15", 7.667180),
]


@pytest.mark.correctness
def test_relaxer_gate_includes_monomial_and_fractional_power():
    """The LP relaxer must engage for purely univariate-power nonconvexity."""
    from discopt._jax.mccormick_lp import MccormickLPRelaxer

    # st_e13: a monomial-only nonconvex model (no product term at all).
    m = dm.from_nl(str(_DATA / "st_e13.nl"))
    relaxer = MccormickLPRelaxer(m)
    assert not relaxer.has_bilinear, "st_e13 has no product term (precondition)"
    assert relaxer.has_relaxable_nonlinearity, (
        "monomial/fractional-power models must engage the LP relaxer"
    )


@pytest.mark.correctness
@pytest.mark.parametrize("instance, optimum", _CERTIFY_CASES)
def test_monomial_model_certifies_with_valid_bound(instance, optimum):
    """Each model reaches its optimum and certifies with a sound dual bound."""
    nl = _DATA / f"{instance}.nl"
    assert nl.exists(), f"missing {nl}"
    r = dm.from_nl(str(nl)).solve(time_limit=30, gap_tolerance=1e-4)

    assert r.objective is not None
    assert abs(r.objective - optimum) <= 1e-3, f"[{instance}] obj={r.objective} != {optimum}"
    # A valid dual bound must exist and never exceed the optimum (soundness).
    assert r.bound is not None, f"[{instance}] no dual bound produced (issue #120 follow-up)"
    assert r.bound <= optimum + 1e-3, f"[{instance}] invalid dual bound {r.bound} > {optimum}"
    assert r.gap_certified, f"[{instance}] expected certified optimality"
