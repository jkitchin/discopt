"""Tests for the factorable reformulation pass (issue #130).

The pass rewrites two families of terms the relaxation pipeline cannot relax
natively into terms it can:

* sign-definite denominators ``N / D`` (``D`` bounded away from zero) are
  cleared by multiplying the constraint through by ``D``;
* mixed repeated-factor products such as ``x*x*y`` are lifted to bilinear form
  via a monomial aux variable ``w == x**2``.

Both rewrites are value-preserving, so the headline guard is that the raw
``nvs01`` instance — exp/quadratic objective, a trilinear equality, and a
division constraint — now certifies to its MINLPLib optimum with a *sound* dual
bound where before the constraint was silently dropped from the relaxation.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import discopt.modeling as dm
import pytest
from discopt._jax.factorable_reform import factorable_reformulate
from discopt._jax.term_classifier import classify_nonlinear_terms

_DATA = Path(__file__).parent / "data" / "minlplib"


def _aux_names(model):
    return [v.name for v in model._variables if v.name.startswith("_fr_aux")]


def test_noop_when_nothing_applies():
    """A model with only bilinear/monomial terms is returned unchanged."""
    m = dm.Model("plain")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x * y + x**2)
    m.subject_to(x + y <= 5)
    out = factorable_reformulate(m)
    assert out is m  # same object, no rewrite performed


def test_mixed_repeated_product_is_lifted_to_bilinear():
    """``x*x*y`` is unrepresentable natively; the pass lifts ``x**2`` to an aux
    so the product becomes a bilinear term the classifier accepts."""
    m = dm.Model("mixed")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.minimize(y)
    m.subject_to(x * x * y >= 8)

    # Before: the mixed product lands in general_nl (dropped by the relaxer).
    pre = classify_nonlinear_terms(m)
    assert pre.general_nl, "expected x*x*y to be unhandled before reformulation"

    out = factorable_reformulate(m)
    assert out is not m
    assert len(_aux_names(out)) == 1

    post = classify_nonlinear_terms(out)
    # The lifted product is now a genuine bilinear term, and the aux defining
    # equality contributes the x**2 monomial — nothing left in general_nl.
    assert post.bilinear, "lifted product should be classified as bilinear"
    assert (0, 2) in post.monomial  # w == x**2
    assert not post.general_nl


def _inequality_sense(model):
    """The (single) non-equality constraint sense, as stored after discopt's
    internal canonicalisation."""
    senses = [c.sense for c in model._constraints if c.sense != "=="]
    assert len(senses) == 1
    return senses[0]


def test_positive_denominator_cleared_sense_preserved():
    """``N / (x**2 + c)`` with a strictly positive denominator is cleared and the
    constraint sense is preserved (relative to discopt's stored form)."""
    m = dm.Model("posdiv")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.minimize(y)
    m.subject_to((y + 1) / (x**2 + 1) - 2 >= 0)
    orig_sense = _inequality_sense(m)

    out = factorable_reformulate(m)
    assert out is not m
    # The division by a variable expression is gone; only the aux-defining
    # equality and a polynomial body remain.
    bodies = [repr(c.body) for c in out._constraints]
    assert all("/ (" not in b and "/(" not in b for b in bodies)
    # Positive denominator → sense unchanged from the stored original.
    assert _inequality_sense(out) == orig_sense


def test_negative_denominator_flips_sense():
    """A strictly negative denominator flips the inequality sense on clearing."""
    m = dm.Model("negdiv")
    x = m.continuous("x", lb=1, ub=5)  # so -(x**2)-1 < 0 strictly
    y = m.continuous("y", lb=0, ub=5)
    m.minimize(y)
    # denominator -(x**2) - 1 is in [-26, -2] < 0 over the box
    m.subject_to((y + 1) / (-(x**2) - 1) <= 3)
    orig_sense = _inequality_sense(m)

    out = factorable_reformulate(m)
    assert out is not m
    flip = {"<=": ">=", ">=": "<="}
    assert _inequality_sense(out) == flip[orig_sense]


def test_grazing_denominator_not_cleared():
    """A denominator whose interval includes zero is NOT sign-definite and must
    be left untouched (no unsound multiply-through)."""
    m = dm.Model("graze")
    x = m.continuous("x", lb=-2, ub=2)  # x**2 - 1 spans [-1, 3], crosses zero
    y = m.continuous("y", lb=0, ub=5)
    m.minimize(y)
    m.subject_to((y + 1) / (x**2 - 1) >= 1)
    out = factorable_reformulate(m)
    # No clearable denominator and no mixed product → unchanged.
    assert out is m


def test_transcendental_product_not_lifted():
    """``sqrt(x) * y`` is not a pure polynomial product and must be left for the
    native composite/general handling, not lifted."""
    m = dm.Model("trans")
    x = m.continuous("x", lb=1, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.minimize(dm.sqrt(x) * y)
    m.subject_to(x + y >= 3)
    out = factorable_reformulate(m)
    assert out is m


@pytest.mark.correctness
def test_lifted_model_preserves_optimum():
    """The reformulated model is equivalent: solving it reaches the same
    optimum as the hand-checked value of the original."""
    m = dm.Model("equiv")
    x = m.integer("x", lb=1, ub=5)
    y = m.integer("y", lb=1, ub=5)
    m.minimize(y)
    # x*x*y >= 20 with x,y integer in [1,5]: minimum y is 1 (x=5 -> 25>=20).
    m.subject_to(x * x * y >= 20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(time_limit=60, gap_tolerance=1e-4)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(1.0, abs=1e-4)
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-4


@pytest.mark.correctness
@pytest.mark.slow
def test_nvs01_certifies_with_sound_bound():
    """Raw nvs01 (division + trilinear + composite objective) certifies to the
    MINLPLib optimum 12.4697 with a valid dual bound (bound <= objective)."""
    nl = _DATA / "nvs01.nl"
    if not nl.exists():
        pytest.skip("nvs01.nl not vendored")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = dm.from_nl(str(nl)).solve(
            time_limit=120, gap_tolerance=1e-4, max_nodes=2_000_000, subnlp_frequency=1
        )
    assert r.status == "optimal"
    assert r.objective == pytest.approx(12.4697, abs=1e-2)
    assert r.bound is not None
    assert r.bound <= r.objective + 1e-4  # sound dual bound
