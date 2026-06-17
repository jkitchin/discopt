"""Linear-fractional relaxation for a ratio of products (issue #185).

discopt previously dumped a quotient of products ``(c·Πx)/(Πy)`` into
``general_nl`` and dropped the enclosing constraint from the McCormick
relaxation, leaving a trivial dual bound (the gear4 "wall #2"). This module
locks the new behavior:

* the term classifier recognizes the ratio (including gear4's *negative*
  numerator coefficient) and records the embedded products + partition
  candidates;
* :class:`MccormickLPRelaxer` lifts the quotient via the bilinear identity
  ``r·q = m`` (``r`` the pure ratio, ``m``/``q`` the lifted numerator/denominator
  products) and produces a **sound** lower bound — verified by a property test
  over random positive-denominator ratios — that is *exact* when the variables
  are pinned.

The large numerator constant (gear4's ``1e6``) is applied by the linearizer as a
substitution coefficient, never injected into the envelope rows, so the
relaxation LP stays well-conditioned (this is what distinguishes the sound lift
from the naive ``y·D = N`` route that produced a false-optimal on gear4 — see
``test_quotient_certification`` scope note and issue #145).
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.term_classifier import (
    classify_nonlinear_terms,
    extract_ratio_of_products,
)
from discopt.modeling.core import from_nl

_DATA = Path(__file__).parent / "data" / "minlplib"


def _ratio_model(coeff: float) -> dm.Model:
    """``minimize coeff·(x0·x1)/(x2·x3)`` over a positive-denominator box."""
    m = dm.Model("ratio")
    x0 = m.continuous("x0", lb=2.0, ub=4.0)
    x1 = m.continuous("x1", lb=2.0, ub=4.0)
    x2 = m.continuous("x2", lb=1.0, ub=2.0)
    x3 = m.continuous("x3", lb=1.0, ub=2.0)
    m.minimize(coeff * (x0 * x1) / (x2 * x3))
    return m


# ---------------------------------------------------------------------------
# Classifier recognition
# ---------------------------------------------------------------------------


def test_classifier_recognizes_ratio_of_products():
    """The classifier records the ratio and its embedded products / candidates."""
    m = _ratio_model(1.0)
    terms = classify_nonlinear_terms(m)
    assert terms.ratio_of_products == [((0, 1), (2, 3))]
    # Numerator and denominator products are recorded as bilinear terms (so they
    # get McCormick envelopes) and their variables become partition candidates.
    assert (0, 1) in terms.bilinear
    assert (2, 3) in terms.bilinear
    assert set(terms.partition_candidates) == {0, 1, 2, 3}


def test_classifier_recognizes_negative_coefficient_ratio():
    """gear4's wall #2 shape: a *negative* numerator coefficient is recognized."""
    m = dm.Model("gear4_shape")
    i1 = m.integer("i1", lb=12, ub=60)
    i2 = m.integer("i2", lb=12, ub=60)
    i3 = m.integer("i3", lb=12, ub=60)
    i4 = m.integer("i4", lb=12, ub=60)
    x6 = m.continuous("x6", lb=0.0, ub=1e6)
    x7 = m.continuous("x7", lb=0.0, ub=1e6)
    m.minimize(x6 + x7)
    m.subject_to(-(1e6 * i1 * i2) / (i3 * i4) - x6 + x7 == -144279.32477276)

    terms = classify_nonlinear_terms(m)
    assert terms.ratio_of_products == [((0, 1), (2, 3))]


def test_extract_ratio_of_products_rejects_non_ratio():
    """The matcher declines non-ratio / constant-denominator divisions."""
    m = dm.Model("nr")
    x0 = m.continuous("x0", lb=1.0, ub=2.0)
    x1 = m.continuous("x1", lb=1.0, ub=2.0)
    # x / constant is plain scaling, not a ratio of products.
    assert extract_ratio_of_products((x0 * x1) / 3.0, m) is None
    # numerator is additive, not a product.
    assert extract_ratio_of_products((x0 + x1) / (x0 * x1), m) is None


# ---------------------------------------------------------------------------
# Relaxation soundness (the issue's must-gate property test)
# ---------------------------------------------------------------------------


def _sample_true_min(coeff, lb, ub, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    pts = lb + rng.random((n, 4)) * (ub - lb)
    vals = coeff * (pts[:, 0] * pts[:, 1]) / (pts[:, 2] * pts[:, 3])
    return float(vals.min())


@pytest.mark.parametrize("coeff", [1.0, -2.0, -1e6, 5.0])
def test_relaxation_is_a_valid_lower_bound(coeff):
    """The lifted relaxation LB never exceeds the true minimum over the box.

    Lower-bound soundness: a valid relaxation underestimates the objective at
    *every* feasible point, so its LP optimum is ≤ the true minimum sampled by
    Monte Carlo (allowing a small tolerance for the finite sample)."""
    m = _ratio_model(coeff)
    lb = np.array([2.0, 2.0, 1.0, 1.0])
    ub = np.array([4.0, 4.0, 2.0, 2.0])
    relaxer = MccormickLPRelaxer(m)
    res = relaxer.solve_at_node(lb, ub)
    assert res.status == "optimal" and res.lower_bound is not None
    true_min = _sample_true_min(coeff, lb, ub)
    # Relaxation must lie at or below the true minimum (sound lower bound).
    rel_scale = max(1.0, abs(true_min))
    assert res.lower_bound <= true_min + 1e-4 * rel_scale, (
        f"unsound: LB={res.lower_bound} > true_min={true_min} (coeff={coeff})"
    )


@pytest.mark.parametrize("coeff", [1.0, -2.0, -1e6])
def test_relaxation_exact_when_pinned(coeff):
    """Pinning all variables makes the lifted relaxation reproduce the exact ratio."""
    m = _ratio_model(coeff)
    pt = np.array([3.0, 4.0, 1.0, 2.0])  # x0=3,x1=4,x2=1,x3=2
    relaxer = MccormickLPRelaxer(m)
    res = relaxer.solve_at_node(pt.copy(), pt.copy())
    assert res.status == "optimal" and res.lower_bound is not None
    exact = coeff * (pt[0] * pt[1]) / (pt[2] * pt[3])
    assert res.lower_bound == pytest.approx(exact, rel=1e-6, abs=1e-6)


def test_relaxation_declines_sign_indefinite_denominator():
    """A denominator straddling zero is *not* lifted (the must-gate guard).

    The relaxation must stay sound: it declines the ratio lift and the bound is
    never above the true minimum (here the LP simply omits the unbounded term).
    """
    m = dm.Model("signflip")
    x0 = m.continuous("x0", lb=1.0, ub=2.0)
    x1 = m.continuous("x1", lb=1.0, ub=2.0)
    x2 = m.continuous("x2", lb=-1.0, ub=1.0)  # straddles 0
    x3 = m.continuous("x3", lb=1.0, ub=2.0)
    m.minimize((x0 * x1) / (x2 * x3))
    relaxer = MccormickLPRelaxer(m)
    # Build must not raise and any produced bound must be sound (≤ true values).
    res = relaxer.solve_at_node(np.array([1.0, 1.0, -1.0, 1.0]), np.array([2.0, 2.0, 1.0, 2.0]))
    if res.status == "optimal" and res.lower_bound is not None:
        # true ratio is unbounded below near x2→0^-, so any finite LB is sound;
        # the key property is that the lift did not fabricate a finite *upper*
        # cut that excludes feasible points. A simple sound check: LB ≤ value at
        # an interior feasible point.
        val = (1.5 * 1.5) / (0.5 * 1.5)
        assert res.lower_bound <= val + 1e-6


# ---------------------------------------------------------------------------
# gear4: the motivating instance
# ---------------------------------------------------------------------------


def test_gear4_relaxation_retains_constraint_and_is_exact_at_optimum():
    """gear4's quotient constraint is retained by the relaxation and the pinned
    node reproduces the true global optimum 1.6434 (issue #185 acceptance #1)."""
    from discopt._jax.model_utils import flat_variable_bounds

    m = from_nl(str(_DATA / "gear4.nl"))
    relaxer = MccormickLPRelaxer(m)
    # Pin the integers to the known optimum (16,19,43,49); leave the slacks free.
    lb, ub = (np.asarray(a, dtype=float).copy() for a in flat_variable_bounds(m))
    n = lb.size
    # Variables order: i1,i2,i3,i4, then continuous slacks.
    opt = {0: 16.0, 1: 19.0, 2: 43.0, 3: 49.0}
    for idx, val in opt.items():
        lb[idx] = ub[idx] = val
    res = relaxer.solve_at_node(lb, ub)
    assert res.status == "optimal" and res.lower_bound is not None, (
        "gear4 constraint e1 was dropped from the relaxation (wall #2 not fixed)"
    )
    assert res.lower_bound == pytest.approx(1.6434284730639774, abs=1e-3)
    assert n >= 6


def test_gear4_solve_stays_sound():
    """End-to-end gear4 must never be certified infeasible (soundness must-gate)."""
    r = from_nl(str(_DATA / "gear4.nl")).solve(time_limit=20, max_nodes=500)
    assert not (r.status == "infeasible" and r.gap_certified), (
        f"gear4 falsely certified: status={r.status} cert={r.gap_certified}"
    )
    assert r.status != "infeasible", f"gear4 is feasible but status={r.status}"
    if r.bound is not None:
        assert r.bound <= 1.6434284730639774 + 1e-3, f"unsound dual bound {r.bound}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
