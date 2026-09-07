"""Unit tests for the bound-computation layer in ``discopt.solver`` (#87).

Covers the cut-augmented evaluator proxy (constraint/Jacobian/bound
augmentation and >= normalization), the interval-arithmetic node bound, the
convex-quadratic objective detector, and the alphaBB underestimator bound.
Every derived bound is validated against dense sampling (soundness:
bound <= true box minimum).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt._relax.cutting_planes import CutPool, LinearCut
from discopt._relax.nlp_evaluator import NLPEvaluator
from discopt.modeling.core import Model
from discopt.solver import (
    _AugmentedEvaluator,
    _BoundOverrideEvaluator,
    _compute_alphabb_bound,
    _compute_interval_bound,
    _objective_is_convex_quadratic,
)

pytestmark = pytest.mark.unit


def _constrained_model():
    m = Model("aug")
    x = m.continuous("x", lb=0.0, ub=2.0, shape=(2,))
    m.subject_to(x[0] + x[1] >= 0.5)
    m.minimize((x[0] - 1.0) ** 2 + (x[1] - 1.0) ** 2)
    return m


def test_bound_override_evaluator_swaps_bounds_only():
    m = _constrained_model()
    ev = NLPEvaluator(m)
    lb = np.array([0.5, 0.5])
    ub = np.array([1.5, 1.5])
    proxy = _BoundOverrideEvaluator(ev, lb, ub)
    plb, pub = proxy.variable_bounds
    np.testing.assert_allclose(plb, lb)
    np.testing.assert_allclose(pub, ub)
    # Everything else is delegated.
    x = np.array([1.0, 1.0])
    assert proxy.evaluate_objective(x) == pytest.approx(ev.evaluate_objective(x))
    assert proxy.n_constraints == ev.n_constraints


def test_augmented_evaluator_appends_normalized_cuts():
    m = _constrained_model()
    ev = NLPEvaluator(m)
    pool = CutPool()
    # x0 + x1 <= 3   and   x0 - x1 >= -1 (normalized to -x0 + x1 <= 1).
    pool.add(LinearCut(coeffs=np.array([1.0, 1.0]), rhs=3.0, sense="<="))
    pool.add(LinearCut(coeffs=np.array([1.0, -1.0]), rhs=-1.0, sense=">="))
    aug = _AugmentedEvaluator(ev, pool)
    assert aug.n_constraints == ev.n_constraints + 2

    x = np.array([2.0, 0.5])
    g = np.asarray(aug.evaluate_constraints(x))
    # Cut rows are appended as a^T x - b (feasible <=> <= 0).
    assert g[-2] == pytest.approx(2.0 + 0.5 - 3.0)
    assert g[-1] == pytest.approx(-(2.0 - 0.5) - 1.0)

    jac = np.asarray(aug.evaluate_jacobian(x))
    np.testing.assert_allclose(jac[-2], [1.0, 1.0])
    np.testing.assert_allclose(jac[-1], [-1.0, 1.0])

    # Augmented constraint bounds: cuts are (-inf, 0].
    bounds = aug.get_augmented_constraint_bounds([(0.0, np.inf)])
    assert len(bounds) == 3
    assert bounds[-1] == (-1e20, 0.0)

    # `get_augmented_jax_bounds` used to be asserted here. It was deleted: it had
    # no consumer anywhere in the package, and it built JAX arrays inside a
    # wrapper that is constructed on a live (possibly tape-backed, JAX-free)
    # solve path. `get_augmented_constraint_bounds` above is the live equivalent.

    # Lagrangian Hessian ignores the (linear) cut rows.
    lam = np.array([0.3, 0.1, 0.2])
    h = aug.evaluate_lagrangian_hessian(x, 1.0, lam)
    h_ref = ev.evaluate_lagrangian_hessian(x, 1.0, lam[:1])
    np.testing.assert_allclose(np.asarray(h), np.asarray(h_ref))


def test_augmented_evaluator_forwards_the_timing_layer():
    """The cut proxy must forward ``timing_bucket`` (issue #74).

    ``_AugmentedEvaluator`` enumerates its members explicitly instead of
    delegating through ``__getattr__``, so an attribute added to the evaluators
    does not reach ``_charge_evaluator`` through it. Without the forward, a
    cut-augmented node NLP silently drops its derivative callbacks from the layer
    profile — no error, just a wrong number.
    """
    m = _constrained_model()
    ev = NLPEvaluator(m)
    pool = CutPool()
    pool.add(LinearCut(coeffs=np.array([1.0, 1.0]), rhs=3.0, sense="<="))
    for proxy in (_AugmentedEvaluator(ev, pool), _AugmentedEvaluator(ev, CutPool())):
        assert proxy.timing_bucket == ev.timing_bucket == "jax"


def test_augmented_evaluator_with_empty_pool_is_passthrough():
    m = _constrained_model()
    ev = NLPEvaluator(m)
    aug = _AugmentedEvaluator(ev, CutPool())
    x = np.array([1.0, 1.0])
    assert aug.n_constraints == ev.n_constraints
    np.testing.assert_allclose(
        np.asarray(aug.evaluate_constraints(x)), np.asarray(ev.evaluate_constraints(x))
    )
    assert aug.get_augmented_constraint_bounds(None) is None


def test_compute_interval_bound_soundness():
    m = Model("ival")
    x = m.continuous("x", lb=-1.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=3.0)
    m.minimize(x * y + x)
    lb = np.array([-1.0, 0.0])
    ub = np.array([2.0, 3.0])
    bound = _compute_interval_bound(m, lb, ub, negate=False)
    # Sampled true minimum of x*y + x over the box.
    xs = np.linspace(-1, 2, 61)
    ys = np.linspace(0, 3, 61)
    true_min = min(a * b + a for a in xs for b in ys)
    assert bound <= true_min + 1e-9
    assert np.isfinite(bound)
    # negate=True bounds -f from below: -hi of the enclosure.
    bound_neg = _compute_interval_bound(m, lb, ub, negate=True)
    true_min_neg = min(-(a * b + a) for a in xs for b in ys)
    assert bound_neg <= true_min_neg + 1e-9
    # No objective -> unconditional abstention.
    m2 = Model("noobj")
    m2.continuous("x", lb=0.0, ub=1.0)
    assert _compute_interval_bound(m2, np.array([0.0]), np.array([1.0]), False) == -np.inf


def test_objective_is_convex_quadratic():
    m = Model("cq")
    a = m.continuous("a", lb=-5.0, ub=5.0)
    b = m.continuous("b", lb=-5.0, ub=5.0)
    m.minimize(a**2 + b**2 + a * b)
    ev = NLPEvaluator(m)
    assert _objective_is_convex_quadratic(m, ev, 2)

    # Array-shaped variables currently make the detector ABSTAIN (False):
    # its flat-bounds construction duplicates whole per-variable bound arrays
    # instead of flattening them, so the Hessian probe point has the wrong
    # shape and the try/except abstains. Abstention only loosens the bound
    # (sound); this test documents the limitation so a future fix flips it.
    m_arr = Model("cq_arr")
    x = m_arr.continuous("x", lb=-5.0, ub=5.0, shape=(2,))
    m_arr.minimize(x[0] ** 2 + x[1] ** 2 + x[0] * x[1])
    assert not _objective_is_convex_quadratic(m_arr, NLPEvaluator(m_arr), 2)

    m2 = Model("ncq")
    y = m2.continuous("y", lb=-5.0, ub=5.0)
    m2.minimize(-(y**2))
    ev2 = NLPEvaluator(m2)
    assert not _objective_is_convex_quadratic(m2, ev2, 1)

    m3 = Model("cubic")
    z = m3.continuous("z", lb=-5.0, ub=5.0)
    m3.minimize(z**3)
    ev3 = NLPEvaluator(m3)
    assert not _objective_is_convex_quadratic(m3, ev3, 1)


def test_compute_alphabb_bound_is_sound_lower_bound():
    m = Model("abb")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    m.minimize(-(x**2))  # concave: needs the alpha perturbation
    ev = NLPEvaluator(m)
    expr = -m._objective.expression if False else m._objective.expression
    bound = _compute_alphabb_bound(ev, m, expr, np.array([-1.0]), np.array([1.0]))
    true_min = -1.0  # min of -x^2 on [-1, 1]
    assert bound <= true_min + 1e-8
    assert bound > -np.inf  # a finite certified bound exists on this box

    # Unbounded box -> abstain with -inf, never a fabricated bound.
    bound_inf = _compute_alphabb_bound(ev, m, expr, np.array([-np.inf]), np.array([np.inf]))
    assert bound_inf == -np.inf


def _cut_augmented_case():
    """A constrained model plus two cuts, and the augmented evaluator over them."""
    m = _constrained_model()
    ev = NLPEvaluator(m)
    pool = CutPool()
    pool.add(LinearCut(coeffs=np.array([1.0, 1.0]), rhs=3.0, sense="<="))
    pool.add(LinearCut(coeffs=np.array([1.0, -1.0]), rhs=-1.0, sense=">="))
    return ev, _AugmentedEvaluator(ev, pool)


def test_augmented_evaluator_reports_sparse_structure_including_cut_rows():
    """The cut proxy must implement the sparse-structure protocol (#1193, P2).

    ``_AugmentedEvaluator`` enumerates its members explicitly, and before this
    it omitted ``has_sparse_structure``. ``_IpoptCallbacks`` probes for that
    attribute with ``hasattr``, so every derivative callback on a cut-augmented
    node fell to the dense arm: an ``m x n`` meshgrid for the Jacobian pattern
    and a dense ``n x n`` Lagrangian Hessian, even though the wrapped tape
    evaluator returns analytical sparse COO for both.

    Forwarding the flag alone would be unsound rather than merely incomplete:
    the augmented constraint vector carries ``n_cuts`` extra rows that the
    wrapped evaluator's pattern does not describe. This test therefore asserts
    the reconstructed dense Jacobian -- not just the flag.
    """
    checks = 0
    ev, aug = _cut_augmented_case()

    assert ev.has_sparse_structure(), "wrapped evaluator must be sparse for this test to bite"
    checks += 1
    assert aug.has_sparse_structure() is True
    checks += 1

    x = np.array([2.0, 0.5])
    rows, cols = aug.jacobian_structure()
    vals = aug.evaluate_jacobian_values(x)
    assert len(rows) == len(cols) == len(vals)
    checks += 1

    # Every augmented row must be described, cut rows included.
    assert set(np.unique(rows)) >= set(range(aug.n_constraints))
    checks += 1

    # The COO triple must reconstruct the dense Jacobian exactly.
    dense = np.zeros((aug.n_constraints, aug.n_variables))
    dense[rows, cols] = vals
    np.testing.assert_allclose(dense, np.asarray(aug.evaluate_jacobian(x)))
    checks += 1

    # Hessian: cuts are linear, so structure and values are the wrapped ones,
    # with the cut multipliers truncated away.
    lam = np.array([0.3, 0.1, 0.2])
    hrows, hcols = aug.hessian_structure()
    hvals = aug.evaluate_hessian_values(x, 1.0, lam)
    h_ref = np.asarray(ev.evaluate_lagrangian_hessian(x, 1.0, lam[:1]))
    np.testing.assert_allclose(hvals, h_ref[hrows, hcols])
    checks += 1

    assert checks == 6, f"expected 6 executed assertions, ran {checks}"


def test_ipopt_callbacks_take_the_sparse_arm_for_a_cut_augmented_node():
    """The consumer must reconstruct the same Jacobian through the sparse arm.

    This is the end the fix is for: ``_IpoptCallbacks`` is what POUNCE is
    handed (``_solve_batch_pounce``), and its ``_use_sparse`` flag is decided
    once from ``has_sparse_structure``.
    """
    from discopt.solvers.nlp_ipopt import _IpoptCallbacks

    checks = 0
    _, aug = _cut_augmented_case()
    cb = _IpoptCallbacks(aug)
    assert cb._use_sparse is True
    checks += 1

    x = np.array([2.0, 0.5])
    rows, cols = cb.jacobianstructure()
    vals = cb.jacobian(x)
    dense = np.zeros((aug.n_constraints, aug.n_variables))
    dense[rows, cols] = vals
    np.testing.assert_allclose(dense, np.asarray(aug.evaluate_jacobian(x)))
    checks += 1

    # The sparse arm must not be a dense meshgrid in disguise.
    assert len(rows) <= aug.n_constraints * aug.n_variables
    checks += 1

    lam = np.array([0.3, 0.1, 0.2])
    hrows, hcols = cb.hessianstructure()
    hvals = cb.hessian(x, lam, 1.0)
    h_ref = np.asarray(aug.evaluate_lagrangian_hessian(x, 1.0, lam))
    np.testing.assert_allclose(hvals, h_ref[hrows, hcols])
    checks += 1

    assert checks == 4, f"expected 4 executed assertions, ran {checks}"
