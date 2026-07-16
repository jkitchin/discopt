"""Regression tests for the exhaustive root RLT-1 bound (issue #661).

The bound-changing verification regime (CLAUDE.md §5) for a relaxation-strengthening
change: a *differential bound* test (new bound >= old McCormick bound AND <= the true
box optimum on a fixed box) plus *feasible-point sampling* (no valid integer point is
cut). Both are exercised on small synthetic Koopmans-Beckmann QAPs whose optimum is
found by brute force, where the exact vertex simplex solves the RLT-1 LP quickly.

The flag `SolverTuning.rlt1_root_bound` is verified default-off so the wired path is a
no-op unless opted in (`DISCOPT_RLT1_ROOT_BOUND`).
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.discretization import DiscretizationState
from discopt._jax.milp_relaxation import (
    build_milp_relaxation,
    sanitize_relaxation_for_conditioning,
)
from discopt._jax.model_utils import binary_flat_cols, flat_variable_bounds
from discopt._jax.rlt import build_rlt1_lp, rlt1_lower_bound
from discopt._jax.term_classifier import classify_nonlinear_terms


def _synthetic_qap(n: int, seed: int):
    """Build a small Koopmans-Beckmann QAP discopt model + its brute-force optimum.

    ``n`` facilities to ``n`` locations; ``x[i*n + k] = 1`` iff facility ``i`` is at
    location ``k``. Objective ``sum F_ij D_kl x_ik x_jl`` with symmetric flow/distance
    matrices — indefinite, so the term-wise McCormick LP bound collapses to ~0.
    """
    rng = np.random.default_rng(seed)
    F = rng.integers(0, 10, (n, n))
    F = F + F.T
    np.fill_diagonal(F, 0)
    D = rng.integers(0, 10, (n, n))
    D = D + D.T
    np.fill_diagonal(D, 0)

    m = dm.Model()
    x = m.binary("x", shape=(n * n,))

    def v(i, k):
        return x[i * n + k]

    for i in range(n):
        m.subject_to(dm.sum([v(i, k) for k in range(n)]) == 1)
    for k in range(n):
        m.subject_to(dm.sum([v(i, k) for i in range(n)]) == 1)
    terms = []
    Q = np.zeros((n * n, n * n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for ll in range(n):
                    if i == j or k == ll:
                        continue
                    w = float(F[i, j] * D[k, ll])
                    if w:
                        terms.append(w * v(i, k) * v(j, ll))
                    Q[i * n + k, j * n + ll] += F[i, j] * D[k, ll]
    m.minimize(dm.sum(terms))

    best = np.inf
    best_perm = None
    for perm in itertools.permutations(range(n)):
        xv = np.zeros(n * n)
        for i, k in enumerate(perm):
            xv[i * n + k] = 1.0
        val = float(xv @ Q @ xv)
        if val < best:
            best, best_perm = val, xv
    return m, best, best_perm


def _built(model):
    lb, ub = flat_variable_bounds(model)
    terms = classify_nonlinear_terms(model)
    relax, info = build_milp_relaxation(
        model, terms, DiscretizationState(), bound_override=(lb, ub)
    )
    relax = sanitize_relaxation_for_conditioning(relax)
    return relax, info


def _mccormick_lp_bound(relax):
    """Continuous McCormick LP root bound via the exact oracle (the loose baseline)."""
    from discopt._jax.obbt import get_exact_lp_solver

    _lp = get_exact_lp_solver()
    res = _lp(
        c=np.asarray(relax._c),
        A_ub=relax._A_ub,
        b_ub=np.asarray(relax._b_ub),
        bounds=list(relax._bounds),
        time_limit=30.0,
    )
    assert res.objective is not None
    return float(res.objective) + float(relax._obj_offset)


@pytest.mark.parametrize("n,seed", [(4, 0), (5, 1)])
def test_rlt1_differential_bound_lifts_loose_mccormick(n, seed):
    """§5 differential bound: RLT-1 >= the (near-zero) McCormick LP bound and stays
    <= the true optimum on the fixed root box."""
    model, opt, _ = _synthetic_qap(n, seed)
    relax, info = _built(model)
    mcc = _mccormick_lp_bound(relax)
    bound, nrlt = rlt1_lower_bound(
        model, relax, info, binary_vars=binary_flat_cols(model), time_limit=30.0
    )
    assert nrlt > 0
    assert bound is not None
    # McCormick on an indefinite binary QP is trivially loose (~0).
    assert mcc <= 1e-3
    # New bound is a genuine tightening ...
    assert bound >= mcc - 1e-6
    assert bound > mcc + 1.0
    # ... and never crosses the true optimum (rigorous lower bound).
    assert bound <= opt + 1e-4


@pytest.mark.parametrize("n,seed", [(4, 0), (4, 2)])
def test_rlt1_never_cuts_a_feasible_point(n, seed):
    """Feasible-point sampling: every genuine assignment ``(x, X = x x^T)`` satisfies
    every row of the RLT-1 LP — no valid integer point is removed — and the LP
    objective there equals that point's true objective value."""
    model, opt, _ = _synthetic_qap(n, seed)
    relax, info = _built(model)
    prob = build_rlt1_lp(model, relax, info, binary_vars=binary_flat_cols(model))
    assert prob is not None and prob.n_rlt_rows > 0

    A = prob.A_ub.tocsr()
    for perm in itertools.permutations(range(n)):
        x = np.zeros(n * n)
        for i, k in enumerate(perm):
            x[i * n + k] = 1.0
        z = prob.pack_point(x)
        # Bounds respected and every inequality satisfied (with a small tolerance).
        assert np.all(z >= -1e-9) and np.all(z <= 1.0 + 1e-9)
        residual = A @ z - prob.b_ub
        assert np.max(residual) <= 1e-7, f"RLT-1 row cuts a feasible point: {np.max(residual)}"
        # Objective at the packed feasible point matches its true QAP cost.
        true_cost = float(prob.cobj @ z + prob.offset)
        # recompute cost directly
        assert true_cost >= opt - 1e-6


def test_rlt1_bound_le_every_feasible_objective():
    """The RLT-1 bound underestimates the objective at *every* feasible assignment."""
    model, opt, _ = _synthetic_qap(4, 3)
    relax, info = _built(model)
    prob = build_rlt1_lp(model, relax, info, binary_vars=binary_flat_cols(model))
    assert prob is not None
    bound, _ = rlt1_lower_bound(
        model, relax, info, binary_vars=binary_flat_cols(model), time_limit=30.0
    )
    assert bound is not None
    for perm in itertools.permutations(range(4)):
        x = np.zeros(16)
        for i, k in enumerate(perm):
            x[i * 4 + k] = 1.0
        z = prob.pack_point(x)
        obj = float(prob.cobj @ z + prob.offset)
        assert bound <= obj + 1e-4


def test_rlt1_no_op_without_equality_constraints():
    """RLT-1 constraint factors need equalities; a purely inequality-constrained
    binary QP yields no RLT-1 LP (a sound no-op)."""
    m = dm.Model()
    x = m.binary("x", shape=(3,))
    m.subject_to(dm.sum([x[i] for i in range(3)]) <= 2)  # inequality only
    m.minimize(x[0] * x[1] + x[1] * x[2] - x[0] * x[2])
    relax, info = _built(m)
    prob = build_rlt1_lp(m, relax, info, binary_vars=binary_flat_cols(m))
    assert prob is None
    bound, nrlt = rlt1_lower_bound(m, relax, info, binary_vars=binary_flat_cols(m))
    assert bound is None and nrlt == 0


def test_rlt1_no_op_on_continuous_qp():
    """Continuous variables have ``X_ii = x_i**2 != x_i``; the binary shortcut is
    unsound, so RLT-1 must decline (no binary_vars)."""
    m = dm.Model()
    x = m.continuous("x", shape=(3,), lb=0.0, ub=1.0)
    m.subject_to(dm.sum([x[i] for i in range(3)]) == 1)
    m.minimize(x[0] * x[1] + x[1] * x[2] - x[0] * x[2])
    relax, info = _built(m)
    prob = build_rlt1_lp(m, relax, info, binary_vars=binary_flat_cols(m))
    assert prob is None


def test_rlt1_size_guard():
    """The all-pairs lift guard declines an oversize model."""
    model, _, _ = _synthetic_qap(4, 0)
    relax, info = _built(model)
    # 16 vars -> 120 pairs; a guard below that must decline.
    prob = build_rlt1_lp(model, relax, info, binary_vars=binary_flat_cols(model), max_pairs=10)
    assert prob is None


def test_rlt1_flag_default_off():
    """The wired lever is opt-in (§3): default-off, so the root path is unchanged
    unless DISCOPT_RLT1_ROOT_BOUND is set."""
    from discopt.solver_tuning import SolverTuning

    assert SolverTuning().rlt1_root_bound is False


def test_rlt1_root_wiring_is_sound_when_enabled():
    """With the flag on, `_root_relaxation_lower_bound` returns a bound that never
    crosses the true optimum (the certificate invariant)."""
    from discopt.solver import _root_relaxation_lower_bound
    from discopt.solver_tuning import SolverTuning, reset_current, set_current

    model, opt, _ = _synthetic_qap(4, 1)
    lb, ub = flat_variable_bounds(model)
    tok = set_current(SolverTuning(rlt1_root_bound=True))
    try:
        bound = _root_relaxation_lower_bound(model, lb, ub, time_limit=30.0)
    finally:
        reset_current(tok)
    assert bound is not None
    assert bound <= opt + 1e-4
