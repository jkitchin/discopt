"""Relaxation-level tests for PSD cuts (bound improvement + validity).

These exercise ``separate_psd_cuts_on_relaxation`` / ``psd_strengthen_relaxation_bound``
against the real McCormick LP relaxation: PSD cuts must *tighten* the dual bound
(close the QCQP relaxation gap) while keeping it **valid** — the strengthened
bound never exceeds the true optimum, so it never excludes it.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling.core as dm
import numpy as np
import pytest
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.milp_relaxation import build_milp_relaxation
from discopt._jax.psd_cuts import (
    psd_strengthen_relaxation_bound,
    separate_psd_cuts_on_relaxation,
)


def _relax(model):
    r = MccormickLPRelaxer(model)
    milp, info = build_milp_relaxation(r._model, r._terms, r._disc)
    return milp, info


def test_psd_cut_closes_indefinite_qcqp_root_gap():
    """min x0^2 + x1^2 - 3 x0 x1 on [0,1]^2: optimum -1; McCormick bound -1.5."""
    m = dm.Model("indef")
    x = m.continuous("x", shape=(2,), lb=0, ub=1)
    m.minimize(x[0] * x[0] + x[1] * x[1] - 3 * x[0] * x[1])
    milp, info = _relax(m)
    z_before, z_after, n_cuts = psd_strengthen_relaxation_bound(milp, info, max_rounds=8)

    assert n_cuts >= 1
    assert z_after > z_before + 1e-6  # the cut tightens the bound
    assert z_after <= -1.0 + 1e-6  # still a VALID bound (<= true optimum)


def test_strengthened_bound_is_valid_on_several_instances():
    """Across indefinite QCQPs, the strengthened bound stays <= the true optimum."""
    cases = [
        # (Q upper-tri as dict {(i,j): q}, true optimum on [0,1]^2)
        ({(0, 0): 1.0, (1, 1): 1.0, (0, 1): -3.0}, -1.0),
        ({(0, 0): 0.0, (1, 1): 0.0, (0, 1): -2.0}, -2.0),  # min -2 x0 x1 -> -2
    ]
    for q, true_opt in cases:
        m = dm.Model("q")
        x = m.continuous("x", shape=(2,), lb=0, ub=1)
        expr = q[(0, 0)] * x[0] * x[0] + q[(1, 1)] * x[1] * x[1] + q[(0, 1)] * x[0] * x[1]
        m.minimize(expr)
        milp, info = _relax(m)
        z_before, z_after, n_cuts = psd_strengthen_relaxation_bound(milp, info, max_rounds=8)
        # Soundness: a valid bound never exceeds the true optimum.
        assert z_after <= true_opt + 1e-6
        # And it never loosens.
        assert z_after >= z_before - 1e-9


def test_separator_emits_no_cut_at_a_consistent_moment_point():
    """If X already equals x x^T at the point, no PSD cut is separable."""
    m = dm.Model("q")
    x = m.continuous("x", shape=(2,), lb=0, ub=1)
    m.minimize(x[0] * x[0] + x[1] * x[1] - 3 * x[0] * x[1])
    milp, info = _relax(m)
    n_total = len(milp._bounds)
    # Construct a point with X_ij = x_i x_j exactly -> moment matrix PSD (rank 1).
    x_full = np.zeros(n_total)
    xi, xj = 0.4, 0.7
    x_full[info["original"][0]] = xi
    x_full[info["original"][1]] = xj
    x_full[info["bilinear"][(0, 1)]] = xi * xj
    x_full[info["monomial"][(0, 2)]] = xi * xi
    x_full[info["monomial"][(1, 2)]] = xj * xj
    cuts = separate_psd_cuts_on_relaxation(info, x_full, n_total)
    assert cuts == []


def test_no_op_on_model_without_lifted_squares():
    """A purely bilinear model with no lifted squares yields no PSD cuts (sound no-op).

    The variables here are CONTINUOUS, so ``X_ii = x_i**2 != x_i`` and the moment
    diagonal genuinely needs a lifted square (absent) — the separator correctly finds
    nothing. This is the soundness boundary of the binary-diagonal shortcut below.
    """
    m = dm.Model("bilin")
    x = m.continuous("x", shape=(2,), lb=0, ub=1)
    m.minimize(x[0] * x[1])  # only the cross term; no x_i^2 lifted
    milp, info = _relax(m)
    z_before, z_after, n_cuts = psd_strengthen_relaxation_bound(milp, info, max_rounds=3)
    assert n_cuts == 0
    if z_before is not None:
        assert z_after == pytest.approx(z_before)


def test_binary_products_get_moment_cuts_via_diagonal_shortcut():
    """Pure products of DISTINCT binaries (QAP-class) carry no lifted square column,
    yet ``X_ii = x_i`` holds for a binary — so the moment separator must still fire.

    Before the binary-diagonal fix ``_diag_col`` returned ``None`` for every variable
    (no ``monomial``/``univariate_square`` lift), so ``_lifted_cliques`` was empty and
    PSD strengthening was a silent no-op on all such models (the qap ``-1e-9`` bound).

    Model: min x0x1 + x0x2 + x1x2 s.t. x0+x1+x2 >= 1.5, xi in {0,1}. Binary-feasible
    sums are {2,3} so the true minimum is 1; the McCormick relaxation reaches 0 at
    x=(.5,.5,.5), X_ij=0, whose 4x4 moment matrix has eigenvalue -1/4. The k=3 moment
    cut is violated there and tightens the bound, staying valid (<= 1).
    """
    from discopt._jax.model_utils import binary_flat_cols

    m = dm.Model("tri")
    x = [m.integer(f"x{i}", lb=0, ub=1) for i in range(3)]
    m.minimize(x[0] * x[1] + x[0] * x[2] + x[1] * x[2])
    m.subject_to(x[0] + x[1] + x[2] >= 1.5)
    bvars = binary_flat_cols(m)
    assert bvars == frozenset({0, 1, 2})

    # Old behavior (no binary_vars): still a no-op — gates the fix (fails after if the
    # shortcut leaked into the default path).
    milp0, info0 = _relax(m)
    zb0, za0, nc0 = psd_strengthen_relaxation_bound(milp0, info0, max_rounds=10, binary_vars=None)
    assert nc0 == 0

    # With binary diagonals: cuts fire and tighten the bound toward the true min 1.
    milp1, info1 = _relax(m)
    zb1, za1, nc1 = psd_strengthen_relaxation_bound(milp1, info1, max_rounds=10, binary_vars=bvars)
    assert nc1 >= 1
    assert za1 > zb1 + 1e-3  # genuine tightening (McCormick 0 -> ~0.36)
    assert za1 <= 1.0 + 1e-6  # SOUND: never above the true minimum

    # Soundness / feasible-point sampling (CLAUDE.md §5): every separated cut must
    # hold at every feasible integer point (X_ij = x_i x_j, X_ii = x_i there).
    x_at_pt = np.asarray(milp1._c, dtype=np.float64)  # length = #relaxation columns
    n_total = x_at_pt.shape[0]
    cuts = separate_psd_cuts_on_relaxation(info1, np.zeros(n_total), n_total, binary_vars=bvars)
    orig = info1["original"]
    bil = info1["bilinear"]
    feas = [p for p in [(1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)] if sum(p) >= 1.5]
    for pt in feas:
        z = np.zeros(n_total)
        for i in range(3):
            z[orig[i]] = pt[i]
        for (i, j), col in bil.items():
            z[col] = pt[i] * pt[j]  # X_ij = x_i x_j exactly at an integer point
        for c in cuts:
            # coeffs . z >= rhs must hold at every feasible integer point.
            assert float(c.coeffs @ z) >= c.rhs - 1e-7, (pt, "cut removes a feasible point")
