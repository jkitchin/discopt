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
    """A purely bilinear model with no lifted squares yields no PSD cuts (sound no-op)."""
    m = dm.Model("bilin")
    x = m.continuous("x", shape=(2,), lb=0, ub=1)
    m.minimize(x[0] * x[1])  # only the cross term; no x_i^2 lifted
    milp, info = _relax(m)
    z_before, z_after, n_cuts = psd_strengthen_relaxation_bound(milp, info, max_rounds=3)
    assert n_cuts == 0
    if z_before is not None:
        assert z_after == pytest.approx(z_before)
