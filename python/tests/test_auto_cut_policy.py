"""Tests for the structure-gated auto cut policy (cuts="auto").

The policy (see the Wave-2 A/B sweep) picks at most one QCQP cut family by
structure: RLT when the model has linear constraints, PSD on pure box-QP, and
neither above the size gate. It is purely a performance choice — every cut family
is sound — so it must always preserve the optimum.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling.core as dm
import numpy as np
import pytest
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt.solver import _AUTO_CUTS_MAX_VARS, _apply_auto_cut_policy


def _qcqp(n: int, seed: int, constrained: bool) -> dm.Model:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Q = (A + A.T) / 2
    m = dm.Model("q")
    x = m.continuous("x", shape=(n,), lb=0, ub=1)
    expr = None
    for i in range(n):
        for j in range(n):
            term = float(Q[i, j]) * x[i] * x[j]
            expr = term if expr is None else expr + term
    m.minimize(expr)
    if constrained:
        m.subject_to(dm.sum([x[i] for i in range(n)]) <= 0.6 * n)
        m.subject_to(x[0] + x[1] <= 1.2)
    return m


# ───────────────────────── policy unit tests (fast) ─────────────────────────


def test_policy_picks_psd_on_box_qp():
    m = _qcqp(5, 0, constrained=False)
    r = MccormickLPRelaxer(m)
    _apply_auto_cut_policy(m, r)
    assert r._psd_cuts is True and r._rlt_cuts is False


def test_policy_picks_rlt_on_constrained_qcqp():
    m = _qcqp(5, 0, constrained=True)
    r = MccormickLPRelaxer(m)
    _apply_auto_cut_policy(m, r)
    assert r._rlt_cuts is True and r._psd_cuts is False


def test_policy_declines_above_size_gate():
    # A cheap-to-build diagonal QCQP with > gate variables (sum of squares + a
    # linear constraint): quadratic + constrained, but oversize -> no cuts.
    n = _AUTO_CUTS_MAX_VARS + 2
    m = dm.Model("big")
    x = m.continuous("x", shape=(n,), lb=0, ub=1)
    m.minimize(dm.sum([x[i] * x[i] for i in range(n)]) - dm.sum([x[i] for i in range(n)]))
    m.subject_to(dm.sum([x[i] for i in range(n)]) <= 0.5 * n)
    r = MccormickLPRelaxer(m)
    _apply_auto_cut_policy(m, r)
    assert r._psd_cuts is False and r._rlt_cuts is False


# ───────────────────────── end-to-end (slow) ─────────────────────────


@pytest.mark.slow
def test_auto_matches_best_family_and_preserves_optimum():
    # Box-QP: auto should match PSD's node count.
    base_b = _qcqp(6, 0, constrained=False).solve(time_limit=120)
    auto_b = _qcqp(6, 0, constrained=False).solve(cuts="auto", time_limit=120)
    assert abs(float(base_b.objective) - float(auto_b.objective)) < 1e-3
    assert auto_b.node_count < base_b.node_count / 2

    # Constrained QCQP: auto should match RLT's node count.
    base_c = _qcqp(6, 3, constrained=True).solve(time_limit=120)
    auto_c = _qcqp(6, 3, constrained=True).solve(cuts="auto", time_limit=120)
    assert abs(float(base_c.objective) - float(auto_c.objective)) < 1e-3
    assert auto_c.node_count < base_c.node_count / 2
