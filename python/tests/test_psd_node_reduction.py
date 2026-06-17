"""End-to-end node-count win from per-node PSD cuts (Wave 2, W2e).

W2c applied PSD cuts only at the root global bound, which cannot reduce the B&B
node count (pruning is driven by the *per-node* relaxation bounds). W2e wires PSD
separation into ``MccormickLPRelaxer.solve_at_node``, so every node's bound is
tightened toward the SDP bound. On dense indefinite QCQP with a non-trivial
search tree this measurably reduces nodes — while always returning the same
global optimum (PSD cuts are valid, so they never remove a feasible point).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling.core as dm
import numpy as np
import pytest


def _dense_indefinite_qcqp(n: int, seed: int) -> dm.Model:
    """min x^T Q x over [0,1]^n with a dense symmetric (indefinite) Q."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Q = (A + A.T) / 2
    m = dm.Model(f"qcqp_n{n}_s{seed}")
    x = m.continuous("x", shape=(n,), lb=0, ub=1)
    expr = None
    for i in range(n):
        for j in range(n):
            term = float(Q[i, j]) * x[i] * x[j]
            expr = term if expr is None else expr + term
    m.minimize(expr)
    return m


def test_psd_preserves_optimum_and_never_adds_nodes():
    """Soundness + no-harm: same optimum, and PSD never increases the node count."""
    base = _dense_indefinite_qcqp(6, 8).solve(cuts="manual", time_limit=60)
    psd = _dense_indefinite_qcqp(6, 8).solve(psd_cuts=True, time_limit=60)
    assert base.status == "optimal" and psd.status == "optimal"
    assert abs(float(base.objective) - float(psd.objective)) < 1e-3
    assert psd.node_count <= base.node_count


@pytest.mark.slow
def test_psd_substantially_reduces_nodes_on_hard_instance():
    """n6_s0 has a non-trivial tree; per-node PSD cuts cut it down sharply."""
    base = _dense_indefinite_qcqp(6, 0).solve(cuts="manual", time_limit=120)
    psd = _dense_indefinite_qcqp(6, 0).solve(psd_cuts=True, time_limit=120)
    assert abs(float(base.objective) - float(psd.objective)) < 1e-3
    # Baseline explores a real tree; PSD cuts more than halve it.
    assert base.node_count > 20
    assert psd.node_count < base.node_count / 2
