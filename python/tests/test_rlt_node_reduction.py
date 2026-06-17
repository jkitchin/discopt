"""End-to-end node-count win from per-node targeted RLT cuts.

RLT (constraint-factor x bound-factor) cuts apply to QCQP with *linear
constraints* (pure box-QP has none, so RLT is a no-op there). On constrained
indefinite QCQP they tighten the per-node bound and reduce the search tree, while
always returning the same global optimum (every cut is valid).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling.core as dm
import numpy as np
import pytest


def _constrained_qcqp(n: int, seed: int) -> dm.Model:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Q = (A + A.T) / 2
    m = dm.Model(f"cqcqp_n{n}_s{seed}")
    x = m.continuous("x", shape=(n,), lb=0, ub=1)
    expr = None
    for i in range(n):
        for j in range(n):
            term = float(Q[i, j]) * x[i] * x[j]
            expr = term if expr is None else expr + term
    m.minimize(expr)
    m.subject_to(dm.sum([x[i] for i in range(n)]) <= 0.6 * n, name="budget")
    m.subject_to(x[0] + x[1] <= 1.2, name="c2")
    return m


def test_rlt_preserves_optimum_and_never_adds_nodes():
    base = _constrained_qcqp(6, 0).solve(rlt_cuts=False, time_limit=60)
    rlt = _constrained_qcqp(6, 0).solve(rlt_cuts=True, time_limit=60)
    assert base.status == "optimal" and rlt.status == "optimal"
    assert abs(float(base.objective) - float(rlt.objective)) < 1e-3
    assert rlt.node_count <= base.node_count


@pytest.mark.slow
def test_rlt_substantially_reduces_nodes_on_constrained_qcqp():
    base = _constrained_qcqp(6, 3).solve(rlt_cuts=False, time_limit=120)
    rlt = _constrained_qcqp(6, 3).solve(rlt_cuts=True, time_limit=120)
    assert abs(float(base.objective) - float(rlt.objective)) < 1e-3
    assert base.node_count > 50  # a genuinely non-trivial tree
    assert rlt.node_count < base.node_count / 2
