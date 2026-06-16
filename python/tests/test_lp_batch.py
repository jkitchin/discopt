"""Batched / multiple-RHS LP solving over a shared constraint matrix.

The Rust `solve_lp_batch` path computes the equilibration scaling and scaled
matrix once and reuses them across every instance. These tests pin that a batch
solve is observationally identical to solving each LP on its own (status and
objective), including on an ill-scaled matrix where the shared-scaling path runs.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.solvers import SolveStatus
from discopt.solvers.lp_simplex import SIMPLEX_AVAILABLE, solve_lp, solve_lp_batch

pytestmark = pytest.mark.skipif(not SIMPLEX_AVAILABLE, reason="Rust simplex binding not built")


def _assert_matches(c, A_ub, instances):
    batch = solve_lp_batch(c, A_ub, instances)
    assert len(batch) == len(instances)
    for (b_ub, bounds), got in zip(instances, batch):
        single = solve_lp(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        assert got.status == single.status
        if single.status == SolveStatus.OPTIMAL:
            assert got.objective == pytest.approx(single.objective, abs=1e-7)
            np.testing.assert_allclose(got.x, single.x, atol=1e-6)


def test_batch_matches_individual_solves():
    c = np.array([-1.0, -2.0])
    A_ub = np.array([[1.0, 1.0], [1.0, 3.0]])
    bounds = [(0.0, 10.0), (0.0, 10.0)]
    instances = [
        (np.array([4.0, 6.0]), bounds),
        (np.array([5.0, 6.0]), bounds),
        (np.array([2.0, 9.0]), bounds),
        (np.array([10.0, 1.0]), bounds),
    ]
    _assert_matches(c, A_ub, instances)


def test_batch_varies_bounds_too():
    # Instances differ in both rhs and bounds (full batching, not just multi-rhs).
    c = np.array([-3.0, -1.0])
    A_ub = np.array([[2.0, 1.0], [1.0, 1.0]])
    instances = [
        (np.array([10.0, 8.0]), [(0.0, 3.0), (0.0, 5.0)]),
        (np.array([10.0, 8.0]), [(0.0, 1.0), (0.0, 5.0)]),
        (np.array([6.0, 6.0]), [(0.0, 10.0), (0.0, 10.0)]),
    ]
    _assert_matches(c, A_ub, instances)


def test_batch_on_ill_scaled_matrix_shares_scaling():
    # A 1e8 linking coefficient gives the matrix a wide dynamic range, so the
    # shared equilibration path runs; the batch must still match single solves.
    c = np.array([1.0, 1.0])
    A_ub = np.array([[1e8, 1.0], [1.0, 1.0]])
    bounds = [(0.0, 1e6), (0.0, 1e6)]
    instances = [
        (np.array([2e8, 5.0]), bounds),
        (np.array([1e8, 9.0]), bounds),
    ]
    _assert_matches(c, A_ub, instances)


def test_empty_batch():
    c = np.array([1.0, 1.0])
    A_ub = np.array([[1.0, 1.0]])
    assert solve_lp_batch(c, A_ub, []) == []


def test_multiple_rhs_shared_bounds():
    # The common multiple-RHS case: same matrix and bounds, several right-hand
    # sides. Expressed as a batch with a shared bounds list.
    c = np.array([-1.0, -1.0])
    A_ub = np.array([[1.0, 2.0], [3.0, 1.0]])
    bounds = [(0.0, 100.0), (0.0, 100.0)]
    rhs_list = [np.array([r, r + 1.0]) for r in (4.0, 7.0, 12.0, 20.0, 1.0)]
    instances = [(b, bounds) for b in rhs_list]
    _assert_matches(c, A_ub, instances)
