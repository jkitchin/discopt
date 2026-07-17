"""Regression tests for issue #681 — builder-resident linear constraint rows
dropped on the nonlinear (spatial-B&B) solve path.

Linear constraints emitted through the fast-construction API
(``Model.add_linear_constraints`` and the default ``Model.constraint(..., fast=True)``
linear fast path) live only in ``model._builder_linear_blocks`` / the Rust builder,
NOT in ``model._constraints``. The JAX spatial-B&B consumers (the ``NLPEvaluator``
feasibility gate and the McCormick relaxer) read only ``model._constraints``, so on
a nonlinear solve path those rows were silently dropped and the solver certified a
FALSE OPTIMUM on an infeasible incumbent.

The X-1 fix (#413) closed this blind spot for export / validation / gp but MISSED
the core solve path. The fix normalizes builder linear rows into equivalent
expression constraints (``Model._materialize_builder_linear_rows``) before the
nonlinear path builds its evaluator/relaxer.

Each solve assertion below FAILS on pre-fix ``main`` (the false optimum) and passes
after the fix.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt import Model
from discopt._rust import model_to_repr

# ── the core false-optimal: multi-variable fast row + nonlinear objective ────


@pytest.mark.smoke
def test_continuous_bilinear_fast_row_not_dropped():
    """max x0*x1 s.t. x0+x1<=5 (fast row). True optimum -6.25 at (2.5, 2.5).

    Pre-fix: the fast row is dropped -> solver returns -100 at (10, 10), which
    violates x0+x1<=5. This is a false optimum (infeasible incumbent, dual bound
    below the true optimum).
    """
    m = Model("t")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=10.0)
    m.minimize(-(x[0] * x[1]))  # nonconvex bilinear -> spatial B&B
    m.add_linear_constraints(sp.csr_matrix(np.array([[1.0, 1.0]])), x, "<=", np.array([5.0]))
    r = m.solve(time_limit=20, gap_tolerance=1e-4)
    xv = np.asarray(r.x["x"])
    assert r.status == "optimal"
    assert xv.sum() <= 5.0 + 1e-4, f"incumbent violates the fast row: x0+x1={xv.sum()}"
    assert abs(r.objective - (-6.25)) < 1e-3, f"got {r.objective}, want -6.25"


@pytest.mark.smoke
def test_default_fast_constraint_path_binary_partition_not_dropped():
    """Balanced 2-partition of a K4 clique via the DEFAULT ``m.constraint(fast=True)``
    path with a nonconvex quadratic objective. True optimum 2 within-partition edges.

    Pre-fix: the assignment/balance rows are dropped -> solver returns obj 0 at the
    all-zeros point (violates every row) and certifies it optimal.
    """
    N, K = 4, 2
    edges = [(i, j, 1.0) for i in range(N) for j in range(i + 1, N)]
    m = dm.Model("mini")
    nodes = m.set("n", list(range(N)))
    parts = m.set("p", list(range(K)))
    x = m.binary("x", over=nodes * parts)
    m.constraint(nodes, lambda i: dm.sum(x[i, k] for k in range(K)) == 1, name="assign")
    m.constraint(parts, lambda k: dm.sum(x[i, k] for i in range(N)) == 2, name="bal")
    m.minimize(dm.sum(w * dm.sum(x[i, k] * x[j, k] for k in range(K)) for (i, j, w) in edges))
    r = m.solve(time_limit=20, gap_tolerance=1e-4)
    xv = np.asarray(r.x["x"]).reshape(N, K)
    assert r.status == "optimal"
    assert np.allclose(xv.sum(axis=1), 1), "each node must be in exactly one partition"
    assert np.allclose(xv.sum(axis=0), 2), "each partition must hold exactly 2 nodes"
    assert abs(r.objective - 2.0) < 1e-6, f"got {r.objective}, want 2.0"


@pytest.mark.smoke
def test_mixed_expression_and_fast_rows_with_nonlinear_objective():
    """Expression row + fast row together, nonlinear objective: both must bind."""
    m = Model("mix")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=10.0)
    m.minimize(-(x[0] * x[1]))
    m.subject_to(x[0] <= 3, name="expr")  # expression row
    m.add_linear_constraints(  # fast row
        sp.csr_matrix(np.array([[1.0, 1.0]])), x, "<=", np.array([5.0])
    )
    r = m.solve(time_limit=20, gap_tolerance=1e-4)
    xv = np.asarray(r.x["x"])
    assert r.status == "optimal"
    assert xv[0] <= 3.0 + 1e-4 and xv.sum() <= 5.0 + 1e-4
    # max x0*x1 with x0<=3, x0+x1<=5 -> x0=2.5, x1=2.5 (x0<=3 slack) -> -6.25.
    assert abs(r.objective - (-6.25)) < 1e-3, f"got {r.objective}, want -6.25"


# ── the normalization primitive: model-preserving relocation ─────────────────


@pytest.mark.smoke
def test_materialize_preserves_counts_and_avoids_double_count():
    m = Model("t")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=10.0)
    m.minimize(-(x[0] * x[1]))
    m.add_linear_constraints(sp.csr_matrix(np.array([[1.0, 1.0]])), x, "<=", np.array([5.0]))
    assert m.num_constraints == 1
    assert m._num_builder_constraint_rows() == 1
    assert len(m._constraints) == 0

    moved = m._materialize_builder_linear_rows()
    assert moved == 1
    assert m.num_constraints == 1  # count preserved, not doubled
    assert m._num_builder_constraint_rows() == 0
    assert len(m._constraints) == 1
    # model_to_repr must not double-count (row now only in _constraints).
    assert model_to_repr(m, getattr(m, "_builder", None)).n_constraints == 1
    # idempotent.
    assert m._materialize_builder_linear_rows() == 0
    assert m.num_constraints == 1


@pytest.mark.smoke
def test_materialize_preserves_builder_linear_objective():
    """A builder-resident linear objective survives normalization + solves right."""
    m = Model("bobj")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=10.0)
    m.add_linear_objective(np.array([1.0, 2.0]), x, sense="minimize")
    m.add_linear_constraints(sp.csr_matrix(np.array([[1.0, 1.0]])), x, ">=", np.array([3.0]))
    m._materialize_builder_linear_rows()
    assert model_to_repr(m, getattr(m, "_builder", None)).n_constraints == 1
    r = m.solve(time_limit=10, gap_tolerance=1e-4)
    assert r.status == "optimal"
    assert abs(r.objective - 3.0) < 1e-4  # x0=3, x1=0


@pytest.mark.smoke
def test_materialize_preserves_builder_quadratic_objective():
    """A builder-resident quadratic objective survives normalization + solves right."""
    m = Model("qobj")
    y = m.continuous("y", shape=(2,), lb=0.0, ub=10.0)
    m.add_quadratic_objective(sp.csr_matrix(np.eye(2)), np.zeros(2), y, sense="minimize")
    m.add_linear_constraints(sp.csr_matrix(np.array([[1.0, 1.0]])), y, ">=", np.array([2.0]))
    m._materialize_builder_linear_rows()
    assert model_to_repr(m, getattr(m, "_builder", None)).n_constraints == 1
    r = m.solve(time_limit=10, gap_tolerance=1e-4)
    assert r.status == "optimal"
    assert abs(r.objective - 1.0) < 1e-4  # min 0.5(y0^2+y1^2) s.t. y0+y1>=2 -> (1,1)


@pytest.mark.smoke
def test_expression_only_model_untouched_by_materialize():
    """A model with no builder rows is a no-op for the normalization."""
    m = Model("expr")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=10.0)
    m.minimize(-(x[0] * x[1]))
    m.subject_to(x[0] + x[1] <= 5, name="c")
    before = len(m._constraints)
    assert m._materialize_builder_linear_rows() == 0
    assert len(m._constraints) == before
