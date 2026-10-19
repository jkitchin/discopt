"""#1808: the deep-recursion gate could not see fast-API rows or disjunct bodies.

``_scoped_deep_recursion`` decides between the default recursion limit and the
large-stack worker from ``_max_expr_node_count(model)``, which read only the
objective and the plain ``Constraint`` bodies in ``model._constraints``. Two kinds
of row become deep expressions later, *inside* the already-inline solve:

* fast-API rows (``add_linear_constraints``), materialised by
  ``_materialize_builder_linear_rows`` -- repro A overflowed in
  ``_model_contains_nonsmooth_node._walk``;
* the bodies of disjuncts (``either_or``), walked by the GDP lowering -- repro B
  overflowed in ``_collect_variables`` (hull) and ``_bound_expression`` (big-M).

Indicator rows (``if_then``) hide their body the same way. The count now covers
all three; the gate over-approximates by design, so the tests check it never
under-counts what is later materialised.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.factorable_reform import _expr_node_count, _max_expr_node_count
from discopt.solver import _DEEP_SOLVE_DEPTH_GATE

N = 2500


def _chain(x, n):
    row = x[0]
    for i in range(1, n):
        row = row + x[i]
    return row


@pytest.mark.smoke
def test_builder_row_bound_covers_materialised_row():
    m = dm.Model("count")
    x = m.continuous("x", shape=(300,), lb=0, ub=1)
    A = sp.csr_matrix(np.vstack([np.ones(300), np.r_[np.ones(10), np.zeros(290)]]))
    m.add_linear_constraints(A, x, ">=", np.array([0.0, 0.0]))
    bound = _max_expr_node_count(m)
    assert len(m._constraints) == 0
    assert m._materialize_builder_linear_rows() == 2
    actual = max(_expr_node_count(c.body) for c in m._constraints)
    assert actual > _DEEP_SOLVE_DEPTH_GATE  # the row is deep enough to matter
    assert bound >= actual


@pytest.mark.smoke
def test_count_sees_disjunct_and_indicator_bodies():
    m = dm.Model("count")
    x = m.continuous("x", shape=(400,), lb=0, ub=1)
    deep = _chain(x, 400)
    depth = _expr_node_count((deep >= 1).body)
    m.either_or([[x[0] <= 0.5], [m.disjunction([[deep >= 1], [x[1] <= 0.5]])]])
    assert _max_expr_node_count(m) >= depth

    m2 = dm.Model("ind")
    y = m2.continuous("y", shape=(400,), lb=0, ub=1)
    z = m2.binary("z")
    m2.if_then(z, [_chain(y, 400) >= 1])
    assert _max_expr_node_count(m2) >= depth


@pytest.mark.slow
def test_repro_a_fast_api_row_nonconvex_objective():
    b = np.random.default_rng(0).uniform(0.1, 0.9, N)
    m = dm.Model("nc")
    x = m.continuous("x", shape=(N,), lb=0, ub=1)
    m.add_linear_constraints(sp.csr_matrix(np.ones((1, N))), x, ">=", np.array([0.0]))
    m.minimize(dm.sum(-(x * x)) + dm.sum(b * x))
    assert _max_expr_node_count(m) > _DEEP_SOLVE_DEPTH_GATE
    r = m.solve(time_limit=20)
    # Same status as the pure-row control in the issue (the gate saw that row).
    assert r.status in ("feasible", "optimal")
    assert r.objective is not None and np.isfinite(r.objective)


@pytest.mark.slow
@pytest.mark.parametrize("gdp_method", ["hull", "big-m"])
def test_repro_b_long_row_inside_disjunct(gdp_method):
    m = dm.Model("gdp")
    x = m.continuous("x", shape=(N,), lb=0, ub=1)
    m.either_or([[_chain(x, N) >= 1], [x[0] <= 0.5]])
    m.minimize(dm.sum(x))
    assert _max_expr_node_count(m) > _DEEP_SOLVE_DEPTH_GATE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(time_limit=30, gdp_method=gdp_method)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(0.0, abs=1e-6)


# ``m.logical`` rows hold a ``LogicalExpression``, not an ``Expression``: the gate
# skipped them, and the GDP lowering's ``_to_nnf`` (inside the solve) and the
# post-solve ``DeclaredLogic`` check both recursed once per level of a loop-built
# ``Y[0] | Y[1] | ...``.


@pytest.fixture
def default_recursion_limit():
    """CPython's default limit. pytest runs tests at 3000, under which the old
    recursive walks survived a 2500-deep chain that overflows a user's script."""
    import sys

    old = sys.getrecursionlimit()
    sys.setrecursionlimit(1000)
    try:
        yield
    finally:
        sys.setrecursionlimit(old)


def _logic_chain(Y, n, op):
    e = Y[0]
    for i in range(1, n):
        e = op(e, Y[i])
    return e


@pytest.mark.smoke
def test_count_sees_logical_constraint():
    m = dm.Model("logic")
    Y = m.boolean("Y", shape=(400,))
    m.logical(_logic_chain(Y, 400, lambda a, b: a | b))
    assert _max_expr_node_count(m) >= 2 * 400 - 1


@pytest.mark.smoke
def test_declared_logic_check_is_iterative(default_recursion_limit):
    from discopt.validation.feasibility import DeclaredLogic

    m = dm.Model("logic")
    Y = m.boolean("Y", shape=(N,))
    m.logical(_logic_chain(Y, N, lambda a, b: a | b))
    m.logical(~_logic_chain(Y, N, lambda a, b: a & b))
    m.minimize(dm.sum(Y.variable))
    logic = DeclaredLogic(m)
    one_on = np.zeros(N)
    one_on[N // 2] = 1.0
    assert logic.check(m, one_on).ok
    assert not logic.check(m, np.zeros(N)).ok  # the OR is false
    assert not logic.check(m, np.ones(N)).ok  # the NOT-AND is false


@pytest.mark.slow
@pytest.mark.parametrize(
    "shape, sense, expected",
    [
        ("or", "min", 1.0),  # Y[0] | ... | Y[N-1]: at least one true
        ("not_and", "max", N - 1.0),  # ~(Y[0] & ... & Y[N-1]): at least one false
        ("implies", "max", float(N)),  # ((Y0 -> Y1) -> Y2) -> ...: all true holds
    ],
)
def test_deep_logical_constraint_solves(shape, sense, expected, default_recursion_limit):
    m = dm.Model("logic")
    Y = m.boolean("Y", shape=(N,))
    if shape == "or":
        e = _logic_chain(Y, N, lambda a, b: a | b)
    elif shape == "not_and":
        e = ~_logic_chain(Y, N, lambda a, b: a & b)
    else:
        e = _logic_chain(Y, N, lambda a, b: a.implies(b))
    m.logical(e)
    obj = dm.sum(Y.variable)
    m.minimize(obj) if sense == "min" else m.maximize(obj)
    assert _max_expr_node_count(m) > _DEEP_SOLVE_DEPTH_GATE
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(expected, abs=1e-6)
