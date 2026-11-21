"""The decomposition structure layer must see variables inside array reductions.

``gdp_reformulate._collect_variables`` has no arm for ``SumExpression``,
``MatMulExpression`` or ``CustomCall``. The decomposition layer used it, so a
row written with ``dm.sum(x)`` or ``A @ x`` looked as if it touched only its
other variables (or none). Three consequences, each pinned here:

* ``detect_decomposition`` split a coupled model into "independent" blocks;
* the Lagrangian solver dropped such rows from its block subproblems as
  "references no variable", leaving a valid but vacuous bound (0.0 where the
  optimum is 24.0);
* the GBD master-only nonlinear guard (C2) saw an empty support and let a
  master-only nonlinear row through instead of refusing it.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.decomposition import solve_lagrangian
from discopt.decomposition.graph.base import ModelGraph, variable_names_in
from discopt.decomposition.structure import detect_decomposition


def _sum_coupled():
    m = dm.Model("sum_coupled")
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    m.subject_to(dm.sum(x) + y <= 1)
    m.subject_to(x[0] <= 0.5)
    m.minimize(-dm.sum(x) - 2 * y)
    return m


def test_variable_names_in_sees_every_reduction():
    m = dm.Model("walk")
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    w = m.continuous("w", shape=(3,), lb=0, ub=1)
    a = np.ones((1, 3))
    assert set(variable_names_in((dm.sum(x) + y <= 1).body)) == {"x", "y"}
    assert set(variable_names_in(((a @ w)[0] <= 1).body)) == {"w"}
    assert variable_names_in(None) == []


def test_sum_row_couples_blocks():
    m = _sum_coupled()
    structure = detect_decomposition(m)
    assert structure.num_blocks == 1
    assert not structure.is_separable
    assert structure.block_of_constraint[0] == 0
    assert ModelGraph.from_model(m).num_blocks() == 1


def test_matmul_row_is_not_variable_free():
    m = dm.Model("matmul")
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    a = np.ones((1, 3))
    m.subject_to((a @ x)[0] >= 1)
    m.minimize(dm.sum(x))
    assert detect_decomposition(m).block_of_constraint == [0]


def _blocks_with_sum_rows():
    """Three (x_b, z_b) blocks joined by one annotated budget row."""
    m = dm.Model("lag_sum")
    zs, xs = [], []
    for b, (k, need, cap) in enumerate([(3, 1.0, 1.0), (2, 5.0, 2.0), (3, 3.0, 2.0)]):
        x = m.continuous(f"x{b}", shape=(k,), lb=0, ub=6)
        z = m.integer(f"z{b}", lb=0, ub=4)
        for i in range(k):
            m.subject_to(x[i] <= cap * z)
        m.subject_to(dm.sum(x) >= need)
        zs.append(z)
        xs.append(x)
    m.subject_to(zs[0] + zs[1] + zs[2] <= 6, name="budget")
    m.mark_coupling("budget")
    m.minimize(sum((b + 1) * zs[b] + dm.sum(xs[b]) for b in range(3)))
    return m


@pytest.mark.requires_highs
def test_lagrangian_keeps_sum_rows_in_its_blocks():
    mono = _blocks_with_sum_rows().solve()
    assert mono.status == "optimal"
    result = solve_lagrangian(_blocks_with_sum_rows(), max_iterations=40)
    assert result.bound is not None
    # With the ``dm.sum`` rows dropped every block subproblem is ``min c.z``
    # over a box with c >= 0: bound 0.0. Kept, the bound is the optimum here
    # (the budget is slack), and never above it.
    assert result.bound <= mono.objective + 1e-6
    assert result.bound >= mono.objective - 1e-4


def test_gbd_refuses_master_only_nonlinear_sum_row():
    from discopt.decomposition import solve_gbd

    m = dm.Model("gbd_c2")
    z = m.integer("z", shape=(2,), lb=0, ub=3)
    y = m.continuous("y", lb=0, ub=10)
    m.subject_to(dm.sum(z) ** 2 <= 4)  # master-only, nonlinear, non-binary master
    m.subject_to(y >= z[0] + 0.5)
    m.subject_to(y**2 <= 50)
    m.minimize(y + dm.sum(z))
    with pytest.raises(NotImplementedError, match="master-only nonlinear"):
        solve_gbd(m)


def _block_diagonal():
    m = dm.Model("diag")
    x = m.continuous("x", shape=(2,), lb=0, ub=1)
    y = m.continuous("y", shape=(2,), lb=0, ub=1)
    m.subject_to(x[0] + x[1] <= 1)
    m.subject_to(y[0] + y[1] <= 1.5)
    m.minimize(-dm.sum(x) - dm.sum(y))
    return m


def test_independent_blocks_route_solves():
    """``INDEPENDENT_BLOCKS`` falls back to ``Model.solve``; the driver keyword
    ``max_iterations`` must not reach it (it raised TypeError)."""
    from discopt.decomposition import MethodKind, analyze_decomposition

    advisor = analyze_decomposition(_block_diagonal())
    assert advisor.recommendation().recommendation is MethodKind.INDEPENDENT_BLOCKS
    result = advisor.decompose().solve(max_iterations=1000, time_limit=60.0)
    assert result.status == "optimal"
    assert result.objective == pytest.approx(-2.5, abs=1e-6)

    auto = _block_diagonal().solve(decomposition="auto")
    assert auto.status == "optimal"
    assert auto.objective == pytest.approx(-2.5, abs=1e-6)
