"""``dm.sum(Q[i])`` over a multi-dimensional array is a sum over a sub-array.

The Rust core read a short integer index (``Q[g]`` on a 4-D ``Q``, or
``Q[g, p]``) as the single flat element ``Q.flat[g]``. Every consumer of
``index_spec_collect_flat`` inherited it: the LP/MILP extraction optimized a
different objective (the default route's reconciliation guard withdrew the
certificate, reporting ``feasible`` at a non-optimal point), and a Benders
solve -- which has no such guard -- reported a CERTIFIED optimum of 67.07 for a
point whose true objective is 193.49 (a mixed-integer capacity-expansion model).
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.decomposition._linear import extract_linear


def _fixed_model(shape, build):
    """Every variable fixed to 1..n, so the objective's value is known exactly."""
    vals = np.arange(1, int(np.prod(shape)) + 1, dtype=float).reshape(shape)
    m = dm.Model("partial_index")
    q = m.continuous("Q", shape=shape, lb=vals, ub=vals)
    m.minimize(build(q))
    return m, vals


def _np_sum(a):
    return a.sum() if isinstance(a, np.ndarray) else dm.sum(a)


CASES = [
    ((2, 3), lambda q: _np_sum(q[0]) + 2 * _np_sum(q[1])),
    ((2, 2, 3), lambda q: _np_sum(q[1])),
    ((2, 2, 3), lambda q: 3 * _np_sum(q[1, 0]) - _np_sum(q[0])),
    ((2, 3, 2, 2), lambda q: _np_sum(q[1]) + _np_sum(q[0, 2])),
    ((2, 3), lambda q: _np_sum(q[:, 1])),  # slices were already right
]


@pytest.mark.parametrize("shape,build", CASES)
def test_extract_linear_reads_subarray_sums(shape, build):
    m, vals = _fixed_model(shape, build)
    lin = extract_linear(m)
    assert float(lin.c @ vals.ravel() + lin.c_offset) == pytest.approx(float(build(vals)))


@pytest.mark.parametrize("shape,build", CASES)
def test_solve_is_certified_at_the_true_objective(shape, build):
    m, vals = _fixed_model(shape, build)
    res = m.solve()
    assert res.status == "optimal" and res.gap_certified
    assert res.objective == pytest.approx(float(build(vals)))


@pytest.mark.requires_highs
def test_benders_certificate_matches_its_point():
    """The reported false certificate: Benders' objective is the true value."""
    from discopt.decomposition import solve_benders

    G, P, S, T = 2, 2, 2, 3
    rng = np.random.default_rng(0)
    demand = rng.uniform(2.0, 6.0, size=(P, S, T))
    var = np.array([0.3, 0.7])
    m = dm.Model("benders_partial_index")
    n = m.integer("N", shape=(G, P), lb=0, ub=5)
    q = m.continuous("Q", shape=(G, P, S, T), lb=0, ub=20)
    u = m.continuous("U", shape=(P, S, T), lb=0, ub=50)
    for p in range(P):
        for s in range(S):
            for t in range(T):
                m.subject_to(q[0, p, s, t] + q[1, p, s, t] + u[p, s, t] == float(demand[p, s, t]))
                for g in range(G):
                    m.subject_to(q[g, p, s, t] <= 2.0 * n[g, p])
    m.minimize(
        dm.sum(n)
        + (float(var[0]) * dm.sum(q[0]) + float(var[1]) * dm.sum(q[1])) / S
        + 6.0 * dm.sum(u) / S
    )
    mono = m.solve()
    res = solve_benders(m)
    assert mono.status == "optimal" and res.status == "optimal"
    assert res.objective == pytest.approx(mono.objective, abs=1e-6)
    x = res.x
    true_obj = (
        np.asarray(x["N"]).sum()
        + (var[0] * np.asarray(x["Q"])[0].sum() + var[1] * np.asarray(x["Q"])[1].sum()) / S
        + 6.0 * np.asarray(x["U"]).sum() / S
    )
    assert res.objective == pytest.approx(true_obj, abs=1e-6)


# The fast-API builder addresses a FLAT column of a variable block. With a short
# integer index meaning the numpy sub-array, a bare `Scalar(col)` on a 2-D block
# named a whole row: an LP built through `add_linear_constraints` on `X` of shape
# (2, 3) returned `feasible` 35.96 instead of `optimal` 27.06 (reported in review
# of jkitchin/discopt#2282). The builder now writes a full-arity index there.
_A = np.vstack([np.eye(6), np.random.default_rng(0).uniform(0, 1, size=(3, 6))])
_B = np.concatenate([[2, 0, 0, 0, 0, 4], [1.0, 2.0, 1.5]])
_W = np.array([3.0, 1.0, 2.0, 1.5, 2.5, 5.0])
_SHAPES = [(6,), (2, 3), (3, 2), (2, 1, 3)]


def _fast_lp(shape, objective):
    m = dm.Model(f"fast_{'x'.join(map(str, shape))}")
    x = m.continuous("X", shape=shape, lb=0, ub=10)
    m.add_linear_constraints(_A, x, ">=", _B)
    if objective == "expression":
        m.minimize(dm.sum(_W.reshape(shape) * x))
    elif objective == "linear":
        m.add_linear_objective(_W, x)
    else:  # quadratic: 0.5 x'Qx + c'x with Q = diag(1)
        m.add_quadratic_objective(np.eye(6), _W, x)
    return m


@pytest.mark.parametrize("objective", ["expression", "linear", "quadratic"])
def test_fast_api_rows_on_multidimensional_variables(objective):
    reference = _fast_lp((6,), objective).solve()
    assert reference.status == "optimal"
    checked = 0
    for shape in _SHAPES[1:]:
        res = _fast_lp(shape, objective).solve()
        assert res.status == "optimal" and res.gap_certified, (shape, res.status)
        assert res.objective == pytest.approx(reference.objective, rel=1e-6), shape
        x = np.asarray(res.x["X"]).ravel()
        assert np.all(_A @ x >= _B - 1e-6), shape
        checked += 1
    assert checked == len(_SHAPES) - 1
