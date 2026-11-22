"""A long Python ``sum(...)`` in an array-walked body must not hit the recursion limit.

``canonicalize_expression`` sends a body whose static shape is unknown --
``dm.sum(y[0]) + sum(x[i] for i in range(4000))`` -- through the exact *array*
walk, which recursed once per ``BinaryOp`` of the chain: ``Model.validate``
(so every ``solve``) raised ``RecursionError`` at ~1000 terms. The walk is now
iterative over ``+``/``-``/``*`` chains, visiting nodes in the same order.
"""

import discopt.modeling as dm
import numpy as np
import pytest

N = 4000


def test_unknown_shape_body_with_long_chain_validates_and_solves():
    m = dm.Model("deep_chain")
    x = m.continuous("x", shape=(N,), lb=0, ub=1)
    y = m.continuous("y", shape=(4, 3), lb=0, ub=1)
    m.subject_to(x[0] + y[0, 0] >= 0.5)
    # Maximise a sum: the optimum is every variable at its upper bound.
    m.maximize(dm.sum(y[0]) + sum(x[i] for i in range(N)))
    m.validate()
    res = m.solve()
    assert res.status == "optimal"
    assert res.objective == pytest.approx(N + 3.0)


def test_array_valued_long_chain_validates():
    m = dm.Model("deep_array_chain")
    y = m.continuous("Y", shape=(3, 5), lb=0, ub=1)
    c = np.linspace(0.5, 1.5, N)
    # An array-valued chain: sum over i of c_i * Y[i % 3] (shape (5,)).
    m.subject_to(sum(float(c[i]) * y[i % 3] for i in range(N)) <= 10.0)
    m.minimize(-dm.sum(y))
    m.validate()
    res = m.solve()
    assert res.status == "optimal"
    # Each column j: sum_i c_i Y[i%3, j] <= 10 with Y in [0, 1].
    yv = np.asarray(res.x["Y"])
    lhs = np.array([sum(c[i] * yv[i % 3, j] for i in range(N)) for j in range(5)])
    assert np.all(lhs <= 10.0 + 1e-6)
