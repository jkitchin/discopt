"""#1289: an object array combined with an operand of statically-unknown shape.

``arr * xs`` is refused as ambiguous (#1239), but ``arr * (np.eye(3) @ xs)``, the
same 3-vector, went through: the construction-time shape of a matmul or an axis
sum is unknown, so every element of the result was a 3-vector, three
constraints became nine rows, and the solve certified 6.0 against 6.5 for the
element-wise pairing. Also ``mean(axis=(0, 1))`` crashed in the tape compiler.
"""

from __future__ import annotations

import discopt as do
import discopt.modeling.core as dm
import numpy as np
import pytest


def _model():
    m = do.Model("b")
    y = m.continuous("y", shape=(3,), lb=0, ub=[2, 1, 1])
    xs = m.continuous("xs", shape=(3,), lb=0, ub=1)
    X = m.continuous("X", shape=(2, 3), lb=0, ub=1)
    arr = np.array([y[0], y[1], y[2]], dtype=object)
    return m, y, xs, X, arr


@pytest.mark.parametrize(
    "build",
    [
        lambda xs, X: np.eye(3) @ xs,
        lambda xs, X: X.sum(axis=0),
        lambda xs, X: X.mean(axis=0),
        lambda xs, X: np.eye(3) @ xs + 1.0,
        lambda xs, X: -(np.ones(2) @ X),
    ],
    ids=["matmul", "axis_sum", "axis_mean", "matmul_plus", "neg_matmul"],
)
@pytest.mark.parametrize("op", ["*", "+", "-", "/"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_array_valued_operand_of_unknown_construction_shape_is_refused(build, op, side):
    _, _, xs, X, arr = _model()
    vec = build(xs, X)
    fns = {
        "*": lambda a, b: a * b,
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "/": lambda a, b: a / b,
    }
    with pytest.raises(TypeError, match="elementwise|pair"):
        if side == "left":
            fns[op](arr, vec)
        else:
            fns[op](vec, arr)


@pytest.mark.parametrize(
    "build",
    [
        lambda xs, X: dm.sum(xs),
        lambda xs, X: xs.mean(),
        lambda xs, X: np.ones(3) @ xs,
        lambda xs, X: xs @ xs,
        lambda xs, X: dm.norm(xs),
        lambda xs, X: dm.norm(xs, 1),
        lambda xs, X: dm.norm(xs, "inf"),
        lambda xs, X: xs.prod(),
        lambda xs, X: 2.0 * dm.sum(xs) + dm.norm(xs),
        lambda xs, X: X.sum(axis=0) @ xs,
        lambda xs, X: X.mean(axis=(0, 1)),
    ],
    ids=[
        "sum",
        "mean",
        "dot_const",
        "dot_self",
        "norm2",
        "norm1",
        "norminf",
        "prod",
        "composite",
        "axis_sum_dot",
        "tuple_mean",
    ],
)
def test_scalar_operand_still_broadcasts(build):
    _, y, xs, X, arr = _model()
    out = arr * build(xs, X)
    assert out.shape == (3,)
    assert all(dm._known_shape(e) in ((), None) for e in out)


def test_issue_repro_is_refused():
    _, _, xs, _, arr = _model()
    with pytest.raises(TypeError):
        arr * (np.eye(3) @ xs)


def test_tuple_axis_mean_solves():
    # max sum(X) s.t. mean over both axes <= 0.5  ->  sum(X) = 3.
    m = do.Model("tm")
    X = m.continuous("X", shape=(2, 3), lb=0, ub=1)
    m.subject_to(X.mean(axis=(0, 1)) <= 0.5)
    m.maximize(X.sum())
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(3.0, abs=1e-6)


def test_partial_tuple_axis_sum_solves():
    # X in (2, 2, 3); summing axes (0, 2) leaves one row per middle index.
    m = do.Model("ta")
    X = m.continuous("X", shape=(2, 2, 3), lb=0, ub=1)
    m.subject_to(X.sum(axis=(0, 2)) <= np.array([1.0, 2.0]))
    m.maximize(X.sum())
    r = m.solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(3.0, abs=1e-6)
