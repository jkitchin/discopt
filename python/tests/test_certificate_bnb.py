"""Tier-3 (spatial branch-and-bound) checker-primitive tests.

These pin the exact-rational soundness kernel a full Tier-3 certificate composes:
box covering, McCormick envelope validity, LP weak-duality bounds, Farkas
infeasibility, and the per-leaf composition (valid relaxation rows + certified
bound). They run on small hand-built data -- no solve -- so they are ``smoke``.

Worked nonconvex example: ``min -x^2`` over ``x in [0, 2]`` (concave; global min
``-4`` at ``x=2``). Lifting ``w = x^2`` and relaxing with McCormick gives a linear
program whose weak-duality bound is exactly ``-4`` -- the gap closes at the root.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest
from discopt.certificate.bnb import (
    certified_leaf_bound,
    check_tree_covers,
    farkas_infeasible,
    lp_lower_bound,
    mccormick_bilinear,
    mccormick_square,
    row_holds,
)


def _leaf(lo, ub):
    return {"box": {"lb": lo, "ub": ub}, "kind": "leaf"}


def _branch(lo, ub, var, point, children):
    return {
        "box": {"lb": lo, "ub": ub},
        "kind": "branch",
        "branch": {"var": var, "point": point, "children": children},
    }


@pytest.mark.smoke
def test_covering_spatial_and_integer():
    spatial = {
        "r": _branch([[0, 1]], [[2, 1]], 0, [1, 1], ["a", "b"]),
        "a": _leaf([[0, 1]], [[1, 1]]),
        "b": _leaf([[1, 1]], [[2, 1]]),
    }
    assert check_tree_covers(spatial, "r", set())[0]

    integer = {
        "r": _branch([[0, 1]], [[5, 1]], 0, [5, 2], ["a", "b"]),  # split at 2.5
        "a": _leaf([[0, 1]], [[2, 1]]),
        "b": _leaf([[3, 1]], [[5, 1]]),
    }
    assert check_tree_covers(integer, "r", {0})[0]


@pytest.mark.smoke
def test_covering_rejects_gap():
    broken = {
        "r": _branch([[0, 1]], [[2, 1]], 0, [1, 1], ["a", "b"]),
        "a": _leaf([[0, 1]], [[1, 1]]),
        "b": _leaf([[3, 2]], [[2, 1]]),  # left edge 1.5 != split point 1 -> gap
    }
    ok, reason = check_tree_covers(broken, "r", set())
    assert not ok and "split" in reason.lower()


@pytest.mark.smoke
def test_mccormick_bilinear_valid_at_samples():
    rows = mccormick_bilinear(F(0), F(2), F(1), F(3), 0, 1, 2)
    for x in (F(0), F(1, 2), F(1), F(2)):
        for y in (F(1), F(2), F(5, 2), F(3)):
            pt = {0: x, 1: y, 2: x * y}
            assert all(row_holds(r, pt) for r in rows), (x, y)


@pytest.mark.smoke
def test_mccormick_square_valid_at_samples():
    rows = mccormick_square(F(0), F(2), 0, 1)
    for x in (F(0), F(1, 2), F(1), F(3, 2), F(2)):
        assert all(row_holds(r, {0: x, 1: x * x}) for r in rows), x


@pytest.mark.smoke
def test_lp_weak_duality_bound():
    # min -w s.t. 2x-w>=0, x>=0, -x>=-2, w>=0  (relaxation of min -x^2 on [0,2])
    a = [[F(2), F(-1)], [F(1), F(0)], [F(-1), F(0)], [F(0), F(1)]]
    b = [F(0), F(0), F(-2), F(0)]
    c = [F(0), F(-1)]
    ok, bound = lp_lower_bound(a, b, c, [F(1), F(0), F(2), F(0)])
    assert ok and bound == F(-4)
    # dual infeasibility is rejected
    assert not lp_lower_bound(a, b, c, [F(-1), F(0), F(2), F(0)])[0]
    assert not lp_lower_bound(a, b, c, [F(1), F(0), F(1), F(0)])[0]  # Aᵀy != c


@pytest.mark.smoke
def test_certified_leaf_bound_and_unsound_cut():
    box = ([F(0), None], [F(2), None])  # x in [0,2]; w (aux) unbounded
    lp = {
        "A": [[F(2), F(-1)], [F(1), F(0)], [F(-1), F(0)], [F(0), F(1)]],
        "b": [F(0), F(0), F(-2), F(0)],
        "c": [F(0), F(-1)],
    }
    aux = [{"op": "square", "x": 0, "w": 1}]
    ok, bound = certified_leaf_bound(box, lp, [F(1), F(0), F(2), F(0)], aux)
    assert ok and bound == F(-4)

    # Injecting a cut that is NOT a valid McCormick/box row is rejected outright.
    lp2 = {"A": lp["A"] + [[F(1), F(-1)]], "b": lp["b"] + [F(0)], "c": lp["c"]}
    ok2, reason = certified_leaf_bound(box, lp2, [F(1), F(0), F(2), F(0), F(0)], aux)
    assert not ok2 and "unsound" in reason.lower()


@pytest.mark.smoke
def test_farkas_infeasible_leaf():
    # x >= 1 and x <= 0 is empty; ray y=(1,1) gives Aᵀy=0, b·y=1>0.
    a = [[F(1)], [F(-1)]]
    b = [F(1), F(0)]
    assert farkas_infeasible(a, b, [F(1), F(1)])[0]
    # a feasible system yields no ray
    assert not farkas_infeasible([[F(1)]], [F(0)], [F(1)])[0]
