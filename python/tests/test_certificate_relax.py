"""Exact McCormick relaxation compiler + dual recovery (Tier-3 untrusted).

These pin the trusted relaxation compiler the untrusted Tier-3 checker relies on:
the quadratic-form extractor, the McCormick LP assembly, and the exact rational
dual recovery. Small LPs (SciPy for the active set, exact Fraction for the dual),
so ``smoke``.
"""

from __future__ import annotations

from fractions import Fraction as F

import pytest
from discopt.certificate.bnb import lp_lower_bound
from discopt.certificate.relax import (
    NotQuadratic,
    build_leaf_lp,
    extract,
    leaf_dual,
)


def _const(n, d=1):
    return {"k": "const", "v": [n, d]}


def _var(i):
    return {"k": "var", "i": i}


def _model(objective_body, constraints=(), ncols=1):
    cols = [{"name": f"x{i}", "type": "continuous"} for i in range(ncols)]
    return {
        "columns": cols,
        "constraints": list(constraints),
        "objective": {"sense": "min", "body": objective_body},
    }


@pytest.mark.smoke
def test_quadform_extract():
    # x*y - 2x  ->  lin {0:-2}, quad {(0,1):1}
    q = extract(
        {
            "k": "sub",
            "l": {"k": "mul", "l": _var(0), "r": _var(1)},
            "r": {"k": "mul", "l": _const(2), "r": _var(0)},
        }
    )
    assert dict(q.lin) == {0: F(-2)}
    assert dict(q.quad) == {(0, 1): F(1)}
    assert q.const == 0


@pytest.mark.smoke
def test_extract_refuses_cubic_and_transcendental():
    with pytest.raises(NotQuadratic):
        extract({"k": "pow", "l": _var(0), "r": _const(3)})
    with pytest.raises(NotQuadratic):
        extract({"k": "fn", "name": "exp", "args": [_var(0)]})
    with pytest.raises(NotQuadratic):
        extract({"k": "div", "l": _var(0), "r": _var(1)})  # non-constant denominator


@pytest.mark.smoke
def test_leaf_lp_square_bound_with_hand_dual():
    # min -x^2 over [0,2]; lifting w=x^2, McCormick bound is -4 (the global optimum).
    model = _model({"k": "neg", "x": {"k": "pow", "l": _var(0), "r": _const(2)}})
    lp = build_leaf_lp(model, [F(0)], [F(2)])
    assert lp["n_total"] == 2 and lp["aux"][0]["op"] == "square"
    # A hand-verified dual over rows [x>=0, x<=2, tangent1, tangent2, secant].
    y = [F(0), F(2), F(0), F(0), F(1)]
    ok, bound = lp_lower_bound(lp["A"], lp["b"], lp["c"], y)
    assert ok and bound + lp["obj_const"] == F(-4)


@pytest.mark.smoke
def test_leaf_dual_recovers_exact_bound():
    # Automatic exact dual recovery matches the hand value (-4), degenerate vertex.
    model = _model({"k": "neg", "x": {"k": "pow", "l": _var(0), "r": _const(2)}})
    lp = build_leaf_lp(model, [F(0)], [F(2)])
    res = leaf_dual(lp)
    assert res is not None
    bound, y = res
    assert bound == F(-4)
    # The recovered dual is exactly feasible for the rebuilt LP.
    ok, b2 = lp_lower_bound(lp["A"], lp["b"], lp["c"], y)
    assert ok and b2 + lp["obj_const"] == bound


@pytest.mark.smoke
def test_leaf_dual_bilinear():
    # min x*y over [0,3]^2 s.t. x+y>=2; McCormick lower bound is 0 (valid).
    model = _model(
        {"k": "mul", "l": _var(0), "r": _var(1)},
        constraints=[
            {
                "name": "c",
                "sense": "ge",
                "body": {"k": "add", "l": _var(0), "r": _var(1)},
                "rhs": [2, 1],
            }
        ],
        ncols=2,
    )
    lp = build_leaf_lp(model, [F(0), F(0)], [F(3), F(3)])
    res = leaf_dual(lp)
    assert res is not None
    bound, _y = res
    # A valid lower bound on x*y over the box (the true min is 0).
    assert bound <= F(0)
