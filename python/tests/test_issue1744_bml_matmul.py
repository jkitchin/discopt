"""#1744: the binary-multilinear MILP linearization must read a scalar ``@``.

``c @ x`` and ``x @ Q @ x`` are the same polynomials as ``dm.sum(c * x)`` and
``sum_ij Q[i,j] x_i x_j``, but the pass's walks refused ``MatMulExpression``, so
one ``@`` anywhere in a model aborted the rewrite and left a binary QP on spatial
branch and bound (the #1731 subset-sum stalled at bound 834.5 for the whole time
limit). The pass now reads a scalar-valued ``@`` as its explicit contraction.
Every value here is judged against brute-force enumeration.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import binary_multilinear_reform as B
from discopt.modeling.core import MatMulExpression

pytestmark = pytest.mark.relaxation


def _has_matmul(expr) -> bool:
    stack, seen = [expr], set()
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, MatMulExpression):
            return True
        for attr in ("left", "right", "operand", "base"):
            child = getattr(node, attr, None)
            if isinstance(child, dm.Expression):
                stack.append(child)
        stack.extend(getattr(node, "terms", None) or [])
    return False


def _subset_sum(w, W, *, vec_con: bool, vec_obj: bool):
    n = len(w)
    m = dm.Model("ss")
    x = m.binary("x", shape=(n,))
    lhs = (w @ x) if vec_con else sum(int(w[i]) * x[i] for i in range(n))
    obj = (w @ x) if vec_obj else sum(int(w[i]) * x[i] for i in range(n))
    m.subject_to(lhs <= W)
    m.maximize(obj)
    m.subject_to(x[0] * x[1] <= 1)  # redundant binary product
    return m


def _qkp(Q, wt, cap):
    m = dm.Model("qkp")
    x = m.binary("x", shape=(len(wt),))
    m.subject_to(wt @ x <= cap)
    m.maximize(x @ Q @ x)
    return m


def _brute_qkp(Q, wt, cap):
    best = -np.inf
    for bits in itertools.product((0, 1), repeat=len(wt)):
        v = np.asarray(bits, dtype=float)
        if wt @ v <= cap + 1e-9:
            best = max(best, float(v @ Q @ v))
    return best


def test_expansion_of_quadratic_form_matches_every_vertex():
    """``x @ Q @ x + c @ x`` expands to the exact multilinear polynomial."""
    rng = np.random.default_rng(0)
    n = 6
    Q = rng.integers(-5, 6, (n, n)).astype(float)
    c = rng.integers(-5, 6, n).astype(float)
    m = dm.Model("q")
    x = m.binary("x", shape=(n,))
    expr = x @ Q @ x + c @ x
    ctx = B._ExpandCtx()
    poly = B._expand_to_multilinear(expr, ctx)
    checked = 0
    for bits in itertools.product((0, 1), repeat=n):
        v = np.asarray(bits, dtype=float)
        val = sum(coef * np.prod([v[e] for _, e in mono]) for mono, coef in poly.items())
        assert val == pytest.approx(float(v @ Q @ v + c @ v), abs=1e-9)
        checked += 1
    assert checked == 2**n


@pytest.mark.parametrize("vec_con,vec_obj", [(True, True), (True, False), (False, True)])
def test_matmul_spelling_reformulates_to_matmul_free_milp(vec_con, vec_obj):
    w = np.array([13, 41, 27, 8, 55, 32])
    m = _subset_sum(w, 70.5, vec_con=vec_con, vec_obj=vec_obj)
    assert B.has_binary_multilinear_work(m)
    r = B.reformulate_binary_multilinear(m)
    assert r is not m, "an `@` in the model must not abort the pass (#1744)"
    assert not _has_matmul(r._objective.expression)
    assert not any(_has_matmul(c.body) for c in r._constraints)


def test_quadratic_form_alone_is_a_witness():
    m = dm.Model("q")
    x = m.binary("x", shape=(4,))
    m.maximize(x @ np.triu(np.ones((4, 4))) @ x)
    assert B.has_binary_multilinear_work(m)


def test_array_valued_matmul_still_abstains():
    """An array-valued ``A @ x`` row is not a scalar body; the pass keeps refusing."""
    m = dm.Model("a")
    x = m.binary("x", shape=(3,))
    m.subject_to(np.ones((2, 3)) @ x <= 2)
    m.maximize(x[0] * x[1] + x[2])
    assert B.reformulate_binary_multilinear(m) is m


def test_continuous_factor_inside_quadratic_form_abstains():
    m = dm.Model("c")
    x = m.binary("x", shape=(3,))
    y = m.continuous("y", shape=(3,), lb=0, ub=1)
    m.maximize(x @ np.ones((3, 3)) @ y)
    assert B.reformulate_binary_multilinear(m) is m


@pytest.mark.parametrize("vec_con,vec_obj", [(True, True), (True, False), (False, True)])
def test_subset_sum_with_matmul_certifies_on_milp_route(vec_con, vec_obj):
    """The #1744 repro at a brute-forceable size: certified, exact, MILP route."""
    rng = np.random.default_rng(1)
    w = rng.integers(10, 100, size=12)
    W = int(w.sum() / 2) + 0.5
    opt = max(
        float(w @ np.asarray(b))
        for b in itertools.product((0, 1), repeat=len(w))
        if w @ np.asarray(b) <= W
    )
    r = _subset_sum(w, W, vec_con=vec_con, vec_obj=vec_obj).solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(opt, abs=1e-6)
    assert r.bound >= opt - 1e-6
    assert r.bound <= opt + 1e-6
    assert "spatial-bb" not in (r.algorithm_route or "")


def test_quadratic_knapsack_matmul_matches_brute_force():
    rng = np.random.default_rng(3)
    n = 10
    Q = np.triu(rng.integers(0, 20, (n, n))).astype(float)
    wt = rng.integers(5, 30, n)
    cap = int(wt.sum() * 0.4)
    opt = _brute_qkp(Q, wt, cap)
    r = _qkp(Q, wt, cap).solve(time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(opt, abs=1e-6)
    assert r.bound >= opt - 1e-6
    assert "spatial-bb" not in (r.algorithm_route or "")
