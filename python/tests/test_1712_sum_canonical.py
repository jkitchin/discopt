"""#1712: ``DISCOPT_SUM_CANONICAL`` folds per-element linear sums into ``dm.sum(C * x)``.

The rewrite is exact up to summation order: it fires only when a variable's terms
name every entry once, so the folded sum holds exactly the products written.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt._relax.dag_compiler import compile_expression
from discopt.modeling.core import (
    _SUM_CANONICAL_MIN_TERMS,
    BinaryOp,
    Constant,
    SumExpression,
    SumOverExpression,
    Variable,
    affine_array_sum,
)

N = _SUM_CANONICAL_MIN_TERMS + 8


@pytest.fixture
def canonical_on(monkeypatch):
    monkeypatch.setenv("DISCOPT_SUM_CANONICAL", "1")


def _flat_point(m: Model, rng) -> np.ndarray:
    return rng.normal(size=sum(v.size for v in m._variables)) * 3.0


def _value(expr, m, x):
    return float(compile_expression(expr, m)(x))


@pytest.mark.smoke
def test_full_cover_folds_to_vectorized_form(canonical_on):
    rng = np.random.default_rng(0)
    m = Model("f")
    x = m.continuous("x", shape=(N,), lb=-1, ub=1)
    c = rng.normal(size=N)
    order = rng.permutation(N)  # any order: coverage, not order, is what counts
    e = dm.sum(c[i] * x[i] for i in order)
    assert isinstance(e, SumExpression) and e.axis is None
    prod = e.operand
    assert isinstance(prod, BinaryOp) and prod.op == "*"
    assert isinstance(prod.left, Constant) and prod.right is x
    assert np.array_equal(prod.left.value, c)


@pytest.mark.smoke
def test_folded_value_equals_written_sum(canonical_on, monkeypatch):
    rng = np.random.default_rng(1)
    compared = 0
    for _ in range(20):
        m = Model("v")
        x = m.continuous("x", shape=(4, N // 4 + 1), lb=-5, ub=5)
        y = m.continuous("y", shape=(3,), lb=0, ub=2)
        keys = [(i, j) for i in range(x.shape[0]) for j in range(x.shape[1])]
        c = rng.normal(size=len(keys)) * 10.0 ** rng.integers(-3, 4, size=len(keys))

        def terms():
            out = []
            for (i, j), ck in zip(keys, c):
                k = rng.integers(4)
                out.append([ck * x[i, j], x[i, j] * ck, x[i, j], -x[i, j]][k])
            out += [y[0] * y[1], dm.exp(y[2]), 2.5]  # kept as written
            return out

        written = terms()
        folded = dm.sum(t for t in written)
        with monkeypatch.context() as mp:
            mp.setenv("DISCOPT_SUM_CANONICAL", "0")
            plain = dm.sum(t for t in written)
        assert isinstance(plain, SumOverExpression) and len(plain.terms) == len(written)
        assert isinstance(folded, SumOverExpression) and len(folded.terms) == 4
        assert isinstance(folded.terms[0], SumExpression)
        pt = _flat_point(m, rng)
        a, b = _value(folded, m, pt), _value(plain, m, pt)
        scale = sum(abs(_value(t, m, pt)) for t in plain.terms)
        assert abs(a - b) <= 1e-12 * max(1.0, scale)
        compared += 1
    assert compared == 20


@pytest.mark.parametrize(
    "case",
    ["partial_cover", "duplicate", "parameter_coef", "too_few_terms", "slice_index"],
)
def test_not_folded(case, canonical_on):
    m = Model("n")
    x = m.continuous("x", shape=(N,), lb=-1, ub=1)
    if case == "partial_cover":
        terms = [2.0 * x[i] for i in range(N - 1)]
    elif case == "duplicate":
        terms = [2.0 * x[i] for i in range(N - 1)] + [x[0]]
    elif case == "parameter_coef":
        p = m.parameter("p", 3.0)
        terms = [p * x[i] for i in range(N)]
    elif case == "too_few_terms":
        z = m.continuous("z", shape=(_SUM_CANONICAL_MIN_TERMS - 1,))
        terms = [z[i] for i in range(_SUM_CANONICAL_MIN_TERMS - 1)]
    else:
        terms = [x[0:2]] + [x[i] for i in range(2, N)]
    e = dm.sum(t for t in terms)
    assert isinstance(e, SumOverExpression) and len(e.terms) == len(terms)


def test_parameter_coefficient_stays_live(canonical_on):
    """A Parameter re-valued after the build still changes the objective."""
    m = Model("p")
    x = m.continuous("x", shape=(N,), lb=0, ub=1)
    p = m.parameter("p", 1.0)
    m.maximize(dm.sum(p * x[i] for i in range(N)))
    assert m.solve().objective == pytest.approx(N)
    p.value = 2.0
    assert m.solve().objective == pytest.approx(2 * N)


@pytest.mark.smoke
def test_default_is_on_and_zero_opts_out(monkeypatch):
    """Graduated (#1712): the per-element sum folds by default; ``=0`` keeps it."""
    monkeypatch.delenv("DISCOPT_SUM_CANONICAL", raising=False)
    m = Model("default")
    x = m.continuous("x", shape=(N,))
    assert affine_array_sum(dm.sum(2.0 * x[i] for i in range(N))) is not None
    monkeypatch.setenv("DISCOPT_SUM_CANONICAL", "0")
    e = dm.sum(2.0 * x[i] for i in range(N))
    assert isinstance(e, SumOverExpression) and len(e.terms) == N


def _transport(n: int, integer: bool):
    rng = np.random.default_rng(n)
    a, b, c = rng.uniform(50, 100, n), rng.uniform(10, 40, n), rng.uniform(1, 10, (n, n))
    m = Model("t")
    P = m.set("P", [f"p{k}" for k in range(n)])
    K = m.set("K", [f"m{k}" for k in range(n)])
    ip, ik = {p: i for i, p in enumerate(P)}, {k: j for j, k in enumerate(K)}
    x = (
        m.integer("x", over=P * K, lb=0, ub=100)
        if integer
        else m.continuous("x", over=P * K, lb=0, ub=1e4)
    )
    m.minimize(dm.sum(c[ip[p], ik[k]] * x[p, k] for p in P for k in K))
    m.constraint(P, lambda p: dm.sum(x[p, k] for k in K) <= a[ip[p]], name="sup")
    m.constraint(K, lambda k: dm.sum(x[p, k] for p in P) >= b[ik[k]], name="dem")
    return m


@pytest.mark.smoke
@pytest.mark.parametrize("integer", [False, True])
def test_solve_agrees_with_flag_off(integer, monkeypatch):
    """The issue's model: the folded objective solves to the same certified answer."""
    n = 7 if integer else 12
    monkeypatch.setenv("DISCOPT_SUM_CANONICAL", "0")
    off = _transport(n, integer).solve()
    monkeypatch.setenv("DISCOPT_SUM_CANONICAL", "1")
    m_on = _transport(n, integer)
    assert isinstance(m_on._objective.expression, SumExpression)  # the fold fired
    on = m_on.solve()
    assert on.status == off.status == "optimal"
    assert on.gap_certified and off.gap_certified
    assert on.objective == pytest.approx(off.objective, rel=1e-9, abs=1e-9)
    assert on.bound == pytest.approx(off.bound, rel=1e-9, abs=1e-9)
    assert on.constraint_duals.keys() == off.constraint_duals.keys()
    assert isinstance(m_on._objective.expression.operand.right, Variable)


def _bilinear(n: int, vectorized: bool) -> Model:
    rng = np.random.default_rng(n)
    c, d = rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)
    m = Model("bil")
    x = m.continuous("x", shape=(n,), lb=0, ub=2)
    y = m.continuous("y", shape=(n,), lb=0, ub=2)
    if vectorized:
        m.minimize(dm.sum(c * x) + dm.sum(d * y))
    else:
        m.minimize(dm.sum(c[i] * x[i] for i in range(n)) + dm.sum(d[i] * y[i] for i in range(n)))
    for i in range(0, n, 4):
        m.subject_to(x[i] * y[i] <= 1.5)
    m.subject_to(dm.sum(x) + dm.sum(y) >= 0.25 * n)
    return m


@pytest.mark.smoke
def test_array_constant_objective_with_bilinear_rows_solves(monkeypatch):
    """A hand-written ``dm.sum(c * x)`` objective beside a bilinear row raised
    ``TypeError`` in ``milp_relaxation`` (``float()`` of an array constant); the
    flag only made that path reachable from the per-element idiom (#1712)."""
    monkeypatch.setenv("DISCOPT_SUM_CANONICAL", "0")
    vec = _bilinear(8, vectorized=True).solve(time_limit=60)
    per = _bilinear(8, vectorized=False).solve(time_limit=60)
    assert vec.status == per.status == "optimal"
    assert vec.objective == pytest.approx(per.objective, rel=1e-6, abs=1e-6)
