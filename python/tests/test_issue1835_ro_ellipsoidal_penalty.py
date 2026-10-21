"""#1835: the ellipsoidal RO penalty uses the row's aggregated coefficient.

For a scalar row ``g(x, p) <= 0`` affine in ``p`` over ``{p_bar + Sigma^1/2 u : ||u|| <= rho}``
the robust counterpart is ``g(x, p_bar) + rho * sqrt(a(x)' Sigma a(x)) <= 0`` with
``a(x) = d g / d p`` over *every* occurrence of ``p``. The former implementation
attached an unscaled ``||Sigma^1/2 M||`` to each ``M @ p`` node, ignoring the outer
coefficient / sum / index, so ``x + 5 * (c @ p) <= 70`` certified ``x = 20`` (truth 10).

Oracle: the closed-form worst case of each row, computed from the row's own numpy
definition by exact finite differences (the row is affine in ``p``), checked against
a dense sample of the ellipsoid boundary; the reference optimum is an independent
SLSQP solve of the closed-form robust problem.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.ro import EllipsoidalUncertaintySet, RobustCounterpart
from scipy.optimize import minimize

pytestmark = pytest.mark.smoke

C = np.array([3.0, 4.0])
M = np.array([[1.0, 2.0], [-3.0, 0.5]])
SIG = np.array([[2.0, 1.0], [1.0, 1.0]])


def _worst(g, z, sets):
    """Closed-form ``max_p g(z, p)`` over the product of ellipsoids (``g`` affine in p)."""
    nom = {k: v[0] for k, v in sets.items()}
    base = g(z, nom)
    total = base
    for k, (pbar, rho, sig) in sets.items():
        a = np.zeros(pbar.size)
        for j in range(pbar.size):
            pp = dict(nom)
            pp[k] = pbar + np.eye(pbar.size)[j].reshape(pbar.shape)
            a[j] = g(z, pp) - base
        total = total + rho * np.sqrt(max(a @ sig @ a, 0.0))
    return total


def _sampled_max(g, z, sets, rng, n=2000):
    best = -np.inf
    for _ in range(n):
        pp = {}
        for k, (pbar, rho, sig) in sets.items():
            L = np.linalg.cholesky(sig)
            u = rng.normal(size=pbar.size)
            pp[k] = pbar + (L @ (rho * u / np.linalg.norm(u))).reshape(pbar.shape)
        best = max(best, g(z, pp))
    return best


def _reference(bounds, rows, obj, sense, sets, rng):
    sgn = 1.0 if sense == "min" else -1.0
    cons = [{"type": "ineq", "fun": (lambda z, g=g: -_worst(g, z, sets))} for g in rows]
    best = None
    for _ in range(20):
        z0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        r = minimize(
            lambda z: _worst(lambda zz, P: sgn * obj(zz, P), z, sets),
            z0,
            bounds=bounds,
            constraints=cons,
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if all(c["fun"](r.x) >= -1e-7 for c in cons) and (best is None or r.fun < best):
            best = r.fun
    assert best is not None
    return sgn * best


def _case(name):
    """Case *name*: (model, param, vars, bounds, rows, obj, sense, (pbar, rho, Sigma))."""
    m = dm.Model(name)
    I2 = np.eye(2)
    if name == "repro1_scaled":
        x = m.continuous("x", lb=0, ub=20)
        p = m.parameter("p", value=np.ones(2))
        m.subject_to(x + 5.0 * (C @ p) <= 70)
        m.maximize(x)
        rows = [lambda z, P: z[0] + 5.0 * (C @ P["p"]) - 70]
        return m, p, [x], [(0, 20)], rows, lambda z, P: z[0], "max", (np.ones(2), 1.0, I2)
    if name == "repro2_sum_of_matmul":
        x = m.continuous("x", lb=0, ub=20)
        p = m.parameter("p", value=np.ones(2))
        m.subject_to(x + dm.sum(np.ones((2, 2)) @ p) <= 10)
        m.maximize(x)
        rows = [lambda z, P: z[0] + np.sum(np.ones((2, 2)) @ P["p"]) - 10]
        return m, p, [x], [(0, 20)], rows, lambda z, P: z[0], "max", (np.ones(2), 1.0, I2)
    if name == "repro3_vector_rows":
        x = m.continuous("x", lb=0, ub=20)
        p = m.parameter("p", value=np.ones(2))
        m.subject_to(x + M @ p <= np.array([10.0, 8.0]))
        m.maximize(x)
        rows = [
            lambda z, P: z[0] + (M @ P["p"])[0] - 10,
            lambda z, P: z[0] + (M @ P["p"])[1] - 8,
        ]
        return m, p, [x], [(0, 20)], rows, lambda z, P: z[0], "max", (np.ones(2), 1.0, I2)
    if name == "negated_scaled":
        x = m.continuous("x", lb=0, ub=50)
        p = m.parameter("p", value=np.ones(2))
        m.subject_to(x - 2.0 * (C @ p) <= 10)
        m.maximize(x)
        rows = [lambda z, P: z[0] - 2.0 * (C @ P["p"]) - 10]
        return m, p, [x], [(0, 50)], rows, lambda z, P: z[0], "max", (np.ones(2), 1.0, I2)
    if name == "indexed_matmul":
        x = m.continuous("x", lb=0, ub=50)
        p = m.parameter("p", value=np.ones(2))
        m.subject_to(x + (M @ p)[0] <= 10)
        m.maximize(x)
        rows = [lambda z, P: z[0] + (M @ P["p"])[0] - 10]
        return m, p, [x], [(0, 50)], rows, lambda z, P: z[0], "max", (np.ones(2), 0.5, I2)
    if name == "scalar_p_times_x":
        x = m.continuous("x", lb=0, ub=10)
        p = m.parameter("p", value=1.0)
        m.subject_to(p * x <= 4)
        m.maximize(x)
        rows = [lambda z, P: float(P["p"]) * z[0] - 4]
        return m, p, [x], [(0, 10)], rows, lambda z, P: z[0], "max", (np.array(1.0), 0.5, np.eye(1))
    x = m.continuous("x", shape=2, lb=0, ub=5)
    pbar = np.array([1.0, 2.0])
    p = m.parameter("p", value=pbar)
    box = [(0, 5)] * 2
    if name == "two_occurrences":
        m.subject_to(p @ x + 2.0 * (p @ x) <= 3)
        m.maximize(dm.sum(x))
        rows = [lambda z, P: 3.0 * (P["p"] @ z) - 3]
        return m, p, [x], box, rows, lambda z, P: z.sum(), "max", (pbar, 0.3, I2)
    if name == "sigma_correlated":
        m.subject_to(p @ x <= 2)
        m.maximize(x[0] + 0.5 * x[1])
        rows = [lambda z, P: P["p"] @ z - 2]
        return m, p, [x], box, rows, lambda z, P: z[0] + 0.5 * z[1], "max", (pbar, 0.5, SIG)
    if name == "objective_min_scaled":
        m.subject_to(dm.sum(x) >= 1)
        m.minimize(3.0 * (p @ x))
        rows = [lambda z, P: 1 - z.sum()]
        return m, p, [x], box, rows, lambda z, P: 3.0 * (P["p"] @ z), "min", (pbar, 0.5, I2)
    if name == "objective_max_negscaled":
        m.subject_to(dm.sum(x) >= 1)
        m.maximize(-2.0 * (p @ x))
        rows = [lambda z, P: 1 - z.sum()]
        return m, p, [x], box, rows, lambda z, P: -2.0 * (P["p"] @ z), "max", (pbar, 0.5, I2)
    if name == "ge_row_scaled":
        m.subject_to(2.0 * (p @ x) >= 4)
        m.minimize(dm.sum(x))
        rows = [lambda z, P: 4 - 2.0 * (P["p"] @ z)]
        return m, p, [x], box, rows, lambda z, P: z.sum(), "min", (pbar, 0.4, I2)
    if name == "sum_elementwise_scaled":
        m.subject_to(3.0 * dm.sum(p * x) <= 3)
        m.maximize(dm.sum(x))
        rows = [lambda z, P: 3.0 * np.sum(P["p"] * z) - 3]
        return m, p, [x], box, rows, lambda z, P: z.sum(), "max", (pbar, 0.3, I2)
    if name == "x_matmul_M_matmul_p":
        m.subject_to(x @ (M @ p) <= 3)
        m.maximize(dm.sum(x))
        rows = [lambda z, P: z @ (M @ P["p"]) - 3]
        return m, p, [x], box, rows, lambda z, P: z.sum(), "max", (pbar, 0.3, I2)
    if name == "divided":
        m.subject_to((p @ x) / 0.5 <= 3)
        m.maximize(dm.sum(x))
        rows = [lambda z, P: (P["p"] @ z) / 0.5 - 3]
        return m, p, [x], box, rows, lambda z, P: z.sum(), "max", (pbar, 0.3, I2)
    if name == "x_dependent_vector_coeff":
        # a(x) = x[0] * x: the coefficient of p depends on x nonlinearly.
        m.subject_to(x[0] * (p @ x) + x[1] <= 4)
        m.maximize(x[0] + x[1])
        rows = [lambda z, P: z[0] * (P["p"] @ z) + z[1] - 4]
        return m, p, [x], box, rows, lambda z, P: z[0] + z[1], "max", (pbar, 0.3, I2)
    if name == "matrix_param_rows":
        Pb = np.array([[1.0, 2.0], [2.0, 1.0]])
        Pm = m.parameter("P", value=Pb)
        m.subject_to(Pm @ x <= np.array([3.0, 4.0]))
        m.maximize(dm.sum(x))
        rows = [lambda z, Q: (Q["P"] @ z)[0] - 3, lambda z, Q: (Q["P"] @ z)[1] - 4]
        return m, Pm, [x], box, rows, lambda z, Q: z.sum(), "max", (Pb, 0.3, np.eye(4))
    if name == "matrix_param_correlated":
        Pb = np.array([[1.0, 2.0], [2.0, 1.0]])
        Pm = m.parameter("P", value=Pb)
        A = np.array([[1.0, 0.3, 0, 0], [0.3, 1.0, 0.2, 0], [0, 0.2, 1.0, 0.1], [0, 0, 0.1, 1.0]])
        m.subject_to(Pm @ x <= np.array([3.0, 4.0]))
        m.maximize(dm.sum(x))
        rows = [lambda z, Q: (Q["P"] @ z)[0] - 3, lambda z, Q: (Q["P"] @ z)[1] - 4]
        return m, Pm, [x], box, rows, lambda z, Q: z.sum(), "max", (Pb, 0.3, A)
    raise KeyError(name)


CASES = [
    "repro1_scaled",
    "repro2_sum_of_matmul",
    "repro3_vector_rows",
    "negated_scaled",
    "indexed_matmul",
    "scalar_p_times_x",
    "two_occurrences",
    "sigma_correlated",
    "objective_min_scaled",
    "objective_max_negscaled",
    "ge_row_scaled",
    "sum_elementwise_scaled",
    "x_matmul_M_matmul_p",
    "divided",
    "x_dependent_vector_coeff",
    "matrix_param_rows",
    "matrix_param_correlated",
]


@pytest.mark.parametrize("name", CASES)
def test_counterpart_matches_closed_form_worst_case(name):
    rng = np.random.default_rng(1835)
    m, p, vars_, bounds, rows, obj, sense, (pbar, rho, sig) = _case(name)
    sets = {p.name: (np.asarray(pbar, float), rho, sig)}
    ref = _reference(bounds, rows, obj, sense, sets, rng)

    RobustCounterpart(m, EllipsoidalUncertaintySet(p, rho=rho, Sigma=sig)).formulate()
    r = m.solve()
    assert r.status == "optimal"
    assert r.objective == pytest.approx(ref, rel=1e-4, abs=1e-5)

    z = np.concatenate([np.atleast_1d(np.asarray(r.value(v), float)) for v in vars_])
    checks = 0
    for g in rows:
        wc = _worst(g, z, sets)
        assert _sampled_max(g, z, sets, rng) <= wc + 1e-8  # the oracle itself
        assert wc <= 1e-5, f"returned point is not robust: worst case {wc}"
        checks += 2
    assert checks > 0


def test_issue_repro_values():
    """The three numbers in #1835, stated in closed form."""
    for name, truth in [
        ("repro1_scaled", 10.0),
        ("repro2_sum_of_matmul", 10.0 - 4.0 - np.sqrt(8.0)),
        ("repro3_vector_rows", 10.0 - 3.0 - np.sqrt(5.0)),
    ]:
        m, p, *_ = _case(name)
        RobustCounterpart(m, EllipsoidalUncertaintySet(p, rho=1.0)).formulate()
        assert m.solve().objective == pytest.approx(truth, abs=1e-5), name


def test_two_independent_parameters_add_their_worst_cases():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=50)
    p = m.parameter("p", value=np.ones(2))
    q = m.parameter("q", value=np.array([2.0]))
    m.subject_to(x + 2.0 * (C @ p) - 3.0 * q[0] <= 20)
    m.maximize(x)
    RobustCounterpart(
        m,
        [EllipsoidalUncertaintySet(p, rho=0.5), EllipsoidalUncertaintySet(q, rho=0.2)],
    ).formulate()
    # 20 - 14 + 6 - 0.5 * ||2c|| - 0.2 * 3 = 12 - 5 - 0.6
    assert m.solve().objective == pytest.approx(6.4, abs=1e-5)


def test_equality_row_requires_vanishing_coefficient():
    """``y + p @ x == 1`` holds for every p only at x = 0, y = 1."""
    m = dm.Model()
    x = m.continuous("x", shape=2, lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=2)
    p = m.parameter("p", value=np.array([1.0, 2.0]))
    m.subject_to(y + p @ x == 1)
    m.maximize(0.1 * dm.sum(x) + y)
    RobustCounterpart(m, EllipsoidalUncertaintySet(p, rho=0.2)).formulate()
    r = m.solve()
    # The two robust rows meet at the norm's kink, so the solver may not certify
    # the bound; whatever it returns must be the robust point.
    assert r.objective == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(r.value(x), 0.0, atol=1e-6)


def test_equality_row_with_constant_coefficient_is_infeasible():
    m = dm.Model()
    x = m.continuous("x", lb=-50, ub=50)
    p = m.parameter("p", value=np.ones(2))
    m.subject_to(x + C @ p == 5)
    m.maximize(x)
    RobustCounterpart(m, EllipsoidalUncertaintySet(p, rho=1.0)).formulate()
    assert m.solve().status == "infeasible"


@pytest.mark.parametrize(
    "build",
    [
        lambda x, p: (p @ x) * (p @ x) <= 4,
        lambda x, p: dm.exp(p @ x) <= 4,
        lambda x, p: dm.sum(x / p) <= 4,
        lambda x, p: dm.sum(p**2 * x) <= 4,
        lambda x, p: dm.sum(abs(p) * x) <= 4,
    ],
    ids=["p_times_p", "exp", "p_in_divisor", "power", "abs"],
)
def test_row_not_affine_in_p_is_refused(build):
    m = dm.Model()
    x = m.continuous("x", shape=2, lb=0.1, ub=5)
    p = m.parameter("p", value=np.array([1.0, 2.0]))
    m.subject_to(build(x, p), name="nl")
    m.maximize(dm.sum(x))
    with pytest.raises(NotImplementedError, match="not supported"):
        RobustCounterpart(m, EllipsoidalUncertaintySet(p, rho=0.3)).formulate()


def test_objective_not_affine_in_p_is_refused():
    m = dm.Model()
    x = m.continuous("x", shape=2, lb=0, ub=5)
    p = m.parameter("p", value=np.array([1.0, 2.0]))
    m.subject_to(dm.sum(x) <= 1)
    m.minimize(dm.exp(p @ x))
    with pytest.raises(NotImplementedError, match="objective"):
        RobustCounterpart(m, EllipsoidalUncertaintySet(p, rho=0.3)).formulate()


def test_canonical_matmul_row_is_formulated_unchanged():
    """Bound-neutral lock: ``p @ x`` keeps the exact pre-#1835 counterpart node."""
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        FunctionCall,
        MatMulExpression,
        SumExpression,
        Variable,
    )

    for build in (lambda p, x: p @ x, lambda p, x: -(p @ x), lambda p, x: x @ p):
        m = dm.Model()
        x = m.continuous("x", shape=3, lb=0, ub=1)
        p = m.parameter("p", value=np.array([0.1, 0.2, 0.3]))
        m.minimize(build(p, x))
        unc = EllipsoidalUncertaintySet(p, rho=2.0)
        RobustCounterpart(m, unc).formulate()
        expr = m._objective.expression
        assert isinstance(expr, BinaryOp) and expr.op == "+"
        pen = expr.right
        assert isinstance(pen, BinaryOp) and isinstance(pen.left, Constant)
        assert float(pen.left.value) == 2.0
        sqrt = pen.right
        assert isinstance(sqrt, FunctionCall) and sqrt.func_name == "sqrt"
        inner = sqrt.args[0]
        assert isinstance(inner, SumExpression) and inner.axis is None
        scaled = inner.operand.left
        assert isinstance(scaled, MatMulExpression)
        np.testing.assert_array_equal(scaled.left.value, unc.Sigma_sqrt)
        assert isinstance(scaled.right, Variable) and scaled.right.name == "x"
