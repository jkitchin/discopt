"""#1836: box RO sign tracking must not pass a sign through a non-monotone node.

``sign_tracking_substitute`` substitutes each uncertain parameter at the end of
its box that maximizes the row, carrying a sign down the tree. It flipped the
sign only for negation, subtraction and constant multipliers; ``1/p``, ``p**k``,
``abs`` and every ``FunctionCall`` passed the parent's sign through unchanged.
Over ``x in [0, 20]``, maximize ``x``:

* ``x + 1/p <= 10``, ``p in [0.5, 1.5]``: returned ``9.33`` (truth ``8``);
* ``x + p**(-1) <= 10``: the same;
* ``x + abs(p) <= 10``, ``p in [-2, 1]``: returned ``9`` (truth ``8``);
* ``x + p**2 <= 10``, ``p in [-2, 1]``: returned ``9`` (truth ``6``).

Each node now either proves monotonicity over the box (and the variable bounds)
and propagates the right sign, takes the exact interval bound of a var-free
subtree, decomposes an even function through ``|u|``, or refuses with
``NotImplementedError``. Every case here is checked against an independent
oracle: a dense grid over the box, with the robust ``x`` found by bisection, and
the returned point is verified feasible at every grid point.
"""

from __future__ import annotations

import types

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.ro import BoxUncertaintySet, RobustCounterpart

pytestmark = pytest.mark.smoke

NP = types.SimpleNamespace(
    exp=np.exp, log=np.log, sqrt=np.sqrt, fabs=np.abs, sin=np.sin, cos=np.cos,
    cosh=np.cosh, sum=np.sum, maximum=np.maximum, acos=np.arccos,
)  # fmt: skip
DM = types.SimpleNamespace(
    exp=dm.exp, log=dm.log, sqrt=dm.sqrt, fabs=dm.abs, sin=dm.sin, cos=dm.cos,
    cosh=dm.cosh, sum=dm.sum, maximum=dm.maximum, acos=dm.acos,
)  # fmt: skip

# name: (g(p, ns), pbar, delta). Rows and objectives below are built from g.
G = {
    "inv": (lambda p, n: 1.0 / p, 1.0, 0.5),
    "pow_m1": (lambda p, n: p ** (-1.0), 1.0, 0.5),
    "abs_indef": (lambda p, n: abs(p), -0.5, 1.5),
    "fabs_indef": (lambda p, n: n.fabs(p), -0.5, 1.5),
    "abs_pos": (lambda p, n: abs(p), 1.0, 0.5),
    "abs_neg": (lambda p, n: abs(p), -1.0, 0.5),
    "sq_indef": (lambda p, n: p**2, -0.5, 1.5),
    "sq_pos": (lambda p, n: p**2, 1.0, 0.5),
    "sq_neg": (lambda p, n: p**2, -1.0, 0.5),
    "same_object_square": (lambda p, n: p * p, -0.5, 1.5),
    "neg_sq": (lambda p, n: -(p**2), -0.5, 1.5),
    "cube": (lambda p, n: p**3, -0.5, 1.5),
    "pow_m2_pos": (lambda p, n: p ** (-2), 1.0, 0.5),
    "pow_m2_neg": (lambda p, n: p ** (-2), -1.0, 0.5),
    "pow_m3_neg": (lambda p, n: p ** (-3), -1.0, 0.5),
    "pow_half": (lambda p, n: p**0.5, 1.0, 0.5),
    "exp": (lambda p, n: n.exp(p), 0.0, 1.0),
    "exp_neg": (lambda p, n: n.exp(-p), 0.0, 1.0),
    "log": (lambda p, n: n.log(p), 1.0, 0.5),
    "sqrt": (lambda p, n: n.sqrt(p), 1.0, 0.5),
    "cosh_indef": (lambda p, n: n.cosh(p), -0.5, 1.5),
    "sin": (lambda p, n: n.sin(p), 1.0, 1.0),
    "cos": (lambda p, n: n.cos(p), 0.0, 1.0),
    "acos": (lambda p, n: n.acos(p), 0.0, 0.5),
    "exp_of_sq": (lambda p, n: n.exp(p**2), -0.5, 1.5),
    "inv_of_sq_plus_1": (lambda p, n: 1.0 / (p**2 + 1.0), -0.5, 1.5),
    "c_times_inv": (lambda p, n: 3.0 * (1.0 / p), 1.0, 0.5),
    "negc_times_inv": (lambda p, n: -3.0 * (1.0 / p), 1.0, 0.5),
    "neg_numerator_div": (lambda p, n: -2.0 / p, 1.0, 0.5),
    "minus_inv": (lambda p, n: 0.0 - 1.0 / p, 1.0, 0.5),
    "inv_negbox": (lambda p, n: 1.0 / p, -1.0, 0.5),
    "two_pow_p": (lambda p, n: 2.0**p, 0.0, 1.0),
    "half_pow_p": (lambda p, n: 0.5**p, 0.0, 1.0),
    "max_p_0": (lambda p, n: n.maximum(p, 0.0), -0.5, 1.0),
    # Affine rows: already correct, locked here.
    "lin_3p": (lambda p, n: 3.0 * p, 1.0, 0.5),
    "lin_negp": (lambda p, n: -p, 1.0, 0.5),
    "vec_sum_inv": (lambda p, n: n.sum(1.0 / p), np.array([1.0, 2.0]), np.array([0.5, 0.5])),
    "vec_sum_sq": (lambda p, n: n.sum(p**2), np.array([-0.5, 0.5]), np.array([1.0, 1.0])),
    "vec_mixed_c_inv": (
        lambda p, n: n.sum(np.array([3.0, -1.0]) * (1.0 / p)),
        np.array([1.0, 2.0]),
        np.array([0.5, 0.5]),
    ),
    "vec_index_abs_sq": (
        lambda p, n: abs(p[0]) + p[1] ** 2,
        np.array([-0.5, 0.5]),
        np.array([1.0, 1.0]),
    ),
}

# Row shapes over g, each h(x, g) <= 0 and increasing in x.
KINDS = {
    "le": (lambda x, g: x + g <= 10.0, lambda x, g: x + g - 10.0),
    "ge": (lambda x, g: g - x >= -10.0, lambda x, g: x - g - 10.0),
}

# Rows with x inside the nonlinearity: (h(x, p, ns), pbar, delta), h <= 0.
H = {
    "exp_x_plus_p": (lambda x, p, n: n.exp(x + p) - 1e4, 1.0, 0.5),
    "sq_x_plus_p_indef": (lambda x, p, n: (x + p) ** 2 - 10.0, -1.0, 1.5),
    "x_div_p": (lambda x, p, n: x / p - 10.0, 1.0, 0.5),
    "x_times_inv": (lambda x, p, n: x * (1.0 / p) - 10.0, 1.0, 0.5),
    "x_times_sq": (lambda x, p, n: x * p**2 - 10.0, -0.5, 1.5),
    "x_times_same_object_square": (lambda x, p, n: x * (p * p) - 10.0, -0.5, 1.5),
    "abs_x_minus_p": (lambda x, p, n: abs(x - p) - 3.0, 1.0, 0.5),
    "sqrt_x_plus_p": (lambda x, p, n: n.sqrt(x + p) - 3.0, 1.0, 0.5),
    "x_plus_sin_small": (lambda x, p, n: x + n.sin(0.01 * x + p) - 10.0, 0.0, 0.5),
    "x_cos_p": (lambda x, p, n: x * n.cos(p) - 5.0, 0.0, 1.0),
}

# Rows the formulation cannot prove sound and must refuse.
REFUSED = {
    "inv_box_contains_zero": (lambda x, p: x + 1.0 / p <= 10.0, 0.2, 0.5),
    "x_div_p_box_contains_zero": (lambda x, p: x / p <= 10.0, 0.2, 0.5),
    "sign_indefinite_factor_times_sq": (lambda x, p: x + (x - 5.0) * p**2 <= 10.0, -0.5, 1.5),
    "sin_x_plus_p": (lambda x, p: dm.sin(x + p) + x <= 10.0, 1.0, 0.5),
    "p_pow_x": (lambda x, p: x + p ** (0.1 * x) <= 10.0, 2.0, 0.5),
}


def _grid(pbar, delta):
    pbar = np.atleast_1d(np.asarray(pbar, float))
    delta = np.atleast_1d(np.asarray(delta, float))
    npts = 1001 if pbar.size == 1 else 101
    axes = []
    for b, d in zip(pbar, delta):
        a = np.linspace(b - d, b + d, npts)
        if b - d < 0 < b + d:
            a = np.append(a, 0.0)
        axes.append(a)
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


def _values(fun, pts, scalar):
    with np.errstate(all="ignore"):
        return np.array([float(fun(float(pt[0]) if scalar else pt)) for pt in pts])


def _oracle_x(h, pts, scalar):
    """Largest x in [0, 20] with h(x, p) <= 0 on the whole grid (h increasing in x)."""

    def feas(x):
        v = _values(lambda p: h(x, p), pts, scalar)
        return bool(np.all(np.isfinite(v)) and np.max(v) <= 0)

    if feas(20.0):
        return 20.0
    if not feas(0.0):
        return None
    lo, hi = 0.0, 20.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if feas(mid) else (lo, mid)
    return lo


def _parameter(m, pbar):
    return m.parameter("p", value=np.asarray(pbar, float).copy() if np.ndim(pbar) else float(pbar))


def _check_row(build, h, pbar, delta):
    scalar = np.ndim(pbar) == 0
    pts = _grid(pbar, delta)
    m = dm.Model("ro")
    x = m.continuous("x", lb=0, ub=20)
    p = _parameter(m, pbar)
    m.subject_to(build(x, p))
    m.maximize(x)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=delta)).formulate()
    r = m.solve()
    xo = _oracle_x(h, pts, scalar)
    if xo is None:  # no robust point at all, e.g. exp(p**2) <= 10 - x
        assert r.status == "infeasible", r.status
        return None
    assert r.status == "optimal", r.status
    xv = float(r.value(x))
    viol = np.max(_values(lambda q: h(xv, q), pts, scalar))
    assert viol <= 1e-6, f"x={xv} is not robust: max violation {viol} on the grid"
    assert xv == pytest.approx(xo, abs=1e-4), f"x={xv}, robust optimum {xo}"
    return xv


@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("name", sorted(G))
def test_row_matches_grid_oracle(name, kind):
    g, pbar, delta = G[name]
    build, h = KINDS[kind]
    _check_row(lambda x, p: build(x, g(p, DM)), lambda x, p: h(x, g(p, NP)), pbar, delta)


@pytest.mark.parametrize("sense", ["min", "max"])
@pytest.mark.parametrize("name", sorted(G))
def test_uncertain_objective_matches_worst_case(name, sense):
    g, pbar, delta = G[name]
    scalar = np.ndim(pbar) == 0
    gv = _values(lambda q: g(q, NP), _grid(pbar, delta), scalar)
    m = dm.Model("ro")
    x = m.continuous("x", lb=1, ub=20)
    p = _parameter(m, pbar)
    if sense == "min":
        m.minimize(x + g(p, DM))
        truth = 1.0 + np.max(gv)  # min_x max_p
    else:
        m.maximize(g(p, DM) - x)
        truth = np.min(gv) - 1.0  # max_x min_p
    RobustCounterpart(m, BoxUncertaintySet(p, delta=delta)).formulate()
    r = m.solve()
    assert r.status == "optimal", r.status
    assert float(r.objective) == pytest.approx(truth, abs=1e-5)


@pytest.mark.parametrize("name", sorted(H))
def test_row_with_variable_inside_matches_grid_oracle(name):
    h, pbar, delta = H[name]
    _check_row(lambda x, p: h(x, p, DM) <= 0, lambda x, p: h(x, p, NP), pbar, delta)


@pytest.mark.parametrize("name", ["inv", "sq_indef", "abs_indef", "cube", "lin_3p"])
def test_linearization_path_matches_grid_oracle(name):
    """A bilinear ``y * (p - pbar)`` row sends the model down the |coeff| path,
    whose finite-difference coefficients are exact only for jointly affine rows."""
    g, pbar, delta = G[name]
    build, h = KINDS["le"]
    pts = _grid(pbar, delta)
    m = dm.Model("ro")
    x = m.continuous("x", lb=0, ub=20)
    y = m.continuous("y", lb=-1, ub=1)
    p = _parameter(m, pbar)
    m.subject_to(build(x, g(p, DM)))
    m.subject_to(y * (p - pbar) <= 1)
    m.maximize(x)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=delta)).formulate()
    r = m.solve()
    assert r.status == "optimal", r.status
    xv = float(r.value(x))
    assert np.max(_values(lambda q: h(xv, g(q, NP)), pts, True)) <= 1e-6
    assert xv == pytest.approx(_oracle_x(lambda x, q: h(x, g(q, NP)), pts, True), abs=1e-4)


@pytest.mark.parametrize(
    "build, pbar, delta, truth",
    [
        (lambda x, p: x + 1.0 / p <= 10, 1.0, 0.5, 8.0),
        (lambda x, p: x + p ** (-1.0) <= 10, 1.0, 0.5, 8.0),
        (lambda x, p: x + abs(p) <= 10, -0.5, 1.5, 8.0),
        (lambda x, p: x + p**2 <= 10, -0.5, 1.5, 6.0),
    ],
    ids=["inv", "pow_m1", "abs", "square"],
)
def test_issue_repros(build, pbar, delta, truth):
    m = dm.Model("ro")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=pbar)
    m.subject_to(build(x, p))
    m.maximize(x)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=delta)).formulate()
    r = m.solve()
    assert r.status == "optimal"
    assert float(r.value(x)) == pytest.approx(truth, abs=1e-6)


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_unprovable_rows_refuse(name):
    build, pbar, delta = REFUSED[name]
    m = dm.Model("ro")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=pbar)
    m.subject_to(build(x, p))
    m.maximize(x)
    with pytest.raises(NotImplementedError, match="#1836"):
        RobustCounterpart(m, BoxUncertaintySet(p, delta=delta)).formulate()
