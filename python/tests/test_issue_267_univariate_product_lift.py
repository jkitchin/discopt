"""Issue #267: lift univariate-function products instead of dropping them.

The MILP relaxation builder relaxes a *single* univariate transcendental
(``sin(x)``, ``exp(x)``, ``log(x)``, ``sqrt(x)``, …) fine, but when such a
function appears as a *factor of a product* — ``sin(x0) * cos(x0 - x0*x0)`` in
inscribedsquare02 — the factor was never lifted to an aux column. The product
then failed to decompose and the whole constraint was dropped from the
relaxation, leaving no valid dual bound.

The fix lifts any liftable univariate-function factor (recursively lifting its
inner argument when that argument is itself nonlinear, e.g. ``x - x*x``) to its
own aux column, after which the existing bilinear/trilinear McCormick machinery
relaxes the product of aux columns. These tests assert the general behaviour
across multiple atom combinations:

* the product term is no longer omitted from the relaxation;
* the emitted relaxation is a *valid outer approximation* — for many sampled
  points the true product (with the true auxiliary-column values) satisfies
  every relaxation row, so no true-feasible point is cut off (soundness);
* inscribedsquare02 returns a finite, valid dual bound.
"""

from __future__ import annotations

import logging

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.discretization import DiscretizationState
from discopt._jax.milp_relaxation import build_milp_relaxation
from discopt._jax.term_classifier import classify_nonlinear_terms
from discopt.modeling.core import Model

pytestmark = [pytest.mark.claim_boundary]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build(model: Model):
    terms = classify_nonlinear_terms(model)
    relax, info = build_milp_relaxation(model, terms, DiscretizationState())
    return relax, info, terms


def _aux_value(func_name: str, arg: float) -> float:
    return {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
    }[func_name](arg)


def _column_value(info, flat, sample):
    """Return the *true* value of MILP column ``flat`` at a sampled point.

    ``sample`` maps original flat indices -> value. Univariate aux columns are
    evaluated from their stored ``arg_coeff @ z + arg_const`` over already-known
    columns; bilinear/trilinear/monomial aux columns are the product of their
    factor columns. This reconstructs the *exact* lifted assignment a true
    feasible point induces, which a sound relaxation must admit.
    """
    # Resolve columns iteratively until every referenced column has a value.
    values: dict[int, float] = dict(sample)

    uni_by_col = {r.aux_col: r for r in info["univariate_relaxations"]}
    bilinear = info["bilinear"]
    monomial = info["monomial"]
    trilinear = info["trilinear"]
    multilinear = info["multilinear"]
    # invert maps: aux col -> factor columns
    bil_factors = {col: key for key, col in bilinear.items()}
    mono_factors = {col: key for key, col in monomial.items()}
    tri_factors = {col: key for key, col in trilinear.items()}
    multi_factors = {col: key for key, col in multilinear.items()}

    def resolve(col: int) -> float:
        if col in values:
            return values[col]
        if col in uni_by_col:
            r = uni_by_col[col]
            arg = float(r.arg_const)
            for j in np.flatnonzero(r.arg_coeff):
                arg += float(r.arg_coeff[j]) * resolve(int(j))
            val = _aux_value(r.func_name, arg)
            values[col] = val
            return val
        if col in bil_factors:
            i, j = bil_factors[col]
            val = resolve(i) * resolve(j)
            values[col] = val
            return val
        if col in mono_factors:
            base, n = mono_factors[col]
            val = resolve(base) ** n
            values[col] = val
            return val
        if col in tri_factors:
            i, j, k = tri_factors[col]
            val = resolve(i) * resolve(j) * resolve(k)
            values[col] = val
            return val
        if col in multi_factors:
            val = 1.0
            for c in multi_factors[col]:
                val *= resolve(c)
            values[col] = val
            return val
        raise KeyError(f"cannot resolve column {col}")

    return resolve(flat)


def _assert_relaxation_encloses(relax, info, n_orig, samplers, n_samples=3000, seed=0):
    """Sample the box; assert the true point (with true aux values) satisfies
    every relaxation inequality. This is the rigorous enclosure / soundness
    check: a valid relaxation never cuts off a true-feasible point."""
    rng = np.random.default_rng(seed)
    A = relax._A_ub
    b = np.asarray(relax._b_ub, dtype=np.float64)
    ncol = A.shape[1]
    max_violation = 0.0
    for _ in range(n_samples):
        sample = {idx: float(s(rng)) for idx, s in samplers.items()}
        z = np.zeros(ncol)
        for col in range(ncol):
            try:
                z[col] = _column_value(info, col, sample)
            except KeyError:
                # Columns the test does not model (selector binaries, unrelated
                # auxes) are left at zero; the rows that reference them are not
                # part of the product envelope under test. Skip rows touching
                # such columns below by masking.
                z[col] = np.nan
        lhs = A.dot(np.nan_to_num(z, nan=0.0))
        # Mask rows that reference an unresolved (NaN) column.
        nan_cols = np.where(np.isnan(z))[0]
        if len(nan_cols):
            referenced = np.asarray((A[:, nan_cols] != 0).sum(axis=1)).ravel()
            active = referenced == 0
        else:
            active = np.ones(A.shape[0], dtype=bool)
        viol = float(np.max((lhs - b)[active])) if active.any() else 0.0
        max_violation = max(max_violation, viol)
    assert max_violation <= 1e-6, (
        f"relaxation cuts off a true point (violation {max_violation:.3e})"
    )


# ---------------------------------------------------------------------------
# 1. The product term is no longer dropped (single vs product)
# ---------------------------------------------------------------------------


def test_single_univariate_constraint_linearizes():
    m = Model()
    x = m.continuous("x", lb=-3.0, ub=3.0)
    m.minimize(x)
    m.subject_to(dm.sin(x) <= 0.5)
    relax, info, _ = _build(m)
    # exactly the sin aux column, no product needed.
    assert any(r.func_name == "sin" for r in info["univariate_relaxations"])


@pytest.mark.parametrize(
    "make_constraint,desc",
    [
        (lambda x, y: dm.sin(x) * dm.cos(x - x * x), "sin*cos(x-x*x)"),
        (lambda x, y: dm.exp(x) * dm.log(y), "exp*log"),
        (lambda x, y: dm.sqrt(x) * y * y, "sqrt*y*y"),
        (lambda x, y: dm.sin(x) * (1.0 / y), "sin*(1/y)"),
        (lambda x, y: dm.log(x) * x * x, "log*x*x"),
        (lambda x, y: dm.exp(x - x * x) * y, "exp(x-x*x)*y"),
        (lambda x, y: dm.sin(x) * dm.cos(y) * y, "sin*cos*y (3-factor)"),
    ],
)
def test_univariate_product_not_dropped(make_constraint, desc, caplog):
    """The product factor is lifted, so the constraint is NOT omitted."""
    m = Model()
    x = m.continuous("x", lb=0.5, ub=1.5)
    y = m.continuous("y", lb=0.5, ub=1.5)
    m.minimize(x + y)
    m.subject_to(make_constraint(x, y) <= 5.0)
    with caplog.at_level(logging.WARNING, logger="discopt._jax.milp_relaxation"):
        relax, info, _ = _build(m)
    omitted = [
        rec.message
        for rec in caplog.records
        if "omitting constraint" in rec.message or "cannot be linearized" in rec.message
    ]
    assert not omitted, f"{desc}: constraint was dropped: {omitted}"
    # at least one univariate aux + one product aux must have been allocated.
    assert info["univariate_relaxations"], f"{desc}: no univariate aux lifted"
    n_products = len(info["bilinear"]) + len(info["trilinear"]) + len(info["multilinear"])
    assert n_products >= 1, f"{desc}: no product envelope allocated"


# ---------------------------------------------------------------------------
# 2. Soundness: the relaxation encloses the true product (no true point cut off)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_constraint,xb,yb,desc",
    [
        (lambda x, y: dm.sin(x) * dm.cos(x - x * x), (-3.0, 3.0), (0.5, 1.5), "sin*cos(x-x*x)"),
        (lambda x, y: dm.exp(x) * dm.log(y), (0.5, 2.0), (0.5, 2.0), "exp*log"),
        (lambda x, y: dm.sqrt(x) * y * y, (0.5, 2.0), (0.5, 2.0), "sqrt*y*y"),
        (lambda x, y: dm.sin(x) * (1.0 / y), (-3.0, 3.0), (0.5, 2.0), "sin*(1/y)"),
        (lambda x, y: dm.log(x) * x * x, (0.5, 2.0), (0.5, 2.0), "log*x*x"),
        (lambda x, y: dm.exp(x - x * x) * y, (0.0, 1.5), (0.5, 2.0), "exp(x-x*x)*y"),
        (lambda x, y: dm.sin(x) * dm.cos(y) * y, (-2.0, 2.0), (0.5, 1.5), "sin*cos*y"),
    ],
)
def test_univariate_product_relaxation_is_sound(make_constraint, xb, yb, desc):
    m = Model()
    x = m.continuous("x", lb=xb[0], ub=xb[1])
    y = m.continuous("y", lb=yb[0], ub=yb[1])
    m.minimize(x + y)
    m.subject_to(make_constraint(x, y) <= 100.0)
    relax, info, _ = _build(m)
    n_orig = 2
    samplers = {
        0: (lambda r, lo=xb[0], hi=xb[1]: r.uniform(lo, hi)),
        1: (lambda r, lo=yb[0], hi=yb[1]: r.uniform(lo, hi)),
    }
    _assert_relaxation_encloses(relax, info, n_orig, samplers)


# ---------------------------------------------------------------------------
# 3. inscribedsquare02: finite, valid dual bound
# ---------------------------------------------------------------------------


def _inscribedsquare02_model():
    """The four sin*cos*poly equality constraints of inscribedsquare02 plus the
    objective, reconstructed in the modeling API so the test is self-contained
    (no dependency on the external benchmark .nl file)."""
    m = Model()
    xs = [m.continuous(f"x{i}", lb=-np.pi, ub=np.pi) for i in range(4)]
    x4 = m.continuous("x4", lb=0.0, ub=2.0)
    x5 = m.continuous("x5", lb=0.0, ub=2.0)
    x6 = m.continuous("x6", lb=-1.0, ub=1.0)
    x7 = m.continuous("x7", lb=-np.pi, ub=np.pi)
    m.maximize(x4**2 + x5**2)
    m.subject_to(dm.sin(xs[0]) * dm.cos(xs[0] - xs[0] * xs[0]) - x6 == 0)
    m.subject_to(dm.sin(xs[0]) * xs[0] - x7 == 0)
    m.subject_to(dm.sin(xs[1]) * dm.cos(xs[1] - xs[1] * xs[1]) - x4 - x6 == 0)
    m.subject_to(dm.sin(xs[1]) * xs[1] - x5 - x7 == 0)
    m.subject_to(dm.sin(xs[2]) * dm.cos(xs[2] - xs[2] * xs[2]) + x5 - x6 == 0)
    m.subject_to(dm.sin(xs[2]) * xs[2] - x4 - x7 == 0)
    m.subject_to(dm.sin(xs[3]) * dm.cos(xs[3] - xs[3] * xs[3]) - x4 + x5 - x6 == 0)
    m.subject_to(dm.sin(xs[3]) * xs[3] - x4 - x5 - x7 == 0)
    return m


def test_inscribedsquare02_constraints_not_dropped(caplog):
    m = _inscribedsquare02_model()
    with caplog.at_level(logging.WARNING, logger="discopt._jax.milp_relaxation"):
        _build(m)
    omitted = [r.message for r in caplog.records if "omitting constraint" in r.message]
    assert not omitted, f"inscribedsquare02 constraints dropped: {omitted}"


@pytest.mark.slow
def test_inscribedsquare02_finite_valid_bound():
    m = _inscribedsquare02_model()
    r = m.solve(time_limit=20, gap_tolerance=1e-4)
    assert r.bound is not None and np.isfinite(r.bound), "no finite dual bound"
    # MAXIMIZE: the reported bound is an UPPER bound, so it must be >= the true
    # optimum (0.968017) — never below it (a false certification).
    assert r.bound >= 0.968017 - 1e-6, f"bound {r.bound} below true optimum 0.968017"
