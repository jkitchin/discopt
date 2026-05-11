"""M9 regression: reverse-mode interval AD on benchmark-style constraints.

The reverse-mode propagator (``discopt._jax.convexity.interval_ad_reverse``)
is not yet wired into the LP relaxation compiler — that integration
ships alongside the M2/M3 kernel and M11 wrapper integration as a
follow-up. This regression exercises the pass directly on constraint
patterns drawn from typical benchmark problems and asserts:

1. **Soundness:** for every sampled feasible point of the original
   model (≥ 10⁴ samples), the tightened per-variable box still
   contains it. No bound-tightening pass ever removes a feasible
   point.
2. **Monotone tightening:** every tightened sub-expression bound is a
   subset of the corresponding forward-interval enclosure.
3. **Parity with Rust FBBT:** on shared models, the per-variable
   tightened bounds match ``ModelRepr.fbbt`` within 1e-9.

These properties are the contract M9 makes with downstream consumers.
Breaking any of them is a correctness regression.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt import Model
from discopt._jax.convexity.interval import Interval
from discopt._jax.convexity.interval_ad_reverse import (
    reverse_propagate,
    tighten_box,
)
from discopt._jax.convexity.interval_eval import evaluate_interval
from discopt.modeling.core import FunctionCall

N_REGRESSION_SAMPLES = 10_000


def _scalar(iv):
    return float(np.asarray(iv.lo).ravel()[0]), float(np.asarray(iv.hi).ravel()[0])


# ---------------------------------------------------------------------------
# 1. Soundness on benchmark-style constraints
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.regression
def test_unit_disk_inside_outside_box():
    """Quadratic feasibility constraint x² + y² ≤ 1 inside a wider box."""
    rng = np.random.default_rng(0)
    m = Model()
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.minimize(x + y)

    target = Interval(np.float64(0.0), np.float64(1.0))
    box = tighten_box([(x * x + y * y, target)], m, max_iter=10)

    xs = rng.uniform(-2.0, 2.0, size=N_REGRESSION_SAMPLES)
    ys = rng.uniform(-2.0, 2.0, size=N_REGRESSION_SAMPLES)
    feas = (xs * xs + ys * ys) <= 1.0 + 1e-12
    xs_f = xs[feas]
    ys_f = ys[feas]

    xlo, xhi = _scalar(box[x])
    ylo, yhi = _scalar(box[y])
    assert (xs_f >= xlo - 1e-9).all()
    assert (xs_f <= xhi + 1e-9).all()
    assert (ys_f >= ylo - 1e-9).all()
    assert (ys_f <= yhi + 1e-9).all()


@pytest.mark.regression
def test_log_constraint_tightens_positive_orthant():
    """log(x) + log(y) >= 0 with x,y > 0 ⟹ xy >= 1; reverse pass should
    cut off corners where x or y are too small to satisfy this."""
    rng = np.random.default_rng(7)
    m = Model()
    x = m.continuous("x", lb=0.1, ub=10.0)
    y = m.continuous("y", lb=0.1, ub=10.0)
    m.minimize(x + y)

    expr = FunctionCall("log", x) + FunctionCall("log", y)
    target = Interval(np.float64(0.0), np.float64(np.inf))
    box = tighten_box([(expr, target)], m, max_iter=15)

    xs = rng.uniform(0.1, 10.0, size=N_REGRESSION_SAMPLES)
    ys = rng.uniform(0.1, 10.0, size=N_REGRESSION_SAMPLES)
    feas = (np.log(xs) + np.log(ys)) >= -1e-12
    xs_f = xs[feas]
    ys_f = ys[feas]

    xlo, xhi = _scalar(box[x])
    ylo, yhi = _scalar(box[y])
    assert (xs_f >= xlo - 1e-9).all()
    assert (xs_f <= xhi + 1e-9).all()
    assert (ys_f >= ylo - 1e-9).all()
    assert (ys_f <= yhi + 1e-9).all()


@pytest.mark.regression
def test_polynomial_equality_tightens_box():
    """Cubic feasibility cut x³ - y = 0 with y ∈ [-8, 27] ⟹ x ∈ [-2, 3]."""
    rng = np.random.default_rng(3)
    m = Model()
    x = m.continuous("x", lb=-5.0, ub=5.0)
    y = m.continuous("y", lb=-8.0, ub=27.0)
    m.minimize(x + y)

    expr = x**3 - y
    target = Interval(np.float64(0.0), np.float64(0.0))
    box = tighten_box([(expr, target)], m, max_iter=15)

    xlo, xhi = _scalar(box[x])
    assert xlo >= -2.0 - 1e-6
    assert xhi <= 3.0 + 1e-6

    xs = rng.uniform(-5.0, 5.0, size=N_REGRESSION_SAMPLES)
    ys = rng.uniform(-8.0, 27.0, size=N_REGRESSION_SAMPLES)
    feas = np.abs(xs**3 - ys) <= 1e-9
    xs_f = xs[feas]
    if len(xs_f):
        assert (xs_f >= xlo - 1e-6).all()
        assert (xs_f <= xhi + 1e-6).all()


# ---------------------------------------------------------------------------
# 2. Monotone-tightening invariant on every node
# ---------------------------------------------------------------------------


@pytest.mark.regression
@pytest.mark.parametrize("seed", range(3))
def test_monotone_tightening_on_compound_expression(seed):
    rng = np.random.default_rng(seed)
    m = Model()
    x = m.continuous("x", lb=float(rng.uniform(-2, -0.5)), ub=float(rng.uniform(0.5, 2)))
    y = m.continuous("y", lb=float(rng.uniform(-2, -0.5)), ub=float(rng.uniform(0.5, 2)))
    m.minimize(x + y)

    expr = (x * y) + (x - y) * (x - y)
    target = Interval(np.float64(-1.0), np.float64(2.0))

    fcache: dict = {}
    evaluate_interval(expr, m, _cache=fcache)
    tight = reverse_propagate(expr, target, m, forward=fcache)

    for nid, t in tight.items():
        if nid not in fcache:
            continue
        f = fcache[nid]
        assert np.all(np.asarray(t.lo) >= np.asarray(f.lo) - 1e-12)
        assert np.all(np.asarray(t.hi) <= np.asarray(f.hi) + 1e-12)


# ---------------------------------------------------------------------------
# 3. Parity with Rust FBBT on shared models
# ---------------------------------------------------------------------------


def _run_rust_fbbt(model: Model):
    from discopt._rust import model_to_repr

    repr_ = model_to_repr(model)
    lo, hi = repr_.fbbt(50, 1e-9)
    lo = np.asarray(lo).ravel()
    hi = np.asarray(hi).ravel()
    out = {}
    for i, v in enumerate(model._variables):
        out[v] = (float(lo[i]), float(hi[i]))
    return out


@pytest.mark.regression
def test_parity_with_rust_fbbt_on_compound_linear():
    m = Model()
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    z = m.continuous("z", lb=0.0, ub=5.0)
    m.minimize(x + y + z)
    m.subject_to(x + 2.0 * y + 3.0 * z <= 6.0)
    m.subject_to(x + y + z >= 1.0)

    py_box = tighten_box(
        [
            (x + 2.0 * y + 3.0 * z, Interval(np.float64(-np.inf), np.float64(6.0))),
            (x + y + z, Interval(np.float64(1.0), np.float64(np.inf))),
        ],
        m,
        max_iter=30,
    )
    rust_box = _run_rust_fbbt(m)
    for v in [x, y, z]:
        py_lo, py_hi = _scalar(py_box[v])
        ru_lo, ru_hi = rust_box[v]
        assert py_lo == pytest.approx(ru_lo, abs=1e-9)
        assert py_hi == pytest.approx(ru_hi, abs=1e-9)


@pytest.mark.regression
def test_parity_with_rust_fbbt_on_squared_constraint():
    m = Model()
    x = m.continuous("x", lb=-3.0, ub=3.0)
    m.minimize(x)
    m.subject_to(x * x <= 4.0)

    py_box = tighten_box(
        [(x * x, Interval(np.float64(-np.inf), np.float64(4.0)))],
        m,
        max_iter=20,
    )
    rust_box = _run_rust_fbbt(m)
    py_lo, py_hi = _scalar(py_box[x])
    ru_lo, ru_hi = rust_box[x]
    # Both should arrive at x ∈ [-2, 2] (or wider if a side abstains).
    assert py_lo == pytest.approx(ru_lo, abs=1e-9)
    assert py_hi == pytest.approx(ru_hi, abs=1e-9)
