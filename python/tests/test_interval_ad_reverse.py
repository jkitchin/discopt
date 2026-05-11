"""Tests for ``discopt._jax.convexity.interval_ad_reverse`` (M9 of issue #51).

Acceptance criteria from issue #51:

1. Reverse-propagated bounds match the Rust FBBT reference implementation
   (``presolve/fbbt.rs``) within 1e-9 on a shared test set.
2. Tightened per-subexpression bounds are always subsets of the original
   forward bounds (monotone tightening).
3. No bound-tightening pass ever removes a feasible point of the original
   model (validated by sampling).
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

N_SAMPLES = 10_000


def _scalar(iv: Interval) -> tuple[float, float]:
    return float(np.asarray(iv.lo).ravel()[0]), float(np.asarray(iv.hi).ravel()[0])


# ---------------------------------------------------------------------------
# Linear constraints — reverse propagation should reproduce textbook FBBT
# ---------------------------------------------------------------------------


def test_linear_equality_x_plus_y_equals_one():
    m = Model()
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.minimize(x + y)
    box = tighten_box([(x + y, Interval(np.float64(1.0), np.float64(1.0)))], m)
    xlo, xhi = _scalar(box[x])
    ylo, yhi = _scalar(box[y])
    assert xlo == pytest.approx(0.0)
    assert xhi == pytest.approx(1.0)
    assert ylo == pytest.approx(0.0)
    assert yhi == pytest.approx(1.0)


def test_linear_inequality_two_x_plus_three_y_le_six():
    m = Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize(x + y)
    target = Interval(np.float64(-np.inf), np.float64(6.0))
    box = tighten_box([(2.0 * x + 3.0 * y, target)], m)
    xlo, xhi = _scalar(box[x])
    ylo, yhi = _scalar(box[y])
    # 2x + 3y <= 6, y >= 0 ⟹ 2x <= 6 ⟹ x <= 3.
    assert xhi <= 3.0 + 1e-9
    # 3y <= 6 ⟹ y <= 2.
    assert yhi <= 2.0 + 1e-9


# ---------------------------------------------------------------------------
# Multiplicative constraint
# ---------------------------------------------------------------------------


def test_multiplicative_xy_eq_one_with_positive_x():
    m = Model()
    x = m.continuous("x", lb=0.5, ub=4.0)
    y = m.continuous("y", lb=0.1, ub=10.0)
    m.minimize(x + y)
    target = Interval(np.float64(1.0), np.float64(1.0))
    box = tighten_box([(x * y, target)], m)
    ylo, yhi = _scalar(box[y])
    # y = 1/x with x ∈ [0.5, 4] ⟹ y ∈ [0.25, 2.0].
    assert ylo >= 0.25 - 1e-9
    assert yhi <= 2.0 + 1e-9


# ---------------------------------------------------------------------------
# Power inverse
# ---------------------------------------------------------------------------


def test_squaring_inverse_with_unsigned_base():
    m = Model()
    x = m.continuous("x", lb=-3.0, ub=3.0)
    m.minimize(x)
    box = tighten_box([(x**2, Interval(np.float64(0.0), np.float64(4.0)))], m)
    xlo, xhi = _scalar(box[x])
    # x² ≤ 4, x ∈ [-3, 3] ⟹ x ∈ [-2, 2].
    assert xlo >= -2.0 - 1e-9
    assert xhi <= 2.0 + 1e-9


def test_squaring_inverse_with_positive_base():
    m = Model()
    x = m.continuous("x", lb=0.5, ub=5.0)
    m.minimize(x)
    box = tighten_box([(x**2, Interval(np.float64(1.0), np.float64(4.0)))], m)
    xlo, xhi = _scalar(box[x])
    # 1 ≤ x² ≤ 4 ∧ x ≥ 0 ⟹ 1 ≤ x ≤ 2.
    assert xlo >= 1.0 - 1e-9
    assert xhi <= 2.0 + 1e-9


def test_cubic_inverse_is_monotone():
    m = Model()
    x = m.continuous("x", lb=-10.0, ub=10.0)
    m.minimize(x)
    box = tighten_box([(x**3, Interval(np.float64(-8.0), np.float64(27.0)))], m)
    xlo, xhi = _scalar(box[x])
    assert xlo >= -2.0 - 1e-6
    assert xhi <= 3.0 + 1e-6


# ---------------------------------------------------------------------------
# Function inverses
# ---------------------------------------------------------------------------


def test_exp_inverse():
    import jax.numpy as jnp  # noqa: F401  (jax registration ensures FunctionCall)

    m = Model()
    x = m.continuous("x", lb=-5.0, ub=5.0)
    m.minimize(x)
    from discopt.modeling.core import FunctionCall

    expr = FunctionCall("exp", x)
    box = tighten_box([(expr, Interval(np.float64(1.0), np.float64(np.e)))], m)
    xlo, xhi = _scalar(box[x])
    # exp(x) ∈ [1, e] ⟹ x ∈ [0, 1].
    assert xlo >= 0.0 - 1e-9
    assert xhi <= 1.0 + 1e-9


def test_log_inverse():
    from discopt.modeling.core import FunctionCall

    m = Model()
    x = m.continuous("x", lb=0.01, ub=100.0)
    m.minimize(x)
    expr = FunctionCall("log", x)
    box = tighten_box([(expr, Interval(np.float64(0.0), np.float64(1.0)))], m)
    xlo, xhi = _scalar(box[x])
    # log(x) ∈ [0, 1] ⟹ x ∈ [1, e].
    assert xlo >= 1.0 - 1e-9
    assert xhi <= np.e + 1e-9


def test_sqrt_inverse():
    from discopt.modeling.core import FunctionCall

    m = Model()
    x = m.continuous("x", lb=0.0, ub=100.0)
    m.minimize(x)
    expr = FunctionCall("sqrt", x)
    box = tighten_box([(expr, Interval(np.float64(1.0), np.float64(3.0)))], m)
    xlo, xhi = _scalar(box[x])
    # sqrt(x) ∈ [1, 3] ⟹ x ∈ [1, 9].
    assert xlo >= 1.0 - 1e-9
    assert xhi <= 9.0 + 1e-9


# ---------------------------------------------------------------------------
# Monotone tightening invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_monotone_tightening_subset_of_forward(seed):
    rng = np.random.default_rng(seed)
    m = Model()
    x = m.continuous("x", lb=float(rng.uniform(-3, -1)), ub=float(rng.uniform(1, 3)))
    y = m.continuous("y", lb=float(rng.uniform(-3, -1)), ub=float(rng.uniform(1, 3)))
    m.minimize(x + y)
    expr = (x + y) * (x - y)
    target = Interval(np.float64(-2.0), np.float64(2.0))

    # Forward enclosure
    fcache: dict = {}
    fwd_root = evaluate_interval(expr, m, _cache=fcache)
    tight = reverse_propagate(expr, target, m, forward=fcache)
    for nid, t in tight.items():
        if nid not in fcache:
            continue
        f = fcache[nid]
        assert np.all(np.asarray(t.lo) >= np.asarray(f.lo) - 1e-12)
        assert np.all(np.asarray(t.hi) <= np.asarray(f.hi) + 1e-12)
    assert np.all(np.asarray(tight[id(expr)].lo) >= np.asarray(fwd_root.lo) - 1e-12)


# ---------------------------------------------------------------------------
# Soundness via sampling: no feasible point removed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_no_feasible_point_removed_polynomial(seed):
    rng = np.random.default_rng(seed)
    m = Model()
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.minimize(x + y)

    expr = x * x + y * y
    target = Interval(np.float64(0.0), np.float64(1.0))  # unit disk
    box = tighten_box([(expr, target)], m, max_iter=10)

    xs = rng.uniform(-2.0, 2.0, size=N_SAMPLES)
    ys = rng.uniform(-2.0, 2.0, size=N_SAMPLES)
    feasible = (xs * xs + ys * ys) <= 1.0 + 1e-12
    xs_f = xs[feasible]
    ys_f = ys[feasible]

    xlo, xhi = _scalar(box[x])
    ylo, yhi = _scalar(box[y])
    assert (xs_f >= xlo - 1e-9).all()
    assert (xs_f <= xhi + 1e-9).all()
    assert (ys_f >= ylo - 1e-9).all()
    assert (ys_f <= yhi + 1e-9).all()


def test_no_feasible_point_removed_transcendental():
    rng = np.random.default_rng(11)
    from discopt.modeling.core import FunctionCall

    m = Model()
    x = m.continuous("x", lb=-3.0, ub=3.0)
    y = m.continuous("y", lb=0.1, ub=20.0)
    m.minimize(x + y)

    # exp(x) <= y, with target [-inf, 0] on (exp(x) - y).
    expr = FunctionCall("exp", x) - y
    target = Interval(np.float64(-np.inf), np.float64(0.0))
    box = tighten_box([(expr, target)], m, max_iter=10)

    xs = rng.uniform(-3.0, 3.0, size=N_SAMPLES)
    ys = rng.uniform(0.1, 20.0, size=N_SAMPLES)
    feasible = np.exp(xs) - ys <= 1e-12
    xs_f = xs[feasible]
    ys_f = ys[feasible]

    xlo, xhi = _scalar(box[x])
    ylo, yhi = _scalar(box[y])
    assert (xs_f >= xlo - 1e-9).all()
    assert (xs_f <= xhi + 1e-9).all()
    assert (ys_f >= ylo - 1e-9).all()
    assert (ys_f <= yhi + 1e-9).all()


# ---------------------------------------------------------------------------
# Idempotence: running the pass twice gives the same box.
# ---------------------------------------------------------------------------


def test_idempotence():
    m = Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize(x + y)

    target = Interval(np.float64(-np.inf), np.float64(5.0))
    constraints = [(2.0 * x + y, target)]
    box1 = tighten_box(constraints, m, max_iter=10)
    box2 = tighten_box(constraints, m, box=box1, max_iter=10)
    for v in [x, y]:
        a_lo, a_hi = _scalar(box1[v])
        b_lo, b_hi = _scalar(box2[v])
        assert a_lo == pytest.approx(b_lo, abs=1e-12)
        assert a_hi == pytest.approx(b_hi, abs=1e-12)


# ---------------------------------------------------------------------------
# Cross-check with Rust FBBT on shared models.
# ---------------------------------------------------------------------------


def _run_rust_fbbt(model: Model) -> dict:
    from discopt._rust import model_to_repr

    repr_ = model_to_repr(model)
    lo, hi = repr_.fbbt(50, 1e-9)
    lo = np.asarray(lo).ravel()
    hi = np.asarray(hi).ravel()
    out = {}
    for i, v in enumerate(model._variables):
        out[v] = (float(lo[i]), float(hi[i]))
    return out


def test_matches_rust_fbbt_on_linear_le():
    """Linear ≤ : 2 x + 3 y ≤ 6 with x, y ∈ [0, 10]."""
    m = Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize(x + y)
    m.subject_to(2.0 * x + 3.0 * y <= 6.0)

    py_box = tighten_box(
        [(2.0 * x + 3.0 * y, Interval(np.float64(-np.inf), np.float64(6.0)))],
        m,
        max_iter=20,
    )
    rust_box = _run_rust_fbbt(m)
    for v in [x, y]:
        py_lo, py_hi = _scalar(py_box[v])
        ru_lo, ru_hi = rust_box[v]
        assert py_lo == pytest.approx(ru_lo, abs=1e-9)
        assert py_hi == pytest.approx(ru_hi, abs=1e-9)


def test_matches_rust_fbbt_on_linear_eq():
    """Linear = : x + y = 1 with x, y ∈ [0, 2]."""
    m = Model()
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.minimize(x + y)
    m.subject_to(x + y == 1.0)

    py_box = tighten_box(
        [(x + y, Interval(np.float64(1.0), np.float64(1.0)))],
        m,
        max_iter=20,
    )
    rust_box = _run_rust_fbbt(m)
    for v in [x, y]:
        py_lo, py_hi = _scalar(py_box[v])
        ru_lo, ru_hi = rust_box[v]
        assert py_lo == pytest.approx(ru_lo, abs=1e-9)
        assert py_hi == pytest.approx(ru_hi, abs=1e-9)
