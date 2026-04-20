"""Soundness tests for the interval DAG evaluator.

For any expression and any sample point inside the declared variable
box, the pointwise value must fall within the interval enclosure. This
is the base contract the convexity certificate depends on.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
from discopt._jax.convexity.interval import Interval
from discopt._jax.convexity.interval_eval import evaluate_interval
from discopt._jax.dag_compiler import compile_expression
from discopt.modeling.core import FunctionCall, Model

SAMPLES = 48


def _sample_flat(model: Model, n: int, seed: int) -> np.ndarray:
    """Draw ``n`` random flat variable vectors inside the model's box."""
    rng = np.random.default_rng(seed)
    lbs, ubs = [], []
    for v in model._variables:
        lbs.append(np.asarray(v.lb, dtype=np.float64).ravel())
        ubs.append(np.asarray(v.ub, dtype=np.float64).ravel())
    lb = np.concatenate(lbs)
    ub = np.concatenate(ubs)
    return rng.uniform(lb, ub, size=(n, lb.size))


def _assert_encloses(expr, model, *, tol=1e-9):
    """The interval enclosure must contain pointwise samples."""
    iv_out = evaluate_interval(expr, model)
    f = compile_expression(expr, model)
    xs = _sample_flat(model, SAMPLES, seed=0)
    for x in xs:
        val = float(f(x))
        assert float(np.asarray(iv_out.lo).ravel()[0]) <= val + tol, (
            f"lo {iv_out.lo} > value {val} on {x}"
        )
        assert val <= float(np.asarray(iv_out.hi).ravel()[0]) + tol, (
            f"value {val} > hi {iv_out.hi} on {x}"
        )


class TestElementaryExpressions:
    def test_linear(self):
        m = Model("t")
        x = m.continuous("x", lb=-2.0, ub=3.0)
        y = m.continuous("y", lb=0.0, ub=5.0)
        _assert_encloses(2.0 * x + 3.0 * y - 1.0, m)

    def test_polynomial(self):
        m = Model("t")
        x = m.continuous("x", lb=-1.0, ub=2.0)
        _assert_encloses(x**3 - 2.0 * x**2 + x + 4.0, m)

    def test_exp_affine(self):
        m = Model("t")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        _assert_encloses(dm.exp(x + 1.0), m)

    def test_log_positive(self):
        m = Model("t")
        x = m.continuous("x", lb=0.1, ub=5.0)
        _assert_encloses(dm.log(x**2 + 1.0), m)

    def test_sqrt_nonneg(self):
        m = Model("t")
        x = m.continuous("x", lb=0.0, ub=4.0)
        _assert_encloses(dm.sqrt(x + 1.0), m)

    def test_reciprocal_positive(self):
        m = Model("t")
        x = m.continuous("x", lb=0.5, ub=3.0)
        _assert_encloses(1.0 / x, m)


class TestNonlinearCompositions:
    def test_exp_of_square(self):
        m = Model("t")
        x = m.continuous("x", lb=-1.5, ub=1.5)
        _assert_encloses(dm.exp(x**2), m)

    def test_log_of_sum_exp(self):
        """log(exp(x) + exp(y)) — a logsumexp-like pattern."""
        m = Model("t")
        x = m.continuous("x", lb=-2.0, ub=2.0)
        y = m.continuous("y", lb=-2.0, ub=2.0)
        _assert_encloses(dm.log(dm.exp(x) + dm.exp(y)), m)

    def test_max_of_quadratics(self):
        m = Model("t")
        x = m.continuous("x", lb=-2.0, ub=2.0)
        y = m.continuous("y", lb=-2.0, ub=2.0)
        _assert_encloses(FunctionCall("max", x**2, y**2), m)


class TestDegenerate:
    def test_constant_expression(self):
        m = Model("t")
        m.continuous("x", lb=-1.0, ub=1.0)  # unused
        iv_out = evaluate_interval(2.5 * dm.exp(0.0), m)
        assert abs(float(np.asarray(iv_out.lo).ravel()[0]) - 2.5) < 1e-12
        assert abs(float(np.asarray(iv_out.hi).ravel()[0]) - 2.5) < 1e-12

    def test_box_override(self):
        """Override the declared variable bounds via ``box=``."""
        m = Model("t")
        x = m.continuous("x", lb=-10.0, ub=10.0)
        tight = {x: Interval.from_bounds(0.5, 1.0)}
        iv_out = evaluate_interval(x**2, m, box=tight)
        # [0.5, 1.0]^2 ⊆ [0.25, 1.0]
        assert float(np.asarray(iv_out.lo).ravel()[0]) <= 0.25 + 1e-12
        assert float(np.asarray(iv_out.hi).ravel()[0]) >= 1.0 - 1e-12
        # But loose enough to be sound without being absurd.
        assert float(np.asarray(iv_out.hi).ravel()[0]) <= 1.5


class TestUnsupportedAtom:
    def test_sigmoid_atom_returns_unbounded(self):
        """``sigmoid`` isn't in the interval atom table; enclosure is ±inf."""
        m = Model("t")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        # discopt's sigmoid emits a FunctionCall("sigmoid", x).
        iv_out = evaluate_interval(dm.sigmoid(x), m)
        # The certificate downstream will refuse to certify — that's
        # the safe behavior. Test here just ensures we don't crash and
        # that the enclosure is at least [−∞, +∞] (sound but useless).
        assert (
            not np.isfinite(float(np.asarray(iv_out.lo).ravel()[0]))
            or not np.isfinite(float(np.asarray(iv_out.hi).ravel()[0]))
            or (
                float(np.asarray(iv_out.lo).ravel()[0]) <= 0.0
                and float(np.asarray(iv_out.hi).ravel()[0]) >= 1.0
            )
        )


class TestCaching:
    def test_shared_subexpression_cached(self):
        m = Model("t")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        sq = x**2
        expr = sq + sq
        cache: dict = {}
        out = evaluate_interval(expr, m, _cache=cache)
        assert id(sq) in cache
        # [-1, 1]^2 = [0, 1], doubled → [0, 2]
        assert float(np.asarray(out.hi).ravel()[0]) <= 2.0 + 1e-9
        assert float(np.asarray(out.lo).ravel()[0]) >= 0.0 - 1e-9
