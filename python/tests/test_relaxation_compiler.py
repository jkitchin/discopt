"""Tests for the McCormick relaxation compiler."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, "/Users/jkitchin/Dropbox/projects/discopt/jaxminlp_benchmarks")
sys.path.insert(0, "/Users/jkitchin/Dropbox/projects/discopt/python")

from jaxminlp_api.core import (
    Constant,
    Constraint,
    Model,
    Parameter,
    _wrap,
)
import jaxminlp_api.core as jm
from jaxminlp_api import examples

from discopt._jax.dag_compiler import compile_expression
from discopt._jax.relaxation_compiler import (
    compile_relaxation,
    compile_objective_relaxation,
    compile_constraint_relaxation,
)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _flat_size(model: Model) -> int:
    return sum(v.size for v in model._variables)


def _get_var_bounds(model: Model):
    """Return (lb_flat, ub_flat) arrays from model variable bounds."""
    lbs, ubs = [], []
    for v in model._variables:
        lbs.append(np.clip(v.lb.flatten(), -1e3, 1e3))
        ubs.append(np.clip(v.ub.flatten(), -1e3, 1e3))
    lb = jnp.array(np.concatenate(lbs), dtype=jnp.float64)
    ub = jnp.array(np.concatenate(ubs), dtype=jnp.float64)
    return lb, ub


def _random_point_in_bounds(lb, ub, rng):
    """Random point strictly inside [lb, ub]."""
    width = jnp.maximum(ub - lb, 1e-6)
    t = jnp.array(rng.uniform(0.05, 0.95, size=lb.shape), dtype=jnp.float64)
    return lb + t * width


def _check_soundness(relax_fn, true_fn, model, n_samples=10000, seed=42):
    """Verify cv <= f(x) <= cc at many random points within variable bounds."""
    rng = np.random.default_rng(seed)
    lb, ub = _get_var_bounds(model)
    n = _flat_size(model)

    violations_cv = 0
    violations_cc = 0

    for _ in range(n_samples):
        x = _random_point_in_bounds(lb, ub, rng)
        # For soundness check: x_cv = x_cc = x (point relaxation)
        cv, cc = relax_fn(x, x, lb, ub)
        true_val = true_fn(x)

        # Allow small numerical tolerance
        if float(cv) > float(true_val) + 1e-8:
            violations_cv += 1
        if float(cc) < float(true_val) - 1e-8:
            violations_cc += 1

    return violations_cv, violations_cc


# ─────────────────────────────────────────────────────────────
# Test 1: Soundness on simple expressions
# ─────────────────────────────────────────────────────────────

class TestSimpleExpressionSoundness:

    def test_linear_2x_plus_3y(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=5)
        y = m.continuous("y", lb=0, ub=5)
        m.minimize(x)

        expr = 2 * x + 3 * y
        relax_fn = compile_relaxation(expr, m)
        true_fn = compile_expression(expr, m)

        cv_viol, cc_viol = _check_soundness(relax_fn, true_fn, m)
        assert cv_viol == 0, f"cv violations: {cv_viol}"
        assert cc_viol == 0, f"cc violations: {cc_viol}"

    def test_bilinear_xy(self):
        m = Model("test")
        x = m.continuous("x", lb=0.5, ub=5)
        y = m.continuous("y", lb=0.5, ub=5)
        m.minimize(x)

        expr = x * y
        relax_fn = compile_relaxation(expr, m)
        true_fn = compile_expression(expr, m)

        cv_viol, cc_viol = _check_soundness(relax_fn, true_fn, m)
        assert cv_viol == 0, f"cv violations: {cv_viol}"
        assert cc_viol == 0, f"cc violations: {cc_viol}"

    def test_exp_plus_log(self):
        m = Model("test")
        x = m.continuous("x", lb=0.1, ub=3)
        y = m.continuous("y", lb=0.1, ub=3)
        m.minimize(x)

        expr = jm.exp(x) + jm.log(y)
        relax_fn = compile_relaxation(expr, m)
        true_fn = compile_expression(expr, m)

        cv_viol, cc_viol = _check_soundness(relax_fn, true_fn, m)
        assert cv_viol == 0, f"cv violations: {cv_viol}"
        assert cc_viol == 0, f"cc violations: {cc_viol}"

    def test_x_squared(self):
        m = Model("test")
        x = m.continuous("x", lb=-3, ub=3)
        m.minimize(x)

        expr = x ** 2
        relax_fn = compile_relaxation(expr, m)
        true_fn = compile_expression(expr, m)

        cv_viol, cc_viol = _check_soundness(relax_fn, true_fn, m)
        assert cv_viol == 0, f"cv violations: {cv_viol}"
        assert cc_viol == 0, f"cc violations: {cc_viol}"

    def test_sqrt_x(self):
        m = Model("test")
        x = m.continuous("x", lb=0.1, ub=10)
        m.minimize(x)

        expr = jm.sqrt(x)
        relax_fn = compile_relaxation(expr, m)
        true_fn = compile_expression(expr, m)

        cv_viol, cc_viol = _check_soundness(relax_fn, true_fn, m)
        assert cv_viol == 0, f"cv violations: {cv_viol}"
        assert cc_viol == 0, f"cc violations: {cc_viol}"

    def test_sin_x(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=3)
        m.minimize(x)

        expr = jm.sin(x)
        relax_fn = compile_relaxation(expr, m)
        true_fn = compile_expression(expr, m)

        cv_viol, cc_viol = _check_soundness(relax_fn, true_fn, m)
        assert cv_viol == 0, f"cv violations: {cv_viol}"
        assert cc_viol == 0, f"cc violations: {cc_viol}"

    def test_neg_x(self):
        m = Model("test")
        x = m.continuous("x", lb=-5, ub=5)
        m.minimize(x)

        expr = -x
        relax_fn = compile_relaxation(expr, m)
        true_fn = compile_expression(expr, m)

        cv_viol, cc_viol = _check_soundness(relax_fn, true_fn, m)
        assert cv_viol == 0, f"cv violations: {cv_viol}"
        assert cc_viol == 0, f"cc violations: {cc_viol}"

    def test_subtraction(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=5)
        y = m.continuous("y", lb=0, ub=5)
        m.minimize(x)

        expr = x - y
        relax_fn = compile_relaxation(expr, m)
        true_fn = compile_expression(expr, m)

        cv_viol, cc_viol = _check_soundness(relax_fn, true_fn, m)
        assert cv_viol == 0, f"cv violations: {cv_viol}"
        assert cc_viol == 0, f"cc violations: {cc_viol}"

    def test_division_by_constant(self):
        m = Model("test")
        x = m.continuous("x", lb=1, ub=10)
        m.minimize(x)

        expr = x / 3.0
        relax_fn = compile_relaxation(expr, m)
        true_fn = compile_expression(expr, m)

        cv_viol, cc_viol = _check_soundness(relax_fn, true_fn, m)
        assert cv_viol == 0, f"cv violations: {cv_viol}"
        assert cc_viol == 0, f"cc violations: {cc_viol}"

    def test_cos_x(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=2)
        m.minimize(x)

        expr = jm.cos(x)
        relax_fn = compile_relaxation(expr, m)
        true_fn = compile_expression(expr, m)

        cv_viol, cc_viol = _check_soundness(relax_fn, true_fn, m)
        assert cv_viol == 0, f"cv violations: {cv_viol}"
        assert cc_viol == 0, f"cc violations: {cc_viol}"


# ─────────────────────────────────────────────────────────────
# Test 2: JIT + vmap compatibility
# ─────────────────────────────────────────────────────────────

class TestJitVmap:

    def test_jit_works(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=5)
        y = m.continuous("y", lb=0, ub=5)
        m.minimize(x)

        expr = 2 * x + 3 * y
        relax_fn = compile_relaxation(expr, m)
        jitted = jax.jit(relax_fn)

        lb = jnp.array([0.0, 0.0])
        ub = jnp.array([5.0, 5.0])
        x_cv = jnp.array([1.0, 2.0])
        x_cc = jnp.array([1.0, 2.0])

        cv, cc = jitted(x_cv, x_cc, lb, ub)
        expected = 2 * 1.0 + 3 * 2.0
        assert jnp.allclose(cv, expected)
        assert jnp.allclose(cc, expected)

    def test_vmap_batch(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=5)
        y = m.continuous("y", lb=0, ub=5)
        m.minimize(x)

        expr = x ** 2 + jm.exp(y)
        relax_fn = compile_relaxation(expr, m)

        batch_size = 128
        rng = np.random.default_rng(42)

        # Create batch of bounds (simulating B&B with different subproblems)
        lb_batch = jnp.array(rng.uniform(0, 2, (batch_size, 2)), dtype=jnp.float64)
        ub_batch = lb_batch + jnp.array(rng.uniform(0.5, 3, (batch_size, 2)), dtype=jnp.float64)

        x_cv_batch = 0.5 * (lb_batch + ub_batch)
        x_cc_batch = 0.5 * (lb_batch + ub_batch)

        vmapped = jax.vmap(relax_fn)
        cv_batch, cc_batch = vmapped(x_cv_batch, x_cc_batch, lb_batch, ub_batch)

        assert cv_batch.shape == (batch_size,)
        assert cc_batch.shape == (batch_size,)
        # cv should be <= cc for all entries
        assert jnp.all(cv_batch <= cc_batch + 1e-8)

    def test_jit_vmap_combined(self):
        m = Model("test")
        x = m.continuous("x", lb=0.1, ub=5)
        y = m.continuous("y", lb=0.1, ub=5)
        m.minimize(x)

        expr = jm.log(x) + y * y
        relax_fn = compile_relaxation(expr, m)

        batch_size = 128
        rng = np.random.default_rng(99)

        lb_batch = jnp.array(rng.uniform(0.1, 2, (batch_size, 2)), dtype=jnp.float64)
        ub_batch = lb_batch + jnp.array(rng.uniform(0.5, 3, (batch_size, 2)), dtype=jnp.float64)

        x_cv_batch = 0.5 * (lb_batch + ub_batch)
        x_cc_batch = 0.5 * (lb_batch + ub_batch)

        combined = jax.jit(jax.vmap(relax_fn))
        cv_batch, cc_batch = combined(x_cv_batch, x_cc_batch, lb_batch, ub_batch)

        assert cv_batch.shape == (batch_size,)
        assert cc_batch.shape == (batch_size,)
        assert jnp.all(cv_batch <= cc_batch + 1e-8)


# ─────────────────────────────────────────────────────────────
# Test 3: All 7 example model objectives
# ─────────────────────────────────────────────────────────────

class TestExampleObjectives:

    def _test_example_model(self, build_fn, n_samples=5000):
        """Generic test: compile objective relaxation and verify soundness."""
        m = build_fn()
        relax_fn = compile_objective_relaxation(m)
        true_fn = compile_expression(m._objective.expression, m)

        cv_viol, cc_viol = _check_soundness(relax_fn, true_fn, m, n_samples=n_samples)
        assert cv_viol == 0, f"cv violations: {cv_viol}"
        assert cc_viol == 0, f"cc violations: {cc_viol}"

    def test_simple_minlp(self):
        self._test_example_model(examples.example_simple_minlp)

    def test_pooling_haverly(self):
        self._test_example_model(examples.example_pooling_haverly)

    def test_process_synthesis(self):
        self._test_example_model(examples.example_process_synthesis)

    def test_portfolio(self):
        self._test_example_model(examples.example_portfolio, n_samples=1000)

    def test_reactor_design(self):
        self._test_example_model(examples.example_reactor_design)

    def test_facility_location(self):
        self._test_example_model(examples.example_facility_location, n_samples=1000)

    def test_parametric(self):
        self._test_example_model(examples.example_parametric)


# ─────────────────────────────────────────────────────────────
# Test 4: Gap monotonicity — tighter bounds -> smaller gap
# ─────────────────────────────────────────────────────────────

class TestGapMonotonicity:

    def test_gap_decreases_with_tighter_bounds(self):
        m = Model("test")
        x = m.continuous("x", lb=0.5, ub=5)
        y = m.continuous("y", lb=0.5, ub=5)
        m.minimize(x)

        expr = x * y + jm.exp(x)
        relax_fn = compile_relaxation(expr, m)

        # Pick a target point
        x_target = jnp.array([2.0, 3.0])

        # Progressive tightening: bounds contract around x_target
        widths = [4.0, 2.0, 1.0, 0.5, 0.1]
        gaps = []

        for w in widths:
            lb = jnp.maximum(jnp.array([0.5, 0.5]), x_target - w)
            ub = jnp.minimum(jnp.array([5.0, 5.0]), x_target + w)
            cv, cc = relax_fn(x_target, x_target, lb, ub)
            gap = float(cc - cv)
            gaps.append(gap)

        # Gap should be monotonically non-increasing as bounds tighten
        for i in range(len(gaps) - 1):
            assert gaps[i] >= gaps[i + 1] - 1e-8, (
                f"Gap increased: {gaps[i]:.6f} -> {gaps[i+1]:.6f} "
                f"at step {i} -> {i+1}"
            )

    def test_gap_approaches_zero_at_point(self):
        """When bounds collapse to a point, gap should be ~0."""
        m = Model("test")
        x = m.continuous("x", lb=0.1, ub=10)
        m.minimize(x)

        expr = jm.exp(x) + x ** 2
        relax_fn = compile_relaxation(expr, m)

        x_pt = jnp.array([3.0])
        # Very tight bounds
        lb = x_pt - 1e-10
        ub = x_pt + 1e-10
        cv, cc = relax_fn(x_pt, x_pt, lb, ub)
        gap = float(cc - cv)
        assert gap < 1e-4, f"Gap at near-point should be ~0, got {gap}"


# ─────────────────────────────────────────────────────────────
# Test 5: Constant expressions
# ─────────────────────────────────────────────────────────────

class TestConstantExpressions:

    def test_scalar_constant(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=1)
        m.minimize(x)

        expr = Constant(7.5)
        relax_fn = compile_relaxation(expr, m)

        x_cv = jnp.array([0.5])
        cv, cc = relax_fn(x_cv, x_cv, jnp.array([0.0]), jnp.array([1.0]))
        assert jnp.allclose(cv, 7.5)
        assert jnp.allclose(cc, 7.5)

    def test_constant_expression_sum(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=1)
        m.minimize(x)

        expr = Constant(3.0) + Constant(4.0)
        relax_fn = compile_relaxation(expr, m)

        x_cv = jnp.array([0.5])
        cv, cc = relax_fn(x_cv, x_cv, jnp.array([0.0]), jnp.array([1.0]))
        assert jnp.allclose(cv, 7.0)
        assert jnp.allclose(cc, 7.0)

    def test_constant_times_constant(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=1)
        m.minimize(x)

        expr = Constant(2.0) * Constant(5.0)
        relax_fn = compile_relaxation(expr, m)

        x_cv = jnp.array([0.5])
        cv, cc = relax_fn(x_cv, x_cv, jnp.array([0.0]), jnp.array([1.0]))
        assert jnp.allclose(cv, 10.0)
        assert jnp.allclose(cc, 10.0)


# ─────────────────────────────────────────────────────────────
# Test 6: Linear expressions — relaxation is exact
# ─────────────────────────────────────────────────────────────

class TestLinearExact:

    def test_linear_relaxation_exact(self):
        """For a*x + b, relaxation should be exact when x_cv = x_cc = x."""
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)

        expr = 3.0 * x + 7.0
        relax_fn = compile_relaxation(expr, m)
        true_fn = compile_expression(expr, m)

        rng = np.random.default_rng(42)
        lb = jnp.array([0.0])
        ub = jnp.array([10.0])

        for _ in range(1000):
            x_val = _random_point_in_bounds(lb, ub, rng)
            cv, cc = relax_fn(x_val, x_val, lb, ub)
            true_val = true_fn(x_val)
            assert jnp.allclose(cv, true_val, atol=1e-10), (
                f"cv={float(cv)} != true={float(true_val)}"
            )
            assert jnp.allclose(cc, true_val, atol=1e-10), (
                f"cc={float(cc)} != true={float(true_val)}"
            )

    def test_multivariate_linear_exact(self):
        """Linear expression in multiple variables should be exact."""
        m = Model("test")
        x = m.continuous("x", lb=0, ub=5)
        y = m.continuous("y", lb=0, ub=5)
        z = m.continuous("z", lb=0, ub=5)
        m.minimize(x)

        expr = 2 * x + 3 * y - z + 1.0
        relax_fn = compile_relaxation(expr, m)
        true_fn = compile_expression(expr, m)

        rng = np.random.default_rng(123)
        lb = jnp.array([0.0, 0.0, 0.0])
        ub = jnp.array([5.0, 5.0, 5.0])

        for _ in range(1000):
            x_val = _random_point_in_bounds(lb, ub, rng)
            cv, cc = relax_fn(x_val, x_val, lb, ub)
            true_val = true_fn(x_val)
            assert jnp.allclose(cv, true_val, atol=1e-10)
            assert jnp.allclose(cc, true_val, atol=1e-10)


# ─────────────────────────────────────────────────────────────
# Test 7: Constraint relaxation
# ─────────────────────────────────────────────────────────────

class TestConstraintRelaxation:

    def test_constraint_body_relaxation(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=5)
        y = m.continuous("y", lb=0, ub=5)
        m.minimize(x)

        # Constraint: x^2 + y <= 10
        constraint = (x ** 2 + y <= 10)
        relax_fn = compile_constraint_relaxation(constraint, m)
        true_fn = compile_expression(constraint.body, m)

        cv_viol, cc_viol = _check_soundness(relax_fn, true_fn, m, n_samples=5000)
        assert cv_viol == 0, f"cv violations: {cv_viol}"
        assert cc_viol == 0, f"cc violations: {cc_viol}"

    def test_nonlinear_constraint(self):
        m = Model("test")
        x = m.continuous("x", lb=0.1, ub=5)
        y = m.continuous("y", lb=0.1, ub=5)
        m.minimize(x)

        # Constraint: exp(x) + log(y) <= 20
        constraint = (jm.exp(x) + jm.log(y) <= 20)
        relax_fn = compile_constraint_relaxation(constraint, m)
        true_fn = compile_expression(constraint.body, m)

        cv_viol, cc_viol = _check_soundness(relax_fn, true_fn, m, n_samples=5000)
        assert cv_viol == 0, f"cv violations: {cv_viol}"
        assert cc_viol == 0, f"cc violations: {cc_viol}"


# ─────────────────────────────────────────────────────────────
# Test: Parameter handling
# ─────────────────────────────────────────────────────────────

class TestParameterHandling:

    def test_parameter_relaxation_exact(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=5)
        p = m.parameter("price", value=3.0)
        m.minimize(x)

        expr = p * x + 1.0
        relax_fn = compile_relaxation(expr, m)
        true_fn = compile_expression(expr, m)

        rng = np.random.default_rng(42)
        lb = jnp.array([0.0])
        ub = jnp.array([5.0])

        for _ in range(1000):
            x_val = _random_point_in_bounds(lb, ub, rng)
            cv, cc = relax_fn(x_val, x_val, lb, ub)
            true_val = true_fn(x_val)
            assert jnp.allclose(cv, true_val, atol=1e-10)
            assert jnp.allclose(cc, true_val, atol=1e-10)


# ─────────────────────────────────────────────────────────────
# Test: IndexExpression
# ─────────────────────────────────────────────────────────────

class TestIndexExpression:

    def test_array_indexing(self):
        m = Model("test")
        x = m.continuous("x", shape=(3,), lb=0, ub=5)
        m.minimize(jm.sum(x))

        expr = x[1] * 2.0 + x[0]
        relax_fn = compile_relaxation(expr, m)
        true_fn = compile_expression(expr, m)

        cv_viol, cc_viol = _check_soundness(relax_fn, true_fn, m, n_samples=5000)
        assert cv_viol == 0, f"cv violations: {cv_viol}"
        assert cc_viol == 0, f"cc violations: {cc_viol}"


# ─────────────────────────────────────────────────────────────
# Test: SumExpression
# ─────────────────────────────────────────────────────────────

class TestSumExpression:

    def test_sum_of_array_variable(self):
        m = Model("test")
        x = m.continuous("x", shape=(4,), lb=0, ub=5)
        m.minimize(jm.sum(x))

        expr = jm.sum(x)
        relax_fn = compile_relaxation(expr, m)
        true_fn = compile_expression(expr, m)

        cv_viol, cc_viol = _check_soundness(relax_fn, true_fn, m, n_samples=5000)
        assert cv_viol == 0, f"cv violations: {cv_viol}"
        assert cc_viol == 0, f"cc violations: {cc_viol}"


# ─────────────────────────────────────────────────────────────
# Test: No-objective error
# ─────────────────────────────────────────────────────────────

class TestErrors:

    def test_no_objective_raises(self):
        m = Model("test")
        m.continuous("x", lb=0, ub=1)
        with pytest.raises(ValueError, match="no objective"):
            compile_objective_relaxation(m)
