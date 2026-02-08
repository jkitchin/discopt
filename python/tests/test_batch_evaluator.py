"""Tests for the BatchRelaxationEvaluator."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import time

import discopt.modeling.core as dm
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.batch_evaluator import (
    BatchRelaxationEvaluator,
    batch_evaluator_from_constraint,
    batch_evaluator_from_expression,
    batch_evaluator_from_objective,
)
from discopt._jax.dag_compiler import compile_expression
from discopt._jax.relaxation_compiler import compile_relaxation
from discopt.modeling import examples
from discopt.modeling.core import Model

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


def _make_batch_bounds(model: Model, batch_size: int, seed: int = 42):
    """Create random batch bounds within model variable bounds."""
    rng = np.random.default_rng(seed)
    lb_model, ub_model = _get_var_bounds(model)
    n = _flat_size(model)

    # Generate random subintervals within [lb_model, ub_model]
    lb_np = np.array(lb_model)
    ub_np = np.array(ub_model)
    width = ub_np - lb_np

    # Random lower bounds within [lb, lb + 0.7*width]
    t_lb = rng.uniform(0.0, 0.7, (batch_size, n))
    lb_batch = lb_np + t_lb * width

    # Random upper bounds within [lb_batch, ub]
    remaining = ub_np - lb_batch
    t_ub = rng.uniform(0.3, 1.0, (batch_size, n))
    ub_batch = lb_batch + t_ub * remaining

    return (
        jnp.array(lb_batch, dtype=jnp.float64),
        jnp.array(ub_batch, dtype=jnp.float64),
    )


# ─────────────────────────────────────────────────────────────
# Test 1: Functional correctness (batch matches serial)
# ─────────────────────────────────────────────────────────────


class TestBatchMatchesSerial:
    """Verify batch evaluation gives identical results to serial evaluation."""

    def _check_batch_serial_match(self, model: Model, batch_size: int = 32):
        """Check batch eval matches serial for objective relaxation."""
        evaluator = batch_evaluator_from_objective(model)
        relax_fn = compile_relaxation(model._objective.expression, model)

        lb_batch, ub_batch = _make_batch_bounds(model, batch_size)

        # Batch evaluation
        cv_batch, cc_batch = evaluator.evaluate_batch(lb_batch, ub_batch)

        # Serial evaluation
        for i in range(batch_size):
            lb_i = lb_batch[i]
            ub_i = ub_batch[i]
            mid_i = 0.5 * (lb_i + ub_i)
            cv_serial, cc_serial = relax_fn(mid_i, mid_i, lb_i, ub_i)

            assert jnp.allclose(cv_batch[i], cv_serial, atol=1e-10), (
                f"cv mismatch at node {i}: batch={float(cv_batch[i])}, serial={float(cv_serial)}"
            )
            assert jnp.allclose(cc_batch[i], cc_serial, atol=1e-10), (
                f"cc mismatch at node {i}: batch={float(cc_batch[i])}, serial={float(cc_serial)}"
            )

    def test_simple_minlp(self):
        m = examples.example_simple_minlp()
        self._check_batch_serial_match(m)

    def test_pooling_haverly(self):
        m = examples.example_pooling_haverly()
        self._check_batch_serial_match(m)

    def test_process_synthesis(self):
        m = examples.example_process_synthesis()
        self._check_batch_serial_match(m)

    def test_reactor_design(self):
        m = examples.example_reactor_design()
        self._check_batch_serial_match(m)

    def test_parametric(self):
        m = examples.example_parametric()
        self._check_batch_serial_match(m)

    def test_simple_expression(self):
        m = Model("test")
        x = m.continuous("x", lb=0.1, ub=5)
        y = m.continuous("y", lb=0.1, ub=5)
        m.minimize(dm.exp(x) + dm.log(y) + x * y)

        self._check_batch_serial_match(m)


# ─────────────────────────────────────────────────────────────
# Test 2: Shape correctness
# ─────────────────────────────────────────────────────────────


class TestShapes:
    def test_output_shapes(self):
        m = Model("test")
        m.continuous("x", lb=0, ub=5)
        m.continuous("y", lb=0, ub=5)
        m.minimize(dm.exp(m._variables[0]) + m._variables[1] ** 2)

        evaluator = batch_evaluator_from_objective(m)
        assert evaluator.n_vars == 2

        batch_size = 64
        lb_batch, ub_batch = _make_batch_bounds(m, batch_size)

        cv, cc = evaluator.evaluate_batch(lb_batch, ub_batch)
        assert cv.shape == (batch_size,)
        assert cc.shape == (batch_size,)

    def test_single_element_batch(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=5)
        m.minimize(x**2)

        evaluator = batch_evaluator_from_objective(m)
        lb = jnp.array([[1.0]])
        ub = jnp.array([[4.0]])

        cv, cc = evaluator.evaluate_batch(lb, ub)
        assert cv.shape == (1,)
        assert cc.shape == (1,)

    def test_evaluate_batch_at_shapes(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x + y)

        evaluator = batch_evaluator_from_objective(m)
        batch_size = 16

        lb_batch, ub_batch = _make_batch_bounds(m, batch_size)
        mid = 0.5 * (lb_batch + ub_batch)

        cv, cc = evaluator.evaluate_batch_at(mid, mid, lb_batch, ub_batch)
        assert cv.shape == (batch_size,)
        assert cc.shape == (batch_size,)


# ─────────────────────────────────────────────────────────────
# Test 3: JIT compatibility
# ─────────────────────────────────────────────────────────────


class TestJitCompatibility:
    def test_jit_produces_same_results(self):
        m = Model("test")
        x = m.continuous("x", lb=0.1, ub=5)
        y = m.continuous("y", lb=0.1, ub=5)
        m.minimize(x**2 + dm.exp(y))

        evaluator = batch_evaluator_from_objective(m)
        lb_batch, ub_batch = _make_batch_bounds(m, 32)

        # The evaluator already uses jit internally
        cv1, cc1 = evaluator.evaluate_batch(lb_batch, ub_batch)

        # Call again to exercise JIT cache
        cv2, cc2 = evaluator.evaluate_batch(lb_batch, ub_batch)

        assert jnp.allclose(cv1, cv2, atol=1e-12)
        assert jnp.allclose(cc1, cc2, atol=1e-12)

    def test_second_call_faster(self):
        """After JIT compilation, subsequent calls should be faster."""
        m = Model("test")
        x = m.continuous("x", lb=0, ub=5)
        y = m.continuous("y", lb=0, ub=5)
        m.minimize(x**2 + y**2 + x * y)

        evaluator = batch_evaluator_from_objective(m)
        lb_batch, ub_batch = _make_batch_bounds(m, 128)

        # Warm-up: triggers JIT compilation
        evaluator.evaluate_batch(lb_batch, ub_batch)

        # Timed run after JIT
        start = time.perf_counter()
        for _ in range(100):
            evaluator.evaluate_batch(lb_batch, ub_batch)
        jit_time = time.perf_counter() - start

        # JIT-compiled call for 100 iterations should be fast
        assert jit_time < 5.0, f"JIT calls too slow: {jit_time:.3f}s for 100 iters"


# ─────────────────────────────────────────────────────────────
# Test 4: Soundness (cv <= f(x) <= cc)
# ─────────────────────────────────────────────────────────────


class TestSoundness:
    def test_cv_leq_cc(self):
        """cv should be <= cc for all nodes in the batch."""
        m = Model("test")
        x = m.continuous("x", lb=0.5, ub=5)
        y = m.continuous("y", lb=0.5, ub=5)
        m.minimize(x * y + dm.exp(x))

        evaluator = batch_evaluator_from_objective(m)
        lb_batch, ub_batch = _make_batch_bounds(m, 256)

        cv, cc = evaluator.evaluate_batch(lb_batch, ub_batch)
        assert jnp.all(cv <= cc + 1e-8), "cv > cc at some nodes"

    def test_bounds_contain_true_value(self):
        """cv <= f(midpoint) <= cc at every node."""
        m = Model("test")
        x = m.continuous("x", lb=0.1, ub=5)
        y = m.continuous("y", lb=0.1, ub=5)
        m.minimize(dm.exp(x) + dm.log(y))

        evaluator = batch_evaluator_from_objective(m)
        true_fn = compile_expression(m._objective.expression, m)

        batch_size = 128
        lb_batch, ub_batch = _make_batch_bounds(m, batch_size)

        cv, cc = evaluator.evaluate_batch(lb_batch, ub_batch)

        # Check that true function value at midpoint is within [cv, cc]
        for i in range(batch_size):
            mid_i = 0.5 * (lb_batch[i] + ub_batch[i])
            true_val = true_fn(mid_i)
            assert float(cv[i]) <= float(true_val) + 1e-8, (
                f"cv > f(x) at node {i}: cv={float(cv[i])}, f={float(true_val)}"
            )
            assert float(cc[i]) >= float(true_val) - 1e-8, (
                f"cc < f(x) at node {i}: cc={float(cc[i])}, f={float(true_val)}"
            )


# ─────────────────────────────────────────────────────────────
# Test 5: Multiple batch sizes
# ─────────────────────────────────────────────────────────────


class TestBatchSizes:
    @pytest.mark.parametrize("batch_size", [1, 32, 64, 128, 512])
    def test_various_batch_sizes(self, batch_size: int):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=5)
        y = m.continuous("y", lb=0, ub=5)
        m.minimize(x**2 + y**2)

        evaluator = batch_evaluator_from_objective(m)
        lb_batch, ub_batch = _make_batch_bounds(m, batch_size)

        cv, cc = evaluator.evaluate_batch(lb_batch, ub_batch)
        assert cv.shape == (batch_size,)
        assert cc.shape == (batch_size,)
        assert jnp.all(cv <= cc + 1e-8)


# ─────────────────────────────────────────────────────────────
# Test 6: Performance (vmap vs serial loop)
# ─────────────────────────────────────────────────────────────


class TestPerformance:
    @pytest.mark.slow
    def test_vmap_faster_than_serial(self):
        """vmap batch evaluation should be faster than a serial Python loop."""
        m = Model("test")
        x = m.continuous("x", lb=0.1, ub=5)
        y = m.continuous("y", lb=0.1, ub=5)
        m.minimize(dm.exp(x) + dm.log(y) + x * y)

        evaluator = batch_evaluator_from_objective(m)
        relax_fn = compile_relaxation(m._objective.expression, m)
        jit_relax = jax.jit(relax_fn)

        batch_size = 512
        lb_batch, ub_batch = _make_batch_bounds(m, batch_size)

        # Warm up
        evaluator.evaluate_batch(lb_batch, ub_batch)
        mid = 0.5 * (lb_batch[0] + ub_batch[0])
        jit_relax(mid, mid, lb_batch[0], ub_batch[0])

        # Time vmap batch
        n_iters = 50
        start = time.perf_counter()
        for _ in range(n_iters):
            evaluator.evaluate_batch(lb_batch, ub_batch)
        vmap_time = time.perf_counter() - start

        # Time serial loop
        start = time.perf_counter()
        for _ in range(n_iters):
            for i in range(batch_size):
                mid_i = 0.5 * (lb_batch[i] + ub_batch[i])
                jit_relax(mid_i, mid_i, lb_batch[i], ub_batch[i])
        serial_time = time.perf_counter() - start

        speedup = serial_time / vmap_time
        # vmap should be at least 2x faster for batch_size=512
        assert speedup > 2.0, (
            f"vmap speedup too low: {speedup:.1f}x "
            f"(vmap={vmap_time:.3f}s, serial={serial_time:.3f}s)"
        )


# ─────────────────────────────────────────────────────────────
# Test 7: Factory functions
# ─────────────────────────────────────────────────────────────


class TestFactoryFunctions:
    def test_from_objective(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=5)
        m.minimize(x**2)

        evaluator = batch_evaluator_from_objective(m)
        assert evaluator.n_vars == 1
        assert isinstance(evaluator, BatchRelaxationEvaluator)

    def test_from_expression(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=5)
        y = m.continuous("y", lb=0, ub=5)
        m.minimize(x)

        expr = dm.exp(x) + y
        evaluator = batch_evaluator_from_expression(expr, m)
        assert evaluator.n_vars == 2

        lb_batch, ub_batch = _make_batch_bounds(m, 16)
        cv, cc = evaluator.evaluate_batch(lb_batch, ub_batch)
        assert cv.shape == (16,)

    def test_from_constraint(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=5)
        y = m.continuous("y", lb=0, ub=5)
        m.minimize(x)

        constraint = x**2 + y <= 10
        evaluator = batch_evaluator_from_constraint(constraint, m)
        assert evaluator.n_vars == 2

        lb_batch, ub_batch = _make_batch_bounds(m, 16)
        cv, cc = evaluator.evaluate_batch(lb_batch, ub_batch)
        assert cv.shape == (16,)

    def test_no_objective_raises(self):
        m = Model("test")
        m.continuous("x", lb=0, ub=1)
        with pytest.raises(ValueError, match="no objective"):
            batch_evaluator_from_objective(m)


# ─────────────────────────────────────────────────────────────
# Test 8: evaluate_batch_at matches evaluate_batch
# ─────────────────────────────────────────────────────────────


class TestEvaluateBatchAt:
    def test_midpoint_matches(self):
        """evaluate_batch_at with midpoints should match evaluate_batch."""
        m = Model("test")
        x = m.continuous("x", lb=0, ub=5)
        y = m.continuous("y", lb=0, ub=5)
        m.minimize(x**2 + y**2)

        evaluator = batch_evaluator_from_objective(m)
        lb_batch, ub_batch = _make_batch_bounds(m, 32)
        mid = 0.5 * (lb_batch + ub_batch)

        cv1, cc1 = evaluator.evaluate_batch(lb_batch, ub_batch)
        cv2, cc2 = evaluator.evaluate_batch_at(mid, mid, lb_batch, ub_batch)

        assert jnp.allclose(cv1, cv2, atol=1e-12)
        assert jnp.allclose(cc1, cc2, atol=1e-12)


# ─────────────────────────────────────────────────────────────
# Test 9: Example models with larger batch sizes
# ─────────────────────────────────────────────────────────────


class TestExampleModels:
    @pytest.mark.slow
    def test_portfolio(self):
        m = examples.example_portfolio()
        evaluator = batch_evaluator_from_objective(m)
        lb_batch, ub_batch = _make_batch_bounds(m, 64)

        cv, cc = evaluator.evaluate_batch(lb_batch, ub_batch)
        assert cv.shape == (64,)
        assert cc.shape == (64,)
        assert jnp.all(cv <= cc + 1e-6)

    @pytest.mark.slow
    def test_facility_location(self):
        m = examples.example_facility_location()
        evaluator = batch_evaluator_from_objective(m)
        lb_batch, ub_batch = _make_batch_bounds(m, 64)

        cv, cc = evaluator.evaluate_batch(lb_batch, ub_batch)
        assert cv.shape == (64,)
        assert cc.shape == (64,)
