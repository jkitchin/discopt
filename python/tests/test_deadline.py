"""Tests for the JAX-side wall-clock deadline plumbing (issue #80).

These verify that ``deadline_scope`` actually interrupts a JAX-compiled
``while_loop`` mid-execution, that an unset deadline leaves convergence
behavior unchanged, and that the host-callback predicate behaves on the
process-global state.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.deadline import (
    clear_deadline,
    deadline_exceeded,
    deadline_exceeded_jax,
    deadline_scope,
    set_deadline,
)


@pytest.fixture(autouse=True)
def _reset_deadline():
    """Ensure no leaked deadline from a previous test."""
    clear_deadline()
    yield
    clear_deadline()


def test_deadline_unset_returns_false():
    assert deadline_exceeded() is False


def test_deadline_scope_set_and_clear():
    with deadline_scope(60.0):
        assert deadline_exceeded() is False  # 60s in the future
    assert deadline_exceeded() is False  # cleared on exit


def test_deadline_scope_trips_after_sleep():
    with deadline_scope(0.05):
        assert deadline_exceeded() is False
        time.sleep(0.1)
        assert deadline_exceeded() is True


def test_set_deadline_none_clears():
    set_deadline(60.0)
    set_deadline(None)
    assert deadline_exceeded() is False


def test_deadline_exceeded_jax_in_while_loop_terminates_loop():
    """A jax.lax.while_loop guarded by deadline_exceeded_jax() must exit
    promptly once the deadline trips, instead of running to its bound."""

    @jax.jit
    def run(n_max):
        def cond(c):
            return jnp.logical_and(c < n_max, jnp.logical_not(deadline_exceeded_jax()))

        def body(c):
            return c + 1

        return jax.lax.while_loop(cond, body, jnp.int32(0))

    # Warm-compile under a wide deadline.
    with deadline_scope(60.0):
        _ = int(run(jnp.int32(1_000)))

    # Now trip the deadline mid-loop.
    with deadline_scope(0.1):
        t0 = time.perf_counter()
        iters = int(run(jnp.int32(10_000_000)))
        elapsed = time.perf_counter() - t0

    assert iters < 10_000_000, "while_loop ran to its bound; deadline was not honored"
    assert elapsed < 2.0, f"deadline trip took {elapsed:.2f}s — should be ≲ 1s"


def test_lp_ipm_honors_deadline():
    """LP IPM (lp_ipm.py main solve) must self-terminate on deadline."""
    from discopt._jax.lp_ipm import LPIPMOptions, lp_ipm_solve

    # Small, well-conditioned LP so without a deadline the solver converges
    # quickly; we then re-run with a tight deadline and verify the wall is
    # bounded.
    rng = np.random.default_rng(0)
    n, m = 20, 5
    A = rng.standard_normal((m, n))
    x_star = np.abs(rng.standard_normal(n))
    b = A @ x_star
    c = rng.standard_normal(n)
    x_l = jnp.zeros(n)
    x_u = jnp.full(n, 1e6)

    # Warm-compile with the SAME options used in the timed call below.
    # max_iter is a static_argnum for _lp_ipm_solve_jit, so a different value
    # triggers a recompile that can dominate the wall-clock measurement on CI.
    state_ok = lp_ipm_solve(
        jnp.asarray(c),
        jnp.asarray(A),
        jnp.asarray(b),
        x_l,
        x_u,
        options=LPIPMOptions(max_iter=10_000),
    )
    assert int(state_ok.converged) in (1, 2, 3)

    # Trip the deadline before calling.
    set_deadline(-1.0)  # already past
    t0 = time.perf_counter()
    _ = lp_ipm_solve(
        jnp.asarray(c),
        jnp.asarray(A),
        jnp.asarray(b),
        x_l,
        x_u,
        options=LPIPMOptions(max_iter=10_000),  # would otherwise iterate long
    )
    elapsed = time.perf_counter() - t0
    clear_deadline()

    # The loop should exit immediately. converged stays at 0 (running) when
    # the deadline trips before the first iteration completes — that's the
    # expected signal to callers that the solve was preempted.
    assert elapsed < 1.0, f"deadline-tripped LP IPM took {elapsed:.2f}s"


def test_lp_ipm_unchanged_when_no_deadline():
    """A normal solve (no deadline) must still converge to optimal."""
    from discopt._jax.lp_ipm import LPIPMOptions, lp_ipm_solve

    rng = np.random.default_rng(1)
    n, m = 15, 4
    A = rng.standard_normal((m, n))
    x_star = np.abs(rng.standard_normal(n))
    b = A @ x_star
    c = rng.standard_normal(n)

    state = lp_ipm_solve(
        jnp.asarray(c),
        jnp.asarray(A),
        jnp.asarray(b),
        jnp.zeros(n),
        jnp.full(n, 1e6),
        options=LPIPMOptions(max_iter=200),
    )
    assert int(state.converged) in (1, 2)
