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
