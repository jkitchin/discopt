"""Regression for #383: expression -> jaxpr lowering must CSE shared DAG nodes.

An expression built by repeatedly wrapping the *same* object is a linear DAG
(each node shared, so O(depth) distinct nodes) whose *tree* unfolding is ~3^depth.
The lowering must memoize by node identity so compile + Hessian-build stay
polynomial in the DAG size; before #383 they were exponential in the sharing
depth (depth 10 took ~55 s; depth 14 was intractable).
"""

import time

import pytest

jax = pytest.importorskip("jax")
import discopt.modeling as dm  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from discopt._jax.dag_compiler import compile_expression  # noqa: E402


def _shared_dag(depth):
    m = dm.Model(f"cse{depth}")
    x = m.continuous("x", lb=-2, ub=2)
    d = x
    for _ in range(depth):
        d = 0.5 * (d + 0.05 * d * d)  # d reused -> shared object -> linear DAG
    m.minimize(d)
    return m


def _ref(xv, depth):
    d = xv
    for _ in range(depth):
        d = 0.5 * (d + 0.05 * d * d)
    return d


@pytest.mark.parametrize("depth", [4, 8, 12])
def test_shared_dag_lowers_to_correct_value(depth):
    """CSE must not change results: the lowered value matches the direct recurrence."""
    m = _shared_dag(depth)
    f = compile_expression(m._objective.expression, m)
    assert float(f(jnp.array([0.7]))) == pytest.approx(_ref(0.7, depth), rel=1e-9)


def test_shared_dag_hessian_build_is_not_exponential():
    """A depth-10 shared DAG (tree unfolding ~3^10) must build a Hessian fast.

    With CSE this is milliseconds; an un-memoized lowering took ~55 s here, so a
    generous 15 s ceiling cleanly separates the two without being timing-fragile.
    """
    m = _shared_dag(10)
    f = compile_expression(m._objective.expression, m)
    x0 = jnp.array([0.5])
    t0 = time.perf_counter()
    hess = jax.hessian(lambda xf: f(xf))(x0)
    hess.block_until_ready()
    assert time.perf_counter() - t0 < 15.0
