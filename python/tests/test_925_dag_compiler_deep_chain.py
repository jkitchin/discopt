"""Regression tests for #925: the DAG compiler must not overflow the C stack.

``dag_compiler`` used to lower an Expression by recursing once per node
(``_compile_node`` -> ``_raw_node`` -> ``_compile_node`` per child) into *nested
child closures*, so the same depth was pushed onto the stack again at every
evaluation. A plain Python ``sum``/``+=`` over a list of terms builds a
**left-nested** ``BinaryOp`` chain whose depth equals the term count, so both
lowering and evaluating one raised ``RecursionError``.

Measured on ``main`` @ 5800bfc (CPython 3.11, ``sys.getrecursionlimit() == 1000``)
with the reproduction in #925: n=200 ok, n=300 ok, n=500 ``RecursionError``. The
exact cliff moves with the interpreter and with how deep the caller already is,
but it is far below any real model size — a #75 differential sweep skipped
``edgecross10-030``/``-040``/``-090`` on it.

The fix flattens the DAG once into a post-order tape and evaluates it with a
flat loop, so compile *and* eval depth are bounded by the heap. These tests use
left-nested chains an order of magnitude past the observed cliff, which is the
shape that actually breaks — a balanced tree of the same term count never
reproduced the defect (see the workaround in
``test_923_dense_lagrangian_hessian.py``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from discopt._jax.dag_compiler import (
    _build_param_index,
    _build_tape,
    compile_constraint,
    compile_expression,
    compile_expression_params,
    compile_objective,
)
from discopt.modeling.core import BinaryOp, Model

# Comfortably past the n=500 cliff measured on the pre-fix compiler, and past the
# ~1000-frame default recursion limit that produced it.
DEEP = 5000
# Reverse mode is split across two depths on purpose. Tracing `jax.grad` is
# linear and cheap (measured: 0.44 s at n=300, 0.99 s at n=600, 2.8 s at
# n=2000) and is the step that would hit the recursion cliff, so the *depth*
# probe traces at DEEP_AD. Actually running the gradient pays XLA compilation,
# which is steeply superlinear in HLO op count (2.3 s at n=60, 8.5 s at n=200,
# 340 s at n=5000) for reasons that have nothing to do with this compiler, so
# the *numeric* probe stays at SHALLOW_AD.
DEEP_AD = 2000
SHALLOW_AD = 120


def _left_nested_sum(terms):
    """Build ``((t0 + t1) + t2) + ...`` — what ``sum(xs)``/``+=`` produces.

    Depth equals ``len(terms)``, unlike a balanced tree's ``log2(len(terms))``.
    """
    acc = terms[0]
    for t in terms[1:]:
        acc = acc + t
    return acc


def _chain_model(n: int, name: str = "deep") -> tuple[Model, list]:
    m = Model(name)
    xs = [m.continuous(f"x{i}", lb=0.0, ub=1.0) for i in range(n)]
    return m, xs


def _assert_left_nested(expr, expected_depth: int) -> None:
    """Fail loudly if the fixture degenerated into a shallow/balanced tree.

    Without this the tests would still pass on the *pre-fix* compiler if the
    modeling layer ever started folding chains, and would prove nothing.
    """
    depth = 0
    node = expr
    while isinstance(node, BinaryOp):
        depth += 1
        node = node.left
    assert depth == expected_depth, f"chain depth {depth}, expected {expected_depth}"


def test_left_nested_objective_compiles_and_evaluates():
    """The #925 reproduction, an order of magnitude past the measured cliff."""
    m, xs = _chain_model(DEEP)
    obj = _left_nested_sum(xs)
    _assert_left_nested(obj, DEEP - 1)
    m.minimize(obj)

    fn = compile_objective(m)
    assert float(fn(jnp.ones(DEEP))) == pytest.approx(float(DEEP))

    rng = np.random.default_rng(0)
    pt = rng.uniform(0.0, 1.0, size=DEEP)
    assert float(fn(jnp.asarray(pt))) == pytest.approx(float(pt.sum()), rel=1e-6)


def test_left_nested_constraint_body_compiles_and_evaluates():
    """The same shape reached through ``compile_constraint`` rather than the objective."""
    m, xs = _chain_model(DEEP, "deep_con")
    body = _left_nested_sum(xs)
    _assert_left_nested(body, DEEP - 1)
    m.subject_to(body <= float(DEEP))
    m.minimize(xs[0])

    con = m._constraints[0]
    # ``subject_to`` normalizes to ``sum(x) - DEEP <= 0``, one level deeper.
    _assert_left_nested(con.body, DEEP)

    fn = compile_constraint(con, m)
    assert float(fn(jnp.ones(DEEP))) == pytest.approx(0.0, abs=1e-6)

    rng = np.random.default_rng(2)
    pt = rng.uniform(0.0, 1.0, size=DEEP)
    assert float(fn(jnp.asarray(pt))) == pytest.approx(float(pt.sum()) - DEEP, rel=1e-6)


def test_left_nested_mixed_operators_match_numpy():
    """Depth is not specific to ``+``: mix the binary ops and a unary/function layer."""
    n = DEEP
    m, xs = _chain_model(n, "deep_mixed")
    rng = np.random.default_rng(1)
    pt = rng.uniform(0.2, 0.9, size=n)

    acc = xs[0]
    expected = pt[0]
    for i in range(1, n):
        which = i % 3
        if which == 0:
            acc = acc + xs[i]
            expected = expected + pt[i]
        elif which == 1:
            acc = acc - xs[i]
            expected = expected - pt[i]
        else:
            # Keep the running value bounded so the comparison stays meaningful.
            acc = acc * 1.0 + xs[i] * 0.5
            expected = expected * 1.0 + pt[i] * 0.5
    deep = abs(acc)
    m.minimize(deep)

    fn = compile_objective(m)
    assert float(fn(jnp.asarray(pt))) == pytest.approx(float(abs(expected)), rel=1e-5)


def test_deep_reverse_mode_trace_survives_depth():
    """``jax.grad`` re-enters the tape to build its jaxpr — that trace is the
    step that hit the cliff, so it must survive well past it."""
    m, xs = _chain_model(DEEP_AD, "deep_ad")
    m.minimize(_left_nested_sum(xs))
    fn = compile_objective(m)

    jaxpr = jax.make_jaxpr(jax.grad(fn))(jnp.ones(DEEP_AD))
    # One primal add per link, plus the reverse sweep — the point is that the
    # whole chain is present rather than truncated or folded away.
    assert len(jaxpr.jaxpr.eqns) >= DEEP_AD


def test_gradient_of_left_nested_sum_is_correct():
    """AD through the tape must give the right numbers: d/dx_i of sum(x) is 1.

    Shallow by design (see ``SHALLOW_AD``); this guards the rewritten kernels'
    differentiability, not the depth property.
    """
    m, xs = _chain_model(SHALLOW_AD, "shallow_grad")
    m.minimize(_left_nested_sum(xs))
    fn = compile_objective(m)

    g = np.asarray(jax.grad(fn)(jnp.ones(SHALLOW_AD)))
    assert g.shape == (SHALLOW_AD,)
    np.testing.assert_allclose(g, np.ones(SHALLOW_AD), rtol=0, atol=1e-9)


def test_left_nested_params_path_compiles_and_evaluates():
    """``compile_expression_params`` threads parameters and must handle the same depth."""
    n = DEEP
    m, xs = _chain_model(n, "deep_params")
    scale = m.parameter("scale", value=2.0)
    expr = _left_nested_sum([x * scale for x in xs])

    fn = compile_expression_params(expr, m)
    params = (jnp.asarray(3.0),)
    assert float(fn(jnp.ones(n), params)) == pytest.approx(3.0 * n)
    # Same trace, new parameter value — no recompile, no recursion.
    assert float(fn(jnp.ones(n), (jnp.asarray(0.5),))) == pytest.approx(0.5 * n)


def test_deep_shared_subexpression_still_evaluates_once():
    """Depth-independence must not cost the #383 common-subexpression sharing.

    The chain is referenced twice at the root; a tape that lost node sharing
    would emit ~2x the slots (and, before #383, lower exponentially).
    """
    n = 400
    m, xs = _chain_model(n, "deep_shared")
    chain = _left_nested_sum(xs)
    expr = chain + chain

    tape = _build_tape(expr, m, _build_param_index(m))
    # n leaves + (n-1) chain adds + 1 root add. One slot per *distinct* node.
    assert len(tape) == n + (n - 1) + 1

    fn = compile_expression(expr, m)
    assert float(fn(jnp.ones(n))) == pytest.approx(2.0 * n)


def test_cyclic_expression_graph_is_refused_loudly():
    """A self-referential node has no valid post-order; refuse rather than emit
    a tape with a dangling child slot."""
    m, xs = _chain_model(3, "cyclic")
    node = xs[0] + xs[1]
    assert isinstance(node, BinaryOp)
    node.left = node  # not reachable through the modeling API; defensive guard

    with pytest.raises(ValueError, match="Cyclic expression graph"):
        _build_tape(node, m, _build_param_index(m))
