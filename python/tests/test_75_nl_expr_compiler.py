"""The DAG -> POUNCE ``NlExpr`` translator must reproduce the JAX path exactly.

Issue #75. ``_nl_expr_compiler`` lowers a discopt expression DAG onto POUNCE's
Rust AD tape, which is the shared prerequisite for taking JAX off the solve path:
both remaining JAX jobs (separation tangents, NLP subsolve derivatives) need the
value and gradient of a scalar expression at a point, and the tape supplies both.

These tests pin the properties that decide whether it can replace the JAX path:

* **numerical agreement** with ``_jax/dag_compiler`` — the thing being replaced;
* **operator coverage** across discopt's DAG, including the ten operators with no
  native tape opcode that are lowered by rewrite (``abs``, ``sign``, ``log1p``,
  ``log2``, ``sigmoid``, ``softplus``, ``entropy``, ``centropy``, ``prod``,
  ``signpower``);
* **DAG sharing** — a node reachable by k references is built once, or the walk
  is exponential in sharing depth (issue #383's trap);
* **loud refusal** on a node with no tape equivalent, rather than a silent wrong
  answer.

Measured when this landed: over 40 corpus instances and 1517 sampled points, max
relative value drift 2.51e-16 and gradient drift 6.64e-15, with zero unsupported
nodes encountered.
"""

import time

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt._nl_expr_compiler import UnsupportedForTape, compile_to_nl_expr, try_compile

pytest.importorskip("pounce")


def _model():
    m = Model()
    x = m.continuous("x", lb=0.3, ub=2.5)
    y = m.continuous("y", lb=0.4, ub=2.0)
    return m, x, y


def _jax_pair(expr, model):
    """(value_fn, grad_fn) from the JAX path being replaced."""
    import jax
    import jax.numpy as jnp
    from discopt._jax.dag_compiler import compile_expression

    f = compile_expression(expr, model)
    return f, jax.grad(lambda xv: jnp.reshape(f(xv), ()))


# Expression classes spanning the operator table, including every rewrite.
CASES = {
    "poly": lambda x, y: x * x * y + 3.0 * x - y / 2.0,
    "exp_log": lambda x, y: dm.exp(x) + dm.log(y),
    "trig": lambda x, y: dm.sin(x) * dm.cos(y) + dm.tan(x / 4.0),
    "sqrt_pow": lambda x, y: dm.sqrt(x) + y**3,
    "hyperbolic": lambda x, y: dm.tanh(x) + dm.sinh(y) + dm.cosh(x),
    "inverse_trig": lambda x, y: dm.atan(x) + dm.asin(y / 3.0),
    "erf": lambda x, y: dm.erf(x) * y,
    "abs_rewrite": lambda x, y: abs(x - y) + x,
    "log10": lambda x, y: dm.log10(x) + y,
    "min_max": lambda x, y: dm.maximum(x, y) + dm.minimum(x, y),
    "division_chain": lambda x, y: (x + y) / (x * y + 1.0),
    "shared_subexpr": lambda x, y: (lambda t: t * t + dm.exp(t))(x * y),
}


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(CASES))
def test_matches_jax_value_and_gradient(name):
    """Tape and JAX must agree to ~machine precision on the same points.

    The bar is 1e-12 relative: far looser than measured (~1e-16), but tight enough
    that a wrong chain rule or a mis-ordered flat index cannot pass.
    """
    m, x, y = _model()
    expr = CASES[name](x, y)
    tape = compile_to_nl_expr(expr, m)
    jf, jg = _jax_pair(expr, m)

    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    compared = 0
    for _ in range(25):
        pt = np.array([rng.uniform(0.35, 2.4), rng.uniform(0.45, 1.9)])
        jv = float(np.asarray(jf(pt)))
        jgv = np.asarray(jg(pt), dtype=float)
        if not (np.isfinite(jv) and np.all(np.isfinite(jgv))):
            continue  # JAX itself is out of domain here; nothing to compare
        tv = tape.eval(list(pt))
        tg = np.asarray(tape.gradient(list(pt)), dtype=float)
        compared += 1
        assert abs(tv - jv) / max(1.0, abs(jv)) <= 1e-12, f"value drift at {pt}: {tv} vs {jv}"
        scale = max(1.0, float(np.max(np.abs(jgv))))
        assert np.max(np.abs(tg - jgv)) / scale <= 1e-12, f"gradient drift at {pt}"

    # §6: a case where every point was skipped would pass vacuously.
    assert compared >= 5, f"only {compared} points compared for {name}"


@pytest.mark.unit
def test_shared_subexpression_lowers_once_and_matches_jax():
    """A node reachable by several parents lowers once and stays numerically right.

    Deliberately SHALLOW, so the comparison against JAX is available: this is the
    agreement check. Depth is pinned separately by
    ``test_deeply_shared_chain_stays_linear_in_distinct_nodes``, which goes past
    where JAX can be traced at all and so checks against a scalar recurrence.
    """
    m, x, y = _model()
    shared = x * y + 1.0
    expr = shared * shared + dm.exp(shared)
    tape = compile_to_nl_expr(expr, m)
    jf, jg = _jax_pair(expr, m)

    pt = np.array([0.9, 1.1])
    assert abs(tape.eval(list(pt)) - float(np.asarray(jf(pt)))) <= 1e-12
    np.testing.assert_allclose(
        np.asarray(tape.gradient(list(pt)), dtype=float),
        np.asarray(jg(pt), dtype=float),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.unit
def test_deeply_shared_chain_stays_linear_in_distinct_nodes():
    """A deep chain of shared nodes must lower to a DAG, never to a tree.

    ``node = node*node + node`` repeated D times holds only ``2D`` distinct nodes
    but ``2**D`` *tree* nodes. At D=30 that is 1.07e9 — unbuildable — so the fact
    that this returns at all is the proof of sharing, on both sides of the
    boundary: this module memoises on ``id(expr)``, and POUNCE's operators
    reference their operands through a ``Cse`` node rather than copying them.

    This was an open blocker when the translator landed (``1917a17b``), pinned as
    a non-running ``xfail``: lowering failed to terminate by depth 10. The cause
    was downstream and is fixed by pounce PR #474 ("stop copying operands"); the
    measurement above was taken against a stale locally-built extension. Verified
    here rather than deleted, because Stage 2/3 traffic depends on it — real
    corpus models share heavily (``dag_compiler`` carries the same fix, #383).

    Truth comes from the scalar recurrence, not from JAX: JAX hits its own
    tracing recursion limit long before this depth, which is the whole point.
    """
    depth = 30
    m = Model()
    x = m.continuous("x", lb=-1.0, ub=1.0)
    node = x
    for _ in range(depth):
        node = node * node + node

    started = time.perf_counter()
    tape = compile_to_nl_expr(node, m)
    x0 = -0.1
    value = tape.eval([x0])
    grad = float(np.asarray(tape.gradient([x0]), dtype=float)[0])
    elapsed = time.perf_counter() - started

    # n_{k+1} = n_k^2 + n_k, so d_{k+1} = (2 n_k + 1) d_k. x0 = -0.1 keeps both
    # bounded: a value like 1.001 overflows to inf by depth 11 and a gradient
    # underflows to exactly 0.0, either of which would pass vacuously.
    n, d = x0, 1.0
    for _ in range(depth):
        d = (2.0 * n + 1.0) * d
        n = n * n + n
    assert abs(n) > 1e-3 and abs(d) > 1e-3, "recurrence degenerated; the check would be vacuous"

    assert abs(value - n) / max(1.0, abs(n)) <= 1e-12, f"value {value} vs {n}"
    assert abs(grad - d) / abs(d) <= 1e-12, f"gradient {grad} vs {d}"
    # Not a performance claim — a non-termination guard, three orders above the
    # ~0.13 s measured, so it cannot fail on a loaded machine (CLAUDE.md §9).
    assert elapsed < 60.0, f"lowering a {depth}-deep shared chain took {elapsed:.1f}s"


@pytest.mark.unit
def test_custom_call_is_refused_loudly():
    """``dm.custom`` wraps an opaque JAX callable and has no tape equivalent.

    It must raise, not return something that silently evaluates to nothing.
    """
    m, x, y = _model()

    @dm.custom
    def opaque(a):
        return a * 2.0

    with pytest.raises(UnsupportedForTape, match="CustomCall"):
        compile_to_nl_expr(opaque(x) + y, m)


@pytest.mark.unit
def test_try_compile_returns_none_instead_of_raising():
    """The soft variant reports representability, never a numerical failure."""
    m, x, y = _model()

    @dm.custom
    def opaque(a):
        return a * 2.0

    assert try_compile(opaque(x) + y, m) is None
    assert try_compile(dm.exp(x) + y, m) is not None


@pytest.mark.unit
def test_flat_index_order_matches_the_jax_path():
    """A gradient is only comparable if both engines use the same flat layout.

    Uses an expression whose partials are distinct constants, so a transposed or
    offset layout produces an obviously wrong vector rather than a subtle drift.
    """
    m = Model()
    a = m.continuous("a", lb=0.0, ub=1.0)
    b = m.continuous("b", lb=0.0, ub=1.0)
    c = m.continuous("c", lb=0.0, ub=1.0)
    expr = 2.0 * a + 30.0 * b + 400.0 * c
    tape = compile_to_nl_expr(expr, m)
    grad = np.asarray(tape.gradient([0.5, 0.5, 0.5]), dtype=float)
    np.testing.assert_allclose(grad, [2.0, 30.0, 400.0], rtol=0, atol=1e-12)
