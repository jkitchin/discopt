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

import math
import time
from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt._jax.dag_compiler import compile_expression
from discopt._nl_expr_compiler import UnsupportedForTape, compile_to_nl_expr, try_compile
from discopt.modeling.core import Constant, FunctionCall, IndexExpression

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


# Expression classes spanning the operator table.
#
# The REWRITE_CASES half is load-bearing and cannot be replaced by a corpus run.
# Measured over 316 MINLPLib instances, a `.nl` corpus exercises exactly six
# operators — log, sqrt, exp, abs, sin, cos — because `.nl` has no opcode for
# sigmoid/softplus/entropy/centropy/signpower at all. Those reach the DAG only
# through the modeling API and `factorable_reform`. Three defects lived in that
# blind spot: entropy's sign was inverted, `_sign` passed `compare`'s arguments
# in the wrong order (a TypeError, which escapes `try_compile`'s fallback), and
# `prod` was lowered as a variadic `*` chain when it is a one-argument array
# reduction. `test_every_rewrite_is_covered` keeps this list honest.
CASES = {
    "poly": lambda x, y: x * x * y + 3.0 * x - y / 2.0,
    "exp_log": lambda x, y: dm.exp(x) + dm.log(y),
    "trig": lambda x, y: dm.sin(x) * dm.cos(y) + dm.tan(x / 4.0),
    "sqrt_pow": lambda x, y: dm.sqrt(x) + y**3,
    "hyperbolic": lambda x, y: dm.tanh(x) + dm.sinh(y) + dm.cosh(x),
    "inverse_trig": lambda x, y: dm.atan(x) + dm.asin(y / 3.0),
    "erf": lambda x, y: dm.erf(x) * y,
    "log10": lambda x, y: dm.log10(x) + y,
    "min_max": lambda x, y: dm.maximum(x, y) + dm.minimum(x, y),
    "division_chain": lambda x, y: (x + y) / (x * y + 1.0),
    "shared_subexpr": lambda x, y: (lambda t: t * t + dm.exp(t))(x * y),
}

# One entry per operator lowered by rewrite rather than by a native tape opcode.
# Keyed by the DAG's `func_name` so the completeness check below can compare
# against the compiler's own table.
REWRITE_CASES = {
    "abs": lambda x, y: abs(x - y) + x,
    "sign": lambda x, y: dm.sign(x - y) + y,
    "log1p": lambda x, y: dm.log1p(x) + y,
    "log2": lambda x, y: dm.log2(x) + y,
    "sigmoid": lambda x, y: dm.sigmoid(x) + y,
    "softplus": lambda x, y: dm.softplus(x) + y,
    "entropy": lambda x, y: FunctionCall("entropy", x) + y,
    "centropy": lambda x, y: FunctionCall("centropy", x, y),
    "signpower": lambda x, y: FunctionCall("signpower", x, Constant(3.0)) + y,
}
CASES.update(REWRITE_CASES)


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
def test_every_rewrite_is_covered():
    """Every operator lowered by rewrite must have a differential case.

    The rewrites are hand-written and are where the bugs were; the corpus cannot
    reach them (``.nl`` has no opcode for most). Without this guard, adding an
    11th rewrite silently ships untested — which is exactly how ``entropy``,
    ``sign`` and ``prod`` shipped wrong. Reads the compiler's source rather than
    a hand-copied list, so the two cannot drift apart.
    """
    import re

    from discopt import _nl_expr_compiler as mod

    src = Path(mod.__file__).read_text()
    body = src[src.index("def _lower_function") :]
    named = set(re.findall(r'name == "([a-z0-9_]+)"', body))
    # Refusals are not rewrites: they raise instead of lowering.
    refused = {"prod"}
    rewrites = named - refused

    assert rewrites, "found no rewrite branches; the regex no longer matches the source"
    missing = rewrites - set(REWRITE_CASES)
    assert not missing, f"rewrites with no differential case: {sorted(missing)}"


@pytest.mark.unit
@pytest.mark.parametrize("func_name", ["prod", "norm1", "norm2", "norminf"])
def test_array_reductions_are_refused_not_approximated(func_name):
    """An array reduction has no scalar tape lowering and must say so.

    ``prod`` is ``jnp.prod`` of ONE array argument, not a variadic multiply.
    Lowering it as a ``*`` chain computed a different function and agreed with
    nothing (measured: reldiff 0.90 on value, 1.70 on gradient) while reporting
    success. A missing tape is recoverable; a wrong one is not.
    """
    m, x, y = _model()
    expr = FunctionCall(func_name, x, y)
    with pytest.raises(UnsupportedForTape):
        compile_to_nl_expr(expr, m)
    assert try_compile(expr, m) is None


@pytest.mark.unit
def test_entropy_matches_the_dag_compiler_sign():
    """``entropy`` is ``x*log(x)`` in discopt's DAG, not ``-x*log(x)``.

    Pinned separately from the parametrized differential because the failure is a
    clean factor of -1: it passes every structural check, raises nothing, and is
    invisible to any corpus sweep. The authority is ``_jax/dag_compiler.py``.
    """
    m, x, _y = _model()
    tape = compile_to_nl_expr(FunctionCall("entropy", x), m)
    for x0 in (0.4, 1.0, 2.0):
        assert tape.eval([x0, 1.0]) == pytest.approx(x0 * np.log(x0), rel=1e-14)


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


# --------------------------------------------------------------------------------
# Numerical hardening of the rewrites
#
# The rewrites were verified for correct CALCULUS but never for numerical
# stability: `test_matches_jax_value_and_gradient` samples `uniform(0.35, 2.4)`,
# mid-domain, where every naive form agrees with its stabilized JAX counterpart to
# ~1e-16. A tail sweep found four defects that a mid-domain sample cannot see:
#
#   softplus  log(1+exp(a))  -> `exp(710)` is inf, so softplus(745) returned inf
#   log1p     log(1+a)       -> a=1e-17 returned 0.0 (every digit lost)
#   entropy   a*log(a)       -> a=0 returned nan, gradient -inf
#   centropy  a*log(a/b)     -> same, 3 non-finite values / 4 non-finite gradients
#
# The entropy/centropy points are REACHABLE: `factorable_reform._try_entropy`
# refuses only `lo < 0.0`, so a box with lower bound exactly 0 is admitted.
# --------------------------------------------------------------------------------

# Points that are extreme but strictly inside each operator's mathematical domain,
# so a non-finite result means an implementation defect and never a domain error.
_TAIL_POINTS = {
    "log1p": [-0.9, -1e-17, 0.0, 1e-17, 1e-13, 1e-8, 1.0, 700.0, 1e300],
    "sigmoid": [-1e300, -745.0, -300.0, -40.0, 0.0, 40.0, 300.0, 745.0, 1e300],
    "softplus": [-1e300, -745.0, -300.0, 0.0, 300.0, 710.0, 745.0, 1e300],
    "entropy": [0.0, 1e-300, 1e-30, 1e-5, 0.5, 1e300],
    "log2": [1e-300, 1e-8, 1.0, 1e300],
    "abs": [-1e300, -1.0, 0.0, 1.0, 1e300],
    "sign": [-1e300, -1.0, 0.0, 1.0, 1e300],
}


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(_TAIL_POINTS))
def test_rewrite_is_finite_wherever_the_authority_is(name):
    """No rewrite may return nan/inf at a point where `dag_compiler` is finite.

    This is the property the mid-domain test cannot reach. It is asserted only
    where JAX itself is finite, so a genuine domain edge (`log2(0)`) is not
    scored as a tape defect.
    """
    import jax
    import jax.numpy as jnp

    m = Model()
    x = m.continuous("x", lb=-1e309, ub=1e309)
    expr = {
        "log1p": lambda v: dm.log1p(v),
        "sigmoid": lambda v: dm.sigmoid(v),
        "softplus": lambda v: dm.softplus(v),
        "entropy": lambda v: FunctionCall("entropy", v),
        "log2": lambda v: dm.log2(v),
        "abs": lambda v: abs(v),
        "sign": lambda v: dm.sign(v),
    }[name](x)

    tape = compile_to_nl_expr(expr, m)
    jf = compile_expression(expr, m)
    jg = jax.grad(lambda xv: jnp.reshape(jf(xv), ()))

    checked = 0
    for p in _TAIL_POINTS[name]:
        pt = np.array([float(p)])
        jv = float(np.asarray(jf(pt)))
        jgv = float(np.asarray(jg(pt))[0])
        if not (np.isfinite(jv) and np.isfinite(jgv)):
            continue  # the authority is non-finite here; nothing to require
        tv = float(tape.eval([float(p)]))
        tgv = float(np.asarray(tape.gradient([float(p)]), dtype=float)[0])
        assert np.isfinite(tv), f"{name}: value {tv} at x={p} where jax gives {jv}"
        assert np.isfinite(tgv), f"{name}: gradient {tgv} at x={p} where jax gives {jgv}"
        checked += 2

    # §6: an operator whose every point was skipped would pass vacuously.
    assert checked >= 6, f"only {checked} finiteness checks ran for {name}"


@pytest.mark.unit
def test_softplus_does_not_overflow_in_the_upper_tail():
    """`log(1 + exp(a))` overflows for a >~ 710; the shifted form never
    exponentiates a positive argument. softplus(a) -> a for large a."""
    m = Model()
    x = m.continuous("x", lb=-1e309, ub=1e309)
    tape = compile_to_nl_expr(dm.softplus(x), m)

    for a in (710.0, 745.0, 1e6, 1e300):
        v = float(tape.eval([a]))
        assert np.isfinite(v), f"softplus({a}) = {v} (the naive form overflows here)"
        assert abs(v - a) <= 1e-9 * abs(a), f"softplus({a}) = {v}, expected ~{a}"

    # ...and the lower tail is not sacrificed to get it: softplus(-300) is
    # exp(-300), which `log(1 + exp(a))` collapsed to 0.0.
    assert abs(float(tape.eval([-300.0])) - math.exp(-300.0)) <= 1e-12 * math.exp(-300.0)


@pytest.mark.unit
def test_log1p_keeps_full_precision_for_tiny_arguments():
    """`log(1 + a)` loses every significant digit once `a` is below the rounding
    gap of 1.0; log1p(1e-17) must be 1e-17, not 0.0."""
    m = Model()
    x = m.continuous("x", lb=-1e309, ub=1e309)
    tape = compile_to_nl_expr(dm.log1p(x), m)

    for a in (1e-17, 1e-13, 1e-8, -1e-17):
        v = float(tape.eval([a]))
        assert abs(v - math.log1p(a)) <= 1e-15 * abs(math.log1p(a)), f"log1p({a}) = {v}"

    # The large-argument arm must stay correct too -- bounding the Kahan arm was
    # required because the quotient rule squares the denominator, and at a=1e300
    # that overflowed and returned a gradient of 6.918e-298 against a true 1e-300.
    g = float(np.asarray(tape.gradient([1e300]), dtype=float)[0])
    assert abs(g - 1e-300) <= 1e-310, f"log1p'(1e300) = {g}, expected 1e-300"


@pytest.mark.unit
@pytest.mark.parametrize("floored", ["entropy", "centropy"])
def test_xlogx_family_is_regularized_at_zero_like_the_authority(floored):
    """`x*log(x)` at x=0 is `0 * -inf = nan`, and its gradient is -inf.

    `dag_compiler` floors the log argument at 1e-300 precisely to keep both
    finite, and the tape must reproduce that: the point is reachable because
    `factorable_reform._try_entropy` admits a box whose lower bound is exactly 0.
    The value is the x->0+ limit (-0.0) and the gradient the floored stand-in.
    """
    import jax
    import jax.numpy as jnp

    m = Model()
    x = m.continuous("x", lb=-1e309, ub=1e309)
    if floored == "entropy":
        expr, pt = FunctionCall("entropy", x), [0.0]
    else:
        y = m.continuous("y", lb=-1e309, ub=1e309)
        expr, pt = FunctionCall("centropy", x, y), [0.0, 2.0]

    tape = compile_to_nl_expr(expr, m)
    jf = compile_expression(expr, m)
    jg = jax.grad(lambda xv: jnp.reshape(jf(xv), ()))

    tv = float(tape.eval(pt))
    tg = np.asarray(tape.gradient(pt), dtype=float)
    jv = float(np.asarray(jf(np.array(pt))))
    jgv = np.asarray(jg(np.array(pt)), dtype=float)

    assert np.isfinite(tv), f"{floored} value at x=0 is {tv} (nan means the floor is missing)"
    assert np.all(np.isfinite(tg)), f"{floored} gradient at x=0 is {tg}"
    assert abs(tv - jv) <= 1e-12, f"{floored} value {tv} vs authority {jv}"
    np.testing.assert_allclose(tg, jgv, rtol=1e-12, atol=0)


@pytest.mark.unit
def test_xlogx_residual_drift_is_a_subgradient_tie_not_error():
    """The one place entropy/centropy still disagree with JAX is `x == 1e-300`.

    That is exactly where `max(x, 1e-300)` switches branches, so the function is
    NOT differentiable there: `jnp.maximum` splits the tie 0.5/0.5 while pounce's
    `max` takes one branch. Both are valid subgradients, and the gap is therefore
    exactly 0.5 -- which is what makes the residual 7.24e-4 relative against
    log(1e-300) = -690.78, rather than a defect of that size.

    Pinned as arithmetic, not as a tolerance: a real error of this magnitude would
    slip through `assert drift < 1e-3`, and drift anywhere OFF the tie point would
    mean the floor moved or a branch was rewritten.
    """
    import jax
    import jax.numpy as jnp

    m = Model()
    x = m.continuous("x", lb=-1e309, ub=1e309)
    tape = compile_to_nl_expr(FunctionCall("entropy", x), m)
    jg = jax.grad(lambda v: (v[0] * jnp.log(jnp.maximum(v[0], 1e-300))).sum())

    checked = 0
    for p, expected_gap in ((1e-300, 0.5), (1e-320, 0.0), (1e-299, 0.0), (1e-8, 0.0)):
        tg = float(np.asarray(tape.gradient([p]), dtype=float)[0])
        jgv = float(np.asarray(jg(np.array([p])))[0])
        assert abs((tg - jgv) - expected_gap) <= 1e-9, (
            f"entropy'({p}): tape {tg} - jax {jgv} = {tg - jgv}, expected {expected_gap}"
        )
        checked += 1

    # 1e-320 is a subnormal STRICTLY below the floor, so max() is not at a tie
    # there and both backends take the constant branch -- gap 0, not 0.5.
    #
    # That point, not the tie itself, is what gives this test teeth: measured
    # against the pre-floor compiler, 1e-300 still produced a gap of exactly 0.5
    # (unfloored `log(x)+1` happens to sit 0.5 from the split subgradient) and
    # would have PASSED, while 1e-320 failed by 45.05. A tie-break assertion at
    # the tie point alone would have been decorative.
    assert checked == 4, f"only {checked} tie-break points asserted"


def _dense_hessian(tape, n, pt):
    """Full dense objective Hessian of a compiled ``NlExpr``.

    ``NlExpr`` exposes only ``eval``/``gradient``; second order needs a built
    problem. Mirrors the strictly-lower triangle rather than
    ``h + h.T - diag(diag(h))``, because these points are chosen to sit where an
    infinity is the correct answer and the add-then-subtract form would turn one
    into a nan (the same defect fixed in ``_tape_nlp_evaluator``).
    """
    import pounce

    prob = pounce.build_nl_problem(n, tape, constraints=None)
    rows, cols = prob.hessian_structure()
    vals = np.asarray(prob.hessian(list(pt)), dtype=float)
    lower = np.zeros((n, n), dtype=float)
    np.add.at(lower, (np.asarray(rows), np.asarray(cols)), vals)
    return lower + np.tril(lower, -1).T


@pytest.mark.unit
def test_xlogx_family_reaches_derivatives_the_chain_rule_cannot():
    """entropy/centropy lower onto pounce's FUSED opcodes, not onto `x*log(x)`.

    Every point here is one where the ANSWER is an ordinary double but some
    intermediate of the decomposed form is not -- a structural limit of the chain
    rule, not a sloppy rule, so no amount of care in the product/log/quotient
    rules reaches them. All three failed before pounce #489 and the folded
    `log(floor)` constant; they are the residuals
    `issue75_derivative_audit.py` used to carry on its allowlist.

    Each assertion states the intermediate that overflows, so a future rewrite
    that reintroduces one fails here with the reason attached.
    """
    checked = 0

    # 1. (x log x)'' = 1/x. Finite for every positive x down to 1e-308, but any
    #    decomposition goes through log''(x) = -1/x**2 = -1e598 at this point.
    m = Model()
    x = m.continuous("x", lb=-1e309, ub=1e309)
    h = _dense_hessian(compile_to_nl_expr(FunctionCall("entropy", x), m), 1, [1e-299])
    assert h[0][0] == pytest.approx(1e299, rel=1e-12), (
        f"entropy''(1e-299) = {h[0][0]}, want 1e299 -- decomposed via log'' = -1/x**2"
    )
    checked += 1

    # 2. d/dy [x log(x/y)] = -x/y = -1 at x = y = 1e300. The quotient rule's
    #    y**2 is 1e600, so the unfused form returned nan for this and for the
    #    whole second-order block.
    m = Model()
    x = m.continuous("x", lb=-1e309, ub=1e309)
    y = m.continuous("y", lb=-1e309, ub=1e309)
    ce = compile_to_nl_expr(FunctionCall("centropy", x, y), m)
    g = np.asarray(ce.gradient([1e300, 1e300]), dtype=float)
    np.testing.assert_allclose(g, [1.0, -1.0], rtol=1e-12, atol=0)
    h = _dense_hessian(ce, 2, [1e300, 1e300])
    np.testing.assert_allclose(h, [[1e-300, -1e-300], [-1e-300, 1e-300]], rtol=1e-12, atol=0)
    checked += 2

    # 3. Below the floor the clamped branch is x*(log(floor) - log(y)). Written
    #    as log(floor/y) instead, log'' forms -1/q**2 with q = 1e-300, so the
    #    y-block came back [[0, -inf], [-inf, nan]] where every entry of the
    #    truth is representable (1e-320 is subnormal but exact here).
    h = _dense_hessian(ce, 2, [1e-320, 1.0])
    assert h[0][0] == 0.0, f"centropy d2/dx2 below the floor is {h[0][0]}, want 0 (clamped)"
    assert h[0][1] == -1.0 and h[1][0] == -1.0, f"centropy d2/dxdy = {h[0][1]}, want -1/y = -1"
    assert h[1][1] == 1e-320, f"centropy d2/dy2 = {h[1][1]}, want x/y**2 = 1e-320"
    checked += 3

    assert checked == 6, f"only {checked} derivative comparisons executed"


# ─────────────────────────────────────────────────────────────
# Static array indexing (`x[i]`, `y[i, j]`)
# ─────────────────────────────────────────────────────────────
#
# The gap these close: before this, EVERY `IndexExpression` was refused, so any
# model written against the Python API with `shape=(...)` variables — the
# ordinary way to write one — fell back to the JAX evaluator. Measured over
# three model families at the time, 40 of 43 reachable `IndexExpression` nodes
# were the static-scalar form lowered here. The other 3 are the vectorized
# slices `dae/collocation.py` emits; those still refuse (see the refusal test
# below), and taking DAE off JAX needs the array lowering, not this.

# (name, n_vars_before, shape, expr-builder taking the shaped Variable).
# `n_vars_before` puts a scalar variable ahead of the shaped one in some cases so
# a lost `_flat_var_offset` would be caught, not masked by a zero offset.
INDEX_CASES = {
    "1d": ((), (6,), lambda v: sum(dm.exp(v[i]) * v[(i + 1) % 6] for i in range(6))),
    "2d": (
        (),
        (3, 3),
        lambda v: sum(dm.log(v[i, j]) * v[j, i] for i in range(3) for j in range(3)),
    ),
    "1d_offset": (("s",), (5,), lambda v: dm.sqrt(v[0]) + v[3] ** 2 + dm.sin(v[4])),
    "2d_offset": (("s", "u"), (2, 4), lambda v: v[0, 3] * v[1, 0] + dm.exp(v[1, 3])),
    "negative": ((), (4,), lambda v: v[-1] * v[-4] + dm.log(v[-2])),
    "repeated_leaf": ((), (3,), lambda v: (lambda t: t * t + dm.exp(t))(v[1] * v[2])),
}


def _indexed_model(leading, shape):
    m = Model()
    for nm in leading:
        m.continuous(nm, lb=0.3, ub=2.0)
    return m, m.continuous("v", shape=shape, lb=0.3, ub=2.0)


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(INDEX_CASES))
def test_static_index_matches_jax_value_and_gradient(name):
    """`x[i]` must lower to the SAME flat slot the JAX path reads.

    A wrong slot is the whole risk here: it is silent, it still produces a finite
    value and a plausible gradient, and only a point-for-point comparison against
    `_jax/dag_compiler` catches it. The gradient half is what pins the slot — the
    value alone is insensitive to a permutation of equal-bounded entries.
    """
    leading, shape, build = INDEX_CASES[name]
    m, v = _indexed_model(leading, shape)
    expr = build(v)
    tape = compile_to_nl_expr(expr, m)
    jf, jg = _jax_pair(expr, m)

    n = sum(var.size for var in m._variables)
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    compared = 0
    for _ in range(25):
        pt = rng.uniform(0.35, 1.9, size=n)
        jv = float(np.asarray(jf(pt)))
        jgv = np.asarray(jg(pt), dtype=float)
        if not (np.isfinite(jv) and np.all(np.isfinite(jgv))):
            continue
        tv = tape.eval(list(pt))
        tg = np.asarray(tape.gradient(list(pt)), dtype=float)
        compared += 1
        assert abs(tv - jv) / max(1.0, abs(jv)) <= 1e-12, f"value drift at {pt}: {tv} vs {jv}"
        scale = max(1.0, float(np.max(np.abs(jgv))))
        assert np.max(np.abs(tg - jgv)) / scale <= 1e-12, (
            f"gradient drift at {pt} -- a mis-resolved flat slot: {tg} vs {jgv}"
        )

    assert compared >= 5, f"only {compared} points compared for {name}"


@pytest.mark.unit
def test_static_index_resolves_the_c_order_slot():
    """Pin the slot arithmetic directly, independent of JAX.

    `grad(v[i, j])` is the unit vector at that entry, so this reads the resolved
    index straight off the gradient. C order and the leading-variable offset are
    both load-bearing: row-major vs column-major differ for any non-square
    access, and a dropped offset silently reads another variable's storage.
    """
    m = Model()
    m.continuous("s", lb=0.0, ub=1.0)  # one scalar ahead of `v` => offset 1
    v = m.continuous("v", shape=(2, 3), lb=0.0, ub=1.0)
    n = 1 + v.size

    checked = 0
    for i in range(2):
        for j in range(3):
            g = np.asarray(compile_to_nl_expr(v[i, j], m).gradient([0.5] * n), dtype=float)
            want = np.zeros(n)
            want[1 + i * 3 + j] = 1.0  # offset 1, C order over shape (2, 3)
            np.testing.assert_array_equal(g, want, err_msg=f"v[{i}, {j}] resolved to {g}")
            checked += 1

    # Negative indices normalize against the dimension, not the flat size.
    g = np.asarray(compile_to_nl_expr(v[-1, -1], m).gradient([0.5] * n), dtype=float)
    assert g[6] == 1.0 and g.sum() == 1.0, f"v[-1, -1] resolved to {g}"
    checked += 1

    assert checked == 7, f"only {checked} slot resolutions executed"


# Every form that does NOT name one scalar slot. Each must refuse, which degrades
# to the JAX evaluator — i.e. to the previous behaviour. Constructed directly
# where `Expression.__getitem__`'s own guard would reject the form first.
NON_STATIC_INDEX = {
    "slice": lambda v, w: v[1:],
    "partial_2d": lambda v, w: IndexExpression(w, 0),
    "slice_2d": lambda v, w: IndexExpression(w, (slice(None), 1)),
    "non_variable_base": lambda v, w: (v + v)[0],
    "boolean": lambda v, w: v[True],
    "out_of_range": lambda v, w: IndexExpression(v, 9),
    "negative_out_of_range": lambda v, w: IndexExpression(v, -9),
    "ellipsis": lambda v, w: IndexExpression(v, Ellipsis),
    "float_index": lambda v, w: IndexExpression(v, 1.0),
    "too_many_indices": lambda v, w: IndexExpression(v, (0, 0)),
}


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(NON_STATIC_INDEX))
def test_non_scalar_index_forms_are_still_refused(name):
    """Refusing is the sound direction: the caller degrades to JAX unchanged.

    `out_of_range` is here on purpose rather than as an error: numpy raises where
    `jnp` silently CLAMPS, so resolving it would let the tape disagree with the
    path it replaces. Refusing hands the case back to JAX with its own semantics.
    """
    m = Model()
    v = m.continuous("v", shape=(4,), lb=0.3, ub=2.0)
    w = m.continuous("w", shape=(2, 3), lb=0.3, ub=2.0)
    expr = NON_STATIC_INDEX[name](v, w)
    with pytest.raises(UnsupportedForTape, match="static scalar slot"):
        compile_to_nl_expr(expr, m)


@pytest.mark.unit
def test_shaped_variable_model_builds_a_tape_evaluator():
    """The gap itself: a shaped-variable model no longer falls back to JAX.

    This is the end-to-end statement the unit tests above cannot make —
    `try_build` returning None is exactly the JAX fallback, and it did so for
    every `shape=(...)` model before static indexing lowered.
    """
    from discopt._tape_nlp_evaluator import try_build

    m = Model()
    x = m.continuous("x", shape=(4,), lb=0.1, ub=2.0)
    m.minimize(sum(dm.exp(x[i]) * x[(i + 1) % 4] for i in range(4)))
    m.subject_to(sum(x[i] ** 2 for i in range(4)) <= 4.0)

    assert try_build(m) is not None, "shaped-variable model still degrades to the JAX evaluator"


@pytest.mark.unit
def test_vectorized_collocation_bodies_still_refuse():
    """Guard the retraction: this change does NOT take DAE collocation off JAX.

    `dae/collocation.py` emits ONE vector-valued constraint per state, built as a
    `MatMulExpression` against the collocation matrix with slice indexing — not
    scalar `x[i, k]`. An earlier reading of this gap claimed the opposite. If a
    future array lowering makes these representable, this test should be updated
    deliberately, not silently.
    """
    pytest.importorskip("discopt.dae")
    from discopt.dae import ContinuousSet, DAEBuilder

    m = Model("second_order_decay")
    dae = DAEBuilder(m, ContinuousSet("t", bounds=(0, 2), nfe=4, ncp=3))
    dae.add_state("A", initial=1.0, bounds=(0.0, 2.0))
    dae.set_ode(lambda t, s, a, c: {"A": -0.7 * s["A"] ** 2})
    dae.discretize()

    assert m._constraints, "collocation emitted no Constraint objects to check"
    refused = 0
    for con in m._constraints:
        with pytest.raises(UnsupportedForTape):
            compile_to_nl_expr(con.body, m)
        refused += 1
    assert refused == len(m._constraints) and refused >= 2, (
        f"only {refused} collocation bodies checked"
    )
