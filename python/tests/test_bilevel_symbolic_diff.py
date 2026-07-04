"""Differential test for the bilevel symbolic differentiator.

``discopt.bilevel.symbolic_diff.diff`` returns a symbolic ``∂expr/∂var`` as an
ordinary Expression. Its acceptance gate (bilevel-module-plan.md §3, Phase 0) is
that the *symbolic* derivative, compiled through the same DAG compiler and
evaluated, matches JAX's *numeric* ``jax.grad`` to machine tolerance across a
fuzz of random expression DAGs covering every supported node type — plus
targeted rule/domain checks and the loud-refusal contracts.

These tests need JAX (the numeric oracle) and the pure-Python modeling layer;
they do not need the Rust extension.
"""

from __future__ import annotations

import math

import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from discopt._jax.dag_compiler import compile_expression_params  # noqa: E402
from discopt.bilevel.symbolic_diff import diff, grad  # noqa: E402
from discopt.modeling.core import (  # noqa: E402
    BinaryOp,
    Constant,
    CustomCall,
    FunctionCall,
    Model,
)

pytestmark = pytest.mark.smoke

_NO_PARAMS = ()


def _numeric_grad(expr, model, x_val):
    """Full gradient vector via jax.grad (the oracle)."""
    f = compile_expression_params(expr, model)
    return jax.grad(lambda xx: f(xx, _NO_PARAMS))(x_val)


def _assert_grad_matches(expr, model, scalar_vars, x_val, *, rtol=1e-7, atol=1e-9):
    """Symbolic ∂expr/∂v == jax.grad[v] for every scalar var, at x_val.

    Returns False (skip) if the primal or numeric gradient is non-finite at this
    point (overflow from exp/sinh, etc.); asserts equality otherwise.
    """
    fval = float(compile_expression_params(expr, model)(x_val, _NO_PARAMS))
    if not math.isfinite(fval):
        return False
    jg = _numeric_grad(expr, model, x_val)
    if not bool(jnp.all(jnp.isfinite(jg))):
        return False
    for k, v in enumerate(scalar_vars):
        d = diff(expr, v)
        sym = float(compile_expression_params(d, model)(x_val, _NO_PARAMS))
        num = float(jg[k])
        assert math.isfinite(sym), f"non-finite symbolic deriv for {v.name}: {expr!r}"
        assert abs(sym - num) <= atol + rtol * abs(num), (
            f"∂/∂{v.name} mismatch: symbolic={sym!r} jax={num!r}\n  expr={expr!r}\n"
            f"  deriv={d!r}\n  x={list(map(float, x_val))}"
        )
    return True


# ---------------------------------------------------------------------------
# 1. Random-DAG fuzz vs jax.grad — the headline acceptance test.
# ---------------------------------------------------------------------------

# Smooth-on-all-of-R univariate functions (no kinks, no domain restrictions);
# overflow from exp/sinh/cosh is filtered per-point.
_FUZZ_FUNCS = ["exp", "sin", "cos", "tanh", "atan", "sinh", "cosh"]


def _random_expr(rng, variables, depth):
    """A random expression over the given scalar Variables (safe by construction)."""
    if depth <= 0 or rng.random() < 0.30:
        if rng.random() < 0.75:
            return variables[int(rng.integers(len(variables)))]
        return Constant(float(rng.uniform(-2.0, 2.0)))

    r = rng.random()
    if r < 0.55:
        op = ["+", "-", "*", "/", "**"][int(rng.integers(5))]
        a = _random_expr(rng, variables, depth - 1)
        b = _random_expr(rng, variables, depth - 1)
        if op == "**":
            # integer exponent in {2, 3}: smooth on R, exercises the power rule
            return BinaryOp("**", a, Constant(float(int(rng.integers(2, 4)))))
        if op == "/":
            # denominator bounded in [1.5, 3.5] -> never near zero
            safe = BinaryOp("+", Constant(2.5), FunctionCall("tanh", b))
            return BinaryOp("/", a, safe)
        return BinaryOp(op, a, b)

    fn = _FUZZ_FUNCS[int(rng.integers(len(_FUZZ_FUNCS)))]
    return FunctionCall(fn, _random_expr(rng, variables, depth - 1))


def test_symbolic_diff_matches_jax_fuzz():
    rng = __import__("numpy").random.default_rng(20260703)
    m = Model("fuzz")
    variables = [m.continuous(f"v{i}", lb=-5, ub=5) for i in range(3)]

    checks = 0
    attempts = 0
    target = 300
    while checks < target and attempts < 6000:
        attempts += 1
        expr = _random_expr(rng, variables, depth=int(rng.integers(2, 5)))
        x_val = jnp.asarray(rng.uniform(-1.2, 1.2, size=len(variables)))
        if _assert_grad_matches(expr, m, variables, x_val):
            checks += 1
    assert checks >= target, f"only {checks} finite fuzz checks in {attempts} attempts"


# ---------------------------------------------------------------------------
# 2. Every univariate derivative rule at a safe in-domain point.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn,x0",
    [
        ("exp", 0.7),
        ("log", 2.0),
        ("log2", 2.0),
        ("log10", 2.0),
        ("log1p", 0.5),
        ("sqrt", 4.0),
        ("sin", 0.6),
        ("cos", 0.6),
        ("tan", 0.5),
        ("sinh", 0.4),
        ("cosh", 0.4),
        ("tanh", 0.4),
        ("atan", 0.5),
        ("asin", 0.3),
        ("acos", 0.3),
        ("asinh", 0.5),
        ("acosh", 1.7),
        ("atanh", 0.3),
        ("softplus", 0.5),
    ],
)
def test_function_derivative_rule(fn, x0):
    m = Model("f")
    x = m.continuous("x", lb=-10, ub=10)
    # chain-rule stress: f(2*x + 0.3), so the derivative must carry the inner 2.
    expr = FunctionCall(fn, BinaryOp("+", BinaryOp("*", Constant(2.0), x), Constant(0.3)))
    x_val = jnp.asarray([x0])
    jg = float(_numeric_grad(expr, m, x_val)[0])
    sym = float(compile_expression_params(diff(expr, x), m)(x_val, _NO_PARAMS))
    assert abs(sym - jg) <= 1e-8 + 1e-7 * abs(jg), f"{fn}: {sym} vs {jg}"


# ---------------------------------------------------------------------------
# 3. Specific arithmetic rules.
# ---------------------------------------------------------------------------


def test_product_quotient_power_general():
    m = Model("rules")
    x = m.continuous("x", lb=0.5, ub=5)
    y = m.continuous("y", lb=0.5, ub=5)
    x_val = jnp.asarray([1.7, 2.3])
    exprs = [
        x * y,  # product: dx=y, dy=x
        (x * y) / (x + y),  # quotient
        x ** Constant(3.0),  # constant-exponent power
        FunctionCall("exp", x) ** y,  # general u**v (u=exp(x)>0)
        x * FunctionCall("sin", x * y),  # nested product+chain
    ]
    for e in exprs:
        assert _assert_grad_matches(e, m, [x, y], x_val)


def test_abs_subgradient_away_from_kink():
    m = Model("abs")
    x = m.continuous("x", lb=-5, ub=5)
    expr = abs(x * Constant(2.0))  # UnaryOp abs; smooth away from 0
    for xv in (1.3, -0.8):
        jg = float(_numeric_grad(expr, m, jnp.asarray([xv]))[0])
        sym = float(compile_expression_params(diff(expr, x), m)(jnp.asarray([xv]), _NO_PARAMS))
        assert abs(sym - jg) <= 1e-9 + 1e-7 * abs(jg)


# ---------------------------------------------------------------------------
# 4. DAG sharing, parameters, multi-var gradient.
# ---------------------------------------------------------------------------


def test_shared_subexpression_differentiates_correctly():
    m = Model("share")
    x = m.continuous("x", lb=-3, ub=3)
    u = FunctionCall("sin", x * x)  # shared node
    expr = u * u + u  # u referenced three times
    xv = jnp.asarray([0.9])
    jg = float(_numeric_grad(expr, m, xv)[0])
    sym = float(compile_expression_params(diff(expr, x), m)(xv, _NO_PARAMS))
    assert abs(sym - jg) <= 1e-8 + 1e-7 * abs(jg)


def test_parameter_treated_as_constant():
    m = Model("param")
    x = m.continuous("x", lb=-3, ub=3)
    p = m.parameter("p", value=4.0)
    expr = p * x * x  # d/dx = 2*p*x ; p is data, not differentiated
    params = (jnp.asarray(4.0),)
    d = diff(expr, x)
    val = float(compile_expression_params(d, m)(jnp.asarray([1.5]), params))
    assert abs(val - (2 * 4.0 * 1.5)) <= 1e-9


def test_grad_returns_component_list():
    m = Model("g")
    x = m.continuous("x", lb=-3, ub=3)
    y = m.continuous("y", lb=-3, ub=3)
    expr = x * x + Constant(3.0) * x * y
    g = grad(expr, [x, y])
    assert len(g) == 2
    xv = jnp.asarray([1.1, 0.4])
    dx = float(compile_expression_params(g[0], m)(xv, _NO_PARAMS))  # 2x + 3y
    dy = float(compile_expression_params(g[1], m)(xv, _NO_PARAMS))  # 3x
    assert abs(dx - (2 * 1.1 + 3 * 0.4)) <= 1e-9
    assert abs(dy - (3 * 1.1)) <= 1e-9


# ---------------------------------------------------------------------------
# 5. Loud-refusal contracts.
# ---------------------------------------------------------------------------


def test_scalar_only_guard():
    m = Model("vec")
    xa = m.continuous("xa", shape=(3,), lb=-1, ub=1)
    with pytest.raises(NotImplementedError, match="scalar Variable"):
        diff(xa * xa, xa)


def test_customcall_refused():
    m = Model("cc")
    x = m.continuous("x", lb=-1, ub=1)
    node = CustomCall(lambda a: a, x, name="opaque")
    with pytest.raises(NotImplementedError, match="CustomCall"):
        diff(node, x)


def test_multiarg_function_refused():
    m = Model("mm")
    x = m.continuous("x", lb=-1, ub=1)
    y = m.continuous("y", lb=-1, ub=1)
    with pytest.raises(NotImplementedError):
        diff(FunctionCall("max", x, y), x)


def test_non_variable_wrt_rejected():
    m = Model("t")
    x = m.continuous("x", lb=-1, ub=1)
    with pytest.raises(TypeError):
        diff(x * x, Constant(1.0))
