"""Symbolic differentiation over the discopt expression DAG.

Phase 0 of the bilevel module (``docs/dev/bilevel-module-plan.md`` §3). A bilevel
program's convex lower level ``min_y f(x, y) s.t. g(x, y) <= 0`` is replaced by
its KKT conditions; the **stationarity** part ``∇_y L == 0`` (where
``L = f + Σ μ_i g_i``) must be emitted as *ordinary model constraints* so the
resulting single-level MPEC stays inside discopt's certified global path.

discopt differentiates only **numerically** (JAX ``jax.grad`` / ``jax.jacfwd``
over ``compile_expression_params``). A numeric gradient can be embedded as a
:class:`~discopt.modeling.core.CustomCall`, but that node is *AD-only* — it
forfeits the global certificate and rejects integer variables. So a certifiable
KKT reformulation needs a **symbolic** derivative that returns an ordinary
:class:`~discopt.modeling.core.Expression`, built from the same DAG node
constructors and therefore compilable, relaxable, and certifiable like any
user-written expression. That is exactly what this module provides:

    diff(expr, wrt)  ->  Expression   # ∂expr/∂wrt, wrt a scalar Variable
    grad(expr, vars) ->  list[Expression]

Scope (Phase 0)
---------------
Partial derivatives with respect to a **scalar** :class:`Variable`. Supported
nodes: the arithmetic operators (``+ - * / **``), unary ``neg``/``abs``, the
smooth univariate :class:`FunctionCall` families the DAG compiler can evaluate,
and the linear aggregation nodes (``sum``, indexed sums, matmul via the product
rule). The result is simplified on the fly (``0·e = 0``, ``1·e = e``,
``e + 0 = e``, ``a**1 = a`` …) so the emitted stationarity system stays small.

Nonsmoothness. ``abs`` and ``x**c`` have kinks; at a kink this returns a
*subgradient* (``sign`` for ``abs``), which is documented and excluded from the
differential-test tolerance. Opaque :class:`CustomCall` nodes have no symbolic
form and are refused loudly. Multi-argument functions (``min``, ``max``,
``atan2``, ``norm``, ``entropy``) are not differentiated in Phase 0 and raise.

Correctness is pinned by a differential test against ``jax.grad``
(``python/tests/test_bilevel_symbolic_diff.py``).
"""

from __future__ import annotations

import math

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    CustomCall,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)

__all__ = ["diff", "grad"]


# ---------------------------------------------------------------------------
# Simplifying constructors — keep the derivative DAG small and domain-clean.
# ---------------------------------------------------------------------------


def _const(v: float) -> Constant:
    return Constant(float(v))


def _is_const(e: Expression) -> bool:
    return isinstance(e, Constant) and e.value.ndim == 0


def _is_zero(e: Expression) -> bool:
    return _is_const(e) and float(e.value) == 0.0


def _is_one(e: Expression) -> bool:
    return _is_const(e) and float(e.value) == 1.0


def _add(a: Expression, b: Expression) -> Expression:
    if _is_zero(a):
        return b
    if _is_zero(b):
        return a
    if _is_const(a) and _is_const(b):
        return _const(float(a.value) + float(b.value))
    return BinaryOp("+", a, b)


def _sub(a: Expression, b: Expression) -> Expression:
    if _is_zero(b):
        return a
    if _is_zero(a):
        return _neg(b)
    if _is_const(a) and _is_const(b):
        return _const(float(a.value) - float(b.value))
    return BinaryOp("-", a, b)


def _mul(a: Expression, b: Expression) -> Expression:
    if _is_zero(a) or _is_zero(b):
        return _const(0.0)
    if _is_one(a):
        return b
    if _is_one(b):
        return a
    if _is_const(a) and _is_const(b):
        return _const(float(a.value) * float(b.value))
    return BinaryOp("*", a, b)


def _div(a: Expression, b: Expression) -> Expression:
    if _is_zero(a):
        return _const(0.0)
    if _is_one(b):
        return a
    return BinaryOp("/", a, b)


def _neg(a: Expression) -> Expression:
    if _is_zero(a):
        return _const(0.0)
    if _is_const(a):
        return _const(-float(a.value))
    return UnaryOp("neg", a)


def _pow(a: Expression, b: Expression) -> Expression:
    if _is_const(b):
        c = float(b.value)
        if c == 0.0:
            return _const(1.0)
        if c == 1.0:
            return a
    return BinaryOp("**", a, b)


def _sq(u: Expression) -> Expression:
    return _pow(u, _const(2.0))


# ---------------------------------------------------------------------------
# Outer-derivative table for smooth univariate FunctionCall nodes.
# Each entry maps ``u`` (the argument Expression) to f'(u) as an Expression.
# Every FunctionCall name emitted here (exp, log, sqrt, sin, cos, tan, sinh,
# cosh, tanh, sign) is in the DAG compiler's supported set, so the derivative
# compiles on the same path as the primal.
# ---------------------------------------------------------------------------

_FUNC_DERIV = {
    "exp": lambda u: FunctionCall("exp", u),
    "log": lambda u: _div(_const(1.0), u),
    "log2": lambda u: _div(_const(1.0), _mul(u, _const(math.log(2.0)))),
    "log10": lambda u: _div(_const(1.0), _mul(u, _const(math.log(10.0)))),
    "log1p": lambda u: _div(_const(1.0), _add(_const(1.0), u)),
    "sqrt": lambda u: _div(_const(1.0), _mul(_const(2.0), FunctionCall("sqrt", u))),
    "sin": lambda u: FunctionCall("cos", u),
    "cos": lambda u: _neg(FunctionCall("sin", u)),
    "tan": lambda u: _add(_const(1.0), _sq(FunctionCall("tan", u))),
    "sinh": lambda u: FunctionCall("cosh", u),
    "cosh": lambda u: FunctionCall("sinh", u),
    "tanh": lambda u: _sub(_const(1.0), _sq(FunctionCall("tanh", u))),
    "atan": lambda u: _div(_const(1.0), _add(_const(1.0), _sq(u))),
    "asin": lambda u: _div(_const(1.0), FunctionCall("sqrt", _sub(_const(1.0), _sq(u)))),
    "acos": lambda u: _neg(_div(_const(1.0), FunctionCall("sqrt", _sub(_const(1.0), _sq(u))))),
    "asinh": lambda u: _div(_const(1.0), FunctionCall("sqrt", _add(_sq(u), _const(1.0)))),
    "acosh": lambda u: _div(_const(1.0), FunctionCall("sqrt", _sub(_sq(u), _const(1.0)))),
    "atanh": lambda u: _div(_const(1.0), _sub(_const(1.0), _sq(u))),
    # softplus'(u) = sigmoid(u) = 1 / (1 + exp(-u))
    "softplus": lambda u: _div(_const(1.0), _add(_const(1.0), FunctionCall("exp", _neg(u)))),
    # Nonsmooth: subgradient. abs'(u) = sign(u); sign'(u) = 0 a.e.
    "abs": lambda u: FunctionCall("sign", u),
    "sign": lambda u: _const(0.0),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def diff(expr: Expression, wrt: Variable, *, _memo: dict | None = None) -> Expression:
    """Symbolic partial derivative ``∂expr/∂wrt`` as a discopt Expression.

    Parameters
    ----------
    expr
        The expression to differentiate.
    wrt
        A **scalar** :class:`~discopt.modeling.core.Variable` to differentiate
        with respect to. Matching is by object identity.

    Returns
    -------
    Expression
        A DAG node for the derivative, built from ordinary expression
        constructors (so it compiles / relaxes / certifies like any expression).

    Raises
    ------
    TypeError
        If ``wrt`` is not a :class:`Variable`.
    NotImplementedError
        If ``wrt`` is not scalar, or ``expr`` contains a node with no Phase 0
        symbolic derivative (opaque ``CustomCall``, multi-arg function, …).
    """
    if not isinstance(wrt, Variable):
        raise TypeError(f"diff() 'wrt' must be a Variable, got {type(wrt).__name__}")
    if wrt.size != 1:
        raise NotImplementedError(
            f"symbolic_diff Phase 0 differentiates w.r.t. a scalar Variable; "
            f"'{wrt.name}' has shape {wrt.shape}. Differentiate each component "
            f"(e.g. wrt[i]) separately, or await the array-variable phase."
        )
    if not isinstance(expr, Expression):
        # Bare python/numpy scalars are constants -> derivative 0.
        return _const(0.0)
    memo = {} if _memo is None else _memo
    key = id(expr)
    cached = memo.get(key)
    if cached is not None:
        return cached
    d = _diff_node(expr, wrt, memo)
    memo[key] = d
    return d


def grad(expr: Expression, wrt_vars) -> list[Expression]:
    """Symbolic gradient: ``[∂expr/∂v for v in wrt_vars]``.

    A fresh memo is used per variable (a node's derivative depends on which
    variable we differentiate against), so DAG sharing is exploited within each
    component but not incorrectly reused across components.
    """
    return [diff(expr, v) for v in wrt_vars]


# ---------------------------------------------------------------------------
# Node dispatch
# ---------------------------------------------------------------------------


def _diff_node(expr: Expression, wrt: Variable, memo: dict) -> Expression:
    if isinstance(expr, Constant) or isinstance(expr, Parameter):
        # Parameters are held fixed (upper-level decisions / data) -> constant.
        return _const(0.0)

    if isinstance(expr, Variable):
        return _const(1.0) if expr is wrt else _const(0.0)

    if isinstance(expr, BinaryOp):
        return _diff_binary(expr, wrt, memo)

    if isinstance(expr, UnaryOp):
        d = diff(expr.operand, wrt, _memo=memo)
        if expr.op == "neg":
            return _neg(d)
        if expr.op == "abs":
            # subgradient at 0
            return _mul(FunctionCall("sign", expr.operand), d)
        raise NotImplementedError(f"symbolic_diff: UnaryOp '{expr.op}' unsupported")

    if isinstance(expr, FunctionCall):
        if len(expr.args) != 1:
            raise NotImplementedError(
                f"symbolic_diff Phase 0 does not differentiate the multi-argument "
                f"function '{expr.func_name}' (min/max/atan2/norm/…)."
            )
        rule = _FUNC_DERIV.get(expr.func_name)
        if rule is None:
            raise NotImplementedError(
                f"symbolic_diff: no derivative rule for function '{expr.func_name}'."
            )
        u = expr.args[0]
        du = diff(u, wrt, _memo=memo)
        if _is_zero(du):
            return _const(0.0)
        return _mul(rule(u), du)

    if isinstance(expr, SumOverExpression):
        return _sum_terms([diff(t, wrt, _memo=memo) for t in expr.terms])

    if isinstance(expr, SumExpression):
        d = diff(expr.operand, wrt, _memo=memo)
        if _is_zero(d):
            return _const(0.0)
        return SumExpression(d, expr.axis)

    if isinstance(expr, IndexExpression):
        d = diff(expr.base, wrt, _memo=memo)
        if _is_zero(d):
            return _const(0.0)
        return IndexExpression(d, expr.index)

    if isinstance(expr, MatMulExpression):
        # Product rule: d(A @ B) = dA @ B + A @ dB.
        da = diff(expr.left, wrt, _memo=memo)
        db = diff(expr.right, wrt, _memo=memo)
        parts = []
        if not _is_zero(da):
            parts.append(MatMulExpression(da, expr.right))
        if not _is_zero(db):
            parts.append(MatMulExpression(expr.left, db))
        if not parts:
            return _const(0.0)
        if len(parts) == 1:
            return parts[0]
        return BinaryOp("+", parts[0], parts[1])

    if isinstance(expr, CustomCall):
        raise NotImplementedError(
            "symbolic_diff: CustomCall is an opaque AD-only node with no symbolic "
            "derivative. It also forfeits the global certificate and rejects "
            "integer variables, so it cannot appear in a certifiable KKT "
            "reformulation. Express the lower level with algebraic operators."
        )

    raise NotImplementedError(
        f"symbolic_diff: no derivative rule for node type {type(expr).__name__}"
    )


def _diff_binary(expr: BinaryOp, wrt: Variable, memo: dict) -> Expression:
    op = expr.op
    a, b = expr.left, expr.right
    da = diff(a, wrt, _memo=memo)
    db = diff(b, wrt, _memo=memo)

    if op == "+":
        return _add(da, db)
    if op == "-":
        return _sub(da, db)
    if op == "*":
        # product rule
        return _add(_mul(da, b), _mul(a, db))
    if op == "/":
        # quotient rule: (da*b - a*db) / b**2
        if _is_zero(db):
            # denominator constant w.r.t. wrt: (da) / b
            return _div(da, b)
        num = _sub(_mul(da, b), _mul(a, db))
        return _div(num, _sq(b))
    if op == "**":
        return _diff_pow(a, b, da, db)

    raise NotImplementedError(f"symbolic_diff: BinaryOp '{op}' unsupported")


def _diff_pow(a: Expression, b: Expression, da: Expression, db: Expression) -> Expression:
    """Derivative of ``a ** b``.

    Constant exponent ``c``: ``c * a**(c-1) * da`` (the common polynomial case).
    General ``a ** b``: ``a**b * (db*ln(a) + b*da/a)`` — valid for ``a > 0``.
    """
    if _is_const(b):
        c = float(b.value)
        if c == 0.0 or _is_zero(da):
            return _const(0.0)
        if c == 1.0:
            return da
        return _mul(_mul(_const(c), _pow(a, _const(c - 1.0))), da)

    # General base**exponent (both may depend on wrt). Requires a > 0.
    term1 = _mul(db, FunctionCall("log", a))
    term2 = _div(_mul(b, da), a) if not _is_zero(da) else _const(0.0)
    return _mul(_pow(a, b), _add(term1, term2))


def _sum_terms(terms: list[Expression]) -> Expression:
    live = [t for t in terms if not _is_zero(t)]
    if not live:
        return _const(0.0)
    if len(live) == 1:
        return live[0]
    return SumOverExpression(live)
