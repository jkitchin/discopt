"""Compile a discopt expression DAG into a POUNCE ``NlExpr`` tape expression.

This is the shared prerequisite for taking JAX off the solve path (issue #75).
Both remaining JAX jobs need the same thing — the value and gradient of a scalar
expression at a point — and POUNCE's Rust AD tape provides exactly that:

* **separation tangents** (``_jax/uniform_relax.py``) need ``g(x0)`` and
  ``grad g(x0)`` for the Kelley cutting-plane loop;
* **NLP subsolve derivatives** (``_jax/nlp_evaluator.py``) need ``f``, ``grad f``,
  ``g``, ``J``, and the Lagrangian Hessian.

Previously those were headed for two *different* replacement backends. They no
longer need to be: ``pounce.NlExpr`` covers discopt's *scalar* DAG operators — 20
natively and 9 by the exact rewrites below — where the in-tree interval-AD engine
covers 6, and it agrees with analytic truth exactly on expressions the interval
engine cannot evaluate at all (``tanh``, ``erf``).

The **array reductions** are the exception and are refused, not approximated:
``prod`` (``jnp.prod`` of one array argument), ``norm1``/``norm2``/``norminf``,
``SumExpression``, ``IndexExpression`` and ``MatMulExpression``. A tape node is a
scalar; there is no array to reduce. An earlier revision lowered ``prod`` as a
variadic ``*`` chain, which silently computed a different function.

*Coverage claims here must be checked against operators, not instances.* ``.nl``
has no opcode for ``sigmoid``/``softplus``/``entropy``/``centropy``/``signpower``,
so a MINLPLib corpus sweep — measured across 316 instances — exercises exactly six
of these (``log``, ``sqrt``, ``exp``, ``abs``, ``sin``, ``cos``) and can never
reach the rest. Three defects lived in the unreached rewrites: ``entropy`` had an
inverted sign, ``_sign`` passed ``compare``'s arguments in the wrong order, and
``prod`` was the wrong function entirely. See
``python/tests/test_75_nl_expr_compiler.py`` for the per-operator differential
that catches this class.

Why not ``.nl`` as the intermediate? Because it is lossy in the wrong direction:
discopt's ``.nl`` writer refuses ``atan2``, ``min``, ``max``, ``erf`` and ``sign``
(``export/nl.py``), several of which the tape *does* differentiate. Building the
tape expression directly keeps them.

Deliberately NOT under ``_jax/``: nothing here touches JAX.
"""

from __future__ import annotations

import math
from typing import Any

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    CustomCall,
    Expression,
    FunctionCall,
    IndexExpression,
    MatMulExpression,
    Model,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)


class UnsupportedForTape(Exception):
    """``expr`` contains a node the tape cannot represent.

    Raised rather than returned so a caller cannot mistake "no tape" for "a tape
    that evaluates to nothing". Callers are expected to catch this and fall back
    to their existing path; the only expected trigger is :class:`CustomCall`,
    whose body is an opaque JAX callable by contract (``modeling/core.py``) and
    therefore has no tape equivalent.
    """


def _compute_var_offset(var: Variable, model: Model) -> int:
    """Start of ``var`` in the flat ``x`` vector.

    Delegates to the model's memoized prefix-sum table, so per-leaf resolution
    during the DAG walk is O(1). A linear scan here would be O(n) per leaf and
    O(n^2) over the build — the regression issue #654 fixed, and the reason
    ``dag_compiler`` routes through the same table.
    """
    return model._flat_var_offset(var)


def compile_to_nl_expr(expr: Expression, model: Model) -> Any:
    """Compile ``expr`` to a ``pounce.NlExpr`` over the model's flat variable layout.

    The returned object supports ``.eval(x)`` and ``.gradient(x)`` for a flat ``x``
    ordered exactly like the JAX path's ``x_flat``, so the two are directly
    comparable point-for-point.

    Raises:
        UnsupportedForTape: if the DAG contains a node with no tape equivalent.
    """
    import pounce

    E = pounce.NlExpr
    # id(node) -> NlExpr. The DAG is a DAG, not a tree: a node reachable by k
    # references must be built once, or a linear DAG lowers in time exponential in
    # its sharing depth (the same trap `dag_compiler` documents as issue #383).
    memo: dict[int, Any] = {}
    return _lower(expr, model, E, memo)


def _lower(expr: Expression, model: Model, E: Any, memo: dict[int, Any]) -> Any:
    key = id(expr)
    hit = memo.get(key)
    if hit is not None:
        return hit
    built = _lower_uncached(expr, model, E, memo)
    memo[key] = built
    return built


def _lower_uncached(expr: Expression, model: Model, E: Any, memo: dict[int, Any]) -> Any:
    def rec(child: Expression) -> Any:
        return _lower(child, model, E, memo)

    if isinstance(expr, Constant):
        return E.const_(float(expr.value))

    if isinstance(expr, Variable):
        if expr.size != 1:
            raise UnsupportedForTape(
                f"array variable {expr.name!r} (size {expr.size}); the tape path is scalar"
            )
        return E.var(_compute_var_offset(expr, model))

    if isinstance(expr, Parameter):
        # Parameters are constants at compile time, matching the legacy
        # `compile_expression` behaviour (the params-as-runtime-args variant has
        # no tape analogue -- a tape is built for fixed structure).
        return E.const_(float(expr.value))

    if isinstance(expr, BinaryOp):
        left, right, op = rec(expr.left), rec(expr.right), expr.op
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        if op == "**":
            return left**right
        raise UnsupportedForTape(f"binary operator {op!r}")

    if isinstance(expr, UnaryOp):
        operand, op = rec(expr.operand), expr.op
        if op == "neg":
            return -operand
        if op == "abs":
            return _abs(E, operand)
        raise UnsupportedForTape(f"unary operator {op!r}")

    if isinstance(expr, FunctionCall):
        return _lower_function(expr, E, [rec(a) for a in expr.args])

    if isinstance(expr, SumExpression):
        # An ARRAY reduction (``jnp.sum(operand, axis=...)``), not a list of scalar
        # terms -- ``.operand``/``.axis``, per ``dag_compiler``. The tape is scalar,
        # so refuse rather than guess a flattening.
        raise UnsupportedForTape("SumExpression (array reduction) has no scalar tape lowering")

    if isinstance(expr, SumOverExpression):
        terms = [rec(t) for t in expr.terms]
        if not terms:
            return E.const_(0.0)
        acc = terms[0]
        for t in terms[1:]:
            acc = acc + t
        return acc

    if isinstance(expr, IndexExpression):
        raise UnsupportedForTape("IndexExpression (array indexing) is not yet lowered")

    if isinstance(expr, MatMulExpression):
        raise UnsupportedForTape("MatMulExpression is not yet lowered")

    if isinstance(expr, CustomCall):
        # By contract (`modeling/core.py`) a dm.custom body is an opaque callable
        # that must be JAX-differentiable. There is no tape equivalent, and there
        # should not be a silent one -- relaxing an opaque callable needs AD
        # *through* it, which is exactly what the tape cannot do.
        raise UnsupportedForTape("CustomCall (dm.custom) has no tape equivalent")

    raise UnsupportedForTape(f"expression node {type(expr).__name__}")


def _abs(E: Any, a: Any) -> Any:
    """``abs(a) = max(a, -a)``. The tape has ``max``; ASL/Ipopt treat the kink as a
    non-smooth event and route derivatives through the active branch."""
    return E.max(a, -a)


def _sign(E: Any, a: Any) -> Any:
    """``sign(a)`` via nested selects; zero maps to 0.0 as in ``jnp.sign``.

    ``compare`` takes the operator FIRST (``compare(op, lhs, rhs)``). Passing it
    last raises ``TypeError``, not ``UnsupportedForTape``, so it escapes
    ``try_compile``'s fallback and crashes the caller.
    """
    zero = E.const_(0.0)
    pos = E.compare(">", a, zero)
    neg = E.compare("<", a, zero)
    return E.select(pos, E.const_(1.0), E.select(neg, E.const_(-1.0), zero))


def _lower_function(expr: FunctionCall, E: Any, args: list) -> Any:
    """Lower a ``FunctionCall``. 20 operators are native to the tape; the rest are
    exact rewrites into operators that are."""
    name = expr.func_name

    def arg0() -> Any:
        """First argument, after ``_require`` has proven it exists."""
        return args[0]

    native_unary = {
        "exp",
        "log",
        "log10",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "erf",
    }
    if name in native_unary:
        _require(args, 1, name)
        return getattr(E, name)(arg0())

    if name in ("min", "max", "atan2"):
        _require(args, 2, name)
        return getattr(E, name)(args[0], args[1])

    # --- exact rewrites (no tape opcode, but expressible) ---------------------
    if name == "abs":
        _require(args, 1, name)
        return _abs(E, arg0())
    if name == "sign":
        _require(args, 1, name)
        return _sign(E, arg0())
    if name == "log1p":
        _require(args, 1, name)
        return E.log(E.const_(1.0) + arg0())
    if name == "log2":
        _require(args, 1, name)
        return E.log(arg0()) * E.const_(1.0 / math.log(2.0))
    if name == "sigmoid":
        _require(args, 1, name)
        return E.const_(1.0) / (E.const_(1.0) + E.exp(-arg0()))
    if name == "softplus":
        _require(args, 1, name)
        return E.log(E.const_(1.0) + E.exp(arg0()))
    if name == "entropy":
        # x*log(x) -- discopt's DAG semantics, NOT the information-theory
        # convention -x*log(x). The authority is `_jax/dag_compiler.py`, which
        # this must reproduce bit-for-bit: `lambda x: x * jnp.log(jnp.maximum(x,
        # 1e-300))`. Getting the sign from the name instead of the reference
        # cost a silent factor of -1 (reldiff 2.0) that no corpus instance could
        # have caught -- `.nl` has no entropy opcode, so 316 MINLPLib instances
        # exercise this line zero times.
        _require(args, 1, name)
        return arg0() * E.log(arg0())
    if name == "centropy":
        # x*log(x/y), matching dag_compiler's GAMS centropy.
        _require(args, 2, name)
        return args[0] * E.log(args[0] / args[1])
    if name == "signpower":
        # sign(x) * |x|**p -- the standard smooth-away-from-zero signed power.
        _require(args, 2, name)
        return _sign(E, args[0]) * (_abs(E, args[0]) ** args[1])
    if name == "prod":
        # NOT a variadic multiply. `dag_compiler` compiles prod as
        # `jnp.prod(arg)` -- ONE argument, which is an ARRAY, reduced to a
        # scalar. Lowering it as a `*` chain over `args` silently computed a
        # different function (measured: reldiff 0.90 on value, 1.70 on
        # gradient). The scalar tape has no array to reduce, so refuse, exactly
        # as SumExpression does. `norm1`/`norm2`/`norminf` are array reductions
        # too and fall through to the same refusal below.
        raise UnsupportedForTape(
            "prod is an array reduction (jnp.prod of one array argument); "
            "the scalar tape path has no array to reduce"
        )

    raise UnsupportedForTape(f"function {name!r}")


def _require(args: list, n: int, name: str) -> None:
    if len(args) != n:
        raise UnsupportedForTape(f"{name} expects {n} argument(s), got {len(args)}")


def try_compile(expr: Expression, model: Model) -> Any | None:
    """``compile_to_nl_expr`` returning ``None`` instead of raising.

    For call sites that already have a working fallback and want the tape only
    when it is available. The distinction matters: this returns ``None`` only for
    *representability*, never for a numerical failure, so a caller cannot confuse
    "unsupported operator" with "bad point".
    """
    try:
        return compile_to_nl_expr(expr, model)
    except UnsupportedForTape:
        return None
