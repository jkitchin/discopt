"""
Relaxation Compiler: Expression tree -> McCormick relaxation function.

Walks the Expression DAG and produces a pure jax.numpy function that computes
compositional McCormick relaxations (convex underestimator cv, concave
overestimator cc). The returned functions are compatible with jax.jit and
jax.vmap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from discopt._jax.learned_relaxations import LearnedRelaxationRegistry

from discopt._jax.mccormick import (
    relax_abs,
    relax_acos,
    relax_add,
    relax_asin,
    relax_atan,
    relax_bilinear,
    relax_cos,
    relax_cosh,
    relax_div,
    relax_exp,
    relax_log,
    relax_log2,
    relax_log10,
    relax_neg,
    relax_pow,
    relax_sign,
    relax_sin,
    relax_sinh,
    relax_sqrt,
    relax_sub,
    relax_tan,
    relax_tanh,
)
from discopt._jax.piecewise_mccormick import (
    piecewise_mccormick_bilinear,
    piecewise_relax_cos,
    piecewise_relax_exp,
    piecewise_relax_log,
    piecewise_relax_sin,
    piecewise_relax_sqrt,
)

# Import expression types from the modeling API
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    FunctionCall,
    IndexExpression,
    Model,
    Parameter,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
)


def _compute_var_offset(var: Variable, model: Model) -> int:
    """Compute the starting offset of a variable in the flat x vector."""
    offset = 0
    for v in model._variables[: var._index]:
        offset += v.size
    return offset


def _is_constant_expr(expr: Expression) -> bool:
    """Check if an expression is a Constant."""
    return isinstance(expr, Constant)


def _get_constant_value(expr: Expression):
    """Get the numeric value from a Constant expression."""
    return jnp.array(expr.value)


def _compile_relax_node(
    expr: Expression,
    model: Model,
    partitions: int = 0,
    mode: str = "standard",
    learned_registry: Optional["LearnedRelaxationRegistry"] = None,
) -> Callable:
    """
    Recursively compile an Expression node into a relaxation function.

    Each returned function takes (x_cv, x_cc, lb, ub) and returns (cv, cc)
    where cv is a convex underestimator and cc is a concave overestimator.

    Args:
        expr: Expression node to compile.
        model: Model containing variable definitions.
        partitions: If > 0, use piecewise McCormick relaxations with this
            many partitions for supported operations (bilinear, exp, log,
            sqrt, sin, cos). If 0, use standard McCormick.
        mode: Relaxation mode — ``"standard"`` (default McCormick),
            ``"piecewise"`` (piecewise McCormick), or ``"learned"``
            (ICNN-based learned relaxations with McCormick fallback).
        learned_registry: Registry of trained learned relaxations.
            Required when ``mode="learned"``.
    """

    if isinstance(expr, Constant):
        val = jnp.array(expr.value)

        def fn(x_cv, x_cc, lb, ub):
            return val, val

        return fn

    if isinstance(expr, Variable):
        offset = _compute_var_offset(expr, model)
        size = expr.size
        shape = expr.shape
        if shape == () or (len(shape) == 1 and shape[0] == 1 and shape == ()):

            def fn(x_cv, x_cc, lb, ub):
                return x_cv[offset], x_cc[offset]

            return fn
        else:

            def fn(x_cv, x_cc, lb, ub, _offset=offset, _size=size, _shape=shape):
                return (
                    x_cv[_offset : _offset + _size].reshape(_shape),
                    x_cc[_offset : _offset + _size].reshape(_shape),
                )

            return fn

    if isinstance(expr, Parameter):
        val = jnp.array(expr.value)

        def fn(x_cv, x_cc, lb, ub):
            return val, val

        return fn

    if isinstance(expr, BinaryOp):
        left_fn = _compile_relax_node(expr.left, model, partitions, mode, learned_registry)
        right_fn = _compile_relax_node(expr.right, model, partitions, mode, learned_registry)
        op = expr.op

        if op == "+":

            def fn(x_cv, x_cc, lb, ub):
                cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
                return relax_add(cv_l, cc_l, cv_r, cc_r)

            return fn

        if op == "-":

            def fn(x_cv, x_cc, lb, ub):
                cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
                return relax_sub(cv_l, cc_l, cv_r, cc_r)

            return fn

        if op == "*":
            # Optimize constant * expr and expr * constant
            if _is_constant_expr(expr.left):
                c = _get_constant_value(expr.left)

                def fn(x_cv, x_cc, lb, ub, _c=c):
                    cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
                    pos = _c >= 0
                    new_cv = jnp.where(pos, _c * cv_r, _c * cc_r)
                    new_cc = jnp.where(pos, _c * cc_r, _c * cv_r)
                    return new_cv, new_cc

                return fn

            if _is_constant_expr(expr.right):
                c = _get_constant_value(expr.right)

                def fn(x_cv, x_cc, lb, ub, _c=c):
                    cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                    pos = _c >= 0
                    new_cv = jnp.where(pos, _c * cv_l, _c * cc_l)
                    new_cc = jnp.where(pos, _c * cc_l, _c * cv_l)
                    return new_cv, new_cc

                return fn

            # General bilinear: use cv/cc as bounds for McCormick envelopes,
            # but also try to propagate tighter variable-level bounds.

            # Learned relaxation for bilinear
            if mode == "learned" and learned_registry is not None:
                lr_bilinear = learned_registry.get("bilinear")
                if lr_bilinear is not None:

                    def fn(x_cv, x_cc, lb, ub, _lr=lr_bilinear):
                        cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                        cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
                        mid_l = 0.5 * (cv_l + cc_l)
                        mid_r = 0.5 * (cv_r + cc_r)
                        true_val = mid_l * mid_r
                        xy = jnp.stack([mid_l, mid_r])
                        xy_lb = jnp.stack([cv_l, cv_r])
                        xy_ub = jnp.stack([cc_l, cc_r])
                        return _lr(xy, xy_lb, xy_ub, true_val)

                    return fn

            if partitions > 0:
                _k = partitions

                def fn(x_cv, x_cc, lb, ub, _pw_k=_k):
                    cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                    cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
                    mid_l = 0.5 * (cv_l + cc_l)
                    mid_r = 0.5 * (cv_r + cc_r)
                    return piecewise_mccormick_bilinear(
                        mid_l, mid_r, cv_l, cc_l, cv_r, cc_r, k=_pw_k
                    )

                return fn

            def fn(x_cv, x_cc, lb, ub):
                cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
                # Use the midpoint of cv/cc as the "x" value for bilinear
                mid_l = 0.5 * (cv_l + cc_l)
                mid_r = 0.5 * (cv_r + cc_r)
                return relax_bilinear(mid_l, mid_r, cv_l, cc_l, cv_r, cc_r)

            return fn

        if op == "/":
            if _is_constant_expr(expr.right):
                c = _get_constant_value(expr.right)
                inv_c = 1.0 / c

                def fn(x_cv, x_cc, lb, ub, _inv_c=inv_c):
                    cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                    pos = _inv_c >= 0
                    new_cv = jnp.where(pos, _inv_c * cv_l, _inv_c * cc_l)
                    new_cc = jnp.where(pos, _inv_c * cc_l, _inv_c * cv_l)
                    return new_cv, new_cc

                return fn

            def fn(x_cv, x_cc, lb, ub):
                cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
                mid_l = 0.5 * (cv_l + cc_l)
                mid_r = 0.5 * (cv_r + cc_r)
                return relax_div(mid_l, mid_r, cv_l, cc_l, cv_r, cc_r)

            return fn

        if op == "**":
            # Integer power: use tight envelope when base is a plain variable
            if _is_constant_expr(expr.right):
                n_val = expr.right.value
                n_int = int(n_val)
                if np.isclose(float(n_val), float(n_int)):
                    # Check if the base is a plain variable for tight bounds
                    if isinstance(expr.left, (Variable, IndexExpression)):
                        from discopt._jax.envelopes import relax_power_int

                        if isinstance(expr.left, Variable) and expr.left.size == 1:
                            vi = _compute_var_offset(expr.left, model)

                            def fn(x_cv, x_cc, lb, ub, _n=n_int, _vi=vi):
                                return relax_power_int(x_cv[_vi], lb[_vi], ub[_vi], _n)

                            return fn
                        elif isinstance(expr.left, IndexExpression):
                            if isinstance(expr.left.base, Variable):
                                base_off = _compute_var_offset(expr.left.base, model)
                                idx = expr.left.index
                                flat_idx = (
                                    base_off + idx
                                    if isinstance(idx, int)
                                    else base_off + idx[0]
                                    if isinstance(idx, tuple) and len(idx) == 1
                                    else None
                                )
                                if flat_idx is not None:

                                    def fn(x_cv, x_cc, lb, ub, _n=n_int, _fi=flat_idx):
                                        return relax_power_int(x_cv[_fi], lb[_fi], ub[_fi], _n)

                                    return fn

                    # Fallback: compositional McCormick
                    def fn(x_cv, x_cc, lb, ub, _n=n_int):
                        cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                        mid = 0.5 * (cv_l + cc_l)
                        return relax_pow(mid, cv_l, cc_l, _n)

                    return fn

            # General case: x^y = exp(y * log(x))
            def fn(x_cv, x_cc, lb, ub):
                cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
                # Use midpoints for evaluation
                mid_l = 0.5 * (cv_l + cc_l)
                mid_r = 0.5 * (cv_r + cc_r)
                # Relaxation of log(x)
                log_cv, log_cc = relax_log(mid_l, cv_l, cc_l)
                # Relaxation of y * log(x) via bilinear
                mid_log = 0.5 * (log_cv + log_cc)
                prod_cv, prod_cc = relax_bilinear(mid_r, mid_log, cv_r, cc_r, log_cv, log_cc)
                # Relaxation of exp(product)
                mid_prod = 0.5 * (prod_cv + prod_cc)
                return relax_exp(mid_prod, prod_cv, prod_cc)

            return fn

        raise ValueError(f"Unknown binary operator: {op!r}")

    if isinstance(expr, UnaryOp):
        operand_fn = _compile_relax_node(expr.operand, model, partitions, mode, learned_registry)
        op = expr.op

        if op == "neg":

            def fn(x_cv, x_cc, lb, ub):
                cv_child, cc_child = operand_fn(x_cv, x_cc, lb, ub)
                return relax_neg(cv_child, cc_child)

            return fn

        if op == "abs":

            def fn(x_cv, x_cc, lb, ub):
                cv_child, cc_child = operand_fn(x_cv, x_cc, lb, ub)
                mid = 0.5 * (cv_child + cc_child)
                return relax_abs(mid, cv_child, cc_child)

            return fn

        raise ValueError(f"Unknown unary operator: {op!r}")

    if isinstance(expr, FunctionCall):
        arg_fns = [
            _compile_relax_node(a, model, partitions, mode, learned_registry) for a in expr.args
        ]
        name = expr.func_name

        # Learned relaxation dispatch: use ICNN-based relaxations when available
        _learned_univariate_ops = {"exp", "log", "sqrt", "sin"}
        if mode == "learned" and learned_registry is not None and name in _learned_univariate_ops:
            lr_model = learned_registry.get(name)
            if lr_model is not None:
                a_fn = arg_fns[0]
                _true_fns = {
                    "exp": jnp.exp,
                    "log": jnp.log,
                    "sqrt": jnp.sqrt,
                    "sin": jnp.sin,
                }
                _tfn = _true_fns[name]

                def fn(x_cv, x_cc, lb, ub, _lr=lr_model, _af=a_fn, _tf=_tfn):
                    cv_child, cc_child = _af(x_cv, x_cc, lb, ub)
                    mid = 0.5 * (cv_child + cc_child)
                    true_val = _tf(mid)
                    return _lr(mid, cv_child, cc_child, true_val)

                return fn

        # Piecewise-capable univariate operations
        _piecewise_relax = {
            "exp": piecewise_relax_exp,
            "log": piecewise_relax_log,
            "sqrt": piecewise_relax_sqrt,
            "sin": piecewise_relax_sin,
            "cos": piecewise_relax_cos,
        }

        _univariate_relax = {
            "exp": relax_exp,
            "log": relax_log,
            "log2": relax_log2,
            "log10": relax_log10,
            "sqrt": relax_sqrt,
            "sin": relax_sin,
            "cos": relax_cos,
            "tan": relax_tan,
            "atan": relax_atan,
            "sinh": relax_sinh,
            "cosh": relax_cosh,
            "asin": relax_asin,
            "acos": relax_acos,
            "tanh": relax_tanh,
            "abs": relax_abs,
        }

        if partitions > 0 and name in _piecewise_relax:
            pw_fn = _piecewise_relax[name]
            a_fn = arg_fns[0]
            _k = partitions

            def fn(x_cv, x_cc, lb, ub, _pw_fn=pw_fn, _a_fn=a_fn, _pw_k=_k):
                cv_a, cc_a = _a_fn(x_cv, x_cc, lb, ub)
                mid = 0.5 * (cv_a + cc_a)
                return _pw_fn(mid, cv_a, cc_a, k=_pw_k)

            return fn

        if name in _univariate_relax:
            relax_fn = _univariate_relax[name]
            a_fn = arg_fns[0]

            def fn(x_cv, x_cc, lb, ub, _relax_fn=relax_fn, _a_fn=a_fn):
                cv_a, cc_a = _a_fn(x_cv, x_cc, lb, ub)
                mid = 0.5 * (cv_a + cc_a)
                return _relax_fn(mid, cv_a, cc_a)

            return fn

        if name == "sign":
            a_fn = arg_fns[0]

            def fn(x_cv, x_cc, lb, ub):
                cv_a, cc_a = a_fn(x_cv, x_cc, lb, ub)
                mid = 0.5 * (cv_a + cc_a)
                return relax_sign(mid, cv_a, cc_a)

            return fn

        if name == "min":
            a_fn, b_fn = arg_fns[0], arg_fns[1]

            def fn(x_cv, x_cc, lb, ub):
                cv_a, cc_a = a_fn(x_cv, x_cc, lb, ub)
                cv_b, cc_b = b_fn(x_cv, x_cc, lb, ub)
                from discopt._jax.mccormick import relax_min

                mid_a = 0.5 * (cv_a + cc_a)
                mid_b = 0.5 * (cv_b + cc_b)
                return relax_min(mid_a, mid_b, cv_a, cc_a, cv_b, cc_b)

            return fn

        if name == "max":
            a_fn, b_fn = arg_fns[0], arg_fns[1]

            def fn(x_cv, x_cc, lb, ub):
                cv_a, cc_a = a_fn(x_cv, x_cc, lb, ub)
                cv_b, cc_b = b_fn(x_cv, x_cc, lb, ub)
                from discopt._jax.mccormick import relax_max

                mid_a = 0.5 * (cv_a + cc_a)
                mid_b = 0.5 * (cv_b + cc_b)
                return relax_max(mid_a, mid_b, cv_a, cc_a, cv_b, cc_b)

            return fn

        raise ValueError(f"Unknown function: {name!r}")

    if isinstance(expr, IndexExpression):
        base_fn = _compile_relax_node(expr.base, model, partitions, mode, learned_registry)
        idx = expr.index

        def fn(x_cv, x_cc, lb, ub, _idx=idx):
            cv_base, cc_base = base_fn(x_cv, x_cc, lb, ub)
            return cv_base[_idx], cc_base[_idx]

        return fn

    if isinstance(expr, SumExpression):
        operand_fn = _compile_relax_node(expr.operand, model, partitions, mode, learned_registry)
        axis = expr.axis

        def fn(x_cv, x_cc, lb, ub, _axis=axis):
            cv_op, cc_op = operand_fn(x_cv, x_cc, lb, ub)
            return jnp.sum(cv_op, axis=_axis), jnp.sum(cc_op, axis=_axis)

        return fn

    if isinstance(expr, SumOverExpression):
        term_fns = [
            _compile_relax_node(t, model, partitions, mode, learned_registry) for t in expr.terms
        ]

        def fn(x_cv, x_cc, lb, ub):
            cv_acc, cc_acc = term_fns[0](x_cv, x_cc, lb, ub)
            for t_fn in term_fns[1:]:
                cv_t, cc_t = t_fn(x_cv, x_cc, lb, ub)
                cv_acc = cv_acc + cv_t
                cc_acc = cc_acc + cc_t
            return cv_acc, cc_acc

        return fn

    raise TypeError(f"Unhandled expression type: {type(expr).__name__}")


def compile_relaxation(
    expr: Expression,
    model: Model,
    partitions: int = 0,
    mode: str = "standard",
    learned_registry: Optional["LearnedRelaxationRegistry"] = None,
) -> Callable:
    """
    Compile an Expression into a McCormick relaxation function.

    Args:
        expr: Expression to relax
        model: Model containing variable definitions
        partitions: If > 0, use piecewise McCormick relaxations with this
            many partitions for supported operations (bilinear, exp, log,
            sqrt, sin, cos). If 0 (default), use standard McCormick.
        mode: Relaxation mode — ``"standard"`` (default), ``"piecewise"``,
            or ``"learned"`` (ICNN-based with McCormick fallback).
        learned_registry: Registry of trained learned relaxations.
            Required when ``mode="learned"``.

    Returns:
        A function f(x_cv, x_cc, lb, ub) -> (cv, cc) where:
          - x_cv: convex relaxation values for variables (1D flat array)
          - x_cc: concave relaxation values for variables (1D flat array)
          - lb: lower bounds for all variables (1D flat array)
          - ub: upper bounds for all variables (1D flat array)
          - cv: convex underestimator of expr
          - cc: concave overestimator of expr

        The function is compatible with jax.jit and jax.vmap.
    """
    return _compile_relax_node(expr, model, partitions, mode, learned_registry)


def compile_objective_relaxation(
    model: Model,
    partitions: int = 0,
    mode: str = "standard",
    learned_registry: Optional["LearnedRelaxationRegistry"] = None,
) -> Callable:
    """Compile relaxation of the model's objective."""
    if model._objective is None:
        raise ValueError("Model has no objective set.")
    return compile_relaxation(
        model._objective.expression, model, partitions, mode, learned_registry
    )


def compile_constraint_relaxation(
    constraint: Constraint,
    model: Model,
    partitions: int = 0,
    mode: str = "standard",
    learned_registry: Optional["LearnedRelaxationRegistry"] = None,
) -> Callable:
    """Compile relaxation of a constraint body."""
    return compile_relaxation(constraint.body, model, partitions, mode, learned_registry)
