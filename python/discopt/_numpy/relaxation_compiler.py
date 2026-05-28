"""Numpy McCormick relaxation compiler.

Walks the Expression DAG and emits pure-numpy closures. Mirrors the
core dispatch of ``discopt._jax.relaxation_compiler`` but skips:

  * trilinear-exact / signomial-multi / learned / piecewise / oa_relax
  * envelope-based ops (asinh, acosh, atanh, erf, log1p, tight sin/cos)
  * non-default mode and arithmetic

For these, raises ``NotImplementedError`` so the dispatcher can fall
back to JAX. The numpy backend trades relaxation tightness for compile
speed — it is only useful on small B&B nodes where the JAX trace/compile
floor dominates wall time.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from discopt._numpy.mccormick import (
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
    relax_max,
    relax_min,
    relax_neg,
    relax_pow,
    relax_sigmoid,
    relax_sign,
    relax_sin,
    relax_sinh,
    relax_softplus,
    relax_sqrt,
    relax_sub,
    relax_tan,
    relax_tanh,
)
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

_SUPPORTED_FUNCS = {
    "exp",
    "log",
    "log2",
    "log10",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "atan",
    "asin",
    "acos",
    "sinh",
    "cosh",
    "tanh",
    "sigmoid",
    "softplus",
    "abs",
    "sign",
    "min",
    "max",
}


def supported_for_model(model: Model) -> bool:
    """Best-effort check whether all expressions in ``model`` are supported.

    Returns False on any unsupported op so the dispatcher can pick JAX
    instead of paying the cost of a failed numpy compile.
    """
    try:
        if model._objective is not None:
            _check_supported(model._objective.expression)
        for c in model._constraints:
            _check_supported(c.body)
    except NotImplementedError:
        return False
    return True


def _check_supported(expr: Expression) -> None:
    if isinstance(expr, (Constant, Variable, Parameter)):
        return
    if isinstance(expr, BinaryOp):
        if expr.op not in ("+", "-", "*", "/", "**"):
            raise NotImplementedError(f"BinaryOp {expr.op!r} not supported in numpy backend")
        _check_supported(expr.left)
        _check_supported(expr.right)
        return
    if isinstance(expr, UnaryOp):
        if expr.op not in ("neg", "abs"):
            raise NotImplementedError(f"UnaryOp {expr.op!r} not supported in numpy backend")
        _check_supported(expr.operand)
        return
    if isinstance(expr, FunctionCall):
        if expr.func_name not in _SUPPORTED_FUNCS:
            raise NotImplementedError(
                f"FunctionCall {expr.func_name!r} not supported in numpy backend"
            )
        for a in expr.args:
            _check_supported(a)
        return
    if isinstance(expr, IndexExpression):
        _check_supported(expr.base)
        return
    if isinstance(expr, SumExpression):
        _check_supported(expr.operand)
        return
    if isinstance(expr, SumOverExpression):
        for t in expr.terms:
            _check_supported(t)
        return
    raise NotImplementedError(f"Expression type {type(expr).__name__} not supported")


def _compute_var_offset(var: Variable, model: Model) -> int:
    offset = 0
    for v in model._variables[: var._index]:
        offset += v.size
    return offset


def _is_constant_expr(expr: Expression) -> bool:
    return isinstance(expr, Constant)


def _get_constant_value(expr: Expression):
    return np.asarray(expr.value)


def _compile_relax_node(expr: Expression, model: Model) -> Callable:
    if isinstance(expr, Constant):
        val = np.asarray(expr.value)

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

        def fn(x_cv, x_cc, lb, ub, _offset=offset, _size=size, _shape=shape):
            return (
                x_cv[_offset : _offset + _size].reshape(_shape),
                x_cc[_offset : _offset + _size].reshape(_shape),
            )

        return fn

    if isinstance(expr, Parameter):
        val = np.asarray(expr.value)

        def fn(x_cv, x_cc, lb, ub):
            return val, val

        return fn

    if isinstance(expr, BinaryOp):
        left_fn = _compile_relax_node(expr.left, model)
        right_fn = _compile_relax_node(expr.right, model)
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
            if _is_constant_expr(expr.left):
                c = _get_constant_value(expr.left)

                def fn(x_cv, x_cc, lb, ub, _c=c):
                    cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
                    pos = _c >= 0
                    new_cv = np.where(pos, _c * cv_r, _c * cc_r)
                    new_cc = np.where(pos, _c * cc_r, _c * cv_r)
                    return new_cv, new_cc

                return fn

            if _is_constant_expr(expr.right):
                c = _get_constant_value(expr.right)

                def fn(x_cv, x_cc, lb, ub, _c=c):
                    cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                    pos = _c >= 0
                    new_cv = np.where(pos, _c * cv_l, _c * cc_l)
                    new_cc = np.where(pos, _c * cc_l, _c * cv_l)
                    return new_cv, new_cc

                return fn

            def fn(x_cv, x_cc, lb, ub):
                cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
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
                    new_cv = np.where(pos, _inv_c * cv_l, _inv_c * cc_l)
                    new_cc = np.where(pos, _inv_c * cc_l, _inv_c * cv_l)
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
            if _is_constant_expr(expr.right):
                n_val = expr.right.value
                n_int = int(n_val)
                if np.isclose(float(n_val), float(n_int)):

                    def fn(x_cv, x_cc, lb, ub, _n=n_int):
                        cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                        mid = 0.5 * (cv_l + cc_l)
                        return relax_pow(mid, cv_l, cc_l, _n)

                    return fn

                alpha = float(n_val)
                if 0.0 < alpha < 1.0:
                    _alpha = alpha

                    def fn(x_cv, x_cc, lb, ub, _a=_alpha):
                        cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                        cv_l = np.maximum(cv_l, 1e-30)
                        cc_l = np.maximum(cc_l, 1e-30)
                        cc_out = cc_l**_a
                        fa = cv_l**_a
                        fb = cc_l**_a
                        slope = (fb - fa) / np.maximum(cc_l - cv_l, 1e-30)
                        mid = 0.5 * (cv_l + cc_l)
                        cv_out = fa + slope * (mid - cv_l)
                        return cv_out, cc_out

                    return fn

                if alpha > 1.0:
                    _alpha = alpha

                    def fn(x_cv, x_cc, lb, ub, _a=_alpha):
                        cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                        cv_l = np.maximum(cv_l, 1e-30)
                        cc_l = np.maximum(cc_l, 1e-30)
                        cv_out = cv_l**_a
                        fa = cv_l**_a
                        fb = cc_l**_a
                        slope = (fb - fa) / np.maximum(cc_l - cv_l, 1e-30)
                        mid = 0.5 * (cv_l + cc_l)
                        cc_out = fa + slope * (mid - cv_l)
                        return cv_out, cc_out

                    return fn

            # Non-constant exponent: x^y = exp(y * log(x))
            def fn(x_cv, x_cc, lb, ub):
                cv_l, cc_l = left_fn(x_cv, x_cc, lb, ub)
                cv_r, cc_r = right_fn(x_cv, x_cc, lb, ub)
                mid_l = 0.5 * (cv_l + cc_l)
                mid_r = 0.5 * (cv_r + cc_r)
                log_cv, log_cc = relax_log(mid_l, cv_l, cc_l)
                mid_log = 0.5 * (log_cv + log_cc)
                prod_cv, prod_cc = relax_bilinear(mid_r, mid_log, cv_r, cc_r, log_cv, log_cc)
                mid_prod = 0.5 * (prod_cv + prod_cc)
                return relax_exp(mid_prod, prod_cv, prod_cc)

            return fn

        raise NotImplementedError(f"Unknown binary operator: {op!r}")

    if isinstance(expr, UnaryOp):
        operand_fn = _compile_relax_node(expr.operand, model)
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

        raise NotImplementedError(f"Unknown unary operator: {op!r}")

    if isinstance(expr, FunctionCall):
        arg_fns = [_compile_relax_node(a, model) for a in expr.args]
        name = expr.func_name

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
            "sigmoid": relax_sigmoid,
            "softplus": relax_softplus,
            "abs": relax_abs,
        }

        if name in _univariate_relax:
            a_fn = arg_fns[0]
            relax_fn = _univariate_relax[name]

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
                mid_a = 0.5 * (cv_a + cc_a)
                mid_b = 0.5 * (cv_b + cc_b)
                return relax_min(mid_a, mid_b, cv_a, cc_a, cv_b, cc_b)

            return fn

        if name == "max":
            a_fn, b_fn = arg_fns[0], arg_fns[1]

            def fn(x_cv, x_cc, lb, ub):
                cv_a, cc_a = a_fn(x_cv, x_cc, lb, ub)
                cv_b, cc_b = b_fn(x_cv, x_cc, lb, ub)
                mid_a = 0.5 * (cv_a + cc_a)
                mid_b = 0.5 * (cv_b + cc_b)
                return relax_max(mid_a, mid_b, cv_a, cc_a, cv_b, cc_b)

            return fn

        raise NotImplementedError(f"Unknown function: {name!r}")

    if isinstance(expr, IndexExpression):
        base_fn = _compile_relax_node(expr.base, model)
        idx = expr.index

        def fn(x_cv, x_cc, lb, ub, _idx=idx):
            cv_base, cc_base = base_fn(x_cv, x_cc, lb, ub)
            return cv_base[_idx], cc_base[_idx]

        return fn

    if isinstance(expr, SumExpression):
        operand_fn = _compile_relax_node(expr.operand, model)
        axis = expr.axis

        def fn(x_cv, x_cc, lb, ub, _axis=axis):
            cv_op, cc_op = operand_fn(x_cv, x_cc, lb, ub)
            return np.sum(cv_op, axis=_axis), np.sum(cc_op, axis=_axis)

        return fn

    if isinstance(expr, SumOverExpression):
        term_fns = [_compile_relax_node(t, model) for t in expr.terms]

        def fn(x_cv, x_cc, lb, ub):
            cv_acc, cc_acc = term_fns[0](x_cv, x_cc, lb, ub)
            for t_fn in term_fns[1:]:
                cv_t, cc_t = t_fn(x_cv, x_cc, lb, ub)
                cv_acc = cv_acc + cv_t
                cc_acc = cc_acc + cc_t
            return cv_acc, cc_acc

        return fn

    raise NotImplementedError(f"Unhandled expression type: {type(expr).__name__}")


def compile_relaxation(expr: Expression, model: Model) -> Callable:
    return _compile_relax_node(expr, model)


def compile_objective_relaxation(model: Model) -> Callable:
    if model._objective is None:
        raise ValueError("Model has no objective set.")
    return compile_relaxation(model._objective.expression, model)


def compile_constraint_relaxation(constraint: Constraint, model: Model) -> Callable:
    return compile_relaxation(constraint.body, model)
