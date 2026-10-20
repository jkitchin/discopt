"""Shared sign-tracking worst-case substitution for robust reformulations.

Both box and polyhedral formulations need to traverse the expression DAG
while tracking the effective sign of each uncertain parameter to determine
whether the upper or lower bound is the worst case.  This module provides
that traversal as a single reusable function.

The sign starts at +1 at the tree root and flips through ``-`` (subtraction)
and unary negation nodes.  Multiplication by a negative constant also flips
the sign.

Constant multipliers (``c * e``, ``e * c``, ``e / c``, ``C @ e``, ``e @ C``)
are signed element-wise, never by a fold such as ``np.sign(np.sum(c))``
(#1831). A constant whose nonzero entries share one sign flips (or keeps) the
scalar sign as before. A *mixed-sign* constant is split exactly as
``c = c+ + c-`` (``c+ = max(c, 0)``, ``c- = min(c, 0)``) and each part is
robustified with its own sign, so every element of ``e`` is substituted at the
worst-case bound for the sign of its own coefficient. Each occurrence of a
parameter is worst-cased independently, which is conservative (sound) in
general and exact when every parameter element enters the row once, affinely.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


def sign_tracking_substitute(
    expr,
    wc_lower: dict[str, np.ndarray],
    wc_upper: dict[str, np.ndarray],
    param_names: set[str],
    maximize: bool,
    sign: int,
):
    """Recursively replace uncertain Parameters with worst-case constants.

    Parameters
    ----------
    expr : Expression
        Sub-expression to rewrite.
    wc_lower : dict[str, ndarray]
        Maps parameter names to their worst-case lower bound arrays.
    wc_upper : dict[str, ndarray]
        Maps parameter names to their worst-case upper bound arrays.
    param_names : set[str]
        Names of uncertain parameters to substitute.
    maximize : bool
        Whether we are maximising (True) or minimising (False) the root
        expression.
    sign : int
        Accumulated sign (+1 or -1) of the current sub-expression relative
        to the root.  Flips at subtraction and unary negation.
    """
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )

    if isinstance(expr, Parameter) and expr.name in param_names:
        # Effective direction: if (maximize XOR sign<0), use upper bound.
        use_upper = maximize ^ (sign < 0)
        wc_value = wc_upper[expr.name] if use_upper else wc_lower[expr.name]
        return Constant(wc_value)

    if isinstance(expr, Constant):
        return expr

    if isinstance(expr, BinaryOp):
        if expr.op == "+":
            nl = sign_tracking_substitute(
                expr.left, wc_lower, wc_upper, param_names, maximize, sign
            )
            nr = sign_tracking_substitute(
                expr.right, wc_lower, wc_upper, param_names, maximize, sign
            )
        elif expr.op == "-":
            nl = sign_tracking_substitute(
                expr.left, wc_lower, wc_upper, param_names, maximize, sign
            )
            nr = sign_tracking_substitute(
                expr.right, wc_lower, wc_upper, param_names, maximize, sign * -1
            )
        elif expr.op == "*":
            # Propagate sign through multiplication by a constant-valued side
            # (the most common case: scalar * variable), element-wise (#1831).
            if _needs_multiplier_sign(expr.left, expr.right, param_names):
                return _signed_constant_product(
                    expr.left,
                    expr.right,
                    lambda c, e: BinaryOp("*", c, e),
                    wc_lower,
                    wc_upper,
                    param_names,
                    maximize,
                    sign,
                )
            if _needs_multiplier_sign(expr.right, expr.left, param_names):
                return _signed_constant_product(
                    expr.right,
                    expr.left,
                    lambda c, e: BinaryOp("*", e, c),
                    wc_lower,
                    wc_upper,
                    param_names,
                    maximize,
                    sign,
                )
            if isinstance(expr.left, Constant):
                nl = expr.left
                nr = sign_tracking_substitute(
                    expr.right, wc_lower, wc_upper, param_names, maximize, sign
                )
            elif isinstance(expr.right, Constant):
                nl = sign_tracking_substitute(
                    expr.left, wc_lower, wc_upper, param_names, maximize, sign
                )
                nr = expr.right
            else:
                # Neither side is a Constant node.  This could be:
                # (a) Parameter * Variable: the Parameter will be substituted
                #     by the recursive call, which is fine.
                # (b) True bilinear: two non-constant, non-parameter expressions
                #     both involving uncertain parameters.  This case cannot be
                #     handled correctly by sign tracking alone.
                # We warn only for case (b): both sides contain uncertain params.
                left_has_param = _contains_uncertain_param(expr.left, param_names)
                right_has_param = _contains_uncertain_param(expr.right, param_names)
                if left_has_param and right_has_param:
                    warnings.warn(
                        "Bilinear term detected in robust reformulation: "
                        "both sides of a multiplication contain uncertain "
                        "parameters.  The sign-tracking substitution may "
                        "not produce a conservative robust counterpart.  "
                        "Consider reformulating to separate uncertain "
                        "parameters from decision variables.",
                        stacklevel=4,
                    )
                nl = sign_tracking_substitute(
                    expr.left, wc_lower, wc_upper, param_names, maximize, sign
                )
                nr = sign_tracking_substitute(
                    expr.right, wc_lower, wc_upper, param_names, maximize, sign
                )
        elif expr.op == "/" and _needs_multiplier_sign(expr.right, expr.left, param_names):
            # ``e / d`` with a constant-valued divisor is ``e * (1/d)``: the
            # sign of each entry of ``d`` is the sign of its coefficient. A
            # negative divisor used to propagate without a flip (#1831 audit).
            return _signed_constant_product(
                expr.right,
                expr.left,
                lambda d, e: BinaryOp("/", e, d),
                wc_lower,
                wc_upper,
                param_names,
                maximize,
                sign,
                reciprocal=True,
            )
        else:
            # Division by a non-constant, power: propagate sign through without flip.
            nl = sign_tracking_substitute(
                expr.left, wc_lower, wc_upper, param_names, maximize, sign
            )
            nr = sign_tracking_substitute(
                expr.right, wc_lower, wc_upper, param_names, maximize, sign
            )
        return BinaryOp(expr.op, nl, nr)

    if isinstance(expr, UnaryOp):
        child_sign = sign * -1 if expr.op == "neg" else sign
        return UnaryOp(
            expr.op,
            sign_tracking_substitute(
                expr.operand, wc_lower, wc_upper, param_names, maximize, child_sign
            ),
        )

    if isinstance(expr, FunctionCall):
        new_args = [
            sign_tracking_substitute(a, wc_lower, wc_upper, param_names, maximize, sign)
            for a in expr.args
        ]
        return FunctionCall(expr.func_name, *new_args)

    if isinstance(expr, MatMulExpression):
        # ``C @ e`` / ``e @ C`` with a constant-valued ``C``: entry ``C[i, j]``
        # is the coefficient of ``e[j]`` in row ``i``, so its sign must reach
        # ``e`` just as a ``*`` multiplier's does. Passing the sign through
        # unchanged ignored every negative entry (#1831 audit).
        if _needs_multiplier_sign(expr.left, expr.right, param_names):
            return _signed_constant_product(
                expr.left,
                expr.right,
                lambda c, e: MatMulExpression(c, e),
                wc_lower,
                wc_upper,
                param_names,
                maximize,
                sign,
            )
        if _needs_multiplier_sign(expr.right, expr.left, param_names):
            return _signed_constant_product(
                expr.right,
                expr.left,
                lambda c, e: MatMulExpression(e, c),
                wc_lower,
                wc_upper,
                param_names,
                maximize,
                sign,
            )
        nl = sign_tracking_substitute(expr.left, wc_lower, wc_upper, param_names, maximize, sign)
        nr = sign_tracking_substitute(expr.right, wc_lower, wc_upper, param_names, maximize, sign)
        return MatMulExpression(nl, nr)

    if isinstance(expr, IndexExpression):
        nb = sign_tracking_substitute(expr.base, wc_lower, wc_upper, param_names, maximize, sign)
        return IndexExpression(nb, expr.index)

    if isinstance(expr, SumExpression):
        new_operand = sign_tracking_substitute(
            expr.operand, wc_lower, wc_upper, param_names, maximize, sign
        )
        return SumExpression(new_operand, expr.axis)

    if isinstance(expr, SumOverExpression):
        new_terms = [
            sign_tracking_substitute(t, wc_lower, wc_upper, param_names, maximize, sign)
            for t in expr.terms
        ]
        return SumOverExpression(new_terms)

    # Variable, unknown node: return as-is.
    return expr


def _needs_multiplier_sign(factor, other, param_names: set[str]) -> bool:
    """True when *factor* is a constant-valued multiplier of an uncertain *other*.

    Only then does the sign of *factor* decide a worst case. A factor that
    mentions a variable or an uncertain parameter is not a constant multiplier
    (those products are handled by the bilinear routing and the generic branch).
    """
    from discopt.modeling.core import Constant

    if not _contains_uncertain_param(other, param_names):
        return False
    if isinstance(factor, Constant):
        return True
    from discopt.ro.formulations.box import _contains_variable

    return not _contains_variable(factor) and not _contains_uncertain_param(factor, param_names)


def _signed_constant_product(
    factor,
    other,
    rebuild,
    wc_lower,
    wc_upper,
    param_names: set[str],
    maximize: bool,
    sign: int,
    *,
    reciprocal: bool = False,
):
    """Robustify ``rebuild(factor, other)`` where *factor* is constant-valued.

    The sign of each entry of *factor* is the sign of the coefficient it
    applies to (for ``reciprocal=True``, i.e. a divisor, the sign of ``1/d``,
    which is the same). If the nonzero entries share one sign, that sign is
    folded into ``sign`` and the node is rebuilt as before. If they are mixed,
    ``factor`` is split exactly into its positive and negative parts and each
    part is robustified with its own sign (#1831) -- the former
    ``np.sign(np.sum(c))`` gave all entries the sign of the sum, substituting
    every element whose coefficient had the other sign at its *best* case.
    """
    from discopt.modeling.core import BinaryOp, Constant

    value = np.asarray(eval_constant_expr(factor), dtype=np.float64)
    has_pos = bool(np.any(value > 0))
    has_neg = bool(np.any(value < 0))

    if not (has_pos and has_neg):
        factor_sign = -1 if has_neg else 1
        new_other = sign_tracking_substitute(
            other, wc_lower, wc_upper, param_names, maximize, sign * factor_sign
        )
        return rebuild(factor, new_other)

    if reciprocal:
        if np.any(value == 0):
            raise ValueError(
                "Robust counterpart: a constant divisor of an uncertain term has "
                "zero entries; the division is undefined."
            )
        # e / d == e * (1/d); split 1/d by sign and multiply.
        recip = 1.0 / value
        pos = Constant(np.where(recip > 0, recip, 0.0))
        neg = Constant(np.where(recip < 0, recip, 0.0))

        def rebuild_part(c, e):
            return BinaryOp("*", e, c)

    else:
        pos = Constant(np.where(value > 0, value, 0.0))
        neg = Constant(np.where(value < 0, value, 0.0))
        rebuild_part = rebuild
    pos_term = rebuild_part(
        pos, sign_tracking_substitute(other, wc_lower, wc_upper, param_names, maximize, sign)
    )
    neg_term = rebuild_part(
        neg, sign_tracking_substitute(other, wc_lower, wc_upper, param_names, maximize, -sign)
    )
    return BinaryOp("+", pos_term, neg_term)


_NUMPY_FUNCS = {
    "abs": np.abs,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "log2": np.log2,
    "log1p": np.log1p,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "asinh": np.arcsinh,
    "acosh": np.arccosh,
    "atanh": np.arctanh,
    "sign": np.sign,
}


def eval_constant_expr(expr) -> np.ndarray:
    """Evaluate a variable-free expression to its full numpy value.

    Every node is evaluated with numpy semantics (broadcasting, ``axis``,
    indexing) so a vector constant is never collapsed to a scalar. A node this
    evaluator does not know raises ``NotImplementedError`` rather than
    contributing a guessed value -- the former polyhedral evaluator returned
    ``float(np.sum(value))`` for a vector constant and ``0.0`` for any node it
    did not recognise, so ``sum(c * p)`` got a zero coefficient (#1831 audit).
    """
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )

    if isinstance(expr, (Constant, Parameter)):
        return np.asarray(expr.value, dtype=np.float64)
    if isinstance(expr, BinaryOp):
        lv = eval_constant_expr(expr.left)
        rv = eval_constant_expr(expr.right)
        if expr.op == "+":
            return np.asarray(lv + rv)
        if expr.op == "-":
            return np.asarray(lv - rv)
        if expr.op == "*":
            return np.asarray(lv * rv)
        if expr.op == "/":
            return np.asarray(lv / rv)
        if expr.op == "**":
            return np.asarray(np.power(lv, rv))
    elif isinstance(expr, UnaryOp):
        v = eval_constant_expr(expr.operand)
        if expr.op == "neg":
            return np.asarray(-v)
        if expr.op == "abs":
            return np.asarray(np.abs(v))
    elif isinstance(expr, MatMulExpression):
        return np.asarray(eval_constant_expr(expr.left) @ eval_constant_expr(expr.right))
    elif isinstance(expr, SumExpression):
        return np.asarray(np.sum(eval_constant_expr(expr.operand), axis=expr.axis))
    elif isinstance(expr, SumOverExpression):
        total = np.asarray(0.0)
        for t in expr.terms:
            total = total + eval_constant_expr(t)
        return np.asarray(total)
    elif isinstance(expr, IndexExpression):
        return np.asarray(eval_constant_expr(expr.base)[expr.index])
    elif isinstance(expr, FunctionCall) and expr.func_name in _NUMPY_FUNCS:
        if len(expr.args) == 1:
            return np.asarray(_NUMPY_FUNCS[expr.func_name](eval_constant_expr(expr.args[0])))
    raise NotImplementedError(
        f"Robust counterpart: cannot evaluate the constant sub-expression {expr!r} "
        f"({type(expr).__name__}) numerically, so the sign of the coefficient it "
        "applies to an uncertain parameter is unknown. Refusing rather than guessing."
    )


def _shape_of(expr) -> Optional[tuple[int, ...]]:
    """Static shape of *expr*, extending ``_known_shape`` to matmul and reductions.

    ``None`` when it cannot be determined statically. ``_known_shape`` gives up
    at a matmul or a reduction, and so does every element-wise node above one:
    ``x + M @ p`` has no known shape there, which is what kept such a row from
    being split per element (#1831).
    """
    from discopt.modeling.core import (
        BinaryOp,
        IndexExpression,
        MatMulExpression,
        SumExpression,
        SumOverExpression,
        UnaryOp,
        _known_shape,
    )

    known = _known_shape(expr)
    if known is not None:
        return known
    if isinstance(expr, (BinaryOp, SumOverExpression)):
        parts = [expr.left, expr.right] if isinstance(expr, BinaryOp) else expr.terms
        shapes = [_shape_of(t) for t in parts]
        if any(sh is None for sh in shapes):
            return None
        try:
            return tuple(int(d) for d in np.broadcast_shapes(*shapes))
        except ValueError:
            return None
    if isinstance(expr, UnaryOp):
        return _shape_of(expr.operand)
    if isinstance(expr, MatMulExpression):
        sl, sr = _shape_of(expr.left), _shape_of(expr.right)
        if sl is None or sr is None or not (1 <= len(sl) <= 2 and 1 <= len(sr) <= 2):
            return None
        if sl[-1] != sr[0]:
            return None
        return tuple(sl[:-1]) + tuple(sr[1:])
    if isinstance(expr, SumExpression):
        if expr.axis is None:
            return ()
        so = _shape_of(expr.operand)
        if so is None or not so:
            return None
        axis = expr.axis % len(so)
        return tuple(d for i, d in enumerate(so) if i != axis)
    if isinstance(expr, IndexExpression):
        sb = _shape_of(expr.base)
        if sb is None:
            return None
        try:
            return tuple(np.empty(sb, dtype=np.int8)[expr.index].shape)
        except (IndexError, TypeError, ValueError):
            return None
    return None


def substitute_param(expr, param_name: str, replacement):
    """Replace every occurrence of ``Parameter(param_name)`` with *replacement*.

    This is the parameter analogue of ``_substitute_var`` in affine_policy.py.
    Used by the coefficient-extraction approach for bilinear robust reformulation.
    """
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )

    if isinstance(expr, Parameter):
        return replacement if expr.name == param_name else expr
    if isinstance(expr, Constant):
        return expr
    if isinstance(expr, BinaryOp):
        return BinaryOp(
            expr.op,
            substitute_param(expr.left, param_name, replacement),
            substitute_param(expr.right, param_name, replacement),
        )
    if isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, substitute_param(expr.operand, param_name, replacement))
    if isinstance(expr, MatMulExpression):
        return MatMulExpression(
            substitute_param(expr.left, param_name, replacement),
            substitute_param(expr.right, param_name, replacement),
        )
    if isinstance(expr, FunctionCall):
        return FunctionCall(
            expr.func_name,
            *[substitute_param(a, param_name, replacement) for a in expr.args],
        )
    if isinstance(expr, IndexExpression):
        return IndexExpression(substitute_param(expr.base, param_name, replacement), expr.index)
    if isinstance(expr, SumExpression):
        return SumExpression(substitute_param(expr.operand, param_name, replacement), expr.axis)
    if isinstance(expr, SumOverExpression):
        return SumOverExpression([substitute_param(t, param_name, replacement) for t in expr.terms])
    return expr


def _contains_uncertain_param(expr, param_names: set[str]) -> bool:
    """Check whether *expr* contains any Parameter whose name is in *param_names*."""
    from discopt.modeling.core import (
        BinaryOp,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )

    if isinstance(expr, Parameter):
        return expr.name in param_names
    if isinstance(expr, (BinaryOp, MatMulExpression)):
        return _contains_uncertain_param(expr.left, param_names) or _contains_uncertain_param(
            expr.right, param_names
        )
    if isinstance(expr, UnaryOp):
        return _contains_uncertain_param(expr.operand, param_names)
    if isinstance(expr, FunctionCall):
        return any(_contains_uncertain_param(a, param_names) for a in expr.args)
    if isinstance(expr, IndexExpression):
        return _contains_uncertain_param(expr.base, param_names)
    if isinstance(expr, SumExpression):
        return _contains_uncertain_param(expr.operand, param_names)
    if isinstance(expr, SumOverExpression):
        return any(_contains_uncertain_param(t, param_names) for t in expr.terms)
    return False


def assert_no_uncertain_params_remain(model, param_names: set[str], *, kind: str) -> None:
    """Refuse loudly if any uncertain parameter survived robustification (RO-2).

    A correct robust counterpart eliminates every uncertain parameter (by
    worst-case substitution, an SOC/dual penalty, etc.). If one remains, the
    expression pattern was silently left at its nominal value — the user asked
    for a robust model and would get a non-robust one. This universal post-check
    turns every unsupported-pattern gap (RO-2, RO-4, RO-5, RO-6, RO-8) from a
    silent wrong answer into a loud ``NotImplementedError`` at ``formulate()``
    time, per the development philosophy (refuse loudly over silent approximation).
    """
    from discopt.modeling.core import Constraint

    leftovers: list[str] = []
    for con in model._constraints:
        if isinstance(con, Constraint) and _contains_uncertain_param(con.body, param_names):
            leftovers.append(con.name or "<unnamed constraint>")
    obj = model._objective
    if obj is not None and _contains_uncertain_param(obj.expression, param_names):
        leftovers.append("<objective>")

    if leftovers:
        raise NotImplementedError(
            f"The {kind} robust counterpart could not robustify uncertain "
            f"parameter(s) appearing in {leftovers}. The expression pattern in "
            "those terms is not supported by this formulation, so they were left "
            "at their nominal value — which would silently produce a NON-robust "
            "model. Rewrite the uncertain terms into a supported pattern (e.g. a "
            "linear/affine dependence on the parameter) or use a different "
            "uncertainty set. Refusing to return a silently non-robust model."
        )
