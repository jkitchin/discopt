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
worst-case bound for the sign of its own coefficient.

Sign tracking worst-cases each *occurrence* of a parameter independently. That
is exact only when no element's coefficients cancel across occurrences:
``sum(M @ p)`` with ``M = [[3, -1], [-2, 1]]`` is ``p0`` exactly, yet tracking
the two rows of ``M`` separately charges ``|3| + |-2|`` for ``p0`` and
``|-1| + |1|`` for ``p1`` -- a counterpart so conservative it reported a
feasible model infeasible (#1831 review). :func:`affine_param_jacobians`
therefore extracts the exact aggregate coefficient of a row that is
structurally affine in the uncertain parameters with constant coefficients, and
:func:`aggregate_affine_counterpart` builds the exact box counterpart
``g(x, p_bar) +- sum_j delta_j * |coeff_j|`` whenever sign tracking would
over-count. Rows that are not affine keep sign tracking.

A *certain* :class:`Parameter` that multiplies an uncertain one (``q * p`` with
a box on ``p`` only) stays symbolic in the counterpart: a later
``RobustCounterpart`` pass over ``q`` must still find it, and freezing it into a
:class:`Constant` made that pass robustify nothing (#1839 review). Its value
does decide the worst-case direction of ``p``, so that decision is recorded as
a sign assumption on the model (:data:`SIGN_GUARD_ATTR`): a later pass that
makes ``q`` uncertain checks the assumption over ``q``'s set, and
``Model.validate`` re-checks it before every solve, so a ``q.value`` that
changes sign is refused instead of silently mis-robustified. The exact
aggregate route takes no coefficient that holds a Parameter.
Above a *nonlinear* node the sign may pass to the uncertain operand only when the
node is proven monotone in that operand over the whole uncertainty box (#1836).
The former traversal passed the sign through ``1/p``, ``p**k``, ``abs`` and every
``FunctionCall`` unchanged, so a decreasing or non-monotone node had its operand
substituted at the *best* case: ``x + 1/p <= 10`` over ``p in [0.5, 1.5]``
certified ``x = 9.33`` (robust truth ``x = 8``). Each nonlinear node is now
handled in exactly one of these ways, and never by passing the sign unproven:

* **Monotone by proof.** The direction is the sign of the node's partial
  derivative in the uncertain operand, decided by interval evaluation of the
  *other* operands over the box and the variable bounds (``a / d`` is
  decreasing in ``d`` when ``a >= 0`` and ``d`` excludes 0; ``u**k`` follows the
  parity of ``k`` and the sign of ``u``; ``exp``/``log``/``sqrt``/... increase;
  ``acos`` decreases; ``sin``/``cos``/``tan`` by the sign of their derivative over
  the operand's range; ``min``/``max`` increase in every argument; a product by
  the signs of its factors). Sign-definite is enough: the sign then flips or
  stays, and the node is rebuilt.
* **Even nodes over a sign-indefinite operand** (``abs(u)``, ``u**(2k)``,
  ``cosh(u)``) use the exact identities ``max |u| = max(max u, -min u)`` and
  ``min |u| = max(min u, -max u, 0)``, then the even function is increasing in
  ``|u|``.
* **Exact interval worst case.** A variable-free subtree for which no proof is
  available (``p * p`` with ``p`` of indefinite sign, a mixed-sign numerator over
  ``p``) is replaced by the endpoint of its interval over the box.
* **Refused** with ``NotImplementedError`` naming the node otherwise -- in
  particular a division by, or a negative power of, an uncertain quantity whose
  range contains 0, and any node with no monotonicity rule. A certain
  Parameter whose value would decide the direction at a nonlinear node (``c/p``,
  ``p**c``, the sign of ``q * x`` multiplying ``1/p``) is refused too: there is no
  symbolic split of such a node to record a sign assumption on, and evaluating it
  would freeze the Parameter.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

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

    ctx = _Ctx(wc_lower, wc_upper, param_names, maximize, sign)

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
            elif _contains_uncertain_param(expr, param_names):
                # Neither side is a constant multiplier: ``p * x`` with ``x >= 0``
                # (RO-1 routes a sign-indefinite ``x`` to the |coeff| path),
                # ``p * p``, ``(1/p) * y``, ... ``A * B`` is increasing in ``A``
                # where ``B >= 0`` and decreasing where ``B <= 0``, so each
                # uncertain factor takes the sign of the other factor over the
                # box. The former code passed the sign to both factors whatever
                # their signs and only warned when both held a parameter (#1836).
                directions = _product_directions(expr, ctx)
                if isinstance(directions, str) and expr.left is expr.right:
                    # ``u * u`` of indefinite sign is ``|u| * |u|``.
                    abs_wc = _abs_worst_case(expr.left, ctx)
                    return BinaryOp("*", abs_wc, abs_wc)
                return _nonlinear_substitute(expr, ctx, [expr.left, expr.right], directions)
            else:
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
        elif expr.op in ("/", "**") and _contains_uncertain_param(expr, param_names):
            # Division by a non-constant, or a power, over an uncertain operand.
            # The former code passed the sign through unchanged, which is right
            # only where the node is increasing in that operand -- ``1/p`` and
            # ``p**-1`` are decreasing, ``p**2`` is not monotone (#1836).
            if expr.op == "/":
                directions = _division_directions(expr, ctx)
            else:
                directions = _power_directions(expr, ctx)
            if directions is _EVEN_IN_BASE:
                # ``u ** (2k)`` with ``u`` of indefinite sign: ``|u| ** (2k)``.
                return BinaryOp("**", _abs_worst_case(expr.left, ctx), expr.right)
            return _nonlinear_substitute(expr, ctx, [expr.left, expr.right], directions)
        else:
            nl = sign_tracking_substitute(
                expr.left, wc_lower, wc_upper, param_names, maximize, sign
            )
            nr = sign_tracking_substitute(
                expr.right, wc_lower, wc_upper, param_names, maximize, sign
            )
        return BinaryOp(expr.op, nl, nr)

    if isinstance(expr, UnaryOp):
        if expr.op == "abs" and _contains_uncertain_param(expr.operand, param_names):
            # ``abs`` follows the sign of its operand; it passed the sign through
            # unchanged, so ``abs(p)`` over a negative box was substituted at its
            # best case (#1836).
            return _even_function_substitute(expr, expr.operand, ctx, "abs")
        if expr.op not in ("neg", "abs") and _contains_uncertain_param(expr, param_names):
            _refuse(expr, f"no monotonicity rule for the unary operator {expr.op!r}")
        child_sign = sign * -1 if expr.op == "neg" else sign
        return UnaryOp(
            expr.op,
            sign_tracking_substitute(
                expr.operand, wc_lower, wc_upper, param_names, maximize, child_sign
            ),
        )

    if isinstance(expr, FunctionCall):
        if _contains_uncertain_param(expr, param_names):
            # A named function is monotone only by proof (#1836): ``exp`` and
            # ``log`` increase, ``acos`` decreases, ``abs``/``cosh`` follow the
            # sign of their argument, ``sin``/``cos``/``tan`` the sign of their
            # derivative over the argument's range.
            if len(expr.args) == 1 and expr.func_name in ("abs", "cosh"):
                return _even_function_substitute(expr, expr.args[0], ctx, expr.func_name)
            directions = _function_directions(expr, ctx)
            return _nonlinear_substitute(expr, ctx, list(expr.args), directions)
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
        if _contains_uncertain_param(expr, param_names):
            # ``p @ x`` with ``x >= 0`` (RO-1 routes any other ``x`` to the
            # |coeff| path), ``p @ p``, ...: as for ``*``, each uncertain side
            # takes the sign of the other side's entries (#1836).
            return _nonlinear_substitute(
                expr, ctx, [expr.left, expr.right], _product_directions(expr, ctx)
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

    if _contains_uncertain_param(expr, param_names):
        # An opaque node (``CustomCall``) over an uncertain parameter has no
        # monotonicity rule; returning it as-is left the parameter at nominal.
        _refuse(expr, f"no worst-case rule for a {type(expr).__name__} node")
    # Variable, unknown node: return as-is.
    return expr


#: Model attribute holding the sign assumptions a robust counterpart relied on,
#: as ``(factor_expr, required_sign, strict, label)``: ``factor_expr`` mentions
#: no uncertain parameter, ``required_sign`` (+1 / -1, broadcast over the
#: factor's entries) is the sign its worst-case direction was read from, and
#: ``strict`` excludes 0. ``required_sign=None`` marks an expression whose
#: Parameters must stay certain (:func:`record_must_stay_certain`). Set only
#: when a record exists. Re-checked by ``Model.validate`` before every solve (it is one of
#: ``discopt.modeling.core.VALIDATION_GUARD_ATTRS``) and by a later
#: ``RobustCounterpart.formulate`` over the set it makes uncertain. Not written
#: by :mod:`discopt.serialize`, which therefore refuses a model that carries it
#: rather than reloading it unguarded (only a model whose counterpart kept a
#: certain Parameter symbolic carries one).
SIGN_GUARD_ATTR = "_ro_sign_preconditions"


class RobustSignAssumptionError(ValueError):
    """A sign a robust counterpart was built on no longer holds."""


def _contains_any_parameter(expr, exclude: frozenset[str] | set[str] = frozenset()) -> bool:
    """True when *expr* mentions a :class:`Parameter` not named in *exclude*."""
    from discopt.modeling.core import Parameter, _iter_model_leaves

    return any(
        isinstance(leaf, Parameter) and leaf.name not in exclude
        for leaf in _iter_model_leaves(expr)
    )


def record_sign_assumption(factor, required_sign, *, strict: bool, label: str) -> None:
    # ``required_sign=None`` records a must-stay-certain assumption instead
    # (:func:`record_must_stay_certain`).
    """Record that a worst-case direction was read from the sign of *factor*.

    Callers record only factors that mention a certain :class:`Parameter`; a
    factor with no model leaf at all is ignored here.
    """
    from discopt.modeling.core import _iter_model_leaves

    model = next((leaf.model for leaf in _iter_model_leaves(factor)), None)
    if model is None:
        return
    records = getattr(model, SIGN_GUARD_ATTR, None)
    if records is None:
        records = []
        setattr(model, SIGN_GUARD_ATTR, records)
    records.append(
        (
            factor,
            None if required_sign is None else np.asarray(required_sign, dtype=np.int8),
            bool(strict),
            f"robust counterpart: {label}",
        )
    )


def record_must_stay_certain(expr, label: str) -> None:
    """Record that the counterpart is exact only while *expr*'s Parameters are certain.

    The polyhedral dual equality ``coeff(q) = A^T lam`` is exact for any fixed
    ``q``, but a later pass making ``q`` uncertain would demand it for every
    ``q`` with one ``lam`` -- a false infeasibility (``lam`` must adjust with
    ``q``). Value changes are fine; only a later robustification is refused.
    """
    record_sign_assumption(expr, None, strict=False, label=label)


def check_sign_assumptions(model, ranged: Optional[dict] = None) -> None:
    """Raise :class:`RobustSignAssumptionError` if a recorded sign no longer holds.

    Without *ranged* each factor is enclosed over the current variable bounds
    with every Parameter at its current value (``Model.validate``). With
    *ranged* -- ``{parameter_name: (lo, hi)}`` for parameters a new
    ``RobustCounterpart`` pass is about to make uncertain -- only the factors
    mentioning one of them are checked, with those parameters ranged over
    their sets: the earlier pass's direction stays a worst case for every
    value in the set only if the sign does.
    """
    from discopt._relax.convexity.interval import Interval
    from discopt._relax.convexity.interval_eval import evaluate_interval, param_box_key

    box = {}
    for name, (lo, hi) in (ranged or {}).items():
        box[param_box_key(name)] = Interval(
            np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
        )
    for factor, required, strict, label in getattr(model, SIGN_GUARD_ATTR, None) or []:
        if ranged is not None and not _contains_uncertain_param(factor, set(ranged)):
            continue
        if required is None:
            if ranged is not None:
                raise RobustSignAssumptionError(
                    f"{label}: an earlier robust counterpart is exact only while the "
                    f"parameter(s) in {factor!r} are certain (a dual equality holds them), "
                    f"and this pass makes {', '.join(sorted(ranged))} uncertain. Robustify "
                    "every uncertain parameter in one RobustCounterpart instead."
                )
            continue
        enc = evaluate_interval(factor, None, box)
        lo = np.asarray(enc.lo, dtype=np.float64)
        hi = np.asarray(enc.hi, dtype=np.float64)
        req = np.broadcast_to(required, lo.shape) if lo.shape else required.reshape(())
        pos, neg = req > 0, req < 0
        if strict:
            ok = bool(np.all(lo[pos] > 0) and np.all(hi[neg] < 0))
        else:
            ok = bool(np.all(lo[pos] >= 0) and np.all(hi[neg] <= 0))
        if not ok:
            where = (
                "over the uncertainty set of " + ", ".join(sorted(ranged))
                if ranged
                else "at the current parameter values and variable bounds"
            )
            raise RobustSignAssumptionError(
                f"{label}: the robust counterpart chose the worst case of an uncertain "
                f"parameter from the sign of {factor!r}, which no longer holds {where} "
                f"(enclosure [{np.min(lo)}, {np.max(hi)}]). The counterpart would be "
                "non-robust. Robustify every uncertain parameter in one "
                "RobustCounterpart, or rebuild the model after changing the value or "
                "bound."
            )


def uncertainty_bounding_box(unc) -> tuple[np.ndarray, np.ndarray]:
    """Element-wise bounds ``(lo, hi)`` of the parameter over its uncertainty set."""
    pbar = np.asarray(unc.parameter.value, dtype=np.float64)
    if unc.kind == "box":
        return np.asarray(unc.lower), np.asarray(unc.upper)
    if unc.kind == "ellipsoidal":
        half = unc.rho * np.sqrt(np.maximum(np.diag(unc.Sigma), 0.0))
        half = half.reshape(pbar.shape)
        return pbar - half, pbar + half
    from discopt.ro.formulations.polyhedral import _support_function_lp

    k = pbar.size
    lo = np.empty(k)
    hi = np.empty(k)
    for j in range(k):
        e = np.zeros(unc.A.shape[1])
        e[j] = 1.0
        hi[j] = _support_function_lp(e, unc.A, unc.b, True)
        lo[j] = _support_function_lp(e, unc.A, unc.b, False)
    return pbar + lo.reshape(pbar.shape), pbar + hi.reshape(pbar.shape)


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
    from discopt.modeling.core import BinaryOp, Constant, Expression

    value = np.asarray(eval_constant_expr(factor), dtype=np.float64)
    has_pos = bool(np.any(value > 0))
    has_neg = bool(np.any(value < 0))

    if reciprocal and np.any(value == 0):
        raise ValueError(
            "Robust counterpart: a constant divisor of an uncertain term has "
            "zero entries; the division is undefined."
        )

    symbolic = not isinstance(factor, Constant) and _contains_any_parameter(factor)

    if not (has_pos and has_neg):
        factor_sign = -1 if has_neg else 1
        if symbolic:
            # The factor keeps its Parameter; the direction read from its value
            # is a recorded assumption (module docstring, #1839 review).
            record_sign_assumption(
                factor, np.array(factor_sign), strict=reciprocal, label="constant multiplier"
            )
        new_other = sign_tracking_substitute(
            other, wc_lower, wc_upper, param_names, maximize, sign * factor_sign
        )
        return rebuild(factor, new_other)

    if reciprocal:
        # e / d == e * (1/d); split 1/d by sign and multiply.
        recip = 1.0 / value
        if symbolic:
            record_sign_assumption(
                factor, np.where(value < 0, -1, 1), strict=True, label="mixed-sign divisor"
            )
            pos_mask = Constant(np.where(recip > 0, 1.0, 0.0))
            neg_mask = Constant(np.where(recip < 0, 1.0, 0.0))

            def masked(mask, e):
                return BinaryOp("*", BinaryOp("/", e, factor), mask)

            pos_term = masked(
                pos_mask,
                sign_tracking_substitute(other, wc_lower, wc_upper, param_names, maximize, sign),
            )
            neg_term = masked(
                neg_mask,
                sign_tracking_substitute(other, wc_lower, wc_upper, param_names, maximize, -sign),
            )
            return BinaryOp("+", pos_term, neg_term)
        pos: Expression = Constant(np.where(recip > 0, recip, 0.0))
        neg: Expression = Constant(np.where(recip < 0, recip, 0.0))

        def rebuild_part(c, e):
            return BinaryOp("*", e, c)

    elif symbolic:
        # Split the Parameter-valued factor by the sign of each entry at its
        # current value, keeping it symbolic (zero entries join the positive
        # part, so a later value may be >= 0 there).
        record_sign_assumption(
            factor, np.where(value < 0, -1, 1), strict=False, label="mixed-sign multiplier"
        )
        pos = BinaryOp("*", factor, Constant(np.where(value >= 0, 1.0, 0.0)))
        neg = BinaryOp("*", factor, Constant(np.where(value < 0, 1.0, 0.0)))
        rebuild_part = rebuild
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


class _NotAffine(Exception):
    """Internal: the expression is not affine in the uncertain parameters with
    constant coefficients (or uses a node the extractor does not model)."""


def _value_shape(expr) -> tuple[int, ...]:
    """Static shape of *expr* (:func:`_shape_of`), or :class:`_NotAffine`."""
    shape = _shape_of(expr)
    if shape is None:
        raise _NotAffine(f"shape of {type(expr).__name__} is not statically known")
    return shape


def affine_param_jacobians(expr, param_names: set[str]):
    """Exact constant coefficients of *expr* in its uncertain parameters.

    Returns ``{name: (J, S)}`` when *expr* is structurally affine in every
    uncertain parameter it mentions and each coefficient is variable-free, or
    ``None`` otherwise. ``J`` has shape ``out + P`` (``out`` the shape of the
    parameter-bearing part of *expr*, which broadcasts against *expr*; ``P``
    the parameter's shape) and is the exact derivative: *expr* equals
    ``expr(p_bar) + sum J * (p - p_bar)`` for every ``p``. ``S`` has the same
    shape and accumulates ``|J|`` *per occurrence* -- the coefficient magnitude
    sign tracking charges. ``S > |J|`` exactly where occurrences cancel.

    Affinity is decided structurally (``p`` reaches the root only through
    ``+ - neg``, sums, indexing, and products, quotients or matmuls with a
    constant-valued side), never by sampling: a finite second difference
    passes ``abs(p - p_bar)``.
    """
    from discopt.modeling.core import (
        BinaryOp,
        IndexExpression,
        MatMulExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )
    from discopt.ro.formulations.box import _contains_variable

    def add(a, b, sign_b):
        out = dict(a)
        for name, (jb, sb) in b.items():
            if name in out:
                ja, sa = out[name]
                out[name] = (ja + sign_b * jb, sa + sb)
            else:
                out[name] = (sign_b * jb, sb)
        return out

    def broadcast(jac, shape):
        out = {}
        for name, (j, s) in jac.items():
            pshape = j.shape[j.ndim - pdims[name] :]
            target = tuple(shape) + pshape
            out[name] = (np.broadcast_to(j, target), np.broadcast_to(s, target))
        return out

    def constant_factor(factor):
        if _contains_variable(factor):
            raise _NotAffine("coefficient contains a decision variable")
        if _contains_any_parameter(factor):
            # A certain Parameter coefficient must stay symbolic (module
            # docstring); sign tracking handles it under a recorded assumption.
            raise _NotAffine("coefficient contains a certain Parameter")
        return np.asarray(eval_constant_expr(factor), dtype=np.float64)

    def scale(jac, c):
        out = {}
        for name, (j, s) in jac.items():
            cx = c.reshape(c.shape + (1,) * pdims[name])
            out[name] = (j * cx, s * np.abs(cx))
        return out

    def matmul(jac, c, c_left):
        out = {}
        for name, (j, s) in jac.items():
            k = pdims[name]
            pshape = j.shape[j.ndim - k :]
            eshape = j.shape[: j.ndim - k]
            jf = j.reshape(eshape + (-1,))
            sf = s.reshape(eshape + (-1,))
            ca = np.abs(c)
            if c_left:
                cols_j = [c @ jf[..., i] for i in range(jf.shape[-1])]
                cols_s = [ca @ sf[..., i] for i in range(sf.shape[-1])]
            else:
                cols_j = [jf[..., i] @ c for i in range(jf.shape[-1])]
                cols_s = [sf[..., i] @ ca for i in range(sf.shape[-1])]
            nj = np.stack(cols_j, axis=-1)
            ns = np.stack(cols_s, axis=-1)
            oshape = nj.shape[:-1]
            out[name] = (nj.reshape(oshape + pshape), ns.reshape(oshape + pshape))
        return out

    def has_p(e):
        return _contains_uncertain_param(e, param_names)

    def jac(e):
        if not has_p(e):
            return {}
        if isinstance(e, Parameter):
            pshape = tuple(np.shape(e.value))
            n = int(np.prod(pshape)) if pshape else 1
            pdims[e.name] = len(pshape)
            eye = np.eye(n).reshape(pshape + pshape)
            return {e.name: (eye, eye.copy())}
        if isinstance(e, BinaryOp):
            if e.op in ("+", "-"):
                return add(jac(e.left), jac(e.right), 1.0 if e.op == "+" else -1.0)
            if e.op == "*":
                if has_p(e.left) and has_p(e.right):
                    raise _NotAffine("product of two uncertain terms")
                if has_p(e.left):
                    return scale(jac(e.left), constant_factor(e.right))
                return scale(jac(e.right), constant_factor(e.left))
            if e.op == "/" and not has_p(e.right):
                d = constant_factor(e.right)
                if np.any(d == 0):
                    raise ValueError(
                        "Robust counterpart: a constant divisor of an uncertain term "
                        "has zero entries; the division is undefined."
                    )
                return scale(jac(e.left), 1.0 / d)
            raise _NotAffine(f"operator {e.op!r} on an uncertain term")
        if isinstance(e, UnaryOp):
            if e.op == "neg":
                return {n: (-j, s) for n, (j, s) in jac(e.operand).items()}
            raise _NotAffine(f"unary {e.op!r} on an uncertain term")
        if isinstance(e, MatMulExpression):
            if has_p(e.left) and has_p(e.right):
                raise _NotAffine("matmul of two uncertain terms")
            if has_p(e.right):
                c = constant_factor(e.left)
                inner = broadcast(jac(e.right), _value_shape(e.right))
                return matmul(inner, c, c_left=True)
            c = constant_factor(e.right)
            inner = broadcast(jac(e.left), _value_shape(e.left))
            return matmul(inner, c, c_left=False)
        if isinstance(e, SumExpression):
            oshape = _value_shape(e.operand)
            inner = broadcast(jac(e.operand), oshape)
            nd = len(oshape)
            if e.axis is None:
                axes: tuple[int, ...] = tuple(range(nd))
            else:
                raw = e.axis if isinstance(e.axis, tuple) else (e.axis,)
                axes = tuple(a % nd for a in raw)
            return {n: (np.sum(j, axis=axes), np.sum(s, axis=axes)) for n, (j, s) in inner.items()}
        if isinstance(e, SumOverExpression):
            total: dict = {}
            for t in e.terms:
                total = add(total, jac(t), 1.0)
            return total
        if isinstance(e, IndexExpression):
            idx = e.index if isinstance(e.index, tuple) else (e.index,)
            if any(i is Ellipsis or i is None for i in idx):
                raise _NotAffine("ellipsis/newaxis index on an uncertain term")
            inner = broadcast(jac(e.base), _value_shape(e.base))
            return {n: (j[idx], s[idx]) for n, (j, s) in inner.items()}
        raise _NotAffine(f"{type(e).__name__} on an uncertain term")

    pdims: dict[str, int] = {}
    try:
        return {n: (np.asarray(j), np.asarray(s)) for n, (j, s) in jac(expr).items()}
    except _NotAffine:
        return None


def aggregate_affine_counterpart(expr, unc_sets, param_names: set[str], maximize: bool):
    """Exact box counterpart of an affine row whose occurrences cancel, else ``None``.

    For ``g(x, p) = g(x, p_bar) + sum_j coeff_j * (p_j - p_bar_j)`` with constant
    ``coeff`` the worst case over ``|p_j - p_bar_j| <= delta_j`` is
    ``g(x, p_bar) + sum_j delta_j * |coeff_j|`` (``-`` when minimising),
    element-wise for a vector row. ``coeff`` is the *aggregate* coefficient from
    :func:`affine_param_jacobians`, so occurrences that cancel are netted before
    the absolute value -- the step sign tracking cannot take.

    Returns ``None`` (the caller keeps sign tracking) when *expr* is not affine
    with constant coefficients, or when no coefficient cancels: sign tracking is
    then already exact, and keeping it leaves every such formulation unchanged.
    *unc_sets* maps parameter names to box uncertainty sets.
    """
    from discopt.modeling.core import BinaryOp, Constant

    jacs = affine_param_jacobians(expr, param_names)
    if not jacs:
        return None
    cancels = False
    for j, s in jacs.values():
        # A relative threshold: a cancellation smaller than ~1e-12 * S (e.g. a
        # residue of ``1e12 * p - 1e12 * p`` rounding) is not detected and the
        # row keeps sign tracking. That miss is only conservative -- sign
        # tracking charges S >= |J| -- never optimistic.
        if np.any(s - np.abs(j) > 1e-12 * np.maximum(1.0, s)):
            cancels = True
    if not cancels:
        return None
    result = expr
    for name in jacs:
        result = substitute_param(result, name, Constant(unc_sets[name].parameter.value))
    for name, (j, _s) in jacs.items():
        delta = np.asarray(unc_sets[name].delta, dtype=np.float64)
        k = delta.ndim
        pen = np.sum(np.abs(j) * delta, axis=tuple(range(j.ndim - k, j.ndim)))
        result = BinaryOp("+" if maximize else "-", result, Constant(np.asarray(pen)))
    return result


class _Ctx(NamedTuple):
    """The arguments of one ``sign_tracking_substitute`` call, for the helpers."""

    wc_lower: dict
    wc_upper: dict
    param_names: set
    maximize: bool
    sign: int

    def rec(self, expr, sign: int):
        return sign_tracking_substitute(
            expr, self.wc_lower, self.wc_upper, self.param_names, self.maximize, sign
        )

    @property
    def want_upper(self) -> bool:
        """Whether the worst case of the current node is its largest value."""
        return self.maximize ^ (self.sign < 0)


# A direction function returns one entry per operand -- +1 (increasing in it),
# -1 (decreasing), None (the operand holds no uncertain parameter) -- or, when no
# direction is proven, a string saying why. ``_EVEN_IN_BASE`` marks
# ``u ** (2k)`` with ``u`` of indefinite sign, robustified through ``|u|``.
_EVEN_IN_BASE = object()

# Named functions increasing (non-decreasing) on their whole domain, and decreasing.
_INCREASING_FUNCS = frozenset(
    {
        "exp",
        "log",
        "log2",
        "log10",
        "log1p",
        "sqrt",
        "sinh",
        "tanh",
        "asin",
        "atan",
        "asinh",
        "acosh",
        "atanh",
        "erf",
        "sigmoid",
        "softplus",
        "sign",
    }
)
_DECREASING_FUNCS = frozenset({"acos"})


class _NoInterval(Exception):
    """The interval evaluator has no rule for a node."""


def _refuse(expr, reason: str):
    text = repr(expr)
    if len(text) > 200:
        text = text[:200] + "..."
    raise NotImplementedError(
        f"Robust counterpart: cannot robustify {text}: {reason}. Sign tracking may "
        "pass a worst-case direction through a nonlinear node only where the node "
        "is proven monotone over the uncertainty box; passing it unproven "
        "certifies non-robust points (#1836). Refusing rather than guessing. "
        "Rewrite the uncertain term (e.g. introduce q = 1/p as the uncertain "
        "parameter so it enters affinely) or bound its operands so the sign is fixed."
    )


def _refuse_certain_parameter(param):
    """A certain Parameter cannot decide a worst-case direction (#1836).

    The direction through a nonlinear node would be read from the parameter's
    ``formulate()``-time value and silently frozen into the counterpart, which a
    later update of the parameter could not honour.
    """
    _refuse(
        param,
        f"the worst-case direction through this nonlinear node depends on the value "
        f"of the certain Parameter {param.name!r}, which would be frozen at its "
        "formulate()-time value; declare it uncertain or replace it by a constant",
    )


def _first_certain_parameter(expr, ctx: _Ctx):
    """The first Parameter in *expr* that is not one of the uncertain ones, or None."""
    from discopt.modeling.core import (
        BinaryOp,
        CustomCall,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )

    stack = [expr]
    while stack:
        e = stack.pop()
        if isinstance(e, Parameter):
            if e.name not in ctx.param_names:
                return e
        elif isinstance(e, (BinaryOp, MatMulExpression)):
            stack += [e.left, e.right]
        elif isinstance(e, UnaryOp):
            stack.append(e.operand)
        elif isinstance(e, (FunctionCall, CustomCall)):
            stack += list(e.args)
        elif isinstance(e, IndexExpression):
            stack.append(e.base)
        elif isinstance(e, SumExpression):
            stack.append(e.operand)
        elif isinstance(e, SumOverExpression):
            stack += list(e.terms)
    return None


def _has_param(expr, ctx: _Ctx) -> bool:
    return _contains_uncertain_param(expr, ctx.param_names)


def _rebuild(expr, operands):
    from discopt.modeling.core import BinaryOp, FunctionCall, MatMulExpression

    if isinstance(expr, BinaryOp):
        return BinaryOp(expr.op, *operands)
    if isinstance(expr, MatMulExpression):
        return MatMulExpression(*operands)
    return FunctionCall(expr.func_name, *operands)


def _nonlinear_substitute(expr, ctx: _Ctx, operands, directions):
    """Robustify a nonlinear node from its proven operand directions.

    Each uncertain operand is substituted in the direction the node moves with
    it. Without a proof (``directions`` is a reason string) a variable-free node
    is replaced by its exact interval endpoint over the box; anything else is
    refused.
    """
    if isinstance(directions, str):
        return _interval_worst_case(expr, ctx, directions)
    new = [ctx.rec(op, ctx.sign * (1 if d is None else d)) for op, d in zip(operands, directions)]
    return _rebuild(expr, new)


def _interval_worst_case(expr, ctx: _Ctx, reason: str):
    """Replace a variable-free *expr* by the worst endpoint of its range over the box."""
    from discopt.modeling.core import Constant
    from discopt.ro.formulations.box import _contains_variable

    if _contains_variable(expr):
        _refuse(expr, reason)
    try:
        lo, hi = _interval(expr, ctx)
    except _NoInterval as exc:
        _refuse(expr, f"{reason}, and {exc}")
    value = hi if ctx.want_upper else lo
    if not np.all(np.isfinite(value)):
        _refuse(expr, f"{reason}, and its range over the box is unbounded")
    return Constant(np.asarray(value, dtype=np.float64))


def _sign_over_box(expr, ctx: _Ctx, *, strict: bool = False) -> int:
    """+1 / -1 when every element of *expr* is >= 0 / <= 0 over the box, else 0.

    The range comes from interval evaluation over the uncertainty box and the
    variable bounds. Interval arithmetic treats each occurrence independently,
    so its range contains the true one: a sign it proves holds. ``strict``
    requires the range to exclude 0.
    """
    try:
        lo, hi = _interval(expr, ctx)
    except _NoInterval:
        return 0
    if strict:
        if np.all(lo > 0):
            return 1
        if np.all(hi < 0):
            return -1
        return 0
    if np.all(lo >= 0):
        return 1
    if np.all(hi <= 0):
        return -1
    return 0


def _product_directions(expr, ctx: _Ctx):
    """``A * B`` / ``A @ B``: each uncertain factor takes the sign of the other."""
    lp, rp = _has_param(expr.left, ctx), _has_param(expr.right, ctx)
    dl = _sign_over_box(expr.right, ctx) if lp else None
    dr = _sign_over_box(expr.left, ctx) if rp else None
    if dl == 0 or dr == 0:
        return "the sign of the other factor is not fixed over the box"
    return [dl, dr]


def _division_directions(expr, ctx: _Ctx):
    """``A / B``: increasing in ``A`` with the sign of ``1/B``, in ``B`` with ``-sign(A)``."""
    lp, rp = _has_param(expr.left, ctx), _has_param(expr.right, ctx)
    sb = _sign_over_box(expr.right, ctx, strict=True)
    if sb == 0:
        return "the divisor's range over the box contains 0"
    da = sb if lp else None
    db = None
    if rp:
        sa = _sign_over_box(expr.left, ctx)
        if sa == 0:
            return (
                "the numerator's sign is not fixed, so the quotient is not monotone in the divisor"
            )
        db = -sa
    return [da, db]


def _power_directions(expr, ctx: _Ctx):
    """``B ** E``: the sign of ``E * B**(E-1)`` in ``B`` and of ``log B`` in ``E``."""
    from discopt.ro.formulations.box import _contains_variable

    base, exponent = expr.left, expr.right
    bp, ep = _has_param(base, ctx), _has_param(exponent, ctx)
    if ep:
        # d/dE B**E = B**E log B, which needs B > 0; d/dB = E B**(E-1).
        try:
            blo, bhi = _interval(base, ctx)
        except _NoInterval:
            return "the base of an uncertain exponent has no interval rule"
        if not np.all(blo > 0):
            return "an uncertain exponent over a base whose range is not positive"
        if np.all(blo >= 1):
            de = 1
        elif np.all(bhi <= 1):
            de = -1
        else:
            return "the base's range straddles 1, so the power is not monotone in the exponent"
        db = None
        if bp:
            db = _sign_over_box(exponent, ctx)
            if db == 0:
                return "the exponent's sign is not fixed, so the power is not monotone in the base"
        return [db, de]

    if not bp:
        return [None, None]
    if _contains_variable(exponent):
        return "the exponent depends on a decision variable"
    certain = _first_certain_parameter(exponent, ctx)
    if certain is not None:
        _refuse_certain_parameter(certain)
    k = np.asarray(eval_constant_expr(exponent), dtype=np.float64)
    if np.all((k == 0) | (k == 1)):
        return [1, None]
    if np.all(k == np.round(k)):
        odd = np.mod(k, 2) == 1
        if np.all(k > 0):
            if np.all(odd):
                return [1, None]
            sb = _sign_over_box(base, ctx)
            if sb > 0:
                return [1, None]
            if sb < 0 and not np.any(odd):
                return [-1, None]
            if sb == 0 and not np.any(odd):
                return _EVEN_IN_BASE
            return "mixed-parity exponents over a base of indefinite sign"
        if np.all(k < 0):
            sb = _sign_over_box(base, ctx, strict=True)
            if sb == 0:
                return "a negative power of an uncertain quantity whose range contains 0"
            # k odd: k-1 even, so the derivative has the sign of k; k even: -sign(B).
            dirs = np.where(odd, -1, -sb)
            if np.all(dirs == dirs.flat[0]):
                return [int(dirs.flat[0]), None]
            return "mixed-parity negative exponents over a negative base"
        return "exponents of mixed sign"
    # A fractional power is defined for B >= 0 (B > 0 for k < 0) and moves with k.
    sb = _sign_over_box(base, ctx, strict=bool(np.any(k < 0)))
    if sb <= 0:
        return "a fractional power of an uncertain quantity whose range is not nonnegative"
    if np.all(k > 0):
        return [1, None]
    if np.all(k < 0):
        return [-1, None]
    return "exponents of mixed sign"


def _function_directions(expr, ctx: _Ctx):
    """Directions of a named function in each argument."""
    name, args = expr.func_name, expr.args
    if name in ("min", "max"):
        # Non-decreasing in every argument (and, one-argument, in every element).
        return [1 if _has_param(a, ctx) else None for a in args]
    if len(args) != 1:
        return f"no monotonicity rule for {name!r} with {len(args)} arguments"
    if name in _INCREASING_FUNCS:
        return [1]
    if name in _DECREASING_FUNCS:
        return [-1]
    if name in ("sin", "cos", "tan"):
        try:
            lo, hi = _interval(args[0], ctx)
        except _NoInterval:
            return f"the argument of {name!r} has no interval rule"
        if name == "cos":
            # d/du cos u = -sin u.
            slo, shi = _cos_interval(lo - np.pi / 2, hi - np.pi / 2)
            d = -1 if np.all(slo >= 0) else (1 if np.all(shi <= 0) else 0)
        else:
            clo, chi = _cos_interval(lo, hi)
            if name == "tan":
                d = 1 if (np.all(clo > 0) or np.all(chi < 0)) else 0
            else:
                d = 1 if np.all(clo >= 0) else (-1 if np.all(chi <= 0) else 0)
        if d == 0:
            return f"{name!r} is not monotone over its argument's range"
        return [d]
    return f"no monotonicity rule for the function {name!r}"


def _even_function_substitute(expr, operand, ctx: _Ctx, name: str):
    """``abs(u)`` / ``cosh(u)``: follow the sign of ``u``, else go through ``|u|``."""
    from discopt.modeling.core import FunctionCall, UnaryOp

    s = _sign_over_box(operand, ctx)
    if s != 0:
        new = ctx.rec(operand, ctx.sign * s)
        return UnaryOp(expr.op, new) if isinstance(expr, UnaryOp) else FunctionCall(name, new)
    abs_wc = _abs_worst_case(operand, ctx)
    return abs_wc if name == "abs" else FunctionCall(name, abs_wc)


def _abs_worst_case(u, ctx: _Ctx):
    """Exact worst case of ``|u|`` over the box, for ``u`` of indefinite sign.

    ``max |u| = max(max u, -min u)`` and ``min |u| = max(min u, -max u, 0)``;
    ``max u`` and ``min u`` are the sign-tracking worst cases of ``u`` in the two
    directions. The result is >= 0, so an even function increasing in ``|u|``
    (``u**2``, ``cosh``) may be applied to it.
    """
    from discopt.modeling.core import Constant, FunctionCall, UnaryOp

    same = ctx.rec(u, ctx.sign)
    other = ctx.rec(u, -ctx.sign)
    core = FunctionCall("max", same, UnaryOp("neg", other))
    if ctx.want_upper:
        return core
    return FunctionCall("max", core, Constant(np.asarray(0.0)))


def _cos_interval(lo, hi):
    """Exact range of ``cos`` over ``[lo, hi]``, element-wise."""
    two_pi = 2.0 * np.pi
    with np.errstate(invalid="ignore", over="ignore"):
        clo = np.minimum(np.cos(lo), np.cos(hi))
        chi = np.maximum(np.cos(lo), np.cos(hi))
        # cos reaches 1 at 2k*pi and -1 at (2k+1)*pi inside the interval.
        has_max = np.floor(hi / two_pi) >= np.ceil(lo / two_pi)
        has_min = np.floor((hi - np.pi) / two_pi) >= np.ceil((lo - np.pi) / two_pi)
        wide = ~np.isfinite(lo) | ~np.isfinite(hi) | (hi - lo >= two_pi)
    chi = np.where(has_max | wide, 1.0, chi)
    clo = np.where(has_min | wide, -1.0, clo)
    return clo, chi


def _mul_interval(alo, ahi, blo, bhi):
    with np.errstate(invalid="ignore"):
        prods = [alo * blo, alo * bhi, ahi * blo, ahi * bhi]
    # 0 * inf is 0 in interval arithmetic (the 0 is exact).
    prods = [np.where(np.isnan(q), 0.0, q) for q in prods]
    return np.minimum.reduce(prods), np.maximum.reduce(prods)


def _pow_interval_scalar(lo: float, hi: float, k: float) -> tuple[float, float]:
    if k == 0:
        return 1.0, 1.0
    if k == round(k):
        if k > 0:
            if int(k) % 2 == 1:
                return lo**k, hi**k
            alo = 0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
            return alo**k, max(abs(lo), abs(hi)) ** k
        if lo <= 0 <= hi:
            return -np.inf, np.inf
        a, b = lo**k, hi**k
        return min(a, b), max(a, b)
    if lo < 0 or (k < 0 and lo == 0):
        return -np.inf, np.inf
    a, b = lo**k, hi**k
    return min(a, b), max(a, b)


def _monotone_values(name: str, v):
    from math import erf

    with np.errstate(all="ignore"):
        if name == "erf":
            return np.vectorize(erf, otypes=[float])(v)
        if name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-v))
        if name == "softplus":
            return np.logaddexp(v, 0.0)
        return _NUMPY_FUNCS[name](v)


def _interval(expr, ctx: _Ctx):
    """Range ``(lo, hi)`` of *expr* over the uncertainty box and the variable bounds.

    Each occurrence of a parameter or variable ranges independently, so the
    result contains the true range (and equals it when each enters once). A
    node without a rule raises ``_NoInterval``; NaN (a domain violation) widens
    to an infinite endpoint.
    """
    lo, hi = _interval_raw(expr, ctx)
    lo = np.where(np.isnan(lo), -np.inf, lo)
    hi = np.where(np.isnan(hi), np.inf, hi)
    return lo, hi


def _interval_raw(expr, ctx: _Ctx):
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
        Variable,
    )

    if isinstance(expr, Parameter) and expr.name in ctx.param_names:
        lo = np.asarray(ctx.wc_lower[expr.name], dtype=np.float64)
        hi = np.asarray(ctx.wc_upper[expr.name], dtype=np.float64)
        return np.minimum(lo, hi), np.maximum(lo, hi)
    if isinstance(expr, Parameter):
        _refuse_certain_parameter(expr)
    if isinstance(expr, Constant):
        v = np.asarray(expr.value, dtype=np.float64)
        return v, v
    if isinstance(expr, Variable):
        lb = np.broadcast_to(np.asarray(expr.lb, dtype=np.float64), expr.shape)
        ub = np.broadcast_to(np.asarray(expr.ub, dtype=np.float64), expr.shape)
        # The modelling layer's "unbounded" is a finite +-9.999e19 sentinel.
        return np.where(lb <= -1e19, -np.inf, lb), np.where(ub >= 1e19, np.inf, ub)
    if isinstance(expr, IndexExpression):
        lo, hi = _interval(expr.base, ctx)
        return np.asarray(lo[expr.index]), np.asarray(hi[expr.index])
    if isinstance(expr, SumExpression):
        lo, hi = _interval(expr.operand, ctx)
        return np.sum(lo, axis=expr.axis), np.sum(hi, axis=expr.axis)
    if isinstance(expr, SumOverExpression):
        tlo, thi = np.asarray(0.0), np.asarray(0.0)
        for t in expr.terms:
            lo, hi = _interval(t, ctx)
            tlo, thi = tlo + lo, thi + hi
        return tlo, thi
    if isinstance(expr, UnaryOp):
        lo, hi = _interval(expr.operand, ctx)
        if expr.op == "neg":
            return -hi, -lo
        if expr.op == "abs":
            return _abs_interval(lo, hi)
        raise _NoInterval(f"no interval rule for the unary operator {expr.op!r}")
    if isinstance(expr, BinaryOp):
        alo, ahi = _interval(expr.left, ctx)
        blo, bhi = _interval(expr.right, ctx)
        if expr.op == "+":
            return alo + blo, ahi + bhi
        if expr.op == "-":
            return alo - bhi, ahi - blo
        if expr.op == "*":
            if expr.left is expr.right:
                # ``u * u`` is a square, not a product of independent factors.
                return _abs_square_interval(alo, ahi)
            return _mul_interval(alo, ahi, blo, bhi)
        if expr.op == "/":
            if not (np.all(blo > 0) or np.all(bhi < 0)):
                shape = np.broadcast_shapes(np.shape(alo), np.shape(blo))
                return np.full(shape, -np.inf), np.full(shape, np.inf)
            with np.errstate(divide="ignore"):
                return _mul_interval(alo, ahi, 1.0 / bhi, 1.0 / blo)
        if expr.op == "**":
            if np.all(blo == bhi):
                lo_b, hi_b, k = np.broadcast_arrays(alo, ahi, blo)
                f = np.vectorize(_pow_interval_scalar, otypes=[float, float])
                with np.errstate(all="ignore"):
                    return f(lo_b, hi_b, k)
            if np.all(alo > 0):
                # B**E = exp(E log B) for B > 0.
                plo, phi = _mul_interval(np.log(alo), np.log(ahi), blo, bhi)
                return np.exp(plo), np.exp(phi)
            raise _NoInterval("no interval rule for an uncertain exponent over a non-positive base")
        raise _NoInterval(f"no interval rule for the binary operator {expr.op!r}")
    if isinstance(expr, MatMulExpression):
        alo, ahi = _interval(expr.left, ctx)
        blo, bhi = _interval(expr.right, ctx)
        if not (1 <= alo.ndim <= 2 and 1 <= blo.ndim <= 2):
            raise _NoInterval("no interval rule for a matmul of this rank")
        a2lo, a2hi = np.atleast_2d(alo), np.atleast_2d(ahi)
        b2lo = blo[:, None] if blo.ndim == 1 else blo
        b2hi = bhi[:, None] if bhi.ndim == 1 else bhi
        plo, phi = _mul_interval(
            a2lo[:, :, None], a2hi[:, :, None], b2lo[None, :, :], b2hi[None, :, :]
        )
        lo, hi = plo.sum(axis=1), phi.sum(axis=1)
        if blo.ndim == 1:
            lo, hi = lo[:, 0], hi[:, 0]
        if alo.ndim == 1:
            lo, hi = lo[0], hi[0]
        return lo, hi
    if isinstance(expr, FunctionCall):
        name, args = expr.func_name, expr.args
        ivs = [_interval(a, ctx) for a in args]
        if name in ("min", "max") and len(args) >= 2:
            red = np.minimum if name == "min" else np.maximum
            return red.reduce([iv[0] for iv in ivs]), red.reduce([iv[1] for iv in ivs])
        if len(args) != 1:
            raise _NoInterval(f"no interval rule for {name!r} with {len(args)} arguments")
        lo, hi = ivs[0]
        if name in ("min", "max"):
            red1 = np.min if name == "min" else np.max
            return np.asarray(red1(lo)), np.asarray(red1(hi))
        if name in _INCREASING_FUNCS:
            return _monotone_values(name, lo), _monotone_values(name, hi)
        if name in _DECREASING_FUNCS:
            return _monotone_values(name, hi), _monotone_values(name, lo)
        if name == "abs":
            return _abs_interval(lo, hi)
        if name == "cosh":
            alo, ahi = _abs_interval(lo, hi)
            return np.cosh(alo), np.cosh(ahi)
        if name == "cos":
            return _cos_interval(lo, hi)
        if name == "sin":
            return _cos_interval(lo - np.pi / 2, hi - np.pi / 2)
        if name == "prod":
            plo, phi = np.asarray(1.0), np.asarray(1.0)
            for el, eh in zip(np.ravel(lo), np.ravel(hi)):
                plo, phi = _mul_interval(plo, phi, np.asarray(el), np.asarray(eh))
            return plo, phi
        raise _NoInterval(f"no interval rule for the function {name!r}")
    raise _NoInterval(f"no interval rule for a {type(expr).__name__} node")


def _abs_square_interval(lo, hi):
    alo, ahi = _abs_interval(lo, hi)
    return alo * alo, ahi * ahi


def _abs_interval(lo, hi):
    straddle = (lo < 0) & (hi > 0)
    alo = np.where(straddle, 0.0, np.minimum(np.abs(lo), np.abs(hi)))
    ahi = np.maximum(np.abs(lo), np.abs(hi))
    return alo, ahi


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
        shapes: list[tuple[int, ...]] = []
        for t in parts:
            sh = _shape_of(t)
            if sh is None:
                return None
            shapes.append(sh)
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


def is_affine_in_params(expr, param_names: set[str]) -> bool:
    """True when *expr* is structurally affine in the uncertain parameters.

    The parameters may reach the root only through ``+ - neg``, sums, indexing,
    and products, quotients or matmuls whose other side is free of uncertain
    parameters (it may hold decision variables: ``y * p`` is affine in ``p``).
    The finite-difference coefficient ``g(p_bar + e_j) - g(p_bar)`` that the
    polyhedral and box-linearization counterparts dualize is the exact
    coefficient of ``xi_j`` only then; for ``exp(p)`` or ``p * p`` it is a
    secant, and the dual reformulation treated a nonlinear row as linear,
    certifying non-robust points (#1841). Decided structurally, never by
    sampling.
    """
    from discopt.modeling.core import (
        BinaryOp,
        IndexExpression,
        MatMulExpression,
        Parameter,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )

    def has_p(e):
        return _contains_uncertain_param(e, param_names)

    def affine(e) -> bool:
        if not has_p(e):
            return True
        if isinstance(e, Parameter):
            return True
        if isinstance(e, BinaryOp):
            if e.op in ("+", "-"):
                return affine(e.left) and affine(e.right)
            if e.op == "*":
                if has_p(e.left) and has_p(e.right):
                    return False
                return affine(e.left) and affine(e.right)
            if e.op == "/":
                return not has_p(e.right) and affine(e.left)
            return False
        if isinstance(e, MatMulExpression):
            if has_p(e.left) and has_p(e.right):
                return False
            return affine(e.left) and affine(e.right)
        if isinstance(e, UnaryOp):
            return e.op == "neg" and affine(e.operand)
        if isinstance(e, (SumExpression, IndexExpression)):
            return affine(e.operand if isinstance(e, SumExpression) else e.base)
        if isinstance(e, SumOverExpression):
            return all(affine(t) for t in e.terms)
        return False

    return affine(expr)


def refuse_non_affine(expr, param_names: set[str], *, kind: str) -> None:
    """Raise ``NotImplementedError`` unless *expr* is affine in the parameters."""
    if not is_affine_in_params(expr, param_names):
        raise NotImplementedError(
            f"The {kind} robust counterpart dualizes the coefficient of each "
            "uncertain parameter, which is exact only for a row affine in the "
            f"parameter; {expr!r} is not (the parameter enters through a "
            "nonlinear node or a product of uncertain terms). Its finite-difference "
            "coefficient is a secant, and the counterpart would certify non-robust "
            "points. Rewrite the row to be affine in the uncertain parameter, or "
            "use a box set on a row monotone in it. Refusing (#1841)."
        )


def substitute_variables(expr, value_of):
    """Replace every :class:`Variable` ``v`` by ``Constant(value_of(v))``."""
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        SumExpression,
        SumOverExpression,
        UnaryOp,
        Variable,
    )

    def sub(e):
        if isinstance(e, Variable):
            return Constant(np.asarray(value_of(e), dtype=np.float64))
        if isinstance(e, BinaryOp):
            return BinaryOp(e.op, sub(e.left), sub(e.right))
        if isinstance(e, UnaryOp):
            return UnaryOp(e.op, sub(e.operand))
        if isinstance(e, MatMulExpression):
            return MatMulExpression(sub(e.left), sub(e.right))
        if isinstance(e, FunctionCall):
            return FunctionCall(e.func_name, *[sub(a) for a in e.args])
        if isinstance(e, IndexExpression):
            return IndexExpression(sub(e.base), e.index)
        if isinstance(e, SumExpression):
            return SumExpression(sub(e.operand), e.axis)
        if isinstance(e, SumOverExpression):
            return SumOverExpression([sub(t) for t in e.terms])
        return e

    return sub(expr)


def _contains_uncertain_param(expr, param_names: set[str]) -> bool:
    """Check whether *expr* contains any Parameter whose name is in *param_names*."""
    from discopt.modeling.core import (
        BinaryOp,
        CustomCall,
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
    if isinstance(expr, CustomCall):
        # An opaque callable over an uncertain parameter is not robustified by
        # any formulation; missing it here let the post-check pass a parameter
        # left at its nominal value (#1836 audit).
        return any(_contains_uncertain_param(a, param_names) for a in expr.args)
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
