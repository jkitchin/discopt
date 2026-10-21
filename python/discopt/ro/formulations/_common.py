"""Shared sign-tracking worst-case substitution for robust reformulations.

Both box and polyhedral formulations need to traverse the expression DAG
while tracking the effective sign of each uncertain parameter to determine
whether the upper or lower bound is the worst case.  This module provides
that traversal as a single reusable function.

The sign starts at +1 at the tree root and flips through ``-`` (subtraction)
and unary negation nodes.  Multiplication by a negative constant also flips
the sign.

A sign may pass only through a node proven monotone in its uncertain operand
(#1836): ``+ - neg``, sums and indexing, products, quotients and matmuls with a
side of known sign, odd positive integer powers and the monotone intrinsics in
``_INCREASING_FUNCS`` / ``_DECREASING_FUNCS``. Below any other node (``1/p``,
``p**2``, ``abs(p)``, ``cos(p)``, ``p*p``) a variable-free sub-expression is
replaced by the matching end of a sound interval enclosure over the box, and
one that mentions a decision variable is refused with ``NotImplementedError``
-- passing the sign through those nodes certified non-robust points.

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
"""

from __future__ import annotations

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
            left_has_param = _contains_uncertain_param(expr.left, param_names)
            right_has_param = _contains_uncertain_param(expr.right, param_names)
            if left_has_param and right_has_param:
                # ``p * p``, ``p * exp(p)``: not monotone in either occurrence
                # with a known direction, so no per-occurrence substitution is
                # a worst case (#1836; this used to warn and substitute anyway).
                return _interval_worst_case(expr, wc_lower, wc_upper, param_names, maximize, sign)
            if not (left_has_param or right_has_param):
                return expr
            # One side carries the uncertainty, the other is a decision-variable
            # expression (a constant-valued one was handled above). The
            # multiplier's sign decides the worst case, so it must be provably
            # sign-definite over the variable box.
            factor, other = (expr.right, expr.left) if left_has_param else (expr.left, expr.right)
            factor_sign = _definite_sign(factor, strict=False)
            if factor_sign is None:
                raise NotImplementedError(
                    "Robust counterpart (sign tracking): an uncertain term is "
                    f"multiplied by {factor!r}, whose sign is not fixed over the "
                    "variable bounds, so the worst-case direction of the uncertain "
                    "parameter is unknown. Refusing rather than guessing (#1836)."
                )
            new_other = sign_tracking_substitute(
                other, wc_lower, wc_upper, param_names, maximize, sign * factor_sign
            )
            if left_has_param:
                return BinaryOp("*", new_other, factor)
            return BinaryOp("*", factor, new_other)
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
        elif expr.op == "/":
            if _contains_uncertain_param(expr.right, param_names):
                # ``1 / p`` is decreasing on each side of 0: the sign does not
                # pass through a divisor (#1836).
                return _interval_worst_case(expr, wc_lower, wc_upper, param_names, maximize, sign)
            if not _contains_uncertain_param(expr.left, param_names):
                return expr
            # ``e / d`` with a variable divisor: monotone in ``e`` only when
            # ``d`` keeps one strict sign over the variable box.
            divisor_sign = _definite_sign(expr.right, strict=True)
            if divisor_sign is None:
                raise NotImplementedError(
                    "Robust counterpart (sign tracking): an uncertain term is "
                    f"divided by {expr.right!r}, whose sign is not fixed (or which "
                    "can be 0) over the variable bounds, so the worst-case "
                    "direction of the uncertain parameter is unknown. Refusing "
                    "rather than guessing (#1836)."
                )
            nl = sign_tracking_substitute(
                expr.left, wc_lower, wc_upper, param_names, maximize, sign * divisor_sign
            )
            nr = expr.right
        elif expr.op == "**":
            if not _contains_uncertain_param(expr, param_names):
                return expr
            if not _contains_uncertain_param(expr.right, param_names) and _is_increasing_power(
                expr.right
            ):
                nl = sign_tracking_substitute(
                    expr.left, wc_lower, wc_upper, param_names, maximize, sign
                )
                nr = expr.right
            else:
                # ``p**2``, ``p**-1``, ``2**p`` with a base of unknown sign: not
                # monotone with a known direction (#1836).
                return _interval_worst_case(expr, wc_lower, wc_upper, param_names, maximize, sign)
        else:
            if not _contains_uncertain_param(expr, param_names):
                return expr
            return _interval_worst_case(expr, wc_lower, wc_upper, param_names, maximize, sign)
        return BinaryOp(expr.op, nl, nr)

    if isinstance(expr, UnaryOp):
        if expr.op != "neg":
            # ``abs(p)`` is not monotone: passing the sign through substituted
            # the bound of largest ``p``, not of largest ``|p|`` (#1836).
            if not _contains_uncertain_param(expr, param_names):
                return expr
            return _interval_worst_case(expr, wc_lower, wc_upper, param_names, maximize, sign)
        return UnaryOp(
            expr.op,
            sign_tracking_substitute(
                expr.operand, wc_lower, wc_upper, param_names, maximize, -sign
            ),
        )

    if isinstance(expr, FunctionCall):
        if not _contains_uncertain_param(expr, param_names):
            return expr
        if len(expr.args) == 1 and expr.func_name in _INCREASING_FUNCS:
            fn_sign = sign
        elif len(expr.args) == 1 and expr.func_name in _DECREASING_FUNCS:
            fn_sign = -sign
        else:
            # ``cos(p)``, ``p**2`` spelled as a function, ``max(p, q)``: no
            # known monotone direction (#1836).
            return _interval_worst_case(expr, wc_lower, wc_upper, param_names, maximize, sign)
        return FunctionCall(
            expr.func_name,
            sign_tracking_substitute(
                expr.args[0], wc_lower, wc_upper, param_names, maximize, fn_sign
            ),
        )

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
        left_has_param = _contains_uncertain_param(expr.left, param_names)
        right_has_param = _contains_uncertain_param(expr.right, param_names)
        if left_has_param and right_has_param:
            return _interval_worst_case(expr, wc_lower, wc_upper, param_names, maximize, sign)
        if not (left_has_param or right_has_param):
            return expr
        # ``p @ x`` / ``x @ p`` with a variable side: as for ``*``, the variable
        # entries must share one provable sign (#1836).
        factor = expr.right if left_has_param else expr.left
        factor_sign = _definite_sign(factor, strict=False)
        if factor_sign is None:
            raise NotImplementedError(
                "Robust counterpart (sign tracking): an uncertain term is "
                f"matrix-multiplied by {factor!r}, whose entries do not share a "
                "sign fixed over the variable bounds, so the worst-case direction "
                "of the uncertain parameter is unknown. Refusing rather than "
                "guessing (#1836)."
            )
        if left_has_param:
            nl = sign_tracking_substitute(
                expr.left, wc_lower, wc_upper, param_names, maximize, sign * factor_sign
            )
            return MatMulExpression(nl, expr.right)
        nr = sign_tracking_substitute(
            expr.right, wc_lower, wc_upper, param_names, maximize, sign * factor_sign
        )
        return MatMulExpression(expr.left, nr)

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


# Single-argument intrinsics that are monotone over their whole domain, so the
# worst case of ``f(e)`` is ``f`` of the worst case of ``e`` in the same
# (increasing) or the opposite (decreasing) direction.
_INCREASING_FUNCS = frozenset(
    {
        "exp",
        "log",
        "log2",
        "log10",
        "log1p",
        "sqrt",
        "atan",
        "asin",
        "sinh",
        "asinh",
        "acosh",
        "atanh",
        "tanh",
        "erf",
        "sigmoid",
        "softplus",
    }
)
_DECREASING_FUNCS = frozenset({"acos"})


def _is_increasing_power(exponent) -> bool:
    """True when ``b ** exponent`` is increasing in ``b`` over the whole real line.

    That is a constant scalar exponent equal to 1 or a positive odd integer
    (``b**3``). Every other exponent -- even (``b**2``), negative (``b**-1``),
    fractional -- is non-monotone or decreasing on part of the line.
    """
    from discopt.ro.formulations.box import _contains_variable

    if _contains_variable(exponent) or _contains_any_parameter(exponent):
        return False
    k = np.asarray(eval_constant_expr(exponent), dtype=np.float64)
    if k.size != 1:
        return False
    kv = float(k.reshape(()))
    return kv > 0 and kv == int(kv) and int(kv) % 2 == 1


def _definite_sign(expr, *, strict: bool) -> Optional[int]:
    """+1 / -1 when *expr* (free of uncertain parameters) keeps one sign over the
    declared variable bounds, else ``None``. ``strict`` excludes 0.

    Decided by a sound interval enclosure, so a ``None`` may be conservative but
    a sign is never wrong.
    """
    from discopt._relax.convexity.interval_eval import evaluate_interval

    enc = evaluate_interval(expr, None)
    lo = np.asarray(enc.lo, dtype=np.float64)
    hi = np.asarray(enc.hi, dtype=np.float64)
    if strict:
        found = 1 if np.all(lo > 0) else -1 if np.all(hi < 0) else None
    else:
        found = 1 if np.all(lo >= 0) else -1 if np.all(hi <= 0) else None
    if found is not None and _contains_any_parameter(expr):
        # A certain Parameter's value is part of the sign, and it stays symbolic
        # in the counterpart, so the sign is a recorded assumption. (Reliance on
        # the variable bounds alone is the pre-existing RO-1 contract: bounds
        # are read at ``formulate()`` time, unguarded, as on main.)
        record_sign_assumption(expr, np.array(found), strict=strict, label="sign-definite factor")
    return found


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


def _interval_worst_case(expr, wc_lower, wc_upper, param_names: set[str], maximize, sign):
    """Worst case of a variable-free, non-monotone uncertain sub-expression.

    Sign tracking may pass a direction only through a node that is monotone in
    its uncertain operand (#1836). Below any other node -- ``1/p``, ``p**2``,
    ``abs(p)``, ``cos(p)``, ``p*p`` -- the worst case is replaced by the matching
    end of a sound interval enclosure of the whole sub-expression over the
    uncertainty box: exact for a single occurrence of a scalar parameter under a
    monotone-by-pieces node, conservative (never optimistic) otherwise. The
    enclosing path is monotone (everything above was sign-tracked), so the
    result is a valid worst case of the row.

    A sub-expression that also mentions a decision variable has no constant
    worst case, and is refused.
    """
    from discopt._relax.convexity.interval import Interval
    from discopt._relax.convexity.interval_eval import evaluate_interval, param_box_key
    from discopt.modeling.core import Constant
    from discopt.ro.formulations.box import _contains_variable

    if _contains_variable(expr):
        raise NotImplementedError(
            "Robust counterpart (sign tracking): the uncertain parameter enters "
            f"{expr!r} through a node that is not monotone (or not monotone in a "
            "known direction) together with a decision variable, so no worst-case "
            "substitution is valid. Rewrite the row to be monotone in the uncertain "
            "parameter. Refusing rather than certifying a non-robust point (#1836)."
        )
    if _contains_any_parameter(expr, exclude=param_names):
        # Evaluating would freeze the certain Parameter at its value, and a later
        # pass over it would robustify nothing (#1839 review).
        raise NotImplementedError(
            "Robust counterpart (sign tracking): a certain Parameter sits under a "
            f"node that is not monotone in the uncertain parameter, in {expr!r}. "
            "Its worst case cannot be kept symbolic. Robustify every uncertain "
            "parameter in one RobustCounterpart (or make the Parameter a constant)."
        )
    box = {
        param_box_key(n): Interval(
            np.asarray(wc_lower[n], dtype=np.float64), np.asarray(wc_upper[n], dtype=np.float64)
        )
        for n in param_names
    }
    enc = evaluate_interval(expr, None, box)
    use_upper = maximize ^ (sign < 0)
    value = np.asarray(enc.hi if use_upper else enc.lo, dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise NotImplementedError(
            f"Robust counterpart (sign tracking): {expr!r} has no finite worst case "
            "over the uncertainty set (an unbounded or unsupported enclosure, e.g. a "
            "divisor that can be 0). Refusing rather than substituting a guess (#1836)."
        )
    return Constant(value)


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
