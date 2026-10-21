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

A *certain* :class:`Parameter` (or any variable-free expression of one) that
multiplies an uncertain parameter decides a worst-case direction from its
``formulate()``-time value, so the counterpart is only valid while that value
(or, for some uses, its sign) holds. Nothing is frozen silently (#1839 review):

* A multiplier whose entries share one sign stays *live* in the counterpart --
  a later update of its value is honoured -- and, when it is a bare Parameter,
  a **sign commitment** is recorded: the counterpart is valid for every value
  with that sign.
* Where the value itself is baked into the counterpart -- the ``c+ + c-`` split
  of a mixed-sign multiplier, the aggregate coefficient of
  :func:`aggregate_affine_counterpart`, a polyhedral support coefficient, or a
  compound multiplier ``f(q)`` -- a **value commitment** is recorded.

Commitments live on the model (``model._ro_commitments``, see
:func:`check_ro_commitments`), travel with copies and pickles (the
:mod:`discopt.serialize` format has no key for them, so ``dumps`` refuses such
a model), and are enforced twice: a later
:class:`~discopt.ro.RobustCounterpart` whose uncertainty set covers a committed
Parameter refuses unless the whole set satisfies a sign commitment, and
``Model.solve`` refuses when a committed Parameter's current value breaks its
commitment. Pass every uncertainty set to one ``RobustCounterpart``, or rebuild
the model, instead of updating such a parameter.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

import numpy as np

# Commitments recorded while one ``RobustCounterpart.formulate()`` builds its
# counterpart (module docstring). ``None`` outside that scope: an evaluation
# made anywhere else commits nothing.
_COMMITMENT_SINK: ContextVar[Optional[dict]] = ContextVar("_ro_commitment_sink", default=None)


@contextmanager
def recording_commitments():
    """Collect the commitments made while the body runs into a fresh dict."""
    sink: dict = {}
    token = _COMMITMENT_SINK.set(sink)
    try:
        yield sink
    finally:
        _COMMITMENT_SINK.reset(token)


def merge_commitment(store: dict, name: str, entry: dict) -> None:
    """Add *entry* for Parameter *name* to *store*; a value commitment dominates.

    Two sign commitments of opposite sign cannot both hold except at zero, so
    they are merged into a value commitment rather than either being dropped.
    """
    old = store.get(name)
    if old is None or entry["kind"] == "value":
        store[name] = entry
    elif old["kind"] == "sign" and old["sign"] != entry["sign"]:
        store[name] = {"kind": "value", "value": entry["value"]}


def _commit(param, kind: str, sign: int = 1) -> None:
    sink = _COMMITMENT_SINK.get()
    if sink is None:
        return
    entry: dict = {"kind": kind, "value": np.asarray(param.value, dtype=np.float64).tolist()}
    if kind == "sign":
        entry["sign"] = int(sign)
    merge_commitment(sink, param.name, entry)


def _commit_values_in(expr) -> None:
    """Record a value commitment for every Parameter that *expr* reads."""
    eval_constant_expr(expr, record=True)


def check_ro_commitments(model, *, context: str) -> None:
    """Refuse a solve whose committed Parameters no longer satisfy their commitments.

    A no-op on a model no robust counterpart has committed. ``ValueError``
    names the Parameter, the commitment and the offending value.
    """
    commitments = getattr(model, "_ro_commitments", None)
    if not commitments:
        return
    by_name = {p.name: p for p in getattr(model, "_parameters", [])}
    for name, entry in commitments.items():
        param = by_name.get(name)
        if param is None:
            raise ValueError(
                f"{context}: the robust counterpart was built from the value of "
                f"Parameter {name!r}, which the model no longer holds; refusing to "
                "solve a counterpart whose validity cannot be checked."
            )
        current = np.asarray(param.value, dtype=np.float64)
        recorded = np.asarray(entry["value"], dtype=np.float64)
        if entry["kind"] == "value":
            if current.shape == recorded.shape and np.array_equal(current, recorded):
                continue
            raise ValueError(
                f"{context}: Parameter {name!r} was changed after RobustCounterpart."
                f"formulate() baked its value {recorded.tolist()!r} into the robust "
                f"counterpart; it is now {current.tolist()!r}. The counterpart is not "
                "valid for the new value. Rebuild the model and re-run formulate() "
                "with the new value (or, for a second uncertain parameter, pass all "
                "uncertainty sets to one RobustCounterpart)."
            )
        sign = int(entry["sign"])
        if current.shape == recorded.shape and np.all(sign * current >= 0):
            continue
        raise ValueError(
            f"{context}: Parameter {name!r} multiplies an uncertain parameter, and "
            "RobustCounterpart.formulate() chose the worst case from its sign "
            f"({'non-negative' if sign > 0 else 'non-positive'}); it is now "
            f"{current.tolist()!r}, which breaks that sign, so the counterpart is no "
            "longer robust. Rebuild the model and re-run formulate()."
        )


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
    from discopt.modeling.core import BinaryOp, Constant, Parameter

    value = np.asarray(eval_constant_expr(factor, record=False), dtype=np.float64)
    has_pos = bool(np.any(value > 0))
    has_neg = bool(np.any(value < 0))

    if reciprocal and np.any(value == 0):
        raise ValueError(
            "Robust counterpart: a constant divisor of an uncertain term has "
            "zero entries; the division is undefined."
        )

    if not (has_pos and has_neg):
        factor_sign = -1 if has_neg else 1
        new_other = sign_tracking_substitute(
            other, wc_lower, wc_upper, param_names, maximize, sign * factor_sign
        )
        # The factor stays live, so a later update of a certain Parameter is
        # honoured; the counterpart is valid only while its sign holds, so that
        # is committed (#1839 review). A compound factor ``f(q)`` has no
        # per-Parameter sign to commit and commits the values it reads.
        if isinstance(factor, Parameter):
            _commit(factor, "sign", factor_sign)
        elif not isinstance(factor, Constant):
            _commit_values_in(factor)
        return rebuild(factor, new_other)

    # The split below bakes the value of a certain Parameter into c+ and c-.
    _commit_values_in(factor)

    if reciprocal:
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


class _NotAffine(Exception):
    """Internal: the expression is not affine in the uncertain parameters with
    constant coefficients (or uses a node the extractor does not model)."""


def _value_shape(expr) -> tuple[int, ...]:
    """Static shape of *expr* (:func:`_shape_of`), or :class:`_NotAffine`."""
    shape = _shape_of(expr)
    if shape is None:
        raise _NotAffine(f"shape of {type(expr).__name__} is not statically known")
    return shape


def affine_param_jacobians(expr, param_names: set[str], *, factors: Optional[list] = None):
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

    When *factors* is a list, each constant-valued coefficient expression read
    is appended to it, so a caller that bakes ``J`` can commit the Parameters
    those expressions read. The extraction itself commits nothing: it runs on
    every row, most of which keep sign tracking and their live multipliers.
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
        if factors is not None:
            factors.append(factor)
        return np.asarray(eval_constant_expr(factor, record=False), dtype=np.float64)

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

    factors: list = []
    jacs = affine_param_jacobians(expr, param_names, factors=factors)
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
    # The penalty below bakes J, i.e. the values of these coefficients.
    for f in factors:
        _commit_values_in(f)
    result = freeze_constant_multipliers(expr, param_names)
    for name in jacs:
        result = substitute_param(result, name, Constant(unc_sets[name].parameter.value))
    for name, (j, _s) in jacs.items():
        delta = np.asarray(unc_sets[name].delta, dtype=np.float64)
        k = delta.ndim
        pen = np.sum(np.abs(j) * delta, axis=tuple(range(j.ndim - k, j.ndim)))
        result = BinaryOp("+" if maximize else "-", result, Constant(np.asarray(pen)))
    return result


def freeze_constant_multipliers(expr, param_names: set[str]):
    """Replace each non-Constant constant-valued multiplier of an uncertain term
    by its current value, committing that value (see the module docstring on
    certain Parameters)."""
    from discopt.modeling.core import (
        BinaryOp,
        Constant,
        FunctionCall,
        IndexExpression,
        MatMulExpression,
        SumExpression,
        SumOverExpression,
        UnaryOp,
    )

    def fr(e):
        return freeze_constant_multipliers(e, param_names)

    def fix(factor, other):
        if isinstance(factor, Constant) or not _needs_multiplier_sign(factor, other, param_names):
            return fr(factor)
        return Constant(np.asarray(eval_constant_expr(factor), dtype=np.float64))

    if isinstance(expr, BinaryOp):
        if expr.op in ("*", "/"):
            left = fix(expr.left, expr.right) if expr.op == "*" else fr(expr.left)
            return BinaryOp(expr.op, left, fix(expr.right, expr.left))
        return BinaryOp(expr.op, fr(expr.left), fr(expr.right))
    if isinstance(expr, MatMulExpression):
        return MatMulExpression(fix(expr.left, expr.right), fix(expr.right, expr.left))
    if isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, fr(expr.operand))
    if isinstance(expr, FunctionCall):
        return FunctionCall(expr.func_name, *[fr(a) for a in expr.args])
    if isinstance(expr, IndexExpression):
        return IndexExpression(fr(expr.base), expr.index)
    if isinstance(expr, SumExpression):
        return SumExpression(fr(expr.operand), expr.axis)
    if isinstance(expr, SumOverExpression):
        return SumOverExpression([fr(t) for t in expr.terms])
    return expr


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


def eval_constant_expr(expr, *, record: bool = True) -> np.ndarray:
    """Evaluate a variable-free expression to its full numpy value.

    Every node is evaluated with numpy semantics (broadcasting, ``axis``,
    indexing) so a vector constant is never collapsed to a scalar. A node this
    evaluator does not know raises ``NotImplementedError`` rather than
    contributing a guessed value -- the former polyhedral evaluator returned
    ``float(np.sum(value))`` for a vector constant and ``0.0`` for any node it
    did not recognise, so ``sum(c * p)`` got a zero coefficient (#1831 audit).

    With ``record`` (the default) every Parameter read records a value
    commitment while a ``RobustCounterpart.formulate()`` is building (module
    docstring): a caller that bakes the result into the counterpart must not be
    able to forget it. A caller that only reads a sign, or keeps the
    expression live, passes ``record=False`` and commits what it relies on
    itself.
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

    def ev(e):
        return eval_constant_expr(e, record=record)

    if isinstance(expr, Parameter) and record:
        _commit(expr, "value")
    if isinstance(expr, (Constant, Parameter)):
        return np.asarray(expr.value, dtype=np.float64)
    if isinstance(expr, BinaryOp):
        lv = ev(expr.left)
        rv = ev(expr.right)
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
        v = ev(expr.operand)
        if expr.op == "neg":
            return np.asarray(-v)
        if expr.op == "abs":
            return np.asarray(np.abs(v))
    elif isinstance(expr, MatMulExpression):
        return np.asarray(ev(expr.left) @ ev(expr.right))
    elif isinstance(expr, SumExpression):
        return np.asarray(np.sum(ev(expr.operand), axis=expr.axis))
    elif isinstance(expr, SumOverExpression):
        total = np.asarray(0.0)
        for t in expr.terms:
            total = total + ev(t)
        return np.asarray(total)
    elif isinstance(expr, IndexExpression):
        return np.asarray(ev(expr.base)[expr.index])
    elif isinstance(expr, FunctionCall) and expr.func_name in _NUMPY_FUNCS:
        if len(expr.args) == 1:
            return np.asarray(_NUMPY_FUNCS[expr.func_name](ev(expr.args[0])))
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
