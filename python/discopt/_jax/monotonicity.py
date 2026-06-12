"""Per-expression monotonicity detection for expression DAGs.

Classifies a scalar expression by how it responds to *jointly* increasing the
decision variables over the model's box:

* ``NONDECREASING`` -- ``∂f/∂xᵢ ≥ 0`` for every variable ``xᵢ`` over the box.
* ``NONINCREASING`` -- ``∂f/∂xᵢ ≤ 0`` for every variable over the box.
* ``CONSTANT``      -- the expression does not depend on any variable.
* ``UNKNOWN``       -- neither could be proven (mixed signs, an unsupported
  atom, or a domain issue forcing a sound abstention).

This mirrors SUSPECT's ``Monotonicity`` verdict (Ceccon, Siirola, Misener,
2020) and is the monotonicity counterpart to :mod:`discopt._jax.convexity`.

Soundness invariant
-------------------
A ``NONDECREASING`` / ``NONINCREASING`` verdict is a *proof*. It is obtained
from an interval enclosure of the gradient via forward-mode interval automatic
differentiation: if the interval enclosing ``∇f`` over the whole box lies in
the nonnegative (resp. nonpositive) orthant, then the true gradient has that
sign everywhere, so the monotonicity holds. Anything that cannot be proven --
an atom without a derivative rule here, a non-finite enclosure (e.g. ``tan``
across an asymptote), a ``log`` of a possibly-nonpositive argument -- abstains
to ``UNKNOWN`` rather than guessing.

Scope
-----
Scalar (size-1) variables only; array-valued variables and constructs the
evaluator does not recognise abstain to ``UNKNOWN``. The public surface is
intentionally small::

    classify_monotonicity(expr, model=None, box=None) -> Monotonicity
    Monotonicity  (enum)
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional, cast

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Expression,
    FunctionCall,
    UnaryOp,
    Variable,
)

from .convexity import interval as iv
from .convexity.interval import Interval


class Monotonicity(Enum):
    """Monotonicity of a scalar expression over the model box."""

    NONDECREASING = "nondecreasing"
    NONINCREASING = "nonincreasing"
    CONSTANT = "constant"
    UNKNOWN = "unknown"


# A node's forward-mode payload: an interval enclosing its value, and a sparse
# gradient mapping each variable it depends on to an interval enclosing the
# corresponding partial derivative over the box. Absent keys are exact zeros.
class _Abstain(Exception):
    """Raised internally to force a sound UNKNOWN verdict."""


_LN2 = math.log(2.0)
_LN10 = math.log(10.0)


def classify_monotonicity(
    expr: Expression,
    model: Optional[object] = None,
    box: Optional[dict] = None,
) -> Monotonicity:
    """Classify the monotonicity of a scalar expression over its box.

    Args:
        expr: A scalar expression from :mod:`discopt.modeling.core`.
        model: Unused; accepted for signature symmetry with
            :func:`discopt._jax.convexity.classify_expr`. Bounds are read from
            the variables themselves (or ``box``).
        box: Optional ``{Variable: Interval}`` overriding declared bounds.

    Returns:
        A :class:`Monotonicity` verdict. ``NONDECREASING`` / ``NONINCREASING``
        are proofs; ``UNKNOWN`` is a sound abstention.
    """
    box = box or {}
    try:
        _value, grad = _walk(expr, box, {})
    except _Abstain:
        return Monotonicity.UNKNOWN

    if not grad:
        return Monotonicity.CONSTANT

    partials = list(grad.values())
    if all(_is_zero(p) for p in partials):
        return Monotonicity.CONSTANT
    if all(_finite(p) and bool(np.all(np.asarray(p.lo) >= 0.0)) for p in partials):
        return Monotonicity.NONDECREASING
    if all(_finite(p) and bool(np.all(np.asarray(p.hi) <= 0.0)) for p in partials):
        return Monotonicity.NONINCREASING
    return Monotonicity.UNKNOWN


def _is_zero(p: Interval) -> bool:
    return bool(np.all(np.asarray(p.lo) == 0.0)) and bool(np.all(np.asarray(p.hi) == 0.0))


def _finite(p: Interval) -> bool:
    return bool(np.all(np.isfinite(np.asarray(p.lo)))) and bool(
        np.all(np.isfinite(np.asarray(p.hi)))
    )


def _scalar(x) -> float:
    """First element of a (possibly shape-``(1,)``) interval endpoint as a float."""
    return float(np.asarray(x).ravel()[0])


def _scalar_box(v: Variable, box: dict) -> Interval:
    if v in box:
        return cast(Interval, box[v])
    if v.size != 1:
        raise _Abstain  # array-valued variables are out of scope
    lb = float(np.asarray(v.lb).ravel()[0])
    ub = float(np.asarray(v.ub).ravel()[0])
    return Interval.from_bounds(lb, ub)


def _walk(node: Expression, box: dict, cache: dict) -> tuple[Interval, dict]:
    """Return ``(value_interval, {Variable: partial_interval})`` for ``node``."""
    key = id(node)
    if key in cache:
        return cast("tuple[Interval, dict]", cache[key])
    out = _impl(node, box, cache)
    cache[key] = out
    return out


def _impl(node: Expression, box: dict, cache: dict) -> tuple[Interval, dict]:
    if isinstance(node, (int, float)):
        return Interval.point(float(node)), {}
    if isinstance(node, Constant):
        val = np.asarray(node.value, dtype=np.float64)
        if val.ndim != 0:
            raise _Abstain
        return Interval.point(float(val)), {}
    if isinstance(node, Variable):
        return _scalar_box(node, box), {node: Interval.point(1.0)}
    if isinstance(node, BinaryOp):
        return _binary(node, box, cache)
    if isinstance(node, UnaryOp):
        return _unary(node, box, cache)
    if isinstance(node, FunctionCall):
        return _function(node, box, cache)
    raise _Abstain  # IndexExpression, MatMul, SumOver, CustomCall, ... -> abstain


# ----------------------------------------------------------------------
# Gradient combinators (interval product / quotient rules)
# ----------------------------------------------------------------------


def _vars(*grads: dict) -> set:
    out: set = set()
    for g in grads:
        out.update(g.keys())
    return out


def _get(g: dict, v) -> Interval:
    return cast(Interval, g.get(v, Interval.point(0.0)))


def _binary(node: BinaryOp, box: dict, cache: dict) -> tuple[Interval, dict]:
    lo_val, lo_g = _walk(node.left, box, cache)
    ro_val, ro_g = _walk(node.right, box, cache)
    op = node.op

    if op == "+":
        return lo_val + ro_val, {v: _get(lo_g, v) + _get(ro_g, v) for v in _vars(lo_g, ro_g)}
    if op == "-":
        return lo_val - ro_val, {v: _get(lo_g, v) - _get(ro_g, v) for v in _vars(lo_g, ro_g)}
    if op == "*":
        # (uv)' = u' v + u v'
        grad = {v: _get(lo_g, v) * ro_val + lo_val * _get(ro_g, v) for v in _vars(lo_g, ro_g)}
        return lo_val * ro_val, grad
    if op == "/":
        if bool(np.asarray(ro_val.contains_zero())):
            raise _Abstain
        # (u/v)' = (u' v - u v') / v^2
        denom = ro_val * ro_val
        grad = {
            v: (_get(lo_g, v) * ro_val - lo_val * _get(ro_g, v)) / denom for v in _vars(lo_g, ro_g)
        }
        return lo_val / ro_val, grad
    if op == "**":
        return _power(lo_val, lo_g, node.right)
    raise _Abstain


def _power(base_val: Interval, base_g: dict, exponent: Expression) -> tuple[Interval, dict]:
    p = _as_constant(exponent)
    if p is None:
        raise _Abstain  # variable exponent -> abstain
    value = _pow_value(base_val, p)
    if not base_g:
        return value, {}
    # f = u^p ;  f' = p * u^(p-1) * u'
    deriv_factor = Interval.point(float(p)) * _pow_value(base_val, p - 1.0)
    grad = {v: deriv_factor * g for v, g in base_g.items()}
    return value, grad


def _pow_value(u: Interval, q: float) -> Interval:
    """Interval enclosure of ``u**q`` for a constant exponent ``q``."""
    if float(q).is_integer():
        return u ** int(q)
    # Fractional power needs a positive base; use exp(q log u).
    if bool(np.any(np.asarray(u.lo) <= 0.0)):
        raise _Abstain
    return iv.exp(Interval.point(float(q)) * iv.log(u))


def _as_constant(node: Expression) -> Optional[float]:
    if isinstance(node, (int, float)):
        return float(node)
    if isinstance(node, Constant):
        val = np.asarray(node.value)
        if val.ndim == 0:
            return float(val)
    return None


# ----------------------------------------------------------------------
# Unary ops and named functions: value + derivative-factor per atom
# ----------------------------------------------------------------------


def _unary(node: UnaryOp, box: dict, cache: dict) -> tuple[Interval, dict]:
    val, g = _walk(node.operand, box, cache)
    if node.op == "neg":
        return -val, {v: -gi for v, gi in g.items()}
    if node.op == "abs":
        factor = _abs_deriv(val)
        return iv.absolute(val), {v: factor * gi for v, gi in g.items()}
    raise _Abstain


def _abs_deriv(u: Interval) -> Interval:
    lo = _scalar(u.lo)
    hi = _scalar(u.hi)
    if lo >= 0.0:
        return Interval.point(1.0)
    if hi <= 0.0:
        return Interval.point(-1.0)
    return Interval.from_bounds(-1.0, 1.0)


def _function(node: FunctionCall, box: dict, cache: dict) -> tuple[Interval, dict]:
    if len(node.args) != 1:
        raise _Abstain
    u_val, u_g = _walk(node.args[0], box, cache)
    name = node.func_name
    value = _func_value(name, u_val)
    factor = _func_deriv(name, u_val)
    if not _finite(factor):
        raise _Abstain
    return value, {v: factor * gi for v, gi in u_g.items()}


def _func_value(name: str, u: Interval) -> Interval:
    if name == "exp":
        return iv.exp(u)
    if name == "log":
        return _log_or_abstain(u)
    if name == "log2":
        return _log_or_abstain(u) * Interval.point(1.0 / _LN2)
    if name == "log10":
        return _log_or_abstain(u) * Interval.point(1.0 / _LN10)
    if name == "sqrt":
        if bool(np.any(np.asarray(u.lo) < 0.0)):
            raise _Abstain
        return iv.sqrt(u)
    if name == "sin":
        return iv.sin(u)
    if name == "cos":
        return iv.cos(u)
    if name == "tan":
        return iv.tan(u)
    if name == "tanh":
        return iv.tanh(u)
    if name == "sinh":
        return iv.sinh(u)
    if name == "cosh":
        return iv.cosh(u)
    if name in ("asin", "acos", "atan"):
        return _inverse_trig_value(name, u)
    raise _Abstain


def _log_or_abstain(u: Interval) -> Interval:
    if bool(np.any(np.asarray(u.lo) <= 0.0)):
        raise _Abstain
    return iv.log(u)


def _inverse_trig_value(name: str, u: Interval) -> Interval:
    """Value enclosure for asin/acos/atan (all monotone on their domains)."""
    lo = _scalar(u.lo)
    hi = _scalar(u.hi)
    if name in ("asin", "acos") and (lo < -1.0 or hi > 1.0):
        raise _Abstain
    if name == "asin":  # increasing
        return Interval.from_bounds(float(np.arcsin(lo)), float(np.arcsin(hi)))
    if name == "acos":  # decreasing
        return Interval.from_bounds(float(np.arccos(hi)), float(np.arccos(lo)))
    return Interval.from_bounds(float(np.arctan(lo)), float(np.arctan(hi)))  # atan, increasing


def _func_deriv(name: str, u: Interval) -> Interval:
    """Interval enclosure of ``g'(u)`` for a named unary atom ``g``."""
    one = Interval.point(1.0)
    if name == "exp":
        return iv.exp(u)
    if name == "log":
        return one / _positive(u)
    if name == "log2":
        return Interval.point(1.0 / _LN2) / _positive(u)
    if name == "log10":
        return Interval.point(1.0 / _LN10) / _positive(u)
    if name == "sqrt":
        return Interval.point(0.5) / iv.sqrt(_positive(u))
    if name == "sin":
        return iv.cos(u)
    if name == "cos":
        return -iv.sin(u)
    if name == "tan":
        # sec^2 = 1 + tan^2; non-finite (asymptote crossing) -> abstain upstream
        return one + iv.tan(u) ** 2
    if name == "tanh":
        return one - iv.tanh(u) ** 2
    if name == "sinh":
        return iv.cosh(u)
    if name == "cosh":
        return iv.sinh(u)
    if name in ("asin", "acos"):
        # d/du asin = 1/sqrt(1-u^2) ; acos is its negation
        radicand = one - u**2
        if bool(np.any(np.asarray(radicand.lo) <= 0.0)):
            raise _Abstain
        d = one / iv.sqrt(radicand)
        return d if name == "asin" else -d
    if name == "atan":
        return one / (one + u**2)
    raise _Abstain


def _positive(u: Interval) -> Interval:
    if bool(np.any(np.asarray(u.lo) <= 0.0)):
        raise _Abstain
    return u
