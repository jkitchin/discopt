"""Per-expression log-curvature lattice for geometric programming.

This is the log-space analogue of the original-space :class:`Curvature`
lattice in :mod:`discopt._jax.convexity.lattice`, propagated through the
*same* :class:`~discopt.modeling.core.Expression` DAG. Where the x-space
walker (:mod:`discopt._jax.convexity.rules`) answers "is this expression
convex in ``x``?", this walker answers "is this expression **log-convex**
in ``x``?" — i.e. is ``F(u) = log f(exp(u))`` convex/concave/affine after
the geometric-programming change of variables ``u_j = log x_j``.

The two lattices are kept **strictly separate** (a distinct enum, a
distinct walker, no cross-contamination). This is deliberate and a
soundness requirement: a genuine posynomial is log-convex but *not*
convex in ``x`` (its Hessian is indefinite on the positive orthant), so
folding a log-curvature verdict into the x-space verdict would mis-gate
the x-space convex fast path — a soundness break. See
:func:`discopt.gp.is_log_convex` for the same argument at whole-model
scope.

Why a lattice and not the whole-model verdict
----------------------------------------------
:func:`discopt.gp.classify_gp` returns a single verdict for an *entire*
model. This walker instead tags **every sub-expression**, so
log-curvature *composes*: a posynomial buried deep inside a constraint is
tagged ``LOG_CONVEX`` and that verdict propagates upward through products,
ratios, powers and sums. That unlocks reasoning about **partially**
log-convex models — a model that is a GP in some variables and something
else in others — which the whole-model verdict cannot express. This is the
richer form requested in issue #115 (a follow-up to #111); it is additive
and does not touch the shipped whole-model GP path.

The lattice
-----------
For a strictly-positive function ``f`` write ``F(u) = log f(exp(u))``:

* ``LOG_AFFINE``  — ``F`` is affine. Exactly the monomials
  ``c * prod_j x_j**a_j`` (``c > 0``): ``F = log c + sum_j a_j u_j``.
* ``LOG_CONVEX``  — ``F`` is convex. Posynomials (sums of monomials) are
  the canonical example: ``F = log sum_k exp(affine_k)`` (log-sum-exp).
* ``LOG_CONCAVE`` — ``F`` is concave. E.g. a reciprocal of a posynomial.
* ``UNKNOWN``     — no proof; the conservative top element.

Soundness invariant
--------------------
A non-``UNKNOWN`` label is a **proof**, and it certifies two things at
once: the expression is strictly **positive** on the model's declared
box, *and* it has the stated log-curvature. Positivity is carried
implicitly — every leaf rule that emits a non-``UNKNOWN`` label requires a
strictly-positive lower bound (variables) or value (constants), and every
composition rule below preserves positivity — so an operation that cannot
stay in the positive domain (negation, subtraction) abstains to
``UNKNOWN``. When a precondition fails, the walker propagates ``UNKNOWN``
rather than guess.

Composition rules (all sound; references below)
-----------------------------------------------
* **product** ``f*g``: ``log(f g) = log f + log g`` — the sum of two
  log-log-curvatures, so this mirrors :func:`lattice.combine_sum` exactly
  (convex+convex→convex, concave+concave→concave, convex+concave→unknown,
  affine is the identity).
* **ratio** ``f/g``: ``log(f/g) = log f - log g`` — product against the
  *reciprocal* log-curvature of ``g`` (:func:`log_negate`).
* **power** ``f**a`` (constant ``a``): ``log(f**a) = a log f`` — scales the
  log-curvature by ``sign(a)`` (:func:`log_scale_pow`).
* **sum** ``f+g``: log-convexity is preserved under addition (Boyd &
  Vandenberghe §3.5.2), so a sum of ``LOG_AFFINE``/``LOG_CONVEX`` terms is
  ``LOG_CONVEX``; anything else abstains. (Log-*concavity* is **not**
  preserved under addition — hence the asymmetry.)
* **max / min**: ``max`` of log-convex is log-convex; ``min`` of
  log-concave is log-concave (pointwise max/min of convex/concave).

References
----------
Boyd, Kim, Vandenberghe, Hassibi (2007), "A tutorial on geometric
  programming," *Optimization and Engineering* 8(1):67-127.
Boyd, Vandenberghe (2004), *Convex Optimization*, §3.5 (log-concave /
  log-convex functions), §4.5 (geometric programming).
Agrawal, Diamond, Boyd (2019), "Disciplined Geometric Programming,"
  *Optimization Letters* — the DGP composition calculus this mirrors.

Issue #115 (follow-up to #111).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

from discopt.modeling.core import (
    BinaryOp,
    Constant,
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

# Treat a lower bound / value at or below this as non-positive (out of the
# strictly-positive GP domain). Matches the posynomial recogniser's tolerance.
_POS_TOL = 1e-12


class LogCurvature(Enum):
    """Log-space curvature of an expression over the positive orthant.

    For strictly-positive ``f`` and ``F(u) = log f(exp(u))``:

    ``LOG_AFFINE``  — ``F`` affine (a monomial); both log-convex and
        log-concave.
    ``LOG_CONVEX``  — ``F`` convex (a posynomial's log-sum-exp form).
    ``LOG_CONCAVE`` — ``F`` concave.
    ``UNKNOWN``     — no sound verdict; the conservative top element. Also
        the verdict whenever strict positivity could not be proven.

    Deliberately distinct from :class:`discopt._jax.convexity.Curvature`;
    the two lattices must never be conflated (see the module docstring).
    """

    LOG_AFFINE = "log-affine"
    LOG_CONVEX = "log-convex"
    LOG_CONCAVE = "log-concave"
    UNKNOWN = "unknown"


# ──────────────────────────────────────────────────────────────────────
# Lattice operators (the log-space analogues of lattice.py)
# ──────────────────────────────────────────────────────────────────────


def log_negate(c: LogCurvature) -> LogCurvature:
    """Log-curvature of the reciprocal ``1/f`` given that of ``f``.

    ``log(1/f) = -log f``, so convexity and concavity swap while an affine
    log stays affine (and ``UNKNOWN`` stays ``UNKNOWN``).
    """
    if c == LogCurvature.LOG_CONVEX:
        return LogCurvature.LOG_CONCAVE
    if c == LogCurvature.LOG_CONCAVE:
        return LogCurvature.LOG_CONVEX
    return c


def log_combine_product(a: LogCurvature, b: LogCurvature) -> LogCurvature:
    """Log-curvature of a product ``f*g``.

    ``log(f g) = log f + log g`` is the *sum* of the two log-log
    curvatures, so this is the log-space twin of
    :func:`lattice.combine_sum`: ``LOG_AFFINE`` is the identity, like
    curvatures reinforce, and convex-against-concave collapses to
    ``UNKNOWN``.
    """
    if a == LogCurvature.UNKNOWN or b == LogCurvature.UNKNOWN:
        return LogCurvature.UNKNOWN
    if a == LogCurvature.LOG_AFFINE:
        return b
    if b == LogCurvature.LOG_AFFINE:
        return a
    if a == b:
        return a
    return LogCurvature.UNKNOWN


def log_combine_sum(a: LogCurvature, b: LogCurvature) -> LogCurvature:
    """Log-curvature of a sum ``f+g``.

    Log-convexity is preserved under addition (Boyd & Vandenberghe
    §3.5.2): a sum of ``LOG_AFFINE``/``LOG_CONVEX`` terms is ``LOG_CONVEX``
    (a sum of two monomials is a posynomial — log-convex, not log-affine).
    Log-*concavity* is **not** preserved under addition, so any
    ``LOG_CONCAVE`` or ``UNKNOWN`` operand yields ``UNKNOWN``.
    """
    convexish = (LogCurvature.LOG_AFFINE, LogCurvature.LOG_CONVEX)
    if a in convexish and b in convexish:
        return LogCurvature.LOG_CONVEX
    return LogCurvature.UNKNOWN


def log_scale_pow(c: LogCurvature, exponent: float) -> LogCurvature:
    """Log-curvature of a constant power ``f**exponent``.

    ``log(f**a) = a * log f``: an affine log stays affine under any real
    ``a``; ``a == 0`` gives the constant ``1`` (``LOG_AFFINE``); ``a > 0``
    preserves the log-curvature; ``a < 0`` flips it (like
    :func:`log_negate`).
    """
    if c == LogCurvature.UNKNOWN:
        return LogCurvature.UNKNOWN
    if c == LogCurvature.LOG_AFFINE:
        return LogCurvature.LOG_AFFINE
    if exponent == 0.0:
        return LogCurvature.LOG_AFFINE
    if exponent > 0.0:
        return c
    return log_negate(c)


# ──────────────────────────────────────────────────────────────────────
# Leaf positivity helpers
# ──────────────────────────────────────────────────────────────────────


def _all_positive(value: object) -> bool:
    """True iff every entry of a numeric value is strictly positive."""
    arr = np.asarray(value, dtype=np.float64)
    return arr.size > 0 and bool(np.all(arr > _POS_TOL))


def _const_scalar(expr: Expression) -> Optional[float]:
    """Scalar value of a constant/parameter leaf, else ``None``."""
    if isinstance(expr, (Constant, Parameter)):
        val = np.asarray(expr.value)
        if val.ndim == 0:
            return float(val)
    return None


def _variable_is_positive(v: Variable) -> bool:
    lb = np.asarray(v.lb, dtype=np.float64)
    return lb.size > 0 and float(lb.min()) > _POS_TOL


def _indexed_variable_is_positive(expr: IndexExpression) -> bool:
    base = expr.base
    if not isinstance(base, Variable):
        return False
    lb = np.asarray(base.lb, dtype=np.float64)
    try:
        idx_lb = float(np.asarray(lb[expr.index]).min())
    except (IndexError, TypeError, ValueError):
        return False
    return idx_lb > _POS_TOL


# ──────────────────────────────────────────────────────────────────────
# The DAG walker
# ──────────────────────────────────────────────────────────────────────


def classify_log_curvature(
    expr: Expression,
    model: Optional[Model] = None,
    _cache: Optional[dict] = None,
) -> LogCurvature:
    """Return the proven :class:`LogCurvature` of ``expr``.

    Walks the expression DAG and propagates log-curvature bottom-up via
    the sound composition rules in this module. ``LogCurvature.UNKNOWN``
    is a conservative verdict (no proof / not provably positive), never a
    claim that the expression is *not* log-convex; callers must treat it
    as such.

    ``model`` is accepted for signature parity with
    :func:`discopt._jax.convexity.classify_expr`; leaf positivity is read
    from each variable's own declared bounds, so it may be ``None``.
    """
    if _cache is None:
        _cache = {}
    eid = id(expr)
    cached = _cache.get(eid)
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    result = _classify_impl(expr, model, _cache)
    _cache[eid] = result
    return result


def _classify_impl(expr: Expression, model: Optional[Model], cache: dict) -> LogCurvature:
    # --- Leaves -----------------------------------------------------
    if isinstance(expr, (Constant, Parameter)):
        # A strictly-positive constant has affine (constant) log.
        return LogCurvature.LOG_AFFINE if _all_positive(expr.value) else LogCurvature.UNKNOWN

    if isinstance(expr, Variable):
        # x > 0  =>  log x = u  is affine in u.
        return LogCurvature.LOG_AFFINE if _variable_is_positive(expr) else LogCurvature.UNKNOWN

    if isinstance(expr, IndexExpression):
        if isinstance(expr.base, Variable):
            return (
                LogCurvature.LOG_AFFINE
                if _indexed_variable_is_positive(expr)
                else LogCurvature.UNKNOWN
            )
        # Indexing a non-variable base preserves elementwise log-curvature.
        return classify_log_curvature(expr.base, model, cache)

    # --- Unary ops --------------------------------------------------
    if isinstance(expr, UnaryOp):
        if expr.op == "abs":
            # On the positive domain |f| = f, so a proven label carries
            # through; an UNKNOWN operand stays UNKNOWN.
            return classify_log_curvature(expr.operand, model, cache)
        # Negation leaves the positive orthant (`-f <= 0`), where log-space
        # curvature is undefined — abstain.
        return LogCurvature.UNKNOWN

    # --- Binary ops -------------------------------------------------
    if isinstance(expr, BinaryOp):
        return _classify_binary(expr, model, cache)

    # --- Function calls --------------------------------------------
    if isinstance(expr, FunctionCall):
        return _classify_function_call(expr, model, cache)

    # --- Aggregations ----------------------------------------------
    if isinstance(expr, SumExpression):
        # A reduction ``sum(operand)`` is an additive fold; a sum over a
        # log-convex operand is log-convex.
        return classify_log_curvature(expr.operand, model, cache)

    if isinstance(expr, SumOverExpression):
        result: Optional[LogCurvature] = None
        for t in expr.terms:
            ti = classify_log_curvature(t, model, cache)
            result = ti if result is None else log_combine_sum(result, ti)
            if result == LogCurvature.UNKNOWN:
                return LogCurvature.UNKNOWN
        return result if result is not None else LogCurvature.UNKNOWN

    # MatMulExpression and everything else: no sound log-curvature rule.
    return LogCurvature.UNKNOWN


def _classify_binary(expr: BinaryOp, model: Optional[Model], cache: dict) -> LogCurvature:
    op = expr.op

    if op == "+":
        a = classify_log_curvature(expr.left, model, cache)
        b = classify_log_curvature(expr.right, model, cache)
        return log_combine_sum(a, b)

    if op == "-":
        # A difference of positive quantities need not be positive, so it
        # leaves the log-space domain — abstain (mirrors ``neg``).
        return LogCurvature.UNKNOWN

    if op == "*":
        a = classify_log_curvature(expr.left, model, cache)
        b = classify_log_curvature(expr.right, model, cache)
        return log_combine_product(a, b)

    if op == "/":
        a = classify_log_curvature(expr.left, model, cache)
        b = classify_log_curvature(expr.right, model, cache)
        return log_combine_product(a, log_negate(b))

    if op == "**":
        # Only a constant real exponent is GP-representable; a variable
        # exponent has no sound log-curvature rule here.
        exponent = _const_scalar(expr.right)
        if exponent is None:
            return LogCurvature.UNKNOWN
        base = classify_log_curvature(expr.left, model, cache)
        return log_scale_pow(base, exponent)

    return LogCurvature.UNKNOWN


def _classify_function_call(
    expr: FunctionCall, model: Optional[Model], cache: dict
) -> LogCurvature:
    name = expr.func_name

    if name == "sqrt" and len(expr.args) == 1:
        # sqrt(f) = f**0.5.
        return log_scale_pow(classify_log_curvature(expr.args[0], model, cache), 0.5)

    if name == "max" and len(expr.args) >= 2:
        # Pointwise max of log-convex (incl. log-affine) is log-convex.
        for a in expr.args:
            if classify_log_curvature(a, model, cache) not in (
                LogCurvature.LOG_AFFINE,
                LogCurvature.LOG_CONVEX,
            ):
                return LogCurvature.UNKNOWN
        return LogCurvature.LOG_CONVEX

    if name == "min" and len(expr.args) >= 2:
        # Pointwise min of log-concave (incl. log-affine) is log-concave.
        for a in expr.args:
            if classify_log_curvature(a, model, cache) not in (
                LogCurvature.LOG_AFFINE,
                LogCurvature.LOG_CONCAVE,
            ):
                return LogCurvature.UNKNOWN
        return LogCurvature.LOG_CONCAVE

    # exp / log and other transcendentals bridge between x-space and
    # log-space; no sound single-lattice rule — abstain.
    return LogCurvature.UNKNOWN


__all__ = [
    "LogCurvature",
    "classify_log_curvature",
    "log_combine_product",
    "log_combine_sum",
    "log_negate",
    "log_scale_pow",
]
