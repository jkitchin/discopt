"""Special convex-pattern recognizers beyond the DCP composition walker.

The DCP walker in :mod:`rules` is sound but incomplete: it leaves
non-constant products and quotients at ``UNKNOWN`` because the
generic rules cannot prove convexity of cone primitives whose
structure is only visible at a supra-node level. This module
supplies targeted recognizers for a small, disciplined set of
shapes where a dedicated proof exists:

* **Homogeneous PSD quadratic ``x^T Q x``** (:func:`is_homogeneous_psd_quadratic`)
  — used as a building block for norm and quadratic-over-linear.
* **Quadratic-over-affine with positive affine denominator**
  (:func:`classify_division_pattern`) — convex when the numerator is
  a homogeneous PSD quadratic.
* **Perspective of exp: ``y * exp(x / y)`` with ``y > 0``**
  (:func:`classify_product_pattern`).
* **Weighted geometric mean ``prod_i x_i^{a_i}``** with ``x_i >= 0``,
  ``0 <= a_i <= 1``, and ``sum_i a_i == 1`` (:func:`classify_product_pattern`).
* **Norm ``sqrt(x^T Q x)`` with Q PSD**
  (:func:`classify_sqrt_pattern`).
* **Quadratic-over-affine epigraph constraint**
  (:func:`classify_fractional_epigraph_constraint`) — the MINLPTests
  ``nlp_cvx_108_*`` family rearranges to ``y >= q(x) / (d x + e)`` with
  ``q`` a PSD quadratic and the linear denominator strictly signed on
  the box.
* **Global quadratic fallback ``f(x) = x^T Q x + c^T x + d``**
  (:func:`quadratic_curvature`) — eigendecomposition of the symmetrised
  Q determines CONVEX / CONCAVE / AFFINE.

Each recognizer has a precise mathematical precondition (PSD of an
extracted matrix, a proven strict sign on a subexpression, a
specific syntactic shape). When the precondition is not met the
recognizer returns ``None`` — the caller preserves its existing
``UNKNOWN`` verdict.

The helpers are ported from the AMP-branch detector on
``bernalde/discopt`` so this branch covers the same MINLPTests
convex families without regressing soundness.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

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

from .lattice import Curvature

# ──────────────────────────────────────────────────────────────────────
# Local utilities (ported with minor adaptation from bernalde/discopt
# branch ``fix/amp-false-infeasible-minlptests``)
# ──────────────────────────────────────────────────────────────────────


def _total_scalar_variables(model: Model) -> int:
    return sum(v.size for v in model._variables)


def _scalar_var_offset(model: Model, target: Variable) -> Optional[int]:
    offset = 0
    for v in model._variables:
        if v is target or v.name == target.name:
            return offset if v.size == 1 else None
        offset += v.size
    return None


def _same_expr(lhs: Expression, rhs: Expression) -> bool:
    """Identity-or-structural equality for the small patterns we match."""
    if lhs is rhs:
        return True
    if isinstance(lhs, Variable) and isinstance(rhs, Variable):
        return lhs.name == rhs.name
    if isinstance(lhs, IndexExpression) and isinstance(rhs, IndexExpression):
        return _same_expr(lhs.base, rhs.base) and lhs.index == rhs.index
    return False


def _contains_var(expr: Expression, target: Variable) -> bool:
    if isinstance(expr, Variable):
        return expr is target or expr.name == target.name
    if isinstance(expr, IndexExpression):
        return isinstance(expr.base, Variable) and (
            expr.base is target or expr.base.name == target.name
        )
    if isinstance(expr, BinaryOp):
        return _contains_var(expr.left, target) or _contains_var(expr.right, target)
    if isinstance(expr, UnaryOp):
        return _contains_var(expr.operand, target)
    if isinstance(expr, FunctionCall):
        return any(_contains_var(arg, target) for arg in expr.args)
    if isinstance(expr, SumExpression):
        return _contains_var(expr.operand, target)
    if isinstance(expr, SumOverExpression):
        return any(_contains_var(term, target) for term in expr.terms)
    return False


def _constant_expr(value: float) -> Constant:
    return Constant(np.array(float(value), dtype=np.float64))


def _add_expr(lhs: Optional[Expression], rhs: Expression) -> Expression:
    if lhs is None:
        return rhs
    return BinaryOp("+", lhs, rhs)


def _scale_expr(expr: Expression, scale: float) -> Expression:
    if abs(scale - 1.0) <= 1e-12:
        return expr
    if abs(scale + 1.0) <= 1e-12:
        return UnaryOp("neg", expr)
    return BinaryOp("*", _constant_expr(scale), expr)


def _flatten_sum_terms(expr: Expression, scale: float, out: list[tuple[float, Expression]]) -> None:
    if isinstance(expr, BinaryOp) and expr.op == "+":
        _flatten_sum_terms(expr.left, scale, out)
        _flatten_sum_terms(expr.right, scale, out)
        return
    if isinstance(expr, BinaryOp) and expr.op == "-":
        _flatten_sum_terms(expr.left, scale, out)
        _flatten_sum_terms(expr.right, -scale, out)
        return
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        _flatten_sum_terms(expr.operand, -scale, out)
        return
    out.append((scale, expr))


def _flatten_product(expr: Expression, out: list[Expression]) -> None:
    if isinstance(expr, BinaryOp) and expr.op == "*":
        _flatten_product(expr.left, out)
        _flatten_product(expr.right, out)
        return
    out.append(expr)


def _extract_power_factor(expr: Expression) -> Optional[tuple[Expression, float]]:
    """Return ``(base, exponent)`` if ``expr`` is ``base ** const`` or ``base``."""
    if isinstance(expr, BinaryOp) and expr.op == "**":
        if isinstance(expr.right, (Constant, Parameter)):
            val = np.asarray(expr.right.value)
            if val.ndim == 0:
                return expr.left, float(val)
        return None
    return expr, 1.0


def _extract_linear_factor(expr: Expression, target: Variable) -> Optional[Expression]:
    """Extract the coefficient expression of ``target`` if ``expr`` is linear in it.

    Returns ``None`` when ``expr`` has a nonlinear (or zero) dependence on
    ``target``. ``Constant(1.0)`` is returned for the variable itself.
    """
    if isinstance(expr, Variable) and (expr is target or expr.name == target.name):
        return _constant_expr(1.0)
    if isinstance(expr, IndexExpression):
        if isinstance(expr.base, Variable) and (
            expr.base is target or expr.base.name == target.name
        ):
            return _constant_expr(1.0)
        return None
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        inner = _extract_linear_factor(expr.operand, target)
        return None if inner is None else UnaryOp("neg", inner)
    if isinstance(expr, BinaryOp) and expr.op == "*":
        left_has = _contains_var(expr.left, target)
        right_has = _contains_var(expr.right, target)
        if left_has and right_has:
            return None
        if left_has:
            inner = _extract_linear_factor(expr.left, target)
            return None if inner is None else BinaryOp("*", inner, expr.right)
        if right_has:
            inner = _extract_linear_factor(expr.right, target)
            return None if inner is None else BinaryOp("*", expr.left, inner)
    return None


def _affine_range_1d(alpha: float, beta: float, lb: float, ub: float) -> tuple[float, float]:
    """Range of ``alpha * x + beta`` for ``x in [lb, ub]``."""
    if alpha >= 0.0:
        lo = alpha * lb + beta if np.isfinite(lb) else (-np.inf if alpha > 0.0 else beta)
        hi = alpha * ub + beta if np.isfinite(ub) else (np.inf if alpha > 0.0 else beta)
    else:
        lo = alpha * ub + beta if np.isfinite(ub) else (-np.inf)
        hi = alpha * lb + beta if np.isfinite(lb) else (np.inf)
    return float(lo), float(hi)


# ──────────────────────────────────────────────────────────────────────
# Sign-on-domain checks
# ──────────────────────────────────────────────────────────────────────


def _has_positive_lower_bound(expr: Expression, model: Model) -> bool:
    """True when ``expr`` is provably strictly positive on the declared box.

    Handles variables, indexed variables, positive constants, and affine
    combinations where every term is strictly positive. Conservative:
    returns False on anything it cannot prove.
    """
    if isinstance(expr, Constant):
        val = np.asarray(expr.value)
        return bool(val.ndim == 0 and float(val) > 0.0)
    if isinstance(expr, Variable):
        lb = float(np.asarray(expr.lb).min())
        return lb > 0.0
    if isinstance(expr, IndexExpression):
        if isinstance(expr.base, Variable):
            try:
                lb = float(np.asarray(np.asarray(expr.base.lb)[expr.index]).min())
            except (IndexError, TypeError, ValueError):
                return False
            return lb > 0.0
    if isinstance(expr, BinaryOp) and expr.op == "+":
        return _has_positive_lower_bound(expr.left, model) and _has_positive_lower_bound(
            expr.right, model
        )
    if isinstance(expr, BinaryOp) and expr.op == "*":
        if isinstance(expr.left, (Constant, Parameter)):
            v = np.asarray(expr.left.value)
            if v.ndim == 0 and float(v) > 0.0:
                return _has_positive_lower_bound(expr.right, model)
        if isinstance(expr.right, (Constant, Parameter)):
            v = np.asarray(expr.right.value)
            if v.ndim == 0 and float(v) > 0.0:
                return _has_positive_lower_bound(expr.left, model)
    return False


def _is_nonneg_domain(expr: Expression, model: Model) -> bool:
    """True when ``expr`` is provably >= 0 on the declared box."""
    if isinstance(expr, Constant):
        val = np.asarray(expr.value)
        return bool(val.ndim == 0 and float(val) >= 0.0)
    if isinstance(expr, Variable):
        lb = float(np.asarray(expr.lb).min())
        return lb >= 0.0
    if isinstance(expr, IndexExpression):
        if isinstance(expr.base, Variable):
            try:
                lb = float(np.asarray(np.asarray(expr.base.lb)[expr.index]).min())
            except (IndexError, TypeError, ValueError):
                return False
            return lb >= 0.0
    if isinstance(expr, BinaryOp) and expr.op == "+":
        return _is_nonneg_domain(expr.left, model) and _is_nonneg_domain(expr.right, model)
    if isinstance(expr, BinaryOp) and expr.op == "*":
        if isinstance(expr.left, (Constant, Parameter)):
            v = np.asarray(expr.left.value)
            if v.ndim == 0 and float(v) >= 0.0:
                return _is_nonneg_domain(expr.right, model)
        if isinstance(expr.right, (Constant, Parameter)):
            v = np.asarray(expr.right.value)
            if v.ndim == 0 and float(v) >= 0.0:
                return _is_nonneg_domain(expr.left, model)
    return False


# ──────────────────────────────────────────────────────────────────────
# Quadratic-form analysis
# ──────────────────────────────────────────────────────────────────────


def _quadratic_data(expr: Expression, model: Model):
    """Extract ``(Q_sym, c, const)`` from ``f = x^T Q x + c^T x + const`` or return None.

    Uses :func:`problem_classifier._extract_quadratic_coefficients` and
    symmetrises ``Q``. Returns ``None`` if the expression is not degree-2.
    """
    from discopt._jax.problem_classifier import _extract_quadratic_coefficients

    try:
        Q, c, const = _extract_quadratic_coefficients(expr, model, _total_scalar_variables(model))
    except Exception:
        return None
    Q = 0.5 * (Q + Q.T)
    return Q, np.asarray(c, dtype=np.float64), float(const)


def is_homogeneous_psd_quadratic(expr: Expression, model: Model) -> bool:
    """True when ``expr`` is ``x^T Q x`` (no linear/constant term) with Q PSD."""
    data = _quadratic_data(expr, model)
    if data is None:
        return False
    Q, c, const = data
    if not np.allclose(c, 0.0, atol=1e-10):
        return False
    if abs(const) > 1e-10:
        return False
    eigvals = np.linalg.eigvalsh(Q)
    return bool(float(np.min(eigvals)) >= -1e-10)


def quadratic_curvature(expr: Expression, model: Model) -> Optional[Curvature]:
    """Return the curvature of a scalar quadratic, if one can be extracted.

    ``None`` when the expression is not extractable as a quadratic form
    (e.g., it contains a non-polynomial atom). Used as a whole-expression
    fallback when the DCP walker leaves a degree-2 polynomial at UNKNOWN
    because its structure only becomes visible after symbolic expansion.
    """
    data = _quadratic_data(expr, model)
    if data is None:
        return None
    Q, _c, _const = data
    if np.allclose(Q, 0.0, atol=1e-10):
        return Curvature.AFFINE
    eigvals = np.linalg.eigvalsh(Q)
    if float(np.min(eigvals)) >= -1e-10:
        return Curvature.CONVEX
    if float(np.max(eigvals)) <= 1e-10:
        return Curvature.CONCAVE
    return Curvature.UNKNOWN


# ──────────────────────────────────────────────────────────────────────
# Product / division / sqrt pattern recognizers
# ──────────────────────────────────────────────────────────────────────


def classify_product_pattern(
    expr: BinaryOp,
    model: Model,
    classify_expr,  # noqa: ANN001 — forward reference avoids circular import
    cache: dict,
) -> Optional[Curvature]:
    """Return CONVEX / CONCAVE for recognised product patterns, else None.

    Recognised:

    * ``y * exp(x / y)`` with ``y > 0`` and ``x`` affine — perspective of
      ``exp``: CONVEX.
    * ``prod_i base_i ** a_i`` with every base affine and nonneg,
      ``a_i in [0, 1]``, ``sum a_i == 1`` — weighted geometric mean:
      CONCAVE.
    """
    # Perspective of exp: y * exp(x / y), y > 0.
    for scale_expr, exp_expr in ((expr.left, expr.right), (expr.right, expr.left)):
        if (
            classify_expr(scale_expr, model, cache) == Curvature.AFFINE
            and _has_positive_lower_bound(scale_expr, model)
            and isinstance(exp_expr, FunctionCall)
            and exp_expr.func_name == "exp"
            and len(exp_expr.args) == 1
        ):
            inner = exp_expr.args[0]
            if isinstance(inner, BinaryOp) and inner.op == "/":
                if (
                    _same_expr(scale_expr, inner.right)
                    and classify_expr(inner.left, model, cache) == Curvature.AFFINE
                    and classify_expr(inner.right, model, cache) == Curvature.AFFINE
                ):
                    return Curvature.CONVEX

    # Weighted geometric mean: prod_i x_i^{a_i}, each x_i affine & nonneg,
    # a_i in [0, 1], sum a_i = 1.
    factors: list[Expression] = []
    _flatten_product(expr, factors)
    if len(factors) < 2:
        return None

    parsed: list[tuple[Expression, float]] = []
    for factor in factors:
        extracted = _extract_power_factor(factor)
        if extracted is None:
            return None
        base, exponent = extracted
        if exponent < -1e-10 or exponent > 1.0 + 1e-10:
            return None
        if classify_expr(base, model, cache) != Curvature.AFFINE:
            return None
        if not _is_nonneg_domain(base, model):
            return None
        parsed.append((base, exponent))

    if abs(sum(exp for _, exp in parsed) - 1.0) <= 1e-10:
        return Curvature.CONCAVE
    return None


def classify_division_pattern(
    expr: BinaryOp,
    model: Model,
    classify_expr,  # noqa: ANN001
    cache: dict,
) -> Optional[Curvature]:
    """Return CONVEX for ``x^T Q x / affine(y)`` with ``affine > 0`` and Q PSD."""
    if classify_expr(expr.right, model, cache) != Curvature.AFFINE:
        return None
    if not _has_positive_lower_bound(expr.right, model):
        return None
    if is_homogeneous_psd_quadratic(expr.left, model):
        return Curvature.CONVEX
    return None


def classify_sqrt_pattern(
    arg: Expression,
    model: Model,
) -> Optional[Curvature]:
    """Return CONVEX for ``sqrt(x^T Q x)`` with Q PSD (Euclidean-style norm)."""
    if is_homogeneous_psd_quadratic(arg, model):
        return Curvature.CONVEX
    return None


# ──────────────────────────────────────────────────────────────────────
# Constraint-level: quadratic-over-affine epigraph recognition
# ──────────────────────────────────────────────────────────────────────


def classify_fractional_epigraph_constraint(
    constraint: Constraint,
    model: Model,
) -> Optional[bool]:
    """Detect scalar epigraphs of univariate quadratic-over-affine forms.

    Recognises a ``<=`` constraint whose body linearises as

        ``coeff(x) * y + remainder(x) <= 0``

    with:

    * ``y`` a scalar model variable appearing only linearly,
    * ``coeff(x)`` an affine form in a single other scalar variable ``x``
      with a proven non-zero sign on the declared box,
    * ``remainder(x)`` a scalar quadratic in ``x`` only (no dependence on
      ``y`` or any other variable).

    Such a constraint rearranges to ``y >= q(x) / L(x)`` (or ``<=``,
    depending on sign of ``L``), a univariate quadratic-over-affine; it
    is convex iff the Schur-complement discriminant
    ``a e^2 - b d e + c d^2`` has the right sign, where
    ``q(x) = a x^2 + b x + c`` and ``L(x) = d x + e``.

    Covers the MINLPTests ``nlp_cvx_108_*`` family, which the DCP
    walker cannot classify directly (the body is syntactically
    bilinear + quadratic, but algebraically an epigraph of a convex
    quadratic-over-linear).
    """
    from discopt._jax.problem_classifier import _extract_linear_coefficients

    if constraint.sense != "<=":
        return None

    scalar_targets = [v for v in model._variables if v.size == 1]
    if len(scalar_targets) != 2:
        return None

    n = _total_scalar_variables(model)
    for target in scalar_targets:
        terms: list[tuple[float, Expression]] = []
        _flatten_sum_terms(constraint.body, 1.0, terms)

        coeff_expr: Optional[Expression] = None
        remainder_expr: Optional[Expression] = None
        valid = True
        for term_scale, term in terms:
            factor = _extract_linear_factor(term, target)
            if factor is None:
                if _contains_var(term, target):
                    valid = False
                    break
                remainder_expr = _add_expr(remainder_expr, _scale_expr(term, term_scale))
                continue
            coeff_expr = _add_expr(coeff_expr, _scale_expr(factor, term_scale))

        if not valid or coeff_expr is None or remainder_expr is None:
            continue

        try:
            coeff_vec, coeff_const = _extract_linear_coefficients(coeff_expr, model, n)
        except Exception:
            continue

        nonzero_coeff = np.flatnonzero(np.abs(coeff_vec) > 1e-10)
        target_idx = _scalar_var_offset(model, target)
        if target_idx is None:
            continue
        if target_idx in nonzero_coeff:
            continue
        if len(nonzero_coeff) != 1:
            continue
        other_idx = int(nonzero_coeff[0])

        data = _quadratic_data(remainder_expr, model)
        if data is None:
            continue
        Q, c, const = data
        remainder_support: set[int] = {int(i) for i in np.flatnonzero(np.abs(np.diag(Q)) > 1e-10)}
        remainder_support |= {int(i) for i in np.flatnonzero(np.abs(c) > 1e-10)}
        if remainder_support - {other_idx}:
            continue
        row_mask = np.arange(Q.shape[0]) != other_idx
        if np.any(np.abs(Q[row_mask, :]) > 1e-10):
            continue
        if np.any(np.abs(Q[:, row_mask]) > 1e-10):
            continue

        other_var = None
        running = 0
        for var in model._variables:
            if running == other_idx and var.size == 1:
                other_var = var
                break
            running += var.size
        if other_var is None:
            continue

        a = 0.5 * float(Q[other_idx, other_idx])
        b = float(c[other_idx])
        c0 = float(const)
        d = float(coeff_vec[other_idx])
        e = float(coeff_const)
        if abs(a) <= 1e-10:
            continue
        lb = float(np.asarray(other_var.lb).min())
        ub = float(np.asarray(other_var.ub).max())
        coeff_lo, coeff_hi = _affine_range_1d(d, e, lb, ub)

        # Schur-complement discriminant for the quadratic-over-affine
        # q(x) / L(x): the epigraph { (x, y) : q(x)/L(x) <= y } with L>0
        # is convex iff a e^2 - b d e + c d^2 >= 0 (equivalently, the
        # 2x2 matrix [[a, (b d - a e)/... ]] has the right PSD profile).
        curvature_numerator = a * e * e - b * d * e + c0 * d * d
        if coeff_hi < -1e-10:
            # coeff(x) * y + r(x) <= 0 with coeff < 0 ⇒ y >= r(x) / (-coeff).
            return curvature_numerator >= -1e-10
        if coeff_lo > 1e-10:
            # coeff(x) * y + r(x) <= 0 with coeff > 0 ⇒ y <= -r(x) / coeff.
            return curvature_numerator <= 1e-10

    return None


__all__ = [
    "classify_division_pattern",
    "classify_fractional_epigraph_constraint",
    "classify_product_pattern",
    "classify_sqrt_pattern",
    "is_homogeneous_psd_quadratic",
    "quadratic_curvature",
]
