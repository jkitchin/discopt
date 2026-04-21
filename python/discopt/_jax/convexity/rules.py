"""SUSPECT-style convexity / sign propagation over the expression DAG.

This module walks a ``discopt.modeling`` expression tree and assigns a
:class:`~.lattice.ExprInfo` (curvature + sign) to every subexpression
using the disciplined convex programming composition rule
(Grant, Boyd, Ye 2006) combined with sign-aware reasoning about
monotonicity of atoms (SUSPECT; Ceccon, Siirola, Misener 2020).

Soundness invariant
-------------------
A CONVEX or CONCAVE verdict is a mathematical proof, derived only from
rules whose premises are satisfied on the expression's domain. When no
rule applies the walker returns ``Curvature.UNKNOWN`` — never a
speculative classification. Sign information is tightened only when
provable from the expression's bounds or algebraic structure; unknown
signs degrade gracefully without poisoning the curvature verdict.

Rules implemented
-----------------
Leaves: Constant / Parameter / Variable / IndexExpression — AFFINE with
sign derived from value or bounds.

Unary: negation flips curvature and sign; ``abs`` is CONVEX and yields
a NONNEG result.

Binary:

* ``a + b`` — curvature via :func:`combine_sum`, sign via
  :func:`sign_add`.
* ``a - b`` — reduces to ``a + (-b)``.
* ``k * expr`` (constant scalar) — :func:`scale` applied to curvature,
  :func:`sign_mul` applied to sign.
* ``expr * expr`` — curvature UNKNOWN (bilinear); sign is the product
  of the two sign labels.
* ``expr / k`` (constant scalar, nonzero) — behaves as ``(1/k) * expr``.
* ``k / expr`` (constant numerator, argument with strictly known sign)
  — reciprocal rule: ``1/expr`` is convex on a strictly positive
  domain, concave on a strictly negative domain. Scaled by the
  numerator's sign.
* ``a ** p`` (p a literal scalar exponent) — full case analysis on
  parity, magnitude, and base sign (Boyd & Vandenberghe, *Convex
  Optimization*, §3.1.5).

Function calls: the unary atom table (:func:`unary_atom_profile`) is
consulted with the argument's sign to obtain the atom's curvature and
monotonicity; :func:`compose` produces the verdict. ``max`` and
``min`` are handled as sound n-ary extensions (max preserves
convexity, min preserves concavity).

References
----------
Grant, Boyd, Ye (2006), "Disciplined Convex Programming."
Boyd, Vandenberghe (2004), *Convex Optimization*, §3.1.
Ceccon, Siirola, Misener (2020), "SUSPECT," TOP.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from discopt._jax.problem_classifier import (
    _extract_linear_coefficients,
    _extract_quadratic_coefficients,
)
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
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

from .lattice import (
    AtomProfile,
    Curvature,
    ExprInfo,
    Monotonicity,
    Sign,
    combine_sum,
    compose,
    is_nonneg,
    is_nonpos,
    is_pos,
    is_strict,
    negate,
    scale,
    sign_add,
    sign_from_bounds,
    sign_from_value,
    sign_mul,
    sign_negate,
    sign_reciprocal,
    unary_atom_profile,
)
from .linear_context import LinearContext, build_linear_context, extract_affine

_LINEAR_CONTEXT_KEY = "__linear_context__"


def _get_linear_context(cache: dict) -> Optional[LinearContext]:
    """Return the cached linear context, if one was stashed."""
    return cache.get(_LINEAR_CONTEXT_KEY) if cache else None


def _refine_sign(
    expr: Expression,
    model: Optional[Model],
    cache: dict,
    current_sign: Sign,
) -> Sign:
    """Tighten ``current_sign`` using the model's linear relaxation.

    Applied at composition sites where the outer atom has a
    restricted domain (``log``, ``sqrt``, ``1/x``, fractional
    powers). Only affine arguments are handled — nonlinear arguments
    would need a general interval walker and are left at their
    syntactic sign. The refinement is monotone: it never loses
    strength (a STRICT ``current_sign`` is returned unchanged), and
    is sound because ``LinearContext.affine_range`` is a proven
    enclosure over the intersection of the box and the linear
    relaxation.
    """
    if is_strict(current_sign):
        return current_sign
    ctx = _get_linear_context(cache)
    if ctx is None or model is None:
        return current_sign
    aff = extract_affine(expr, model, ctx.n_vars)
    if aff is None:
        return current_sign
    coeffs, const = aff
    lo, hi = ctx.affine_range(coeffs, const)
    return sign_from_bounds(lo, hi)


def _is_nonneg_domain(expr: Expression, model: Optional[Model], cache: dict) -> bool:
    """True when ``expr`` is provably nonnegative on the current box."""
    sign = classify_expr_info(expr, model, cache).sign
    if model is not None and not is_nonneg(sign):
        sign = _refine_sign(expr, model, cache, sign)
    return is_nonneg(sign)


def _has_positive_lower_bound(expr: Expression, model: Optional[Model], cache: dict) -> bool:
    """True when ``expr`` is provably strictly positive on the current box."""
    sign = classify_expr_info(expr, model, cache).sign
    if model is not None and not is_pos(sign):
        sign = _refine_sign(expr, model, cache, sign)
    return is_pos(sign)


def _same_expr(lhs: Expression, rhs: Expression) -> bool:
    """Best-effort structural equality for small pattern checks."""
    if lhs is rhs:
        return True
    if type(lhs) is not type(rhs):
        return False

    if isinstance(lhs, Constant):
        return bool(np.array_equal(lhs.value, rhs.value))
    if isinstance(lhs, Parameter):
        return lhs.name == rhs.name and bool(np.array_equal(lhs.value, rhs.value))
    if isinstance(lhs, Variable):
        return lhs.name == rhs.name
    if isinstance(lhs, IndexExpression):
        return lhs.index == rhs.index and _same_expr(lhs.base, rhs.base)
    if isinstance(lhs, UnaryOp):
        return lhs.op == rhs.op and _same_expr(lhs.operand, rhs.operand)
    if isinstance(lhs, BinaryOp):
        return (
            lhs.op == rhs.op
            and _same_expr(lhs.left, rhs.left)
            and _same_expr(lhs.right, rhs.right)
        )
    if isinstance(lhs, FunctionCall):
        return lhs.func_name == rhs.func_name and len(lhs.args) == len(rhs.args) and all(
            _same_expr(a, b) for a, b in zip(lhs.args, rhs.args)
        )
    if isinstance(lhs, SumExpression):
        return _same_expr(lhs.operand, rhs.operand)
    if isinstance(lhs, SumOverExpression):
        return len(lhs.terms) == len(rhs.terms) and all(
            _same_expr(a, b) for a, b in zip(lhs.terms, rhs.terms)
        )
    return False


def _total_scalar_variables(model: Model) -> int:
    return sum(int(v.size) for v in model._variables)


def _scalar_var_offset(model: Model, target: Variable) -> Optional[int]:
    offset = 0
    for var in model._variables:
        if var is target:
            return offset
        offset += int(var.size)
    return None


def _quadratic_data(
    expr: Expression,
    model: Model,
) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """Extract scalar quadratic data or return ``None``."""
    try:
        Q, c, const = _extract_quadratic_coefficients(expr, model, _total_scalar_variables(model))
    except Exception:
        return None
    return 0.5 * (Q + Q.T), c, float(const)


def _is_homogeneous_psd_quadratic(expr: Expression, model: Model) -> bool:
    """Check if ``expr`` is ``x'Qx`` with PSD ``Q`` and no affine offset."""
    data = _quadratic_data(expr, model)
    if data is None:
        return False
    Q, c, const = data
    if not np.allclose(c, 0.0, atol=1e-10):
        return False
    if abs(const) > 1e-10:
        return False
    eigvals = np.linalg.eigvalsh(Q)
    return float(np.min(eigvals)) >= -1e-10


def _flatten_product(expr: Expression, out: list[Expression]) -> None:
    if isinstance(expr, BinaryOp) and expr.op == "*":
        _flatten_product(expr.left, out)
        _flatten_product(expr.right, out)
        return
    out.append(expr)


def _extract_power_factor(expr: Expression) -> Optional[tuple[Expression, float]]:
    if isinstance(expr, BinaryOp) and expr.op == "**":
        if isinstance(expr.right, (Constant, Parameter)):
            exponent_val = np.asarray(expr.right.value)
            if exponent_val.ndim == 0:
                return expr.left, float(exponent_val)
        return None
    return expr, 1.0


def _flatten_sum_terms(
    expr: Expression, scale: float, out: list[tuple[float, Expression]]
) -> None:
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


def _extract_linear_factor(expr: Expression, target: Variable) -> Optional[Expression]:
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
    if alpha >= 0.0:
        lo = alpha * lb + beta if np.isfinite(lb) else (-np.inf if alpha > 0.0 else beta)
        hi = alpha * ub + beta if np.isfinite(ub) else (np.inf if alpha > 0.0 else beta)
    else:
        lo = alpha * ub + beta if np.isfinite(ub) else -np.inf
        hi = alpha * lb + beta if np.isfinite(lb) else np.inf
    return float(lo), float(hi)


def _classify_fractional_epigraph_constraint(
    constraint: Constraint,
    model: Optional[Model],
) -> Optional[bool]:
    """Detect scalar epigraph/hypograph constraints for quadratic-over-affine forms."""
    if model is None or constraint.sense != "<=":
        return None

    scalar_targets = [v for v in model._variables if v.size == 1]
    if len(scalar_targets) != 2:
        return None

    n_vars = _total_scalar_variables(model)
    for target in scalar_targets:
        terms: list[tuple[float, Expression]] = []
        _flatten_sum_terms(constraint.body, 1.0, terms)

        coeff_expr: Optional[Expression] = None
        remainder_expr: Optional[Expression] = None
        valid = True
        for scale_factor, term in terms:
            factor = _extract_linear_factor(term, target)
            if factor is None:
                if _contains_var(term, target):
                    valid = False
                    break
                remainder_expr = _add_expr(remainder_expr, _scale_expr(term, scale_factor))
                continue
            coeff_expr = _add_expr(coeff_expr, _scale_expr(factor, scale_factor))

        if not valid or coeff_expr is None or remainder_expr is None:
            continue

        try:
            coeff_vec, coeff_const = _extract_linear_coefficients(coeff_expr, model, n_vars)
        except Exception:
            continue

        nonzero_coeff = np.flatnonzero(np.abs(coeff_vec) > 1e-10)
        target_idx = _scalar_var_offset(model, target)
        if target_idx is None or target_idx in nonzero_coeff or len(nonzero_coeff) != 1:
            continue
        other_idx = int(nonzero_coeff[0])

        data = _quadratic_data(remainder_expr, model)
        if data is None:
            continue
        Q, c, const = data
        remainder_support = set(np.flatnonzero(np.abs(np.diag(Q)) > 1e-10))
        remainder_support |= set(np.flatnonzero(np.abs(c) > 1e-10))
        if remainder_support - {other_idx}:
            continue
        mask = np.arange(Q.shape[0]) != other_idx
        if np.any(np.abs(Q[mask, :]) > 1e-10):
            continue
        if np.any(np.abs(Q[:, mask]) > 1e-10):
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

        lb = float(other_var.lb)
        ub = float(other_var.ub)
        coeff_lo, coeff_hi = _affine_range_1d(d, e, lb, ub)
        curvature_numerator = a * e * e - b * d * e + c0 * d * d

        if coeff_hi < -1e-10:
            return curvature_numerator >= -1e-10
        if coeff_lo > 1e-10:
            return curvature_numerator <= 1e-10

    return None


def _classify_product_special(
    expr: BinaryOp,
    model: Optional[Model],
    cache: dict,
) -> Optional[ExprInfo]:
    if model is None:
        return None

    for scale_expr, exp_expr in ((expr.left, expr.right), (expr.right, expr.left)):
        scale_info = classify_expr_info(scale_expr, model, cache)
        if (
            scale_info.curvature == Curvature.AFFINE
            and _has_positive_lower_bound(scale_expr, model, cache)
            and isinstance(exp_expr, FunctionCall)
            and exp_expr.func_name == "exp"
            and len(exp_expr.args) == 1
        ):
            inner = exp_expr.args[0]
            if isinstance(inner, BinaryOp) and inner.op == "/":
                numerator = classify_expr_info(inner.left, model, cache)
                denominator = classify_expr_info(inner.right, model, cache)
                if (
                    _same_expr(scale_expr, inner.right)
                    and numerator.curvature == Curvature.AFFINE
                    and denominator.curvature == Curvature.AFFINE
                ):
                    return ExprInfo(Curvature.CONVEX, Sign.POS)

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
        if classify_expr_info(base, model, cache).curvature != Curvature.AFFINE:
            return None
        if not _is_nonneg_domain(base, model, cache):
            return None
        parsed.append((base, exponent))

    if abs(sum(exponent for _, exponent in parsed) - 1.0) <= 1e-10:
        return ExprInfo(Curvature.CONCAVE, Sign.NONNEG)
    return None


def _classify_division_special(
    expr: BinaryOp,
    model: Optional[Model],
    cache: dict,
) -> Optional[ExprInfo]:
    if model is None:
        return None
    if classify_expr_info(expr.right, model, cache).curvature != Curvature.AFFINE:
        return None
    if not _has_positive_lower_bound(expr.right, model, cache):
        return None
    if _is_homogeneous_psd_quadratic(expr.left, model):
        return ExprInfo(Curvature.CONVEX, Sign.NONNEG)
    return None


def _classify_function_special(
    expr: FunctionCall,
    model: Optional[Model],
    cache: dict,
) -> Optional[ExprInfo]:
    del cache
    if model is None:
        return None
    if expr.func_name == "sqrt" and len(expr.args) == 1:
        if _is_homogeneous_psd_quadratic(expr.args[0], model):
            return ExprInfo(Curvature.CONVEX, Sign.NONNEG)
    return None


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def classify_expr(
    expr: Expression,
    model: Optional[Model] = None,
    _cache: Optional[dict] = None,
) -> Curvature:
    """Return the proven curvature of ``expr``.

    ``Curvature.UNKNOWN`` is a conservative verdict, not a claim of
    nonconvexity; downstream consumers must still treat the result as
    non-convex when no proof is available.
    """
    return classify_expr_info(expr, model, _cache).curvature


def classify_expr_info(
    expr: Expression,
    model: Optional[Model] = None,
    _cache: Optional[dict] = None,
) -> ExprInfo:
    """Internal-detail: return full (curvature, sign) info for ``expr``.

    Exposed so callers (e.g., other detector phases, tests) can reuse
    the sign propagation without re-walking the DAG.
    """
    if _cache is None:
        _cache = {}

    eid = id(expr)
    if eid in _cache:
        return _cache[eid]  # type: ignore[no-any-return]

    info = _classify_impl(expr, model, _cache)
    _cache[eid] = info
    return info


# ──────────────────────────────────────────────────────────────────────
# Sign extraction for leaves
# ──────────────────────────────────────────────────────────────────────


def _variable_sign(v: Variable) -> Sign:
    """Conservative sign of every entry of a variable."""
    lb_arr = np.asarray(v.lb)
    ub_arr = np.asarray(v.ub)
    if lb_arr.size == 0:
        return Sign.UNKNOWN
    return sign_from_bounds(float(lb_arr.min()), float(ub_arr.max()))


def _indexed_variable_sign(expr: IndexExpression) -> Sign:
    """Sign of ``var[idx]`` — sharper than the whole-variable bound."""
    if not isinstance(expr.base, Variable):
        return Sign.UNKNOWN
    lb = np.asarray(expr.base.lb)
    ub = np.asarray(expr.base.ub)
    try:
        lb_val = float(np.asarray(lb[expr.index]).min())
        ub_val = float(np.asarray(ub[expr.index]).max())
    except (IndexError, TypeError, ValueError):
        return Sign.UNKNOWN
    return sign_from_bounds(lb_val, ub_val)


# ──────────────────────────────────────────────────────────────────────
# Core dispatcher
# ──────────────────────────────────────────────────────────────────────


def _classify_impl(expr: Expression, model: Optional[Model], cache: dict) -> ExprInfo:
    """Dispatch on expression node type to an :class:`ExprInfo`."""

    # --- Leaves -----------------------------------------------------
    if isinstance(expr, (Constant, Parameter)):
        return ExprInfo(Curvature.AFFINE, sign_from_value(expr.value))

    if isinstance(expr, Variable):
        return ExprInfo(Curvature.AFFINE, _variable_sign(expr))

    if isinstance(expr, IndexExpression):
        # Tighten the sign when the base is a Variable by looking at
        # the per-index bound; otherwise fall back to recursing into
        # the base expression.
        if isinstance(expr.base, Variable):
            return ExprInfo(Curvature.AFFINE, _indexed_variable_sign(expr))
        base = classify_expr_info(expr.base, model, cache)
        return base  # indexing preserves curvature and sign info.

    # --- Unary ops --------------------------------------------------
    if isinstance(expr, UnaryOp):
        child = classify_expr_info(expr.operand, model, cache)
        if expr.op == "neg":
            return ExprInfo(negate(child.curvature), sign_negate(child.sign))
        if expr.op == "abs":
            # |x| is convex everywhere; the value is nonneg.
            if child.curvature == Curvature.AFFINE:
                return ExprInfo(Curvature.CONVEX, Sign.NONNEG)
            return ExprInfo(Curvature.UNKNOWN, Sign.NONNEG)
        return ExprInfo(Curvature.UNKNOWN, Sign.UNKNOWN)

    # --- Binary ops -------------------------------------------------
    if isinstance(expr, BinaryOp):
        return _classify_binary(expr, model, cache)

    # --- Function calls --------------------------------------------
    if isinstance(expr, FunctionCall):
        return _classify_function_call(expr, model, cache)

    # --- Aggregations ----------------------------------------------
    if isinstance(expr, SumExpression):
        return classify_expr_info(expr.operand, model, cache)

    if isinstance(expr, SumOverExpression):
        curv: Curvature = Curvature.AFFINE
        s: Sign = Sign.ZERO
        for t in expr.terms:
            t_info = classify_expr_info(t, model, cache)
            curv = combine_sum(curv, t_info.curvature)
            s = sign_add(s, t_info.sign)
            if curv == Curvature.UNKNOWN:
                # We can keep refining the sum's sign but curvature is
                # already lost; still return a best-effort sign.
                return ExprInfo(Curvature.UNKNOWN, Sign.UNKNOWN)
        return ExprInfo(curv, s)

    if isinstance(expr, MatMulExpression):
        return _classify_matmul(expr, model, cache)

    return ExprInfo(Curvature.UNKNOWN, Sign.UNKNOWN)


# ──────────────────────────────────────────────────────────────────────
# Binary ops
# ──────────────────────────────────────────────────────────────────────


def _is_scalar_const(expr: Expression) -> bool:
    """True if ``expr`` is a concrete numeric scalar."""
    if isinstance(expr, (Constant, Parameter)):
        val = np.asarray(expr.value)
        return bool(val.ndim == 0)
    return False


def _scalar_value(expr: Expression) -> float:
    return float(np.asarray(expr.value))  # type: ignore[attr-defined]


def _classify_binary(expr: BinaryOp, model: Optional[Model], cache: dict) -> ExprInfo:
    left = classify_expr_info(expr.left, model, cache)
    right = classify_expr_info(expr.right, model, cache)

    if expr.op == "+":
        return ExprInfo(
            combine_sum(left.curvature, right.curvature),
            sign_add(left.sign, right.sign),
        )

    if expr.op == "-":
        return ExprInfo(
            combine_sum(left.curvature, negate(right.curvature)),
            sign_add(left.sign, sign_negate(right.sign)),
        )

    if expr.op == "*":
        return _classify_product(expr, left, right, model, cache)

    if expr.op == "/":
        return _classify_division(expr, left, right, model, cache)

    if expr.op == "**":
        return _classify_power(expr, left, model, cache)

    return ExprInfo(Curvature.UNKNOWN, Sign.UNKNOWN)


def _classify_product(
    expr: BinaryOp,
    left: ExprInfo,
    right: ExprInfo,
    model: Optional[Model],
    cache: dict,
) -> ExprInfo:
    """Classify ``a * b`` with sign-aware curvature."""
    prod_sign = sign_mul(left.sign, right.sign)

    # Constant scaling on either side → curvature scaled by that sign.
    if _is_scalar_const(expr.left):
        val = _scalar_value(expr.left)
        s = 0 if val == 0 else (1 if val > 0 else -1)
        return ExprInfo(scale(right.curvature, s), prod_sign)
    if _is_scalar_const(expr.right):
        val = _scalar_value(expr.right)
        s = 0 if val == 0 else (1 if val > 0 else -1)
        return ExprInfo(scale(left.curvature, s), prod_sign)

    special = _classify_product_special(expr, model, cache)
    if special is not None:
        return special

    # Bilinear / general product: curvature is UNKNOWN even when both
    # factors share a sign (consider x*y on the positive orthant, whose
    # Hessian has eigenvalues ±1). Sign can still be tightened.
    return ExprInfo(Curvature.UNKNOWN, prod_sign)


def _classify_division(
    expr: BinaryOp,
    left: ExprInfo,
    right: ExprInfo,
    model: Optional[Model],
    cache: dict,
) -> ExprInfo:
    """Classify ``a / b``."""
    # Divide by constant: scale by 1/k.
    if _is_scalar_const(expr.right):
        val = _scalar_value(expr.right)
        if abs(val) <= 1e-30:
            return ExprInfo(Curvature.UNKNOWN, Sign.UNKNOWN)
        s = 1 if val > 0 else -1
        inv_sign = Sign.POS if val > 0 else Sign.NEG
        return ExprInfo(scale(left.curvature, s), sign_mul(left.sign, inv_sign))

    special = _classify_division_special(expr, model, cache)
    if special is not None:
        return special

    # Reciprocal with constant numerator and strictly-signed denominator.
    # 1/u is convex + nonincreasing on u>0; concave + nonincreasing on
    # u<0. The DCP composition rule combines this profile with the
    # inner expression's curvature — bypassing it (i.e., trusting the
    # sign alone) would be UNSOUND: e.g., 1/(1 + exp(-x)) has positive
    # denominator but convex inner, so the composite is neither convex
    # nor concave.
    right_sign = right.sign
    if _is_scalar_const(expr.left) and not is_strict(right_sign):
        right_sign = _refine_sign(expr.right, model, cache, right_sign)
    if _is_scalar_const(expr.left) and is_strict(right_sign):
        c = _scalar_value(expr.left)
        recip_curv = Curvature.CONVEX if is_pos(right_sign) else Curvature.CONCAVE
        recip_mono = Monotonicity.NONINC
        composed = compose(recip_curv, recip_mono, right.curvature)
        recip_sign = sign_reciprocal(right_sign)
        if c == 0:
            return ExprInfo(Curvature.AFFINE, Sign.ZERO)
        c_sign = 1 if c > 0 else -1
        return ExprInfo(
            scale(composed, c_sign),
            sign_mul(sign_from_value(c), recip_sign),
        )

    # General quotient — no sound curvature verdict.
    return ExprInfo(Curvature.UNKNOWN, sign_mul(left.sign, sign_reciprocal(right.sign)))


def _classify_power(
    expr: BinaryOp,
    base: ExprInfo,
    model: Optional[Model],
    cache: dict,
) -> ExprInfo:
    """Classify ``base ** exponent`` for a scalar constant ``exponent``.

    Case analysis follows Boyd & Vandenberghe §3.1.5.
    """
    if not _is_scalar_const(expr.right):
        return ExprInfo(Curvature.UNKNOWN, Sign.UNKNOWN)

    n = _scalar_value(expr.right)
    n_int = int(n)
    is_int = np.isclose(n, float(n_int))

    # Tighten base sign when it's needed for a conclusive verdict:
    # fractional exponents and negative exponents both require the
    # base to lie in a known half-line.
    if not is_strict(base.sign) and (
        (is_int and n_int < 0) or (not is_int and (n < 0 or 0 < n < 1 or n > 1))
    ):
        base = ExprInfo(base.curvature, _refine_sign(expr.left, model, cache, base.sign))

    # Trivial exponents.
    if np.isclose(n, 0.0):
        return ExprInfo(Curvature.AFFINE, Sign.POS)  # x^0 = 1
    if np.isclose(n, 1.0):
        return base

    # x^2 is convex on all of R for an affine base; sign is NONNEG.
    if np.isclose(n, 2.0):
        if base.curvature == Curvature.AFFINE:
            return ExprInfo(Curvature.CONVEX, Sign.NONNEG)
        return ExprInfo(Curvature.UNKNOWN, Sign.NONNEG)

    # Even integer power >=2 (n=2 handled above).
    if is_int and n_int >= 2 and n_int % 2 == 0:
        if base.curvature == Curvature.AFFINE:
            return ExprInfo(Curvature.CONVEX, Sign.NONNEG)
        return ExprInfo(Curvature.UNKNOWN, Sign.NONNEG)

    # Odd integer power >=3: sign-dependent curvature, sign inherits base.
    if is_int and n_int >= 3 and n_int % 2 == 1:
        out_sign = base.sign  # odd power preserves sign
        if base.curvature == Curvature.AFFINE:
            if is_nonneg(base.sign):
                return ExprInfo(Curvature.CONVEX, out_sign)
            if is_nonpos(base.sign):
                return ExprInfo(Curvature.CONCAVE, out_sign)
        return ExprInfo(Curvature.UNKNOWN, out_sign)

    # Negative integer exponent — x^(-k) for k a positive integer on a
    # strictly-signed domain. Convex on x>0 for any negative exponent;
    # on x<0 the verdict depends on parity of k.
    if is_int and n_int < 0:
        k = -n_int
        if is_pos(base.sign) and base.curvature == Curvature.AFFINE:
            # x^(-k) = 1/x^k: convex on (0, inf).
            return ExprInfo(Curvature.CONVEX, Sign.POS)
        if base.sign == Sign.NEG and base.curvature == Curvature.AFFINE:
            # x^(-k) on x<0: sign alternates with parity of k, so does
            # curvature. Even k → positive, convex. Odd k → negative,
            # concave.
            if k % 2 == 0:
                return ExprInfo(Curvature.CONVEX, Sign.POS)
            return ExprInfo(Curvature.CONCAVE, Sign.NEG)
        return ExprInfo(Curvature.UNKNOWN, Sign.UNKNOWN)

    # Fractional 0 < n < 1 on nonneg domain: concave; result ≥ 0.
    if 0 < n < 1:
        if base.curvature == Curvature.AFFINE and is_nonneg(base.sign):
            return ExprInfo(Curvature.CONCAVE, Sign.NONNEG)
        return ExprInfo(Curvature.UNKNOWN, Sign.UNKNOWN)

    # n > 1, non-integer, on nonneg domain: convex; result ≥ 0.
    if n > 1:
        if base.curvature == Curvature.AFFINE and is_nonneg(base.sign):
            return ExprInfo(Curvature.CONVEX, Sign.NONNEG)
        return ExprInfo(Curvature.UNKNOWN, Sign.UNKNOWN)

    # n < 0 non-integer on strictly positive domain: convex.
    if n < 0 and is_pos(base.sign) and base.curvature == Curvature.AFFINE:
        return ExprInfo(Curvature.CONVEX, Sign.POS)

    return ExprInfo(Curvature.UNKNOWN, Sign.UNKNOWN)


# ──────────────────────────────────────────────────────────────────────
# Function calls
# ──────────────────────────────────────────────────────────────────────


def _classify_function_call(expr: FunctionCall, model: Optional[Model], cache: dict) -> ExprInfo:
    name = expr.func_name

    # n-ary atoms: max / min / sum_of_squares / norm2.
    if name == "max" and len(expr.args) >= 2:
        return _classify_nary_max(expr, model, cache)
    if name == "min" and len(expr.args) >= 2:
        return _classify_nary_min(expr, model, cache)

    special = _classify_function_special(expr, model, cache)
    if special is not None:
        return special

    if len(expr.args) != 1:
        return ExprInfo(Curvature.UNKNOWN, Sign.UNKNOWN)

    arg_info = classify_expr_info(expr.args[0], model, cache)
    arg_sign = arg_info.sign
    # Restricted-domain atoms need a strict sign proof on the argument
    # to license the DCP rule. Tighten via the linear relaxation when
    # the syntactic sign alone is too weak.
    if name in ("log", "log2", "log10", "sqrt") and not is_strict(arg_sign):
        arg_sign = _refine_sign(expr.args[0], model, cache, arg_sign)
    profile: Optional[AtomProfile] = unary_atom_profile(name, arg_sign)
    if profile is None:
        return ExprInfo(Curvature.UNKNOWN, _function_result_sign(name, arg_sign))

    curv = compose(profile.curvature, profile.monotonicity, arg_info.curvature)
    return ExprInfo(curv, _function_result_sign(name, arg_sign))


def _function_result_sign(name: str, arg_sign: Sign) -> Sign:
    """Best sign for the result of an atom given its argument's sign."""
    if name == "exp":
        return Sign.POS
    if name in ("log", "log2", "log10"):
        # log(x) ≤ 0 iff x ≤ 1, ≥ 0 iff x ≥ 1 — no sign from arg_sign
        # alone without bound comparison to 1; stay UNKNOWN.
        return Sign.UNKNOWN
    if name == "sqrt":
        if is_pos(arg_sign):
            return Sign.POS
        if is_nonneg(arg_sign):
            return Sign.NONNEG
        return Sign.UNKNOWN
    if name == "abs":
        return Sign.NONNEG
    if name == "cosh":
        return Sign.POS
    if name == "sinh":
        return arg_sign
    if name == "tanh":
        return arg_sign
    return Sign.UNKNOWN


def _classify_nary_max(expr: FunctionCall, model: Optional[Model], cache: dict) -> ExprInfo:
    """``max(a_1, ..., a_n)`` is convex when every argument is convex."""
    curv: Curvature = Curvature.AFFINE
    s = Sign.UNKNOWN
    for i, a in enumerate(expr.args):
        info = classify_expr_info(a, model, cache)
        # For max, convex / affine arguments compose to convex; any
        # concave or unknown argument kills the verdict.
        if info.curvature == Curvature.CONCAVE or info.curvature == Curvature.UNKNOWN:
            curv = Curvature.UNKNOWN
        elif curv != Curvature.UNKNOWN:
            if info.curvature == Curvature.CONVEX:
                curv = Curvature.CONVEX
            elif info.curvature == Curvature.AFFINE and curv == Curvature.AFFINE:
                curv = Curvature.AFFINE
        s = info.sign if i == 0 else _sign_join(s, info.sign)
    return ExprInfo(curv, s)


def _classify_nary_min(expr: FunctionCall, model: Optional[Model], cache: dict) -> ExprInfo:
    """``min(a_1, ..., a_n)`` is concave when every argument is concave."""
    curv: Curvature = Curvature.AFFINE
    s = Sign.UNKNOWN
    for i, a in enumerate(expr.args):
        info = classify_expr_info(a, model, cache)
        if info.curvature == Curvature.CONVEX or info.curvature == Curvature.UNKNOWN:
            curv = Curvature.UNKNOWN
        elif curv != Curvature.UNKNOWN:
            if info.curvature == Curvature.CONCAVE:
                curv = Curvature.CONCAVE
            elif info.curvature == Curvature.AFFINE and curv == Curvature.AFFINE:
                curv = Curvature.AFFINE
        s = info.sign if i == 0 else _sign_join(s, info.sign)
    return ExprInfo(curv, s)


def _sign_join(a: Sign, b: Sign) -> Sign:
    """Least-upper-bound in the sign lattice (loses information)."""
    if a == b:
        return a
    if is_nonneg(a) and is_nonneg(b):
        return Sign.NONNEG
    if is_nonpos(a) and is_nonpos(b):
        return Sign.NONPOS
    return Sign.UNKNOWN


# ──────────────────────────────────────────────────────────────────────
# Matrix multiplication
# ──────────────────────────────────────────────────────────────────────


def _classify_matmul(expr: MatMulExpression, model: Optional[Model], cache: dict) -> ExprInfo:
    left = classify_expr_info(expr.left, model, cache)
    right = classify_expr_info(expr.right, model, cache)
    if isinstance(expr.left, (Constant, Parameter)):
        return ExprInfo(right.curvature, Sign.UNKNOWN)
    if isinstance(expr.right, (Constant, Parameter)):
        return ExprInfo(left.curvature, Sign.UNKNOWN)
    return ExprInfo(Curvature.UNKNOWN, Sign.UNKNOWN)


# ──────────────────────────────────────────────────────────────────────
# Constraint- and model-level classification
# ──────────────────────────────────────────────────────────────────────


def classify_constraint(
    constraint: Constraint,
    model: Optional[Model] = None,
    _cache: Optional[dict] = None,
    *,
    use_certificate: bool = False,
) -> bool:
    """Return True when ``constraint`` defines a convex feasible set.

    When ``use_certificate`` is set and the syntactic walker fails to
    prove convexity, :func:`~.certificate.certify_convex` is consulted
    as a sound numerical fallback on the root variable box. The
    certificate never contradicts a syntactic CONVEX/CONCAVE verdict —
    it only tightens UNKNOWN cases.
    """
    if _cache is None:
        _cache = {}

    curv = classify_expr(constraint.body, model, _cache)

    syntactic = _constraint_convex_from_curvature(curv, constraint.sense)
    if syntactic or not use_certificate or model is None:
        if syntactic:
            return syntactic
        special = _classify_fractional_epigraph_constraint(constraint, model)
        if special is not None:
            return special
        return syntactic

    special = _classify_fractional_epigraph_constraint(constraint, model)
    if special is not None:
        return special

    # Fall back to the sound numerical certificate.
    try:
        from .certificate import certify_convex

        cert = certify_convex(constraint.body, model)
    except Exception:
        return syntactic
    if cert is None:
        return syntactic
    return _constraint_convex_from_curvature(cert, constraint.sense)


def _constraint_convex_from_curvature(curv: Curvature, sense: str) -> bool:
    """Decide constraint convexity given a body curvature and sense."""
    if sense == "<=":
        return curv in (Curvature.CONVEX, Curvature.AFFINE)
    if sense == ">=":
        return curv in (Curvature.CONCAVE, Curvature.AFFINE)
    if sense == "==":
        return curv == Curvature.AFFINE
    return False


def classify_model(model: Model, *, use_certificate: bool = False) -> tuple[bool, list[bool]]:
    """Classify a model's convexity.

    Returns ``(is_convex, per_constraint_mask)``. ``max f`` is treated
    as ``min -f``, so a maximization objective is "convex" (in the
    global sense — the overall problem is convex) when its body is
    concave or affine.

    When ``use_certificate`` is set, the sound interval-Hessian
    certificate (:func:`~.certificate.certify_convex`) is consulted
    whenever the syntactic walker leaves a constraint or the objective
    unproven. The certificate only tightens UNKNOWN verdicts; it never
    overrides an already-proven CONVEX/CONCAVE, preserving the
    soundness invariant.
    """
    cache: dict = {}
    ctx = build_linear_context(model)
    if ctx is not None:
        cache[_LINEAR_CONTEXT_KEY] = ctx

    obj_convex = True
    if model._objective is not None:
        from discopt.modeling.core import ObjectiveSense

        obj_curv = classify_expr(model._objective.expression, model, cache)
        if model._objective.sense == ObjectiveSense.MINIMIZE:
            obj_convex = obj_curv in (Curvature.CONVEX, Curvature.AFFINE)
            need_curv_for_obj = Curvature.CONVEX
        else:
            obj_convex = obj_curv in (Curvature.CONCAVE, Curvature.AFFINE)
            need_curv_for_obj = Curvature.CONCAVE

        if not obj_convex and use_certificate:
            try:
                from .certificate import certify_convex

                cert = certify_convex(model._objective.expression, model)
            except Exception:
                cert = None
            if cert == need_curv_for_obj:
                obj_convex = True

    constraint_mask: list[bool] = []
    all_convex = obj_convex
    for c in model._constraints:
        if isinstance(c, Constraint):
            is_cvx = classify_constraint(c, model, cache, use_certificate=use_certificate)
            constraint_mask.append(is_cvx)
            if not is_cvx:
                all_convex = False
        else:
            constraint_mask.append(False)
            all_convex = False

    return all_convex, constraint_mask


__all__ = [
    "classify_expr",
    "classify_expr_info",
    "classify_constraint",
    "classify_model",
]
