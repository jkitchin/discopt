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
        return _classify_product(expr, left, right)

    if expr.op == "/":
        return _classify_division(expr, left, right, model, cache)

    if expr.op == "**":
        return _classify_power(expr, left, model, cache)

    return ExprInfo(Curvature.UNKNOWN, Sign.UNKNOWN)


def _classify_product(expr: BinaryOp, left: ExprInfo, right: ExprInfo) -> ExprInfo:
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

    # Reciprocal with constant numerator and strictly-signed denominator.
    # 1/u is convex + nonincreasing on u>0; concave + nonincreasing on
    # u<0. The DCP composition rule combines this profile with the
    # inner expression's curvature — bypassing it (i.e., trusting the
    # sign alone) would be UNSOUND: e.g., 1/(1 + exp(-x)) has positive
    # denominator but convex inner, so the composite is neither convex
    # nor concave.
    if _is_scalar_const(expr.left) and is_strict(right.sign):
        c = _scalar_value(expr.left)
        recip_curv = Curvature.CONVEX if is_pos(right.sign) else Curvature.CONCAVE
        recip_mono = Monotonicity.NONINC
        composed = compose(recip_curv, recip_mono, right.curvature)
        recip_sign = sign_reciprocal(right.sign)
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

    if len(expr.args) != 1:
        return ExprInfo(Curvature.UNKNOWN, Sign.UNKNOWN)

    arg_info = classify_expr_info(expr.args[0], model, cache)
    profile: Optional[AtomProfile] = unary_atom_profile(name, arg_info.sign)
    if profile is None:
        return ExprInfo(Curvature.UNKNOWN, _function_result_sign(name, arg_info.sign))

    curv = compose(profile.curvature, profile.monotonicity, arg_info.curvature)
    return ExprInfo(curv, _function_result_sign(name, arg_info.sign))


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
) -> bool:
    """Return True when ``constraint`` defines a convex feasible set."""
    if _cache is None:
        _cache = {}

    curv = classify_expr(constraint.body, model, _cache)

    if constraint.sense == "<=":
        return curv in (Curvature.CONVEX, Curvature.AFFINE)
    if constraint.sense == ">=":
        return curv in (Curvature.CONCAVE, Curvature.AFFINE)
    if constraint.sense == "==":
        return curv == Curvature.AFFINE
    return False


def classify_model(model: Model) -> tuple[bool, list[bool]]:
    """Classify a model's convexity.

    Returns ``(is_convex, per_constraint_mask)``. ``max f`` is treated
    as ``min -f``, so a maximization objective is "convex" (in the
    global sense — the overall problem is convex) when its body is
    concave or affine.
    """
    cache: dict = {}

    obj_convex = True
    if model._objective is not None:
        from discopt.modeling.core import ObjectiveSense

        obj_curv = classify_expr(model._objective.expression, model, cache)
        if model._objective.sense == ObjectiveSense.MINIMIZE:
            obj_convex = obj_curv in (Curvature.CONVEX, Curvature.AFFINE)
        else:
            obj_convex = obj_curv in (Curvature.CONCAVE, Curvature.AFFINE)

    constraint_mask: list[bool] = []
    all_convex = obj_convex
    for c in model._constraints:
        if isinstance(c, Constraint):
            is_cvx = classify_constraint(c, model, cache)
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
