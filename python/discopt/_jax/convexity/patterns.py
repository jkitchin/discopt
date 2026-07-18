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
* **Exp times reciprocal powers: ``exp(affine) * prod_i g_i(x)**a_i``**
  with each ``g_i`` affine, strictly positive, and ``a_i <= 0``
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
    MatMulExpression,
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


# Per-model cache of the declared-box arrays + scalar count (issue: qap-class
# convexity classification). ``_box_bounds`` / ``_total_scalar_variables`` are pure
# functions of the model's DECLARED variable bounds, yet the syntactic convexity
# recognizers call them once per product term (``_affine_strictly_positive`` ->
# ``_affine_lower_bound``) — 66k times on qap's 21,424-term objective, each an
# O(n_vars) rebuild (10s of pure recomputation). Declared bounds are static within a
# single classification pass, so we memoize the result on the model and invalidate it
# exactly where the solver resets ``_convexity_classification_cache`` (see
# :func:`clear_declared_box_cache`, called from ``solver.solve_model``), so a
# post-presolve re-classification with tightened bounds recomputes. Bound-neutral: the
# returned arrays/count are identical, only recomputation is removed.
_DECLARED_BOX_CACHE_ATTR = "_patterns_declared_box_cache"


def clear_declared_box_cache(model: Model) -> None:
    """Invalidate the per-model declared-box cache (call when declared bounds change).

    Idempotent and safe on a model that was never cached. The solver calls this at
    every point it resets its convexity-classification cache, so a re-classification
    after presolve/reformulation tightens the declared bounds sees fresh values.
    """
    try:
        if hasattr(model, _DECLARED_BOX_CACHE_ATTR):
            delattr(model, _DECLARED_BOX_CACHE_ATTR)
    except Exception:
        pass


def _total_scalar_variables(model: Model) -> int:
    cached = getattr(model, _DECLARED_BOX_CACHE_ATTR, None)
    if cached is not None:
        return int(cached[0])
    return sum(v.size for v in model._variables)


def _scalar_var_offset(model: Model, target: Variable) -> Optional[int]:
    offset = 0
    for v in model._variables:
        if v is target or v.name == target.name:
            return offset if v.size == 1 else None
        offset += v.size
    return None


def _var_offset(model: Model, target: Variable) -> Optional[int]:
    offset = 0
    for v in model._variables:
        if v is target or v.name == target.name:
            return offset
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


def _box_bounds(model: Model) -> tuple[np.ndarray, np.ndarray]:
    """Per-scalar-slot lower/upper bound vectors over all model variables.

    Memoized per model (see :data:`_DECLARED_BOX_CACHE_ATTR`): pure function of the
    DECLARED bounds, called once per objective term by the syntactic convexity
    recognizers. The cache stores ``(n_scalar, lo, hi)`` so ``_total_scalar_variables``
    shares it, and is invalidated by :func:`clear_declared_box_cache`.
    """
    cached = getattr(model, _DECLARED_BOX_CACHE_ATTR, None)
    if cached is not None:
        return cached[1], cached[2]
    los: list[np.ndarray] = []
    his: list[np.ndarray] = []
    for v in model._variables:
        lb = np.asarray(v.lb, dtype=np.float64).ravel()
        ub = np.asarray(v.ub, dtype=np.float64).ravel()
        if lb.size == 1 and v.size != 1:
            lb = np.full(v.size, float(lb.item()), dtype=np.float64)
        if ub.size == 1 and v.size != 1:
            ub = np.full(v.size, float(ub.item()), dtype=np.float64)
        los.append(lb)
        his.append(ub)
    if not los:
        lo_arr, hi_arr = np.zeros(0), np.zeros(0)
    else:
        lo_arr, hi_arr = np.concatenate(los), np.concatenate(his)
    try:
        setattr(model, _DECLARED_BOX_CACHE_ATTR, (int(lo_arr.size), lo_arr, hi_arr))
    except Exception:
        pass
    return lo_arr, hi_arr


def _affine_lower_bound(expr: Expression, model: Model) -> Optional[float]:
    """Rigorous lower bound of an affine ``expr`` over the declared box.

    Returns ``None`` when ``expr`` is not affine-extractable or the bound is
    not finite (an unbounded contributing variable). Unlike
    :func:`_has_positive_lower_bound` this evaluates the affine form against the
    box, so ``0.001 + 0.999 * x18`` with ``x18 in [0, 1]`` correctly yields
    ``0.001`` even though the ``x18`` term alone is only non-negative.
    """
    from discopt._jax.problem_classifier import _extract_linear_coefficients

    n = _total_scalar_variables(model)
    try:
        vec, const = _extract_linear_coefficients(expr, model, n)
    except Exception:
        return None
    vec = np.asarray(vec, dtype=np.float64).ravel()
    lo_box, hi_box = _box_bounds(model)
    if vec.size != lo_box.size:
        return None
    # min of c·x = Σ (c>0 ? c·lb : c·ub); guard infinities that actually matter.
    contrib = np.where(vec >= 0.0, vec * lo_box, vec * hi_box)
    if not np.all(np.isfinite(contrib)):
        return None
    return float(const) + float(np.sum(contrib))


def _affine_strictly_positive(expr: Expression, model: Model) -> bool:
    """True when an affine ``expr`` has a proven strictly-positive lower bound."""
    lb = _affine_lower_bound(expr, model)
    return lb is not None and lb > 1e-12


def _expr_struct_eq(a: Expression, b: Expression) -> bool:
    """Structural equality for the small affine shapes the recognizers match.

    ``from_nl`` reconstruction rebuilds shared subexpressions as distinct nodes,
    so a denominator appearing in several terms is not object-identical; this
    compares by structure (constants by value, variables by name).
    """
    if a is b:
        return True
    if isinstance(a, Constant) and isinstance(b, Constant):
        va = np.asarray(a.value)
        vb = np.asarray(b.value)
        return va.shape == vb.shape and bool(np.allclose(va, vb))
    if isinstance(a, Variable) and isinstance(b, Variable):
        return a.name == b.name
    if isinstance(a, IndexExpression) and isinstance(b, IndexExpression):
        return _expr_struct_eq(a.base, b.base) and a.index == b.index
    if isinstance(a, Parameter) and isinstance(b, Parameter):
        return a is b or a.name == b.name
    if isinstance(a, BinaryOp) and isinstance(b, BinaryOp):
        return (
            a.op == b.op and _expr_struct_eq(a.left, b.left) and _expr_struct_eq(a.right, b.right)
        )
    if isinstance(a, UnaryOp) and isinstance(b, UnaryOp):
        return a.op == b.op and _expr_struct_eq(a.operand, b.operand)
    return False


def _split_const_mul(expr: Expression) -> tuple[float, Optional[Expression]]:
    """Factor scalar constants out of a product: ``expr == const * core``.

    ``core`` is ``None`` when every factor is a scalar constant.
    """
    factors: list[Expression] = []
    _flatten_product(expr, factors)
    const = 1.0
    core_factors: list[Expression] = []
    for f in factors:
        if isinstance(f, (Constant, Parameter)):
            v = np.asarray(f.value)
            if v.ndim == 0:
                const *= float(v)
                continue
        core_factors.append(f)
    if not core_factors:
        return const, None
    core = core_factors[0]
    for f in core_factors[1:]:
        core = BinaryOp("*", core, f)
    return const, core


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


def _linear_vector_matrix(expr: Expression, model: Model) -> Optional[np.ndarray]:
    """Return A for vector affine form ``A @ x`` with no constant term."""
    n_total = _total_scalar_variables(model)
    if isinstance(expr, MatMulExpression):
        if isinstance(expr.left, Constant) and isinstance(expr.right, Variable):
            mat = np.asarray(expr.left.value, dtype=np.float64)
            var = expr.right
            offset = _var_offset(model, var)
            if offset is None:
                return None
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
            if mat.ndim != 2 or mat.shape[1] != var.size:
                return None
            out = np.zeros((mat.shape[0], n_total), dtype=np.float64)
            out[:, offset : offset + var.size] = mat.reshape(mat.shape[0], var.size)
            return out
        if isinstance(expr.left, Variable) and isinstance(expr.right, Constant):
            mat = np.asarray(expr.right.value, dtype=np.float64)
            var = expr.left
            offset = _var_offset(model, var)
            if offset is None:
                return None
            if mat.ndim == 1:
                mat = mat.reshape(-1, 1)
            if mat.ndim != 2 or mat.shape[0] != var.size:
                return None
            out = np.zeros((mat.shape[1], n_total), dtype=np.float64)
            out[:, offset : offset + var.size] = mat.T.reshape(mat.shape[1], var.size)
            return out
    if isinstance(expr, Variable):
        offset = _var_offset(model, expr)
        if offset is None:
            return None
        out = np.zeros((expr.size, n_total), dtype=np.float64)
        out[:, offset : offset + expr.size] = np.eye(expr.size, dtype=np.float64)
        return out
    return None


def _sum_of_squares_linear_matrix(expr: Expression, model: Model) -> Optional[np.ndarray]:
    """Return A for ``sum((A @ x) * (A @ x))``-style squared norms."""
    if not isinstance(expr, SumExpression):
        return None
    operand = expr.operand
    if not isinstance(operand, BinaryOp) or operand.op != "*":
        return None
    left = _linear_vector_matrix(operand.left, model)
    right = _linear_vector_matrix(operand.right, model)
    if left is None or right is None:
        return None
    if left.shape != right.shape or not np.allclose(left, right, atol=1e-12):
        return None
    return left


def _affine_square_sum_matrix(
    expr: Expression, model: Model
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return ``(A, b)`` when ``expr == sum_i (a_i . x + b_i)**2 == ||A x + b||^2``.

    Each summand of the top-level ``+`` tree must be a perfect square of a
    *scalar affine* form: either ``(affine)**2`` or ``affine * affine`` with
    both factors structurally identical. Rows of ``A`` are the affine
    coefficient vectors and ``b`` the affine constants. Returns ``None`` if the
    expression does not have this shape.

    Because ``sqrt`` of this is the Euclidean norm ``||A x + b||_2`` of an
    affine map, it is convex for *any* ``A`` and ``b`` — no PSD/eigenvalue
    check is needed (the sum-of-squares structure makes it PSD by
    construction, and a nonzero constant ``b`` is still convex). This is the
    affine-difference generalization of :func:`_sum_of_squares_linear_matrix`,
    which only matched homogeneous ``sum((A@x)*(A@x))`` MatMul forms.
    """
    from discopt._jax.problem_classifier import _extract_linear_coefficients

    n_total = _total_scalar_variables(model)

    terms: list[Expression] = []

    def _flatten_sum(node: Expression) -> None:
        if isinstance(node, BinaryOp) and node.op == "+":
            _flatten_sum(node.left)
            _flatten_sum(node.right)
        elif isinstance(node, SumExpression):
            _flatten_sum(node.operand)
        else:
            terms.append(node)

    _flatten_sum(expr)
    if not terms:
        return None

    rows: list[np.ndarray] = []
    consts: list[float] = []
    for term in terms:
        # Peel a nonnegative scalar coefficient: ``c * s**2`` with ``c >= 0`` is the
        # square ``(sqrt(c) * s)**2``, so a *weighted* sum of affine squares is still
        # ``||A x + b||^2`` (absorb ``sqrt(c)`` into the affine row). A negative
        # coefficient would break the sum-of-squares/PSD structure, so it abstains.
        scale, core = _peel_nonneg_scale(term)
        if scale is None:
            return None
        # A bare nonnegative constant term ``c`` is the square ``(sqrt(c))**2`` of a
        # zero affine form — a valid constant row, so ``sqrt(sum affine^2 + c)`` is
        # still ``||A x + b||`` (e.g. smoothed/regularized distances).
        if isinstance(core, Constant) and core.value.ndim == 0:
            cval = scale * float(core.value)
            if cval < 0.0:
                return None
            rows.append(np.zeros(n_total, dtype=np.float64))
            consts.append(float(np.sqrt(cval)))
            continue
        base: Optional[Expression] = None
        if (
            isinstance(core, BinaryOp)
            and core.op == "**"
            and isinstance(core.right, Constant)
            and core.right.value.ndim == 0
            and abs(float(core.right.value) - 2.0) < 1e-12
        ):
            base = core.left
        elif isinstance(core, BinaryOp) and core.op == "*" and _same_expr(core.left, core.right):
            base = core.left
        if base is None:
            return None
        try:
            coeffs, const = _extract_linear_coefficients(base, model, n_total)
        except Exception:
            return None
        root = float(np.sqrt(scale))
        rows.append(root * coeffs)
        consts.append(root * const)

    return np.asarray(rows, dtype=np.float64), np.asarray(consts, dtype=np.float64)


def _peel_nonneg_scale(term: Expression) -> tuple[Optional[float], Expression]:
    """Peel a product of nonnegative constant factors off ``term``.

    Returns ``(scale, core)`` where ``term == scale * core`` and ``scale >= 0``,
    or ``(None, term)`` if a constant factor is negative (which would invalidate
    the sum-of-squares structure the caller relies on). Constants may appear on
    either side and may be nested (``c1 * (c2 * s**2))``.
    """
    scale = 1.0
    node = term
    while isinstance(node, BinaryOp) and node.op == "*":
        if isinstance(node.left, Constant) and node.left.value.ndim == 0:
            c = float(node.left.value)
            if c < 0.0:
                return None, term
            scale *= c
            node = node.right
        elif isinstance(node.right, Constant) and node.right.value.ndim == 0:
            c = float(node.right.value)
            if c < 0.0:
                return None, term
            scale *= c
            node = node.left
        else:
            break
    return scale, node


def is_affine_norm_square(expr: Expression, model: Model) -> bool:
    """True when ``expr == ||A x + b||_2^2`` for some affine map ``A x + b``.

    Recognises a sum of squares of scalar affine forms (see
    :func:`_affine_square_sum_matrix`). ``sqrt`` of such an expression is a
    convex Euclidean norm regardless of the constant term, which covers
    ``sqrt((x0-x2)**2 + (x1-x3)**2)``-style neighbourhood/distance objectives
    (MINLPLib ``tspn*``) that the homogeneous quadratic recognizer misses.
    """
    return _affine_square_sum_matrix(expr, model) is not None


def is_homogeneous_psd_quadratic(expr: Expression, model: Model) -> bool:
    """True when ``expr`` is ``x^T Q x`` (no linear/constant term) with Q PSD."""
    mat = _sum_of_squares_linear_matrix(expr, model)
    if mat is not None:
        q = mat.T @ mat
        eigvals = np.linalg.eigvalsh(q)
        return bool(float(np.min(eigvals)) >= -1e-10)
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
    * ``exp(affine) * prod_i g_i(x)**a_i`` where every ``g_i`` is affine and
      strictly positive and every ``a_i <= 0``: CONVEX because it is
      ``exp(affine + Σ a_i log(g_i(x)))`` with a convex exponent.
    * **Signomial monomial** ``c * prod_i base_i ** a_i`` with every base
      affine, classified by exponent sign pattern on the positive orthant
      (Boyd & Vandenberghe §3.1.5):

        - all ``a_i >= 0`` and ``sum a_i <= 1`` with nonneg bases — a
          generalized weighted geometric mean: CONCAVE.
        - all ``a_i <= 0`` with strictly-positive bases: CONVEX.
        - exactly one ``a_k >= 1``, every other ``a_i <= 0``, ``sum a_i >= 1``,
          strictly-positive bases: CONVEX.

      A negative leading constant ``c`` flips CONVEX <-> CONCAVE.
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

    # Peel the leading scalar constant. The remaining product is inspected by
    # the exp-times-reciprocal-power and signomial recognizers below. A negative
    # constant flips the curvature.
    const, core = _split_const_mul(expr)
    if core is None or abs(const) <= 1e-12:
        return None
    factors: list[Expression] = []
    _flatten_product(core, factors)
    if len(factors) < 2:
        return None

    exp_factor_count = 0
    power_factors: list[tuple[Expression, float]] = []
    exp_power_candidate = True
    for factor in factors:
        if (
            isinstance(factor, FunctionCall)
            and factor.func_name == "exp"
            and len(factor.args) == 1
            and classify_expr(factor.args[0], model, cache) == Curvature.AFFINE
        ):
            exp_factor_count += 1
            continue
        extracted = _extract_power_factor(factor)
        if extracted is None:
            exp_power_candidate = False
            break
        base, exponent = extracted
        if classify_expr(base, model, cache) != Curvature.AFFINE:
            exp_power_candidate = False
            break
        power_factors.append((base, exponent))

    tol = 1e-10
    if (
        exp_power_candidate
        and exp_factor_count >= 1
        and all(exp <= tol for _base, exp in power_factors)
        and all(_affine_strictly_positive(base, model) for base, _exp in power_factors)
    ):
        return Curvature.CONCAVE if const < 0 else Curvature.CONVEX

    parsed: list[tuple[Expression, float]] = []
    for factor in factors:
        extracted = _extract_power_factor(factor)
        if extracted is None:
            return None
        base, exponent = extracted
        if classify_expr(base, model, cache) != Curvature.AFFINE:
            return None
        parsed.append((base, exponent))

    exps = [exp for _, exp in parsed]
    bases = [base for base, _ in parsed]
    total = sum(exps)

    base_curv: Optional[Curvature] = None
    if (
        all(exp >= -tol for exp in exps)
        and total <= 1.0 + tol
        and all(_is_nonneg_domain(base, model) for base in bases)
    ):
        # Generalized weighted geometric mean — concave on the nonneg orthant.
        base_curv = Curvature.CONCAVE
    elif all(_affine_strictly_positive(base, model) for base in bases):
        if all(exp <= tol for exp in exps):
            # All exponents non-positive — convex on the positive orthant.
            base_curv = Curvature.CONVEX
        else:
            big = [exp for exp in exps if exp >= 1.0 - tol]
            rest_nonpos = all(exp <= tol for exp in exps if exp < 1.0 - tol)
            if len(big) == 1 and rest_nonpos and total >= 1.0 - tol:
                # One exponent >= 1, the rest <= 0, total >= 1 — convex.
                base_curv = Curvature.CONVEX

    if base_curv is None:
        return None
    if const < 0:
        return Curvature.CONCAVE if base_curv == Curvature.CONVEX else Curvature.CONVEX
    return base_curv


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
    classify_expr=None,  # noqa: ANN001 — forward reference avoids circular import
    cache: Optional[dict] = None,
) -> Optional[Curvature]:
    """Return curvature for recognised ``sqrt(arg)`` shapes, else None.

    * ``sqrt(x^T Q x)`` with Q PSD — Euclidean-style norm: CONVEX (even though
      the inner quadratic is itself convex, ``concave ∘ convex`` would fail DCP
      composition, so the norm is recognised directly).
    * ``sqrt(prod_i base_i ** p_i)`` with every ``base_i`` affine and nonneg on
      the box, ``p_i >= 0`` and ``sum_i p_i <= 2`` — a weighted geometric mean
      ``prod_i base_i ** (p_i / 2)`` whose halved exponents are nonneg and sum
      to ``<= 1``: CONCAVE. Covers ``sqrt(x_i * x_j)`` geometric-mean concavity
      (MINLPLib ``tls*``). Requires ``classify_expr`` / ``cache`` to verify the
      bases are affine; skipped when they are not supplied.
    """
    if is_homogeneous_psd_quadratic(arg, model):
        return Curvature.CONVEX

    # Euclidean norm of an affine map: sqrt(sum_i (a_i.x + b_i)^2) = ||A x + b||,
    # convex for any affine map (the homogeneous PSD recognizer above only
    # catches the b == 0 / MatMul cases).
    if is_affine_norm_square(arg, model):
        return Curvature.CONVEX

    # Geometric-mean concavity: sqrt of a product of nonneg affine powers.
    # sqrt(prod base_i^{p_i}) = prod base_i^{p_i/2}; with weights w_i = p_i/2
    # >= 0 and sum w_i <= 1 over affine nonneg bases, the product of powers is
    # concave on the nonneg orthant (Boyd & Vandenberghe §3.1.5).
    if classify_expr is not None:
        factors: list[Expression] = []
        _flatten_product(arg, factors)
        if len(factors) >= 1:
            total = 0.0
            ok = True
            for factor in factors:
                extracted = _extract_power_factor(factor)
                if extracted is None:
                    ok = False
                    break
                base, exponent = extracted
                if exponent < -1e-10:
                    ok = False
                    break
                if classify_expr(base, model, cache) != Curvature.AFFINE:
                    ok = False
                    break
                if not _is_nonneg_domain(base, model):
                    ok = False
                    break
                total += exponent
            if ok and total > 1e-10 and total <= 2.0 + 1e-10:
                return Curvature.CONCAVE
    return None


def classify_perspective_product(
    expr: BinaryOp,
    model: Model,
    classify_expr,  # noqa: ANN001 — forward reference avoids circular import
    cache: dict,
) -> Optional[Curvature]:
    """Return CONVEX for a perspective product ``P * L`` with ``L`` affine > 0.

    Recognises the perspective reformulation ``g(x)^2 / L`` written in the
    expanded arrangement ``((g / L) ** 2) * L`` together with its affine
    companions ``(h / L) * L`` and ``const * L``. With ``L`` affine and
    strictly positive on the box, each squared term ``(g / L) ** 2`` (``g``
    affine) is a quadratic-over-affine; multiplying the whole sum by ``L``
    distributes and collapses ``(g / L)^2 * L = g^2 / L`` (convex
    quad-over-linear), ``(h / L) * L = h`` (affine), and ``const * L`` (affine).
    A nonneg-weighted sum of convex + affine terms is convex.

    Covers the MINLPLib ``clay*hfsg`` perspective layout, which the
    quadratic-over-affine recogniser misses (the ``* L`` keeps the node a
    product) and whose box interval Hessian blows up as ``L`` approaches its
    lower bound, defeating the Gershgorin certificate.

    Requires at least one genuine squared (quad-over-affine) term — a product
    whose terms all collapse to affine is left to the generic walker.
    """
    if not isinstance(expr, BinaryOp) or expr.op != "*":
        return None

    for p_expr, l_expr in ((expr.left, expr.right), (expr.right, expr.left)):
        if classify_expr(l_expr, model, cache) != Curvature.AFFINE:
            continue
        if not _affine_strictly_positive(l_expr, model):
            continue

        terms: list[tuple[float, Expression]] = []
        _flatten_sum_terms(p_expr, 1.0, terms)
        if not terms:
            continue

        all_ok = True
        saw_square = False
        for scale_i, term in terms:
            const, core = _split_const_mul(term)
            if core is None:
                # const * L → affine, any sign acceptable.
                continue
            # Affine companion: (h / L) * L = h, with h affine, denominator ≡ L.
            if (
                isinstance(core, BinaryOp)
                and core.op == "/"
                and _expr_struct_eq(core.right, l_expr)
                and classify_expr(core.left, model, cache) == Curvature.AFFINE
            ):
                continue
            # Square term: (g / L) ** 2, g affine, denominator ≡ L; the squared
            # quad-over-affine is convex only with a nonneg effective weight.
            if (
                isinstance(core, BinaryOp)
                and core.op == "**"
                and isinstance(core.right, (Constant, Parameter))
            ):
                pwr = np.asarray(core.right.value)
                inner = core.left
                if (
                    pwr.ndim == 0
                    and abs(float(pwr) - 2.0) <= 1e-10
                    and isinstance(inner, BinaryOp)
                    and inner.op == "/"
                    and _expr_struct_eq(inner.right, l_expr)
                    and classify_expr(inner.left, model, cache) == Curvature.AFFINE
                ):
                    if scale_i * const >= -1e-10:
                        saw_square = True
                        continue
            all_ok = False
            break

        if all_ok and saw_square:
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
            # coeff(x) * y + r(x) <= 0 with coeff > 0 ⇒ y <= -r(x) / coeff, a
            # hypograph { y <= -q(x)/L(x) }; it is convex iff -q/L is concave,
            # i.e. q/L is convex over the positive denominator L>0. That is the
            # SAME discriminant as the coeff_hi<0 branch: a e^2 - b d e + c d^2
            # >= 0. (The earlier `<= 1e-10` inverted this, classifying the
            # nonconvex hypograph of a convex ratio as convex — see #757.)
            return curvature_numerator >= -1e-10

    return None


__all__ = [
    "classify_division_pattern",
    "classify_fractional_epigraph_constraint",
    "classify_perspective_product",
    "classify_product_pattern",
    "classify_sqrt_pattern",
    "is_homogeneous_psd_quadratic",
    "quadratic_curvature",
]
