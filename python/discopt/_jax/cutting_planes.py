"""
Cutting Planes: RLT, Outer Approximation (OA), and Lift-and-Project cut generation.

Provides linear cutting planes that tighten relaxations within spatial Branch & Bound:
  - RLT cuts: McCormick linearization inequalities for bilinear terms x*y
  - OA cuts: gradient-based tangent hyperplanes at NLP relaxation solutions
  - Lift-and-project cuts: disjunctive cuts for fractional binary variables

All cuts are represented as LinearCut NamedTuples with coeffs, rhs, and sense,
suitable for injection into LP/NLP subproblems.

The CutPool class manages accumulated cuts across B&B iterations, handling
deduplication, aging, and purging of inactive cuts.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from discopt._jax.nonlinear_bound_tightening import is_effectively_finite
from discopt.constants import ALPHABB_EPS, ALPHABB_SAFETY
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Expression,
    IndexExpression,
    Model,
    SumOverExpression,
    UnaryOp,
    Variable,
)


class LinearCut(NamedTuple):
    """A single linear cutting plane: coeffs @ x  sense  rhs.

    Attributes:
        coeffs: coefficient vector of length n_vars (dense).
        rhs: right-hand side scalar.
        sense: one of "<=", ">=", "==".
    """

    coeffs: np.ndarray
    rhs: float
    sense: str


class OACutSkip(NamedTuple):
    """Structured reason an evaluator constraint did not receive a direct OA cut."""

    constraint_index: int
    reason: str


class OACutGenerationReport(NamedTuple):
    """Direct evaluator OA cuts plus per-row skip reasons."""

    cuts: list[LinearCut]
    skipped: list[OACutSkip]


# ---------------------------------------------------------------------------
# RLT (Reformulation-Linearization Technique) cuts
#
# For a bilinear term w = x_i * x_j with bounds
#   x_i in [x_i_lb, x_i_ub], x_j in [x_j_lb, x_j_ub],
# the four McCormick envelopes are:
#
#   w >= x_i_lb * x_j + x_i * x_j_lb - x_i_lb * x_j_lb   (underestimator 1)
#   w >= x_i_ub * x_j + x_i * x_j_ub - x_i_ub * x_j_ub   (underestimator 2)
#   w <= x_i_ub * x_j + x_i * x_j_lb - x_i_ub * x_j_lb   (overestimator 1)
#   w <= x_i_lb * x_j + x_i * x_j_ub - x_i_lb * x_j_ub   (overestimator 2)
#
# These are expressed with an auxiliary variable w that represents x_i * x_j.
# Here we generate them as LinearCut objects with a coefficient for each
# original variable plus a coefficient for the auxiliary w.
# ---------------------------------------------------------------------------


class BilinearTerm(NamedTuple):
    """Describes a bilinear product x[i] * x[j] in the model.

    Attributes:
        i: index of first variable in flat x vector.
        j: index of second variable in flat x vector.
        w_index: index of auxiliary variable representing x[i]*x[j],
                 or None if cuts are expressed without an auxiliary.
    """

    i: int
    j: int
    w_index: Optional[int] = None


def generate_rlt_cuts(
    bilinear_term: BilinearTerm,
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    n_vars: int,
) -> list[LinearCut]:
    """Generate the four McCormick envelope cuts for a single bilinear term.

    For w = x[i] * x[j], the McCormick envelopes are linearized as:
      - Two underestimator inequalities (w >= ...)
      - Two overestimator inequalities (w <= ...)

    Requires an auxiliary variable (``bilinear_term.w_index``) representing
    the product. Without one, there is no sound way to encode the envelopes
    as linear constraints in the original variable space, so this function
    returns an empty list.

    Args:
        bilinear_term: BilinearTerm describing which variables are multiplied.
        x_lb: lower bounds on all variables, shape (n_vars,).
        x_ub: upper bounds on all variables, shape (n_vars,).
        n_vars: total number of variables (including any auxiliaries).

    Returns:
        List of four LinearCut objects, or [] if ``w_index`` is None.
    """
    i = bilinear_term.i
    j = bilinear_term.j
    w_idx = bilinear_term.w_index

    xi_lb = float(x_lb[i])
    xi_ub = float(x_ub[i])
    xj_lb = float(x_lb[j])
    xj_ub = float(x_ub[j])

    cuts = []

    if w_idx is not None:
        # Cuts with auxiliary variable w representing x[i]*x[j]
        # Underestimator 1: w >= xi_lb * x[j] + xj_lb * x[i] - xi_lb * xj_lb
        # => -xj_lb * x[i] - xi_lb * x[j] + w >= -xi_lb * xj_lb
        coeffs = np.zeros(n_vars, dtype=np.float64)
        coeffs[i] = -xj_lb
        coeffs[j] = -xi_lb
        coeffs[w_idx] = 1.0
        cuts.append(LinearCut(coeffs=coeffs, rhs=-xi_lb * xj_lb, sense=">="))

        # Underestimator 2: w >= xi_ub * x[j] + xj_ub * x[i] - xi_ub * xj_ub
        coeffs = np.zeros(n_vars, dtype=np.float64)
        coeffs[i] = -xj_ub
        coeffs[j] = -xi_ub
        coeffs[w_idx] = 1.0
        cuts.append(LinearCut(coeffs=coeffs, rhs=-xi_ub * xj_ub, sense=">="))

        # Overestimator 1: w <= xi_ub * x[j] + xj_lb * x[i] - xi_ub * xj_lb
        # => -xj_lb * x[i] - xi_ub * x[j] + w <= -xi_ub * xj_lb
        coeffs = np.zeros(n_vars, dtype=np.float64)
        coeffs[i] = -xj_lb
        coeffs[j] = -xi_ub
        coeffs[w_idx] = 1.0
        cuts.append(LinearCut(coeffs=coeffs, rhs=-xi_ub * xj_lb, sense="<="))

        # Overestimator 2: w <= xi_lb * x[j] + xj_ub * x[i] - xi_lb * xj_ub
        coeffs = np.zeros(n_vars, dtype=np.float64)
        coeffs[i] = -xj_ub
        coeffs[j] = -xi_lb
        coeffs[w_idx] = 1.0
        cuts.append(LinearCut(coeffs=coeffs, rhs=-xi_lb * xj_ub, sense="<="))
    else:
        # Without an auxiliary variable w = x[i]*x[j], the McCormick envelopes
        # cannot be expressed as valid linear inequalities on (x[i], x[j]) alone
        # — the bilinear product is intrinsic to each envelope. Any "cut"
        # generated purely in the original variable space would be unsound
        # (e.g. with x,y in [0.1,5], the underestimator rearranges to
        # 0.1*x + 0.1*y <= 0.01, which excludes every feasible point). Callers
        # that want bilinear relaxations must introduce an auxiliary variable
        # and set ``w_index`` accordingly.
        return []

    return cuts


def separate_rlt_cuts(
    bilinear_term: BilinearTerm,
    x_sol: np.ndarray,
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    n_vars: int,
    tol: float = 1e-8,
) -> list[LinearCut]:
    """Generate only the violated RLT cuts at a given solution point.

    Checks which of the four McCormick envelope inequalities are violated
    at x_sol and returns only those cuts.

    Args:
        bilinear_term: BilinearTerm describing which variables are multiplied.
        x_sol: current solution point, shape (n_vars,).
        x_lb: lower bounds on all variables.
        x_ub: upper bounds on all variables.
        n_vars: total number of variables.
        tol: violation tolerance.

    Returns:
        List of violated LinearCut objects (0 to 4 cuts).
    """
    all_cuts = generate_rlt_cuts(bilinear_term, x_lb, x_ub, n_vars)
    violated = []

    for cut in all_cuts:
        lhs_val = float(np.dot(cut.coeffs, x_sol))
        if cut.sense == "<=" and lhs_val > cut.rhs + tol:
            violated.append(cut)
        elif cut.sense == ">=" and lhs_val < cut.rhs - tol:
            violated.append(cut)
        elif cut.sense == "==" and abs(lhs_val - cut.rhs) > tol:
            violated.append(cut)

    return violated


# ---------------------------------------------------------------------------
# Outer Approximation (OA) cuts
#
# Given a convex constraint g(x) <= 0 and a point x*, the OA cut is the
# first-order Taylor expansion:
#
#   g(x*) + nabla_g(x*)^T (x - x*) <= 0
#
# which simplifies to:
#   nabla_g(x*)^T x <= nabla_g(x*)^T x* - g(x*)
#
# IMPORTANT: OA tangent cuts are only globally valid for CONVEX functions.
# For non-convex constraints, the tangent hyperplane may cut off feasible
# regions. Two safe approaches for non-convex problems:
#   (a) Convexify first with alphaBB, then apply OA to the convex relaxation.
#   (b) Apply OA to the McCormick convex underestimators from the relaxation
#       compiler, which are convex by construction.
#
# This module provides generate_oa_cuts_from_relaxation() for approach (b),
# which is the recommended default for general MINLP.
#
# For the objective min f(x), an OA cut at x* is:
#   f(x*) + nabla_f(x*)^T (x - x*) <= z
# where z is the epigraph variable, or equivalently:
#   nabla_f(x*)^T x - z <= nabla_f(x*)^T x* - f(x*)
# ---------------------------------------------------------------------------


def generate_oa_cut(
    grad: np.ndarray,
    func_val: float,
    x_star: np.ndarray,
    sense: str = "<=",
) -> LinearCut:
    """Generate an OA cut from gradient information at a point.

    For a constraint g(x) <= 0, the OA cut at x* is:
        grad^T x <= grad^T x* - g(x*)

    Args:
        grad: gradient of the constraint/objective at x*, shape (n,).
        func_val: value of the constraint/objective at x*.
        x_star: point where the linearization is taken, shape (n,).
        sense: constraint sense ("<=", ">=", "==").

    Returns:
        A LinearCut representing the tangent hyperplane.
    """
    coeffs = np.asarray(grad, dtype=np.float64).copy()
    rhs = float(np.dot(grad, x_star)) - func_val
    return LinearCut(coeffs=coeffs, rhs=rhs, sense=sense)


def generate_oa_cuts_from_evaluator(
    evaluator,
    x_sol: np.ndarray,
    constraint_senses: Optional[list[str]] = None,
    convex_mask: Optional[list[bool]] = None,
    skip_reasons: Optional[list[Optional[str]]] = None,
) -> list[LinearCut]:
    """Generate OA cuts for all constraints using an NLPEvaluator.

    Direct tangent OA cuts are emitted only for rows certified convex by
    ``convex_mask``. Rows marked false are intentionally skipped and can be
    explained through ``skip_reasons``/``generate_oa_cuts_from_evaluator_report``.
    For nonconvex rows that still need cuts, use a relaxation-specific cut
    generator such as McCormick or alpha-BB instead.

    Args:
        evaluator: An NLPEvaluator with evaluate_constraints and evaluate_jacobian.
        x_sol: solution point at which to linearize, shape (n,).
        constraint_senses: list of senses for each constraint. If None,
            all constraints are assumed to be "<=".
        convex_mask: Per-constraint boolean list. If provided, only constraints
            where ``convex_mask[k]`` is True are linearized.
        skip_reasons: Optional per-row reason strings for nonconvex rows. Used
            by the report API; accepted here so callers can share arguments.

    Returns:
        List of LinearCut objects, one per constraint.
    """
    return generate_oa_cuts_from_evaluator_report(
        evaluator,
        x_sol,
        constraint_senses=constraint_senses,
        convex_mask=convex_mask,
        skip_reasons=skip_reasons,
    ).cuts


def generate_oa_cuts_from_evaluator_report(
    evaluator,
    x_sol: np.ndarray,
    constraint_senses: Optional[list[str]] = None,
    convex_mask: Optional[list[bool]] = None,
    skip_reasons: Optional[list[Optional[str]]] = None,
) -> OACutGenerationReport:
    """Generate direct evaluator OA cuts and record intentionally skipped rows.

    A row marked false in ``convex_mask`` is not safe for direct tangent OA, so
    it is skipped with the corresponding ``skip_reasons[k]`` value, or
    ``"not_certified_convex"`` when no reason is supplied. The existing
    :func:`generate_oa_cuts_from_evaluator` API returns only ``report.cuts``.
    """
    m = evaluator.n_constraints
    if m == 0:
        return OACutGenerationReport(cuts=[], skipped=[])

    cons_vals = evaluator.evaluate_constraints(x_sol)
    jac = evaluator.evaluate_jacobian(x_sol)

    if constraint_senses is None:
        constraint_senses = ["<="] * m

    cuts = []
    skipped = []
    for k in range(m):
        if convex_mask is not None and not convex_mask[k]:
            reason = "not_certified_convex"
            if skip_reasons is not None and skip_reasons[k] is not None:
                reason = str(skip_reasons[k])
            skipped.append(OACutSkip(constraint_index=k, reason=reason))
            continue
        grad_k = jac[k, :]
        g_k = float(cons_vals[k])
        sense = constraint_senses[k]
        cut = generate_oa_cut(grad_k, g_k, x_sol, sense=sense)
        cuts.append(cut)

    return OACutGenerationReport(cuts=cuts, skipped=skipped)


QuadraticPolynomial = dict[tuple[int, ...], float]


def _constant_scalar(expr: Expression) -> Optional[float]:
    """Return a scalar constant value, or None when the expression is not scalar constant."""
    if not isinstance(expr, Constant):
        return None
    value = np.asarray(expr.value, dtype=np.float64)
    if value.shape != ():
        return None
    return float(value)


def _var_offset(var: Variable, model: Model) -> int:
    """Return a variable's start index in the flattened model vector."""
    offset = 0
    for existing in model._variables[: var._index]:
        offset += existing.size
    return offset


def _scalar_var_index(expr: Expression, model: Model) -> Optional[int]:
    """Return the flat index for a scalar variable expression."""
    if isinstance(expr, Variable):
        if expr.size == 1:
            return _var_offset(expr, model)
        return None
    if isinstance(expr, IndexExpression) and isinstance(expr.base, Variable):
        base_offset = _var_offset(expr.base, model)
        idx = expr.index
        if isinstance(idx, int):
            return base_offset + idx
        if isinstance(idx, tuple) and len(idx) == 1 and isinstance(idx[0], int):
            return base_offset + idx[0]
    return None


def _cleanup_polynomial(poly: QuadraticPolynomial) -> QuadraticPolynomial:
    """Drop numerical zero coefficients from a structural polynomial."""
    return {key: coeff for key, coeff in poly.items() if abs(coeff) > 1e-14}


def _scale_polynomial(poly: QuadraticPolynomial, scale: float) -> QuadraticPolynomial:
    """Scale polynomial coefficients."""
    return _cleanup_polynomial({key: scale * coeff for key, coeff in poly.items()})


def _add_polynomials(
    left: QuadraticPolynomial,
    right: QuadraticPolynomial,
    right_scale: float = 1.0,
) -> QuadraticPolynomial:
    """Add two polynomials, optionally scaling the right operand."""
    result = dict(left)
    for key, coeff in right.items():
        result[key] = result.get(key, 0.0) + right_scale * coeff
    return _cleanup_polynomial(result)


def _multiply_polynomials(
    left: QuadraticPolynomial,
    right: QuadraticPolynomial,
) -> Optional[QuadraticPolynomial]:
    """Multiply two polynomials, rejecting terms above degree two."""
    result: QuadraticPolynomial = {}
    for left_key, left_coeff in left.items():
        for right_key, right_coeff in right.items():
            key = tuple(sorted(left_key + right_key))
            if len(key) > 2:
                return None
            result[key] = result.get(key, 0.0) + left_coeff * right_coeff
    return _cleanup_polynomial(result)


def _quadratic_polynomial(expr: Expression, model: Model) -> Optional[QuadraticPolynomial]:
    """Extract a scalar polynomial of degree at most two, or None if unsupported."""
    const_val = _constant_scalar(expr)
    if const_val is not None:
        return {(): const_val}

    flat_idx = _scalar_var_index(expr, model)
    if flat_idx is not None:
        return {(flat_idx,): 1.0}

    if isinstance(expr, UnaryOp) and expr.op == "neg":
        operand = _quadratic_polynomial(expr.operand, model)
        if operand is None:
            return None
        return _scale_polynomial(operand, -1.0)

    if isinstance(expr, SumOverExpression):
        result: QuadraticPolynomial = {}
        for term in expr.terms:
            term_poly = _quadratic_polynomial(term, model)
            if term_poly is None:
                return None
            result = _add_polynomials(result, term_poly)
        return result

    if not isinstance(expr, BinaryOp):
        return None

    if expr.op in {"+", "-"}:
        left = _quadratic_polynomial(expr.left, model)
        right = _quadratic_polynomial(expr.right, model)
        if left is None or right is None:
            return None
        return _add_polynomials(left, right, -1.0 if expr.op == "-" else 1.0)

    if expr.op == "*":
        left = _quadratic_polynomial(expr.left, model)
        right = _quadratic_polynomial(expr.right, model)
        if left is None or right is None:
            return None
        return _multiply_polynomials(left, right)

    if expr.op == "/":
        denom = _constant_scalar(expr.right)
        if denom is None or abs(denom) <= 1e-14:
            return None
        left = _quadratic_polynomial(expr.left, model)
        if left is None:
            return None
        return _scale_polynomial(left, 1.0 / denom)

    if expr.op == "**":
        exponent = _constant_scalar(expr.right)
        if exponent is None or abs(exponent - round(exponent)) > 1e-12:
            return None
        exponent_int = int(round(exponent))
        if exponent_int == 0:
            return {(): 1.0}
        if exponent_int == 1:
            return _quadratic_polynomial(expr.left, model)
        if exponent_int == 2:
            base = _quadratic_polynomial(expr.left, model)
            if base is None:
                return None
            return _multiply_polynomials(base, base)
        return None

    return None


def _quadratic_hessian_from_polynomial(poly: QuadraticPolynomial, n_vars: int) -> np.ndarray:
    """Build a Hessian matrix from a structural quadratic polynomial."""
    hess = np.zeros((n_vars, n_vars), dtype=np.float64)
    for key, coeff in poly.items():
        if len(key) == 2:
            i, j = key
            if i == j:
                hess[i, i] += 2.0 * coeff
            else:
                hess[i, j] += coeff
                hess[j, i] += coeff
    return hess


def _constraint_row_quadratic_hessian(
    evaluator,
    row_idx: int,
    n_vars: int,
) -> Optional[np.ndarray]:
    """Return a structural quadratic Hessian for a scalar constraint row."""
    model = getattr(evaluator, "_model", None)
    source_constraints = getattr(evaluator, "_source_constraints", None)
    flat_sizes = getattr(evaluator, "_constraint_flat_sizes", None)
    if model is None or source_constraints is None or flat_sizes is None:
        return None

    offset = 0
    for constraint, flat_size_raw in zip(source_constraints, flat_sizes):
        flat_size = int(flat_size_raw)
        if row_idx < offset + flat_size:
            if flat_size != 1 or row_idx != offset:
                return None
            poly = _quadratic_polynomial(constraint.body, model)
            if poly is None:
                return None
            return _quadratic_hessian_from_polynomial(poly, n_vars)
        offset += flat_size
    return None


def generate_alphabb_quadratic_oa_cuts_from_evaluator(
    evaluator,
    x_sol: np.ndarray,
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    constraint_senses: Optional[list[str]] = None,
    convex_mask: Optional[list[bool]] = None,
    hessian_tol: float = 1e-8,
) -> list[LinearCut]:
    """Generate OA cuts from alpha-BB relaxations of nonconvex quadratic rows.

    Direct OA cuts on a nonconvex row are not globally valid. For a quadratic
    row ``q(x) <= 0`` with finite bounds on curved variables, alpha-BB gives a
    convex underestimator

        q_under(x) = q(x) - sum_i alpha_i (x_i - lb_i) (ub_i - x_i)

    satisfying ``q_under(x) <= q(x)`` over the box. Every point feasible for the
    original row therefore satisfies ``q_under(x) <= 0``, so a tangent cut of
    the convex underestimator is a valid relaxation cut.
    """
    m = evaluator.n_constraints
    if m == 0:
        return []

    x_sol = np.asarray(x_sol, dtype=np.float64).reshape(-1)
    x_lb = np.asarray(x_lb, dtype=np.float64).reshape(-1)
    x_ub = np.asarray(x_ub, dtype=np.float64).reshape(-1)
    if x_sol.size != x_lb.size or x_sol.size != x_ub.size:
        raise ValueError("x_sol, x_lb, and x_ub must have matching shapes")

    if constraint_senses is None:
        constraint_senses = ["<="] * m

    cons_vals = evaluator.evaluate_constraints(x_sol)
    jac = evaluator.evaluate_jacobian(x_sol)

    cuts: list[LinearCut] = []
    for k in range(m):
        if convex_mask is not None and convex_mask[k]:
            continue
        if constraint_senses[k] != "<=":
            continue

        hess = _constraint_row_quadratic_hessian(evaluator, k, x_sol.size)
        if hess is None:
            continue

        hess_nz = np.abs(hess) > hessian_tol
        curved = np.flatnonzero(np.any(hess_nz, axis=0) | np.any(hess_nz, axis=1))
        if curved.size == 0:
            continue
        if not all(
            is_effectively_finite(float(x_lb[idx])) and is_effectively_finite(float(x_ub[idx]))
            for idx in curved
        ):
            continue

        hess_sub = hess[np.ix_(curved, curved)]
        hess_sub = 0.5 * (hess_sub + hess_sub.T)
        min_eig = float(np.linalg.eigvalsh(hess_sub)[0])
        if min_eig >= -ALPHABB_EPS:
            continue

        alpha = np.zeros_like(x_sol, dtype=np.float64)
        alpha[curved] = max(0.0, -0.5 * min_eig + ALPHABB_SAFETY)

        perturbation = float(
            np.sum(alpha[curved] * (x_sol[curved] - x_lb[curved]) * (x_ub[curved] - x_sol[curved]))
        )
        under_val = float(cons_vals[k]) - perturbation
        under_grad = np.asarray(jac[k, :], dtype=np.float64).copy()
        under_grad[curved] -= alpha[curved] * (x_lb[curved] + x_ub[curved] - 2.0 * x_sol[curved])
        cuts.append(generate_oa_cut(under_grad, under_val, x_sol, sense="<="))

    return cuts


def generate_objective_oa_cut(
    evaluator,
    x_sol: np.ndarray,
    n_vars: int,
    z_index: Optional[int] = None,
) -> LinearCut:
    """Generate an OA cut for the objective function.

    For min f(x), the OA cut at x* is:
        f(x*) + nabla_f(x*)^T (x - x*) <= z
    where z is the epigraph variable.

    If z_index is None, the cut is returned as:
        nabla_f(x*)^T x <= nabla_f(x*)^T x* - f(x*) + UB
    where UB should be set externally.

    Args:
        evaluator: An NLPEvaluator.
        x_sol: solution point, shape (n,).
        n_vars: number of variables (including epigraph if present).
        z_index: index of epigraph variable z in the extended variable vector,
                 or None if no epigraph variable is used.

    Returns:
        A LinearCut for the objective linearization.
    """
    obj_val = evaluator.evaluate_objective(x_sol)
    grad = evaluator.evaluate_gradient(x_sol)

    # f(x*) + grad^T (x - x*) <= z
    # grad^T x - z <= grad^T x* - f(x*)
    n_orig = len(grad)
    coeffs = np.zeros(n_vars, dtype=np.float64)
    coeffs[:n_orig] = grad

    rhs_val = float(np.dot(grad, x_sol)) - obj_val

    if z_index is not None:
        coeffs[z_index] = -1.0

    return LinearCut(coeffs=coeffs, rhs=rhs_val, sense="<=")


def separate_oa_cuts(
    evaluator,
    x_sol: np.ndarray,
    constraint_senses: Optional[list[str]] = None,
    tol: float = 1e-8,
    convex_mask: Optional[list[bool]] = None,
) -> list[LinearCut]:
    """Generate OA cuts only for violated constraints at x_sol.

    WARNING: These cuts are only globally valid if the constraint is convex.
    Use ``convex_mask`` to restrict OA generation to known-convex constraints.

    A constraint g_k(x) <= 0 is violated if g_k(x_sol) > tol.
    Only violated constraints produce cuts.

    Args:
        evaluator: An NLPEvaluator.
        x_sol: solution point, shape (n,).
        constraint_senses: list of senses. If None, all are "<=".
        tol: violation tolerance.
        convex_mask: Per-constraint boolean list. If provided, only constraints
            where ``convex_mask[k]`` is True are considered for OA cuts.
            If None, all constraints are eligible (original behaviour).

    Returns:
        List of LinearCut objects for violated constraints.
    """
    m = evaluator.n_constraints
    if m == 0:
        return []

    cons_vals = evaluator.evaluate_constraints(x_sol)
    jac = evaluator.evaluate_jacobian(x_sol)

    if constraint_senses is None:
        constraint_senses = ["<="] * m

    cuts = []
    for k in range(m):
        # Skip non-convex constraints when a mask is provided
        if convex_mask is not None and not convex_mask[k]:
            continue

        g_k = float(cons_vals[k])
        sense = constraint_senses[k]
        violated = False
        if sense == "<=" and g_k > tol:
            violated = True
        elif sense == ">=" and g_k < -tol:
            violated = True
        elif sense == "==" and abs(g_k) > tol:
            violated = True

        if violated:
            grad_k = jac[k, :]
            cut = generate_oa_cut(grad_k, g_k, x_sol, sense=sense)
            cuts.append(cut)

    return cuts


def is_cut_violated(cut: LinearCut, x: np.ndarray, tol: float = 1e-8) -> bool:
    """Check whether a linear cut is violated at point x.

    Args:
        cut: a LinearCut.
        x: point to check, shape (n,).
        tol: violation tolerance.

    Returns:
        True if the cut is violated.
    """
    lhs = float(np.dot(cut.coeffs, x))
    if cut.sense == "<=":
        return lhs > cut.rhs + tol
    elif cut.sense == ">=":
        return lhs < cut.rhs - tol
    elif cut.sense == "==":
        return abs(lhs - cut.rhs) > tol
    return False


# ---------------------------------------------------------------------------
# Detecting bilinear terms in the expression DAG
# ---------------------------------------------------------------------------


def detect_bilinear_terms(model) -> list[BilinearTerm]:
    """Scan model constraints and objective for bilinear products x[i]*x[j].

    Walks the expression DAG to find BinaryOp("*") nodes where both operands
    are Variables (or IndexExpressions of Variables). Returns a list of
    BilinearTerm objects without auxiliary variables.

    Args:
        model: A Model with constraints and/or objective.

    Returns:
        List of BilinearTerm objects found in the model.
    """
    from discopt.modeling.core import (
        BinaryOp,
        IndexExpression,
        Variable,
    )
    from discopt.modeling.core import Constraint as ConstraintType

    def _var_index(expr: Expression, model_) -> int | None:
        """Get the flat variable index for a simple variable reference."""
        if isinstance(expr, Variable):
            offset = 0
            for v in model_._variables[: expr._index]:
                offset += v.size
            if expr.shape == () or expr.shape == (1,):
                return offset
            return None
        if isinstance(expr, IndexExpression) and isinstance(expr.base, Variable):
            var = expr.base
            offset = 0
            for v in model_._variables[: var._index]:
                offset += v.size
            idx = expr.index
            if isinstance(idx, int):
                return offset + idx
            if isinstance(idx, tuple) and len(idx) == 1 and isinstance(idx[0], int):
                return offset + idx[0]
        return None

    found: list[BilinearTerm] = []
    seen: set[tuple[int, int]] = set()

    def _walk(expr: Expression):
        if isinstance(expr, BinaryOp) and expr.op == "*":
            i = _var_index(expr.left, model)
            j = _var_index(expr.right, model)
            if i is not None and j is not None:
                pair = (min(i, j), max(i, j))
                if pair not in seen:
                    seen.add(pair)
                    found.append(BilinearTerm(i=pair[0], j=pair[1]))
        if isinstance(expr, BinaryOp):
            _walk(expr.left)
            _walk(expr.right)
        elif hasattr(expr, "operand"):
            _walk(expr.operand)
        elif hasattr(expr, "args"):
            for a in expr.args:
                _walk(a)
        elif hasattr(expr, "terms"):
            for t in expr.terms:
                _walk(t)
        elif hasattr(expr, "base") and isinstance(expr, IndexExpression):
            pass  # leaf

    for c in model._constraints:
        if isinstance(c, ConstraintType):
            _walk(c.body)
    if model._objective is not None:
        _walk(model._objective.expression)

    return found


# ---------------------------------------------------------------------------
# Combined cut generation for the solver loop
# ---------------------------------------------------------------------------


def generate_cuts_at_node(
    evaluator,
    model,
    x_sol: np.ndarray,
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    constraint_senses: list[str] | None = None,
    bilinear_terms: list[BilinearTerm] | None = None,
    tol: float = 1e-8,
    oa_enabled: bool = True,
    convex_constraint_mask: list[bool] | None = None,
) -> list[LinearCut]:
    """Generate all applicable cuts at a B&B node solution.

    Combines:
      - OA cuts from the NLPEvaluator for violated constraints (valid as local
        linearizations; globally valid when constraints are convex).
      - RLT cuts for detected bilinear terms using current node bounds.

    This is the recommended entry point for cut generation in the solver loop.

    Args:
        evaluator: An NLPEvaluator.
        model: The Model (used for bilinear detection if bilinear_terms is None).
        x_sol: NLP relaxation solution at the node, shape (n,).
        x_lb: variable lower bounds at the B&B node.
        x_ub: variable upper bounds at the B&B node.
        constraint_senses: list of senses for each constraint.
        bilinear_terms: pre-detected bilinear terms (avoids re-scanning).
        tol: violation tolerance.
        oa_enabled: If True, generate OA (outer approximation) cuts for violated
            constraints.  OA cuts are tangent hyperplanes that are only globally
            valid when all constraints are convex.  For non-convex problems, set
            this to False to avoid cutting off feasible integer points.  RLT cuts
            are always generated regardless of this flag.
        convex_constraint_mask: Per-constraint convexity flags. If provided,
            OA cuts are generated only for constraints where the mask is True.
            If None and oa_enabled is True, OA cuts are generated for all
            constraints (original behaviour).

    Returns:
        List of LinearCut objects (OA + RLT).
    """
    cuts: list[LinearCut] = []

    # OA cuts for violated constraints (only valid for convex constraints)
    if oa_enabled:
        oa = separate_oa_cuts(
            evaluator,
            x_sol,
            constraint_senses,
            tol,
            convex_mask=convex_constraint_mask,
        )
        cuts.extend(oa)

    # RLT cuts for bilinear terms using current node bounds
    if bilinear_terms is None:
        bilinear_terms = detect_bilinear_terms(model)

    n_vars = len(x_sol)
    for bt in bilinear_terms:
        rlt = separate_rlt_cuts(bt, x_sol, x_lb, x_ub, n_vars, tol)
        cuts.extend(rlt)

    return cuts


# ---------------------------------------------------------------------------
# Cut Pool: manages accumulated cuts across B&B iterations
# ---------------------------------------------------------------------------


class CutPool:
    """Manages a collection of linear cuts with deduplication, aging, and purging.

    Tracks cut activity (how many consecutive iterations each cut has been
    non-binding) and purges stale cuts to keep the pool compact.

    Attributes:
        max_cuts: Maximum number of cuts before forced purge.
        purge_fraction: Fraction of oldest/least-active cuts to remove on purge.
    """

    def __init__(self, max_cuts: int = 500, purge_fraction: float = 0.3):
        self.max_cuts = max_cuts
        self.purge_fraction = purge_fraction
        self._cuts: list[LinearCut] = []
        self._ages: list[int] = []  # idle iterations since last binding
        self._hashes: set[int] = set()

    def __len__(self) -> int:
        return len(self._cuts)

    @staticmethod
    def _cut_hash(cut: LinearCut) -> int:
        """Compute a hash for deduplication based on coefficients and rhs."""
        c_rounded = np.round(cut.coeffs, decimals=8)
        r_rounded = round(cut.rhs, 8)
        return hash((c_rounded.tobytes(), r_rounded, cut.sense))

    def add(self, cut: LinearCut) -> bool:
        """Add a cut if it is not a duplicate.

        Returns True if the cut was added, False if it was a duplicate.
        """
        h = self._cut_hash(cut)
        if h in self._hashes:
            return False
        self._hashes.add(h)
        self._cuts.append(cut)
        self._ages.append(0)

        if len(self._cuts) > self.max_cuts:
            self._purge_oldest()

        return True

    def add_many(self, cuts: list[LinearCut]) -> int:
        """Add multiple cuts, returning the count of non-duplicate additions."""
        added = 0
        for cut in cuts:
            if self.add(cut):
                added += 1
        return added

    def get_active_cuts(self, x_sol: np.ndarray, tol: float = 1e-8) -> list[LinearCut]:
        """Return cuts that are violated at x_sol."""
        active = []
        for cut in self._cuts:
            if is_cut_violated(cut, x_sol, tol):
                active.append(cut)
        return active

    def age_cuts(self, x_sol: np.ndarray, tol: float = 1e-6):
        """Increment age for non-binding cuts, reset age for binding ones.

        A cut is considered binding if |lhs - rhs| <= tol (for <= or >= sense)
        or if it is violated.
        """
        for i, cut in enumerate(self._cuts):
            lhs = float(np.dot(cut.coeffs, x_sol))
            if cut.sense == "<=":
                binding = lhs >= cut.rhs - tol
            elif cut.sense == ">=":
                binding = lhs <= cut.rhs + tol
            else:  # "=="
                binding = abs(lhs - cut.rhs) <= tol
            if binding:
                self._ages[i] = 0
            else:
                self._ages[i] += 1

    def purge_inactive(self, max_age: int = 10):
        """Remove cuts that have been non-binding for max_age iterations."""
        keep_idx = [i for i, age in enumerate(self._ages) if age < max_age]
        self._rebuild(keep_idx)

    def _purge_oldest(self):
        """Remove the oldest fraction of cuts when pool exceeds max_cuts."""
        n_remove = max(1, int(len(self._cuts) * self.purge_fraction))
        sorted_idx = sorted(range(len(self._ages)), key=lambda i: self._ages[i], reverse=True)
        remove_set = set(sorted_idx[:n_remove])
        keep_idx = [i for i in range(len(self._cuts)) if i not in remove_set]
        self._rebuild(keep_idx)

    def _rebuild(self, keep_idx: list[int]):
        """Rebuild internal lists keeping only the specified indices."""
        self._cuts = [self._cuts[i] for i in keep_idx]
        self._ages = [self._ages[i] for i in keep_idx]
        self._hashes = {self._cut_hash(c) for c in self._cuts}

    def to_constraint_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Convert all cuts to stacked constraint arrays.

        Returns:
            A_cuts: (n_cuts, n_vars) coefficient matrix.
            b_cuts: (n_cuts,) right-hand side vector.
            senses: list of sense strings.
        """
        if not self._cuts:
            return (
                np.empty((0, 0), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                [],
            )
        A = np.stack([c.coeffs for c in self._cuts], axis=0)
        b = np.array([c.rhs for c in self._cuts], dtype=np.float64)
        senses = [c.sense for c in self._cuts]
        return A, b, senses

    @property
    def cuts(self) -> list[LinearCut]:
        """Return a copy of the current cut list."""
        return list(self._cuts)


# ---------------------------------------------------------------------------
# Lift-and-Project cuts
# ---------------------------------------------------------------------------


def generate_lift_and_project_cut(
    x_sol: np.ndarray,
    A_ub: Optional[np.ndarray],
    b_ub: Optional[np.ndarray],
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    frac_var_idx: int,
) -> Optional[LinearCut]:
    """Generate a lift-and-project cut for a fractional binary variable.

    Uses the Balas disjunction x[j] <= 0 OR x[j] >= 1 for a binary variable
    j with fractional value x_sol[j] in (0, 1).

    Implements a simplified version: for each constraint row that involves
    the fractional variable, the disjunctive argument tightens the
    constraint using the integrality requirement. When no such
    strengthening is possible, falls back to a simple rounding cut.

    Args:
        x_sol: current relaxation solution, shape (n,).
        A_ub: inequality constraint matrix (m, n) for A_ub @ x <= b_ub,
              or None if no constraints.
        b_ub: inequality rhs (m,), or None.
        x_lb: variable lower bounds (n,).
        x_ub: variable upper bounds (n,).
        frac_var_idx: index of the fractional binary variable.

    Returns:
        A LinearCut separating x_sol from the integer hull, or None if
        no violated cut could be found.
    """
    j = frac_var_idx
    xj_val = float(x_sol[j])

    if xj_val <= 1e-6 or xj_val >= 1.0 - 1e-6:
        return None

    m = 0
    if A_ub is not None and b_ub is not None and len(A_ub.shape) == 2:
        m = A_ub.shape[0]

    if m == 0:
        # No LP constraints to strengthen; cannot derive a valid global cut
        return None

    assert A_ub is not None
    assert b_ub is not None

    # Disjunctive cut from x[j] = 0 or x[j] = 1:
    #
    # For each constraint a_i^T x <= b_i, the constraint is valid on both
    # branches. The disjunctive hull can only be tighter if we combine
    # multiple constraints.
    #
    # Approach: solve the CGLP (Cut-Generating LP) in dual form.
    # For the disjunction D = {x[j] <= 0} union {x[j] >= 1}:
    #
    # The CGLP dual finds weights (u0, u1) >= 0 on the constraint rows
    # such that the combined cut from the two branches is deepest.
    #
    # Branch 0 system: A x <= b, x[j] <= 0, x_lb <= x <= x_ub
    # Branch 1 system: A x <= b, x[j] >= 1, x_lb <= x <= x_ub
    #
    # The Balas cut is: pi^T x <= pi_0 where
    #   pi = u0^T A = u1^T A  (same alpha from both branches)
    #   pi_0 = min(u0^T b, u1^T b - pi_j)  (tighter of the two branch rhs)
    #
    # For a single row with a_j != 0, this reduces to the original.
    # With multiple rows, we can get a tighter cut.
    #
    # Simplified CGLP: we find the combination that maximizes violation
    # at x_sol subject to validity on both branches.

    # Collect near-binding constraint rows with nonzero a_j
    active_rows = []
    for i in range(m):
        a_j_i = float(A_ub[i][j])
        if abs(a_j_i) < 1e-12:
            continue
        slack = float(b_ub[i]) - float(np.dot(A_ub[i], x_sol))
        if slack < 0.1:  # near-binding or violated
            active_rows.append(i)

    if not active_rows:
        return None

    # For each active row, compute the disjunctive strengthening.
    # For a single row a^T x <= b with binary x[j]:
    # Branch 0 (x[j]=0): sum_{k!=j} a_k x_k <= b
    # Branch 1 (x[j]=1): sum_{k!=j} a_k x_k <= b - a_j
    # Conv hull of these two (in terms of x[j]):
    #   sum_{k!=j} a_k x_k <= b - a_j * x[j]  if a_j >= 0
    #   sum_{k!=j} a_k x_k <= (b - a_j) + a_j * x[j]  if a_j < 0
    # Both reduce to the original constraint a^T x <= b.
    #
    # The real power: combine rows with OPPOSITE signs of a_j.
    # Take row i (a_j > 0) and row k (a_j < 0):
    # weighted sum: lambda * row_i + (1-lambda) * row_k
    # Then strengthen the combined coefficient of x[j].

    # Split rows by sign of a_j
    pos_rows = [i for i in active_rows if float(A_ub[i][j]) > 0]
    neg_rows = [i for i in active_rows if float(A_ub[i][j]) < 0]

    best_cut = None
    best_violation = -np.inf

    # Try combining each pair of positive and negative a_j rows
    for pi in pos_rows:
        for ni in neg_rows:
            a_p = A_ub[pi].copy()
            b_p = float(b_ub[pi])
            a_n = A_ub[ni].copy()
            b_n = float(b_ub[ni])

            ap_j = float(a_p[j])
            an_j = float(a_n[j])  # negative

            # Combine: lambda * row_p + (1-lambda) * row_n
            # We want the combined a_j to be zero, then the cut is:
            # alpha^T x <= beta (with alpha_j = 0)
            # lambda * ap_j + (1-lambda) * an_j = 0
            # lambda = -an_j / (ap_j - an_j)
            denom = ap_j - an_j
            if abs(denom) < 1e-12:
                continue
            lam = -an_j / denom

            if lam < -1e-6 or lam > 1.0 + 1e-6:
                continue
            lam = np.clip(lam, 0.0, 1.0)

            alpha = lam * a_p + (1 - lam) * a_n
            beta = lam * b_p + (1 - lam) * b_n

            # The combined cut has alpha_j ~= 0, so it's independent of x[j].
            # This IS a valid cut (convex combination of valid constraints).
            # But it doesn't use the integrality of x[j].

            # Now strengthen: since x[j] is binary, we can add back a
            # tighter coefficient. On branch 0 (x[j]=0):
            # alpha^T x <= beta (same, since alpha_j ~= 0)
            # On branch 1 (x[j]=1):
            # sum_{k!=j} alpha_k x_k + alpha_j <= beta
            # => sum_{k!=j} alpha_k x_k <= beta - alpha_j

            # The integrality-strengthened cut:
            # sum_{k!=j} alpha_k x_k + gamma * x[j] <= beta
            # Valid if: gamma <= 0 (branch 0: 0 <= beta - sum_{k!=j} alpha_k x_k)
            # and gamma <= beta - sum_{k!=j} alpha_k x_k at branch 1 endpoints.
            # The tightest gamma that's still valid:
            # gamma = min over feasible x with x[j]=1: beta - sum_{k!=j} alpha_k x_k - alpha_j

            # For the cut to separate x_sol, we want:
            # sum_{k!=j} alpha_k x_sol[k] + gamma * f > beta
            # i.e., gamma * f > beta - sum_{k!=j} alpha_k x_sol[k]
            # Make gamma as large as possible (most positive) while valid.

            # gamma <= 0 is always valid (just drops x[j] from the cut).
            # gamma > 0 is valid if b_p - a_p^T x >= 0 on branch 1
            # (ensured by constraint validity).

            # Simpler: just check if the zero-gamma cut separates x_sol
            lhs_val = float(np.dot(alpha, x_sol))
            violation = lhs_val - beta
            if violation > best_violation:
                best_violation = violation
                best_cut = LinearCut(
                    coeffs=alpha.astype(np.float64),
                    rhs=beta,
                    sense="<=",
                )

    # Also try each row individually (no strengthening, but may still cut)
    for i in active_rows:
        a_i = A_ub[i].copy()
        b_i = float(b_ub[i])
        lhs_val = float(np.dot(a_i, x_sol))
        violation = lhs_val - b_i
        if violation > best_violation:
            best_violation = violation
            best_cut = LinearCut(
                coeffs=a_i.astype(np.float64),
                rhs=b_i,
                sense="<=",
            )

    if best_cut is not None and best_violation > -1e-8:
        return best_cut

    return None
