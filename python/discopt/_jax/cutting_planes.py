"""
Cutting Planes: RLT and Outer Approximation (OA) cut generation.

Provides linear cutting planes that tighten relaxations within spatial Branch & Bound:
  - RLT cuts: McCormick linearization inequalities for bilinear terms x*y
  - OA cuts: gradient-based tangent hyperplanes at NLP relaxation solutions

All cuts are represented as LinearCut NamedTuples with coeffs, rhs, and sense,
suitable for injection into LP/NLP subproblems.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np


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

    If bilinear_term.w_index is None, the cuts are expressed in the
    original variable space by separating the bilinear contribution
    (suitable for use as violated-cut separation in the LP relaxation).

    Args:
        bilinear_term: BilinearTerm describing which variables are multiplied.
        x_lb: lower bounds on all variables, shape (n_vars,).
        x_ub: upper bounds on all variables, shape (n_vars,).
        n_vars: total number of variables (including any auxiliaries).

    Returns:
        List of four LinearCut objects.
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
        # Cuts without auxiliary: express as separation inequalities.
        # For a point x*, the bilinear term x[i]*x[j] is linearized by
        # replacing x[i]*x[j] with the McCormick envelopes. The four
        # cuts bound the bilinear product using only x[i] and x[j].
        #
        # Underestimator 1: x[i]*x[j] >= xi_lb*x[j] + xj_lb*x[i] - xi_lb*xj_lb
        coeffs = np.zeros(n_vars, dtype=np.float64)
        coeffs[i] = xj_lb
        coeffs[j] = xi_lb
        cuts.append(LinearCut(coeffs=coeffs, rhs=xi_lb * xj_lb, sense="<="))

        # Underestimator 2: x[i]*x[j] >= xi_ub*x[j] + xj_ub*x[i] - xi_ub*xj_ub
        coeffs = np.zeros(n_vars, dtype=np.float64)
        coeffs[i] = xj_ub
        coeffs[j] = xi_ub
        cuts.append(LinearCut(coeffs=coeffs, rhs=xi_ub * xj_ub, sense="<="))

        # Overestimator 1: x[i]*x[j] <= xi_ub*x[j] + xj_lb*x[i] - xi_ub*xj_lb
        coeffs = np.zeros(n_vars, dtype=np.float64)
        coeffs[i] = xj_lb
        coeffs[j] = xi_ub
        cuts.append(LinearCut(coeffs=coeffs, rhs=xi_ub * xj_lb, sense=">="))

        # Overestimator 2: x[i]*x[j] <= xi_lb*x[j] + xj_ub*x[i] - xi_lb*xj_ub
        coeffs = np.zeros(n_vars, dtype=np.float64)
        coeffs[i] = xj_ub
        coeffs[j] = xi_lb
        cuts.append(LinearCut(coeffs=coeffs, rhs=xi_lb * xj_ub, sense=">="))

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
) -> list[LinearCut]:
    """Generate OA cuts for all constraints using an NLPEvaluator.

    WARNING: These cuts are only globally valid if all constraints are convex.
    For non-convex constraints, use generate_oa_cuts_from_relaxation() instead,
    which linearizes the McCormick convex underestimators.

    Args:
        evaluator: An NLPEvaluator with evaluate_constraints and evaluate_jacobian.
        x_sol: solution point at which to linearize, shape (n,).
        constraint_senses: list of senses for each constraint. If None,
            all constraints are assumed to be "<=".

    Returns:
        List of LinearCut objects, one per constraint.
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
        grad_k = jac[k, :]
        g_k = float(cons_vals[k])
        sense = constraint_senses[k]
        cut = generate_oa_cut(grad_k, g_k, x_sol, sense=sense)
        cuts.append(cut)

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
) -> list[LinearCut]:
    """Generate OA cuts only for violated constraints at x_sol.

    WARNING: These cuts are only globally valid if all constraints are convex.
    For non-convex constraints, use separate_oa_cuts_from_relaxation() instead.

    A constraint g_k(x) <= 0 is violated if g_k(x_sol) > tol.
    Only violated constraints produce cuts.

    Args:
        evaluator: An NLPEvaluator.
        x_sol: solution point, shape (n,).
        constraint_senses: list of senses. If None, all are "<=".
        tol: violation tolerance.

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
        Expression,
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

    found = []
    seen = set()

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

    Returns:
        List of LinearCut objects (OA + RLT).
    """
    cuts: list[LinearCut] = []

    # OA cuts for violated constraints
    oa = separate_oa_cuts(evaluator, x_sol, constraint_senses, tol)
    cuts.extend(oa)

    # RLT cuts for bilinear terms using current node bounds
    if bilinear_terms is None:
        bilinear_terms = detect_bilinear_terms(model)

    n_vars = len(x_sol)
    for bt in bilinear_terms:
        rlt = separate_rlt_cuts(bt, x_sol, x_lb, x_ub, n_vars, tol)
        cuts.extend(rlt)

    return cuts
