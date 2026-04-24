"""General-purpose Outer Approximation (OA) solver for MINLP.

Implements the Duran-Grossmann (1986) / Fletcher-Leyffer (1994) algorithm
with extensions for feasibility cuts, equality relaxation, and ECP mode.

Decomposes MINLP into alternating NLP subproblems (with fixed integers)
and MILP master problems (with accumulated linearization cuts).

References:
    Duran & Grossmann, Math. Prog. 36, 1986. DOI: 10.1007/BF02592064
    Fletcher & Leyffer, Math. Prog. 66, 1994. DOI: 10.1007/BF01581153
    Viswanathan & Grossmann, C&CE 14(7), 1990. DOI: 10.1016/0098-1354(90)87085-4
    Westerlund & Pettersson, C&CE 19(S1), 1995. DOI: 10.1016/0098-1354(95)00164-W
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from discopt.modeling.core import Constraint, Model, SolveResult, VarType

if TYPE_CHECKING:
    from discopt._jax.nlp_evaluator import NLPEvaluator

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────


@dataclass
class OAConfig:
    """Configuration for the OA solver."""

    time_limit: float = 3600.0
    gap_tolerance: float = 1e-4
    max_iterations: int = 100
    nlp_solver: str = "ipm"
    equality_relaxation: bool = False
    ecp_mode: bool = False
    feasibility_cuts: bool = True
    add_nogood_cuts: bool = True
    log_iterations: bool = True


# ── Problem Decomposition ─────────────────────────────────────


@dataclass
class _DecomposedProblem:
    """Pre-processed model split into linear and nonlinear parts."""

    evaluator: "NLPEvaluator"
    n_vars: int
    n_cons: int
    lb: np.ndarray
    ub: np.ndarray
    int_indices: list[int]
    integrality: np.ndarray
    linear_A_rows: list[np.ndarray]
    linear_b_rows: list[float]
    linear_senses: list[str]
    nonlinear_indices: list[int]
    constraint_senses: list[str]
    obj_coeffs: Optional[tuple] = None
    obj_is_linear: bool = False
    oa_objective_is_convex: bool = True
    oa_constraint_mask: Optional[list[bool]] = None
    master_bound_valid: bool = True
    model: Optional[Model] = None


def _decompose_model(model: Model) -> _DecomposedProblem:
    """Separate model into linear/nonlinear constraints, identify integers."""
    from discopt._jax.convexity import classify_oa_cut_convexity
    from discopt._jax.gdp_reformulate import _extract_body_coeffs, _is_linear
    from discopt._jax.nlp_evaluator import NLPEvaluator

    evaluator = NLPEvaluator(model)
    oa_convexity = classify_oa_cut_convexity(model)
    n_vars = evaluator.n_variables
    n_cons = evaluator.n_constraints
    lb, ub = evaluator.variable_bounds

    # Identify integer/binary variable indices
    int_indices = []
    offset = 0
    for v in model._variables:
        if v.var_type in (VarType.BINARY, VarType.INTEGER):
            for i in range(v.size):
                int_indices.append(offset + i)
        offset += v.size

    integrality = np.zeros(n_vars, dtype=np.int32)
    for idx in int_indices:
        integrality[idx] = 1

    # Classify constraints as linear or nonlinear
    linear_A_rows = []
    linear_b_rows = []
    linear_senses = []
    nonlinear_indices = []

    # Track senses for ALL constraints in evaluator order (nonlinear only)
    all_constraint_senses = []
    eval_idx = 0  # tracks position in evaluator's stacked constraints

    for c in model._constraints:
        if not isinstance(c, Constraint):
            continue
        if _is_linear(c.body):
            coeffs = _extract_body_coeffs(c.body, model, n_vars)
            if coeffs is not None:
                c_vec, off = coeffs
                linear_A_rows.append(c_vec)
                linear_b_rows.append(-off)
                linear_senses.append(c.sense)
            else:
                nonlinear_indices.append(eval_idx)
        else:
            nonlinear_indices.append(eval_idx)
        all_constraint_senses.append(c.sense)
        eval_idx += 1

    # Check if objective is linear
    raw_obj = model._objective
    obj_coeffs = (
        _extract_body_coeffs(raw_obj.expression, model, n_vars) if raw_obj is not None else None
    )
    obj_is_linear = obj_coeffs is not None

    return _DecomposedProblem(
        evaluator=evaluator,
        n_vars=n_vars,
        n_cons=n_cons,
        lb=lb,
        ub=ub,
        int_indices=int_indices,
        integrality=integrality,
        linear_A_rows=linear_A_rows,
        linear_b_rows=linear_b_rows,
        linear_senses=linear_senses,
        nonlinear_indices=nonlinear_indices,
        constraint_senses=all_constraint_senses,
        obj_coeffs=obj_coeffs,
        obj_is_linear=obj_is_linear,
        oa_objective_is_convex=oa_convexity.objective_is_convex,
        oa_constraint_mask=oa_convexity.constraint_mask,
        master_bound_valid=(obj_is_linear or oa_convexity.objective_is_convex),
        model=model,
    )


# ── Bounds Proxy ──────────────────────────────────────────────


class _BoundsProxy:
    """Wraps an NLPEvaluator with overridden variable bounds.

    Forwards all attribute access to the underlying evaluator except
    for variable_bounds which returns the overridden bounds.
    """

    def __init__(self, evaluator, new_lb, new_ub):
        self._eval = evaluator
        self._lb = np.asarray(new_lb, dtype=np.float64)
        self._ub = np.asarray(new_ub, dtype=np.float64)

    def __getattr__(self, name):
        # Forward anything not found on self to the underlying evaluator
        return getattr(self._eval, name)

    @property
    def variable_bounds(self):
        return self._lb, self._ub


# ── NLP Subproblem Solvers ────────────────────────────────────


def _is_primal_feasible(evaluator, x, tol: float = 1e-4) -> bool:
    """Return True if x satisfies all constraints within tol."""
    if evaluator.n_constraints == 0:
        return True
    try:
        from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

        cl, cu = _infer_constraint_bounds(evaluator._model)
        cons = np.asarray(evaluator.evaluate_constraints(x))
        return bool(np.all(cons >= cl - tol) and np.all(cons <= cu + tol))
    except Exception:
        return False


def _solve_nlp(evaluator, lb, ub, nlp_solver: str, max_iter: int = 200):
    """Solve an NLP with given bounds. Returns (x, obj) or (None, None)."""
    lb_clip = np.clip(lb, -1e8, 1e8)
    ub_clip = np.clip(ub, -1e8, 1e8)
    x0 = 0.5 * (lb_clip + ub_clip)

    try:
        if nlp_solver == "ipm" and hasattr(evaluator, "_obj_fn"):
            from discopt._jax.ipm import solve_nlp_ipm

            result = solve_nlp_ipm(evaluator, x0, options={"print_level": 0, "max_iter": max_iter})
        else:
            from discopt.solvers.nlp_ipopt import solve_nlp

            result = solve_nlp(evaluator, x0, options={"print_level": 0, "max_iter": max_iter})

        from discopt.solvers import SolveStatus

        if result.status == SolveStatus.OPTIMAL:
            return result.x, float(evaluator.evaluate_objective(result.x))

        # Accept iteration-limited results if the solution is primal feasible.
        # The IPM may not certify dual convergence (code 4: stalled) yet still
        # find a valid primal point, which is sufficient for OA linearization cuts.
        if result.status == SolveStatus.ITERATION_LIMIT and result.x is not None:
            if _is_primal_feasible(evaluator, result.x):
                return result.x, float(evaluator.evaluate_objective(result.x))
    except Exception:
        pass
    return None, None


def _solve_nlp_relaxation(evaluator, lb, ub, nlp_solver: str):
    """Solve the continuous NLP relaxation (all integers relaxed)."""
    return _solve_nlp(evaluator, lb, ub, nlp_solver)


def _solve_nlp_subproblem(evaluator, lb, ub, int_indices, x_master, nlp_solver):
    """Fix integers at master values and solve NLP subproblem."""
    sub_lb = lb.copy()
    sub_ub = ub.copy()
    for idx in int_indices:
        val = round(x_master[idx])
        sub_lb[idx] = val
        sub_ub[idx] = val

    proxy = _BoundsProxy(evaluator, sub_lb, sub_ub)
    return _solve_nlp(proxy, sub_lb, sub_ub, nlp_solver)


def _solve_feasibility_subproblem(evaluator, lb, ub, int_indices, x_master, nlp_solver):
    """Solve feasibility problem with fixed integers.

    Evaluates constraint violations at the master point and returns the
    point for generating feasibility cuts. When a full feasibility NLP
    cannot be constructed, falls back to returning the master point itself
    so that OA cuts can still be generated there.
    """
    sub_lb = lb.copy()
    sub_ub = ub.copy()
    for idx in int_indices:
        val = round(x_master[idx])
        sub_lb[idx] = val
        sub_ub[idx] = val

    # Try solving the NLP from the master point as initial guess
    proxy = _BoundsProxy(evaluator, sub_lb, sub_ub)
    lb_clip = np.clip(sub_lb, -1e8, 1e8)
    ub_clip = np.clip(sub_ub, -1e8, 1e8)
    x0 = np.clip(x_master[: evaluator.n_variables], lb_clip, ub_clip)

    try:
        if nlp_solver == "ipm" and hasattr(evaluator, "_obj_fn"):
            from discopt._jax.ipm import solve_nlp_ipm

            result = solve_nlp_ipm(proxy, x0, options={"print_level": 0, "max_iter": 200})
        else:
            from discopt.solvers.nlp_ipopt import solve_nlp

            result = solve_nlp(proxy, x0, options={"print_level": 0, "max_iter": 200})

        # Even if infeasible, return the point for cut generation
        if result.x is not None:
            return result.x
    except Exception:
        pass

    # Fallback: return master point (clipped to bounds)
    return x0


# ── Cut Generation ────────────────────────────────────────────


def _add_oa_cuts(
    evaluator,
    x_star,
    n_vars,
    n_cons,
    constraint_senses,
    oa_A_rows,
    oa_b_rows,
    obj_is_linear,
    constraint_convex_mask,
    objective_is_convex,
    equality_relaxation=False,
):
    """Generate OA cuts at x_star and append to cut lists.

    Constraint cuts have length n_vars.
    Objective cuts (when nonlinear) have length n_vars+1, with the last
    element being the -eta epigraph coefficient.
    """
    from discopt._jax.cutting_planes import (
        generate_oa_cuts_from_evaluator,
        generate_objective_oa_cut,
    )

    if n_cons > 0:
        cuts = generate_oa_cuts_from_evaluator(
            evaluator,
            x_star,
            constraint_senses=constraint_senses,
            convex_mask=constraint_convex_mask,
        )
        for cut in cuts:
            coeffs = cut.coeffs.copy()
            # Filter degenerate cuts
            if np.linalg.norm(coeffs) < 1e-12:
                continue

            sense = cut.sense
            if equality_relaxation and sense == "==":
                sense = "<="

            if sense == "<=":
                oa_A_rows.append(coeffs)
                oa_b_rows.append(cut.rhs)
            elif sense == ">=":
                oa_A_rows.append(-coeffs)
                oa_b_rows.append(-cut.rhs)
            elif sense == "==":
                # Equality: add both <= and >= cuts
                oa_A_rows.append(coeffs)
                oa_b_rows.append(cut.rhs)
                oa_A_rows.append(-coeffs)
                oa_b_rows.append(-cut.rhs)

    # Objective OA cut (only if nonlinear): grad^T x - eta <= rhs
    if not obj_is_linear and objective_is_convex:
        n_master = n_vars + 1
        obj_cut = generate_objective_oa_cut(evaluator, x_star, n_master, z_index=n_vars)
        oa_A_rows.append(obj_cut.coeffs.copy())
        oa_b_rows.append(obj_cut.rhs)


def _add_ecp_cuts(
    evaluator,
    x_master,
    n_vars,
    constraint_senses,
    oa_A_rows,
    oa_b_rows,
    obj_is_linear,
    constraint_convex_mask,
    objective_is_convex,
    equality_relaxation=False,
):
    """Generate ECP cuts: OA cuts only for violated constraints at x_master."""
    from discopt._jax.cutting_planes import (
        generate_objective_oa_cut,
        separate_oa_cuts,
    )

    n_added = 0
    if evaluator.n_constraints > 0:
        cuts = separate_oa_cuts(
            evaluator,
            x_master,
            constraint_senses=constraint_senses,
            convex_mask=constraint_convex_mask,
        )
        for cut in cuts:
            coeffs = cut.coeffs.copy()
            if np.linalg.norm(coeffs) < 1e-12:
                continue

            sense = cut.sense
            if equality_relaxation and sense == "==":
                sense = "<="

            if sense == "<=":
                oa_A_rows.append(coeffs)
                oa_b_rows.append(cut.rhs)
                n_added += 1
            elif sense == ">=":
                oa_A_rows.append(-coeffs)
                oa_b_rows.append(-cut.rhs)
                n_added += 1
            elif sense == "==":
                oa_A_rows.append(coeffs)
                oa_b_rows.append(cut.rhs)
                oa_A_rows.append(-coeffs)
                oa_b_rows.append(-cut.rhs)
                n_added += 2

    if not obj_is_linear and objective_is_convex:
        n_master = n_vars + 1
        obj_cut = generate_objective_oa_cut(evaluator, x_master, n_master, z_index=n_vars)
        oa_A_rows.append(obj_cut.coeffs.copy())
        oa_b_rows.append(obj_cut.rhs)
        n_added += 1

    return n_added


def _add_no_good_cut(x_master, int_indices, oa_A_rows, oa_b_rows, n_vars):
    """Add an integer-exclusion (no-good) cut.

    sum_{i: y_i*=1} (1-y_i) + sum_{i: y_i*=0} y_i >= 1
    Equivalently in <= form:
    sum_{y_i*=1} y_i - sum_{y_i*=0} y_i <= count(y_i*=1) - 1
    """
    coeffs = np.zeros(n_vars)
    count_ones = 0
    for idx in int_indices:
        val = round(x_master[idx])
        if val >= 0.5:
            coeffs[idx] = 1.0
            count_ones += 1
        else:
            coeffs[idx] = -1.0
    oa_A_rows.append(coeffs)
    oa_b_rows.append(float(count_ones - 1))


def _add_feasibility_cuts(
    evaluator,
    x_feas,
    n_vars,
    constraint_senses,
    oa_A_rows,
    oa_b_rows,
    constraint_convex_mask,
):
    """Add gradient-based feasibility cuts (Fletcher-Leyffer 1994).

    For each violated constraint g_k(x) <= 0 at x_feas:
        g_k(x_feas) + nabla g_k(x_feas)^T (x - x_feas) <= 0
    """
    from discopt._jax.cutting_planes import separate_oa_cuts

    if evaluator.n_constraints == 0:
        return

    cuts = separate_oa_cuts(
        evaluator,
        x_feas,
        constraint_senses=constraint_senses,
        convex_mask=constraint_convex_mask,
    )
    for cut in cuts:
        coeffs = cut.coeffs.copy()
        if np.linalg.norm(coeffs) < 1e-12:
            continue
        if cut.sense == "<=":
            oa_A_rows.append(coeffs)
            oa_b_rows.append(cut.rhs)
        elif cut.sense == ">=":
            oa_A_rows.append(-coeffs)
            oa_b_rows.append(-cut.rhs)


# ── MILP Master Problem ──────────────────────────────────────


def _solve_master_milp(
    linear_A_rows,
    linear_b_rows,
    linear_senses,
    oa_A_rows,
    oa_b_rows,
    n_vars,
    integrality,
    lb,
    ub,
    obj_coeffs,
    obj_is_linear,
    objective_bound_valid,
    time_limit,
    gap_tolerance,
):
    """Build and solve the master MILP."""
    try:
        from discopt.solvers.milp_highs import solve_milp
    except ImportError as e:
        raise ImportError(
            "OA solver requires highspy for the MILP master. Install with: pip install highspy"
        ) from e

    use_objective_epigraph = (not obj_is_linear) and objective_bound_valid
    n_master = n_vars
    if use_objective_epigraph:
        n_master += 1  # epigraph variable eta

    # Build A_ub, b_ub from linear <= constraints + OA cuts
    A_ub_rows = []
    b_ub_vals = []

    for i, sense in enumerate(linear_senses):
        row = linear_A_rows[i]
        if use_objective_epigraph:
            row = np.append(row, 0.0)
        if sense == "<=":
            A_ub_rows.append(row)
            b_ub_vals.append(linear_b_rows[i])
        elif sense == ">=":
            A_ub_rows.append(-row)
            b_ub_vals.append(-linear_b_rows[i])

    # OA cuts (all in <= form already)
    # Constraint cuts have length n_vars; objective cuts have length n_master
    for i in range(len(oa_A_rows)):
        row = oa_A_rows[i]
        if use_objective_epigraph and len(row) == n_vars:
            row = np.append(row, 0.0)  # extend constraint cuts with 0 for eta
        A_ub_rows.append(row)
        b_ub_vals.append(oa_b_rows[i])

    # Equality constraints from linear
    A_eq_rows = []
    b_eq_vals = []
    for i, sense in enumerate(linear_senses):
        if sense == "==":
            row = linear_A_rows[i]
            if use_objective_epigraph:
                row = np.append(row, 0.0)
            A_eq_rows.append(row)
            b_eq_vals.append(linear_b_rows[i])

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals) if b_ub_vals else None
    A_eq = np.array(A_eq_rows) if A_eq_rows else None
    b_eq = np.array(b_eq_vals) if b_eq_vals else None

    # Objective
    if obj_is_linear:
        c_vec, _off = obj_coeffs
        c = c_vec.copy()
    elif use_objective_epigraph:
        c = np.zeros(n_master)
        c[-1] = 1.0  # minimize eta
    else:
        c = np.zeros(n_master)

    # Bounds
    bounds_list = list(zip(lb.tolist(), ub.tolist()))
    if use_objective_epigraph:
        bounds_list.append((-1e20, 1e20))  # eta unbounded

    # Integrality
    int_vec = np.zeros(n_master, dtype=np.int32)
    int_vec[:n_vars] = integrality

    return solve_milp(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds_list,
        integrality=int_vec,
        time_limit=time_limit,
        gap_tolerance=gap_tolerance,
    )


# ── Result Construction ───────────────────────────────────────


def _build_x_dict(x_flat: np.ndarray, model: Model) -> dict:
    """Convert flat solution vector to {var_name: value} dict."""
    result = {}
    offset = 0
    for v in model._variables:
        result[v.name] = x_flat[offset : offset + v.size].reshape(v.shape)
        offset += v.size
    return result


def _compute_gap(lb: float, ub: float) -> float:
    if ub >= 1e19 or lb <= -1e19:
        return 1.0
    denom = max(1e-10, abs(ub))
    return (ub - lb) / denom


# ── Main Algorithm ────────────────────────────────────────────


def solve_oa(
    model: Model,
    time_limit: float = 3600.0,
    gap_tolerance: float = 1e-4,
    max_iterations: int = 100,
    nlp_solver: str = "ipm",
    equality_relaxation: bool = False,
    ecp_mode: bool = False,
    feasibility_cuts: bool = True,
    **kwargs,
) -> SolveResult:
    """Solve a MINLP via Outer Approximation.

    Decomposes the problem into alternating NLP subproblems (continuous
    optimization with fixed integers) and MILP master problems (linear
    relaxation with accumulated OA cuts).

    Parameters
    ----------
    model : Model
        MINLP model with continuous, binary, and/or integer variables.
    time_limit : float
        Wall-clock time limit in seconds.
    gap_tolerance : float
        Relative optimality gap for convergence.
    max_iterations : int
        Maximum OA iterations.
    nlp_solver : str
        NLP backend: ``"ipm"``, ``"ipopt"``, ``"ripopt"``.
    equality_relaxation : bool
        Relax nonlinear equalities to inequalities in OA cuts
        (Viswanathan & Grossmann 1990). Helps when nonlinear equalities
        cause the MILP master to become infeasible.
    ecp_mode : bool
        Extended Cutting Plane mode (Westerlund & Pettersson 1995):
        skip NLP subproblems entirely, only add cuts at MILP master
        solutions for violated constraints. Simpler but slower convergence.
    feasibility_cuts : bool
        Use gradient-based feasibility cuts (Fletcher & Leyffer 1994)
        when the NLP subproblem is infeasible. Stronger than no-good cuts.

    Returns
    -------
    SolveResult
    """
    t_start = time.perf_counter()

    # 1. Decompose model
    decomp = _decompose_model(model)
    evaluator = decomp.evaluator
    n_vars = decomp.n_vars
    n_cons = decomp.n_cons
    if decomp.oa_constraint_mask is not None and not all(decomp.oa_constraint_mask):
        logger.warning(
            "OA: generating OA cuts only for %d of %d constraints classified convex",
            sum(1 for is_convex in decomp.oa_constraint_mask if is_convex),
            len(decomp.oa_constraint_mask),
        )
    if not decomp.obj_is_linear and not decomp.oa_objective_is_convex:
        logger.warning(
            "OA: nonlinear objective is not convex in the optimization sense; "
            "disabling master lower-bound updates and skipping objective OA cuts"
        )

    # If no integer variables, just solve the NLP directly
    if len(decomp.int_indices) == 0:
        x_sol, obj = _solve_nlp_relaxation(evaluator, decomp.lb, decomp.ub, nlp_solver)
        wall_time = time.perf_counter() - t_start
        if x_sol is not None:
            return SolveResult(
                status="optimal",
                objective=obj,
                bound=obj,
                gap=0.0,
                x=_build_x_dict(x_sol, model),
                wall_time=wall_time,
            )
        return SolveResult(
            status="infeasible",
            objective=None,
            bound=None,
            gap=None,
            x={},
            wall_time=wall_time,
        )

    # 2. Solve initial NLP relaxation for first linearization point
    oa_A_rows: list[np.ndarray] = []
    oa_b_rows: list[float] = []

    x_relax, obj_relax = _solve_nlp_relaxation(evaluator, decomp.lb, decomp.ub, nlp_solver)

    UB = 1e20
    LB = -1e20
    incumbent = None
    incumbent_obj = None

    if x_relax is not None:
        _add_oa_cuts(
            evaluator,
            x_relax,
            n_vars,
            n_cons,
            decomp.constraint_senses,
            oa_A_rows,
            oa_b_rows,
            decomp.obj_is_linear,
            decomp.oa_constraint_mask,
            decomp.oa_objective_is_convex,
            equality_relaxation=equality_relaxation,
        )
        # Check if relaxation solution is already integer-feasible
        is_int_feasible = all(
            abs(x_relax[idx] - round(x_relax[idx])) < 1e-5 for idx in decomp.int_indices
        )
        if is_int_feasible and obj_relax is not None:
            UB = obj_relax
            incumbent = x_relax.copy()
            incumbent_obj = obj_relax
    else:
        # NLP relaxation failed — generate initial cuts at midpoint
        lb_clip = np.clip(decomp.lb, -1e8, 1e8)
        ub_clip = np.clip(decomp.ub, -1e8, 1e8)
        x_mid = 0.5 * (lb_clip + ub_clip)
        _add_oa_cuts(
            evaluator,
            x_mid,
            n_vars,
            n_cons,
            decomp.constraint_senses,
            oa_A_rows,
            oa_b_rows,
            decomp.obj_is_linear,
            decomp.oa_constraint_mask,
            decomp.oa_objective_is_convex,
            equality_relaxation=equality_relaxation,
        )

    # 3. Main OA loop
    for iteration in range(max_iterations):
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            logger.info("OA: Time limit reached at iteration %d", iteration)
            break

        # a. Solve master MILP
        master_result = _solve_master_milp(
            decomp.linear_A_rows,
            decomp.linear_b_rows,
            decomp.linear_senses,
            oa_A_rows,
            oa_b_rows,
            n_vars,
            decomp.integrality,
            decomp.lb,
            decomp.ub,
            decomp.obj_coeffs,
            decomp.obj_is_linear,
            decomp.master_bound_valid,
            time_limit=time_limit - elapsed,
            gap_tolerance=gap_tolerance,
        )

        from discopt.solvers import SolveStatus

        if master_result is None:
            logger.info("OA: Master MILP failed at iteration %d", iteration)
            break

        if master_result.status == SolveStatus.INFEASIBLE:
            logger.info("OA: Master MILP infeasible at iteration %d", iteration)
            break

        if master_result.status == SolveStatus.UNBOUNDED or master_result.x is None:
            # Master unbounded → need more OA cuts. Generate at midpoint.
            logger.info("OA: Master MILP unbounded at iteration %d, adding cuts", iteration)
            lb_clip = np.clip(decomp.lb, -1e8, 1e8)
            ub_clip = np.clip(decomp.ub, -1e8, 1e8)
            x_mid = 0.5 * (lb_clip + ub_clip)
            _add_oa_cuts(
                evaluator,
                x_mid,
                n_vars,
                n_cons,
                decomp.constraint_senses,
                oa_A_rows,
                oa_b_rows,
                decomp.obj_is_linear,
                decomp.oa_constraint_mask,
                decomp.oa_objective_is_convex,
                equality_relaxation=equality_relaxation,
            )
            continue

        x_master = master_result.x[:n_vars]
        if decomp.master_bound_valid and master_result.objective is not None:
            LB = max(LB, master_result.objective)

        # b. ECP mode: add cuts at master point, skip NLP
        if ecp_mode:
            n_violated = _add_ecp_cuts(
                evaluator,
                x_master,
                n_vars,
                decomp.constraint_senses,
                oa_A_rows,
                oa_b_rows,
                decomp.obj_is_linear,
                decomp.oa_constraint_mask,
                decomp.oa_objective_is_convex,
                equality_relaxation=equality_relaxation,
            )
            # In ECP, use master objective as heuristic UB
            master_obj = float(evaluator.evaluate_objective(x_master))
            cons_vals = evaluator.evaluate_constraints(x_master)
            is_feasible = all(cons_vals[k] <= 1e-6 for k in range(n_cons))
            if is_feasible and master_obj < UB:
                UB = master_obj
                incumbent = x_master.copy()
                incumbent_obj = master_obj

            gap = _compute_gap(LB, UB)
            logger.info(
                "OA-ECP iter %d: LB=%.6f UB=%.6f gap=%.4f%% cuts=%d violated=%d",
                iteration,
                LB,
                UB,
                gap * 100,
                len(oa_A_rows),
                n_violated,
            )

            if n_violated == 0 or gap <= gap_tolerance:
                break
            continue

        # c. Fix integers, solve NLP subproblem
        x_nlp, obj_nlp = _solve_nlp_subproblem(
            evaluator,
            decomp.lb,
            decomp.ub,
            decomp.int_indices,
            x_master,
            nlp_solver,
        )

        if x_nlp is not None:
            if obj_nlp < UB:
                UB = obj_nlp
                incumbent = x_nlp.copy()
                incumbent_obj = obj_nlp

            # Generate OA cuts at NLP solution
            _add_oa_cuts(
                evaluator,
                x_nlp,
                n_vars,
                n_cons,
                decomp.constraint_senses,
                oa_A_rows,
                oa_b_rows,
                decomp.obj_is_linear,
                decomp.oa_constraint_mask,
                decomp.oa_objective_is_convex,
                equality_relaxation=equality_relaxation,
            )
        else:
            # NLP infeasible for this integer assignment
            if feasibility_cuts:
                x_feas = _solve_feasibility_subproblem(
                    evaluator,
                    decomp.lb,
                    decomp.ub,
                    decomp.int_indices,
                    x_master,
                    nlp_solver,
                )
                if x_feas is not None:
                    _add_feasibility_cuts(
                        evaluator,
                        x_feas,
                        n_vars,
                        decomp.constraint_senses,
                        oa_A_rows,
                        oa_b_rows,
                        decomp.oa_constraint_mask,
                    )

            # Always add no-good cut as fallback to avoid cycling
            _add_no_good_cut(x_master, decomp.int_indices, oa_A_rows, oa_b_rows, n_vars)

            # Also add OA cuts at master point
            _add_oa_cuts(
                evaluator,
                x_master,
                n_vars,
                n_cons,
                decomp.constraint_senses,
                oa_A_rows,
                oa_b_rows,
                decomp.obj_is_linear,
                decomp.oa_constraint_mask,
                decomp.oa_objective_is_convex,
                equality_relaxation=equality_relaxation,
            )

        # d. Check convergence
        gap = _compute_gap(LB, UB)
        logger.info(
            "OA iter %d: LB=%.6f UB=%.6f gap=%.4f%% cuts=%d",
            iteration,
            LB,
            UB,
            gap * 100,
            len(oa_A_rows),
        )

        if gap <= gap_tolerance:
            break

    # 4. Build result
    wall_time = time.perf_counter() - t_start
    gap = _compute_gap(LB, UB)
    bound = LB if decomp.master_bound_valid and LB > -1e19 else None
    reported_gap = gap if bound is not None and UB < 1e19 else None

    if incumbent is not None:
        status = "optimal" if gap <= gap_tolerance else "feasible"
        return SolveResult(
            status=status,
            objective=incumbent_obj,
            bound=bound,
            gap=reported_gap,
            x=_build_x_dict(incumbent, model),
            wall_time=wall_time,
        )

    return SolveResult(
        status="infeasible",
        objective=None,
        bound=bound,
        gap=None,
        x={},
        wall_time=wall_time,
    )
