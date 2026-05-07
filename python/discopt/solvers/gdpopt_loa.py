"""GDPopt-style Logic-based Outer Approximation (LOA) solver for GDP.

Decomposes a GDP/MINLP into alternating MILP master + NLP subproblems.
Requires highspy for the MILP master problem solver.
"""

from __future__ import annotations

import logging
import time
from typing import cast

import numpy as np

from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.modeling.core import (
    Constraint,
    Model,
    SolveResult,
    VarType,
)

logger = logging.getLogger(__name__)


def solve_gdpopt_loa(
    model: Model,
    time_limit: float = 3600.0,
    gap_tolerance: float = 1e-4,
    max_iterations: int = 100,
    nlp_solver: str = "ipm",
) -> SolveResult:
    """Solve a GDP model via Logic-based Outer Approximation.

    Parameters
    ----------
    model : Model
        The original model (before GDP reformulation).
    time_limit : float
        Wall-clock time limit in seconds.
    gap_tolerance : float
        Relative optimality gap tolerance.
    max_iterations : int
        Maximum LOA iterations.
    nlp_solver : str
        NLP solver backend for subproblems (``"ipm"``, ``"ripopt"``, ``"ipopt"``).

    Returns
    -------
    SolveResult
    """
    t_start = time.perf_counter()

    # 1. Reformulate GDP to standard MINLP via big-M
    from discopt._jax.gdp_reformulate import reformulate_gdp

    reformulated = reformulate_gdp(model, method="big-m")

    # 2. Build NLP evaluator for the reformulated model
    from discopt._jax.convexity import classify_oa_cut_convexity
    from discopt._jax.nlp_evaluator import NLPEvaluator

    evaluator = NLPEvaluator(reformulated)
    oa_convexity = classify_oa_cut_convexity(reformulated)
    n_vars = evaluator.n_variables
    n_cons = evaluator.n_constraints
    lb, ub = evaluator.variable_bounds
    obj_is_linear = False
    master_bound_valid = False

    # 3. Identify integer/binary variable indices
    int_indices = []
    offset = 0
    for v in reformulated._variables:
        if v.var_type in (VarType.BINARY, VarType.INTEGER):
            for i in range(v.size):
                int_indices.append(offset + i)
        offset += v.size
    # Build integrality vector for MILP master
    integrality = np.zeros(n_vars, dtype=np.int32)
    for idx in int_indices:
        integrality[idx] = 1

    # 4. Extract linear constraints as coefficient matrices
    linear_A_rows = []
    linear_b_rows = []
    linear_senses = []
    nonlinear_indices = []

    from discopt._jax.gdp_reformulate import _extract_body_coeffs, _is_linear

    for k, c in enumerate(reformulated._constraints):
        if not isinstance(c, Constraint):
            continue
        if _is_linear(c.body):
            coeffs = _extract_body_coeffs(c.body, reformulated, n_vars)
            if coeffs is not None:
                c_vec, off = coeffs
                linear_A_rows.append(c_vec)
                linear_b_rows.append(-off)  # body - off sense 0 → c_vec @ x sense -off
                linear_senses.append(c.sense)
            else:
                nonlinear_indices.append(k)
        else:
            nonlinear_indices.append(k)

    # 5. Check if objective is linear
    _raw_obj = reformulated._objective
    obj_coeffs = (
        _extract_body_coeffs(_raw_obj.expression, reformulated, n_vars)
        if _raw_obj is not None
        else None
    )
    obj_is_linear = obj_coeffs is not None
    master_bound_valid = obj_is_linear or oa_convexity.objective_is_convex

    if n_cons > 0 and not all(oa_convexity.constraint_mask):
        logger.warning(
            "LOA: generating OA cuts only for %d of %d constraints classified convex",
            sum(1 for is_convex in oa_convexity.constraint_mask if is_convex),
            len(oa_convexity.constraint_mask),
        )
    if not obj_is_linear and not oa_convexity.objective_is_convex:
        logger.warning(
            "LOA: nonlinear objective is not convex in the optimization sense; "
            "disabling master lower-bound updates and skipping objective OA cuts"
        )

    # 6. Solve initial NLP relaxation (continuous)
    x_relax = _solve_nlp_relaxation(evaluator, lb, ub, nlp_solver)

    # 7. Initialize OA cut pool
    oa_A_rows: list[np.ndarray] = []
    oa_b_rows: list[float] = []

    if x_relax is not None:
        _add_oa_cuts(
            evaluator,
            x_relax,
            n_vars,
            n_cons,
            oa_A_rows,
            oa_b_rows,
            obj_is_linear,
            oa_convexity.constraint_mask,
            oa_convexity.objective_is_convex,
        )

    # 8. Main LOA loop
    LB = -1e20
    UB = 1e20
    incumbent = None
    incumbent_obj = None

    for iteration in range(max_iterations):
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            break

        # a. Build and solve master MILP
        master_result = _solve_master_milp(
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
            master_bound_valid,
            time_limit=time_limit - elapsed,
            gap_tolerance=gap_tolerance,
        )

        if master_result is None or master_result.x is None:
            # Master infeasible → problem is infeasible
            logger.info("LOA: Master MILP infeasible at iteration %d", iteration)
            break

        x_master = master_result.x[:n_vars]
        if master_bound_valid and master_result.objective is not None:
            LB = max(LB, master_result.objective)

        # b. Fix integers to master values, solve NLP subproblem
        sub_lb = lb.copy()
        sub_ub = ub.copy()
        for idx in int_indices:
            val = round(x_master[idx])
            sub_lb[idx] = val
            sub_ub[idx] = val

        x_nlp = _solve_nlp_subproblem(evaluator, sub_lb, sub_ub, nlp_solver)

        if x_nlp is not None:
            obj_nlp = float(evaluator.evaluate_objective(x_nlp))
            if obj_nlp < UB:
                UB = obj_nlp
                incumbent = x_nlp.copy()
                incumbent_obj = obj_nlp

            # c. Generate OA cuts at NLP solution
            _add_oa_cuts(
                evaluator,
                x_nlp,
                n_vars,
                n_cons,
                oa_A_rows,
                oa_b_rows,
                obj_is_linear,
                oa_convexity.constraint_mask,
                oa_convexity.objective_is_convex,
            )
        else:
            # NLP infeasible → add no-good cut
            _add_no_good_cut(x_master, int_indices, oa_A_rows, oa_b_rows, n_vars)
            # Also try OA cuts at master point
            _add_oa_cuts(
                evaluator,
                x_master,
                n_vars,
                n_cons,
                oa_A_rows,
                oa_b_rows,
                obj_is_linear,
                oa_convexity.constraint_mask,
                oa_convexity.objective_is_convex,
            )

        # d. Check convergence
        gap = _compute_gap(LB, UB)
        logger.info(
            "LOA iter %d: LB=%.6f UB=%.6f gap=%.4f%% cuts=%d",
            iteration,
            LB,
            UB,
            gap * 100,
            len(oa_A_rows),
        )

        if gap <= gap_tolerance:
            break

    # 9. Build result
    wall_time = time.perf_counter() - t_start
    gap = _compute_gap(LB, UB)
    bound = LB if master_bound_valid and LB > -1e19 else None
    reported_gap = gap if bound is not None and UB < 1e19 else None

    if incumbent is not None:
        status = "optimal" if gap <= gap_tolerance else "feasible"
        x_dict = _build_x_dict(incumbent, reformulated)
        return SolveResult(
            status=status,
            objective=incumbent_obj,
            bound=bound,
            gap=reported_gap,
            x=x_dict,
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


def _compute_gap(lb: float, ub: float) -> float:
    if ub >= 1e19 or lb <= -1e19:
        return 1.0
    abs_gap = max(0.0, ub - lb)
    if abs_gap <= 1e-9:
        return 0.0
    denom = max(abs(ub), abs(lb), 1e-10)
    return abs_gap / denom


def _solve_nlp_relaxation(evaluator, lb, ub, nlp_solver: str) -> np.ndarray | None:
    """Solve the continuous NLP relaxation."""
    lb_clip = np.clip(lb, -1e8, 1e8)
    ub_clip = np.clip(ub, -1e8, 1e8)
    x0 = 0.5 * (lb_clip + ub_clip)

    try:
        if nlp_solver == "ipm" and hasattr(evaluator, "_obj_fn"):
            from discopt._jax.ipm import solve_nlp_ipm

            result = solve_nlp_ipm(evaluator, x0, options={"print_level": 0})
        else:
            from discopt.solvers.nlp_ipopt import solve_nlp

            result = solve_nlp(evaluator, x0, options={"print_level": 0})

        from discopt.solvers import SolveStatus

        if result.status == SolveStatus.OPTIMAL and result.x is not None:
            return np.asarray(result.x, dtype=np.float64)
    except Exception:
        pass
    return None


def _solve_nlp_subproblem(evaluator, sub_lb, sub_ub, nlp_solver: str) -> np.ndarray | None:
    """Solve NLP subproblem with fixed integer bounds."""
    # IPM log-barriers require strict lb < x < ub; relax fixed vars by tiny slack
    ipm_lb = sub_lb.copy()
    ipm_ub = sub_ub.copy()
    fixed = ipm_lb == ipm_ub
    ipm_lb[fixed] -= 1e-8
    ipm_ub[fixed] += 1e-8

    lb_clip = np.clip(ipm_lb, -1e8, 1e8)
    ub_clip = np.clip(ipm_ub, -1e8, 1e8)
    x0 = np.clip(0.5 * (lb_clip + ub_clip), sub_lb, sub_ub)

    # Create a bounds-overriding evaluator proxy
    proxy = _BoundsProxy(evaluator, ipm_lb, ipm_ub)

    try:
        if nlp_solver == "ipm" and hasattr(evaluator, "_obj_fn"):
            from discopt._jax.ipm import solve_nlp_ipm

            result = solve_nlp_ipm(proxy, x0, options={"print_level": 0, "max_iter": 200})
        else:
            from discopt.solvers.nlp_ipopt import solve_nlp

            result = solve_nlp(
                cast(NLPEvaluator, proxy), x0, options={"print_level": 0, "max_iter": 200}
            )

        from discopt.solvers import SolveStatus

        if result.status == SolveStatus.OPTIMAL and result.x is not None:
            return np.asarray(result.x, dtype=np.float64)

        # Accept ITERATION_LIMIT if the solution is primal-feasible.
        # The IPM may fail to certify dual optimality (e.g. on LPs or
        # near-degenerate NLPs) while still finding the correct primal
        # solution.  Any feasible point is a valid LOA upper bound.
        if result.status == SolveStatus.ITERATION_LIMIT and result.x is not None:
            x_sol = np.asarray(result.x, dtype=np.float64)
            atol = 1e-6
            if np.all(x_sol >= sub_lb - atol) and np.all(x_sol <= sub_ub + atol):
                feas = True
                if evaluator.n_constraints > 0:
                    g_val = evaluator.evaluate_constraints(x_sol)
                    model = getattr(evaluator, "_model", None)
                    if model is None:
                        model = proxy._eval._model
                    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

                    c_lb, c_ub = _infer_constraint_bounds(model)
                    for ci in range(len(c_lb)):
                        if g_val[ci] < c_lb[ci] - atol or g_val[ci] > c_ub[ci] + atol:
                            feas = False
                            break
                if feas:
                    return x_sol
    except Exception:
        pass
    return None


class _BoundsProxy:
    """Wraps an NLPEvaluator with overridden variable bounds."""

    def __init__(self, evaluator: NLPEvaluator, new_lb, new_ub) -> None:
        self._eval = evaluator
        self._lb = np.asarray(new_lb, dtype=np.float64)
        self._ub = np.asarray(new_ub, dtype=np.float64)

    @property
    def n_variables(self):
        return self._eval.n_variables

    @property
    def n_constraints(self):
        return self._eval.n_constraints

    @property
    def variable_bounds(self):
        return self._lb, self._ub

    @property
    def _model(self):
        return self._eval._model

    @property
    def _obj_fn(self):
        return self._eval._obj_fn

    @property
    def _cons_fn(self):
        return self._eval._cons_fn

    @property
    def _source_constraints(self):
        return self._eval._source_constraints

    @property
    def _constraint_flat_sizes(self):
        return self._eval._constraint_flat_sizes

    def evaluate_objective(self, x):
        return self._eval.evaluate_objective(x)

    def evaluate_gradient(self, x):
        return self._eval.evaluate_gradient(x)

    def evaluate_hessian(self, x):
        return self._eval.evaluate_hessian(x)

    def evaluate_constraints(self, x):
        return self._eval.evaluate_constraints(x)

    def evaluate_jacobian(self, x):
        return self._eval.evaluate_jacobian(x)

    def evaluate_lagrangian_hessian(self, x, obj_factor, lam):
        return self._eval.evaluate_lagrangian_hessian(x, obj_factor, lam)


def _add_oa_cuts(
    evaluator,
    x_star,
    n_vars,
    n_cons,
    oa_A_rows,
    oa_b_rows,
    obj_is_linear,
    constraint_convex_mask,
    objective_is_convex,
):
    """Generate OA cuts at x_star and append to cut lists."""
    from discopt._jax.cutting_planes import (
        generate_oa_cuts_from_evaluator,
        generate_objective_oa_cut,
    )

    # Constraint OA cuts
    if n_cons > 0:
        cuts = generate_oa_cuts_from_evaluator(
            evaluator,
            x_star,
            convex_mask=constraint_convex_mask,
        )
        for cut in cuts:
            if cut.sense == "<=":
                oa_A_rows.append(cut.coeffs.copy())
                oa_b_rows.append(cut.rhs)
            elif cut.sense == ">=":
                oa_A_rows.append(-cut.coeffs.copy())
                oa_b_rows.append(-cut.rhs)

    # Objective OA cut (only if nonlinear)
    if not obj_is_linear and objective_is_convex:
        obj_cut = generate_objective_oa_cut(evaluator, x_star, n_vars + 1, z_index=n_vars)
        # This cut is: coeffs @ x <= rhs (underestimates the objective)
        oa_A_rows.append(obj_cut.coeffs.copy())
        oa_b_rows.append(obj_cut.rhs)


def _add_no_good_cut(x_master, int_indices, oa_A_rows, oa_b_rows, n_vars):
    """Add an integer-exclusion (no-good) cut."""
    # sum_{i: y_i*=1} (1 - y_i) + sum_{i: y_i*=0} y_i >= 1
    # Equivalently: -sum_{y_i*=1} y_i + sum_{y_i*=0} y_i >= 1 - count(y_i*=1)
    # As <= form: sum_{y_i*=1} y_i - sum_{y_i*=0} y_i <= count(y_i*=1) - 1
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
            "LOA solver requires highspy for the MILP master. Install with: pip install highspy"
        ) from e

    # Determine master problem dimensions
    use_objective_epigraph = (not obj_is_linear) and objective_bound_valid
    n_master = n_vars
    if use_objective_epigraph:
        n_master += 1  # epigraph variable eta

    # Build A_ub, b_ub from linear <= constraints + OA cuts
    A_ub_rows = []
    b_ub_vals = []

    for i, sense in enumerate(linear_senses):
        row = linear_A_rows[i]
        rhs = linear_b_rows[i]
        if use_objective_epigraph:
            row = np.append(row, 0.0)
        if sense == "<=":
            A_ub_rows.append(row)
            b_ub_vals.append(rhs)
        elif sense == ">=":
            A_ub_rows.append(-row)
            b_ub_vals.append(-rhs)

    # OA cuts (all <= form)
    for i in range(len(oa_A_rows)):
        row = oa_A_rows[i]
        if use_objective_epigraph and len(row) == n_vars:
            row = np.append(row, 0.0)
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

    # Build arrays
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals) if b_ub_vals else None
    A_eq = np.array(A_eq_rows) if A_eq_rows else None
    b_eq = np.array(b_eq_vals) if b_eq_vals else None

    # Objective
    if obj_is_linear:
        c_vec, off = obj_coeffs
        c = c_vec.copy()
    elif use_objective_epigraph:
        # min eta, with OA cuts: eta >= f(x*) + grad(x*) . (x - x*)
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


def _build_x_dict(x_flat: np.ndarray, model: Model) -> dict:
    """Convert flat solution vector to {var_name: value} dict."""
    result = {}
    offset = 0
    for v in model._variables:
        result[v.name] = x_flat[offset : offset + v.size].reshape(v.shape)
        offset += v.size
    return result
