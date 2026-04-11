"""
NLP solver wrapper using cyipopt (Python binding for Ipopt).

Phase 1 scaffolding: uses cyipopt for continuous relaxation solves.
Will be replaced by direct Rust Ipopt bindings later.

Maps NLPEvaluator callbacks to cyipopt.Problem interface:
  - objective, gradient, constraints, jacobian, hessian
  - Variable and constraint bounds
  - Ipopt status codes to SolveStatus enum
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.modeling.core import Model
from discopt.solvers import NLPResult, SolveStatus

# Ipopt status code mapping
# See: https://coin-or.github.io/Ipopt/IpReturnCodes_8inc.html
#
# NOTE: Status 2 ("Infeasible_Problem_Detected") is mapped to ERROR rather
# than INFEASIBLE because IPOPT can only detect *local* infeasibility.
# For non-convex NLPs the problem may still be feasible from a different
# starting point.  Mapping to ERROR prevents the solver from confidently
# reporting "infeasible" when the problem is merely hard to solve.
_IPOPT_STATUS_MAP: dict[int, SolveStatus] = {
    0: SolveStatus.OPTIMAL,  # Solve_Succeeded
    1: SolveStatus.OPTIMAL,  # Solved_To_Acceptable_Level
    2: SolveStatus.ERROR,  # Infeasible_Problem_Detected (local only)
    3: SolveStatus.UNBOUNDED,  # Search_Direction_Becomes_Too_Small
    4: SolveStatus.ERROR,  # Diverging_Iterates
    5: SolveStatus.ERROR,  # User_Requested_Stop
    6: SolveStatus.ERROR,  # Feasible_Point_Found (not optimal)
    -1: SolveStatus.ITERATION_LIMIT,  # Maximum_Iterations_Exceeded
    -2: SolveStatus.ERROR,  # Restoration_Failed
    -3: SolveStatus.ERROR,  # Error_In_Step_Computation
    -4: SolveStatus.TIME_LIMIT,  # Maximum_CpuTime_Exceeded
    -5: SolveStatus.TIME_LIMIT,  # Maximum_WallTime_Exceeded
}


class _IpoptCallbacks:
    """Adapter mapping NLPEvaluator methods to cyipopt.Problem callbacks."""

    def __init__(self, evaluator) -> None:
        self._ev = evaluator
        self._n = evaluator.n_variables
        self._m = evaluator.n_constraints
        self._use_sparse = (
            hasattr(evaluator, "has_sparse_structure") and evaluator.has_sparse_structure()
        )

    def objective(self, x: np.ndarray) -> float:
        return self._ev.evaluate_objective(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self._ev.evaluate_gradient(x)

    def constraints(self, x: np.ndarray) -> np.ndarray:
        return self._ev.evaluate_constraints(x)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        if self._use_sparse:
            return self._ev.evaluate_jacobian_values(x)
        # cyipopt wants the Jacobian flattened in the order given by jacobianstructure
        jac = self._ev.evaluate_jacobian(x)
        return jac.flatten()

    def jacobianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        if self._use_sparse:
            return self._ev.jacobian_structure()
        # Dense structure: all (row, col) pairs
        rows, cols = np.meshgrid(np.arange(self._m), np.arange(self._n), indexing="ij")
        return (rows.flatten(), cols.flatten())

    def hessian(self, x: np.ndarray, lagrange: np.ndarray, obj_factor: float) -> np.ndarray:
        if self._use_sparse:
            return self._ev.evaluate_hessian_values(x, obj_factor, lagrange)
        # Hessian of the Lagrangian = obj_factor * H_obj + sum(lagrange[i] * H_c[i])
        if hasattr(self._ev, "evaluate_lagrangian_hessian"):
            h = self._ev.evaluate_lagrangian_hessian(x, obj_factor, lagrange)
        else:
            h = obj_factor * self._ev.evaluate_hessian(x)

        # Extract lower triangle in row-major order matching hessianstructure
        rows, cols = self.hessianstructure()
        return h[rows, cols]

    def hessianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        if self._use_sparse:
            return self._ev.hessian_structure()
        # Lower triangle (including diagonal)
        rows, cols = np.tril_indices(self._n)
        return (rows, cols)


def _infer_constraint_bounds(
    model: Model,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer constraint bounds (cl, cu) from model constraint senses.

    The NLPEvaluator compiles constraints as `body - rhs`, so we need:
      - For `<=` constraints: body - rhs <= 0, so cl = -inf, cu = 0
      - For `==` constraints: body - rhs == 0, so cl = 0, cu = 0
      - For `>=` constraints: these are already normalized to <= by the
        Expression.__ge__ method, so we only see <= and == here.
    """
    from discopt.modeling.core import Constraint

    cl_list = []
    cu_list = []
    for c in model._constraints:
        if not isinstance(c, Constraint):
            continue
        if c.sense == "<=":
            cl_list.append(-1e20)
            cu_list.append(0.0)
        elif c.sense == "==":
            cl_list.append(0.0)
            cu_list.append(0.0)
        elif c.sense == ">=":
            cl_list.append(0.0)
            cu_list.append(1e20)
        else:
            raise ValueError(f"Unknown constraint sense: {c.sense}")

    return np.array(cl_list, dtype=np.float64), np.array(cu_list, dtype=np.float64)


def solve_nlp(
    evaluator: NLPEvaluator,
    x0: np.ndarray,
    constraint_bounds: Optional[list[tuple[float, float]]] = None,
    options: Optional[dict] = None,
) -> NLPResult:
    """
    Solve an NLP using cyipopt with JAX-compiled callbacks.

    Args:
        evaluator: NLPEvaluator providing objective/gradient/Hessian/constraint/Jacobian.
        x0: Initial point (n,).
        constraint_bounds: List of (cl, cu) per constraint. None to infer from model.
        options: Ipopt options dict (e.g., {'max_iter': 1000, 'tol': 1e-8}).

    Returns:
        NLPResult with solution
    """
    try:
        import cyipopt
    except ImportError:
        raise ImportError(
            "cyipopt is required for solve_nlp. Install it with:\n"
            "  pip install cyipopt\n"
            "Note: cyipopt requires the Ipopt C library to be installed."
        )

    opts = dict(options) if options else {}
    opts.setdefault("print_level", 0)

    n = evaluator.n_variables
    m = evaluator.n_constraints
    lb, ub = evaluator.variable_bounds

    # Constraint bounds
    if constraint_bounds is not None:
        cl = np.array([b[0] for b in constraint_bounds], dtype=np.float64)
        cu = np.array([b[1] for b in constraint_bounds], dtype=np.float64)
    elif m > 0:
        cl, cu = _infer_constraint_bounds(evaluator._model)
    else:
        cl = np.empty(0, dtype=np.float64)
        cu = np.empty(0, dtype=np.float64)

    callbacks = _IpoptCallbacks(evaluator)

    problem = cyipopt.Problem(
        n=n,
        m=m,
        problem_obj=callbacks,
        lb=lb,
        ub=ub,
        cl=cl,
        cu=cu,
    )

    # cyipopt requires native Python types (rejects numpy scalars).
    # Some options (e.g. max_wall_time) may not exist in older Ipopt versions.
    import logging as _logging

    import numpy as _np

    _logger = _logging.getLogger(__name__)
    for key, value in opts.items():
        try:
            if isinstance(value, (_np.floating, float)):
                problem.add_option(key, float(value))
            elif isinstance(value, (_np.integer, int)):
                problem.add_option(key, int(value))
            else:
                problem.add_option(key, value)
        except TypeError:
            _logger.debug("Ipopt option '%s' not accepted, skipping", key)

    t0 = time.perf_counter()
    x, info = problem.solve(x0.astype(np.float64))
    wall_time = time.perf_counter() - t0

    status_code = info["status"]
    status = _IPOPT_STATUS_MAP.get(status_code, SolveStatus.ERROR)

    multipliers = info.get("mult_g", None)
    if multipliers is not None and len(multipliers) == 0:
        multipliers = None

    return NLPResult(
        status=status,
        x=np.asarray(x),
        objective=float(info["obj_val"]),
        multipliers=np.asarray(multipliers) if multipliers is not None else None,
        iterations=0,  # Ipopt doesn't expose iteration count via this API
        wall_time=wall_time,
    )


def solve_nlp_from_model(
    model: Model,
    x0: Optional[np.ndarray] = None,
    options: Optional[dict] = None,
) -> NLPResult:
    """Convenience: create NLPEvaluator from model and solve.

    Args:
        model: A Model with objective and constraints set.
        x0: Initial point (n,). If None, uses midpoint of bounds clipped to [-100, 100].
        options: Ipopt options dict.

    Returns:
        NLPResult with solution.
    """
    evaluator = NLPEvaluator(model)

    if x0 is None:
        lb, ub = evaluator.variable_bounds
        lb_clipped = np.clip(lb, -100.0, 100.0)
        ub_clipped = np.clip(ub, -100.0, 100.0)
        x0 = 0.5 * (lb_clipped + ub_clipped)

    return solve_nlp(evaluator, x0, options=options)
