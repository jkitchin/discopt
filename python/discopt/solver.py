"""
Solver orchestrator: end-to-end Model.solve() via NLP-based spatial Branch & Bound.

Connects:
  - PyTreeManager (Rust B&B engine) for node management / branching / pruning
  - NLPEvaluator (JAX) for objective/gradient/Hessian/constraint/Jacobian
  - solve_nlp (cyipopt) for continuous relaxation solves at each node
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from scipy.optimize import minimize as scipy_minimize

from discopt._jax.alphabb import estimate_alpha as _estimate_alpha_jax
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt._rust import PyTreeManager
from discopt.constants import INFEASIBILITY_SENTINEL as _INFEASIBILITY_SENTINEL
from discopt.constants import SENTINEL_THRESHOLD as _SENTINEL_THRESHOLD
from discopt.constants import STARTING_POINT_CLIP as _SPC
from discopt.modeling.core import (
    Constraint,
    Model,
    SolveResult,
    VarType,
)
from discopt.solvers import SolveStatus
from discopt.solvers.nlp_ipopt import solve_nlp

logger = logging.getLogger(__name__)


class _AugmentedEvaluator:
    """Wraps an NLPEvaluator with additional linear cut constraints.

    When cuts are injected, the constraint function becomes:
        [original_constraints; A_cut @ x - b_cut]
    where each cut a^T x <= b becomes a^T x - b <= 0 (upper bounded by 0).
    For >= cuts, a^T x >= b becomes b - a^T x <= 0 (negated).
    """

    def __init__(self, evaluator, cut_pool):
        self._ev = evaluator
        self._cut_pool = cut_pool
        A, b, senses = cut_pool.to_constraint_arrays()
        self._n_cuts = A.shape[0]
        if self._n_cuts > 0:
            # Normalize: convert all cuts to <= form (a^T x - rhs <= 0)
            self._A = A.copy()
            self._b = b.copy()
            for k in range(self._n_cuts):
                if senses[k] == ">=":
                    self._A[k] = -self._A[k]
                    self._b[k] = -self._b[k]
                # "==" treated as <= (conservative)
        else:
            self._A = None
            self._b = None

    @property
    def n_constraints(self):
        return self._ev.n_constraints + self._n_cuts

    @property
    def n_variables(self):
        return self._ev.n_variables

    @property
    def variable_bounds(self):
        return self._ev.variable_bounds

    def evaluate_objective(self, x):
        return self._ev.evaluate_objective(x)

    def evaluate_gradient(self, x):
        return self._ev.evaluate_gradient(x)

    def evaluate_hessian(self, x):
        return self._ev.evaluate_hessian(x)

    def evaluate_constraints(self, x):
        orig = self._ev.evaluate_constraints(x)
        if self._n_cuts == 0:
            return orig
        cut_vals = self._A @ x - self._b
        return np.concatenate([orig, cut_vals])

    def evaluate_jacobian(self, x):
        orig = self._ev.evaluate_jacobian(x)
        if self._n_cuts == 0:
            return orig
        return np.vstack([orig, self._A])

    def evaluate_lagrangian_hessian(self, x, obj_factor, lambda_):
        # Cut constraints are linear so their Hessian contribution is zero
        m_orig = self._ev.n_constraints
        return self._ev.evaluate_lagrangian_hessian(x, obj_factor, lambda_[:m_orig])

    def get_augmented_constraint_bounds(self, original_bounds):
        """Return constraint bounds extended with cut bounds (all <= 0)."""
        if self._n_cuts == 0:
            return original_bounds
        if original_bounds is None:
            original_bounds = []
        cut_bounds = [(-1e20, 0.0)] * self._n_cuts
        return list(original_bounds) + cut_bounds

    def get_augmented_jax_bounds(self, g_l_jax, g_u_jax):
        """Return JAX constraint bound arrays extended with cut bounds."""
        import jax.numpy as jnp

        if self._n_cuts == 0:
            return g_l_jax, g_u_jax
        cut_gl = jnp.full(self._n_cuts, -1e20, dtype=jnp.float64)
        cut_gu = jnp.zeros(self._n_cuts, dtype=jnp.float64)
        if g_l_jax is not None:
            new_gl = jnp.concatenate([g_l_jax, cut_gl])
            new_gu = jnp.concatenate([g_u_jax, cut_gu])
        else:
            new_gl = cut_gl
            new_gu = cut_gu
        return new_gl, new_gu

    @property
    def _obj_fn(self):
        return self._ev._obj_fn

    @property
    def _cons_fn(self):
        if self._n_cuts == 0:
            return self._ev._cons_fn

        import jax.numpy as jnp

        orig_cons_fn = self._ev._cons_fn
        A_jax = jnp.array(self._A, dtype=jnp.float64)
        b_jax = jnp.array(self._b, dtype=jnp.float64)

        if orig_cons_fn is not None:

            def augmented_con(x):
                orig = orig_cons_fn(x)
                cut_vals = A_jax @ x - b_jax
                return jnp.concatenate([orig, cut_vals])
        else:

            def augmented_con(x):
                return A_jax @ x - b_jax

        return augmented_con


def _make_evaluator(model: Model):
    """Create the appropriate evaluator for the model."""
    return NLPEvaluator(model)


def _estimate_alpha_fd(evaluator, lb, ub, n_samples=30):
    """Estimate alphaBB convexification parameters via finite-difference Hessians.

    Samples random points in [lb, ub], computes the FD Hessian at each,
    finds the most negative eigenvalue, and returns alpha = max(0, -lambda_min/2 * 1.5 + 1e-6).
    """
    n = len(lb)
    rng = np.random.RandomState(123)

    # Clip infinite bounds for sampling
    lb_clip = np.clip(lb, -1e4, 1e4)
    ub_clip = np.clip(ub, -1e4, 1e4)
    span = ub_clip - lb_clip
    # Avoid zero-width dimensions
    span = np.maximum(span, 1e-8)

    eps = 1e-6
    global_min_eig = 0.0

    for _ in range(n_samples):
        x = lb_clip + rng.uniform(size=n) * span
        # Central-difference Hessian
        hess = np.empty((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i, n):
                x_pp = x.copy()
                x_pm = x.copy()
                x_mp = x.copy()
                x_mm = x.copy()
                x_pp[i] += eps
                x_pp[j] += eps
                x_pm[i] += eps
                x_pm[j] -= eps
                x_mp[i] -= eps
                x_mp[j] += eps
                x_mm[i] -= eps
                x_mm[j] -= eps
                fpp = evaluator.evaluate_objective(x_pp)
                fpm = evaluator.evaluate_objective(x_pm)
                fmp = evaluator.evaluate_objective(x_mp)
                fmm = evaluator.evaluate_objective(x_mm)
                h = (fpp - fpm - fmp + fmm) / (4.0 * eps * eps)
                hess[i, j] = h
                hess[j, i] = h
        eigs = np.linalg.eigvalsh(hess)
        global_min_eig = min(global_min_eig, float(eigs[0]))

    alpha_scalar = max(0.0, -global_min_eig / 2.0 * 1.5 + 1e-6)
    return np.full(n, alpha_scalar)


def _compute_alphabb_bound(evaluator, node_lb, node_ub, alpha):
    """Compute a valid lower bound by minimizing the alphaBB underestimator.

    L(x) = f(x) - sum_i alpha_i * (x_i - lb_i) * (ub_i - x_i)

    Returns the minimum of L over [node_lb, node_ub], or -inf on failure.
    """

    def underestimator(x):
        f_val = evaluator.evaluate_objective(x)
        perturbation = np.sum(alpha * (x - node_lb) * (node_ub - x))
        return f_val - perturbation

    # Multiple starting points for robustness
    lb_clip = np.clip(node_lb, -1e4, 1e4)
    ub_clip = np.clip(node_ub, -1e4, 1e4)
    mid = 0.5 * (lb_clip + ub_clip)
    bounds = list(zip(node_lb, node_ub))

    best_val = np.inf
    for x0 in [mid, lb_clip + 0.25 * (ub_clip - lb_clip), lb_clip + 0.75 * (ub_clip - lb_clip)]:
        try:
            result = scipy_minimize(underestimator, x0, method="L-BFGS-B", bounds=bounds)
            if result.fun < best_val:
                best_val = result.fun
        except (ValueError, ArithmeticError, RuntimeError):
            continue

    return best_val if np.isfinite(best_val) else -np.inf


def _extract_variable_info(model: Model):
    """Extract flat variable bounds and integer variable group info from a model.

    Returns:
        n_vars: total number of scalar decision variables
        lb: flat lower bounds array
        ub: flat upper bounds array
        int_var_offsets: list of flat offsets for integer/binary variable groups
        int_var_sizes: list of sizes for integer/binary variable groups
    """
    lb_parts = []
    ub_parts = []
    int_var_offsets = []
    int_var_sizes = []
    offset = 0

    for v in model._variables:
        lb_parts.append(v.lb.flatten())
        ub_parts.append(v.ub.flatten())
        if v.var_type in (VarType.BINARY, VarType.INTEGER):
            int_var_offsets.append(offset)
            int_var_sizes.append(v.size)
        offset += v.size

    n_vars = offset
    lb = np.concatenate(lb_parts) if lb_parts else np.array([], dtype=np.float64)
    ub = np.concatenate(ub_parts) if ub_parts else np.array([], dtype=np.float64)

    return n_vars, lb, ub, int_var_offsets, int_var_sizes


def _check_constraint_feasibility(evaluator, x, cl_list, cu_list, tol=1e-6):
    """Return True if x satisfies all constraints within tolerance.

    Parameters
    ----------
    evaluator : NLPEvaluator or _AugmentedEvaluator
        Constraint evaluator with ``evaluate_constraints`` and ``n_constraints``.
    x : np.ndarray
        Candidate solution vector.
    cl_list : list[float]
        Lower bounds on constraints (use -1e20 for no lower bound).
    cu_list : list[float]
        Upper bounds on constraints (use 1e20 for no upper bound).
    tol : float
        Feasibility tolerance.

    Returns
    -------
    bool
        True if all constraints are satisfied within *tol*.
    """
    if evaluator.n_constraints == 0:
        return True
    cons = evaluator.evaluate_constraints(np.asarray(x, dtype=np.float64))
    cl = np.array(cl_list, dtype=np.float64)
    cu = np.array(cu_list, dtype=np.float64)
    max_viol = max(float(np.max(cons - cu)), float(np.max(cl - cons)))
    return max_viol <= tol


def _infer_constraint_bounds(model: Model):
    """Infer (cl, cu) arrays from model constraint senses.

    The NLPEvaluator compiles constraints as `body - rhs`, so:
      - '<=' constraints: body - rhs <= 0 => cl = -inf, cu = 0
      - '==' constraints: body - rhs == 0 => cl = 0, cu = 0
      - '>=' constraints: body - rhs >= 0 => cl = 0, cu = inf
    """
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

    return cl_list, cu_list


def _generate_starting_points(node_lb, node_ub, n_random=2):
    """Generate diverse starting points for multi-start NLP at root node."""
    lb_clipped = np.clip(node_lb, -_SPC, _SPC)
    ub_clipped = np.clip(node_ub, -_SPC, _SPC)
    span = ub_clipped - lb_clipped

    points = [
        0.5 * (lb_clipped + ub_clipped),  # midpoint
        lb_clipped + 0.25 * span,  # lower-quarter
        lb_clipped + 0.75 * span,  # upper-quarter
    ]

    rng = np.random.RandomState(42)
    for _ in range(n_random):
        points.append(lb_clipped + rng.uniform(size=lb_clipped.shape) * span)

    return points


def _solve_root_node_multistart(
    evaluator,
    node_lb,
    node_ub,
    constraint_bounds,
    options,
    nlp_solver,
    n_random=2,
):
    """Solve root NLP relaxation from multiple starting points.

    On nonconvex problems, different starting points can converge to
    different local minima. Multi-start at the root increases the
    chance of finding the global optimum for the initial bound/incumbent.
    """
    starting_points = _generate_starting_points(node_lb, node_ub, n_random=n_random)

    best_result = None
    best_obj = np.inf

    for x0 in starting_points:
        nlp_result = _solve_node_nlp(
            evaluator,
            x0,
            node_lb,
            node_ub,
            constraint_bounds,
            options,
            nlp_solver=nlp_solver,
        )
        if nlp_result.status in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT):
            if nlp_result.objective < best_obj:
                best_obj = nlp_result.objective
                best_result = nlp_result

    if best_result is not None:
        return best_result
    # All failed — return the last result
    return nlp_result


def _solve_root_node_multistart_ipm(
    evaluator,
    node_lb,
    node_ub,
    constraint_bounds,
    g_l_jax,
    g_u_jax,
    options,
    n_random=2,
):
    """Solve root NLP relaxation from multiple starting points via vmap'd IPM.

    Uses jax.vmap to solve all starting points in parallel, giving ~Nx speedup
    over the serial loop when using the pure-JAX IPM backend.
    """
    import jax.numpy as jnp

    from discopt._jax.ipm import IPMOptions, solve_nlp_batch
    from discopt.solvers import NLPResult

    starting_points = _generate_starting_points(node_lb, node_ub, n_random=n_random)
    n_starts = len(starting_points)

    obj_fn = evaluator._obj_fn
    m = evaluator.n_constraints
    con_fn = evaluator._cons_fn if m > 0 else None

    # Stack starting points into (n_starts, n_vars) batch
    x0_batch = jnp.array(np.stack(starting_points), dtype=jnp.float64)
    # Broadcast node bounds to (n_starts, n_vars)
    xl_batch = jnp.broadcast_to(jnp.array(node_lb, dtype=jnp.float64), (n_starts, len(node_lb)))
    xu_batch = jnp.broadcast_to(jnp.array(node_ub, dtype=jnp.float64), (n_starts, len(node_ub)))

    ipm_opts = IPMOptions(max_iter=int(options.get("max_iter", 200)))

    try:
        state = solve_nlp_batch(
            obj_fn, con_fn, x0_batch, xl_batch, xu_batch, g_l_jax, g_u_jax, ipm_opts
        )
    except Exception as e:
        logger.debug("Root multistart batch IPM failed: %s", e)
        # Fall back: return infeasible sentinel
        return NLPResult(
            status=SolveStatus.ERROR,
            x=np.asarray(starting_points[0]),
            objective=_INFEASIBILITY_SENTINEL,
        )

    # Unpack batched results: pick best converged solution
    converged = np.asarray(state.converged)  # (n_starts,)
    obj_vals = np.asarray(state.obj)  # (n_starts,)
    x_vals = np.asarray(state.x)  # (n_starts, n_vars)

    # Mask: converged == 1 (optimal), 2 (acceptable), or 3 (iter limit)
    feasible_mask = (converged == 1) | (converged == 2) | (converged == 3)

    if np.any(feasible_mask):
        # Among feasible, pick the one with lowest objective
        masked_obj = np.where(feasible_mask, obj_vals, np.inf)
        best_idx = int(np.argmin(masked_obj))
        return NLPResult(
            status=SolveStatus.OPTIMAL,
            x=x_vals[best_idx],
            objective=float(obj_vals[best_idx]),
        )
    else:
        # All failed
        return NLPResult(
            status=SolveStatus.ERROR,
            x=x_vals[0],
            objective=_INFEASIBILITY_SENTINEL,
        )


def _unpack_solution(model: Model, x_flat: np.ndarray):
    """Convert flat solution vector to {var_name: array} dict."""
    result = {}
    offset = 0
    for v in model._variables:
        size = v.size
        val = x_flat[offset : offset + size]
        if v.shape == () or v.shape == (1,):
            result[v.name] = val.reshape(v.shape) if v.shape == () else val
        else:
            result[v.name] = val.reshape(v.shape)
        offset += size
    return result


def _strong_branch_lp(
    evaluator,
    solution: np.ndarray,
    node_lb: np.ndarray,
    node_ub: np.ndarray,
    candidate_var_indices: np.ndarray,
    parent_lb: float,
    max_candidates: int = 5,
    time_limit: float = 1.0,
) -> Optional[int]:
    """Perform strong branching via LP relaxations for unreliable candidates.

    For each candidate variable, solves two LP relaxations (down-branch and
    up-branch) and returns the variable index with the best product score.

    Uses the NLP evaluator's gradient at the current solution as LP objective
    (first-order Taylor approximation), with node bounds as variable bounds.

    Parameters
    ----------
    evaluator : NLPEvaluator or _AugmentedEvaluator
        Evaluator for gradient/constraint computation.
    solution : np.ndarray
        Current relaxation solution at this node.
    node_lb, node_ub : np.ndarray
        Variable bounds for this node.
    candidate_var_indices : np.ndarray
        Flat indices of candidate variables to evaluate.
    parent_lb : float
        Parent node's relaxation lower bound.
    max_candidates : int
        Maximum number of candidates to evaluate (most fractional first).
    time_limit : float
        Total time budget for all LP solves.

    Returns
    -------
    int or None
        Best variable index to branch on, or None if no valid candidate.
    """
    from discopt.solvers.lp_highs import solve_lp

    n_vars = len(solution)
    n_candidates = len(candidate_var_indices)
    if n_candidates == 0:
        return None

    # Limit candidates — prioritize those closest to 0.5 fractionality
    if n_candidates > max_candidates:
        fracs = np.array([solution[i] - np.floor(solution[i]) for i in candidate_var_indices])
        closeness_to_half = 0.5 - np.abs(fracs - 0.5)
        top_k = np.argsort(-closeness_to_half)[:max_candidates]
        candidate_var_indices = candidate_var_indices[top_k]

    # LP objective: gradient of the objective at the current solution.
    try:
        c = np.asarray(evaluator.evaluate_gradient(solution), dtype=np.float64).ravel()
    except Exception:
        return None

    # LP constraints from the evaluator's Jacobian (linearized).
    A_ub = None
    b_ub = None
    try:
        if evaluator.n_constraints > 0:
            g_vals = np.asarray(evaluator.evaluate_constraints(solution), dtype=np.float64).ravel()
            J = np.asarray(evaluator.evaluate_jacobian(solution), dtype=np.float64)
            if J.ndim == 1:
                J = J.reshape(1, -1)
            # Linearized constraints: J @ (x - x0) + g(x0) <= 0
            # => J @ x <= J @ x0 - g(x0)
            A_ub = J
            b_ub = J @ solution - g_vals
    except Exception:
        pass  # Proceed without constraints (just variable bounds)

    bounds_list = [(float(node_lb[j]), float(node_ub[j])) for j in range(n_vars)]

    best_var = None
    best_score = -np.inf
    t_start = time.perf_counter()
    per_solve_limit = max(0.05, time_limit / (2 * len(candidate_var_indices) + 1))

    for var_idx in candidate_var_indices:
        if time.perf_counter() - t_start > time_limit:
            break

        var_idx = int(var_idx)
        val = solution[var_idx]
        floor_val = np.floor(val)

        # Down branch: x_i <= floor(val)
        down_bounds = list(bounds_list)
        down_bounds[var_idx] = (down_bounds[var_idx][0], floor_val)
        try:
            down_result = solve_lp(
                c, A_ub=A_ub, b_ub=b_ub, bounds=down_bounds, time_limit=per_solve_limit
            )
            down_obj = down_result.objective
            down_lb = (
                float(down_obj)
                if down_result.status == SolveStatus.OPTIMAL and down_obj is not None
                else np.inf
            )
        except Exception:
            down_lb = np.inf

        # Up branch: x_i >= ceil(val)
        up_bounds = list(bounds_list)
        up_bounds[var_idx] = (floor_val + 1.0, up_bounds[var_idx][1])
        try:
            up_result = solve_lp(
                c, A_ub=A_ub, b_ub=b_ub, bounds=up_bounds, time_limit=per_solve_limit
            )
            up_obj = up_result.objective
            up_lb = (
                float(up_obj)
                if up_result.status == SolveStatus.OPTIMAL and up_obj is not None
                else np.inf
            )
        except Exception:
            up_lb = np.inf

        # Product score: improvement in each direction
        down_gain = max(0.0, down_lb - parent_lb) if np.isfinite(down_lb) else 1e6
        up_gain = max(0.0, up_lb - parent_lb) if np.isfinite(up_lb) else 1e6
        score = (1e-6 + down_gain) * (1e-6 + up_gain)

        if score > best_score:
            best_score = score
            best_var = var_idx

    return best_var


def _is_pure_continuous(model: Model) -> bool:
    """Check if model has no integer/binary variables."""
    return all(v.var_type == VarType.CONTINUOUS for v in model._variables)


def solve_model(
    model: Model,
    time_limit: float = 3600.0,
    gap_tolerance: float = 1e-4,
    threads: int = 1,
    deterministic: bool = True,
    batch_size: int = 16,
    strategy: str = "best_first",
    max_nodes: int = 100_000,
    ipopt_options: Optional[dict] = None,
    nlp_solver: str = "ipm",
    sparse: Optional[bool] = None,
    cutting_planes: bool = False,
    partitions: int = 0,
    branching_policy: str = "fractional",
    use_learned_relaxations: bool = False,
    mccormick_bounds: str = "none",
    gdp_method: str = "big-m",
) -> SolveResult:
    """
    Solve a Model via NLP-based spatial Branch & Bound.

    At each B&B node the solver: (1) solves a continuous NLP relaxation
    with node-tightened bounds, (2) optionally generates OA cutting planes,
    (3) prunes if infeasible, (4) fathoms and updates incumbent if
    integer-feasible, or (5) branches on the most fractional integer variable.

    This function is called by :meth:`Model.solve` and is not typically
    invoked directly.

    Parameters
    ----------
    model : Model
        A Model with objective and constraints set.
    time_limit : float, default 3600.0
        Wall-clock time limit in seconds.
    gap_tolerance : float, default 1e-4
        Relative optimality gap tolerance for termination.
    threads : int, default 1
        Number of CPU threads (reserved for future use).
    deterministic : bool, default True
        Ensure deterministic results.
    batch_size : int, default 16
        Number of B&B nodes to export per iteration.
    strategy : str, default "best_first"
        Node selection strategy: ``"best_first"`` or ``"depth_first"``.
    max_nodes : int, default 100_000
        Maximum number of B&B nodes before stopping.
    ipopt_options : dict, optional
        Options passed to cyipopt (only used when ``nlp_solver="ipopt"``).
    nlp_solver : str, default "ipm"
        NLP solver backend: ``"ripopt"`` (Rust IPM via PyO3),
        ``"ipopt"`` (cyipopt), ``"ipm"`` (pure-JAX IPM), or
        ``"sparse_ipm"`` (sparse KKT + scipy direct solve).
    sparse : bool or None, default None
        Force sparse (True) or dense (False) Jacobian evaluation.
        If None, auto-selects based on problem size and density.
    cutting_planes : bool, default False
        Enable outer-approximation cut generation after NLP relaxation solves.
    partitions : int, default 0
        Number of piecewise McCormick partitions (0 = standard convex
        relaxation, k > 0 = k partitions for tighter relaxations).
    branching_policy : str, default "fractional"
        Variable selection: ``"fractional"`` (most-fractional, default)
        or ``"gnn"`` (GNN scoring hook; Rust handles actual branching).
    use_learned_relaxations : bool, default False
        Use ICNN-based learned convex relaxations instead of standard
        McCormick. Requires ``pip install discopt[gnn]`` (equinox + optax).
        Falls back to standard McCormick for unsupported operations.
    mccormick_bounds : str, default "none"
        McCormick relaxation lower-bounding strategy:
        ``"auto"`` selects ``"nlp"`` when a JAX objective is
        available, ``"none"`` otherwise,
        ``"nlp"`` solves a convex NLP over the McCormick relaxation
        (gives valid lower bounds for pruning),
        ``"midpoint"`` evaluates the convex underestimator at midpoint
        (heuristic, not a valid global lower bound — use with caution),
        ``"none"`` disables (default).
    gdp_method : str, default "big-m"
        Reformulation method for disjunctive constraints:
        ``"big-m"`` (default) or ``"hull"`` (convex hull).

    Returns
    -------
    SolveResult
        Contains solution values, objective, gap, node count, and
        per-layer profiling times (Rust, JAX, Python).
    """
    # --- Enforce float64 precision ---
    # JAX defaults to float32 unless JAX_ENABLE_X64=1 is set *before* importing
    # JAX.  All solver tolerances assume float64; float32 silently degrades
    # convergence and may return incorrect solutions.
    import jax.numpy as jnp

    if jnp.zeros(1).dtype != jnp.float64:
        import warnings

        warnings.warn(
            "JAX is running in float32 mode.  Set the environment variable "
            "JAX_ENABLE_X64=1 *before* importing JAX for full solver precision.  "
            "Results may be inaccurate.",
            stacklevel=2,
        )

    # --- GDP reformulation: convert indicator/disjunctive/SOS to standard MINLP ---
    from discopt._jax.gdp_reformulate import reformulate_gdp

    model = reformulate_gdp(model, method=gdp_method)

    # --- Build Rust model representation for FBBT ---
    _model_repr = None
    try:
        from discopt._rust import model_to_repr

        _model_repr = model_to_repr(model)
    except Exception:
        pass  # FBBT bindings unavailable; skip

    # --- Learned relaxation registry (opt-in) ---
    import warnings

    _learned_registry = None
    _relax_mode = "standard"
    if use_learned_relaxations:
        try:
            from discopt._jax.learned_relaxations import load_pretrained_registry

            _learned_registry = load_pretrained_registry()
            if len(_learned_registry) > 0:
                _relax_mode = "learned"
            else:
                warnings.warn(
                    "No pretrained learned relaxation models found. "
                    "Falling back to standard McCormick.",
                    stacklevel=2,
                )
        except ImportError:
            warnings.warn(
                "Learned relaxations require pip install discopt[gnn] "
                "(equinox + optax). Falling back to standard McCormick.",
                stacklevel=2,
            )

    t_start = time.perf_counter()
    rust_time = 0.0
    jax_time = 0.0

    if nlp_solver == "ripopt":
        logger.info("Using ripopt (Rust interior point method)")
    elif nlp_solver == "sparse_ipm":
        logger.info("Using sparse IPM (scipy direct solve)")
    elif nlp_solver == "ipm":
        logger.info("Using discopt IPM (pure-JAX interior point method)")
    else:
        logger.info("Using Ipopt (via cyipopt)")

    # --- Problem classification: dispatch LP/QP to specialized solvers ---
    try:
        from discopt._jax.problem_classifier import ProblemClass, classify_problem

        problem_class = classify_problem(model)
    except Exception as e:
        logger.debug("Problem classification failed: %s", e)
        problem_class = None

    if problem_class is not None:
        if problem_class == ProblemClass.LP:
            return _solve_lp(model, t_start)
        elif problem_class == ProblemClass.QP:
            return _solve_qp(model, t_start)
        elif problem_class == ProblemClass.MILP:
            return _solve_milp_bb(
                model,
                time_limit,
                gap_tolerance,
                batch_size,
                strategy,
                max_nodes,
                t_start,
            )
        elif problem_class == ProblemClass.MIQP:
            return _solve_miqp_bb(
                model,
                time_limit,
                gap_tolerance,
                batch_size,
                strategy,
                max_nodes,
                t_start,
            )

    # --- Pure continuous: solve directly with NLP, no B&B needed ---
    if _is_pure_continuous(model):
        return _solve_continuous(model, time_limit, ipopt_options, t_start, nlp_solver)

    # --- Extract variable info ---
    n_vars, lb, ub, int_offsets, int_sizes = _extract_variable_info(model)

    # --- Create PyTreeManager (Rust) ---
    t_rust_start = time.perf_counter()
    tree = PyTreeManager(
        n_vars,
        lb.tolist(),
        ub.tolist(),
        int_offsets,
        int_sizes,
        strategy,
    )
    tree.initialize()
    rust_time += time.perf_counter() - t_rust_start

    # --- Compile NLP evaluator ---
    t_jax_start = time.perf_counter()
    evaluator = _make_evaluator(model)
    jax_time += time.perf_counter() - t_jax_start

    # --- Infer constraint bounds ---
    cl_list, cu_list = _infer_constraint_bounds(model)
    constraint_bounds = list(zip(cl_list, cu_list)) if cl_list else None

    # Pre-compute constraint bounds as JAX arrays for batch IPM
    g_l_jax = None
    g_u_jax = None
    if nlp_solver == "ipm" and cl_list:
        import jax.numpy as jnp

        g_l_jax = jnp.array(cl_list, dtype=jnp.float64)
        g_u_jax = jnp.array(cu_list, dtype=jnp.float64)

    # --- Prepare cut generation if enabled ---
    _generate_cuts = None
    _bilinear_terms = None
    _constraint_senses = None
    _cut_pool = None
    if cutting_planes:
        from discopt._jax.cutting_planes import (
            CutPool,
            detect_bilinear_terms,
            generate_cuts_at_node,
        )

        _generate_cuts = generate_cuts_at_node
        _bilinear_terms = detect_bilinear_terms(model)
        _cut_pool = CutPool(max_cuts=500)
        _constraint_senses = [c.sense for c in model._constraints if isinstance(c, Constraint)]

    # --- Convexity detection (Phase E) ---
    # Use the expression DAG convexity detector to:
    # (E2) Skip relaxation overhead for fully convex subproblems
    # (E3) Enable OA cuts per-constraint (not just for affine constraints)
    _model_is_convex = False
    _oa_enabled = False
    _convex_constraint_mask = None
    try:
        from discopt._jax.convexity import classify_model as _classify_model

        _model_is_convex, _convex_constraint_mask = _classify_model(model)
        if _model_is_convex:
            logger.info("Model detected as convex — NLP solutions are valid lower bounds")
        if cutting_planes and any(_convex_constraint_mask):
            _oa_enabled = True
    except Exception as exc:
        logger.debug("Convexity detection failed: %s", exc)
        if cutting_planes and model._constraints:
            _convex_constraint_mask = [False] * len(model._constraints)

    # --- Default Ipopt options ---
    opts = dict(ipopt_options) if ipopt_options else {}
    opts.setdefault("print_level", 0)
    opts.setdefault("max_iter", 3000)
    opts.setdefault("tol", 1e-7)

    # --- Augmented constraint function with cuts (updated each iteration) ---
    _augmented_evaluator = None

    # --- AlphaBB convexification for nonconvex models ---
    # (E2) Skip alphaBB entirely for convex models — NLP gives valid bounds.
    _alphabb_alpha = None
    _use_alphabb = False
    if n_vars <= 50 and not _model_is_convex:
        if hasattr(evaluator, "_obj_fn"):
            # JAX-native path: uses jax.hessian + jax.vmap (10-100x faster)
            try:
                _alphabb_alpha = np.asarray(
                    _estimate_alpha_jax(evaluator._obj_fn, lb, ub, n_samples=100)
                )
                _use_alphabb = bool(np.any(_alphabb_alpha > 1e-8))
            except (ValueError, ArithmeticError, RuntimeError) as e:
                logger.debug("JAX alphaBB estimation failed: %s", e)

    # --- McCormick relaxation bounds ---
    _mc_obj_eval = None  # BatchRelaxationEvaluator for midpoint bounds
    _mc_obj_relax_fn = None  # raw relaxation fn for NLP bounds
    _mc_con_relax_fns = None
    _mc_con_senses = None
    _mc_negate = False
    _mc_mode = mccormick_bounds

    if _mc_mode == "auto":
        if _model_is_convex:
            # (E2) For convex models, NLP relaxation already gives valid
            # lower bounds — no need for McCormick relaxation overhead.
            _mc_mode = "none"
        elif model._objective is not None:
            # "nlp" mode solves a convex relaxation NLP for valid lower bounds.
            # "midpoint" is a heuristic (not a valid bound) and can cause
            # incorrect pruning — do not use it for bound tightening.
            _mc_mode = "nlp"
        else:
            _mc_mode = "none"

    if _mc_mode in ("midpoint", "nlp") and model._objective is not None:
        from discopt._jax.batch_evaluator import BatchRelaxationEvaluator
        from discopt._jax.relaxation_compiler import (
            compile_constraint_relaxation,
            compile_objective_relaxation,
        )
        from discopt.modeling.core import ObjectiveSense

        try:
            _mc_obj_relax_fn = compile_objective_relaxation(
                model,
                partitions=partitions,
                mode=_relax_mode,
                learned_registry=_learned_registry,
            )
            _mc_obj_eval = BatchRelaxationEvaluator(_mc_obj_relax_fn, n_vars)
            _mc_negate = model._objective.sense == ObjectiveSense.MAXIMIZE

            if _mc_mode == "nlp" and model._constraints:
                _mc_con_relax_fns = []
                _mc_con_senses = []
                for c in model._constraints:
                    if isinstance(c, Constraint):
                        _mc_con_relax_fns.append(
                            compile_constraint_relaxation(
                                c,
                                model,
                                partitions=partitions,
                                mode=_relax_mode,
                                learned_registry=_learned_registry,
                            )
                        )
                        _mc_con_senses.append(c.sense)
        except Exception as e:
            logger.debug("McCormick relaxation setup failed: %s", e)
            _mc_obj_eval = None
            _mc_obj_relax_fn = None

    # --- Feasibility pump at root ---
    # Try to find an integer-feasible incumbent before B&B starts.
    _fp_ran = False

    # --- B&B loop ---
    iteration = 0
    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            break

        # Export batch from Rust tree
        t_rust_start = time.perf_counter()
        batch_lb, batch_ub, batch_ids, batch_psols = tree.export_batch(batch_size)
        rust_time += time.perf_counter() - t_rust_start

        n_batch = len(batch_ids)
        if n_batch == 0:
            break

        # Solve NLP relaxation for each node in the batch
        t_jax_start = time.perf_counter()

        # Use augmented evaluator with cuts if available
        _active_evaluator = evaluator
        _active_cb = constraint_bounds
        _active_gl = g_l_jax
        _active_gu = g_u_jax
        if _cut_pool is not None and len(_cut_pool) > 0:
            _augmented_evaluator = _AugmentedEvaluator(evaluator, _cut_pool)
            _active_evaluator = _augmented_evaluator
            _active_cb = _augmented_evaluator.get_augmented_constraint_bounds(constraint_bounds)
            if nlp_solver == "ipm" and hasattr(evaluator, "_obj_fn"):
                _active_gl, _active_gu = _augmented_evaluator.get_augmented_jax_bounds(
                    g_l_jax, g_u_jax
                )

        _use_ipm_batch = nlp_solver == "ipm" and hasattr(evaluator, "_obj_fn")
        if _use_ipm_batch and n_batch > 1:
            result_ids, result_lbs, result_sols, result_feas = _solve_batch_ipm(
                _active_evaluator,
                batch_lb,
                batch_ub,
                batch_ids,
                n_vars,
                _active_cb,
                opts,
                _active_gl,
                _active_gu,
                batch_psols=batch_psols,
            )
            # Constraint feasibility post-check for batch IPM results
            if cl_list:
                for i in range(n_batch):
                    if result_lbs[i] < _SENTINEL_THRESHOLD:
                        if not _check_constraint_feasibility(
                            _active_evaluator,
                            result_sols[i],
                            cl_list,
                            cu_list,
                        ):
                            result_lbs[i] = _INFEASIBILITY_SENTINEL
                            logger.debug(
                                "Batch node %d: IPM solution violates "
                                "constraints, marking infeasible",
                                int(batch_ids[i]),
                            )
            # Tighten lower bounds with alphaBB underestimator
            if _use_alphabb:
                for i in range(n_batch):
                    if result_lbs[i] < _SENTINEL_THRESHOLD:
                        try:
                            node_lb_i = np.array(batch_lb[i])
                            node_ub_i = np.array(batch_ub[i])
                            relax_lb = _compute_alphabb_bound(
                                evaluator, node_lb_i, node_ub_i, _alphabb_alpha
                            )
                            result_lbs[i] = max(result_lbs[i], relax_lb)
                        except (ValueError, ArithmeticError, RuntimeError) as e:
                            logger.debug("alphaBB bound failed at node %d: %s", i, e)
            # Tighten lower bounds with McCormick relaxation
            if _mc_obj_eval is not None:
                try:
                    import jax.numpy as jnp

                    lb_jax = jnp.array(batch_lb, dtype=jnp.float64)
                    ub_jax = jnp.array(batch_ub, dtype=jnp.float64)
                    if _mc_mode == "nlp" and _mc_obj_relax_fn is not None:
                        from discopt._jax.mccormick_nlp import solve_mccormick_batch

                        mc_lbs = np.asarray(
                            solve_mccormick_batch(
                                _mc_obj_relax_fn,
                                _mc_con_relax_fns,
                                _mc_con_senses,
                                lb_jax,
                                ub_jax,
                                negate=_mc_negate,
                            )
                        )
                    else:
                        from discopt._jax.mccormick_nlp import (
                            evaluate_midpoint_bound_batch,
                        )

                        assert _mc_obj_relax_fn is not None
                        mc_lbs = np.asarray(
                            evaluate_midpoint_bound_batch(
                                _mc_obj_relax_fn,
                                lb_jax,
                                ub_jax,
                                negate=_mc_negate,
                            )
                        )
                    for i in range(n_batch):
                        if result_lbs[i] < _SENTINEL_THRESHOLD and np.isfinite(mc_lbs[i]):
                            result_lbs[i] = max(result_lbs[i], float(mc_lbs[i]))
                except (ValueError, ArithmeticError, RuntimeError) as e:
                    logger.debug("Batch McCormick bound failed: %s", e)
        else:
            result_ids = np.empty(n_batch, dtype=np.int64)
            result_lbs = np.empty(n_batch, dtype=np.float64)
            result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
            result_feas = np.empty(n_batch, dtype=bool)

            for i in range(n_batch):
                node_lb = np.array(batch_lb[i])
                node_ub = np.array(batch_ub[i])

                if iteration == 0:
                    if _use_ipm_batch:
                        nlp_result = _solve_root_node_multistart_ipm(
                            _active_evaluator,
                            node_lb,
                            node_ub,
                            _active_cb,
                            _active_gl,
                            _active_gu,
                            opts,
                        )
                    else:
                        nlp_result = _solve_root_node_multistart(
                            _active_evaluator,
                            node_lb,
                            node_ub,
                            _active_cb,
                            opts,
                            nlp_solver,
                        )
                else:
                    # Warm-start from parent solution if available
                    psol_i = np.array(batch_psols[i])
                    if not np.any(np.isnan(psol_i)):
                        # Clip parent solution into child's bounds
                        x0 = np.clip(psol_i, node_lb, node_ub)
                    else:
                        lb_clipped = np.clip(node_lb, -_SPC, _SPC)
                        ub_clipped = np.clip(node_ub, -_SPC, _SPC)
                        x0 = 0.5 * (lb_clipped + ub_clipped)
                    nlp_result = _solve_node_nlp(
                        _active_evaluator,
                        x0,
                        node_lb,
                        node_ub,
                        _active_cb,
                        opts,
                        nlp_solver=nlp_solver,
                    )

                result_ids[i] = int(batch_ids[i])

                if nlp_result.status in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT):
                    nlp_lb = nlp_result.objective
                    if _use_alphabb:
                        try:
                            relax_lb = _compute_alphabb_bound(
                                evaluator, node_lb, node_ub, _alphabb_alpha
                            )
                            nlp_lb = max(nlp_lb, relax_lb)
                        except (ValueError, ArithmeticError, RuntimeError) as e:
                            logger.debug("alphaBB bound failed: %s", e)
                    # McCormick relaxation bound
                    if _mc_obj_relax_fn is not None:
                        try:
                            import jax.numpy as jnp

                            lb_j = jnp.array(node_lb, dtype=jnp.float64)
                            ub_j = jnp.array(node_ub, dtype=jnp.float64)
                            if _mc_mode == "nlp":
                                from discopt._jax.mccormick_nlp import (
                                    solve_mccormick_relaxation_nlp,
                                )

                                mc_lb = solve_mccormick_relaxation_nlp(
                                    _mc_obj_relax_fn,
                                    _mc_con_relax_fns,
                                    _mc_con_senses,
                                    lb_j,
                                    ub_j,
                                    negate=_mc_negate,
                                )
                            else:
                                from discopt._jax.mccormick_nlp import (
                                    evaluate_midpoint_bound,
                                )

                                mc_lb = evaluate_midpoint_bound(
                                    _mc_obj_relax_fn,
                                    lb_j,
                                    ub_j,
                                    negate=_mc_negate,
                                )
                            if np.isfinite(mc_lb):
                                nlp_lb = max(nlp_lb, mc_lb)
                        except (ValueError, ArithmeticError, RuntimeError) as e:
                            logger.debug("McCormick bound failed: %s", e)
                    # Constraint feasibility post-check: reject NLP solutions
                    # that violate constraints (the Rust B&B tree only checks
                    # integrality, not constraint satisfaction).
                    if cl_list and not _check_constraint_feasibility(
                        _active_evaluator, nlp_result.x, cl_list, cu_list
                    ):
                        nlp_lb = _INFEASIBILITY_SENTINEL
                        logger.debug(
                            "Node %d: NLP solution violates constraints, marking infeasible",
                            int(batch_ids[i]),
                        )
                    result_lbs[i] = nlp_lb
                    result_sols[i] = nlp_result.x
                    result_feas[i] = False
                else:
                    result_lbs[i] = _INFEASIBILITY_SENTINEL
                    lb_clipped = np.clip(node_lb, -_SPC, _SPC)
                    ub_clipped = np.clip(node_ub, -_SPC, _SPC)
                    result_sols[i] = 0.5 * (lb_clipped + ub_clipped)
                    result_feas[i] = False
        jax_time += time.perf_counter() - t_jax_start

        # --- Optional GNN branching scoring ---
        # GNN computes variable scores and passes hints to Rust TreeManager,
        # which uses them instead of most-fractional branching.
        if branching_policy == "gnn":
            from discopt._jax.gnn_policy import select_branch_variable_gnn
            from discopt._jax.problem_graph import build_graph

            hint_node_ids = []
            hint_var_indices = []
            for i in range(n_batch):
                if result_lbs[i] < _SENTINEL_THRESHOLD:
                    node_lb_i = np.array(batch_lb[i])
                    node_ub_i = np.array(batch_ub[i])
                    graph = build_graph(model, result_sols[i], node_lb_i, node_ub_i)
                    var_idx = select_branch_variable_gnn(graph, params=None)
                    if var_idx is not None:
                        hint_node_ids.append(int(batch_ids[i]))
                        hint_var_indices.append(var_idx)
            if hint_node_ids:
                tree.set_branch_hints(
                    np.array(hint_node_ids, dtype=np.int64),
                    np.array(hint_var_indices, dtype=np.int64),
                )

        # --- Strong branching for unreliable pseudocost candidates ---
        # For nodes without GNN hints, use LP-based strong branching when
        # pseudocost observations are insufficient for reliable branching.
        if branching_policy != "gnn" and iteration > 0:
            sb_hint_ids: list[int] = []
            sb_hint_vars = []
            rel_thresh = tree.reliability_threshold()
            for i in range(n_batch):
                if result_lbs[i] >= _SENTINEL_THRESHOLD:
                    continue  # skip infeasible nodes
                node_id = int(batch_ids[i])
                sol_i = result_sols[i]
                var_indices, _frac_parts, obs_counts, _scores = tree.score_candidates(sol_i)
                if len(var_indices) == 0:
                    continue
                # Identify unreliable candidates
                unreliable_mask = obs_counts < rel_thresh
                if not np.any(unreliable_mask):
                    continue  # all candidates are reliable, pseudocosts will work
                unreliable_vars = np.asarray(var_indices)[unreliable_mask]
                try:
                    best_var = _strong_branch_lp(
                        _active_evaluator,
                        sol_i,
                        np.array(batch_lb[i]),
                        np.array(batch_ub[i]),
                        unreliable_vars,
                        parent_lb=float(result_lbs[i]),
                        max_candidates=5,
                        time_limit=0.5,
                    )
                    if best_var is not None:
                        sb_hint_ids.append(node_id)
                        sb_hint_vars.append(best_var)
                except Exception as e:
                    logger.debug("Strong branching failed for node %d: %s", node_id, e)
            if sb_hint_ids:
                tree.set_branch_hints(
                    np.array(sb_hint_ids, dtype=np.int64),
                    np.array(sb_hint_vars, dtype=np.int64),
                )

        # --- Optional cut generation (OA + RLT + lift-and-project) ---
        if cutting_planes and _generate_cuts is not None and _cut_pool is not None:
            for i in range(n_batch):
                if result_lbs[i] < _SENTINEL_THRESHOLD:  # skip infeasible nodes
                    node_lb_i = np.array(batch_lb[i])
                    node_ub_i = np.array(batch_ub[i])
                    new_cuts = _generate_cuts(
                        evaluator,
                        model,
                        result_sols[i],
                        node_lb_i,
                        node_ub_i,
                        constraint_senses=_constraint_senses,
                        bilinear_terms=_bilinear_terms,
                        oa_enabled=_oa_enabled,
                        convex_constraint_mask=_convex_constraint_mask,
                    )
                    _cut_pool.add_many(new_cuts)
                    # Age and purge stale cuts
                    _cut_pool.age_cuts(result_sols[i])
            _cut_pool.purge_inactive(max_age=15)

        # --- Feasibility pump after root node ---
        if iteration == 0 and not _fp_ran:
            _fp_ran = True
            # Find the best relaxation solution from this batch
            best_root_idx = None
            best_root_obj = np.inf
            for i in range(n_batch):
                if result_lbs[i] < _SENTINEL_THRESHOLD and result_lbs[i] < best_root_obj:
                    best_root_obj = result_lbs[i]
                    best_root_idx = i
            if best_root_idx is not None:
                try:
                    from discopt._jax.primal_heuristics import feasibility_pump

                    fp_sol = feasibility_pump(model, result_sols[best_root_idx], max_rounds=5)
                    if fp_sol is not None:
                        fp_obj = float(evaluator.evaluate_objective(fp_sol))
                        fp_feas = not cl_list or _check_constraint_feasibility(
                            evaluator, fp_sol, cl_list, cu_list
                        )
                        if np.isfinite(fp_obj) and fp_obj < _SENTINEL_THRESHOLD and fp_feas:
                            tree.inject_incumbent(fp_sol, fp_obj)
                            logger.info("Feasibility pump found incumbent: obj=%.6g", fp_obj)
                except Exception as e:
                    logger.debug("Feasibility pump failed: %s", e)

        # Import results back to Rust tree
        t_rust_start = time.perf_counter()
        tree.import_results(result_ids, result_lbs, result_sols, result_feas)
        proc_stats = tree.process_evaluated()
        rust_time += time.perf_counter() - t_rust_start

        # --- Periodic OBBT with incumbent cutoff (Phase C) ---
        # When a new incumbent is found and bounds are still wide,
        # re-run OBBT with the incumbent objective as a cutoff.
        if proc_stats["incumbent_updates"] > 0 and n_vars <= 200:
            incumbent_info = tree.incumbent()
            if incumbent_info is not None:
                inc_sol, inc_obj = incumbent_info
                if inc_obj < _SENTINEL_THRESHOLD:
                    try:
                        from discopt._jax.obbt import run_obbt

                        obbt_result = run_obbt(
                            model,
                            lb=np.array(lb),
                            ub=np.array(ub),
                            incumbent_cutoff=float(inc_obj),
                            time_limit_per_lp=0.1,
                        )
                        if obbt_result.n_tightened > 0:
                            lb = obbt_result.tightened_lb
                            ub = obbt_result.tightened_ub
                            logger.info(
                                "OBBT tightened %d bounds (incumbent=%.6g)",
                                obbt_result.n_tightened,
                                inc_obj,
                            )
                    except Exception as e:
                        logger.debug("Periodic OBBT failed: %s", e)

        # --- FBBT with incumbent cutoff (Phase C3) ---
        # Cheap bound tightening via Rust FBBT (no LP solves).
        # Run on every incumbent update, complementing OBBT.
        if _model_repr is not None and proc_stats["incumbent_updates"] > 0:
            incumbent_info = tree.incumbent()
            if incumbent_info is not None:
                inc_sol, inc_obj = incumbent_info
                if inc_obj < _SENTINEL_THRESHOLD:
                    try:
                        fbbt_lbs, fbbt_ubs = _model_repr.fbbt_with_cutoff(
                            max_iter=10, tol=1e-8, incumbent_bound=float(inc_obj)
                        )
                        fbbt_lbs = np.asarray(fbbt_lbs)
                        fbbt_ubs = np.asarray(fbbt_ubs)
                        # Map per-block bounds to flat bounds array
                        n_tightened = 0
                        flat_idx = 0
                        for bi, vinfo in enumerate(model._variables):
                            for j in range(vinfo.size):
                                new_lo = fbbt_lbs[bi]
                                new_hi = fbbt_ubs[bi]
                                if new_lo > lb[flat_idx] + 1e-10:
                                    lb[flat_idx] = new_lo
                                    n_tightened += 1
                                if new_hi < ub[flat_idx] - 1e-10:
                                    ub[flat_idx] = new_hi
                                    n_tightened += 1
                                flat_idx += 1
                        if n_tightened > 0:
                            logger.info(
                                "FBBT tightened %d bounds (incumbent=%.6g)",
                                n_tightened,
                                inc_obj,
                            )
                    except Exception as e:
                        logger.debug("FBBT with cutoff failed: %s", e)

        iteration += 1

        # Check termination
        if tree.is_finished():
            break
        if tree.gap() <= gap_tolerance:
            break

        stats = tree.stats()
        if stats["total_nodes"] >= max_nodes:
            break

    # --- Build result ---
    wall_time = time.perf_counter() - t_start
    python_time = wall_time - rust_time - jax_time

    stats = tree.stats()
    incumbent = tree.incumbent()

    if incumbent is not None:
        sol_array, obj_val = incumbent
        # Filter out bogus incumbents from infeasible NLP relaxations
        if obj_val >= _SENTINEL_THRESHOLD:
            incumbent = None

    if incumbent is not None:
        sol_flat = np.array(sol_array)
        x_dict = _unpack_solution(model, sol_flat)

        if tree.gap() <= gap_tolerance or tree.is_finished():
            status = "optimal"
        else:
            status = "feasible"
    else:
        x_dict = None
        obj_val = None
        if stats["total_nodes"] >= max_nodes:
            status = "node_limit"
        elif wall_time >= time_limit:
            status = "time_limit"
        else:
            status = "infeasible"

    return SolveResult(
        status=status,
        objective=obj_val,
        bound=stats["global_lower_bound"],
        gap=stats["gap"],
        x=x_dict,
        wall_time=wall_time,
        node_count=stats["total_nodes"],
        rust_time=rust_time,
        jax_time=jax_time,
        python_time=python_time,
    )


def _solve_continuous(
    model: Model,
    time_limit: float,
    ipopt_options: Optional[dict],
    t_start: float,
    nlp_solver: str = "ipopt",
) -> SolveResult:
    """Solve a purely continuous model directly with NLP solver (no B&B)."""
    t_jax_start = time.perf_counter()
    evaluator = _make_evaluator(model)
    jax_time = time.perf_counter() - t_jax_start

    lb, ub = evaluator.variable_bounds
    lb_clipped = np.clip(lb, -_SPC, _SPC)
    ub_clipped = np.clip(ub, -_SPC, _SPC)
    x0 = 0.5 * (lb_clipped + ub_clipped)

    opts = dict(ipopt_options) if ipopt_options else {}
    opts.setdefault("print_level", 0)

    constraint_bounds = None

    t_jax_start = time.perf_counter()
    if nlp_solver == "ripopt":
        from discopt.solvers.nlp_ripopt import solve_nlp as solve_nlp_ripopt

        nlp_result = solve_nlp_ripopt(
            evaluator, x0, constraint_bounds=constraint_bounds, options=opts
        )
    elif nlp_solver == "sparse_ipm" and hasattr(evaluator, "_obj_fn"):
        from discopt._jax.sparse_ipm import solve_nlp_sparse_ipm

        # Build sparse Jacobian function if beneficial
        sparse_jac_fn = None
        try:
            from discopt._jax.sparsity import detect_and_color

            result = detect_and_color(model)
            if result is not None:
                from discopt._jax.sparse_jacobian import make_sparse_jac_fn

                pattern, colors, n_colors, seed = result
                sparse_jac_fn = make_sparse_jac_fn(evaluator._cons_fn, pattern, colors, seed)
        except Exception as e:
            logger.debug("Sparse Jacobian setup failed: %s", e)
        nlp_result = solve_nlp_sparse_ipm(
            evaluator,
            x0,
            constraint_bounds=constraint_bounds,
            options=opts,
            sparse_jac_fn=sparse_jac_fn,
        )
    elif nlp_solver == "ipm" and hasattr(evaluator, "_obj_fn"):
        from discopt._jax.ipm import solve_nlp_ipm

        nlp_result = solve_nlp_ipm(evaluator, x0, constraint_bounds=constraint_bounds, options=opts)
    else:
        nlp_result = solve_nlp(evaluator, x0, constraint_bounds=constraint_bounds, options=opts)
    jax_time += time.perf_counter() - t_jax_start

    wall_time = time.perf_counter() - t_start
    python_time = wall_time - jax_time

    if nlp_result.status == SolveStatus.OPTIMAL:
        status = "optimal"
    elif nlp_result.status == SolveStatus.INFEASIBLE:
        status = "infeasible"
    else:
        status = nlp_result.status.value

    x_dict = _unpack_solution(model, nlp_result.x) if nlp_result.x is not None else None

    return SolveResult(
        status=status,
        objective=nlp_result.objective,
        bound=nlp_result.objective if status == "optimal" else None,
        gap=0.0 if status == "optimal" else None,
        x=x_dict,
        wall_time=wall_time,
        node_count=0,
        rust_time=0.0,
        jax_time=jax_time,
        python_time=python_time,
    )


def _solve_node_nlp(
    evaluator: NLPEvaluator,
    x0: np.ndarray,
    node_lb: np.ndarray,
    node_ub: np.ndarray,
    constraint_bounds: Optional[list[tuple[float, float]]],
    options: dict,
    nlp_solver: str = "ipopt",
):
    """Solve the NLP relaxation at a single B&B node with tightened bounds.

    We override variable bounds to use the node-specific bounds
    rather than the global bounds.
    """
    if nlp_solver == "ripopt":
        return _solve_node_nlp_ripopt(evaluator, x0, node_lb, node_ub, constraint_bounds, options)
    if nlp_solver == "ipm":
        # JAX IPM requires JAX-compiled _obj_fn/_cons_fn; fall back to ipopt
        # for evaluators without these attributes.
        if not hasattr(evaluator, "_obj_fn"):
            return _solve_node_nlp_ipopt(
                evaluator, x0, node_lb, node_ub, constraint_bounds, options
            )
        return _solve_node_nlp_ipm(evaluator, x0, node_lb, node_ub, constraint_bounds, options)
    return _solve_node_nlp_ipopt(evaluator, x0, node_lb, node_ub, constraint_bounds, options)


def _solve_node_nlp_ripopt(
    evaluator: NLPEvaluator,
    x0: np.ndarray,
    node_lb: np.ndarray,
    node_ub: np.ndarray,
    constraint_bounds: Optional[list[tuple[float, float]]],
    options: dict,
):
    """Solve node NLP with ripopt (Rust IPM)."""
    from discopt.solvers import NLPResult
    from discopt.solvers.nlp_ripopt import solve_nlp as solve_nlp_ripopt

    class _BoundOverride:
        """Thin proxy that overrides variable_bounds on the evaluator."""

        def __init__(self, ev, lb, ub):
            self._ev = ev
            self._lb = lb
            self._ub = ub

        def __getattr__(self, name):
            if name == "variable_bounds":
                return (self._lb, self._ub)
            return getattr(self._ev, name)

    proxy = _BoundOverride(evaluator, node_lb, node_ub)

    try:
        return solve_nlp_ripopt(
            proxy,
            x0,
            constraint_bounds=constraint_bounds,
            options=options,
        )
    except Exception as e:
        logger.debug("ripopt solver failed: %s", e)
        return NLPResult(status=SolveStatus.ERROR, x=x0, objective=_INFEASIBILITY_SENTINEL)


def _solve_node_nlp_ipm(
    evaluator: NLPEvaluator,
    x0: np.ndarray,
    node_lb: np.ndarray,
    node_ub: np.ndarray,
    constraint_bounds: Optional[list[tuple[float, float]]],
    options: dict,
):
    """Solve node NLP with the pure-JAX IPM."""
    import jax.numpy as jnp

    from discopt._jax.ipm import IPMOptions, ipm_solve
    from discopt.solvers import NLPResult

    obj_fn = evaluator._obj_fn
    m = evaluator.n_constraints
    con_fn = evaluator._cons_fn if m > 0 else None

    x0_jax = jnp.array(x0, dtype=jnp.float64)
    x_l = jnp.array(node_lb, dtype=jnp.float64)
    x_u = jnp.array(node_ub, dtype=jnp.float64)

    if constraint_bounds is not None:
        g_l = jnp.array([b[0] for b in constraint_bounds], dtype=jnp.float64)
        g_u = jnp.array([b[1] for b in constraint_bounds], dtype=jnp.float64)
    else:
        g_l = None
        g_u = None

    ipm_opts = IPMOptions(max_iter=int(options.get("max_iter", 200)))

    try:
        state = ipm_solve(obj_fn, con_fn, x0_jax, x_l, x_u, g_l, g_u, ipm_opts)
    except Exception as e:
        logger.debug("IPM solver failed: %s", e)
        return NLPResult(status=SolveStatus.ERROR, x=x0, objective=_INFEASIBILITY_SENTINEL)

    conv = int(state.converged)
    if conv in (1, 2):
        status = SolveStatus.OPTIMAL
    elif conv == 3:
        status = SolveStatus.ITERATION_LIMIT
    else:
        status = SolveStatus.ERROR

    return NLPResult(
        status=status,
        x=np.asarray(state.x),
        objective=float(state.obj),
    )


def _solve_batch_ipm(
    evaluator,
    batch_lb,
    batch_ub,
    batch_ids,
    n_vars,
    constraint_bounds,
    options,
    g_l_jax,
    g_u_jax,
    batch_psols=None,
):
    """Solve a batch of NLP relaxations simultaneously via vmap'd IPM."""
    import jax.numpy as jnp

    from discopt._jax.ipm import IPMOptions, solve_nlp_batch

    n_batch = len(batch_ids)
    obj_fn = evaluator._obj_fn
    m = evaluator.n_constraints
    con_fn = evaluator._cons_fn if m > 0 else None

    # Build (batch, n) JAX arrays for bounds and starting points
    xl_batch = jnp.array(batch_lb, dtype=jnp.float64)
    xu_batch = jnp.array(batch_ub, dtype=jnp.float64)

    # Warm-start: use parent solutions clipped to child bounds, fall back to midpoint
    if batch_psols is not None:
        psols_jax = jnp.array(batch_psols, dtype=jnp.float64)
        has_parent = ~jnp.any(jnp.isnan(psols_jax), axis=1, keepdims=True)
        warm_x0 = jnp.clip(psols_jax, xl_batch, xu_batch)
        lb_clipped = jnp.clip(xl_batch, -_SPC, _SPC)
        ub_clipped = jnp.clip(xu_batch, -_SPC, _SPC)
        midpoint_x0 = 0.5 * (lb_clipped + ub_clipped)
        x0_batch = jnp.where(has_parent, warm_x0, midpoint_x0)
    else:
        lb_clipped = jnp.clip(xl_batch, -_SPC, _SPC)
        ub_clipped = jnp.clip(xu_batch, -_SPC, _SPC)
        x0_batch = 0.5 * (lb_clipped + ub_clipped)

    ipm_opts = IPMOptions(max_iter=int(options.get("max_iter", 200)))

    try:
        state = solve_nlp_batch(
            obj_fn, con_fn, x0_batch, xl_batch, xu_batch, g_l_jax, g_u_jax, ipm_opts
        )
    except Exception as e:
        logger.debug("Batch IPM failed: %s", e)
        # Fallback: mark all as infeasible
        result_ids = np.array(batch_ids, dtype=np.int64)
        x0_np = np.asarray(x0_batch)
        result_lbs = np.full(n_batch, _INFEASIBILITY_SENTINEL, dtype=np.float64)
        result_sols = x0_np
        result_feas = np.zeros(n_batch, dtype=bool)
        return result_ids, result_lbs, result_sols, result_feas

    # Unpack batched IPMState → numpy arrays
    converged = np.asarray(state.converged)  # (batch,)
    obj_vals = np.asarray(state.obj)  # (batch,)
    x_vals = np.asarray(state.x)  # (batch, n)

    result_ids = np.array(batch_ids, dtype=np.int64)
    result_lbs = np.where(
        (converged == 1) | (converged == 2) | (converged == 3),
        obj_vals,
        _INFEASIBILITY_SENTINEL,
    )
    result_sols = x_vals
    result_feas = np.zeros(n_batch, dtype=bool)  # Let Rust check integrality

    return result_ids, result_lbs, result_sols, result_feas


def _solve_node_nlp_ipopt(
    evaluator: NLPEvaluator,
    x0: np.ndarray,
    node_lb: np.ndarray,
    node_ub: np.ndarray,
    constraint_bounds: Optional[list[tuple[float, float]]],
    options: dict,
):
    """Solve node NLP with cyipopt (Ipopt)."""
    try:
        import cyipopt
    except ImportError:
        raise ImportError("cyipopt is required. Install it with: pip install cyipopt")

    from discopt.solvers.nlp_ipopt import _IpoptCallbacks

    n = evaluator.n_variables
    m = evaluator.n_constraints
    callbacks = _IpoptCallbacks(evaluator)

    if constraint_bounds is not None:
        cl = np.array([b[0] for b in constraint_bounds], dtype=np.float64)
        cu = np.array([b[1] for b in constraint_bounds], dtype=np.float64)
    else:
        cl = np.empty(0, dtype=np.float64)
        cu = np.empty(0, dtype=np.float64)

    problem = cyipopt.Problem(
        n=n,
        m=m,
        problem_obj=callbacks,
        lb=node_lb.astype(np.float64),
        ub=node_ub.astype(np.float64),
        cl=cl,
        cu=cu,
    )

    for key, value in options.items():
        problem.add_option(key, value)

    from discopt.solvers import NLPResult

    try:
        x, info = problem.solve(x0.astype(np.float64))
    except Exception as e:
        logger.debug("Ipopt solver failed: %s", e)
        return NLPResult(
            status=SolveStatus.ERROR,
            x=x0,
            objective=_INFEASIBILITY_SENTINEL,
        )

    from discopt.solvers.nlp_ipopt import _IPOPT_STATUS_MAP

    status_code = info["status"]
    status = _IPOPT_STATUS_MAP.get(status_code, SolveStatus.ERROR)

    return NLPResult(
        status=status,
        x=np.asarray(x),
        objective=float(info["obj_val"]),
    )


# ---------------------------------------------------------------------------
# Specialized LP/QP solvers
# ---------------------------------------------------------------------------


def _solve_lp(model: Model, t_start: float) -> SolveResult:
    """Solve an LP using the pure-JAX LP IPM (no NLP evaluator needed)."""
    from discopt._jax.lp_ipm import lp_ipm_solve
    from discopt._jax.problem_classifier import extract_lp_data

    t_jax_start = time.perf_counter()
    lp_data = extract_lp_data(model)
    state = lp_ipm_solve(lp_data.c, lp_data.A_eq, lp_data.b_eq, lp_data.x_l, lp_data.x_u)
    jax_time = time.perf_counter() - t_jax_start
    wall_time = time.perf_counter() - t_start

    n_orig = sum(v.size for v in model._variables)
    x_flat = np.asarray(state.x[:n_orig])
    obj_val = float(state.obj) + lp_data.obj_const

    conv = int(state.converged)
    if conv in (1, 2):
        status = "optimal"
    elif conv == 3:
        status = "iteration_limit"
    else:
        status = "error"

    return SolveResult(
        status=status,
        objective=obj_val,
        bound=obj_val if status == "optimal" else None,
        gap=0.0 if status == "optimal" else None,
        x=_unpack_solution(model, x_flat),
        wall_time=wall_time,
        node_count=0,
        rust_time=0.0,
        jax_time=jax_time,
        python_time=wall_time - jax_time,
    )


def _solve_qp(model: Model, t_start: float) -> SolveResult:
    """Solve a QP using the pure-JAX QP IPM."""
    from discopt._jax.problem_classifier import extract_qp_data
    from discopt._jax.qp_ipm import qp_ipm_solve

    t_jax_start = time.perf_counter()
    qp_data = extract_qp_data(model)
    state = qp_ipm_solve(
        qp_data.Q,
        qp_data.c,
        qp_data.A_eq,
        qp_data.b_eq,
        qp_data.x_l,
        qp_data.x_u,
    )
    jax_time = time.perf_counter() - t_jax_start
    wall_time = time.perf_counter() - t_start

    n_orig = sum(v.size for v in model._variables)
    x_flat = np.asarray(state.x[:n_orig])
    obj_val = float(state.obj) + qp_data.obj_const

    conv = int(state.converged)
    if conv in (1, 2):
        status = "optimal"
    elif conv == 3:
        status = "iteration_limit"
    else:
        status = "error"

    return SolveResult(
        status=status,
        objective=obj_val,
        bound=obj_val if status == "optimal" else None,
        gap=0.0 if status == "optimal" else None,
        x=_unpack_solution(model, x_flat),
        wall_time=wall_time,
        node_count=0,
        rust_time=0.0,
        jax_time=jax_time,
        python_time=wall_time - jax_time,
    )


def _solve_milp_bb(
    model: Model,
    time_limit: float,
    gap_tolerance: float,
    batch_size: int,
    strategy: str,
    max_nodes: int,
    t_start: float,
) -> SolveResult:
    """Solve a MILP via B&B with LP relaxation solves at each node."""
    import jax.numpy as jnp

    from discopt._jax.lp_ipm import lp_ipm_solve
    from discopt._jax.problem_classifier import extract_lp_data

    rust_time = 0.0
    jax_time = 0.0

    t_jax_start = time.perf_counter()
    lp_data = extract_lp_data(model)
    jax_time += time.perf_counter() - t_jax_start

    n_vars, lb, ub, int_offsets, int_sizes = _extract_variable_info(model)
    n_orig = sum(v.size for v in model._variables)

    t_rust_start = time.perf_counter()
    tree = PyTreeManager(n_vars, lb.tolist(), ub.tolist(), int_offsets, int_sizes, strategy)
    tree.initialize()
    rust_time += time.perf_counter() - t_rust_start

    iteration = 0
    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            break

        t_rust_start = time.perf_counter()
        batch_lb, batch_ub, batch_ids, _batch_psols = tree.export_batch(batch_size)
        rust_time += time.perf_counter() - t_rust_start

        n_batch = len(batch_ids)
        if n_batch == 0:
            break

        t_jax_start = time.perf_counter()
        result_ids = np.array(batch_ids, dtype=np.int64)
        n_slack = lp_data.x_l.shape[0] - n_orig

        if n_batch > 1:
            # Batch LP solve via vmap
            from discopt._jax.lp_ipm import lp_ipm_solve_batch

            xl_arr = jnp.array(batch_lb, dtype=jnp.float64)
            xu_arr = jnp.array(batch_ub, dtype=jnp.float64)
            slack_l = jnp.zeros((n_batch, n_slack), dtype=jnp.float64)
            slack_u = jnp.full((n_batch, n_slack), 1e20, dtype=jnp.float64)
            xl_full = jnp.concatenate([xl_arr, slack_l], axis=1)
            xu_full = jnp.concatenate([xu_arr, slack_u], axis=1)

            try:
                state = lp_ipm_solve_batch(lp_data.c, lp_data.A_eq, lp_data.b_eq, xl_full, xu_full)
                converged = np.asarray(state.converged)
                obj_vals = np.asarray(state.obj)
                x_vals = np.asarray(state.x)

                ok = (converged == 1) | (converged == 2) | (converged == 3)
                result_lbs = np.where(ok, obj_vals + lp_data.obj_const, _INFEASIBILITY_SENTINEL)
                result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
                for i in range(n_batch):
                    if ok[i]:
                        result_sols[i] = x_vals[i, :n_vars]
                    else:
                        lb_c = np.clip(np.array(batch_lb[i]), -_SPC, _SPC)
                        ub_c = np.clip(np.array(batch_ub[i]), -_SPC, _SPC)
                        result_sols[i] = 0.5 * (lb_c + ub_c)
            except Exception as e:
                logger.debug("Batch LP solve failed: %s", e)
                result_lbs = np.full(n_batch, _INFEASIBILITY_SENTINEL, dtype=np.float64)
                result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
                for i in range(n_batch):
                    lb_c = np.clip(np.array(batch_lb[i]), -_SPC, _SPC)
                    ub_c = np.clip(np.array(batch_ub[i]), -_SPC, _SPC)
                    result_sols[i] = 0.5 * (lb_c + ub_c)
            result_feas = np.zeros(n_batch, dtype=bool)
        else:
            result_lbs = np.empty(n_batch, dtype=np.float64)
            result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
            result_feas = np.zeros(n_batch, dtype=bool)

            for i in range(n_batch):
                node_lb = np.array(batch_lb[i])
                node_ub = np.array(batch_ub[i])

                x_l_node = jnp.array(node_lb, dtype=jnp.float64)
                x_u_node = jnp.array(node_ub, dtype=jnp.float64)

                x_l_full = jnp.concatenate([x_l_node, jnp.zeros(n_slack)])
                x_u_full = jnp.concatenate([x_u_node, jnp.full(n_slack, 1e20)])

                try:
                    state = lp_ipm_solve(lp_data.c, lp_data.A_eq, lp_data.b_eq, x_l_full, x_u_full)
                    conv = int(state.converged)
                    if conv in (1, 2, 3):
                        result_lbs[i] = float(state.obj) + lp_data.obj_const
                        result_sols[i] = np.asarray(state.x[:n_vars])
                    else:
                        result_lbs[i] = _INFEASIBILITY_SENTINEL
                        lb_c = np.clip(node_lb, -_SPC, _SPC)
                        ub_c = np.clip(node_ub, -_SPC, _SPC)
                        result_sols[i] = 0.5 * (lb_c + ub_c)
                except Exception as e:
                    logger.debug("Per-node LP/QP solve failed: %s", e)
                    result_lbs[i] = _INFEASIBILITY_SENTINEL
                    lb_c = np.clip(node_lb, -_SPC, _SPC)
                    ub_c = np.clip(node_ub, -_SPC, _SPC)
                    result_sols[i] = 0.5 * (lb_c + ub_c)

        jax_time += time.perf_counter() - t_jax_start

        t_rust_start = time.perf_counter()
        tree.import_results(result_ids, result_lbs, result_sols, result_feas)
        tree.process_evaluated()
        rust_time += time.perf_counter() - t_rust_start

        iteration += 1
        if tree.is_finished():
            break
        if tree.gap() <= gap_tolerance:
            break
        stats = tree.stats()
        if stats["total_nodes"] >= max_nodes:
            break

    wall_time = time.perf_counter() - t_start
    python_time = wall_time - rust_time - jax_time
    stats = tree.stats()
    incumbent = tree.incumbent()

    if incumbent is not None:
        sol_array, obj_val = incumbent
        if obj_val >= _SENTINEL_THRESHOLD:
            incumbent = None

    if incumbent is not None:
        sol_flat = np.array(sol_array)
        x_dict = _unpack_solution(model, sol_flat)
        if tree.gap() <= gap_tolerance or tree.is_finished():
            status = "optimal"
        else:
            status = "feasible"
    else:
        x_dict = None
        obj_val = None
        if stats["total_nodes"] >= max_nodes:
            status = "node_limit"
        elif wall_time >= time_limit:
            status = "time_limit"
        else:
            status = "infeasible"

    return SolveResult(
        status=status,
        objective=obj_val,
        bound=stats["global_lower_bound"],
        gap=stats["gap"],
        x=x_dict,
        wall_time=wall_time,
        node_count=stats["total_nodes"],
        rust_time=rust_time,
        jax_time=jax_time,
        python_time=python_time,
    )


def _solve_miqp_bb(
    model: Model,
    time_limit: float,
    gap_tolerance: float,
    batch_size: int,
    strategy: str,
    max_nodes: int,
    t_start: float,
) -> SolveResult:
    """Solve a MIQP via B&B with QP relaxation solves at each node."""
    import jax.numpy as jnp

    from discopt._jax.problem_classifier import extract_qp_data
    from discopt._jax.qp_ipm import qp_ipm_solve

    rust_time = 0.0
    jax_time = 0.0

    t_jax_start = time.perf_counter()
    qp_data = extract_qp_data(model)
    jax_time += time.perf_counter() - t_jax_start

    n_vars, lb, ub, int_offsets, int_sizes = _extract_variable_info(model)
    n_orig = sum(v.size for v in model._variables)

    t_rust_start = time.perf_counter()
    tree = PyTreeManager(n_vars, lb.tolist(), ub.tolist(), int_offsets, int_sizes, strategy)
    tree.initialize()
    rust_time += time.perf_counter() - t_rust_start

    iteration = 0
    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            break

        t_rust_start = time.perf_counter()
        batch_lb, batch_ub, batch_ids, _batch_psols = tree.export_batch(batch_size)
        rust_time += time.perf_counter() - t_rust_start

        n_batch = len(batch_ids)
        if n_batch == 0:
            break

        t_jax_start = time.perf_counter()
        result_ids = np.array(batch_ids, dtype=np.int64)
        n_slack = qp_data.x_l.shape[0] - n_orig

        if n_batch > 1:
            # Batch QP solve via vmap
            from discopt._jax.qp_ipm import qp_ipm_solve_batch

            xl_arr = jnp.array(batch_lb, dtype=jnp.float64)
            xu_arr = jnp.array(batch_ub, dtype=jnp.float64)
            slack_l = jnp.zeros((n_batch, n_slack), dtype=jnp.float64)
            slack_u = jnp.full((n_batch, n_slack), 1e20, dtype=jnp.float64)
            xl_full = jnp.concatenate([xl_arr, slack_l], axis=1)
            xu_full = jnp.concatenate([xu_arr, slack_u], axis=1)

            try:
                state = qp_ipm_solve_batch(
                    qp_data.Q,
                    qp_data.c,
                    qp_data.A_eq,
                    qp_data.b_eq,
                    xl_full,
                    xu_full,
                )
                converged = np.asarray(state.converged)
                obj_vals = np.asarray(state.obj)
                x_vals = np.asarray(state.x)

                ok = (converged == 1) | (converged == 2) | (converged == 3)
                result_lbs = np.where(ok, obj_vals + qp_data.obj_const, _INFEASIBILITY_SENTINEL)
                result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
                for i in range(n_batch):
                    if ok[i]:
                        result_sols[i] = x_vals[i, :n_vars]
                    else:
                        lb_c = np.clip(np.array(batch_ub[i]), -_SPC, _SPC)
                        ub_c = np.clip(np.array(batch_ub[i]), -_SPC, _SPC)
                        result_sols[i] = 0.5 * (lb_c + ub_c)
            except Exception as e:
                logger.debug("Batch QP solve failed: %s", e)
                result_lbs = np.full(n_batch, _INFEASIBILITY_SENTINEL, dtype=np.float64)
                result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
                for i in range(n_batch):
                    lb_c = np.clip(np.array(batch_lb[i]), -_SPC, _SPC)
                    ub_c = np.clip(np.array(batch_ub[i]), -_SPC, _SPC)
                    result_sols[i] = 0.5 * (lb_c + ub_c)
            result_feas = np.zeros(n_batch, dtype=bool)
        else:
            result_lbs = np.empty(n_batch, dtype=np.float64)
            result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
            result_feas = np.zeros(n_batch, dtype=bool)

            for i in range(n_batch):
                node_lb = np.array(batch_lb[i])
                node_ub = np.array(batch_ub[i])

                x_l_node = jnp.array(node_lb, dtype=jnp.float64)
                x_u_node = jnp.array(node_ub, dtype=jnp.float64)

                x_l_full = jnp.concatenate([x_l_node, jnp.zeros(n_slack)])
                x_u_full = jnp.concatenate([x_u_node, jnp.full(n_slack, 1e20)])

                try:
                    state = qp_ipm_solve(
                        qp_data.Q,
                        qp_data.c,
                        qp_data.A_eq,
                        qp_data.b_eq,
                        x_l_full,
                        x_u_full,
                    )
                    conv = int(state.converged)
                    if conv in (1, 2, 3):
                        result_lbs[i] = float(state.obj) + qp_data.obj_const
                        result_sols[i] = np.asarray(state.x[:n_vars])
                    else:
                        result_lbs[i] = _INFEASIBILITY_SENTINEL
                        lb_c = np.clip(node_lb, -_SPC, _SPC)
                        ub_c = np.clip(node_ub, -_SPC, _SPC)
                        result_sols[i] = 0.5 * (lb_c + ub_c)
                except Exception as e:
                    logger.debug("Per-node LP/QP solve failed: %s", e)
                    result_lbs[i] = _INFEASIBILITY_SENTINEL
                    lb_c = np.clip(node_lb, -_SPC, _SPC)
                    ub_c = np.clip(node_ub, -_SPC, _SPC)
                    result_sols[i] = 0.5 * (lb_c + ub_c)

        jax_time += time.perf_counter() - t_jax_start

        t_rust_start = time.perf_counter()
        tree.import_results(result_ids, result_lbs, result_sols, result_feas)
        tree.process_evaluated()
        rust_time += time.perf_counter() - t_rust_start

        iteration += 1
        if tree.is_finished():
            break
        if tree.gap() <= gap_tolerance:
            break
        stats = tree.stats()
        if stats["total_nodes"] >= max_nodes:
            break

    wall_time = time.perf_counter() - t_start
    python_time = wall_time - rust_time - jax_time
    stats = tree.stats()
    incumbent = tree.incumbent()

    if incumbent is not None:
        sol_array, obj_val = incumbent
        if obj_val >= _SENTINEL_THRESHOLD:
            incumbent = None

    if incumbent is not None:
        sol_flat = np.array(sol_array)
        x_dict = _unpack_solution(model, sol_flat)
        if tree.gap() <= gap_tolerance or tree.is_finished():
            status = "optimal"
        else:
            status = "feasible"
    else:
        x_dict = None
        obj_val = None
        if stats["total_nodes"] >= max_nodes:
            status = "node_limit"
        elif wall_time >= time_limit:
            status = "time_limit"
        else:
            status = "infeasible"

    return SolveResult(
        status=status,
        objective=obj_val,
        bound=stats["global_lower_bound"],
        gap=stats["gap"],
        x=x_dict,
        wall_time=wall_time,
        node_count=stats["total_nodes"],
        rust_time=rust_time,
        jax_time=jax_time,
        python_time=python_time,
    )
