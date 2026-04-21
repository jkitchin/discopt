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
from typing import Any, Optional

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


def _evaluator_fingerprint(model: Model) -> tuple:
    """Structural fingerprint of a model for evaluator-cache validity.

    Captures identity of the objective, constraints, variables, and parameters.
    Mutating ``Parameter.value`` does NOT change the fingerprint, so repeated
    solves that only rebind parameter values reuse the same JITed callables
    and hit the XLA cache.
    """
    return (
        id(model._objective),
        tuple(id(c) for c in model._constraints),
        tuple(id(v) for v in model._variables),
        tuple(id(p) for p in model._parameters),
    )


def _make_evaluator(model: Model):
    """Create or reuse a cached NLPEvaluator for the model.

    The first call builds a fresh ``NLPEvaluator`` (which JITs obj/grad/hess/
    cons/jac/lag_hess). Subsequent calls return the same evaluator as long as
    the model's structural fingerprint is unchanged, so the underlying jit
    objects (and their XLA caches) are preserved across solves. Parameter
    value changes are threaded through at call time as a runtime pytree.
    """
    fingerprint = _evaluator_fingerprint(model)
    cached = getattr(model, "_nlp_evaluator_cache", None)
    if cached is not None:
        ev, cached_fp = cached
        if cached_fp == fingerprint:
            return ev
    ev = NLPEvaluator(model)
    model._nlp_evaluator_cache = (ev, fingerprint)
    return ev


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


def _check_lp_solution_feasibility(A_eq, b_eq, x_full, tol=1e-4):
    """Check that an LP/QP solution satisfies A_eq @ x = b_eq within tolerance.

    Returns True if the maximum absolute constraint residual is within *tol*.
    Used by MILP/MIQP B&B to reject LP/QP relaxation solutions where the IPM
    converged to a constraint-violating point.
    """
    if A_eq.shape[0] == 0:
        return True
    residual = np.asarray(A_eq) @ np.asarray(x_full) - np.asarray(b_eq)
    return float(np.max(np.abs(residual))) <= tol


def _check_constraint_feasibility(evaluator, x, cl_list, cu_list, tol=1e-4):
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
    # The evaluator may have more constraints than cl/cu (e.g., augmented with
    # cutting planes).  Only check the original constraints.
    n_check = min(len(cons), len(cl))
    if n_check == 0:
        return True
    max_viol = max(
        float(np.max(cons[:n_check] - cu[:n_check])), float(np.max(cl[:n_check] - cons[:n_check]))
    )
    return max_viol <= tol


def _tighten_node_bounds(evaluator, node_lb, node_ub, cl_list, cu_list, max_rounds=3):
    """Constraint-based bound tightening (FBBT) for a single B&B node.

    Uses the constraint Jacobian to propagate implied variable bounds.
    For linear constraints (e.g., x_i <= M * y_i with y_i fixed), this
    is exact and eliminates degenerate variable bounds that cause IPM
    convergence failures.

    Parameters
    ----------
    evaluator : NLPEvaluator
        Provides evaluate_constraints and evaluate_jacobian.
    node_lb, node_ub : np.ndarray
        Variable bounds at this node.
    cl_list, cu_list : list[float]
        Constraint bounds.
    max_rounds : int
        Maximum propagation rounds.

    Returns
    -------
    lb, ub : np.ndarray
        Tightened variable bounds.
    """
    if evaluator.n_constraints == 0 or not cl_list:
        return node_lb.copy(), node_ub.copy()

    lb = node_lb.copy()
    ub = node_ub.copy()
    n = len(lb)
    m = len(cl_list)
    cu = np.array(cu_list, dtype=np.float64)
    cl = np.array(cl_list, dtype=np.float64)

    # Detect which constraints are linear by checking if the Jacobian
    # changes between two distinct evaluation points.  FBBT via Jacobian
    # linearization is only sound for linear constraints; applying it to
    # nonlinear constraints (e.g. x^1.5) can over-tighten and exclude
    # feasible regions, causing false infeasibility (issue #6).
    try:
        pt_a = np.clip(lb + 0.25 * (ub - lb), -_SPC, _SPC)
        pt_b = np.clip(lb + 0.75 * (ub - lb), -_SPC, _SPC)
        J_a = evaluator.evaluate_jacobian(pt_a)
        J_b = evaluator.evaluate_jacobian(pt_b)
        is_linear = np.all(np.abs(J_a - J_b) < 1e-8, axis=1)  # (m,) bool
    except Exception:
        return lb, ub  # can't determine linearity, skip FBBT

    if not np.any(is_linear):
        return lb, ub  # no linear constraints to tighten

    for _ in range(max_rounds):
        changed = False
        # Evaluate Jacobian at midpoint of current bounds
        mid = np.clip(lb, -_SPC, _SPC)
        span = np.clip(ub, -_SPC, _SPC) - mid
        mid = mid + 0.5 * span
        try:
            J = evaluator.evaluate_jacobian(mid)  # (m, n)
            g = evaluator.evaluate_constraints(mid)  # (m,)
        except Exception:
            break

        for j in range(m):
            if not is_linear[j]:
                continue
            # For constraint g_j(x) <= cu_j:
            # Linear approx: g_j(mid) + J[j,:] @ (x - mid) <= cu_j
            # To find max x_i, set other vars to MINIMIZE g (most room):
            #   J[j,k] > 0 → use lb[k];  J[j,k] < 0 → use ub[k]
            if cu[j] < 1e19:
                for i in range(n):
                    if abs(J[j, i]) < 1e-12 or lb[i] == ub[i]:
                        continue
                    residual = cu[j] - g[j]
                    for k in range(n):
                        if k == i:
                            continue
                        if J[j, k] > 0:
                            residual -= J[j, k] * (lb[k] - mid[k])
                        else:
                            residual -= J[j, k] * (ub[k] - mid[k])
                    # J[j,i] * (x_i - mid_i) <= residual
                    if J[j, i] > 1e-12:
                        new_ub = mid[i] + residual / J[j, i]
                        if new_ub < ub[i] - 1e-10:
                            ub[i] = max(lb[i], new_ub)
                            changed = True
                    elif J[j, i] < -1e-12:
                        new_lb = mid[i] + residual / J[j, i]
                        if new_lb > lb[i] + 1e-10:
                            lb[i] = min(ub[i], new_lb)
                            changed = True

            # For constraint g_j(x) >= cl_j:
            # To find min x_i, set other vars to MAXIMIZE g (most room):
            #   J[j,k] > 0 → use ub[k];  J[j,k] < 0 → use lb[k]
            if cl[j] > -1e19:
                for i in range(n):
                    if abs(J[j, i]) < 1e-12 or lb[i] == ub[i]:
                        continue
                    residual = cl[j] - g[j]
                    for k in range(n):
                        if k == i:
                            continue
                        if J[j, k] > 0:
                            residual -= J[j, k] * (ub[k] - mid[k])
                        else:
                            residual -= J[j, k] * (lb[k] - mid[k])
                    # J[j,i] * (x_i - mid_i) >= residual
                    if J[j, i] > 1e-12:
                        new_lb = mid[i] + residual / J[j, i]
                        if new_lb > lb[i] + 1e-10:
                            lb[i] = min(ub[i], new_lb)
                            changed = True
                    elif J[j, i] < -1e-12:
                        new_ub = mid[i] + residual / J[j, i]
                        if new_ub < ub[i] - 1e-10:
                            ub[i] = max(lb[i], new_ub)
                            changed = True

        if not changed:
            break

    return lb, ub


def _infer_constraint_bounds(model: Model, evaluator=None):
    """Infer (cl, cu) arrays from model constraint senses.

    The NLPEvaluator compiles constraints as `body - rhs`, so:
      - '<=' constraints: body - rhs <= 0 => cl = -inf, cu = 0
      - '==' constraints: body - rhs == 0 => cl = 0, cu = 0
      - '>=' constraints: body - rhs >= 0 => cl = 0, cu = inf

    If an ``evaluator`` is passed, its per-constraint flat sizes are used
    to expand each source Constraint's bounds to ``flat_size`` rows so
    vector-valued bodies (e.g. DAEBuilder's vectorized collocation
    residuals) line up with cyipopt's row count. Without an evaluator,
    each source Constraint contributes one row (legacy scalar behavior).
    """
    cl_list = []
    cu_list = []
    sizes = None
    if evaluator is not None and hasattr(evaluator, "_constraint_flat_sizes"):
        sizes = evaluator._constraint_flat_sizes

    k = 0
    for c in model._constraints:
        if not isinstance(c, Constraint):
            continue
        if c.sense == "<=":
            lo, hi = -1e20, 0.0
        elif c.sense == "==":
            lo, hi = 0.0, 0.0
        elif c.sense == ">=":
            lo, hi = 0.0, 1e20
        else:
            raise ValueError(f"Unknown constraint sense: {c.sense}")
        n = int(sizes[k]) if sizes is not None else 1
        cl_list.extend([lo] * n)
        cu_list.extend([hi] * n)
        k += 1

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
    convex=False,
):
    """Solve root NLP relaxation from multiple starting points via vmap'd IPM.

    Uses jax.vmap to solve all starting points in parallel, giving ~Nx speedup
    over the serial loop when using the pure-JAX IPM backend.

    When ``convex=True``, iterates whose best code is 3 (max_iter) or 4
    (stalled) are polished with cyipopt; IPM's non-KKT obj is not a valid
    lower bound for convex NLP-BB (issue #39).
    """
    import jax.numpy as jnp

    from discopt._jax.ipm import (
        IPMOptions,
        _jax_feasibility_restoration,
        ipm_solve,
        solve_nlp_batch,
    )
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

    # Mask: converged == 1 (optimal), 2 (acceptable), 3 (iter limit), or 4 (stalled).
    # Code 5 (infeasible) is excluded. NaN objectives are also excluded —
    # they indicate IPM divergence (e.g. log of negative argument).
    feasible_mask = (
        (converged == 1) | (converged == 2) | (converged == 3) | (converged == 4)
    ) & np.isfinite(obj_vals)

    if np.any(feasible_mask):
        # Among feasible, pick the one with lowest objective
        masked_obj = np.where(feasible_mask, obj_vals, np.inf)
        best_idx = int(np.argmin(masked_obj))
        best_code = int(converged[best_idx])
        best_obj = float(obj_vals[best_idx])
        best_x = np.asarray(x_vals[best_idx], dtype=np.float64)

        # Convex polish: codes 3 (max_iter) / 4 (stalled) don't certify
        # KKT stationarity — objective is not a valid LB for convex NLP-BB.
        if convex and best_code in (3, 4):
            try:
                polish = _solve_node_nlp_ipopt(
                    evaluator,
                    best_x,
                    np.asarray(node_lb),
                    np.asarray(node_ub),
                    constraint_bounds,
                    options,
                )
            except Exception as e:
                logger.debug("Root IPM convex polish failed: %s", e)
                polish = None
            if polish is not None and polish.status in (
                SolveStatus.OPTIMAL,
                SolveStatus.ITERATION_LIMIT,
            ):
                p_obj = float(polish.objective)
                if np.isfinite(p_obj) and p_obj < _SENTINEL_THRESHOLD:
                    return NLPResult(
                        status=polish.status,
                        x=np.asarray(polish.x),
                        objective=p_obj,
                    )

        return NLPResult(
            status=SolveStatus.OPTIMAL,
            x=best_x,
            objective=best_obj,
        )
    else:
        # All starts infeasible or NaN — attempt feasibility restoration.
        if con_fn is not None and g_l_jax is not None:
            xl_jax = jnp.array(node_lb, dtype=jnp.float64)
            xu_jax = jnp.array(node_ub, dtype=jnp.float64)
            for i in range(n_starts):
                if converged[i] == 5:
                    try:
                        x_restored, rest_ok = _jax_feasibility_restoration(
                            con_fn,
                            jnp.asarray(x_vals[i]),
                            xl_jax,
                            xu_jax,
                            g_l_jax,
                            g_u_jax,
                            ipm_opts,
                        )
                    except Exception:
                        continue
                    if rest_ok:
                        try:
                            state_i = ipm_solve(
                                obj_fn,
                                con_fn,
                                x_restored,
                                xl_jax,
                                xu_jax,
                                g_l_jax,
                                g_u_jax,
                                ipm_opts,
                            )
                        except Exception:
                            continue
                        conv_i = int(state_i.converged)
                        if conv_i in (1, 2, 3, 4):
                            return NLPResult(
                                status=SolveStatus.OPTIMAL,
                                x=np.asarray(state_i.x),
                                objective=float(state_i.obj),
                            )
        return NLPResult(
            status=SolveStatus.ERROR,
            x=x_vals[0],
            objective=_INFEASIBILITY_SENTINEL,
        )


def _solve_node_multistart_ipm(
    evaluator,
    x0,
    node_lb,
    node_ub,
    constraint_bounds,
    g_l_jax,
    g_u_jax,
    options,
    n_extra=2,
):
    """Multistart NLP at a child node: parent warm-start + diverse points.

    Uses parent warm-start as primary starting point, plus midpoint and
    random points within the node bounds. All points are solved in parallel
    via vmap'd IPM, and the best converged solution is returned.
    """
    import jax.numpy as jnp

    from discopt._jax.ipm import (
        IPMOptions,
        _jax_feasibility_restoration,
        ipm_solve,
        solve_nlp_batch,
    )
    from discopt.solvers import NLPResult

    n_vars = len(node_lb)
    lb_clipped = np.clip(node_lb, -_SPC, _SPC)
    ub_clipped = np.clip(node_ub, -_SPC, _SPC)

    # Starting points: parent warm-start + midpoint + random
    starts = [np.asarray(x0, dtype=np.float64)]
    span = np.maximum(ub_clipped - lb_clipped, 0.0)
    starts.append(0.5 * (lb_clipped + ub_clipped))  # midpoint

    # Deterministic seed from node bounds for reproducibility
    seed = int(abs(hash(tuple(node_lb[:4].tolist()) + tuple(node_ub[:4].tolist())))) % (2**31)
    rng = np.random.RandomState(seed)
    starts.append(lb_clipped + rng.uniform(size=n_vars) * span)

    n_starts = len(starts)
    obj_fn = evaluator._obj_fn
    m = evaluator.n_constraints
    con_fn = evaluator._cons_fn if m > 0 else None

    x0_batch = jnp.array(np.stack(starts), dtype=jnp.float64)
    xl_batch = jnp.broadcast_to(jnp.array(node_lb, dtype=jnp.float64), (n_starts, n_vars))
    xu_batch = jnp.broadcast_to(jnp.array(node_ub, dtype=jnp.float64), (n_starts, n_vars))

    ipm_opts = IPMOptions(max_iter=int(options.get("max_iter", 200)))

    try:
        state = solve_nlp_batch(
            obj_fn, con_fn, x0_batch, xl_batch, xu_batch, g_l_jax, g_u_jax, ipm_opts
        )
    except Exception as e:
        logger.debug("Node multistart batch IPM failed: %s", e)
        return NLPResult(
            status=SolveStatus.ERROR,
            x=np.asarray(x0),
            objective=_INFEASIBILITY_SENTINEL,
        )

    converged = np.asarray(state.converged)
    obj_vals = np.asarray(state.obj)
    x_vals = np.asarray(state.x)

    feasible_mask = ((converged == 1) | (converged == 2) | (converged == 3)) & np.isfinite(obj_vals)

    if np.any(feasible_mask):
        masked_obj = np.where(feasible_mask, obj_vals, np.inf)
        best_idx = int(np.argmin(masked_obj))
        return NLPResult(
            status=SolveStatus.OPTIMAL,
            x=x_vals[best_idx],
            objective=float(obj_vals[best_idx]),
        )
    else:
        # All starts infeasible — attempt feasibility restoration.
        if con_fn is not None and g_l_jax is not None:
            xl_jax = jnp.array(node_lb, dtype=jnp.float64)
            xu_jax = jnp.array(node_ub, dtype=jnp.float64)
            for i in range(n_starts):
                if converged[i] == 5:
                    try:
                        x_restored, rest_ok = _jax_feasibility_restoration(
                            con_fn,
                            jnp.asarray(x_vals[i]),
                            xl_jax,
                            xu_jax,
                            g_l_jax,
                            g_u_jax,
                            ipm_opts,
                        )
                    except Exception:
                        continue
                    if rest_ok:
                        try:
                            state_i = ipm_solve(
                                obj_fn,
                                con_fn,
                                x_restored,
                                xl_jax,
                                xu_jax,
                                g_l_jax,
                                g_u_jax,
                                ipm_opts,
                            )
                        except Exception:
                            continue
                        conv_i = int(state_i.converged)
                        if conv_i in (1, 2, 3):
                            return NLPResult(
                                status=SolveStatus.OPTIMAL,
                                x=np.asarray(state_i.x),
                                objective=float(state_i.obj),
                            )
        return NLPResult(
            status=SolveStatus.ERROR,
            x=x_vals[0],
            objective=_INFEASIBILITY_SENTINEL,
        )


def _invoke_pre_import_callbacks(
    *,
    model,
    tree,
    t_start,
    result_ids,
    result_lbs,
    result_sols,
    result_feas,
    n_batch,
    int_offsets,
    int_sizes,
    n_vars,
    lazy_constraints,
    incumbent_callback,
    _cut_pool,
):
    """Check lazy constraints and incumbent callbacks before importing results.

    For each integer-feasible solution in the batch:
    1. Call ``lazy_constraints`` callback. If it returns cuts, add them to the
       cut pool and mark the node as infeasible (preventing it from becoming
       an incumbent). The cuts will tighten subsequent relaxations.
    2. Call ``incumbent_callback``. If it returns False, mark the node as
       infeasible.
    """
    from discopt._jax.cutting_planes import LinearCut
    from discopt.callbacks import CallbackContext, cut_result_to_dense

    incumbent_info = tree.incumbent()
    inc_obj = None
    if incumbent_info is not None:
        _, inc_obj = incumbent_info
        if inc_obj >= _SENTINEL_THRESHOLD:
            inc_obj = None

    stats = tree.stats()
    elapsed = time.perf_counter() - t_start

    for i in range(n_batch):
        if result_lbs[i] >= _SENTINEL_THRESHOLD:
            continue  # skip infeasible nodes

        # Check integrality
        sol_is_int_feas = True
        for off, sz in zip(int_offsets, int_sizes):
            for j in range(off, off + sz):
                if abs(result_sols[i, j] - round(result_sols[i, j])) > 1e-5:
                    sol_is_int_feas = False
                    break
            if not sol_is_int_feas:
                break

        if not sol_is_int_feas:
            continue

        ctx = CallbackContext(
            node_count=stats["total_nodes"],
            incumbent_obj=inc_obj,
            best_bound=stats.get("global_lower_bound", -np.inf),
            gap=stats.get("gap"),
            elapsed_time=elapsed,
            x_relaxation=result_sols[i].copy(),
            node_bound=float(result_lbs[i]),
        )

        # --- Lazy constraints ---
        if lazy_constraints is not None:
            try:
                cuts = lazy_constraints(ctx, model)
                if cuts:
                    for cut in cuts:
                        coeffs, rhs, sense = cut_result_to_dense(cut, model)
                        _cut_pool.add(LinearCut(coeffs=coeffs, rhs=rhs, sense=sense))
                    # Mark as infeasible so it does not become incumbent.
                    result_lbs[i] = _INFEASIBILITY_SENTINEL
                    logger.info(
                        "Lazy constraint callback added %d cut(s) at node %d",
                        len(cuts),
                        int(result_ids[i]),
                    )
                    continue  # skip incumbent callback for cut-separated nodes
            except Exception as e:
                logger.warning("Lazy constraint callback raised an exception: %s", e)

        # --- Incumbent callback ---
        if incumbent_callback is not None:
            try:
                solution = _unpack_solution(model, result_sols[i])
                accept = incumbent_callback(ctx, model, solution)
                if accept is False:
                    result_lbs[i] = _INFEASIBILITY_SENTINEL
                    logger.info(
                        "Incumbent callback rejected solution at node %d",
                        int(result_ids[i]),
                    )
            except Exception as e:
                logger.warning("Incumbent callback raised an exception: %s", e)


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


_BOUND_WARN_THRESHOLD = 1e15


def _check_finite_bounds(model: Model) -> None:
    """Warn if any variable has very large or infinite bounds.

    Interior point methods use barrier terms that require reasonably sized
    bounds. Bounds beyond 1e15 cause numerical difficulties (NaN gradients,
    ill-conditioned KKT systems) and the solver silently produces NaN
    objectives or reports iteration_limit. This check warns users early.
    """
    bad_vars = []
    for v in model._variables:
        lb_flat = v.lb.flatten()
        ub_flat = v.ub.flatten()
        for j in range(v.size):
            lo, hi = float(lb_flat[j]), float(ub_flat[j])
            if (
                not np.isfinite(lo)
                or not np.isfinite(hi)
                or abs(lo) > _BOUND_WARN_THRESHOLD
                or abs(hi) > _BOUND_WARN_THRESHOLD
            ):
                name = v.name if v.size == 1 else f"{v.name}[{j}]"
                bad_vars.append(f"{name} (lb={lo:.2g}, ub={hi:.2g})")
    if bad_vars:
        import warnings

        warnings.warn(
            f"Variables with very large or infinite bounds: "
            f"{', '.join(bad_vars[:5])}. "
            f"NLP solvers may fail (NaN, iteration_limit) when bounds "
            f"exceed ~1e15. Add tighter explicit bounds, e.g. "
            f"m.continuous('x', lb=0, ub=1000).",
            stacklevel=3,
        )


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
    mccormick_bounds: str = "auto",
    gdp_method: str = "big-m",
    initial_point: Optional[np.ndarray] = None,
    skip_convex_check: bool = False,
    nlp_bb: Optional[bool] = None,
    lazy_constraints=None,
    incumbent_callback=None,
    node_callback=None,
    use_highs_milp: bool = True,
    **kwargs,
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

    # --- OA decomposition: general-purpose Outer Approximation ---
    if gdp_method == "oa":
        from discopt.solvers.oa import solve_oa

        # Extract OA-specific kwargs that solve_model doesn't understand
        oa_kwargs = {}
        for key in ("equality_relaxation", "ecp_mode", "feasibility_cuts"):
            if key in kwargs:
                oa_kwargs[key] = kwargs.pop(key)

        return solve_oa(
            model,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            max_iterations=max_nodes,
            nlp_solver=nlp_solver,
            **oa_kwargs,
        )

    # --- LOA decomposition: intercept before GDP reformulation ---
    if gdp_method == "loa":
        from discopt.solvers.gdpopt_loa import solve_gdpopt_loa

        return solve_gdpopt_loa(
            model,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            max_iterations=max_nodes,
            nlp_solver=nlp_solver,
        )

    # --- GDP reformulation: convert indicator/disjunctive/SOS to standard MINLP ---
    from discopt._jax.gdp_reformulate import reformulate_gdp

    model = reformulate_gdp(model, method=gdp_method)

    # --- Build Rust model representation for FBBT ---
    _model_repr = None
    try:
        from discopt._rust import model_to_repr

        _builder = getattr(model, "_builder", None)
        _model_repr = model_to_repr(model, _builder)
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

    # --- Check for very large variable bounds ---
    # All solver paths (LP IPM, QP IPM, NLP) use barrier methods that
    # struggle with bounds beyond ~1e15. Check once before any dispatch.
    _check_finite_bounds(model)

    # --- Explicit NLP-BB override: bypass specialized solvers ---
    if nlp_bb is True and not _is_pure_continuous(model):
        return _solve_nlp_bb(
            model,
            time_limit,
            gap_tolerance,
            batch_size,
            strategy,
            max_nodes,
            t_start,
            nlp_solver,
            skip_convex_check=skip_convex_check,
            initial_point=initial_point,
            lazy_constraints=lazy_constraints,
            incumbent_callback=incumbent_callback,
            node_callback=node_callback,
        )

    # --- Problem classification: dispatch LP/QP to specialized solvers ---
    try:
        from discopt._jax.problem_classifier import ProblemClass, classify_problem

        problem_class = classify_problem(model)
    except Exception as e:
        logger.debug("Problem classification failed: %s", e)
        problem_class = None

    if problem_class is not None:
        if problem_class == ProblemClass.LP:
            return _solve_lp(model, t_start, time_limit)
        elif problem_class == ProblemClass.QP:
            return _solve_qp(model, t_start)
        elif problem_class == ProblemClass.MILP:
            if use_highs_milp:
                highs_result = _solve_milp_highs(model, t_start, time_limit, gap_tolerance)
                if highs_result is not None:
                    return highs_result
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
            # Try HiGHS MIQP first, fall back to B&B with QP relaxations
            highs_result = _solve_qp_highs(model, t_start, time_limit)
            if highs_result is not None:
                return highs_result
            return _solve_miqp_bb(
                model,
                time_limit,
                gap_tolerance,
                batch_size,
                strategy,
                max_nodes,
                t_start,
            )

    # --- Convex NLP fast path: skip B&B for convex continuous problems ---
    if _is_pure_continuous(model) and not skip_convex_check:
        try:
            from discopt._jax.convexity import classify_model as _classify_convexity

            # use_certificate=True enables the sound interval-Hessian
            # fallback for constraints/objective the syntactic walker
            # leaves unproven — tightens UNKNOWN to CONVEX when provable
            # on the root box, enabling the single-NLP fast path for
            # models whose convexity isn't visible at the DAG level.
            is_convex, _ = _classify_convexity(model, use_certificate=True)
            if is_convex:
                logger.info(
                    "Convex NLP detected — solving with single NLP (global optimality guaranteed)"
                )
                result = _solve_continuous(
                    model,
                    time_limit,
                    ipopt_options,
                    t_start,
                    nlp_solver,
                    initial_point=initial_point,
                )
                result.convex_fast_path = True
                return result
        except Exception as exc:
            logger.debug("Convex fast path detection failed: %s", exc)

    # --- Pure continuous: solve directly with NLP, no B&B needed ---
    if _is_pure_continuous(model):
        return _solve_continuous(
            model,
            time_limit,
            ipopt_options,
            t_start,
            nlp_solver,
            initial_point=initial_point,
        )

    # --- NLP-BB auto-select for convex MINLPs (nlp_bb=None) ---
    # Placed after problem classifier so MILP/MIQP use their specialized
    # (faster) solvers. Only genuinely nonlinear convex MINLPs reach here.
    # Also skip when lazy constraints are provided (they need the cut pool
    # infrastructure from the full spatial B&B loop).
    if nlp_bb is None and lazy_constraints is None:
        try:
            from discopt._jax.convexity import classify_model as _cls_conv

            _is_conv, _ = _cls_conv(model, use_certificate=True)
            if _is_conv:
                logger.info("Convex MINLP detected, using NLP-BB (nonlinear Branch and Bound)")
                return _solve_nlp_bb(
                    model,
                    time_limit,
                    gap_tolerance,
                    batch_size,
                    strategy,
                    max_nodes,
                    t_start,
                    nlp_solver,
                    skip_convex_check=skip_convex_check,
                    initial_point=initial_point,
                    lazy_constraints=lazy_constraints,
                    incumbent_callback=incumbent_callback,
                    node_callback=node_callback,
                )
        except Exception:
            pass

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
    cl_list, cu_list = _infer_constraint_bounds(model, evaluator)
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

    # --- Lazy constraint callback requires a cut pool ---
    if lazy_constraints is not None and _cut_pool is None:
        from discopt._jax.cutting_planes import CutPool

        _cut_pool = CutPool(max_cuts=500)

    # --- Convexity detection (Phase E) ---
    # Use the expression DAG convexity detector to:
    # (E2) Skip relaxation overhead for fully convex subproblems
    # (E3) Enable OA cuts per-constraint (not just for affine constraints)
    _model_is_convex = False
    _oa_enabled = False
    _convex_constraint_mask = None
    try:
        from discopt._jax.convexity import classify_model as _classify_model

        # use_certificate consults the sound interval-Hessian fallback
        # for constraints the syntactic walker leaves unproven. Tightens
        # _convex_constraint_mask entries to True when the certificate
        # proves convexity on the root box — enabling OA cuts and the
        # αBB skip below for structurally-convex problems whose
        # convexity the DCP walker alone cannot recognise.
        _model_is_convex, _convex_constraint_mask = _classify_model(model, use_certificate=True)
        if _model_is_convex:
            logger.info("Model detected as convex — NLP solutions are valid lower bounds")
        if cutting_planes and any(_convex_constraint_mask):
            _oa_enabled = True
    except Exception as exc:
        logger.debug("Convexity detection failed: %s", exc)
        if cutting_planes and model._constraints:
            _convex_constraint_mask = [False] * len(model._constraints)

    # Per-node certificate refresh imported lazily so the main
    # classification import above stays the single-point-of-truth for
    # "convexity is available at all". ``None`` disables per-node
    # refresh below (the root mask is then used verbatim).
    _refresh_mask: Any = None
    try:
        from discopt._jax.convexity import refresh_convex_mask as _refresh_mask_import

        _refresh_mask = _refresh_mask_import
    except Exception:
        pass

    # Enable nonconvex spatial branching so integer-feasible nodes are not
    # prematurely fathomed.  The NLP local optimum at such a node may not
    # be the global optimum of the continuous subproblem.
    if not _model_is_convex:
        tree.set_nonconvex(True)

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
            logger.warning("McCormick relaxation setup failed: %s", e)
            _mc_obj_eval = None
            _mc_obj_relax_fn = None

    # --- Warm-start: inject user-provided initial solution as incumbent ---
    if initial_point is not None:
        ws_obj = float(evaluator.evaluate_objective(initial_point))
        # Check integer feasibility of the warm-start point
        ws_int_feas = True
        for off, sz in zip(int_offsets, int_sizes):
            for j in range(off, off + sz):
                if abs(initial_point[j] - round(initial_point[j])) > 1e-5:
                    ws_int_feas = False
                    break
            if not ws_int_feas:
                break
        if ws_int_feas and np.isfinite(ws_obj) and ws_obj < _SENTINEL_THRESHOLD:
            ws_con_feas = not cl_list or _check_constraint_feasibility(
                evaluator, initial_point, cl_list, cu_list
            )
            if ws_con_feas:
                tree.inject_incumbent(initial_point, ws_obj)
                logger.info("Warm-start incumbent injected: obj=%.6g", ws_obj)
            else:
                logger.info(
                    "Warm-start point is integer-feasible but violates "
                    "constraints, using as NLP starting point only"
                )
        else:
            logger.info(
                "Warm-start point is not integer-feasible, using as NLP starting point only"
            )

    # --- Feasibility pump at root ---
    # Try to find an integer-feasible incumbent before B&B starts.
    _fp_ran = False

    # --- B&B loop ---
    # McCormick NLP is expensive (one IPM per node). Run it every N iterations
    # and use cheap midpoint bounds in between. Period=1 means every iteration.
    _mc_nlp_period = 5  # run McCormick NLP every 5th iteration
    iteration = 0
    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            break

        # Update per-iteration time budget for NLP subproblem solves (issue #5).
        remaining = time_limit - elapsed
        opts["max_wall_time"] = max(remaining, 0.1)

        # Export batch from Rust tree
        t_rust_start = time.perf_counter()
        batch_lb, batch_ub, batch_ids, batch_psols = tree.export_batch(batch_size)
        rust_time += time.perf_counter() - t_rust_start

        n_batch = len(batch_ids)
        if n_batch == 0:
            break

        # Tighten node bounds via constraint propagation (FBBT).
        if cl_list:
            for i in range(n_batch):
                node_lb_i = np.array(batch_lb[i])
                node_ub_i = np.array(batch_ub[i])
                t_lb, t_ub = _tighten_node_bounds(evaluator, node_lb_i, node_ub_i, cl_list, cu_list)
                batch_lb[i] = t_lb.tolist()
                batch_ub[i] = t_ub.tolist()

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
                multistart=True,  # IPM needs multistart even for convex models
                convex=_model_is_convex,
            )
            # Constraint feasibility post-check for batch IPM results.
            # When the IPM solution violates constraints (e.g. due to hitting
            # the iteration limit), mark the node as infeasible (SENTINEL).
            # This prevents the invalid solution from becoming the incumbent
            # and causes the Rust tree to prune the node.
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
            # For nonconvex problems, NLP objective is NOT a valid lower
            # bound (local minima can exceed the global optimum).  Reset ALL
            # non-sentinel nodes to -inf so only convex relaxation bounds are
            # used.  For integer-feasible nodes, inject the NLP solution as
            # an incumbent candidate via tree.inject_incumbent() and let the
            # Rust tree continue spatial branching on continuous variables.
            _int_feas_mask = np.zeros(n_batch, dtype=bool)
            if not _model_is_convex:
                _nlp_obj_backup = result_lbs.copy()
                for i in range(n_batch):
                    if result_lbs[i] < _SENTINEL_THRESHOLD:
                        sol_is_int_feas = True
                        for off, sz in zip(int_offsets, int_sizes):
                            for j in range(off, off + sz):
                                if abs(result_sols[i, j] - round(result_sols[i, j])) > 1e-5:
                                    sol_is_int_feas = False
                                    break
                            if not sol_is_int_feas:
                                break
                        _int_feas_mask[i] = sol_is_int_feas
                        if sol_is_int_feas:
                            # Inject NLP solution as incumbent candidate.
                            # The Rust tree will update its incumbent if this
                            # objective improves on the current best.
                            tree.inject_incumbent(result_sols[i].copy(), float(_nlp_obj_backup[i]))
                        # Reset ALL nonconvex nodes to -inf; convex bounds
                        # computed below will provide valid lower bounds.
                        result_lbs[i] = -np.inf
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
                    # Use NLP mode only periodically to avoid per-node IPM overhead.
                    # Midpoint mode is NOT a valid lower bound (cv(mid) >= min f(x)
                    # is not guaranteed), so skip McCormick on non-NLP iterations.
                    _use_mc_nlp = (
                        _mc_mode == "nlp"
                        and _mc_obj_relax_fn is not None
                        and (iteration == 0 or iteration % _mc_nlp_period == 0)
                    )
                    if _use_mc_nlp:
                        from discopt._jax.mccormick_nlp import solve_mccormick_batch

                        assert _mc_obj_relax_fn is not None
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
                    elif _mc_mode != "nlp":
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
                    else:
                        mc_lbs = None
                    if mc_lbs is not None:
                        for i in range(n_batch):
                            if result_lbs[i] < _SENTINEL_THRESHOLD and np.isfinite(mc_lbs[i]):
                                result_lbs[i] = max(result_lbs[i], float(mc_lbs[i]))
                except (ValueError, ArithmeticError, RuntimeError) as e:
                    logger.debug("Batch McCormick bound failed: %s", e)
            # For nonconvex problems with no convex relaxation available,
            # fall back to NLP objective as best-effort bound.
            if not _model_is_convex:
                for i in range(n_batch):
                    if result_lbs[i] == -np.inf and _nlp_obj_backup[i] < _SENTINEL_THRESHOLD:
                        result_lbs[i] = _nlp_obj_backup[i]
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
                        # More random starts for nonconvex problems
                        _root_n_random = 5 if not _model_is_convex else 2
                        nlp_result = _solve_root_node_multistart_ipm(
                            _active_evaluator,
                            node_lb,
                            node_ub,
                            _active_cb,
                            _active_gl,
                            _active_gu,
                            opts,
                            n_random=_root_n_random,
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
                    if _use_ipm_batch:
                        nlp_result = _solve_node_multistart_ipm(
                            _active_evaluator,
                            x0,
                            node_lb,
                            node_ub,
                            _active_cb,
                            _active_gl,
                            _active_gu,
                            opts,
                        )
                    else:
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
                    convex_lb = -np.inf  # accumulate valid convex lower bound

                    if _use_alphabb:
                        try:
                            relax_lb = _compute_alphabb_bound(
                                evaluator, node_lb, node_ub, _alphabb_alpha
                            )
                            nlp_lb = max(nlp_lb, relax_lb)
                            convex_lb = max(convex_lb, relax_lb)
                        except (ValueError, ArithmeticError, RuntimeError) as e:
                            logger.debug("alphaBB bound failed: %s", e)
                    # McCormick relaxation bound
                    if _mc_obj_relax_fn is not None:
                        try:
                            import jax.numpy as jnp

                            lb_j = jnp.array(node_lb, dtype=jnp.float64)
                            ub_j = jnp.array(node_ub, dtype=jnp.float64)
                            _use_mc_nlp_serial = _mc_mode == "nlp" and (
                                iteration == 0 or iteration % _mc_nlp_period == 0
                            )
                            if _use_mc_nlp_serial:
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
                            elif _mc_mode != "nlp":
                                from discopt._jax.mccormick_nlp import (
                                    evaluate_midpoint_bound,
                                )

                                mc_lb = evaluate_midpoint_bound(
                                    _mc_obj_relax_fn,
                                    lb_j,
                                    ub_j,
                                    negate=_mc_negate,
                                )
                            else:
                                mc_lb = -np.inf
                            if np.isfinite(mc_lb):
                                nlp_lb = max(nlp_lb, mc_lb)
                                convex_lb = max(convex_lb, mc_lb)
                        except (ValueError, ArithmeticError, RuntimeError) as e:
                            logger.debug("McCormick bound failed: %s", e)

                    # For nonconvex problems: NLP local min is NOT a valid
                    # lower bound (can exceed global opt → premature pruning).
                    # Use convex relaxation bound for non-integer-feasible
                    # nodes, but keep NLP objective for integer-feasible ones
                    # (the Rust tree uses result_lbs for incumbent values).
                    if not _model_is_convex and convex_lb > -np.inf:
                        sol_is_int_feas = True
                        for off, sz in zip(int_offsets, int_sizes):
                            for j in range(off, off + sz):
                                xj = nlp_result.x[j]
                                if not np.isfinite(xj) or abs(xj - round(xj)) > 1e-5:
                                    sol_is_int_feas = False
                                    break
                            if not sol_is_int_feas:
                                break
                        if not sol_is_int_feas:
                            nlp_lb = convex_lb

                    # Constraint feasibility post-check
                    if cl_list and not _check_constraint_feasibility(
                        _active_evaluator, nlp_result.x, cl_list, cu_list
                    ):
                        nlp_lb = _INFEASIBILITY_SENTINEL
                        logger.debug(
                            "Node %d: NLP solution violates constraints, marking infeasible",
                            int(batch_ids[i]),
                        )
                    # Guard: NaN lower bounds corrupt the Rust B&B tree
                    # (NaN comparisons always return False in IEEE 754).
                    if not np.isfinite(nlp_lb):
                        nlp_lb = _INFEASIBILITY_SENTINEL
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
                    # Refresh the convex-constraint mask on this node's
                    # tightened box: a constraint UNKNOWN at the root
                    # may be provably convex on the subtree. The
                    # refresh only flips False -> True (soundness
                    # preserved) and is skipped when every root entry
                    # is already True (nothing to tighten).
                    node_mask = _convex_constraint_mask
                    node_oa_enabled = _oa_enabled
                    if (
                        _refresh_mask is not None
                        and _convex_constraint_mask is not None
                        and not all(_convex_constraint_mask)
                    ):
                        try:
                            node_mask = _refresh_mask(
                                model, _convex_constraint_mask, node_lb_i, node_ub_i
                            )
                        except Exception as exc:
                            logger.debug("Per-node convexity refresh failed: %s", exc)
                            node_mask = _convex_constraint_mask
                        if (
                            node_mask is not _convex_constraint_mask
                            and node_mask is not None
                            and any(node_mask)
                        ):
                            node_oa_enabled = True
                    new_cuts = _generate_cuts(
                        evaluator,
                        model,
                        result_sols[i],
                        node_lb_i,
                        node_ub_i,
                        constraint_senses=_constraint_senses,
                        bilinear_terms=_bilinear_terms,
                        oa_enabled=node_oa_enabled,
                        convex_constraint_mask=node_mask,
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

        # --- User callbacks: lazy constraints and incumbent filtering ---
        if lazy_constraints is not None or incumbent_callback is not None:
            _invoke_pre_import_callbacks(
                model=model,
                tree=tree,
                t_start=t_start,
                result_ids=result_ids,
                result_lbs=result_lbs,
                result_sols=result_sols,
                result_feas=result_feas,
                n_batch=n_batch,
                int_offsets=int_offsets,
                int_sizes=int_sizes,
                n_vars=n_vars,
                lazy_constraints=lazy_constraints,
                incumbent_callback=incumbent_callback,
                _cut_pool=_cut_pool,
            )

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

        # --- Node callback: notify user after each batch ---
        if node_callback is not None:
            try:
                stats_snap = tree.stats()
                incumbent_info_cb = tree.incumbent()
                inc_obj_cb = None
                if incumbent_info_cb is not None:
                    _, inc_obj_cb = incumbent_info_cb
                    if inc_obj_cb >= _SENTINEL_THRESHOLD:
                        inc_obj_cb = None
                best_idx = 0
                for i in range(n_batch):
                    if result_lbs[i] < result_lbs[best_idx]:
                        best_idx = i
                from discopt.callbacks import CallbackContext

                ctx = CallbackContext(
                    node_count=stats_snap["total_nodes"],
                    incumbent_obj=inc_obj_cb,
                    best_bound=stats_snap.get("global_lower_bound", -np.inf),
                    gap=stats_snap.get("gap"),
                    elapsed_time=time.perf_counter() - t_start,
                    x_relaxation=result_sols[best_idx].copy(),
                    node_bound=float(result_lbs[best_idx]),
                )
                node_callback(ctx, model)
            except Exception as e:
                logger.warning("Node callback raised an exception: %s", e)

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

        # Negate objective back for maximization (B&B tree tracks minimization)
        from discopt.modeling.core import ObjectiveSense

        assert model._objective is not None
        if model._objective.sense == ObjectiveSense.MAXIMIZE:
            obj_val = -obj_val

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

    from discopt.modeling.core import ObjectiveSense

    # Negate bound back for maximization
    bound_val = stats["global_lower_bound"]
    assert model._objective is not None
    if bound_val is not None and model._objective.sense == ObjectiveSense.MAXIMIZE:
        bound_val = -bound_val

    return SolveResult(
        status=status,
        objective=obj_val,
        bound=bound_val,
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
    initial_point: Optional[np.ndarray] = None,
) -> SolveResult:
    """Solve a purely continuous model directly with NLP solver (no B&B)."""
    # Single-NLP solves need reliable KKT convergence. The pure-JAX IPM's
    # acceptable-tolerance check only covers bound complementarity, so on
    # problems with unbounded variables and inequality constraints it can
    # terminate at a non-KKT point and report OPTIMAL (false optimality).
    # B&B subproblems tolerate this because the tree catches it, but single
    # solves don't, so promote the default ipm -> ipopt here. Users who
    # explicitly requested ipm/ripopt/sparse_ipm still get what they asked for.
    if nlp_solver == "ipm":
        nlp_solver = "ipopt"

    t_jax_start = time.perf_counter()
    evaluator = _make_evaluator(model)
    jax_time = time.perf_counter() - t_jax_start

    lb, ub = evaluator.variable_bounds
    lb_clipped = np.clip(lb, -_SPC, _SPC)
    ub_clipped = np.clip(ub, -_SPC, _SPC)
    if initial_point is not None:
        x0 = np.clip(initial_point, lb, ub)
        logger.info("Using warm-start point for continuous NLP")
    else:
        x0 = 0.5 * (lb_clipped + ub_clipped)
        # Variables that are effectively unbounded on both sides collapse
        # to midpoint 0 above. Zero is a stationary point of periodic
        # functions (sin, cos) and other even functions, so single-start
        # local NLP gets stuck at a local max (e.g. cos(0) = 1). Nudge
        # unbounded coordinates to 0.5 so first-order methods can pick a
        # descent direction and escape the pathological start.
        fully_unbounded = (lb <= -_BOUND_WARN_THRESHOLD) & (ub >= _BOUND_WARN_THRESHOLD)
        x0 = np.where(fully_unbounded, 0.5, x0)
        # On problems with one-sided large bounds (e.g. x >= 1e-5 with no
        # upper bound), the midpoint of the clipped [-_SPC, _SPC] range
        # lands at ~50, which sends exp/log NLPs into overflow territory
        # and crashes ipopt. Tighten the starting-point range to keep
        # initial iterates in a numerically safe zone while still
        # respecting actual bounds.
        _X0_CLIP = 10.0
        x0 = np.clip(x0, np.maximum(lb, -_X0_CLIP), np.minimum(ub, _X0_CLIP))

    opts = dict(ipopt_options) if ipopt_options else {}
    opts.setdefault("print_level", 0)

    # Pass remaining time budget to NLP solver so stalled subproblems
    # don't run unbounded (see issue #5).
    remaining = time_limit - (time.perf_counter() - t_start)
    opts["max_wall_time"] = max(remaining, 0.1)

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

    # Negate objective back for maximization (NLPEvaluator solves minimization of -f)
    from discopt.modeling.core import ObjectiveSense

    assert model._objective is not None
    obj_val = nlp_result.objective
    if obj_val is not None and model._objective.sense == ObjectiveSense.MAXIMIZE:
        obj_val = -obj_val

    return SolveResult(
        status=status,
        objective=obj_val,
        bound=obj_val if status == "optimal" else None,
        gap=0.0 if status == "optimal" else None,
        x=x_dict,
        wall_time=wall_time,
        node_count=0,
        rust_time=0.0,
        jax_time=jax_time,
        python_time=python_time,
    )


def _solve_nlp_bb(
    model: Model,
    time_limit: float,
    gap_tolerance: float,
    batch_size: int,
    strategy: str,
    max_nodes: int,
    t_start: float,
    nlp_solver: str,
    skip_convex_check: bool = False,
    initial_point: Optional[np.ndarray] = None,
    lazy_constraints=None,
    incumbent_callback=None,
    node_callback=None,
) -> SolveResult:
    """Solve a MINLP via nonlinear Branch & Bound (NLP-BB).

    Instead of solving convex relaxations (McCormick/alphaBB) at each node,
    NLP-BB solves the original continuous NLP with discrete variables fixed
    via bound tightening.  For convex MINLPs, the NLP objective at each node
    is a valid lower bound, giving certified optimality gaps without any
    relaxation overhead.

    For nonconvex problems the NLP objective is NOT a valid lower bound;
    the solver runs in heuristic mode and reports gap_certified=False.
    """
    import jax.numpy as jnp

    from discopt._jax.gdp_reformulate import reformulate_gdp
    from discopt.modeling.core import ObjectiveSense

    model = reformulate_gdp(model, method="big-m")

    rust_time = 0.0
    jax_time = 0.0

    # --- Convexity gate ---
    _gap_certified = True
    try:
        from discopt._jax.convexity import classify_model as _classify_model

        _model_is_convex, _ = _classify_model(model, use_certificate=True)
    except Exception:
        _model_is_convex = False

    if not _model_is_convex and not skip_convex_check:
        logger.warning(
            "NLP-BB on nonconvex model: running in heuristic mode "
            "(gap not certified). Pass skip_convex_check=True to suppress."
        )
        _gap_certified = False

    # --- Extract variable info and create tree ---
    n_vars, lb, ub, int_offsets, int_sizes = _extract_variable_info(model)

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
    cl_list, cu_list = _infer_constraint_bounds(model, evaluator)
    constraint_bounds = list(zip(cl_list, cu_list)) if cl_list else None

    g_l_jax = None
    g_u_jax = None
    if nlp_solver == "ipm" and cl_list:
        g_l_jax = jnp.array(cl_list, dtype=jnp.float64)
        g_u_jax = jnp.array(cu_list, dtype=jnp.float64)

    opts: dict = {}
    opts.setdefault("print_level", 0)
    opts.setdefault("max_iter", 3000)
    opts.setdefault("tol", 1e-7)

    # --- Warm-start: inject user-provided initial solution as incumbent ---
    if initial_point is not None:
        ws_obj = float(evaluator.evaluate_objective(initial_point))
        ws_int_feas = True
        for off, sz in zip(int_offsets, int_sizes):
            for j in range(off, off + sz):
                if abs(initial_point[j] - round(initial_point[j])) > 1e-5:
                    ws_int_feas = False
                    break
            if not ws_int_feas:
                break
        if ws_int_feas and np.isfinite(ws_obj) and ws_obj < _SENTINEL_THRESHOLD:
            ws_con_feas = not cl_list or _check_constraint_feasibility(
                evaluator, initial_point, cl_list, cu_list
            )
            if ws_con_feas:
                tree.inject_incumbent(initial_point, ws_obj)
                logger.info("NLP-BB warm-start incumbent: obj=%.6g", ws_obj)

    # --- Feasibility pump flag ---
    _fp_ran = False

    # --- NLP-BB loop ---
    _use_ipm_batch = nlp_solver == "ipm" and hasattr(evaluator, "_obj_fn")
    iteration = 0
    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            break

        # Update per-iteration time budget for NLP subproblem solves (issue #5).
        remaining = time_limit - elapsed
        opts["max_wall_time"] = max(remaining, 0.1)

        # Export batch from Rust tree
        t_rust_start = time.perf_counter()
        batch_lb, batch_ub, batch_ids, batch_psols = tree.export_batch(batch_size)
        rust_time += time.perf_counter() - t_rust_start

        n_batch = len(batch_ids)
        if n_batch == 0:
            break

        # Tighten node bounds via constraint propagation (FBBT).
        # This resolves degenerate bounds (e.g., x <= M*y with y fixed at 0)
        # that cause IPM convergence failures.
        if cl_list:
            for i in range(n_batch):
                node_lb_i = np.array(batch_lb[i])
                node_ub_i = np.array(batch_ub[i])
                t_lb, t_ub = _tighten_node_bounds(evaluator, node_lb_i, node_ub_i, cl_list, cu_list)
                batch_lb[i] = t_lb.tolist()
                batch_ub[i] = t_ub.tolist()

        # Solve NLP at each node (no relaxation, no multistart for convex)
        t_jax_start = time.perf_counter()

        if _use_ipm_batch and n_batch > 1:
            result_ids, result_lbs, result_sols, result_feas = _solve_batch_ipm(
                evaluator,
                batch_lb,
                batch_ub,
                batch_ids,
                n_vars,
                constraint_bounds,
                opts,
                g_l_jax,
                g_u_jax,
                batch_psols=batch_psols,
                multistart=True,  # IPM needs multistart even for convex models
                convex=_model_is_convex,
            )
            # Constraint feasibility post-check
            if cl_list:
                for i in range(n_batch):
                    if result_lbs[i] < _SENTINEL_THRESHOLD:
                        if not _check_constraint_feasibility(
                            evaluator, result_sols[i], cl_list, cu_list
                        ):
                            result_lbs[i] = _INFEASIBILITY_SENTINEL
            # For nonconvex: NLP objective is NOT a valid lower bound.
            # Keep it for integer-feasible nodes (incumbent candidates),
            # but reset to -inf for others so we don't prune incorrectly.
            if not _model_is_convex:
                for i in range(n_batch):
                    if result_lbs[i] < _SENTINEL_THRESHOLD:
                        sol_is_int_feas = True
                        for off, sz in zip(int_offsets, int_sizes):
                            for j in range(off, off + sz):
                                frac = abs(result_sols[i, j] - round(result_sols[i, j]))
                                if frac > 1e-5:
                                    sol_is_int_feas = False
                                    break
                            if not sol_is_int_feas:
                                break
                        if not sol_is_int_feas:
                            result_lbs[i] = -np.inf
        else:
            # Serial fallback (batch_size=1 or non-IPM solver)
            result_ids = np.empty(n_batch, dtype=np.int64)
            result_lbs = np.empty(n_batch, dtype=np.float64)
            result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
            result_feas = np.empty(n_batch, dtype=bool)

            for i in range(n_batch):
                node_lb = np.array(batch_lb[i])
                node_ub = np.array(batch_ub[i])

                if iteration == 0 and _use_ipm_batch:
                    n_random = 2 if _model_is_convex else 5
                    nlp_result = _solve_root_node_multistart_ipm(
                        evaluator,
                        node_lb,
                        node_ub,
                        constraint_bounds,
                        g_l_jax,
                        g_u_jax,
                        opts,
                        n_random=n_random,
                        convex=_model_is_convex,
                    )
                elif iteration == 0:
                    nlp_result = _solve_root_node_multistart(
                        evaluator,
                        node_lb,
                        node_ub,
                        constraint_bounds,
                        opts,
                        nlp_solver,
                    )
                else:
                    psol_i = np.array(batch_psols[i])
                    if not np.any(np.isnan(psol_i)):
                        x0 = np.clip(psol_i, node_lb, node_ub)
                    else:
                        lb_c = np.clip(node_lb, -_SPC, _SPC)
                        ub_c = np.clip(node_ub, -_SPC, _SPC)
                        x0 = 0.5 * (lb_c + ub_c)
                    nlp_result = _solve_node_nlp(
                        evaluator,
                        x0,
                        node_lb,
                        node_ub,
                        constraint_bounds,
                        opts,
                        nlp_solver=nlp_solver,
                        convex=_model_is_convex,
                    )

                result_ids[i] = int(batch_ids[i])
                if nlp_result.status in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT):
                    nlp_lb = nlp_result.objective
                    # Constraint feasibility check
                    if cl_list and not _check_constraint_feasibility(
                        evaluator, nlp_result.x, cl_list, cu_list
                    ):
                        nlp_lb = _INFEASIBILITY_SENTINEL
                    # For nonconvex: reset non-integer-feasible to -inf
                    elif not _model_is_convex:
                        sol_is_int_feas = True
                        for off, sz in zip(int_offsets, int_sizes):
                            for j in range(off, off + sz):
                                xj = nlp_result.x[j]
                                if not np.isfinite(xj) or abs(xj - round(xj)) > 1e-5:
                                    sol_is_int_feas = False
                                    break
                            if not sol_is_int_feas:
                                break
                        if not sol_is_int_feas:
                            nlp_lb = -np.inf
                    # Guard: NaN lower bounds corrupt the Rust B&B tree.
                    if not np.isfinite(nlp_lb):
                        nlp_lb = _INFEASIBILITY_SENTINEL
                    result_lbs[i] = nlp_lb
                    result_sols[i] = nlp_result.x
                    result_feas[i] = False
                else:
                    result_lbs[i] = _INFEASIBILITY_SENTINEL
                    lb_c = np.clip(node_lb, -_SPC, _SPC)
                    ub_c = np.clip(node_ub, -_SPC, _SPC)
                    result_sols[i] = 0.5 * (lb_c + ub_c)
                    result_feas[i] = False

        jax_time += time.perf_counter() - t_jax_start

        # --- Feasibility pump after root node ---
        if iteration == 0 and not _fp_ran:
            _fp_ran = True
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
                            logger.info("NLP-BB feasibility pump incumbent: obj=%.6g", fp_obj)
                except Exception as e:
                    logger.debug("Feasibility pump failed: %s", e)

        # --- User callbacks ---
        if lazy_constraints is not None or incumbent_callback is not None:
            _invoke_pre_import_callbacks(
                model=model,
                tree=tree,
                t_start=t_start,
                result_ids=result_ids,
                result_lbs=result_lbs,
                result_sols=result_sols,
                result_feas=result_feas,
                n_batch=n_batch,
                int_offsets=int_offsets,
                int_sizes=int_sizes,
                n_vars=n_vars,
                lazy_constraints=lazy_constraints,
                incumbent_callback=incumbent_callback,
                _cut_pool=None,
            )

        # Import results back to Rust tree
        t_rust_start = time.perf_counter()
        tree.import_results(result_ids, result_lbs, result_sols, result_feas)
        tree.process_evaluated()
        rust_time += time.perf_counter() - t_rust_start

        # --- Node callback ---
        if node_callback is not None:
            try:
                stats_snap = tree.stats()
                incumbent_info_cb = tree.incumbent()
                inc_obj_cb = None
                if incumbent_info_cb is not None:
                    _, inc_obj_cb = incumbent_info_cb
                    if inc_obj_cb >= _SENTINEL_THRESHOLD:
                        inc_obj_cb = None
                best_idx = 0
                for i in range(n_batch):
                    if result_lbs[i] < result_lbs[best_idx]:
                        best_idx = i
                from discopt.callbacks import CallbackContext

                ctx = CallbackContext(
                    node_count=stats_snap["total_nodes"],
                    incumbent_obj=inc_obj_cb,
                    best_bound=stats_snap.get("global_lower_bound", -np.inf),
                    gap=stats_snap.get("gap"),
                    elapsed_time=time.perf_counter() - t_start,
                    x_relaxation=result_sols[best_idx].copy(),
                    node_bound=float(result_lbs[best_idx]),
                )
                node_callback(ctx, model)
            except Exception as e:
                logger.warning("Node callback raised an exception: %s", e)

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
        if obj_val >= _SENTINEL_THRESHOLD:
            incumbent = None

    if incumbent is not None:
        sol_flat = np.array(sol_array)
        x_dict = _unpack_solution(model, sol_flat)

        assert model._objective is not None
        if model._objective.sense == ObjectiveSense.MAXIMIZE:
            obj_val = -obj_val

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

    # Negate bound back for maximization
    bound_val = stats["global_lower_bound"]
    assert model._objective is not None
    if bound_val is not None and model._objective.sense == ObjectiveSense.MAXIMIZE:
        bound_val = -bound_val

    return SolveResult(
        status=status,
        objective=obj_val,
        bound=bound_val,
        gap=stats["gap"],
        x=x_dict,
        wall_time=wall_time,
        node_count=stats["total_nodes"],
        rust_time=rust_time,
        jax_time=jax_time,
        python_time=python_time,
        nlp_bb=True,
        gap_certified=_gap_certified,
    )


def _solve_node_nlp(
    evaluator: NLPEvaluator,
    x0: np.ndarray,
    node_lb: np.ndarray,
    node_ub: np.ndarray,
    constraint_bounds: Optional[list[tuple[float, float]]],
    options: dict,
    nlp_solver: str = "ipopt",
    convex: bool = False,
):
    """Solve the NLP relaxation at a single B&B node with tightened bounds.

    We override variable bounds to use the node-specific bounds
    rather than the global bounds.
    """
    # Pre-screen: detect trivially infeasible nodes by evaluating constraints
    # at the midpoint. When the feasible region is very narrow (most variables
    # pinned) and constraints are violated, NLP solvers like ripopt can stall
    # for thousands of iterations instead of quickly returning infeasible.
    if constraint_bounds is not None and evaluator.n_constraints > 0:
        from discopt.solvers import NLPResult

        x_mid = np.clip(x0, node_lb, node_ub)
        span = node_ub - node_lb
        n_pinned = np.sum(span < 1e-10)
        if n_pinned >= len(span) - 1:
            # Nearly all variables pinned: evaluate constraints at midpoint
            try:
                g = evaluator.evaluate_constraints(x_mid)
                infeasible = False
                for k, (cl, cu) in enumerate(constraint_bounds):
                    if g[k] < cl - 1e-6 or g[k] > cu + 1e-6:
                        infeasible = True
                        break
                if infeasible:
                    # Verify at the bounds midpoint too
                    x_check = 0.5 * (node_lb + node_ub)
                    g2 = evaluator.evaluate_constraints(x_check)
                    still_infeasible = False
                    for k, (cl, cu) in enumerate(constraint_bounds):
                        if g2[k] < cl - 1e-6 or g2[k] > cu + 1e-6:
                            still_infeasible = True
                            break
                    if still_infeasible:
                        return NLPResult(
                            status=SolveStatus.INFEASIBLE,
                            x=x_mid,
                            objective=_INFEASIBILITY_SENTINEL,
                        )
            except Exception:
                pass  # If evaluation fails, fall through to NLP solver

    if nlp_solver == "ripopt":
        return _solve_node_nlp_ripopt(evaluator, x0, node_lb, node_ub, constraint_bounds, options)
    if nlp_solver == "ipm":
        # JAX IPM requires JAX-compiled _obj_fn/_cons_fn; fall back to ipopt
        # for evaluators without these attributes.
        if not hasattr(evaluator, "_obj_fn"):
            return _solve_node_nlp_ipopt(
                evaluator, x0, node_lb, node_ub, constraint_bounds, options
            )
        return _solve_node_nlp_ipm(
            evaluator, x0, node_lb, node_ub, constraint_bounds, options, convex=convex
        )
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

    # Guard against ripopt stalling on degenerate/infeasible subproblems
    # by enforcing a per-node wall time limit. Cap at 30s per node, but
    # also respect the remaining global budget passed via options (issue #5).
    opts = dict(options)
    caller_limit = opts.get("max_wall_time", 30.0)
    if caller_limit <= 0:
        caller_limit = 30.0
    opts["max_wall_time"] = min(30.0, caller_limit)

    try:
        return solve_nlp_ripopt(
            proxy,
            x0,
            constraint_bounds=constraint_bounds,
            options=opts,
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
    convex: bool = False,
):
    """Solve node NLP with the pure-JAX IPM.

    When ``convex=True``, iterates that hit ``max_iter`` / stall (codes 3,
    4) are polished with cyipopt because a non-KKT IPM objective is not a
    valid lower bound for the convex NLP (issue #39).
    """
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
    max_wall_time = options.get("max_wall_time")

    try:
        t0 = time.perf_counter()
        state = ipm_solve(obj_fn, con_fn, x0_jax, x_l, x_u, g_l, g_u, ipm_opts)
        wall_time = time.perf_counter() - t0
    except Exception as e:
        logger.debug("IPM solver failed: %s", e)
        return NLPResult(status=SolveStatus.ERROR, x=x0, objective=_INFEASIBILITY_SENTINEL)

    conv = int(state.converged)

    obj_val = float(state.obj)
    x_sol = np.asarray(state.x)
    needs_recovery = conv == 5 or not np.isfinite(obj_val) or np.any(~np.isfinite(x_sol))

    # Feasibility restoration: when the IPM declares infeasible or diverges
    # to NaN, try to find a feasible point via restoration, then re-solve.
    if needs_recovery and con_fn is not None and g_l is not None and g_u is not None:
        from discopt._jax.ipm import _jax_feasibility_restoration

        x_rest_start = x0_jax if np.any(~np.isfinite(x_sol)) else state.x
        for _rest_attempt in range(3):
            try:
                x_restored, rest_ok = _jax_feasibility_restoration(
                    con_fn,
                    x_rest_start,
                    x_l,
                    x_u,
                    g_l,
                    g_u,
                    ipm_opts,
                )
            except Exception:
                break
            if not rest_ok:
                break
            try:
                state = ipm_solve(
                    obj_fn,
                    con_fn,
                    x_restored,
                    x_l,
                    x_u,
                    g_l,
                    g_u,
                    ipm_opts,
                )
            except Exception:
                break
            conv = int(state.converged)
            obj_val = float(state.obj)
            x_sol = np.asarray(state.x)
            if conv != 5 and np.isfinite(obj_val) and np.all(np.isfinite(x_sol)):
                break
            x_rest_start = x0_jax if np.any(~np.isfinite(x_sol)) else state.x

    # Check wall-time limit post-hoc (issue #5). The JIT-compiled
    # jax.lax.while_loop cannot check wall clock mid-iteration.
    wall_time = time.perf_counter() - t0
    exceeded_time = max_wall_time is not None and wall_time > max_wall_time
    if conv in (1, 2) and not exceeded_time:
        status = SolveStatus.OPTIMAL
    elif conv == 5:
        status = SolveStatus.INFEASIBLE
    elif conv in (3, 4) or exceeded_time:
        status = SolveStatus.ITERATION_LIMIT
    else:
        status = SolveStatus.ERROR

    # Final NaN guard — if restoration also failed, mark as error.
    if not np.isfinite(obj_val) or np.any(~np.isfinite(x_sol)):
        return NLPResult(
            status=SolveStatus.ERROR,
            x=x0,
            objective=_INFEASIBILITY_SENTINEL,
        )

    # Convex polish: codes 3/4 are not KKT, so obj_val is not a valid LB
    # for a convex NLP. Re-solve with cyipopt. See issue #39.
    if convex and conv in (3, 4):
        try:
            polish = _solve_node_nlp_ipopt(
                evaluator, x_sol, node_lb, node_ub, constraint_bounds, options
            )
        except Exception as e:
            logger.debug("IPM convex polish failed (single-node): %s", e)
            polish = None
        if polish is not None and polish.status in (
            SolveStatus.OPTIMAL,
            SolveStatus.ITERATION_LIMIT,
        ):
            p_obj = float(polish.objective)
            if np.isfinite(p_obj) and p_obj < _SENTINEL_THRESHOLD:
                return NLPResult(
                    status=polish.status,
                    x=np.asarray(polish.x),
                    objective=p_obj,
                )

    return NLPResult(
        status=status,
        x=x_sol,
        objective=obj_val,
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
    multistart=False,
    convex=False,
):
    """Solve a batch of NLP relaxations simultaneously via vmap'd IPM.

    When multistart=True, each node gets 3 starting points (warm-start,
    midpoint, random) solved in parallel, with the best converged solution
    selected per node.

    When ``convex=True``, the caller is treating the NLP objective at each
    node as a valid lower bound (sound only for convex models). IPM
    iterates that hit ``max_iter`` (code 3) or stall (code 4) are not at
    KKT stationarity and their objective is not a reliable lower bound, so
    those nodes get a polish pass with cyipopt.
    """
    import jax.numpy as jnp

    from discopt._jax.ipm import (
        IPMOptions,
        _jax_feasibility_restoration,
        ipm_solve,
        solve_nlp_batch,
    )

    n_batch = len(batch_ids)
    obj_fn = evaluator._obj_fn
    m = evaluator.n_constraints
    con_fn = evaluator._cons_fn if m > 0 else None

    # Build (batch, n) JAX arrays for bounds and starting points
    xl_batch = jnp.array(batch_lb, dtype=jnp.float64)
    xu_batch = jnp.array(batch_ub, dtype=jnp.float64)

    # Warm-start: use parent solutions clipped to child bounds, fall back to midpoint
    lb_clipped = jnp.clip(xl_batch, -_SPC, _SPC)
    ub_clipped = jnp.clip(xu_batch, -_SPC, _SPC)
    midpoint_x0 = 0.5 * (lb_clipped + ub_clipped)

    if batch_psols is not None:
        psols_jax = jnp.array(batch_psols, dtype=jnp.float64)
        has_parent = ~jnp.any(jnp.isnan(psols_jax), axis=1, keepdims=True)
        warm_x0 = jnp.clip(psols_jax, xl_batch, xu_batch)
        x0_batch = jnp.where(has_parent, warm_x0, midpoint_x0)
    else:
        x0_batch = midpoint_x0

    n_starts = 1
    if multistart:
        # Expand: 3 starting points per node (warm-start, midpoint, random)
        # More starts give better solutions but 3x cost per node.
        n_starts = 3
        span = jnp.maximum(ub_clipped - lb_clipped, 0.0)
        # Random starting point (deterministic seed)
        rng = np.random.RandomState(42)
        rand_offsets = jnp.array(rng.uniform(size=(n_batch, n_vars)), dtype=jnp.float64)
        rand_x0 = lb_clipped + rand_offsets * span

        # Stack: (n_batch, 3, n_vars) then reshape to (n_batch*3, n_vars)
        x0_expanded = jnp.stack([x0_batch, midpoint_x0, rand_x0], axis=1)
        x0_batch = x0_expanded.reshape(n_batch * n_starts, n_vars)
        # Repeat bounds to match interleaved order
        xl_expanded = jnp.broadcast_to(xl_batch[:, None, :], (n_batch, n_starts, n_vars)).reshape(
            n_batch * n_starts, n_vars
        )
        xu_expanded = jnp.broadcast_to(xu_batch[:, None, :], (n_batch, n_starts, n_vars)).reshape(
            n_batch * n_starts, n_vars
        )
        xl_batch = xl_expanded
        xu_batch = xu_expanded

    ipm_opts = IPMOptions(max_iter=int(options.get("max_iter", 200)))

    try:
        state = solve_nlp_batch(
            obj_fn, con_fn, x0_batch, xl_batch, xu_batch, g_l_jax, g_u_jax, ipm_opts
        )
    except Exception as e:
        logger.debug("Batch IPM failed: %s", e)
        # Fallback: mark all as infeasible
        result_ids = np.array(batch_ids, dtype=np.int64)
        result_lbs = np.full(n_batch, _INFEASIBILITY_SENTINEL, dtype=np.float64)
        result_sols = np.asarray(midpoint_x0 if not multistart else x0_expanded[:, 0, :])
        result_feas = np.zeros(n_batch, dtype=bool)
        return result_ids, result_lbs, result_sols, result_feas

    # Unpack batched IPMState → numpy arrays
    converged = np.asarray(state.converged)
    obj_vals = np.asarray(state.obj)
    x_vals = np.asarray(state.x)

    if multistart:
        # Reshape (n_batch*n_starts,) → (n_batch, n_starts), pick best per node
        converged = converged.reshape(n_batch, n_starts)
        obj_vals = obj_vals.reshape(n_batch, n_starts)
        x_vals = x_vals.reshape(n_batch, n_starts, n_vars)

        # Accept converged (1=optimal, 2=acceptable), iteration-limit (3),
        # and stalled (4) solutions. Code 5 (infeasible) is excluded.
        # Constraint feasibility is checked separately in the caller;
        # iteration-limit solutions that violate constraints are handled
        # there (not pruned, but given a conservative lower bound).
        feasible_mask = (
            (converged == 1) | (converged == 2) | (converged == 3) | (converged == 4)
        ) & np.isfinite(obj_vals)
        masked_obj = np.where(feasible_mask, obj_vals, np.inf)
        best_per_node = np.argmin(masked_obj, axis=1)  # (n_batch,)

        result_obj = np.array([obj_vals[i, best_per_node[i]] for i in range(n_batch)])
        result_x = np.array([x_vals[i, best_per_node[i]] for i in range(n_batch)], dtype=np.float64)
        any_feasible = np.any(feasible_mask, axis=1)
        result_lbs = np.asarray(
            np.where(any_feasible, result_obj, _INFEASIBILITY_SENTINEL), dtype=np.float64
        )
        result_sols = result_x  # already writable np.array

        # Convex polish: IPM codes 3 (max_iter) / 4 (stalled) don't certify
        # KKT stationarity, so their objective is not a valid lower bound
        # for a convex NLP. Re-solve those nodes with cyipopt and use its
        # (reliable) optimum as the LB. See issue #39 (synthes2 B&B node
        # stalls at 92.10 vs true 73.04).
        if convex:
            best_codes = np.array([int(converged[i, best_per_node[i]]) for i in range(n_batch)])
            polish_needed = any_feasible & ((best_codes == 3) | (best_codes == 4))
            for i in np.where(polish_needed)[0]:
                row = int(i * n_starts)
                node_lb_i = np.asarray(xl_batch[row])
                node_ub_i = np.asarray(xu_batch[row])
                try:
                    polish = _solve_node_nlp_ipopt(
                        evaluator,
                        result_sols[i],
                        node_lb_i,
                        node_ub_i,
                        constraint_bounds,
                        options,
                    )
                except Exception as e:
                    logger.debug("IPM convex polish failed at node %d: %s", i, e)
                    continue
                if polish.status in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT):
                    polished_obj = float(polish.objective)
                    if np.isfinite(polished_obj) and polished_obj < _SENTINEL_THRESHOLD:
                        result_lbs[i] = polished_obj
                        result_sols[i] = np.asarray(polish.x, dtype=np.float64)
    else:
        conv_mask = (converged == 1) | (converged == 2) | (converged == 3) | (converged == 4)
        ok_mask = conv_mask & np.isfinite(obj_vals)
        result_lbs = np.asarray(
            np.where(ok_mask, obj_vals, _INFEASIBILITY_SENTINEL),
            dtype=np.float64,
        )
        result_sols = np.array(x_vals, dtype=np.float64)  # writable copy

        # Restoration: for failed nodes (NaN or code 5), try to recover
        # a feasible point via restoration and re-solve individually.
        failed_indices = np.where(~ok_mask)[0]
        if len(failed_indices) > 0 and con_fn is not None and g_l_jax is not None:
            for idx in failed_indices:
                x_start = x0_batch[idx] if not np.all(np.isfinite(x_vals[idx])) else x_vals[idx]
                try:
                    x_restored, rest_ok = _jax_feasibility_restoration(
                        con_fn,
                        x_start,
                        xl_batch[idx] if xl_batch.ndim == 2 else xl_batch,
                        xu_batch[idx] if xu_batch.ndim == 2 else xu_batch,
                        g_l_jax,
                        g_u_jax,
                        ipm_opts,
                    )
                except Exception:
                    continue
                if rest_ok:
                    try:
                        state_i = ipm_solve(
                            obj_fn,
                            con_fn,
                            x_restored,
                            xl_batch[idx] if xl_batch.ndim == 2 else xl_batch,
                            xu_batch[idx] if xu_batch.ndim == 2 else xu_batch,
                            g_l_jax,
                            g_u_jax,
                            ipm_opts,
                        )
                    except Exception:
                        continue
                    conv_i = int(state_i.converged)
                    obj_i = float(state_i.obj)
                    x_i = np.asarray(state_i.x)
                    if conv_i in (1, 2, 3, 4) and np.isfinite(obj_i) and np.all(np.isfinite(x_i)):
                        result_lbs[idx] = obj_i
                        result_sols[idx] = x_i

    result_ids = np.array(batch_ids, dtype=np.int64)
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

    # cyipopt requires native Python types (rejects numpy scalars).
    # Some options (e.g. max_wall_time) may not exist in older Ipopt versions.
    for key, value in options.items():
        try:
            if isinstance(value, (np.floating, float)):
                problem.add_option(key, float(value))
            elif isinstance(value, (np.integer, int)):
                problem.add_option(key, int(value))
            else:
                problem.add_option(key, value)
        except TypeError:
            logger.debug("Ipopt option '%s' not accepted, skipping", key)

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


def _decompose_eq_slack_form(
    A_eq_full: np.ndarray,
    b_eq_full: np.ndarray,
    n_orig: int,
    n_slack: int,
) -> tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """Reconstruct (A_ub, b_ub, A_eq, b_eq) from an equality-plus-slack form.

    `extract_lp_data` / `extract_qp_data` convert inequalities to equalities
    with non-negative slacks. Rows whose slack column is nonzero are the
    original inequalities; rows with no slack are true equalities. This
    helper projects back to inequality/equality form over the original
    variables so HiGHS LP/MILP/QP solvers can consume it.
    """
    if A_eq_full.shape[0] == 0:
        return None, None, None, None

    eq_rows: list[np.ndarray] = []
    eq_rhs: list[float] = []
    ub_rows: list[np.ndarray] = []
    ub_rhs: list[float] = []

    for i in range(A_eq_full.shape[0]):
        slack_part = A_eq_full[i, n_orig:]
        has_slack = n_slack > 0 and np.any(np.abs(slack_part) > 1e-15)
        if has_slack:
            slack_idx = np.argmax(np.abs(slack_part))
            slack_coef = slack_part[slack_idx]
            orig_row = A_eq_full[i, :n_orig]
            rhs = b_eq_full[i]
            if slack_coef > 0:
                ub_rows.append(orig_row)
                ub_rhs.append(rhs)
            else:
                ub_rows.append(-orig_row)
                ub_rhs.append(-rhs)
        else:
            eq_rows.append(A_eq_full[i, :n_orig])
            eq_rhs.append(b_eq_full[i])

    A_ub = np.array(ub_rows, dtype=np.float64) if ub_rows else None
    b_ub = np.array(ub_rhs, dtype=np.float64) if ub_rows else None
    A_eq = np.array(eq_rows, dtype=np.float64) if eq_rows else None
    b_eq = np.array(eq_rhs, dtype=np.float64) if eq_rows else None
    return A_ub, b_ub, A_eq, b_eq


def _solve_lp(model: Model, t_start: float, time_limit: float | None = None) -> SolveResult:
    """Solve an LP, preferring HiGHS and falling back to the pure-JAX LP IPM.

    The pure-JAX IPM struggles on problems whose declared bounds exceed
    ~1e15 (it returns NaN via Newton blow-up on unbounded variables); HiGHS
    handles unbounded columns natively and is also usually faster. We try
    HiGHS first and fall back to the IPM only when HiGHS is unavailable.
    """
    highs_result = _solve_lp_highs(model, t_start, time_limit)
    if highs_result is not None:
        return highs_result

    from discopt._jax.lp_ipm import lp_ipm_solve
    from discopt._jax.problem_classifier import extract_lp_data

    t_jax_start = time.perf_counter()
    lp_data = extract_lp_data(model)
    state = lp_ipm_solve(lp_data.c, lp_data.A_eq, lp_data.b_eq, lp_data.x_l, lp_data.x_u)
    jax_time = time.perf_counter() - t_jax_start
    wall_time = time.perf_counter() - t_start

    from discopt.modeling.core import ObjectiveSense

    n_orig = sum(v.size for v in model._variables)
    x_flat = np.asarray(state.x[:n_orig])
    obj_val = float(state.obj) + lp_data.obj_const

    # Negate objective back for maximization (LP solver always minimizes)
    assert model._objective is not None
    if model._objective.sense == ObjectiveSense.MAXIMIZE:
        obj_val = -obj_val

    conv = int(state.converged)
    if conv in (1, 2):
        status = "optimal"
    elif conv == 3:
        status = "iteration_limit"
    else:
        status = "error"

    sr = SolveResult(
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
    # LPs are convex by definition; mark for parity with the QP/NLP fast paths.
    sr.convex_fast_path = True
    return sr


def _solve_lp_highs(
    model: Model,
    t_start: float,
    time_limit: float | None = None,
) -> SolveResult | None:
    """Solve an LP using HiGHS. Returns None when HiGHS is unavailable or
    the HiGHS wrapper fails, so the caller can fall back to the JAX IPM."""
    try:
        from discopt.solvers.lp_highs import solve_lp as _highs_solve_lp
    except ImportError:
        return None

    from discopt._jax.problem_classifier import extract_lp_data
    from discopt.modeling.core import ObjectiveSense
    from discopt.solvers import SolveStatus

    lp_data = extract_lp_data(model)
    n_orig = sum(v.size for v in model._variables)

    bounds = list(
        zip(
            np.asarray(lp_data.x_l[:n_orig]).tolist(),
            np.asarray(lp_data.x_u[:n_orig]).tolist(),
        )
    )

    n_total = lp_data.A_eq.shape[1] if lp_data.A_eq.shape[0] > 0 else n_orig
    n_slack = n_total - n_orig
    A_eq_full = np.asarray(lp_data.A_eq)
    b_eq_full = np.asarray(lp_data.b_eq)
    A_ub, b_ub, A_eq, b_eq = _decompose_eq_slack_form(A_eq_full, b_eq_full, n_orig, n_slack)

    try:
        result = _highs_solve_lp(
            c=np.asarray(lp_data.c[:n_orig]),
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            time_limit=time_limit,
        )
    except Exception as e:
        logger.debug("HiGHS LP solve failed: %s", e)
        return None

    wall_time = time.perf_counter() - t_start

    if result.status == SolveStatus.OPTIMAL:
        assert result.x is not None and result.objective is not None
        obj_val = float(result.objective) + float(lp_data.obj_const)
        assert model._objective is not None
        if model._objective.sense == ObjectiveSense.MAXIMIZE:
            obj_val = -obj_val

        sr = SolveResult(
            status="optimal",
            objective=obj_val,
            bound=obj_val,
            gap=0.0,
            x=_unpack_solution(model, np.asarray(result.x[:n_orig])),
            wall_time=wall_time,
            node_count=0,
            rust_time=0.0,
            jax_time=0.0,
            python_time=wall_time,
        )
        sr.convex_fast_path = True
        return sr
    if result.status == SolveStatus.INFEASIBLE:
        return SolveResult(status="infeasible", wall_time=wall_time)
    if result.status == SolveStatus.UNBOUNDED:
        return SolveResult(status="unbounded", wall_time=wall_time)
    if result.status == SolveStatus.TIME_LIMIT:
        return SolveResult(status="time_limit", wall_time=wall_time)
    return None


def _solve_qp(model: Model, t_start: float) -> SolveResult:
    """Solve a QP, preferring HiGHS when available, falling back to JAX IPM."""
    result = _solve_qp_highs(model, t_start)
    if result is not None:
        return result
    return _solve_qp_jax(model, t_start)


def _solve_qp_highs(
    model: Model,
    t_start: float,
    time_limit: float | None = None,
) -> SolveResult | None:
    """Solve a QP/MIQP using HiGHS. Returns None if HiGHS is unavailable."""
    try:
        from discopt.solvers.qp_highs import solve_qp as _highs_solve_qp
    except ImportError:
        return None

    from discopt._jax.problem_classifier import extract_qp_data
    from discopt.modeling.core import ObjectiveSense
    from discopt.solvers import SolveStatus

    qp_data = extract_qp_data(model)
    n_orig = sum(v.size for v in model._variables)

    # Build bounds list (original variables only, no slacks)
    bounds = list(
        zip(
            np.asarray(qp_data.x_l[:n_orig]).tolist(),
            np.asarray(qp_data.x_u[:n_orig]).tolist(),
        )
    )

    n_total = qp_data.A_eq.shape[1] if qp_data.A_eq.shape[0] > 0 else n_orig
    n_slack = n_total - n_orig
    A_eq_full = np.asarray(qp_data.A_eq)
    b_eq_full = np.asarray(qp_data.b_eq)
    A_ub, b_ub, A_eq, b_eq = _decompose_eq_slack_form(A_eq_full, b_eq_full, n_orig, n_slack)

    # Build integrality array for MIQP
    integrality = None
    has_integer = any(v.var_type in (VarType.BINARY, VarType.INTEGER) for v in model._variables)
    if has_integer:
        int_arr = np.zeros(n_orig, dtype=np.int32)
        offset = 0
        for v in model._variables:
            if v.var_type in (VarType.BINARY, VarType.INTEGER):
                int_arr[offset : offset + v.size] = 1
            offset += v.size
        integrality = int_arr

    # Q matrix: only the original variable part (no slacks)
    Q_orig = np.asarray(qp_data.Q[:n_orig, :n_orig])
    c_orig = np.asarray(qp_data.c[:n_orig])

    try:
        result = _highs_solve_qp(
            Q=Q_orig,
            c=c_orig,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            integrality=integrality,
            time_limit=time_limit,
        )
    except Exception as e:
        logger.debug("HiGHS QP solve failed: %s", e)
        return None

    wall_time = time.perf_counter() - t_start

    if result.status == SolveStatus.OPTIMAL:
        assert result.x is not None and result.objective is not None
        x_flat = result.x[:n_orig]
        obj_val = result.objective + qp_data.obj_const

        assert model._objective is not None
        if model._objective.sense == ObjectiveSense.MAXIMIZE:
            obj_val = -obj_val

        sr = SolveResult(
            status="optimal",
            objective=obj_val,
            bound=obj_val,
            gap=0.0,
            x=_unpack_solution(model, x_flat),
            wall_time=wall_time,
            node_count=result.node_count,
            rust_time=0.0,
            jax_time=0.0,
            python_time=wall_time,
        )
        # A detected QP with PSD Q is a convex problem solved directly without
        # B&B -- semantically the same as the convex NLP fast path.
        if integrality is None:
            sr.convex_fast_path = True
        return sr
    elif result.status == SolveStatus.INFEASIBLE:
        return SolveResult(status="infeasible", wall_time=wall_time)
    elif result.status == SolveStatus.TIME_LIMIT:
        return SolveResult(status="time_limit", wall_time=wall_time)

    return None


def _solve_milp_highs(
    model: Model,
    t_start: float,
    time_limit: float | None = None,
    gap_tolerance: float = 1e-4,
) -> SolveResult | None:
    """Solve a MILP using HiGHS MIP. Returns None if HiGHS is unavailable."""
    try:
        from discopt.solvers.milp_highs import solve_milp as _highs_solve_milp
    except ImportError:
        return None

    from discopt._jax.problem_classifier import extract_lp_data
    from discopt.modeling.core import ObjectiveSense
    from discopt.solvers import SolveStatus

    lp_data = extract_lp_data(model)
    n_orig = sum(v.size for v in model._variables)

    bounds = list(
        zip(
            np.asarray(lp_data.x_l[:n_orig]).tolist(),
            np.asarray(lp_data.x_u[:n_orig]).tolist(),
        )
    )

    n_total = lp_data.A_eq.shape[1] if lp_data.A_eq.shape[0] > 0 else n_orig
    n_slack = n_total - n_orig
    A_eq_full = np.asarray(lp_data.A_eq)
    b_eq_full = np.asarray(lp_data.b_eq)
    A_ub, b_ub, A_eq, b_eq = _decompose_eq_slack_form(A_eq_full, b_eq_full, n_orig, n_slack)

    int_arr = np.zeros(n_orig, dtype=np.int32)
    offset = 0
    for v in model._variables:
        if v.var_type in (VarType.BINARY, VarType.INTEGER):
            int_arr[offset : offset + v.size] = 1
        offset += v.size

    c_orig = np.asarray(lp_data.c[:n_orig])

    try:
        result = _highs_solve_milp(
            c=c_orig,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            integrality=int_arr,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
        )
    except Exception as e:
        logger.debug("HiGHS MILP solve failed: %s", e)
        return None

    wall_time = time.perf_counter() - t_start

    assert model._objective is not None
    sense = model._objective.sense

    if result.status == SolveStatus.OPTIMAL:
        assert result.x is not None and result.objective is not None
        x_flat = result.x[:n_orig]
        obj_val = result.objective + lp_data.obj_const
        if sense == ObjectiveSense.MAXIMIZE:
            obj_val = -obj_val
        return SolveResult(
            status="optimal",
            objective=obj_val,
            bound=obj_val,
            gap=result.gap if result.gap is not None else 0.0,
            x=_unpack_solution(model, x_flat),
            wall_time=wall_time,
            node_count=result.node_count,
            rust_time=0.0,
            jax_time=0.0,
            python_time=wall_time,
        )
    elif result.status == SolveStatus.INFEASIBLE:
        return SolveResult(status="infeasible", wall_time=wall_time, node_count=result.node_count)
    elif result.status == SolveStatus.TIME_LIMIT:
        return SolveResult(status="time_limit", wall_time=wall_time, node_count=result.node_count)

    return None


def _solve_qp_jax(model: Model, t_start: float) -> SolveResult:
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

    from discopt.modeling.core import ObjectiveSense

    n_orig = sum(v.size for v in model._variables)
    x_flat = np.asarray(state.x[:n_orig])
    obj_val = float(state.obj) + qp_data.obj_const

    # Negate objective back for maximization (QP solver always minimizes)
    assert model._objective is not None
    if model._objective.sense == ObjectiveSense.MAXIMIZE:
        obj_val = -obj_val

    conv = int(state.converged)
    if conv in (1, 2):
        status = "optimal"
    elif conv == 3:
        status = "iteration_limit"
    else:
        status = "error"

    sr = SolveResult(
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
    # QP dispatch only reaches this function for detected convex QPs.
    sr.convex_fast_path = True
    return sr


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
                result_lbs = np.asarray(
                    np.where(ok, obj_vals + float(lp_data.obj_const), _INFEASIBILITY_SENTINEL),
                    dtype=np.float64,
                )
                result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
                for i in range(n_batch):
                    if ok[i]:
                        result_sols[i] = x_vals[i, :n_vars]
                        # Reject LP solutions that violate constraints
                        if not _check_lp_solution_feasibility(
                            lp_data.A_eq, lp_data.b_eq, x_vals[i]
                        ):
                            result_lbs[i] = _INFEASIBILITY_SENTINEL
                            lb_c = np.clip(np.array(batch_lb[i]), -_SPC, _SPC)
                            ub_c = np.clip(np.array(batch_ub[i]), -_SPC, _SPC)
                            result_sols[i] = 0.5 * (lb_c + ub_c)
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
                        # Reject LP solutions that violate constraints
                        if _check_lp_solution_feasibility(lp_data.A_eq, lp_data.b_eq, state.x):
                            result_lbs[i] = float(state.obj) + lp_data.obj_const
                            result_sols[i] = np.asarray(state.x[:n_vars])
                        else:
                            result_lbs[i] = _INFEASIBILITY_SENTINEL
                            lb_c = np.clip(node_lb, -_SPC, _SPC)
                            ub_c = np.clip(node_ub, -_SPC, _SPC)
                            result_sols[i] = 0.5 * (lb_c + ub_c)
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

        # Negate objective back for maximization (B&B tree tracks minimization)
        from discopt.modeling.core import ObjectiveSense

        assert model._objective is not None
        if model._objective.sense == ObjectiveSense.MAXIMIZE:
            obj_val = -obj_val

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

    from discopt.modeling.core import ObjectiveSense

    # Negate bound back for maximization
    bound_val = stats["global_lower_bound"]
    assert model._objective is not None
    if bound_val is not None and model._objective.sense == ObjectiveSense.MAXIMIZE:
        bound_val = -bound_val

    return SolveResult(
        status=status,
        objective=obj_val,
        bound=bound_val,
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
                result_lbs = np.asarray(
                    np.where(ok, obj_vals + float(qp_data.obj_const), _INFEASIBILITY_SENTINEL),
                    dtype=np.float64,
                )
                result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
                for i in range(n_batch):
                    if ok[i]:
                        result_sols[i] = x_vals[i, :n_vars]
                        # Reject QP solutions that violate constraints
                        if not _check_lp_solution_feasibility(
                            qp_data.A_eq, qp_data.b_eq, x_vals[i]
                        ):
                            result_lbs[i] = _INFEASIBILITY_SENTINEL
                            lb_c = np.clip(np.array(batch_lb[i]), -_SPC, _SPC)
                            ub_c = np.clip(np.array(batch_lb[i]), -_SPC, _SPC)
                            result_sols[i] = 0.5 * (lb_c + ub_c)
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
                        # Reject QP solutions that violate constraints
                        if _check_lp_solution_feasibility(qp_data.A_eq, qp_data.b_eq, state.x):
                            result_lbs[i] = float(state.obj) + qp_data.obj_const
                            result_sols[i] = np.asarray(state.x[:n_vars])
                        else:
                            result_lbs[i] = _INFEASIBILITY_SENTINEL
                            lb_c = np.clip(node_lb, -_SPC, _SPC)
                            ub_c = np.clip(node_ub, -_SPC, _SPC)
                            result_sols[i] = 0.5 * (lb_c + ub_c)
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

        # Negate objective back for maximization (B&B tree tracks minimization)
        from discopt.modeling.core import ObjectiveSense

        assert model._objective is not None
        if model._objective.sense == ObjectiveSense.MAXIMIZE:
            obj_val = -obj_val

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

    from discopt.modeling.core import ObjectiveSense

    # Negate bound back for maximization
    bound_val = stats["global_lower_bound"]
    assert model._objective is not None
    if bound_val is not None and model._objective.sense == ObjectiveSense.MAXIMIZE:
        bound_val = -bound_val

    return SolveResult(
        status=status,
        objective=obj_val,
        bound=bound_val,
        gap=stats["gap"],
        x=x_dict,
        wall_time=wall_time,
        node_count=stats["total_nodes"],
        rust_time=rust_time,
        jax_time=jax_time,
        python_time=python_time,
    )
