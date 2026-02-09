"""
Solver orchestrator: end-to-end Model.solve() via NLP-based spatial Branch & Bound.

Connects:
  - PyTreeManager (Rust B&B engine) for node management / branching / pruning
  - NLPEvaluator (JAX) for objective/gradient/Hessian/constraint/Jacobian
  - solve_nlp (cyipopt) for continuous relaxation solves at each node
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt._rust import PyTreeManager
from discopt.modeling.core import (
    Constraint,
    Model,
    SolveResult,
    VarType,
)
from discopt.solvers import SolveStatus
from discopt.solvers.nlp_ipopt import solve_nlp


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


def _has_nl_repr(model: Model) -> bool:
    """Check if model was loaded from a .nl file."""
    return hasattr(model, "_nl_repr") and model._nl_repr is not None


def _make_evaluator(model: Model):
    """Create the appropriate evaluator for the model."""
    if _has_nl_repr(model):
        from discopt._jax.nl_evaluator import NLPEvaluatorFromNl

        return NLPEvaluatorFromNl(model)
    return NLPEvaluator(model)


def _infer_nl_constraint_bounds(model: Model):
    """Infer (cl, cu) from .nl constraint senses and rhs values.

    For .nl files, constraint body is evaluated by Rust, and the sense/rhs
    are stored separately. We need:
      - '<=' constraints: body <= rhs => cl = -inf, cu = rhs
      - '==' constraints: body == rhs => cl = rhs, cu = rhs
      - '>=' constraints: body >= rhs => cl = rhs, cu = inf
    """
    cl_list = []
    cu_list = []
    for i in range(model._nl_n_constraints):
        sense = model._nl_constraint_senses[i]
        rhs = model._nl_constraint_rhs[i]
        if sense == "<=":
            cl_list.append(-1e20)
            cu_list.append(rhs)
        elif sense == "==":
            cl_list.append(rhs)
            cu_list.append(rhs)
        elif sense == ">=":
            cl_list.append(rhs)
            cu_list.append(1e20)
        else:
            raise ValueError(f"Unknown constraint sense: {sense}")
    return cl_list, cu_list


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
    lb_clipped = np.clip(node_lb, -100.0, 100.0)
    ub_clipped = np.clip(node_ub, -100.0, 100.0)
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
    except Exception:
        # Fall back: return infeasible sentinel
        return NLPResult(
            status=SolveStatus.ERROR,
            x=np.asarray(starting_points[0]),
            objective=1e30,
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
            objective=1e30,
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


def _is_pure_continuous(model: Model) -> bool:
    """Check if model has no integer/binary variables."""
    return all(v.var_type == VarType.CONTINUOUS for v in model._variables)


def solve_model(
    model: Model,
    time_limit: float = 3600.0,
    gap_tolerance: float = 1e-4,
    threads: int = 1,
    gpu: bool = True,
    deterministic: bool = True,
    batch_size: int = 16,
    strategy: str = "best_first",
    max_nodes: int = 100_000,
    ipopt_options: Optional[dict] = None,
    nlp_solver: str = "ipm",
    cutting_planes: bool = False,
    partitions: int = 0,
    branching_policy: str = "fractional",
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
    gpu : bool, default True
        Enable GPU for JAX (currently CPU-only on macOS).
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
        ``"ipopt"`` (cyipopt), or ``"ipm"`` (pure-JAX IPM).
    cutting_planes : bool, default False
        Enable outer-approximation cut generation after NLP relaxation solves.
    partitions : int, default 0
        Number of piecewise McCormick partitions (0 = standard convex
        relaxation, k > 0 = k partitions for tighter relaxations).
    branching_policy : str, default "fractional"
        Variable selection: ``"fractional"`` (most-fractional, default)
        or ``"gnn"`` (GNN scoring hook; Rust handles actual branching).

    Returns
    -------
    SolveResult
        Contains solution values, objective, gap, node count, and
        per-layer profiling times (Rust, JAX, Python).
    """
    t_start = time.perf_counter()
    rust_time = 0.0
    jax_time = 0.0

    if nlp_solver == "ripopt":
        print("Using ripopt (Rust interior point method)")
    elif nlp_solver == "ipm":
        print("Using discopt IPM (pure-JAX interior point method)")
    else:
        print("Using Ipopt (via cyipopt)")

    # --- Problem classification: dispatch LP/QP to specialized solvers ---
    # Skip LP/QP classification for .nl models: extract_lp_data/extract_qp_data
    # rely on the DAG expression structure which .nl models don't have.
    if not _has_nl_repr(model):
        try:
            from discopt._jax.problem_classifier import ProblemClass, classify_problem

            problem_class = classify_problem(model)
        except Exception:
            problem_class = None

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
    if _has_nl_repr(model):
        cl_list, cu_list = _infer_nl_constraint_bounds(model)
    else:
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
        if _has_nl_repr(model):
            _constraint_senses = list(model._nl_constraint_senses)
        else:
            _constraint_senses = [c.sense for c in model._constraints if isinstance(c, Constraint)]

    # --- Default Ipopt options ---
    opts = dict(ipopt_options) if ipopt_options else {}
    opts.setdefault("print_level", 0)
    opts.setdefault("max_iter", 3000)
    opts.setdefault("tol", 1e-7)

    # --- Augmented constraint function with cuts (updated each iteration) ---
    _augmented_evaluator = None

    # --- B&B loop ---
    iteration = 0
    while True:
        elapsed = time.perf_counter() - t_start
        if elapsed >= time_limit:
            break

        # Export batch from Rust tree
        t_rust_start = time.perf_counter()
        batch_lb, batch_ub, batch_ids = tree.export_batch(batch_size)
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
            )
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
                    lb_clipped = np.clip(node_lb, -100.0, 100.0)
                    ub_clipped = np.clip(node_ub, -100.0, 100.0)
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
                    result_lbs[i] = nlp_result.objective
                    result_sols[i] = nlp_result.x
                    result_feas[i] = False
                else:
                    result_lbs[i] = 1e30
                    lb_clipped = np.clip(node_lb, -100.0, 100.0)
                    ub_clipped = np.clip(node_ub, -100.0, 100.0)
                    result_sols[i] = 0.5 * (lb_clipped + ub_clipped)
                    result_feas[i] = False
        jax_time += time.perf_counter() - t_jax_start

        # --- Optional GNN branching scoring (advisory) ---
        # GNN computes variable scores at each node. Currently advisory
        # only: actual branching is done by Rust's most-fractional policy
        # inside process_evaluated(). Full GNN-driven branching requires
        # extending the Rust TreeManager with a set_branch_variable() method.
        if branching_policy == "gnn" and not _has_nl_repr(model):
            from discopt._jax.gnn_policy import select_branch_variable_gnn
            from discopt._jax.problem_graph import build_graph

            for i in range(n_batch):
                if result_lbs[i] < 1e20:
                    node_lb_i = np.array(batch_lb[i])
                    node_ub_i = np.array(batch_ub[i])
                    graph = build_graph(model, result_sols[i], node_lb_i, node_ub_i)
                    select_branch_variable_gnn(graph, params=None)

        # --- Optional cut generation (OA + RLT + lift-and-project) ---
        if cutting_planes and _generate_cuts is not None and _cut_pool is not None:
            for i in range(n_batch):
                if result_lbs[i] < 1e20:  # skip infeasible nodes
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
                    )
                    _cut_pool.add_many(new_cuts)
                    # Age and purge stale cuts
                    _cut_pool.age_cuts(result_sols[i])
            _cut_pool.purge_inactive(max_age=15)

        # Import results back to Rust tree
        t_rust_start = time.perf_counter()
        tree.import_results(result_ids, result_lbs, result_sols, result_feas)
        tree.process_evaluated()
        rust_time += time.perf_counter() - t_rust_start

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
        # (1e30 is the sentinel value set in _solve_node_nlp for infeasible nodes)
        if obj_val >= 1e20:
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
    lb_clipped = np.clip(lb, -100.0, 100.0)
    ub_clipped = np.clip(ub, -100.0, 100.0)
    x0 = 0.5 * (lb_clipped + ub_clipped)

    opts = dict(ipopt_options) if ipopt_options else {}
    opts.setdefault("print_level", 0)

    # For .nl models, pass explicit constraint bounds
    constraint_bounds = None
    if _has_nl_repr(model):
        cl_list, cu_list = _infer_nl_constraint_bounds(model)
        if cl_list:
            constraint_bounds = list(zip(cl_list, cu_list))

    t_jax_start = time.perf_counter()
    if nlp_solver == "ripopt":
        from discopt.solvers.nlp_ripopt import solve_nlp as solve_nlp_ripopt

        nlp_result = solve_nlp_ripopt(
            evaluator, x0, constraint_bounds=constraint_bounds, options=opts
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
        # for non-JAX evaluators (e.g. NLPEvaluatorFromNl from .nl files).
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
    except Exception:
        return NLPResult(status=SolveStatus.ERROR, x=x0, objective=1e30)


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
    except Exception:
        return NLPResult(status=SolveStatus.ERROR, x=x0, objective=1e30)

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
    lb_clipped = jnp.clip(xl_batch, -100.0, 100.0)
    ub_clipped = jnp.clip(xu_batch, -100.0, 100.0)
    x0_batch = 0.5 * (lb_clipped + ub_clipped)

    ipm_opts = IPMOptions(max_iter=int(options.get("max_iter", 200)))

    try:
        state = solve_nlp_batch(
            obj_fn, con_fn, x0_batch, xl_batch, xu_batch, g_l_jax, g_u_jax, ipm_opts
        )
    except Exception:
        # Fallback: mark all as infeasible
        result_ids = np.array(batch_ids, dtype=np.int64)
        x0_np = np.asarray(x0_batch)
        result_lbs = np.full(n_batch, 1e30, dtype=np.float64)
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
        1e30,
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
    except Exception:
        return NLPResult(
            status=SolveStatus.ERROR,
            x=x0,
            objective=1e30,
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
        batch_lb, batch_ub, batch_ids = tree.export_batch(batch_size)
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
                result_lbs = np.where(ok, obj_vals + lp_data.obj_const, 1e30)
                result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
                for i in range(n_batch):
                    if ok[i]:
                        result_sols[i] = x_vals[i, :n_vars]
                    else:
                        lb_c = np.clip(np.array(batch_lb[i]), -100, 100)
                        ub_c = np.clip(np.array(batch_ub[i]), -100, 100)
                        result_sols[i] = 0.5 * (lb_c + ub_c)
            except Exception:
                result_lbs = np.full(n_batch, 1e30, dtype=np.float64)
                result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
                for i in range(n_batch):
                    lb_c = np.clip(np.array(batch_lb[i]), -100, 100)
                    ub_c = np.clip(np.array(batch_ub[i]), -100, 100)
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
                        result_lbs[i] = 1e30
                        lb_c = np.clip(node_lb, -100, 100)
                        ub_c = np.clip(node_ub, -100, 100)
                        result_sols[i] = 0.5 * (lb_c + ub_c)
                except Exception:
                    result_lbs[i] = 1e30
                    lb_c = np.clip(node_lb, -100, 100)
                    ub_c = np.clip(node_ub, -100, 100)
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
        if obj_val >= 1e20:
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
        batch_lb, batch_ub, batch_ids = tree.export_batch(batch_size)
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
                result_lbs = np.where(ok, obj_vals + qp_data.obj_const, 1e30)
                result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
                for i in range(n_batch):
                    if ok[i]:
                        result_sols[i] = x_vals[i, :n_vars]
                    else:
                        lb_c = np.clip(np.array(batch_lb[i]), -100, 100)
                        ub_c = np.clip(np.array(batch_ub[i]), -100, 100)
                        result_sols[i] = 0.5 * (lb_c + ub_c)
            except Exception:
                result_lbs = np.full(n_batch, 1e30, dtype=np.float64)
                result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
                for i in range(n_batch):
                    lb_c = np.clip(np.array(batch_lb[i]), -100, 100)
                    ub_c = np.clip(np.array(batch_ub[i]), -100, 100)
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
                        result_lbs[i] = 1e30
                        lb_c = np.clip(node_lb, -100, 100)
                        ub_c = np.clip(node_ub, -100, 100)
                        result_sols[i] = 0.5 * (lb_c + ub_c)
                except Exception:
                    result_lbs[i] = 1e30
                    lb_c = np.clip(node_lb, -100, 100)
                    ub_c = np.clip(node_ub, -100, 100)
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
        if obj_val >= 1e20:
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
