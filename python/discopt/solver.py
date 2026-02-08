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
    if cutting_planes:
        from discopt._jax.cutting_planes import (
            detect_bilinear_terms,
            generate_cuts_at_node,
        )

        _generate_cuts = generate_cuts_at_node
        _bilinear_terms = detect_bilinear_terms(model)
        if _has_nl_repr(model):
            _constraint_senses = list(model._nl_constraint_senses)
        else:
            _constraint_senses = [c.sense for c in model._constraints if isinstance(c, Constraint)]

    # --- Default Ipopt options ---
    opts = dict(ipopt_options) if ipopt_options else {}
    opts.setdefault("print_level", 0)
    opts.setdefault("max_iter", 3000)
    opts.setdefault("tol", 1e-7)

    # --- Accumulated OA cuts across iterations ---
    oa_cuts = []

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
        if nlp_solver == "ipm" and n_batch > 1:
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
            )
        else:
            result_ids = np.empty(n_batch, dtype=np.int64)
            result_lbs = np.empty(n_batch, dtype=np.float64)
            result_sols = np.empty((n_batch, n_vars), dtype=np.float64)
            result_feas = np.empty(n_batch, dtype=bool)

            for i in range(n_batch):
                node_id = int(batch_ids[i])
                node_lb = np.array(batch_lb[i])
                node_ub = np.array(batch_ub[i])

                lb_clipped = np.clip(node_lb, -100.0, 100.0)
                ub_clipped = np.clip(node_ub, -100.0, 100.0)
                x0 = 0.5 * (lb_clipped + ub_clipped)

                nlp_result = _solve_node_nlp(
                    evaluator,
                    x0,
                    node_lb,
                    node_ub,
                    constraint_bounds,
                    opts,
                    nlp_solver=nlp_solver,
                )

                result_ids[i] = node_id

                if nlp_result.status == SolveStatus.INFEASIBLE:
                    result_lbs[i] = 1e30
                    result_sols[i] = x0
                    result_feas[i] = False
                elif nlp_result.status in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT):
                    result_lbs[i] = nlp_result.objective
                    result_sols[i] = nlp_result.x
                    result_feas[i] = False
                else:
                    result_lbs[i] = 1e30
                    result_sols[i] = x0
                    result_feas[i] = False
        jax_time += time.perf_counter() - t_jax_start

        # --- Optional GNN branching suggestion (future hook) ---
        if branching_policy == "gnn" and not _has_nl_repr(model):
            from discopt._jax.gnn_policy import select_branch_variable_gnn
            from discopt._jax.problem_graph import build_graph

            for i in range(n_batch):
                if result_lbs[i] < 1e20:
                    node_lb_i = np.array(batch_lb[i])
                    node_ub_i = np.array(batch_ub[i])
                    graph = build_graph(model, result_sols[i], node_lb_i, node_ub_i)
                    select_branch_variable_gnn(graph, params=None)

        # --- Optional cut generation (OA for violated constraints + RLT) ---
        if cutting_planes and _generate_cuts is not None:
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
                    oa_cuts.extend(new_cuts)

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
    elif nlp_solver == "ipm":
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
