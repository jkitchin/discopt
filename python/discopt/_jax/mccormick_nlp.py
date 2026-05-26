"""
McCormick relaxation NLP solver for computing valid lower bounds in B&B.

Builds a convex NLP from McCormick relaxations of the objective and constraints,
then solves it with the pure-JAX IPM. The optimal value of the convex relaxation
is a valid lower bound on the original nonconvex problem over the node's domain.

Two modes:
  - **midpoint**: Evaluate the McCormick convex underestimator at the midpoint
    of the node bounds. Nearly free but provides a weak bound.
  - **nlp**: Solve a convex NLP minimizing the McCormick underestimator subject
    to McCormick-relaxed constraints. Tighter but costs one IPM solve per node.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np


# Default n_vars threshold below which the numpy/scipy backend is used
# when caller supplies a numpy-compiled relaxation. Overridable via
# DISCOPT_MCCORMICK_NUMPY_THRESHOLD.
#
# Default is 0 (disabled). The numpy backend bypasses JAX trace/compile
# entirely via scipy.optimize.minimize(SLSQP); on small instances it is
# competitive on short solves and more robust at the root node (returns
# finite bounds where the JAX IPM may return -inf), but produces looser
# relaxation bounds than the JAX IPM, so on long solves it explores more
# B&B nodes. Set the env var to N to opt in for problems with <= N vars.
_NUMPY_THRESHOLD_DEFAULT = 0


def _numpy_threshold() -> int:
    raw = os.environ.get("DISCOPT_MCCORMICK_NUMPY_THRESHOLD")
    if raw is None:
        return _NUMPY_THRESHOLD_DEFAULT
    try:
        return int(raw)
    except ValueError:
        return _NUMPY_THRESHOLD_DEFAULT


def _pounce_backend_enabled() -> bool:
    """True when DISCOPT_MCCORMICK_BACKEND=pounce (or the deprecated "ripopt").

    Opt-in: routes the convex relaxation NLP through POUNCE (pure-Rust
    Ipopt port; https://github.com/jkitchin/pounce) instead of the
    JAX-jit'd ``_jax/ipm.py``. The McCormickRelaxationEvaluator builds
    small jit'd objective/gradient/Jacobian/Hessian modules and POUNCE
    drives the IPM. Avoids the trace/compile floor of the inline JAX IPM
    at the cost of one Python<->Rust boundary per IPM iteration.
    """
    val = os.environ.get("DISCOPT_MCCORMICK_BACKEND", "ipm").strip().lower()
    return val in ("pounce", "ripopt")


# Per-(relaxation, options) jit caches. Keyed by Python id() of the
# relaxation functions plus negate/max_iter so repeat B&B nodes hit the
# XLA cache instead of recompiling. Without these caches each call built
# fresh closures over (lb, ub), forcing JAX to retrace+recompile per node
# (the dominant cost on small instances).
_midpoint_batch_cache: dict = {}
_relax_solver_cache: dict = {}
_pounce_evaluator_cache: dict = {}


def _deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def evaluate_midpoint_bound(
    obj_relax_fn: Callable,
    node_lb: jnp.ndarray,
    node_ub: jnp.ndarray,
    negate: bool = False,
) -> float:
    """Evaluate McCormick objective relaxation at the node midpoint.

    Args:
        obj_relax_fn: Compiled relaxation fn(x_cv, x_cc, lb, ub) -> (cv, cc).
        node_lb: Lower bounds for this B&B node, shape (n,).
        node_ub: Upper bounds for this B&B node, shape (n,).
        negate: If True, the original problem is maximization.
            Return -cc as the lower bound on the negated objective.

    Returns:
        A valid lower bound (float), or -inf on failure.
    """
    try:
        mid = 0.5 * (node_lb + node_ub)
        cv, cc = obj_relax_fn(mid, mid, node_lb, node_ub)
        if negate:
            return -float(cc)
        return float(cv)
    except Exception:
        return -np.inf


def evaluate_midpoint_bound_batch(
    obj_relax_fn: Callable,
    lb_batch: jnp.ndarray,
    ub_batch: jnp.ndarray,
    negate: bool = False,
) -> jnp.ndarray:
    """Evaluate McCormick midpoint bounds for a batch of nodes.

    Args:
        obj_relax_fn: Compiled relaxation fn(x_cv, x_cc, lb, ub) -> (cv, cc).
        lb_batch: Lower bounds, shape (N, n_vars).
        ub_batch: Upper bounds, shape (N, n_vars).
        negate: If True, maximization problem.

    Returns:
        Array of lower bounds, shape (N,).
    """
    key = id(obj_relax_fn)
    vmapped_fn = _midpoint_batch_cache.get(key)
    if vmapped_fn is None:
        vmapped_fn = jax.jit(jax.vmap(obj_relax_fn))
        _midpoint_batch_cache[key] = vmapped_fn
    mid = 0.5 * (lb_batch + ub_batch)
    cv_batch, cc_batch = vmapped_fn(mid, mid, lb_batch, ub_batch)
    if negate:
        return jnp.asarray(-cc_batch)
    return jnp.asarray(cv_batch)


def _filter_well_behaved_constraints(
    con_relax_fns: list[Callable],
    con_senses: list[str],
    lb: jnp.ndarray,
    ub: jnp.ndarray,
) -> tuple[list[Callable], list[str]]:
    """Filter out constraints whose McCormick relaxation produces inf/NaN.

    Constraints involving singularities (e.g. 1/(x^3 * sin(x))) can produce
    inf/NaN in their McCormick relaxation at wide bounds. Dropping these
    constraints weakens the relaxation but keeps the NLP well-conditioned.
    The resulting lower bound is still valid (just weaker).
    """
    good_fns = []
    good_senses = []

    # Test at several points to see if the relaxation is well-behaved
    test_points = [
        0.5 * (lb + ub),
        lb + 0.25 * (ub - lb),
        lb + 0.75 * (ub - lb),
    ]

    for fn, sense in zip(con_relax_fns, con_senses):
        is_ok = False
        for pt in test_points:
            try:
                cv, cc = fn(pt, pt, lb, ub)
                cv_val = float(cv)
                cc_val = float(cc)
                if np.isfinite(cv_val) and np.isfinite(cc_val):
                    if abs(cv_val) < 1e12 and abs(cc_val) < 1e12:
                        is_ok = True
                        break
            except Exception:
                continue
        if is_ok:
            good_fns.append(fn)
            good_senses.append(sense)

    return good_fns, good_senses


def _get_or_build_pounce_evaluator(
    obj_relax_fn: Callable,
    con_fns_tuple: Optional[tuple[Callable, ...]],
    senses_tuple: Optional[tuple[str, ...]],
    n_vars: int,
    negate: bool,
):
    """Cache the McCormickRelaxationEvaluator by relaxation identity.

    A single evaluator per problem signature is reused across B&B nodes,
    so the underlying jit'd grad/jacobian/hessian modules are compiled
    once and warm for every subsequent solve.
    """
    from discopt._jax.mccormick_evaluator import McCormickRelaxationEvaluator

    con_key = tuple(id(f) for f in con_fns_tuple) if con_fns_tuple else ()
    sense_key = senses_tuple or ()
    key = (id(obj_relax_fn), con_key, sense_key, int(n_vars), bool(negate))
    cached = _pounce_evaluator_cache.get(key)
    if cached is not None:
        return cached
    ev = McCormickRelaxationEvaluator(
        obj_relax_fn,
        list(con_fns_tuple) if con_fns_tuple else None,
        list(senses_tuple) if senses_tuple else None,
        n_vars=n_vars,
        negate=negate,
    )
    _pounce_evaluator_cache[key] = ev
    return ev


def _solve_relaxation_with_pounce(
    obj_relax_fn: Callable,
    con_fns_tuple: Optional[tuple[Callable, ...]],
    senses_tuple: Optional[tuple[str, ...]],
    node_lb: np.ndarray,
    node_ub: np.ndarray,
    negate: bool,
    max_iter: int,
    deadline: float | None,
) -> float:
    """Solve the McCormick relaxation NLP via POUNCE."""
    from discopt.solvers.nlp_pounce import POUNCE_AVAILABLE, solve_nlp as solve_nlp_pounce
    from discopt.solvers import SolveStatus

    if not POUNCE_AVAILABLE:
        return -np.inf

    lb = np.asarray(node_lb, dtype=np.float64)
    ub = np.asarray(node_ub, dtype=np.float64)
    n_vars = int(lb.size)

    ev = _get_or_build_pounce_evaluator(
        obj_relax_fn, con_fns_tuple, senses_tuple, n_vars, negate
    )
    ev.set_bounds(lb, ub)

    m = ev.n_constraints
    if m > 0:
        constraint_bounds: Optional[list[tuple[float, float]]] = [(-np.inf, 0.0)] * m
    else:
        constraint_bounds = None

    x0 = np.clip(0.5 * (lb + ub), lb, ub).astype(np.float64)

    opts: dict = {"max_iter": int(max_iter), "print_level": 0}
    if deadline is not None:
        remaining = deadline - time.perf_counter()
        if remaining <= 0.0:
            return -np.inf
        opts["max_wall_time"] = float(remaining)

    try:
        result = solve_nlp_pounce(ev, x0, constraint_bounds=constraint_bounds, options=opts)
    except Exception:
        return -np.inf

    # Mirror the prior ripopt acceptance set: optimal, acceptable, stalled
    # (Search_Direction_Becomes_Too_Small → UNBOUNDED in the discopt enum),
    # and max-iterations are all valid B&B lower bounds because the
    # McCormick underestimator is convex.
    if result.status not in (
        SolveStatus.OPTIMAL,
        SolveStatus.ITERATION_LIMIT,
        SolveStatus.UNBOUNDED,
    ):
        return -np.inf
    val = float(result.objective)
    if not np.isfinite(val):
        return -np.inf
    return val


def _get_or_build_relax_solver(
    obj_relax_fn: Callable,
    con_relax_fns: Optional[tuple[Callable, ...]],
    con_senses: Optional[tuple[str, ...]],
    negate: bool,
    max_iter: int,
) -> Callable:
    """Return a jit-compiled solver for a given relaxation signature.

    Cached by function identity so that B&B nodes (which share the same
    relaxation functions and only differ in bounds) hit the XLA cache.
    Bounds are passed as traced arguments to the returned solver, so a
    single compiled module serves every node on a given problem.
    """
    from discopt._jax.ipm import IPMOptions, ipm_solve

    con_key = tuple(id(f) for f in con_relax_fns) if con_relax_fns else ()
    sense_key = tuple(con_senses) if con_senses else ()
    key = (id(obj_relax_fn), con_key, sense_key, bool(negate), int(max_iter))

    cached = _relax_solver_cache.get(key)
    if cached is not None:
        return cached

    has_cons = bool(con_relax_fns) and bool(con_senses)
    fns_local = tuple(con_relax_fns) if has_cons else ()
    senses_local = tuple(con_senses) if has_cons else ()
    n_cons = len(fns_local)
    opts = IPMOptions(max_iter=max_iter)

    @jax.jit
    def _solve(node_lb, node_ub):
        def obj_fn(x):
            cv, cc = obj_relax_fn(x, x, node_lb, node_ub)
            return -cc if negate else cv

        if has_cons:
            def con_fn(x):
                vals = []
                for fn, sense in zip(fns_local, senses_local):
                    cv, cc = fn(x, x, node_lb, node_ub)
                    if sense == ">=":
                        vals.append(-cc)
                    else:  # "<=" and "==" both use cv as upper-bounded by 0
                        vals.append(cv)
                return jnp.stack(vals)

            g_l = jnp.full(n_cons, -1e20, dtype=jnp.float64)
            g_u = jnp.zeros(n_cons, dtype=jnp.float64)
        else:
            con_fn = None
            g_l = None
            g_u = None

        x0 = jnp.clip(0.5 * (node_lb + node_ub), node_lb, node_ub)
        state = ipm_solve(
            obj_fn,
            con_fn,
            x0,
            node_lb,
            node_ub,
            g_l,
            g_u,
            opts,
            check_deadline=False,
        )
        return state.obj, state.converged

    _relax_solver_cache[key] = _solve
    return _solve


def solve_mccormick_relaxation_nlp(
    obj_relax_fn: Callable,
    con_relax_fns: Optional[list[Callable]],
    con_senses: Optional[list[str]],
    node_lb: jnp.ndarray,
    node_ub: jnp.ndarray,
    negate: bool = False,
    max_iter: int = 50,
    deadline: float | None = None,
    obj_relax_fn_numpy: Optional[Callable] = None,
    con_relax_fns_numpy: Optional[list[Callable]] = None,
) -> float:
    """Solve a convex NLP over McCormick relaxations for a tight lower bound.

    Builds a convex objective from the McCormick underestimator (cv) and
    convex constraint relaxations, then solves with the IPM.

    Args:
        obj_relax_fn: Compiled objective relaxation fn.
        con_relax_fns: List of compiled constraint relaxation fns, or None.
        con_senses: List of constraint senses ("<=", ">=", "=="), or None.
        node_lb: Variable lower bounds, shape (n,).
        node_ub: Variable upper bounds, shape (n,).
        negate: True if the original problem is maximization.
        max_iter: Maximum IPM iterations.
        deadline: Absolute ``time.perf_counter()`` deadline. If expired before
            a new relaxation solve starts, return ``-inf`` without solving.

    Returns:
        Valid lower bound (float), or -inf on failure.
    """
    if _deadline_expired(deadline):
        return -np.inf

    # Dispatch to numpy/scipy backend when caller supplied numpy fns and
    # problem is small enough that JAX trace/compile would dominate.
    if obj_relax_fn_numpy is not None:
        n_vars = int(np.asarray(node_lb).size)
        if n_vars <= _numpy_threshold():
            from discopt._numpy.nlp_solver import solve_mccormick_relaxation_nlp_numpy

            return solve_mccormick_relaxation_nlp_numpy(
                obj_relax_fn_numpy,
                con_relax_fns_numpy,
                con_senses,
                np.asarray(node_lb, dtype=np.float64),
                np.asarray(node_ub, dtype=np.float64),
                negate=negate,
                max_iter=max(max_iter, 100),
                deadline=deadline,
            )

    lb = jnp.asarray(node_lb, dtype=jnp.float64)
    ub = jnp.asarray(node_ub, dtype=jnp.float64)

    # Cheap early bailout: if the obj relaxation is degenerate at the
    # midpoint, skip the solve. One sync per node (vs O(m) in the old
    # per-constraint filter) — still cheaper than entering the IPM only
    # to have it blow up on inf/NaN.
    mid = 0.5 * (lb + ub)
    try:
        cv_test, cc_test = obj_relax_fn(mid, mid, lb, ub)
        cv_t = float(cv_test)
        cc_t = float(cc_test)
        if not (np.isfinite(cv_t) and np.isfinite(cc_t)):
            return -np.inf
    except Exception:
        return -np.inf

    if _deadline_expired(deadline):
        return -np.inf

    con_fns_tuple = tuple(con_relax_fns) if con_relax_fns else None
    senses_tuple = tuple(con_senses) if con_senses else None

    # Optional: route the convex relaxation NLP through POUNCE (pure-Rust
    # Ipopt port; same one the original-NLP path uses) instead of the
    # inline jit'd JAX IPM. Avoids the JAX IPM trace/compile floor.
    if _pounce_backend_enabled():
        return _solve_relaxation_with_pounce(
            obj_relax_fn,
            con_fns_tuple,
            senses_tuple,
            np.asarray(node_lb, dtype=np.float64),
            np.asarray(node_ub, dtype=np.float64),
            negate=negate,
            max_iter=max(max_iter, 100),
            deadline=deadline,
        )

    try:
        solver = _get_or_build_relax_solver(
            obj_relax_fn, con_fns_tuple, senses_tuple, negate, max_iter
        )
        obj_val, converged = solver(lb, ub)
        conv = int(converged)
        if conv in (1, 2, 3):
            v = float(obj_val)
            if np.isfinite(v):
                return v
    except Exception:
        pass

    return -np.inf


def solve_mccormick_batch(
    obj_relax_fn: Callable,
    con_relax_fns: Optional[list[Callable]],
    con_senses: Optional[list[str]],
    lb_batch: jnp.ndarray,
    ub_batch: jnp.ndarray,
    negate: bool = False,
    max_iter: int = 50,
    deadline: float | None = None,
    obj_relax_fn_numpy: Optional[Callable] = None,
    con_relax_fns_numpy: Optional[list[Callable]] = None,
) -> jnp.ndarray:
    """Solve McCormick relaxation NLPs for a batch of nodes via vmap.

    Args:
        obj_relax_fn: Compiled objective relaxation fn.
        con_relax_fns: List of compiled constraint relaxation fns, or None.
        con_senses: List of constraint senses, or None.
        lb_batch: Lower bounds, shape (N, n_vars).
        ub_batch: Upper bounds, shape (N, n_vars).
        negate: True for maximization.
        max_iter: Max IPM iterations per node.
        deadline: Absolute ``time.perf_counter()`` deadline. If it expires,
            remaining nodes receive ``-inf`` bounds without starting more
            relaxation solves.

    Returns:
        Array of lower bounds, shape (N,).
    """

    n_batch = lb_batch.shape[0]

    # Build parametric objective that takes bounds as arguments
    def obj_fn_param(x, lb_node, ub_node):
        cv, cc = obj_relax_fn(x, x, lb_node, ub_node)
        if negate:
            return -cc
        return cv

    # For vmap, we need obj_fn(x) with lb/ub captured. But solve_nlp_batch
    # expects obj_fn(x) with shared function. Since bounds differ per node,
    # we can't directly use solve_nlp_batch. Instead, solve serially
    # (still fast since each IPM is JIT'd).
    result_list = []

    for i in range(n_batch):
        if _deadline_expired(deadline):
            result_list.extend([-np.inf] * (n_batch - i))
            break

        lb_i = lb_batch[i]
        ub_i = ub_batch[i]
        val = solve_mccormick_relaxation_nlp(
            obj_relax_fn,
            con_relax_fns,
            con_senses,
            lb_i,
            ub_i,
            negate=negate,
            max_iter=max_iter,
            deadline=deadline,
            obj_relax_fn_numpy=obj_relax_fn_numpy,
            con_relax_fns_numpy=con_relax_fns_numpy,
        )
        result_list.append(val)

    return jnp.array(result_list, dtype=jnp.float64)
