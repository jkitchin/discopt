"""
McCormick relaxation NLP solver for computing valid lower bounds in B&B.

Builds a convex NLP from McCormick relaxations of the objective and constraints,
then solves it with POUNCE (the pure-Rust Ipopt port). The optimal value of the
convex relaxation is a valid lower bound on the original nonconvex problem over the
node's domain. (The JAX IPM previously used here is retired; the relaxation
*functions* remain JAX-built.)

Mode:
  - **nlp**: Solve a convex NLP minimizing the McCormick underestimator subject
    to McCormick-relaxed constraints. Its optimum is a valid lower bound.

The former **midpoint** mode — evaluate the underestimator at the box midpoint
and return that value — was removed (correctness issue C-18): ``u(midpoint)`` is
not a valid lower bound on ``min_box u`` (e.g. for ``x**2`` on ``[1, 3]`` it
returns 3.0 while the true minimum is 1.0), so it could certify a wrong optimum.
The only sound cheap bound is the minimum of the (convex) underestimator over the
box, which is precisely what the ``nlp`` mode computes.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

from discopt._optional_jax import jax_available, missing_jax_error

if not jax_available():  # #1873: JAX is the optional discopt[jax] extra
    raise missing_jax_error(__name__)

import jax.numpy as jnp
import numpy as np

# Per-(relaxation, options) jit caches. Keyed by Python id() of the
# relaxation functions plus negate/max_iter so repeat B&B nodes hit the
# XLA cache instead of recompiling. Without these caches each call built
# fresh closures over (lb, ub), forcing JAX to retrace+recompile per node
# (the dominant cost on small instances).
_pounce_evaluator_cache: dict = {}


def _deadline_expired(deadline: float | None) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


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
    from discopt._relax.mccormick_evaluator import McCormickRelaxationEvaluator

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
    from discopt.solvers import SolveStatus
    from discopt.solvers.nlp_pounce import POUNCE_AVAILABLE
    from discopt.solvers.nlp_pounce import solve_nlp as solve_nlp_pounce

    if not POUNCE_AVAILABLE:
        return float("-inf")

    lb = np.asarray(node_lb, dtype=np.float64)
    ub = np.asarray(node_ub, dtype=np.float64)
    n_vars = int(lb.size)

    ev = _get_or_build_pounce_evaluator(obj_relax_fn, con_fns_tuple, senses_tuple, n_vars, negate)
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
            return float("-inf")
        opts["max_wall_time"] = float(remaining)

    try:
        result = solve_nlp_pounce(ev, x0, constraint_bounds=constraint_bounds, options=opts)
    except Exception as exc:
        # #1520: kept as a sound fallback. POUNCE is native code reached through
        # PyO3; if it raises, the node gets ``-inf`` (no bound from this path),
        # which is always valid. Reported, not silent.
        from discopt._relax._fallback import warn_fallback_once

        warn_fallback_once(
            "McCormick relaxation NLP (POUNCE) solve", exc, "no bound from this path (-inf)"
        )
        return float("-inf")

    # #1520: only a CONVERGED solve is a lower bound. Convexity of the McCormick
    # underestimator makes its *minimum* a valid bound, but an interior-point
    # iterate stopped at ``max_iter`` (``ITERATION_LIMIT``) or on a stalled
    # search direction (mapped to ``UNBOUNDED``) is not a minimum: its objective
    # can sit anywhere above it (and, primal-infeasible, anywhere at all). This
    # used to accept both, handing the tree a node bound nothing proved. The
    # node now keeps its inherited bound (``-inf`` here), which is always valid.
    if result.status != SolveStatus.OPTIMAL:
        return float("-inf")
    if result.objective is None:
        return float("-inf")
    val = float(result.objective)
    if not np.isfinite(val):
        return float("-inf")
    return val


def solve_mccormick_relaxation_nlp(
    obj_relax_fn: Callable,
    con_relax_fns: Optional[list[Callable]],
    con_senses: Optional[list[str]],
    node_lb: jnp.ndarray,
    node_ub: jnp.ndarray,
    negate: bool = False,
    max_iter: int = 50,
    deadline: float | None = None,
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
        return float("-inf")

    lb = jnp.asarray(node_lb, dtype=jnp.float64)
    ub = jnp.asarray(node_ub, dtype=jnp.float64)

    # Cheap early bailout: if the obj relaxation is degenerate at the
    # midpoint, skip the solve. One sync per node (vs O(m) in the old
    # per-constraint filter) — still cheaper than entering the IPM only
    # to have it blow up on inf/NaN.
    mid = 0.5 * (lb + ub)
    # #1520: no except. A compiled relaxation declines by value (inf/NaN, handled
    # here); it has no documented exception, so a raise is a defect -- not a
    # silent ``-inf`` that reads as "the NLP bound never helps".
    cv_test, cc_test = obj_relax_fn(mid, mid, lb, ub)
    cv_t = float(cv_test)
    cc_t = float(cc_test)
    if not (np.isfinite(cv_t) and np.isfinite(cc_t)):
        return float("-inf")

    if _deadline_expired(deadline):
        return float("-inf")

    con_fns_tuple = tuple(con_relax_fns) if con_relax_fns else None
    senses_tuple = tuple(con_senses) if con_senses else None

    # The convex relaxation NLP is solved by POUNCE (pure-Rust Ipopt port; the
    # same engine the original-NLP path uses). The JAX IPM previously used here
    # has been removed — POUNCE has no JAX trace/compile floor and keeps the
    # spatial-B&B relaxation path off any JAX interior-point core.
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


def solve_mccormick_batch(
    obj_relax_fn: Callable,
    con_relax_fns: Optional[list[Callable]],
    con_senses: Optional[list[str]],
    lb_batch: jnp.ndarray,
    ub_batch: jnp.ndarray,
    negate: bool = False,
    max_iter: int = 50,
    deadline: float | None = None,
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
        )
        result_list.append(val)

    return jnp.array(result_list, dtype=jnp.float64)
