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

import time
from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np


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
    import jax

    vmapped_fn = jax.jit(jax.vmap(obj_relax_fn))
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
    from discopt._jax.ipm import IPMOptions, ipm_solve

    if _deadline_expired(deadline):
        return -np.inf

    lb = jnp.asarray(node_lb, dtype=jnp.float64)
    ub = jnp.asarray(node_ub, dtype=jnp.float64)

    # Check objective relaxation is well-behaved
    mid = 0.5 * (lb + ub)
    try:
        cv_test, cc_test = obj_relax_fn(mid, mid, lb, ub)
        if not np.isfinite(float(cv_test)) or not np.isfinite(float(cc_test)):
            return -np.inf
    except Exception:
        return -np.inf

    # Build convex objective: minimize cv(x) for minimization
    def obj_fn(x):
        cv, cc = obj_relax_fn(x, x, lb, ub)
        if negate:
            return -cc
        return cv

    # Build constraint function from relaxations
    g_l = None
    g_u = None
    con_fn = None

    if con_relax_fns and con_senses:
        if _deadline_expired(deadline):
            return -np.inf

        # Filter out constraints that produce inf/NaN at wide bounds
        good_fns, good_senses = _filter_well_behaved_constraints(con_relax_fns, con_senses, lb, ub)

        if good_fns:
            g_l_list = []
            g_u_list = []

            for sense in good_senses:
                if sense == "<=":
                    g_l_list.append(-1e20)
                    g_u_list.append(0.0)
                elif sense == ">=":
                    g_l_list.append(-1e20)
                    g_u_list.append(0.0)
                elif sense == "==":
                    g_l_list.append(-1e20)
                    g_u_list.append(0.0)

            g_l = jnp.array(g_l_list, dtype=jnp.float64)
            g_u = jnp.array(g_u_list, dtype=jnp.float64)

            def con_fn(x, _lb=lb, _ub=ub, _fns=good_fns, _senses=good_senses):
                vals = []
                for fn, sense in zip(_fns, _senses):
                    cv, cc = fn(x, x, _lb, _ub)
                    if sense == "<=":
                        vals.append(cv)
                    elif sense == ">=":
                        vals.append(-cc)
                    elif sense == "==":
                        vals.append(cv)
                return jnp.stack(vals)

    opts = IPMOptions(max_iter=max_iter)
    x0 = jnp.clip(0.5 * (lb + ub), lb, ub)

    if _deadline_expired(deadline):
        return -np.inf

    try:
        state = ipm_solve(obj_fn, con_fn, x0, lb, ub, g_l, g_u, opts)
        conv = int(state.converged)
        if conv in (1, 2, 3):
            obj_val = float(state.obj)
            if np.isfinite(obj_val):
                return obj_val
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
