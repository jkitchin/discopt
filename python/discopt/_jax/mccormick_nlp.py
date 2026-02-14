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

from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np


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


def solve_mccormick_relaxation_nlp(
    obj_relax_fn: Callable,
    con_relax_fns: Optional[list[Callable]],
    con_senses: Optional[list[str]],
    node_lb: jnp.ndarray,
    node_ub: jnp.ndarray,
    negate: bool = False,
    max_iter: int = 50,
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

    Returns:
        Valid lower bound (float), or -inf on failure.
    """
    from discopt._jax.ipm import IPMOptions, ipm_solve

    lb = jnp.asarray(node_lb, dtype=jnp.float64)
    ub = jnp.asarray(node_ub, dtype=jnp.float64)

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
        g_l_list = []
        g_u_list = []

        for sense in con_senses:
            if sense == "<=":
                # cv ≤ 0 is a valid relaxation of body ≤ 0
                g_l_list.append(-1e20)
                g_u_list.append(0.0)
            elif sense == ">=":
                # -cc ≤ 0 is a valid relaxation of body ≥ 0
                # (cc ≥ body, so -cc ≤ -body ≤ 0)
                g_l_list.append(-1e20)
                g_u_list.append(0.0)
            elif sense == "==":
                # For equality: cv ≤ 0 (one-sided relaxation)
                g_l_list.append(-1e20)
                g_u_list.append(0.0)

        g_l = jnp.array(g_l_list, dtype=jnp.float64)
        g_u = jnp.array(g_u_list, dtype=jnp.float64)

        def con_fn(x, _lb=lb, _ub=ub, _fns=con_relax_fns, _senses=con_senses):
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

    x0 = 0.5 * (lb + ub)
    x0 = jnp.clip(x0, lb, ub)

    opts = IPMOptions(max_iter=max_iter)

    try:
        state = ipm_solve(obj_fn, con_fn, x0, lb, ub, g_l, g_u, opts)
        conv = int(state.converged)
        if conv in (1, 2, 3):
            return float(state.obj)
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
        )
        result_list.append(val)

    return jnp.array(result_list, dtype=jnp.float64)
