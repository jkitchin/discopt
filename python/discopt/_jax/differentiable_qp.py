"""
Differentiable QP solver: compute gradients through QP solutions via custom_jvp.

Uses implicit differentiation of the KKT conditions (OptNet-style).
Key advantage over general NLP implicit diff: Q and A are known constants,
so no autodiff calls are needed for the sensitivity computation.

Reference: Amos & Kolter, "OptNet: Differentiable Optimization as a Layer in
Neural Networks" (ICML 2017).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from discopt._jax.qp_ipm import qp_ipm_solve

_EPS = 1e-20


@jax.custom_jvp
def qp_solve(Q, c, A, b, x_l, x_u):
    """Solve a QP and return (obj, x, y, z_l, z_u).

    This function is differentiable w.r.t. all inputs via implicit
    differentiation of the KKT conditions.

    Args:
        Q: (n, n) symmetric positive semi-definite objective matrix.
        c: (n,) linear objective coefficients.
        A: (m, n) equality constraint matrix.
        b: (m,) equality RHS.
        x_l: (n,) lower variable bounds.
        x_u: (n,) upper variable bounds.

    Returns:
        Tuple of (obj, x, y, z_l, z_u).
    """
    state = qp_ipm_solve(Q, c, A, b, x_l, x_u)
    return state.obj, state.x, state.y, state.z_l, state.z_u


@qp_solve.defjvp
def qp_solve_jvp(primals, tangents):
    """JVP rule for QP solve via implicit differentiation of KKT.

    At the optimal solution (x*, y*, z_l*, z_u*), the KKT conditions hold:
        F1: Qx + c - A'y - z_l + z_u = 0    (stationarity)
        F2: Ax - b = 0                        (primal feasibility)
        F3: S_l Z_l e = 0                     (complementarity, lower)
        F4: S_u Z_u e = 0                     (complementarity, upper)

    Differentiating F=0 w.r.t. parameters p = (Q, c, A, b, x_l, x_u):
        dF/d(x,y,z_l,z_u) · d(x,y,z_l,z_u)/dp = -dF/dp

    The Jacobian of F w.r.t. (x, y, z_l, z_u):
        [ Q    -A'   -I    I   ]
        [ A     0     0    0   ]
        [ Z_l   0    S_l   0   ]
        [ -Z_u  0     0   S_u  ]

    Key simplification: Q and A are known constants (no autodiff needed).
    """
    Q, c, A, b, x_l, x_u = primals
    dQ, dc, dA, db, dx_l, dx_u = tangents

    # Forward solve
    obj, x, y, z_l, z_u = qp_solve(Q, c, A, b, x_l, x_u)

    n = c.shape[0]
    m = A.shape[0]

    # Slacks
    s_l = jnp.maximum(x - x_l, _EPS)
    s_u = jnp.maximum(x_u - x, _EPS)

    # Bound masks
    has_lb = (x_l > -1e20).astype(jnp.float64)
    has_ub = (x_u < 1e20).astype(jnp.float64)

    # Diagonal matrices from complementarity
    Z_l = z_l * has_lb
    Z_u = z_u * has_ub
    S_l = s_l * has_lb + (1.0 - has_lb) * 1.0
    S_u = s_u * has_ub + (1.0 - has_ub) * 1.0

    dim = 3 * n + m

    # Handle zero tangents
    dc_safe = jnp.zeros(n) if dc is None else dc
    db_safe = jnp.zeros(m) if db is None else db
    dx_l_safe = jnp.zeros(n) if dx_l is None else dx_l
    dx_u_safe = jnp.zeros(n) if dx_u is None else dx_u
    dQ_safe = jnp.zeros((n, n)) if dQ is None else dQ
    dA_safe = jnp.zeros((m, n)) if dA is None else dA

    # RHS = -dF/dp (implicit function theorem)
    # F1 = Qx + c - A'y - z_l + z_u; -dF1/dp:
    rhs1 = -dQ_safe @ x - dc_safe
    rhs1 = rhs1 + dA_safe.T @ y

    # F2 = Ax - b; -dF2/dp:
    rhs2 = -dA_safe @ x + db_safe

    # F3 = S_l Z_l e; dF3/dx_l = -Z_l => -dF3/dp = Z_l dx_l
    rhs3 = Z_l * dx_l_safe

    # F4 = S_u Z_u e; dF4/dx_u = Z_u => -dF4/dp = -Z_u dx_u
    rhs4 = -Z_u * dx_u_safe

    rhs = jnp.concatenate([rhs1, rhs2, rhs3, rhs4])

    # Build KKT matrix
    KKT = jnp.zeros((dim, dim), dtype=jnp.float64)

    # Row block 1: [Q, -A', -I, I]
    KKT = KKT.at[:n, :n].set(Q)
    KKT = KKT.at[:n, n : n + m].set(-A.T)
    KKT = KKT.at[:n, n + m : n + m + n].set(-jnp.diag(has_lb))
    KKT = KKT.at[:n, n + m + n :].set(jnp.diag(has_ub))

    # Row block 2: [A, 0, 0, 0]
    KKT = KKT.at[n : n + m, :n].set(A)

    # Row block 3: [Z_l, 0, S_l, 0]
    KKT = KKT.at[n + m : n + m + n, :n].set(jnp.diag(Z_l))
    KKT = KKT.at[n + m : n + m + n, n + m : n + m + n].set(jnp.diag(S_l))

    # Row block 4: [-Z_u, 0, 0, S_u]
    KKT = KKT.at[n + m + n :, :n].set(jnp.diag(-Z_u))
    KKT = KKT.at[n + m + n :, n + m + n :].set(jnp.diag(S_u))

    # Regularize slightly for numerical stability
    KKT = KKT + 1e-12 * jnp.eye(dim)

    # Solve for sensitivities
    sol = jnp.linalg.solve(KKT, rhs)

    dx_sol = sol[:n]
    dy_sol = sol[n : n + m]
    dz_l_sol = sol[n + m : n + m + n]
    dz_u_sol = sol[n + m + n :]

    # Tangent of objective: d(0.5 x'Qx + c'x)
    # = 0.5 x' dQ x + (Qx + c)' dx + dc' x
    dobj = 0.5 * x @ dQ_safe @ x + jnp.dot(Q @ x + c, dx_sol) + jnp.dot(dc_safe, x)

    primals_out = (obj, x, y, z_l, z_u)
    tangents_out = (dobj, dx_sol, dy_sol, dz_l_sol, dz_u_sol)
    return primals_out, tangents_out


def qp_solve_grad(Q, c, A, b, x_l, x_u):
    """Convenience: solve QP and return just objective (grad-compatible).

    Usage:
        grad_fn = jax.grad(qp_solve_grad, argnums=1)  # grad w.r.t. c
        dc = grad_fn(Q, c, A, b, x_l, x_u)
    """
    obj, _, _, _, _ = qp_solve(Q, c, A, b, x_l, x_u)
    return obj
