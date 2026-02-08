"""
Differentiable LP solver: compute gradients through LP solutions via custom_jvp.

Uses implicit differentiation of the KKT conditions at the IPM optimal solution.
The IPM produces the analytic center of the optimal face, so the complementarity
slacks are never exactly zero — the KKT sensitivity system is always non-singular
(unlike simplex-based sensitivity where active constraints can be degenerate).

Reference: Amos & Kolter, "OptNet: Differentiable Optimization as a Layer in
Neural Networks" (2017) — adapted for LP (Q=0 special case).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from discopt._jax.lp_ipm import lp_ipm_solve

_EPS = 1e-20


@jax.custom_jvp
def lp_solve(c, A, b, x_l, x_u):
    """Solve an LP and return (obj, x, y, z_l, z_u).

    This function is differentiable w.r.t. all inputs via implicit
    differentiation of the KKT conditions.

    Args:
        c: (n,) objective coefficients.
        A: (m, n) equality constraint matrix.
        b: (m,) equality RHS.
        x_l: (n,) lower variable bounds.
        x_u: (n,) upper variable bounds.

    Returns:
        Tuple of (obj, x, y, z_l, z_u) where:
          obj: scalar optimal objective value
          x: (n,) optimal primal solution
          y: (m,) optimal dual variables for equality constraints
          z_l: (n,) optimal dual variables for lower bounds
          z_u: (n,) optimal dual variables for upper bounds
    """
    state = lp_ipm_solve(c, A, b, x_l, x_u)
    return state.obj, state.x, state.y, state.z_l, state.z_u


@lp_solve.defjvp
def lp_solve_jvp(primals, tangents):
    """JVP rule for LP solve via implicit differentiation of KKT.

    At the optimal solution (x*, y*, z_l*, z_u*), the KKT conditions hold:
        F1: c - A'y - z_l + z_u = 0         (dual feasibility)
        F2: Ax - b = 0                       (primal feasibility)
        F3: S_l Z_l e = 0                    (complementarity, lower)
        F4: S_u Z_u e = 0                    (complementarity, upper)

    Differentiating F=0 w.r.t. parameters p = (c, A, b, x_l, x_u):
        dF/d(x,y,z_l,z_u) · d(x,y,z_l,z_u)/dp = -dF/dp

    The Jacobian of F w.r.t. (x, y, z_l, z_u):
        [ 0    -A'   -I    I   ]
        [ A     0     0    0   ]
        [ Z_l   0    S_l   0   ]
        [ -Z_u  0     0   S_u  ]
    """
    c, A, b, x_l, x_u = primals
    dc, dA, db, dx_l, dx_u = tangents

    # Forward solve
    obj, x, y, z_l, z_u = lp_solve(c, A, b, x_l, x_u)

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
    S_l = s_l * has_lb + (1.0 - has_lb) * 1.0  # avoid zero diagonal
    S_u = s_u * has_ub + (1.0 - has_ub) * 1.0

    # Build the KKT sensitivity system
    # Variables: [dx_sol (n), dy_sol (m), dz_l_sol (n), dz_u_sol (n)]
    dim = 2 * n + m + n  # n + m + n + n = 3n + m

    # Handle zero tangents
    dc_safe = jnp.zeros(n) if dc is None else dc
    db_safe = jnp.zeros(m) if db is None else db
    dx_l_safe = jnp.zeros(n) if dx_l is None else dx_l
    dx_u_safe = jnp.zeros(n) if dx_u is None else dx_u

    # RHS = -dF/dp (implicit function theorem)
    # F1 = c - A'y - z_l + z_u; dF1/dc=I, dF1/dA=-y' => -dF1/dp
    rhs1 = -dc_safe
    if dA is not None:
        rhs1 = rhs1 + dA.T @ y

    # F2 = Ax - b; dF2/db=-I, dF2/dA=x' => -dF2/dp
    rhs2 = db_safe
    if dA is not None:
        rhs2 = rhs2 - dA @ x

    # F3 = S_l Z_l e (complementarity lower); dF3/dx_l = -Z_l
    # => -dF3/dp = Z_l dx_l
    rhs3 = Z_l * dx_l_safe

    # F4 = S_u Z_u e (complementarity upper); dF4/dx_u = Z_u
    # => -dF4/dp = -Z_u dx_u
    rhs4 = -Z_u * dx_u_safe

    rhs = jnp.concatenate([rhs1, rhs2, rhs3, rhs4])

    # Build KKT matrix
    KKT = jnp.zeros((dim, dim), dtype=jnp.float64)

    # Row block 1: [0, -A', -I, I]
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

    # Tangent of objective: d(c'x*) = dc'x* + c'dx*
    dobj = jnp.dot(dc_safe, x) + jnp.dot(c, dx_sol)

    # Tangent outputs
    dz_l_sol = sol[n + m : n + m + n]
    dz_u_sol = sol[n + m + n :]

    primals_out = (obj, x, y, z_l, z_u)
    tangents_out = (dobj, dx_sol, dy_sol, dz_l_sol, dz_u_sol)
    return primals_out, tangents_out


def lp_solve_grad(c, A, b, x_l, x_u):
    """Convenience: solve LP and return just objective (grad-compatible).

    This is the function to use with jax.grad:
        grad_fn = jax.grad(lp_solve_grad)
        dc = grad_fn(c, A, b, x_l, x_u)  # gradient w.r.t. c
    """
    obj, _, _, _, _ = lp_solve(c, A, b, x_l, x_u)
    return obj
