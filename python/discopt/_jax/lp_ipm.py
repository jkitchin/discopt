"""
Pure-JAX Interior Point Method (IPM) for linear programming.

Implements a Mehrotra predictor-corrector primal-dual IPM for LP:

    min  c'x
    s.t. A x = b          (m equality constraints)
         x_l <= x <= x_u   (variable bounds)

Algorithm details:
  - Augmented KKT system formulation (same structure as QP IPM with Q=0)
  - Mehrotra predictor-corrector with centering parameter sigma = (mu_aff/mu)^3
  - Fraction-to-boundary rule for step sizes
  - jax.lax.while_loop for JIT-compatible iteration

All data structures are NamedTuples for JAX pytree compatibility.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INF = 1e20
_EPS = 1e-20
_SLACK_FLOOR = 1e-12

# ---------------------------------------------------------------------------
# Data structures (NamedTuples for JAX pytree compatibility)
# ---------------------------------------------------------------------------


class LPIPMOptions(NamedTuple):
    """Options for the LP IPM solver."""

    tol: float = 1e-8
    max_iter: int = 100
    tau_min: float = 0.99
    bound_push: float = 1e-2


class LPIPMState(NamedTuple):
    """State carried through the while_loop."""

    x: jnp.ndarray  # (n,) primal variables
    y: jnp.ndarray  # (m,) equality constraint multipliers
    z_l: jnp.ndarray  # (n,) lower bound multipliers
    z_u: jnp.ndarray  # (n,) upper bound multipliers
    mu: jnp.ndarray  # scalar barrier parameter
    iteration: jnp.ndarray  # scalar int
    converged: jnp.ndarray  # 0=running, 1=optimal, 3=max_iter
    obj: jnp.ndarray  # scalar objective value


class LPProblemData(NamedTuple):
    """Pre-computed LP problem structure."""

    c: jnp.ndarray  # (n,) cost vector
    A: jnp.ndarray  # (m, n) constraint matrix
    b: jnp.ndarray  # (m,) rhs vector
    x_l: jnp.ndarray  # (n,) lower variable bounds
    x_u: jnp.ndarray  # (n,) upper variable bounds
    has_lb: jnp.ndarray  # (n,) float mask
    has_ub: jnp.ndarray  # (n,) float mask
    n: int
    m: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fraction_to_boundary(vals, dvals, tau):
    """Max step alpha s.t. vals + alpha*dvals >= (1-tau)*vals."""
    neg_mask = dvals < 0.0
    ratios = jnp.where(
        neg_mask,
        -tau * vals / jnp.where(neg_mask, dvals, -1.0),
        jnp.array(1e20),
    )
    return jnp.clip(jnp.min(ratios, initial=1e20), 0.0, 1.0)


def _push_from_bounds(x, x_l, x_u, has_lb, has_ub, bp):
    """Push x away from bounds by at least bp."""
    rng = x_u - x_l
    push = jnp.minimum(bp, 0.5 * rng)
    push = jnp.where(rng < 1e30, push, bp)
    x_new = x
    x_new = jnp.where((has_lb > 0.5) & (x_new < x_l + push), x_l + push, x_new)
    x_new = jnp.where((has_ub > 0.5) & (x_new > x_u - push), x_u - push, x_new)
    x_new = jnp.where(has_lb > 0.5, jnp.maximum(x_new, x_l), x_new)
    x_new = jnp.where(has_ub > 0.5, jnp.minimum(x_new, x_u), x_new)
    return x_new


def _make_problem_data(c, A, b, x_l, x_u):
    """Build LPProblemData with bound masks."""
    n = c.shape[0]
    m = b.shape[0]
    has_lb = (x_l > -_INF).astype(jnp.float64)
    has_ub = (x_u < _INF).astype(jnp.float64)
    return LPProblemData(
        c=c,
        A=A,
        b=b,
        x_l=x_l,
        x_u=x_u,
        has_lb=has_lb,
        has_ub=has_ub,
        n=n,
        m=m,
    )


# ---------------------------------------------------------------------------
# Augmented system solve
# ---------------------------------------------------------------------------


def _solve_augmented(Sig, A, rhs_x, rhs_y, n, m):
    """Solve the augmented KKT system for LP.

    [diag(Sig)   -A'] [dx]   [rhs_x]
    [A            0 ] [dy] = [rhs_y]

    Same structure as QP IPM with Q=0.
    """
    if m > 0:
        W = jnp.diag(Sig)
        KKT = jnp.block(
            [
                [W, -A.T],
                [A, jnp.zeros((m, m), dtype=jnp.float64)],
            ]
        )
        rhs = jnp.concatenate([rhs_x, rhs_y])
        sol = jnp.linalg.solve(KKT, rhs)
        return sol[:n], sol[n:]
    else:
        dx = rhs_x / jnp.maximum(Sig, _EPS)
        return dx, jnp.zeros(0, dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def _initialize_state(pd, opts):
    """Create initial LPIPMState."""
    m = pd.m
    mu = jnp.array(0.1, dtype=jnp.float64)

    x0 = jnp.where(
        (pd.has_lb > 0.5) & (pd.has_ub > 0.5),
        0.5 * (pd.x_l + pd.x_u),
        jnp.where(
            pd.has_lb > 0.5,
            pd.x_l + 1.0,
            jnp.where(pd.has_ub > 0.5, pd.x_u - 1.0, 0.0),
        ),
    )
    x = _push_from_bounds(
        x0,
        pd.x_l,
        pd.x_u,
        pd.has_lb,
        pd.has_ub,
        opts.bound_push,
    )

    s_l = jnp.maximum(x - pd.x_l, _SLACK_FLOOR) * pd.has_lb
    s_u = jnp.maximum(pd.x_u - x, _SLACK_FLOOR) * pd.has_ub
    z_l = jnp.where(
        pd.has_lb > 0.5,
        mu / jnp.maximum(s_l, _SLACK_FLOOR),
        0.0,
    )
    z_u = jnp.where(
        pd.has_ub > 0.5,
        mu / jnp.maximum(s_u, _SLACK_FLOOR),
        0.0,
    )
    y = jnp.zeros(m, dtype=jnp.float64)
    obj = jnp.dot(pd.c, x)

    return LPIPMState(
        x=x,
        y=y,
        z_l=z_l,
        z_u=z_u,
        mu=mu,
        iteration=jnp.array(0, dtype=jnp.int32),
        converged=jnp.array(0, dtype=jnp.int32),
        obj=obj,
    )


# ---------------------------------------------------------------------------
# Solve for m=0 (no equality constraints)
# ---------------------------------------------------------------------------


def _solve_unconstrained(pd, opts):
    """When m=0, the LP reduces to clamping x to bounds based on c."""
    n = pd.n
    x = jnp.zeros(n, dtype=jnp.float64)
    x = jnp.where((pd.c > 0) & (pd.has_lb > 0.5), pd.x_l, x)
    x = jnp.where((pd.c < 0) & (pd.has_ub > 0.5), pd.x_u, x)
    x = jnp.where((pd.c == 0) & (pd.has_lb > 0.5), pd.x_l, x)

    obj = jnp.dot(pd.c, x)
    return LPIPMState(
        x=x,
        y=jnp.zeros(0, dtype=jnp.float64),
        z_l=jnp.maximum(-pd.c, 0.0) * pd.has_lb,
        z_u=jnp.maximum(pd.c, 0.0) * pd.has_ub,
        mu=jnp.array(0.0, dtype=jnp.float64),
        iteration=jnp.array(0, dtype=jnp.int32),
        converged=jnp.array(1, dtype=jnp.int32),
        obj=obj,
    )


# ---------------------------------------------------------------------------
# Core iteration body (Mehrotra predictor-corrector)
# ---------------------------------------------------------------------------


def _make_iteration_body(pd, opts):
    """Build the while_loop body for one LP IPM iteration.

    Uses the augmented KKT system approach (same as QP IPM with Q=0).
    Mehrotra predictor-corrector:
      1. Affine step (mu=0) to get search direction
      2. Centering parameter sigma = (mu_aff / mu)^3
      3. Corrected step with centering and second-order terms
    """
    n, m = pd.n, pd.m
    c, A, b = pd.c, pd.A, pd.b

    def body(state):
        x, y, mu = state.x, state.y, state.mu
        z_l, z_u = state.z_l, state.z_u
        tau = jnp.maximum(1.0 - mu, opts.tau_min)

        s_l = jnp.maximum(x - pd.x_l, _SLACK_FLOOR) * pd.has_lb
        s_u = jnp.maximum(pd.x_u - x, _SLACK_FLOOR) * pd.has_ub

        Sig = (
            pd.has_lb * z_l / jnp.maximum(s_l, _EPS)
            + pd.has_ub * z_u / jnp.maximum(s_u, _EPS)
            + _EPS
        )

        r_dual = c - z_l + z_u - A.T @ y
        r_prim = A @ x - b
        r_comp_l = pd.has_lb * s_l * z_l
        r_comp_u = pd.has_ub * s_u * z_u

        # --- Affine (predictor) step ---
        rhs_x_aff = -r_dual - pd.has_lb * z_l + pd.has_ub * z_u
        rhs_y_aff = -r_prim
        dx_aff, dy_aff = _solve_augmented(
            Sig,
            A,
            rhs_x_aff,
            rhs_y_aff,
            n,
            m,
        )

        dz_l_aff = pd.has_lb * ((-r_comp_l - z_l * dx_aff) / jnp.maximum(s_l, _EPS))
        dz_u_aff = pd.has_ub * ((-r_comp_u + z_u * dx_aff) / jnp.maximum(s_u, _EPS))

        alpha_aff_p = jnp.array(1.0)
        alpha_aff_p = jnp.minimum(
            alpha_aff_p,
            _fraction_to_boundary(
                jnp.where(pd.has_lb > 0.5, s_l, 1.0),
                jnp.where(pd.has_lb > 0.5, dx_aff, 0.0),
                1.0,
            ),
        )
        alpha_aff_p = jnp.minimum(
            alpha_aff_p,
            _fraction_to_boundary(
                jnp.where(pd.has_ub > 0.5, s_u, 1.0),
                jnp.where(pd.has_ub > 0.5, -dx_aff, 0.0),
                1.0,
            ),
        )

        alpha_aff_d = jnp.array(1.0)
        alpha_aff_d = jnp.minimum(
            alpha_aff_d,
            _fraction_to_boundary(
                jnp.where(pd.has_lb > 0.5, z_l, 1.0),
                jnp.where(pd.has_lb > 0.5, dz_l_aff, 0.0),
                1.0,
            ),
        )
        alpha_aff_d = jnp.minimum(
            alpha_aff_d,
            _fraction_to_boundary(
                jnp.where(pd.has_ub > 0.5, z_u, 1.0),
                jnp.where(pd.has_ub > 0.5, dz_u_aff, 0.0),
                1.0,
            ),
        )

        # --- Centering parameter ---
        n_pairs = jnp.maximum(
            jnp.sum(pd.has_lb) + jnp.sum(pd.has_ub),
            1.0,
        )
        s_l_aff = s_l + alpha_aff_p * dx_aff
        s_u_aff = s_u - alpha_aff_p * dx_aff
        z_l_aff = z_l + alpha_aff_d * dz_l_aff
        z_u_aff = z_u + alpha_aff_d * dz_u_aff
        mu_aff = (
            jnp.sum(pd.has_lb * s_l_aff * z_l_aff) + jnp.sum(pd.has_ub * s_u_aff * z_u_aff)
        ) / n_pairs
        sigma = jnp.clip(
            (mu_aff / jnp.maximum(mu, _EPS)) ** 3,
            0.0,
            1.0,
        )
        mu_target = sigma * mu

        # --- Corrected (centering + second-order) step ---
        corr_l = pd.has_lb * (dx_aff * dz_l_aff) / jnp.maximum(s_l, _EPS)
        corr_u = pd.has_ub * (dx_aff * dz_u_aff) / jnp.maximum(s_u, _EPS)
        rhs_x_cc = (
            -r_dual
            + pd.has_lb * (mu_target / jnp.maximum(s_l, _EPS) - z_l)
            - pd.has_ub * (mu_target / jnp.maximum(s_u, _EPS) - z_u)
            - corr_l
            - corr_u
        )
        rhs_y_cc = -r_prim
        dx, dy = _solve_augmented(Sig, A, rhs_x_cc, rhs_y_cc, n, m)

        dz_l = pd.has_lb * (
            (mu_target - z_l * (s_l + dx) - dx_aff * dz_l_aff) / jnp.maximum(s_l, _EPS)
        )
        dz_u = pd.has_ub * (
            (mu_target - z_u * (s_u - dx) + dx_aff * dz_u_aff) / jnp.maximum(s_u, _EPS)
        )

        # --- Step sizes ---
        alpha_p = jnp.array(1.0)
        alpha_p = jnp.minimum(
            alpha_p,
            _fraction_to_boundary(
                jnp.where(pd.has_lb > 0.5, s_l, 1.0),
                jnp.where(pd.has_lb > 0.5, dx, 0.0),
                tau,
            ),
        )
        alpha_p = jnp.minimum(
            alpha_p,
            _fraction_to_boundary(
                jnp.where(pd.has_ub > 0.5, s_u, 1.0),
                jnp.where(pd.has_ub > 0.5, -dx, 0.0),
                tau,
            ),
        )
        alpha_d = jnp.array(1.0)
        alpha_d = jnp.minimum(
            alpha_d,
            _fraction_to_boundary(
                jnp.where(pd.has_lb > 0.5, z_l, 1.0),
                jnp.where(pd.has_lb > 0.5, dz_l, 0.0),
                tau,
            ),
        )
        alpha_d = jnp.minimum(
            alpha_d,
            _fraction_to_boundary(
                jnp.where(pd.has_ub > 0.5, z_u, 1.0),
                jnp.where(pd.has_ub > 0.5, dz_u, 0.0),
                tau,
            ),
        )

        # --- Update variables ---
        x_new = x + alpha_p * dx
        x_new = jnp.where(
            pd.has_lb > 0.5,
            jnp.maximum(x_new, pd.x_l + _SLACK_FLOOR),
            x_new,
        )
        x_new = jnp.where(
            pd.has_ub > 0.5,
            jnp.minimum(x_new, pd.x_u - _SLACK_FLOOR),
            x_new,
        )
        z_l_new = jnp.maximum(z_l + alpha_d * dz_l, _EPS) * pd.has_lb
        z_u_new = jnp.maximum(z_u + alpha_d * dz_u, _EPS) * pd.has_ub
        y_new = y + alpha_d * dy

        # --- Stationarity-based z recovery at active bounds ---
        s_l_new = jnp.maximum(x_new - pd.x_l, _SLACK_FLOOR) * pd.has_lb
        s_u_new = jnp.maximum(pd.x_u - x_new, _SLACK_FLOOR) * pd.has_ub
        grad_stat = c - A.T @ y_new
        z_u_stat = jnp.maximum(-(grad_stat - z_l_new), _EPS)
        z_l_stat = jnp.maximum(grad_stat + z_u_new, _EPS)
        at_ub = pd.has_ub * (s_u_new <= _SLACK_FLOOR * 2.0).astype(jnp.float64)
        at_lb = pd.has_lb * (s_l_new <= _SLACK_FLOOR * 2.0).astype(jnp.float64)
        both = at_lb * at_ub
        at_lb = at_lb * (1.0 - both)
        at_ub = at_ub * (1.0 - both)
        z_u_new = jnp.where(at_ub > 0.5, z_u_stat, z_u_new)
        z_l_new = jnp.where(at_lb > 0.5, z_l_stat, z_l_new)

        # --- Update barrier parameter ---
        compl = jnp.sum(pd.has_lb * z_l_new * s_l_new) + jnp.sum(pd.has_ub * z_u_new * s_u_new)
        mu_new = compl / jnp.maximum(n_pairs, 1.0)
        mu_new = jnp.minimum(mu_new, mu)
        mu_new = jnp.maximum(mu_new, _EPS)

        # --- Check convergence ---
        r_dual_new = c - z_l_new + z_u_new - A.T @ y_new
        r_prim_new = A @ x_new - b
        primal_infeas = jnp.max(jnp.abs(r_prim_new)) / (1.0 + jnp.max(jnp.abs(b)))
        dual_infeas = jnp.max(jnp.abs(r_dual_new)) / (1.0 + jnp.max(jnp.abs(c)))
        obj_p = jnp.dot(c, x_new)
        gap = mu_new / (1.0 + jnp.abs(obj_p))

        new_iter = state.iteration + 1
        optimal = (primal_infeas <= opts.tol) & (dual_infeas <= opts.tol) & (gap <= opts.tol)
        at_max = new_iter >= opts.max_iter
        code = jnp.where(optimal, jnp.int32(1), jnp.int32(0))
        code = jnp.where(
            (code == 0) & at_max,
            jnp.int32(3),
            code,
        )

        return LPIPMState(
            x=x_new,
            y=y_new,
            z_l=z_l_new,
            z_u=z_u_new,
            mu=mu_new,
            iteration=new_iter,
            converged=code.astype(jnp.int32),
            obj=obj_p,
        )

    return body


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lp_ipm_solve(
    c: jnp.ndarray,
    A: jnp.ndarray,
    b: jnp.ndarray,
    x_l: jnp.ndarray,
    x_u: jnp.ndarray,
    options: LPIPMOptions | None = None,
) -> LPIPMState:
    """Solve an LP using a pure-JAX Mehrotra predictor-corrector IPM.

    Standard form::

        min  c'x
        s.t. A x = b
             x_l <= x <= x_u

    Args:
        c: Cost vector (n,).
        A: Equality constraint matrix (m, n).
        b: Equality constraint RHS (m,).
        x_l: Lower variable bounds (n,). Use -1e20 for unbounded.
        x_u: Upper variable bounds (n,). Use 1e20 for unbounded.
        options: LPIPMOptions.

    Returns:
        Final LPIPMState. converged: 1=optimal, 3=max_iter.
    """
    opts = options if options is not None else LPIPMOptions()
    c = jnp.asarray(c, dtype=jnp.float64)
    A = jnp.asarray(A, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    x_l = jnp.asarray(x_l, dtype=jnp.float64)
    x_u = jnp.asarray(x_u, dtype=jnp.float64)

    pd = _make_problem_data(c, A, b, x_l, x_u)

    if pd.m == 0:
        result: LPIPMState = _solve_unconstrained(pd, opts)
        return result

    state = _initialize_state(pd, opts)
    body = _make_iteration_body(pd, opts)

    def cond(st):
        return st.converged == 0

    result = jax.lax.while_loop(cond, body, state)
    return result


def lp_ipm_solve_batch(
    c: jnp.ndarray,
    A: jnp.ndarray,
    b: jnp.ndarray,
    xl_batch: jnp.ndarray,
    xu_batch: jnp.ndarray,
    options: LPIPMOptions | None = None,
) -> LPIPMState:
    """Batch LP solve via jax.vmap over variable bounds.

    All instances share c, A, b but have per-instance bounds.

    Args:
        c: Cost vector (n,).
        A: Equality constraint matrix (m, n).
        b: Equality constraint RHS (m,).
        xl_batch: Lower bounds (batch, n).
        xu_batch: Upper bounds (batch, n).
        options: LPIPMOptions.

    Returns:
        LPIPMState with batched arrays (batch, ...).
    """

    def _solve_single(xl_single, xu_single):
        return lp_ipm_solve(c, A, b, xl_single, xu_single, options)

    result: LPIPMState = jax.vmap(_solve_single)(xl_batch, xu_batch)
    return result
