"""
Callback-based Interior Point Method (IPM) for nonlinear programming.

Same Mehrotra predictor-corrector algorithm as ipm.py, but accepts explicit
derivative callbacks instead of using JAX autodiff.  This allows benchmarking
on CUTEst and other problems whose functions are not JAX-traceable.

Key differences from ipm.py:
  - Python while loop (not jax.lax.while_loop)
  - Calls grad_fn(x), hess_fn(x, obj_factor, y), jac_fn(x) directly
  - Uses jnp.linalg.solve for KKT (dense only)
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np

from discopt._jax.ipm import (
    IPMOptions,
    IPMState,
    _constraint_violation,
    _fraction_to_boundary,
    _make_problem_data,
    _push_from_bounds,
    _safeguard_z,
    _total_violation,
)

_INF = 1e20
_EPS = 1e-20
_SLACK_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# Feasibility restoration helpers
# ---------------------------------------------------------------------------


def _build_restoration_callbacks(
    con_fn: Callable[[np.ndarray], np.ndarray],
    jac_fn: Callable[[np.ndarray], np.ndarray],
    x_ref: np.ndarray,
    g_l: np.ndarray,
    g_u: np.ndarray,
    has_g_lb: np.ndarray,
    has_g_ub: np.ndarray,
    zeta: float = 0.999,
    beta: float = 50.0,
) -> tuple[Callable, Callable, Callable]:
    """Build obj/grad/hess callbacks for the softplus-smoothed restoration subproblem.

    Minimizes constraint violation using softplus(v) = log(1+exp(beta*v))/beta
    as a smooth approximation to max(v, 0), plus a proximity term to x_ref.
    The resulting problem is bound-constrained only (no general constraints).
    """
    rho = 1.0 - zeta
    x_ref_j = jnp.asarray(x_ref, dtype=jnp.float64)
    g_l_j = jnp.asarray(g_l, dtype=jnp.float64)
    g_u_j = jnp.asarray(g_u, dtype=jnp.float64)
    has_lb = jnp.asarray(has_g_lb, dtype=jnp.float64)
    has_ub = jnp.asarray(has_g_ub, dtype=jnp.float64)

    def _softplus(v):
        # Numerically stable softplus: log(1+exp(beta*v))/beta
        bv = beta * v
        return jnp.where(bv > 20.0, v, jnp.log1p(jnp.exp(bv)) / beta)

    def _sigmoid(v):
        bv = beta * v
        return jnp.where(bv > 20.0, 1.0, jnp.where(bv < -20.0, 0.0, 1.0 / (1.0 + jnp.exp(-bv))))

    def obj_fn(x):
        c = jnp.asarray(con_fn(np.asarray(x, dtype=np.float64)), dtype=jnp.float64)
        viol_lb = has_lb * _softplus(g_l_j - c)
        viol_ub = has_ub * _softplus(c - g_u_j)
        feas = zeta * jnp.sum(viol_lb + viol_ub)
        prox = rho / 2.0 * jnp.sum((jnp.asarray(x) - x_ref_j) ** 2)
        return float(feas + prox)

    def grad_fn(x):
        x_j = jnp.asarray(x, dtype=jnp.float64)
        c = jnp.asarray(con_fn(np.asarray(x, dtype=np.float64)), dtype=jnp.float64)
        J = jnp.asarray(jac_fn(np.asarray(x, dtype=np.float64)), dtype=jnp.float64)
        sig_lb = has_lb * _sigmoid(g_l_j - c)  # d/dc softplus(g_l - c) = -sigmoid(g_l - c)
        sig_ub = has_ub * _sigmoid(c - g_u_j)  # d/dc softplus(c - g_u) = sigmoid(c - g_u)
        # gradient w.r.t. x: zeta * J^T @ (sig_ub - sig_lb) + rho * (x - x_ref)
        g = zeta * (J.T @ (sig_ub - sig_lb)) + rho * (x_j - x_ref_j)
        return np.asarray(g, dtype=np.float64)

    def hess_fn(x, obj_factor, y):
        # Gauss-Newton approximation: zeta * J^T @ diag(w) @ J + rho * I
        c = jnp.asarray(con_fn(np.asarray(x, dtype=np.float64)), dtype=jnp.float64)
        J = jnp.asarray(jac_fn(np.asarray(x, dtype=np.float64)), dtype=jnp.float64)
        w_lb = has_lb * beta * _sigmoid(g_l_j - c) * (1.0 - _sigmoid(g_l_j - c))
        w_ub = has_ub * beta * _sigmoid(c - g_u_j) * (1.0 - _sigmoid(c - g_u_j))
        n = J.shape[1]
        H = obj_factor * (zeta * (J.T @ jnp.diag(w_lb + w_ub) @ J) + rho * jnp.eye(n))
        return np.asarray(H, dtype=np.float64)

    return obj_fn, grad_fn, hess_fn


def _feasibility_restoration(
    con_fn: Callable,
    jac_fn: Callable,
    x_current: np.ndarray,
    x_l: np.ndarray,
    x_u: np.ndarray,
    g_l: np.ndarray,
    g_u: np.ndarray,
    has_g_lb: np.ndarray,
    has_g_ub: np.ndarray,
    opts: "IPMOptions",
) -> tuple[np.ndarray, bool]:
    """Run a feasibility restoration subproblem.

    Returns (x_restored, success) where success means violation decreased >= 10%.
    """
    # Compute initial violation
    c0 = jnp.asarray(con_fn(np.asarray(x_current, dtype=np.float64)), dtype=jnp.float64)
    viol0 = float(_total_violation(c0, jnp.asarray(g_l), jnp.asarray(g_u), has_g_lb, has_g_ub))
    if viol0 < opts.tol:
        return np.asarray(x_current), False  # already feasible

    r_obj, r_grad, r_hess = _build_restoration_callbacks(
        con_fn,
        jac_fn,
        np.asarray(x_current),
        g_l,
        g_u,
        has_g_lb,
        has_g_ub,
    )

    rest_opts = IPMOptions(
        max_iter=500,
        mu_init=0.1,
        tol=1e-6,
        acceptable_tol=1e-4,
        acceptable_iter=5,
        predictor_corrector=True,
    )

    state = ipm_solve_callbacks(
        r_obj,
        r_grad,
        r_hess,
        None,
        None,
        np.asarray(x_current, dtype=np.float64),
        np.asarray(x_l, dtype=np.float64),
        np.asarray(x_u, dtype=np.float64),
        None,
        None,
        options=rest_opts,
        _in_restoration=True,
    )

    x_rest = np.asarray(state.x, dtype=np.float64)
    c1 = jnp.asarray(con_fn(x_rest), dtype=jnp.float64)
    viol1 = float(_total_violation(c1, jnp.asarray(g_l), jnp.asarray(g_u), has_g_lb, has_g_ub))

    success = viol1 < viol0 * 0.9  # at least 10% reduction
    return x_rest, success


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------


def ipm_solve_callbacks(
    obj_fn: Callable[[np.ndarray], float],
    grad_fn: Callable[[np.ndarray], np.ndarray],
    hess_fn: Callable[[np.ndarray, float, np.ndarray], np.ndarray],
    con_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    jac_fn: Optional[Callable[[np.ndarray], np.ndarray]],
    x0: np.ndarray,
    x_l: np.ndarray,
    x_u: np.ndarray,
    g_l: Optional[np.ndarray] = None,
    g_u: Optional[np.ndarray] = None,
    options: Optional[IPMOptions] = None,
    _in_restoration: bool = False,
) -> IPMState:
    """Solve an NLP using a callback-based IPM.

    Args:
        obj_fn: f(x) -> scalar objective.
        grad_fn: grad(x) -> (n,) gradient of objective.
        hess_fn: hess(x, obj_factor, y) -> (n,n) Lagrangian Hessian.
        con_fn: c(x) -> (m,) constraints, or None if unconstrained.
        jac_fn: jac(x) -> (m,n) Jacobian of constraints, or None.
        x0: Initial point (n,).
        x_l: Lower variable bounds (n,).  Use -1e20 for unbounded.
        x_u: Upper variable bounds (n,).  Use 1e20 for unbounded.
        g_l: Lower constraint bounds (m,).
        g_u: Upper constraint bounds (m,).
        options: IPMOptions.

    Returns:
        Final IPMState with solution in state.x.
    """
    opts = options if options is not None else IPMOptions()

    x_l = jnp.asarray(x_l, dtype=jnp.float64)
    x_u = jnp.asarray(x_u, dtype=jnp.float64)
    if g_l is None:
        g_l = jnp.zeros(0, dtype=jnp.float64)
    else:
        g_l = jnp.asarray(g_l, dtype=jnp.float64)
    if g_u is None:
        g_u = jnp.zeros(0, dtype=jnp.float64)
    else:
        g_u = jnp.asarray(g_u, dtype=jnp.float64)

    # --- Normalize >= constraints to <= form ---
    # For constraints with only a lower bound (c >= g_l), negate to -c <= -g_l.
    # This ensures all inequality multipliers have consistent sign (y >= 0).
    if g_l.shape[0] > 0 and con_fn is not None:
        has_g_lb_raw = g_l > -_INF
        has_g_ub_raw = g_u < _INF
        is_eq_raw = has_g_lb_raw & has_g_ub_raw & (jnp.abs(g_u - g_l) < 1e-12)
        lb_only = has_g_lb_raw & ~has_g_ub_raw & ~is_eq_raw
        if jnp.any(lb_only):
            # Negate lower-bound-only rows: -c <= -g_l
            negate_mask = lb_only.astype(jnp.float64)
            sign = 1.0 - 2.0 * negate_mask  # +1 for normal, -1 for negated
            new_g_l = jnp.where(lb_only, -_INF * jnp.ones_like(g_u), g_l)
            new_g_u = jnp.where(lb_only, -g_l, g_u)
            g_l = new_g_l
            g_u = new_g_u

            # Wrap callbacks to negate the appropriate constraint rows
            _orig_con_fn = con_fn
            _orig_jac_fn = jac_fn
            _orig_hess_fn = hess_fn
            _sign = np.asarray(sign, dtype=np.float64)

            def con_fn(x, _s=_sign, _f=_orig_con_fn):  # noqa: E731
                return _s * _f(x)

            def jac_fn(x, _s=_sign, _f=_orig_jac_fn):  # noqa: E731
                return _s[:, None] * _f(x)

            def hess_fn(x, obj_factor, y, _s=_sign, _f=_orig_hess_fn):  # noqa: E731
                return _f(x, obj_factor, _s * y)

    pd = _make_problem_data(x_l, x_u, g_l, g_u)
    n, m = pd.n, pd.m

    # --- Initialize ---
    mu = jnp.array(opts.mu_init, dtype=jnp.float64)
    x = _push_from_bounds(
        jnp.asarray(x0, dtype=jnp.float64),
        pd.x_l,
        pd.x_u,
        pd.has_lb,
        pd.has_ub,
        opts.bound_push,
        opts.bound_frac,
    )
    sl = jnp.maximum(x - pd.x_l, _SLACK_FLOOR) * pd.has_lb
    su = jnp.maximum(pd.x_u - x, _SLACK_FLOOR) * pd.has_ub
    z_l = jnp.where(pd.has_lb > 0.5, mu / jnp.maximum(sl, _SLACK_FLOOR), 0.0)
    z_u = jnp.where(pd.has_ub > 0.5, mu / jnp.maximum(su, _SLACK_FLOOR), 0.0)
    y = jnp.zeros(m, dtype=jnp.float64)

    # Least-squares multiplier init (always for constrained problems in callback solver)
    if m > 0 and (opts.least_squares_mult_init or not _in_restoration):
        g0 = jnp.asarray(grad_fn(np.asarray(x)), dtype=jnp.float64)
        J0 = jnp.asarray(jac_fn(np.asarray(x)), dtype=jnp.float64)
        A = J0 @ J0.T + 1e-12 * jnp.eye(m)
        b = -J0 @ g0
        y_ls = jnp.linalg.solve(A, b)
        safe = jnp.max(jnp.abs(y_ls)) <= opts.constr_mult_init_max
        y = jnp.where(safe, y_ls, y)

    obj_val = float(obj_fn(np.asarray(x)))
    nu = jnp.array(0.0, dtype=jnp.float64)  # unused, kept for IPMState compat
    delta_w_last = jnp.array(0.0, dtype=jnp.float64)
    consecutive_acceptable = 0
    stall_count = 0
    no_progress_count = 0  # consecutive iterations with negligible x change

    # --- Helper to convert x for callbacks ---
    def _np(arr):
        return np.asarray(arr, dtype=np.float64)

    # --- Filter line search state ---
    # Compute initial constraint violation for filter initialization
    if m > 0 and con_fn is not None:
        _g_init = jnp.asarray(con_fn(_np(x)), dtype=jnp.float64)
        theta_init = float(_total_violation(_g_init, pd.g_l, pd.g_u, pd.has_g_lb, pd.has_g_ub))
    else:
        theta_init = 0.0
    filter_pairs: list[tuple[float, float]] = []
    theta_max = opts.theta_max_fact * max(1.0, theta_init)
    theta_min = opts.theta_min_fact * max(1.0, theta_init)

    # --- Filter helper ---
    def _is_acceptable_to_filter(f_trial, theta_trial, pairs):
        for theta_f, phi_f in pairs:
            if theta_trial >= theta_f and f_trial >= phi_f:
                return False
        return True

    # --- Restoration tracking ---
    restoration_attempts = 0
    max_restoration_attempts = 5
    best_primal_inf = float("inf")
    iters_since_feas_improve = 0

    # --- Main loop ---
    converged = 0
    iteration = 0

    while converged == 0 and iteration < opts.max_iter:
        tau = max(1.0 - float(mu), opts.tau_min)
        x_np = _np(x)

        # Evaluate
        grad_f = jnp.asarray(grad_fn(x_np), dtype=jnp.float64)
        if m > 0:
            H = jnp.asarray(hess_fn(x_np, 1.0, _np(y)), dtype=jnp.float64)
            g = jnp.asarray(con_fn(x_np), dtype=jnp.float64)
            J = jnp.asarray(jac_fn(x_np), dtype=jnp.float64)
        else:
            H = jnp.asarray(hess_fn(x_np, 1.0, np.zeros(0)), dtype=jnp.float64)
            g = jnp.zeros(0, dtype=jnp.float64)
            J = jnp.zeros((0, n), dtype=jnp.float64)

        # Variable bound slacks and Sigma
        sx_l = jnp.maximum(x - pd.x_l, _SLACK_FLOOR) * pd.has_lb
        sx_u = jnp.maximum(pd.x_u - x, _SLACK_FLOOR) * pd.has_ub
        Sig_l = pd.has_lb * z_l / jnp.maximum(sx_l, _SLACK_FLOOR)
        Sig_u = pd.has_ub * z_u / jnp.maximum(sx_u, _SLACK_FLOOR)

        # RHS for x
        inv_sx_l = pd.has_lb / jnp.maximum(sx_l, _SLACK_FLOOR)
        inv_sx_u = pd.has_ub / jnp.maximum(sx_u, _SLACK_FLOOR)
        rhs_x_base = -(grad_f - z_l + z_u)
        rhs_x = rhs_x_base + mu * inv_sx_l - mu * inv_sx_u

        if m > 0:
            rhs_x = rhs_x - J.T @ y

            slack_floor = jnp.maximum(mu, _EPS)
            s_from_lb = jnp.maximum(g - pd.g_l, slack_floor)
            s_from_ub = jnp.maximum(pd.g_u - g, slack_floor)
            z_s_lb = mu / s_from_lb
            z_s_ub = mu / s_from_ub

            ineq = 1.0 - pd.is_eq
            D_lb = pd.has_g_lb * ineq * s_from_lb / jnp.maximum(z_s_lb, _EPS)
            D_ub = pd.has_g_ub * ineq * s_from_ub / jnp.maximum(z_s_ub, _EPS)
            D_diag = D_lb + D_ub

            rhs_eq = pd.is_eq * (-(g - pd.g_l))
            rhs_ub_ineq_base = pd.has_g_ub * ineq * (pd.g_u - g - s_from_ub)
            rhs_lb_ineq_base = pd.has_g_lb * ineq * (g - pd.g_l - s_from_lb)
            inv_z_s_ub = pd.has_g_ub * ineq / jnp.maximum(z_s_ub, _EPS)
            inv_z_s_lb = pd.has_g_lb * ineq / jnp.maximum(z_s_lb, _EPS)
            rhs_ub_ineq = rhs_ub_ineq_base + mu * inv_z_s_ub
            rhs_lb_ineq = rhs_lb_ineq_base + mu * inv_z_s_lb
            rhs_ineq = rhs_ub_ineq - rhs_lb_ineq
            rhs_y = rhs_eq + rhs_ineq
            rhs_y_base = rhs_eq + rhs_ub_ineq_base - rhs_lb_ineq_base
        else:
            D_diag = jnp.zeros(0, dtype=jnp.float64)
            rhs_y = jnp.zeros(0, dtype=jnp.float64)
            rhs_y_base = jnp.zeros(0, dtype=jnp.float64)
            inv_z_s_ub = jnp.zeros(0, dtype=jnp.float64)
            inv_z_s_lb = jnp.zeros(0, dtype=jnp.float64)

        # --- KKT solver with inertia correction ---
        def _solve_kkt(delta_w, rx, ry):
            W = H + jnp.diag(Sig_l + Sig_u) + delta_w * jnp.eye(n)
            if m > 0:
                D_reg = jnp.diag(D_diag + opts.delta_c)
                KKT_mat = jnp.block([[W, J.T], [J, -D_reg]])
                rhs = jnp.concatenate([rx, ry])
                sol = jnp.linalg.solve(KKT_mat, rhs)
                return sol[:n], sol[n:], W
            else:
                return jnp.linalg.solve(W, rx), jnp.zeros(0, dtype=jnp.float64), W

        # Inertia correction loop
        init_dw = float(delta_w_last)
        if init_dw > 0:
            dw = init_dw / opts.delta_w_growth
        else:
            dw = 0.0
        dw = min(dw, opts.delta_w_max)

        for _ in range(10):
            _, _, W = _solve_kkt(jnp.array(dw), rhs_x, rhs_y)
            try:
                chol = jnp.linalg.cholesky(W)
                if not bool(jnp.any(jnp.isnan(chol))):
                    break
            except Exception:
                pass
            dw = max(dw * opts.delta_w_growth, opts.delta_w_init)
        final_dw = jnp.array(dw, dtype=jnp.float64)

        # --- Mehrotra predictor-corrector ---
        if opts.predictor_corrector and n > 0:
            # Affine predictor (mu=0)
            rhs_x_aff = rhs_x_base
            if m > 0:
                rhs_x_aff = rhs_x_aff - J.T @ y
            rhs_y_aff = rhs_y_base
            dx_aff, dy_aff, _ = _solve_kkt(final_dw, rhs_x_aff, rhs_y_aff)

            dz_l_aff = pd.has_lb * (-z_l * (sx_l + dx_aff) / jnp.maximum(sx_l, _SLACK_FLOOR))
            dz_u_aff = pd.has_ub * (-z_u * (sx_u - dx_aff) / jnp.maximum(sx_u, _SLACK_FLOOR))

            # Adaptive centering
            n_bounds = jnp.maximum(jnp.sum(pd.has_lb) + jnp.sum(pd.has_ub), 1.0)
            comp_curr = (
                jnp.sum(pd.has_lb * z_l * sx_l) + jnp.sum(pd.has_ub * z_u * sx_u)
            ) / n_bounds

            alpha_aff_p = _fraction_to_boundary(
                jnp.where(pd.has_lb > 0.5, sx_l, 1.0),
                jnp.where(pd.has_lb > 0.5, dx_aff, 0.0),
                jnp.array(1.0),
            )
            alpha_aff_p = jnp.minimum(
                alpha_aff_p,
                _fraction_to_boundary(
                    jnp.where(pd.has_ub > 0.5, sx_u, 1.0),
                    jnp.where(pd.has_ub > 0.5, -dx_aff, 0.0),
                    jnp.array(1.0),
                ),
            )
            alpha_aff_d = _fraction_to_boundary(
                jnp.where(pd.has_lb > 0.5, z_l, 1.0),
                jnp.where(pd.has_lb > 0.5, dz_l_aff, 0.0),
                jnp.array(1.0),
            )
            alpha_aff_d = jnp.minimum(
                alpha_aff_d,
                _fraction_to_boundary(
                    jnp.where(pd.has_ub > 0.5, z_u, 1.0),
                    jnp.where(pd.has_ub > 0.5, dz_u_aff, 0.0),
                    jnp.array(1.0),
                ),
            )

            sx_l_aff = sx_l + alpha_aff_p * dx_aff
            sx_u_aff = sx_u - alpha_aff_p * dx_aff
            z_l_aff = z_l + alpha_aff_d * dz_l_aff
            z_u_aff = z_u + alpha_aff_d * dz_u_aff
            comp_aff = (
                jnp.sum(pd.has_lb * z_l_aff * sx_l_aff) + jnp.sum(pd.has_ub * z_u_aff * sx_u_aff)
            ) / n_bounds

            sigma = jnp.where(comp_curr > _EPS, (comp_aff / comp_curr) ** 3, jnp.array(0.1))
            sigma = jnp.clip(sigma, 0.0, 1.0)
            sigma_mu = sigma * mu

            # Cross-product correction
            cross_x = pd.has_lb * dx_aff * dz_l_aff / jnp.maximum(sx_l, _SLACK_FLOOR)
            cross_x = cross_x - pd.has_ub * (-dx_aff) * dz_u_aff / jnp.maximum(sx_u, _SLACK_FLOOR)

            rhs_x_corr = rhs_x_base + sigma_mu * inv_sx_l - sigma_mu * inv_sx_u
            rhs_x_corr = rhs_x_corr - cross_x
            if m > 0:
                rhs_x_corr = rhs_x_corr - J.T @ y
                rhs_y_corr = rhs_y_base + sigma_mu * inv_z_s_ub - sigma_mu * inv_z_s_lb
            else:
                rhs_y_corr = jnp.zeros(0, dtype=jnp.float64)

            dx, dy, _ = _solve_kkt(final_dw, rhs_x_corr, rhs_y_corr)
            mu_target = sigma_mu
        else:
            dx, dy, _ = _solve_kkt(final_dw, rhs_x, rhs_y)
            mu_target = mu

        # --- Recover bound dual steps ---
        dz_l = pd.has_lb * ((mu_target - z_l * (sx_l + dx)) / jnp.maximum(sx_l, _SLACK_FLOOR))
        dz_u = pd.has_ub * ((mu_target - z_u * (sx_u - dx)) / jnp.maximum(sx_u, _SLACK_FLOOR))

        # --- Fraction-to-boundary step sizes ---
        alpha_x = jnp.array(1.0)
        alpha_x = jnp.minimum(
            alpha_x,
            _fraction_to_boundary(
                jnp.where(pd.has_lb > 0.5, sx_l, 1.0),
                jnp.where(pd.has_lb > 0.5, dx, 0.0),
                tau,
            ),
        )
        alpha_x = jnp.minimum(
            alpha_x,
            _fraction_to_boundary(
                jnp.where(pd.has_ub > 0.5, sx_u, 1.0),
                jnp.where(pd.has_ub > 0.5, -dx, 0.0),
                tau,
            ),
        )

        alpha_z = jnp.array(1.0)
        alpha_z = jnp.minimum(
            alpha_z,
            _fraction_to_boundary(
                jnp.where(pd.has_lb > 0.5, z_l, 1.0),
                jnp.where(pd.has_lb > 0.5, dz_l, 0.0),
                tau,
            ),
        )
        alpha_z = jnp.minimum(
            alpha_z,
            _fraction_to_boundary(
                jnp.where(pd.has_ub > 0.5, z_u, 1.0),
                jnp.where(pd.has_ub > 0.5, dz_u, 0.0),
                tau,
            ),
        )

        # --- Filter line search ---
        theta = (
            float(_total_violation(g, pd.g_l, pd.g_u, pd.has_g_lb, pd.has_g_ub)) if m > 0 else 0.0
        )

        # Barrier objective: f(x) - mu * sum(log(bound slacks))
        def _barrier_obj(x_eval):
            """Compute barrier objective at a point."""
            fv = float(obj_fn(_np(x_eval)))
            sl = jnp.maximum(x_eval - pd.x_l, _SLACK_FLOOR) * pd.has_lb
            su = jnp.maximum(pd.x_u - x_eval, _SLACK_FLOOR) * pd.has_ub
            log_sl = jnp.where(pd.has_lb > 0.5, jnp.log(jnp.maximum(sl, _EPS)), 0.0)
            log_su = jnp.where(pd.has_ub > 0.5, jnp.log(jnp.maximum(su, _EPS)), 0.0)
            fv -= float(mu) * float(jnp.sum(log_sl) + jnp.sum(log_su))
            return fv

        phi_barrier = _barrier_obj(x)

        # Directional derivative of barrier objective
        grad_barrier = grad_f - mu * inv_sx_l + mu * inv_sx_u
        grad_barrier_T_dx = float(jnp.dot(grad_barrier, dx))

        # Switching condition: f-type vs h-type
        is_f_type = False
        if m > 0 and grad_barrier_T_dx < 0 and theta <= theta_min:
            neg_deriv = -grad_barrier_T_dx
            if neg_deriv > 0:
                is_f_type = (
                    float(alpha_x) * neg_deriv**opts.s_phi > opts.delta_switch * theta**opts.s_theta
                )

        # For unconstrained: always use Armijo on barrier objective
        if m == 0:
            is_f_type = True

        # Backtracking with filter acceptance
        alpha_ls = float(alpha_x)
        armijo_holds = False
        for _ in range(opts.max_ls_iter):
            x_t = x + alpha_ls * dx
            f_t = float(obj_fn(_np(x_t)))
            if m > 0:
                g_t = jnp.asarray(con_fn(_np(x_t)), dtype=jnp.float64)
                theta_t = float(_total_violation(g_t, pd.g_l, pd.g_u, pd.has_g_lb, pd.has_g_ub))
            else:
                theta_t = 0.0

            # Reject if theta exceeds upper bound
            if theta_t > theta_max:
                alpha_ls *= 0.5
                if alpha_ls < 1e-16:
                    break
                continue

            # Check against filter (uses f(x), not barrier)
            if m > 0 and not _is_acceptable_to_filter(f_t, theta_t, filter_pairs):
                alpha_ls *= 0.5
                if alpha_ls < 1e-16:
                    break
                continue

            if is_f_type or theta < 1e-10:
                # F-type or effectively unconstrained: Armijo on barrier
                # When currently feasible, don't allow going infeasible
                if theta < opts.tol and theta_t > opts.tol:
                    alpha_ls *= 0.5
                    if alpha_ls < 1e-16:
                        break
                    continue
                phi_t = _barrier_obj(x_t)
                armijo_rhs = phi_barrier + opts.eta_phi * alpha_ls * min(grad_barrier_T_dx, -1e-16)
                if phi_t <= armijo_rhs:
                    armijo_holds = True
                    break
            else:
                # H-type: sufficient reduction in EITHER theta OR phi
                if (
                    theta_t <= (1 - opts.gamma_theta) * theta
                    or f_t <= obj_val - opts.gamma_phi * theta
                ):
                    break

            alpha_ls *= 0.5
            if alpha_ls < 1e-16:
                break

        # --- Second-order correction (SOC) for Maratos effect ---
        if alpha_ls < 0.1 * float(alpha_x) and m > 0:
            x_soc_trial = x + float(alpha_x) * dx
            x_soc_trial = jnp.where(
                pd.has_lb > 0.5,
                jnp.maximum(x_soc_trial, pd.x_l + _SLACK_FLOOR),
                x_soc_trial,
            )
            x_soc_trial = jnp.where(
                pd.has_ub > 0.5,
                jnp.minimum(x_soc_trial, pd.x_u - _SLACK_FLOOR),
                x_soc_trial,
            )
            c_soc = jnp.asarray(con_fn(_np(x_soc_trial)), dtype=jnp.float64)

            s_soc_lb = jnp.maximum(c_soc - pd.g_l, jnp.maximum(mu, _EPS))
            s_soc_ub = jnp.maximum(pd.g_u - c_soc, jnp.maximum(mu, _EPS))
            rhs_eq_soc = pd.is_eq * (-(c_soc - pd.g_l))
            rhs_ub_soc = pd.has_g_ub * ineq * (pd.g_u - c_soc - s_soc_ub) + mu * inv_z_s_ub
            rhs_lb_soc = pd.has_g_lb * ineq * (c_soc - pd.g_l - s_soc_lb) + mu * inv_z_s_lb
            rhs_y_soc = rhs_eq_soc + rhs_ub_soc - rhs_lb_soc

            dx_soc, dy_soc, _ = _solve_kkt(final_dw, rhs_x, rhs_y_soc)

            # Fraction-to-boundary for SOC direction
            sx_l_soc = jnp.maximum(x - pd.x_l, _SLACK_FLOOR) * pd.has_lb
            sx_u_soc = jnp.maximum(pd.x_u - x, _SLACK_FLOOR) * pd.has_ub
            alpha_soc = jnp.array(1.0)
            alpha_soc = jnp.minimum(
                alpha_soc,
                _fraction_to_boundary(
                    jnp.where(pd.has_lb > 0.5, sx_l_soc, 1.0),
                    jnp.where(pd.has_lb > 0.5, dx_soc, 0.0),
                    tau,
                ),
            )
            alpha_soc = jnp.minimum(
                alpha_soc,
                _fraction_to_boundary(
                    jnp.where(pd.has_ub > 0.5, sx_u_soc, 1.0),
                    jnp.where(pd.has_ub > 0.5, -dx_soc, 0.0),
                    tau,
                ),
            )

            # Line search SOC with filter acceptance
            alpha_soc_ls = float(alpha_soc)
            soc_accepted = False
            for _ in range(opts.max_ls_iter):
                x_t = x + alpha_soc_ls * dx_soc
                f_t = float(obj_fn(_np(x_t)))
                g_t = jnp.asarray(con_fn(_np(x_t)), dtype=jnp.float64)
                theta_t = float(_total_violation(g_t, pd.g_l, pd.g_u, pd.has_g_lb, pd.has_g_ub))

                if theta_t > theta_max:
                    alpha_soc_ls *= 0.5
                    if alpha_soc_ls < 1e-16:
                        break
                    continue
                if not _is_acceptable_to_filter(f_t, theta_t, filter_pairs):
                    alpha_soc_ls *= 0.5
                    if alpha_soc_ls < 1e-16:
                        break
                    continue

                if is_f_type:
                    grad_soc_T_dx = float(jnp.dot(grad_barrier, dx_soc))
                    phi_t = _barrier_obj(x_t)
                    if phi_t <= phi_barrier + opts.eta_phi * alpha_soc_ls * grad_soc_T_dx:
                        soc_accepted = True
                        break
                else:
                    if (
                        theta_t <= (1 - opts.gamma_theta) * theta
                        or f_t <= obj_val - opts.gamma_phi * theta
                    ):
                        soc_accepted = True
                        break
                alpha_soc_ls *= 0.5
                if alpha_soc_ls < 1e-16:
                    break

            if soc_accepted and alpha_soc_ls > alpha_ls * 10.0:
                alpha_ls = alpha_soc_ls
                dx = dx_soc
                dy = dy_soc
                dz_l = pd.has_lb * (
                    (mu_target - z_l * (sx_l + dx)) / jnp.maximum(sx_l, _SLACK_FLOOR)
                )
                dz_u = pd.has_ub * (
                    (mu_target - z_u * (sx_u - dx)) / jnp.maximum(sx_u, _SLACK_FLOOR)
                )
                alpha_z = jnp.array(1.0)
                alpha_z = jnp.minimum(
                    alpha_z,
                    _fraction_to_boundary(
                        jnp.where(pd.has_lb > 0.5, z_l, 1.0),
                        jnp.where(pd.has_lb > 0.5, dz_l, 0.0),
                        tau,
                    ),
                )
                alpha_z = jnp.minimum(
                    alpha_z,
                    _fraction_to_boundary(
                        jnp.where(pd.has_ub > 0.5, z_u, 1.0),
                        jnp.where(pd.has_ub > 0.5, dz_u, 0.0),
                        tau,
                    ),
                )

        # Augment filter for h-type steps
        if m > 0 and (not is_f_type or not armijo_holds):
            filter_pairs.append(
                (
                    (1 - opts.gamma_theta) * theta,
                    obj_val - opts.gamma_phi * theta,
                )
            )

        # --- Stall detection and recovery ---
        ls_failed = alpha_ls < 1e-10
        if ls_failed:
            stall_count += 1
        else:
            stall_count = 0

        do_recovery = stall_count >= 3

        if do_recovery:
            # Projected steepest descent with filter acceptance
            alpha_sd = 0.1
            for _ in range(20):
                x_t = x - alpha_sd * grad_f
                x_t = jnp.where(pd.has_lb > 0.5, jnp.maximum(x_t, pd.x_l + _SLACK_FLOOR), x_t)
                x_t = jnp.where(pd.has_ub > 0.5, jnp.minimum(x_t, pd.x_u - _SLACK_FLOOR), x_t)
                f_t = float(obj_fn(_np(x_t)))
                if m > 0:
                    g_t = jnp.asarray(con_fn(_np(x_t)), dtype=jnp.float64)
                    theta_t = float(_total_violation(g_t, pd.g_l, pd.g_u, pd.has_g_lb, pd.has_g_ub))
                else:
                    theta_t = 0.0
                # Accept if improves either objective or feasibility
                if theta_t < theta * 0.9 or f_t < obj_val * 0.99:
                    break
                alpha_sd *= 0.5
                if alpha_sd < 1e-8:
                    break

            sd_found = alpha_sd > 1e-8
            if sd_found:
                x_new = x - alpha_sd * grad_f
                x_new = jnp.where(pd.has_lb > 0.5, jnp.maximum(x_new, pd.x_l + _SLACK_FLOOR), x_new)
                x_new = jnp.where(pd.has_ub > 0.5, jnp.minimum(x_new, pd.x_u - _SLACK_FLOOR), x_new)
                alpha_p = alpha_sd
                alpha_d = 0.0
            else:
                do_recovery = False

        if not do_recovery:
            x_new = x + alpha_ls * dx
            x_new = jnp.where(pd.has_lb > 0.5, jnp.maximum(x_new, pd.x_l + _SLACK_FLOOR), x_new)
            x_new = jnp.where(pd.has_ub > 0.5, jnp.minimum(x_new, pd.x_u - _SLACK_FLOOR), x_new)
            alpha_p = alpha_ls
            alpha_d = float(alpha_z)

        # Track progress: detect when the solution isn't changing meaningfully.
        # Use relative change in x scaled by max(1, ||x||) to handle variables
        # of different magnitudes.
        x_scale = max(1.0, float(jnp.max(jnp.abs(x))))
        x_change = float(jnp.max(jnp.abs(x_new - x))) / x_scale
        if x_change < opts.acceptable_tol:
            no_progress_count += 1
        else:
            no_progress_count = 0

        # --- Dual update ---
        z_l_new = jnp.maximum(z_l + alpha_d * dz_l, _EPS) * pd.has_lb
        z_u_new = jnp.maximum(z_u + alpha_d * dz_u, _EPS) * pd.has_ub

        if m > 0:
            y_new = y + alpha_d * dy
            ineq_mask = 1.0 - pd.is_eq
            y_new = jnp.where(ineq_mask > 0.5, jnp.maximum(y_new, _EPS), y_new)
        else:
            y_new = y

        # On recovery: reset bound multipliers from barrier condition
        if do_recovery:
            sx_l_rec = jnp.maximum(x_new - pd.x_l, _SLACK_FLOOR) * pd.has_lb
            sx_u_rec = jnp.maximum(pd.x_u - x_new, _SLACK_FLOOR) * pd.has_ub
            z_l_new = jnp.where(
                pd.has_lb > 0.5,
                mu / jnp.maximum(sx_l_rec, _SLACK_FLOOR),
                0.0,
            )
            z_u_new = jnp.where(
                pd.has_ub > 0.5,
                mu / jnp.maximum(sx_u_rec, _SLACK_FLOOR),
                0.0,
            )

        # Safeguard multipliers
        sx_l_new = jnp.maximum(x_new - pd.x_l, _SLACK_FLOOR) * pd.has_lb
        sx_u_new = jnp.maximum(pd.x_u - x_new, _SLACK_FLOOR) * pd.has_ub

        # Stationarity-based multiplier fix at bounds
        grad_f_upd = jnp.asarray(grad_fn(_np(x_new)), dtype=jnp.float64)
        grad_stat = grad_f_upd
        if m > 0:
            J_new = jnp.asarray(jac_fn(_np(x_new)), dtype=jnp.float64)
            grad_stat = grad_stat + J_new.T @ y_new

        z_u_stat = jnp.maximum(-(grad_stat - z_l_new), _EPS)
        z_l_stat = jnp.maximum(grad_stat + z_u_new, _EPS)
        at_ub = pd.has_ub * (sx_u_new <= _SLACK_FLOOR * 2.0).astype(jnp.float64)
        at_lb = pd.has_lb * (sx_l_new <= _SLACK_FLOOR * 2.0).astype(jnp.float64)
        both = at_lb * at_ub
        at_lb = at_lb * (1.0 - both)
        at_ub = at_ub * (1.0 - both)
        z_u_new = jnp.where(at_ub > 0.5, z_u_stat, z_u_new)
        z_l_new = jnp.where(at_lb > 0.5, z_l_stat, z_l_new)

        z_l_new = _safeguard_z(z_l_new, sx_l_new, mu, pd.has_lb, opts.kappa_sigma)
        z_u_new = _safeguard_z(z_u_new, sx_u_new, mu, pd.has_ub, opts.kappa_sigma)

        # --- Check convergence (compute errors before mu update) ---
        grad_L = grad_f_upd - z_l_new + z_u_new
        if m > 0:
            g_new = jnp.asarray(con_fn(_np(x_new)), dtype=jnp.float64)
            grad_L = grad_L + J_new.T @ y_new
            primal_inf = float(
                _constraint_violation(g_new, pd.g_l, pd.g_u, pd.has_g_lb, pd.has_g_ub)
            )
        else:
            primal_inf = 0.0

        dual_inf = float(jnp.max(jnp.abs(grad_L)))

        # --- Update mu (Loqo rule with monotone option) ---
        compl = jnp.sum(pd.has_lb * z_l_new * sx_l_new) + jnp.sum(pd.has_ub * z_u_new * sx_u_new)
        n_pairs = jnp.sum(pd.has_lb) + jnp.sum(pd.has_ub)
        avg_compl = compl / jnp.maximum(n_pairs, 1.0)
        mu_candidate = avg_compl / opts.mu_decrease_kappa
        if opts.mu_allow_increase:
            mu_new = mu_candidate
        else:
            mu_new = jnp.minimum(mu_candidate, mu)
        mu_new = jnp.maximum(mu_new, opts.mu_min)
        compl_inf = float(jnp.minimum(avg_compl, mu_new))

        # Reset filter when mu changes significantly (new barrier sub-problem)
        if float(mu_new) < 0.5 * float(mu):
            filter_pairs = []
            theta_cur = (
                float(_total_violation(g_new, pd.g_l, pd.g_u, pd.has_g_lb, pd.has_g_ub))
                if m > 0
                else 0.0
            )
            theta_max = opts.theta_max_fact * max(1.0, theta_cur)
            theta_min = opts.theta_min_fact * max(1.0, theta_cur)

        optimal = primal_inf <= opts.tol and dual_inf <= opts.tol and compl_inf <= opts.tol
        acceptable = (
            primal_inf <= opts.acceptable_tol
            and dual_inf <= opts.acceptable_tol
            and compl_inf <= opts.acceptable_tol
        )
        if acceptable:
            consecutive_acceptable += 1
        else:
            consecutive_acceptable = 0
        acc_conv = consecutive_acceptable >= opts.acceptable_iter

        iteration += 1
        # Stagnation convergence: if stalling for many iterations and the
        # solution is near-feasible, accept.  Use a relaxed feasibility
        # threshold (sqrt of acceptable_tol) since the Maratos effect can
        # prevent the solver from reaching tight tolerances near active bounds.
        stagnation = stall_count >= 10
        stag_tol = max(opts.acceptable_tol, opts.acceptable_tol**0.5)
        stag_conv = stagnation and primal_inf <= stag_tol and dual_inf <= stag_tol * 100

        # No-progress convergence: x hasn't changed meaningfully for many
        # iterations.  If the solution is near-feasible, it's converged.
        no_prog_conv = no_progress_count >= 15 and primal_inf <= stag_tol

        # Track slow feasibility progress
        if primal_inf < 0.9 * best_primal_inf:
            best_primal_inf = primal_inf
            iters_since_feas_improve = 0
        else:
            iters_since_feas_improve += 1

        # Infeasible stagnation: stuck at an infeasible point with no progress.
        # Try feasibility restoration before giving up.
        # Trigger on: (a) stall/no-progress as before, OR
        # (b) slow feasibility convergence (many iters without 10% improvement).
        infeas_stag_trigger = (
            not _in_restoration
            and m > 0
            and con_fn is not None
            and jac_fn is not None
            and primal_inf > stag_tol
            and restoration_attempts < max_restoration_attempts
            and (stall_count >= 10 or no_progress_count >= 15 or iters_since_feas_improve >= 50)
        )

        if infeas_stag_trigger:
            x_rest, rest_ok = _feasibility_restoration(
                con_fn,
                jac_fn,
                np.asarray(x),
                x_l,
                x_u,
                np.asarray(g_l),
                np.asarray(g_u),
                pd.has_g_lb,
                pd.has_g_ub,
                opts,
            )
            restoration_attempts += 1
            if rest_ok:
                # Warm-start from restored point
                x = jnp.asarray(x_rest, dtype=jnp.float64)
                x = _push_from_bounds(
                    x, pd.x_l, pd.x_u, pd.has_lb, pd.has_ub, opts.bound_push, opts.bound_frac
                )
                # Reset mu to allow barrier to re-converge
                mu = jnp.maximum(mu, jnp.array(1e-4, dtype=jnp.float64))
                # Reset bound multipliers from barrier condition
                sl_r = jnp.maximum(x - pd.x_l, _SLACK_FLOOR) * pd.has_lb
                su_r = jnp.maximum(pd.x_u - x, _SLACK_FLOOR) * pd.has_ub
                z_l = jnp.where(pd.has_lb > 0.5, mu / jnp.maximum(sl_r, _SLACK_FLOOR), 0.0)
                z_u = jnp.where(pd.has_ub > 0.5, mu / jnp.maximum(su_r, _SLACK_FLOOR), 0.0)
                # Re-init constraint multipliers via least-squares
                g0_r = jnp.asarray(grad_fn(_np(x)), dtype=jnp.float64)
                J0_r = jnp.asarray(jac_fn(_np(x)), dtype=jnp.float64)
                A_r = J0_r @ J0_r.T + 1e-12 * jnp.eye(m)
                b_r = -J0_r @ g0_r
                y = jnp.linalg.solve(A_r, b_r)
                # Reset stall counters, filter, and objective
                stall_count = 0
                no_progress_count = 0
                iters_since_feas_improve = 0
                best_primal_inf = float("inf")
                consecutive_acceptable = 0
                delta_w_last = jnp.array(0.0, dtype=jnp.float64)
                # Reset filter for post-restoration
                filter_pairs = []
                _g_rest = jnp.asarray(con_fn(_np(x)), dtype=jnp.float64)
                theta_rest = float(
                    _total_violation(_g_rest, pd.g_l, pd.g_u, pd.has_g_lb, pd.has_g_ub)
                )
                theta_max = opts.theta_max_fact * max(1.0, theta_rest)
                theta_min = opts.theta_min_fact * max(1.0, theta_rest)
                obj_val = float(obj_fn(_np(x)))
                continue

        # Hard infeasible stagnation: restoration exhausted or not applicable
        infeas_stag = primal_inf > stag_tol and (
            (stall_count >= 50 or no_progress_count >= 50)
            or (
                iters_since_feas_improve >= 100 and restoration_attempts >= max_restoration_attempts
            )
        )

        if optimal:
            converged = 1
        elif acc_conv or stag_conv or no_prog_conv:
            converged = 2
        elif infeas_stag or iteration >= opts.max_iter:
            converged = 3

        # Update state for next iteration
        x = x_new
        y = y_new
        z_l = z_l_new
        z_u = z_u_new
        mu = mu_new
        delta_w_last = final_dw
        obj_val = float(obj_fn(_np(x)))

    return IPMState(
        x=x,
        y=y,
        z_l=z_l,
        z_u=z_u,
        mu=mu,
        nu=nu,
        iteration=jnp.array(iteration, dtype=jnp.int32),
        converged=jnp.array(converged, dtype=jnp.int32),
        obj=jnp.array(obj_val, dtype=jnp.float64),
        consecutive_acceptable=jnp.array(consecutive_acceptable, dtype=jnp.int32),
        alpha_primal=jnp.array(alpha_p, dtype=jnp.float64),
        delta_w_last=delta_w_last,
        stall_count=jnp.array(stall_count, dtype=jnp.int32),
        primal_inf_prev=jnp.array(0.0, dtype=jnp.float64),
        infeas_count=jnp.array(0, dtype=jnp.int32),
    )


# ---------------------------------------------------------------------------
# High-level evaluator adapter
# ---------------------------------------------------------------------------


def solve_nlp_ipm_callbacks(
    evaluator,
    x0=None,
    constraint_bounds: Optional[list[tuple[float, float]]] = None,
    options: Optional[dict] = None,
):
    """Solve an NLP using the callback-based IPM with any evaluator.

    The evaluator must provide:
      - evaluate_objective(x) -> float
      - evaluate_gradient(x) -> (n,)
      - evaluate_lagrangian_hessian(x, obj_factor, y) -> (n,n)
      - evaluate_constraints(x) -> (m,)   (if constrained)
      - evaluate_jacobian(x) -> (m,n)     (if constrained)
      - n_variables, n_constraints, variable_bounds

    Returns:
        NLPResult with solution.
    """
    from discopt.solvers import NLPResult, SolveStatus

    m = evaluator.n_constraints
    lb, ub = evaluator.variable_bounds
    x_l = np.asarray(lb, dtype=np.float64)
    x_u = np.asarray(ub, dtype=np.float64)

    if x0 is None:
        lb_c = np.clip(lb, -100.0, 100.0)
        ub_c = np.clip(ub, -100.0, 100.0)
        x0 = 0.5 * (lb_c + ub_c)
    x0 = np.asarray(x0, dtype=np.float64)

    if constraint_bounds is not None:
        g_l = np.array([b[0] for b in constraint_bounds], dtype=np.float64)
        g_u = np.array([b[1] for b in constraint_bounds], dtype=np.float64)
    elif m > 0 and hasattr(evaluator, "constraint_bounds") and evaluator.constraint_bounds:
        g_l, g_u = evaluator.constraint_bounds
        g_l = np.asarray(g_l, dtype=np.float64)
        g_u = np.asarray(g_u, dtype=np.float64)
    else:
        g_l = None
        g_u = None

    ipm_opts = IPMOptions()
    if options:
        fields = {k: v for k, v in options.items() if k in IPMOptions._fields}
        if fields:
            ipm_opts = ipm_opts._replace(**fields)

    # Build callbacks
    obj_fn = evaluator.evaluate_objective
    grad_fn = evaluator.evaluate_gradient
    hess_fn = evaluator.evaluate_lagrangian_hessian
    con_fn = evaluator.evaluate_constraints if m > 0 else None
    jac_fn = evaluator.evaluate_jacobian if m > 0 else None

    t0 = time.perf_counter()
    state = ipm_solve_callbacks(
        obj_fn,
        grad_fn,
        hess_fn,
        con_fn,
        jac_fn,
        x0,
        x_l,
        x_u,
        g_l,
        g_u,
        ipm_opts,
    )
    wall_time = time.perf_counter() - t0

    conv = int(state.converged)
    if conv in (1, 2):
        status = SolveStatus.OPTIMAL
    elif conv == 3:
        feasible = True
        if m > 0 and con_fn is not None and g_l is not None and g_u is not None:
            g_final = np.asarray(con_fn(np.asarray(state.x)))
            viol_lb = np.maximum(g_l - g_final, 0.0)
            viol_ub = np.maximum(g_final - g_u, 0.0)
            max_viol = float(np.max(viol_lb + viol_ub))
            feasible = max_viol < 1e-4
        status = SolveStatus.OPTIMAL if feasible else SolveStatus.ITERATION_LIMIT
    else:
        status = SolveStatus.ERROR

    return NLPResult(
        status=status,
        x=np.asarray(state.x),
        objective=float(state.obj),
        multipliers=np.asarray(state.y) if m > 0 else None,
        iterations=int(state.iteration),
        wall_time=wall_time,
    )
