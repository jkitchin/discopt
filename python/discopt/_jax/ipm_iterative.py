"""
Tier 2 iterative IPM solver using lineax for large-scale NLP.

Extends the Tier 1 dense IPM (ipm.py) with iterative linear solvers
from the lineax library. For problems with 1K-50K variables where
forming and factoring the full KKT matrix is impractical.

Key features:
  - Matrix-free KKT operators via lineax.FunctionLinearOperator
  - CG solver on condensed normal equations (SPD system)
  - GMRES solver on full augmented KKT system (indefinite)
  - Warm-starting: reuse previous solution as initial guess
  - Inexact Newton: inner tolerance decreases as outer IPM converges
    (Eisenstat-Walker forcing sequence)
  - Falls back to existing PCG path when lineax is not installed

All data structures are NamedTuples for JAX pytree compatibility.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp

from discopt._jax.ipm import (
    IPMOptions,
    IPMState,
    ipm_solve,
)

# ---------------------------------------------------------------------------
# Optional lineax import
# ---------------------------------------------------------------------------

try:
    import lineax as lx

    HAS_LINEAX = True
except ImportError:
    lx = None  # type: ignore[assignment]
    HAS_LINEAX = False


# ---------------------------------------------------------------------------
# Inexact Newton forcing sequence
# ---------------------------------------------------------------------------


def compute_forcing_term(
    mu: jnp.ndarray,
    iteration: jnp.ndarray,
    base_tol: float = 1e-10,
) -> float:
    """Eisenstat-Walker forcing sequence for inexact Newton.

    The inner linear solver tolerance decreases as the outer IPM
    converges (mu -> 0). This avoids over-solving the linear system
    in early iterations when the barrier parameter is large.

    Returns a concrete float suitable for lineax solver construction.
    For use outside jax.lax.while_loop (concrete context only).

    eta_k = max(base_tol, min(0.1, sqrt(mu)))
    """
    mu_val = float(mu)
    eta = min(0.1, mu_val**0.5)
    return float(max(eta, base_tol))


# ---------------------------------------------------------------------------
# Matrix-free operators
# ---------------------------------------------------------------------------


def _make_condensed_matvec(
    W: jnp.ndarray,
    J: jnp.ndarray,
    D_reg_vec: jnp.ndarray,
) -> tuple[Callable, jnp.ndarray, jnp.ndarray]:
    """Create matvec for condensed normal equations: (J W^{-1} J^T + D) v.

    This system is SPD when W is SPD, making it suitable for CG.
    """
    # For the condensed approach, we need W^{-1}. For medium-scale,
    # we compute this via Cholesky. For truly large-scale, we could
    # use an inner iterative solve, but that's beyond Tier 2.
    W_inv = jnp.linalg.solve(W, jnp.eye(W.shape[0]))
    JWinv = J @ W_inv

    def matvec(v):
        return JWinv @ (J.T @ v) + D_reg_vec * v

    return matvec, W_inv, JWinv


def _make_kkt_matvec(
    W: jnp.ndarray,
    J: jnp.ndarray,
    D_reg_vec: jnp.ndarray,
    n: int,
    m: int,
) -> Callable:
    """Create matvec for full augmented KKT system (indefinite).

    [W,    J^T] [vx]
    [J,   -D  ] [vy]
    """

    def matvec(v):
        vx = v[:n]
        vy = v[n:]
        top = W @ vx + J.T @ vy
        bot = J @ vx - D_reg_vec * vy
        return jnp.concatenate([top, bot])

    return matvec


# ---------------------------------------------------------------------------
# IterativeKKTSolver
# ---------------------------------------------------------------------------


class IterativeKKTSolver:
    """Solve the augmented KKT system using lineax iterative methods.

    For CG: uses the condensed normal equations approach
        (J W^{-1} J^T + D) dy = J W^{-1} rhs_x - rhs_y
    which produces an SPD system amenable to CG.

    For GMRES: solves the full indefinite augmented KKT system directly.

    Args:
        linear_solver: "cg" or "gmres".
        rtol: Relative tolerance for lineax solver.
        atol: Absolute tolerance for lineax solver.
        max_steps: Maximum iterations for the inner linear solve.
        warm_start: Whether to use previous solution as initial guess.
        use_preconditioner: Whether to apply Jacobi preconditioning.
    """

    def __init__(
        self,
        linear_solver: str = "cg",
        rtol: float = 1e-8,
        atol: float = 1e-8,
        max_steps: int = 1000,
        warm_start: bool = True,
        use_preconditioner: bool = True,
    ):
        if not HAS_LINEAX:
            raise ImportError(
                "lineax is required for IterativeKKTSolver. Install with: pip install lineax"
            )
        self.linear_solver = linear_solver
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps
        self.warm_start = warm_start
        self.use_preconditioner = use_preconditioner

    def solve(
        self,
        H: jnp.ndarray,
        J: jnp.ndarray,
        Sig_diag: jnp.ndarray,
        D_reg_vec: jnp.ndarray,
        delta_w: jnp.ndarray,
        delta_c: float,
        rhs_x: jnp.ndarray,
        rhs_y: jnp.ndarray,
        prev_sol: Optional[jnp.ndarray] = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, object]]:
        """Solve the KKT system iteratively.

        Args:
            H: (n, n) Hessian of Lagrangian.
            J: (m, n) constraint Jacobian.
            Sig_diag: (n,) barrier Hessian diagonal.
            D_reg_vec: (m,) condensed slack diagonal.
            delta_w: scalar primal regularization.
            delta_c: scalar dual regularization.
            rhs_x: (n,) primal RHS.
            rhs_y: (m,) dual RHS.
            prev_sol: Previous solution for warm-starting.

        Returns:
            (dx, dy, info) where info has 'num_steps' and 'converged'.
        """
        n = rhs_x.shape[0]
        m = rhs_y.shape[0]

        # Build W = H + diag(Sigma) + delta_w * I
        W = H + jnp.diag(Sig_diag) + delta_w * jnp.eye(n)
        D_full = D_reg_vec + delta_c

        if self.linear_solver == "gmres":
            result: tuple[jnp.ndarray, jnp.ndarray, dict[str, object]] = self._solve_gmres(
                W, J, D_full, rhs_x, rhs_y, n, m, prev_sol
            )
            return result
        else:
            result2: tuple[jnp.ndarray, jnp.ndarray, dict[str, object]] = self._solve_cg_condensed(
                W,
                J,
                D_full,
                rhs_x,
                rhs_y,
                n,
                m,
                prev_sol,
            )
            return result2

    def _solve_cg_condensed(
        self,
        W,
        J,
        D_full,
        rhs_x,
        rhs_y,
        n,
        m,
        prev_sol,
    ):
        """Solve via condensed normal equations + CG."""
        if m == 0:
            # Unconstrained: solve W dx = rhs_x directly with CG
            return self._solve_cg_unconstrained(W, rhs_x, n, prev_sol)

        # Condensed approach: (J W^{-1} J^T + D) dy = J W^{-1} rhs_x - rhs_y
        matvec, W_inv, JWinv = _make_condensed_matvec(W, J, D_full)
        rhs_condensed = JWinv @ rhs_x - rhs_y

        input_struct = jax.ShapeDtypeStruct((m,), rhs_condensed.dtype)
        op = lx.FunctionLinearOperator(
            matvec,
            input_struct,
            tags=lx.positive_semidefinite_tag,
        )
        solver = lx.CG(
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
        )

        options: dict = {}
        if self.warm_start and prev_sol is not None:
            # prev_sol is dy from previous iteration
            options["y0"] = prev_sol
        if self.use_preconditioner:
            # Diagonal preconditioner for condensed system
            S_diag = (
                jnp.array([jnp.dot(JWinv[i], J[i]) + D_full[i] for i in range(m)])
                if m <= 1000
                else jnp.sum(JWinv * J, axis=1) + D_full
            )
            S_diag_safe = jnp.where(jnp.abs(S_diag) > 1e-30, S_diag, 1.0)
            precond_diag = 1.0 / S_diag_safe
            precond_fn = lambda v: precond_diag * v  # noqa: E731
            precond_struct = jax.ShapeDtypeStruct((m,), rhs_condensed.dtype)
            precond_op = lx.FunctionLinearOperator(
                precond_fn,
                precond_struct,
                tags=lx.positive_semidefinite_tag,
            )
            options["preconditioner"] = precond_op

        sol = lx.linear_solve(
            op,
            rhs_condensed,
            solver,
            options=options if options else None,
            throw=False,
        )
        dy = sol.value
        # Back-substitute: dx = W^{-1} (rhs_x - J^T dy)
        dx = W_inv @ (rhs_x - J.T @ dy)

        info = {
            "num_steps": sol.stats.get("num_steps", 0),
            "converged": sol.result == lx.RESULTS.successful,
        }
        return dx, dy, info

    def _solve_cg_unconstrained(self, W, rhs_x, n, prev_sol):
        """Solve W dx = rhs_x with CG for unconstrained case."""
        input_struct = jax.ShapeDtypeStruct((n,), rhs_x.dtype)
        op = lx.FunctionLinearOperator(
            lambda v: W @ v,
            input_struct,
            tags=lx.positive_semidefinite_tag,
        )
        solver = lx.CG(
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
        )

        options: dict = {}
        if self.warm_start and prev_sol is not None:
            options["y0"] = prev_sol
        if self.use_preconditioner:
            diag_W = jnp.diag(W)
            diag_safe = jnp.where(jnp.abs(diag_W) > 1e-30, diag_W, 1.0)
            precond_fn = lambda v: v / diag_safe  # noqa: E731
            precond_struct = jax.ShapeDtypeStruct((n,), rhs_x.dtype)
            precond_op = lx.FunctionLinearOperator(
                precond_fn,
                precond_struct,
                tags=lx.positive_semidefinite_tag,
            )
            options["preconditioner"] = precond_op

        sol = lx.linear_solve(
            op,
            rhs_x,
            solver,
            options=options if options else None,
            throw=False,
        )
        info = {
            "num_steps": sol.stats.get("num_steps", 0),
            "converged": sol.result == lx.RESULTS.successful,
        }
        return sol.value, jnp.zeros(0, dtype=rhs_x.dtype), info

    def _solve_gmres(self, W, J, D_full, rhs_x, rhs_y, n, m, prev_sol):
        """Solve full indefinite KKT system with GMRES."""
        if m == 0:
            # Unconstrained: solve W dx = rhs_x with GMRES
            input_struct = jax.ShapeDtypeStruct((n,), rhs_x.dtype)
            op = lx.FunctionLinearOperator(
                lambda v: W @ v,
                input_struct,
            )
            solver = lx.GMRES(
                rtol=self.rtol,
                atol=self.atol,
                max_steps=self.max_steps,
            )
            options: dict = {}
            if self.warm_start and prev_sol is not None:
                options["y0"] = prev_sol
            sol = lx.linear_solve(
                op,
                rhs_x,
                solver,
                options=options if options else None,
                throw=False,
            )
            info = {
                "num_steps": sol.stats.get("num_steps", 0),
                "converged": sol.result == lx.RESULTS.successful,
            }
            return sol.value, jnp.zeros(0, dtype=rhs_x.dtype), info

        matvec = _make_kkt_matvec(W, J, D_full, n, m)
        rhs = jnp.concatenate([rhs_x, rhs_y])
        input_struct = jax.ShapeDtypeStruct((n + m,), rhs.dtype)
        op = lx.FunctionLinearOperator(matvec, input_struct)

        solver = lx.GMRES(
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
        )
        options_dict: dict = {}
        if self.warm_start and prev_sol is not None:
            options_dict["y0"] = prev_sol

        sol = lx.linear_solve(
            op,
            rhs,
            solver,
            options=options_dict if options_dict else None,
            throw=False,
        )
        dx = sol.value[:n]
        dy = sol.value[n:]
        info = {
            "num_steps": sol.stats.get("num_steps", 0),
            "converged": sol.result == lx.RESULTS.successful,
        }
        return dx, dy, info


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def solve_nlp_iterative(
    obj_fn: Callable,
    con_fn: Optional[Callable],
    x0: jnp.ndarray,
    x_l: jnp.ndarray,
    x_u: jnp.ndarray,
    g_l: Optional[jnp.ndarray] = None,
    g_u: Optional[jnp.ndarray] = None,
    options: Optional[IPMOptions] = None,
) -> IPMState:
    """Solve NLP using iterative (Tier 2) IPM.

    For problems with 1K-50K variables. Uses lineax CG by default,
    falling back to the existing PCG path when lineax is unavailable.

    Args:
        obj_fn: Scalar objective f(x) -> scalar.
        con_fn: Constraint function g(x) -> (m,) array, or None.
        x0: Initial point (n,).
        x_l: Lower variable bounds (n,). Use -1e20 for unbounded.
        x_u: Upper variable bounds (n,). Use 1e20 for unbounded.
        g_l: Lower constraint bounds (m,).
        g_u: Upper constraint bounds (m,).
        options: IPMOptions. If linear_solver is not set, defaults to
            "lineax_cg" when lineax is available, else "pcg".

    Returns:
        Final IPMState with solution in state.x.
    """
    if options is None:
        options = IPMOptions()

    # Select appropriate linear solver
    if options.linear_solver == "dense":
        if HAS_LINEAX:
            options = options._replace(linear_solver="lineax_cg")
        else:
            options = options._replace(linear_solver="pcg")

    return ipm_solve(obj_fn, con_fn, x0, x_l, x_u, g_l, g_u, options)


def solve_nlp_iterative_batch(
    obj_fn: Callable,
    con_fn: Optional[Callable],
    x0_batch: jnp.ndarray,
    xl_batch: jnp.ndarray,
    xu_batch: jnp.ndarray,
    g_l: Optional[jnp.ndarray] = None,
    g_u: Optional[jnp.ndarray] = None,
    options: Optional[IPMOptions] = None,
) -> IPMState:
    """Batched version of solve_nlp_iterative using jax.vmap.

    All instances share obj_fn/con_fn/g_l/g_u but have per-instance
    x0 and variable bounds.

    Args:
        obj_fn: Scalar objective f(x) -> scalar.
        con_fn: Constraint function g(x) -> (m,) or None.
        x0_batch: Initial points (batch, n).
        xl_batch: Lower variable bounds (batch, n).
        xu_batch: Upper variable bounds (batch, n).
        g_l: Lower constraint bounds (m,) -- shared.
        g_u: Upper constraint bounds (m,) -- shared.
        options: IPMOptions.

    Returns:
        IPMState with batched arrays (batch, ...).
    """
    if options is None:
        options = IPMOptions()

    if options.linear_solver == "dense":
        if HAS_LINEAX:
            options = options._replace(linear_solver="lineax_cg")
        else:
            options = options._replace(linear_solver="pcg")

    def _solve_single(x0_single, xl_single, xu_single):
        return ipm_solve(
            obj_fn,
            con_fn,
            x0_single,
            xl_single,
            xu_single,
            g_l,
            g_u,
            options,
            check_deadline=False,  # host callback unsupported under vmap (#80)
        )

    return jax.vmap(_solve_single)(  # type: ignore[no-any-return]
        x0_batch,
        xl_batch,
        xu_batch,
    )
