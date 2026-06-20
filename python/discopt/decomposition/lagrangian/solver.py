"""Lagrangian relaxation solver (v1: linear (mixed-integer) models).

Given (internally minimizing)

    min  c^T z   s.t.  coupling rows  A_c z <= r_c,   block rows  A_b z <= r_b,
                       z in bounds, some z integer,

dualize the coupling rows with multipliers ``λ >= 0`` to form the relaxation

    L(λ) = min_z  (c + A_c^T λ)^T z - λ^T r_c   s.t.  block rows, bounds, integrality.

For every ``λ >= 0`` this is a **valid lower bound** on the optimum (the dualized
term is ``<= 0`` at any feasible point). The Lagrangian dual ``max_{λ>=0} L(λ)``
is maximized by:

- **subgradient** ascent (Polyak step with target-level halving), or
- a **bundle** / cutting-plane method (Kelley): a piecewise-linear model of the
  concave ``L`` maximized over a box.

A Lagrangian heuristic recovers a primal incumbent by fixing the integer
variables at the relaxed solution and re-solving the full LP. The reported
``bound`` is the best ``L(λ)`` (rigorous, using the subproblem's dual bound),
and ``objective`` is the best recovered feasible value.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from discopt.decomposition._linear import extract_linear, solution_dict
from discopt.decomposition.structure import (
    DecompositionStructure,
    detect_decomposition,
    flat_bounds,
    restricted_bounds,
)
from discopt.modeling.core import Model, SolveResult, VarType
from discopt.solvers import SolveStatus

logger = logging.getLogger(__name__)

_BIG = 1e20
_LAMBDA_MAX = 1e6


@dataclass
class LagrangianConfig:
    """Configuration for the Lagrangian relaxation solver."""

    time_limit: float = 3600.0
    gap_tolerance: float = 1e-4
    max_iterations: int = 200
    method: str = "subgradient"  # or "bundle"
    prefer_pounce: bool = False
    initial_step: float = 1.5  # Polyak alpha
    patience: int = 8  # non-improving iters before halving alpha
    recover_every: int = 1  # run primal recovery every k iterations


def solve_lagrangian(
    model: Model,
    *,
    structure: DecompositionStructure | None = None,
    time_limit: float = 3600.0,
    gap_tolerance: float = 1e-4,
    max_iterations: int = 200,
    method: str = "subgradient",
    nlp_solver: str = "pounce",
    config: LagrangianConfig | None = None,
    **_ignored,
) -> SolveResult:
    """Solve a linear (mixed-integer) model by Lagrangian relaxation.

    Parameters
    ----------
    model : Model
        Linear-constraint, linear-objective model.
    structure : DecompositionStructure, optional
        Decomposition structure; auto-detected when omitted. Coupling
        constraints can be annotated with ``model.mark_coupling(...)``.
    method : {"subgradient", "bundle"}
        Dual maximization method.

    Returns
    -------
    SolveResult
        ``bound`` is the rigorous Lagrangian dual lower bound; ``objective`` is
        the best feasible value found by the recovery heuristic (if any).
    """
    from discopt.solvers.lp_backend import get_lp_solver, get_milp_solver

    cfg = config or LagrangianConfig(
        time_limit=time_limit,
        gap_tolerance=gap_tolerance,
        max_iterations=max_iterations,
        method=method,
        prefer_pounce=(nlp_solver == "pounce"),
    )
    if cfg.method not in ("subgradient", "bundle"):
        raise ValueError(f"Unknown method={cfg.method!r}; choose 'subgradient' or 'bundle'.")
    t0 = time.time()

    milp = get_milp_solver(prefer_pounce=cfg.prefer_pounce)
    lp = get_lp_solver(prefer_pounce=cfg.prefer_pounce)

    if structure is None:
        structure = detect_decomposition(model)

    lin = extract_linear(model)
    n = lin.n
    coupling_src = set(structure.coupling_constraints)
    if not coupling_src:
        raise NotImplementedError(
            "Lagrangian relaxation found no coupling constraints to dualize. "
            "Annotate them with model.mark_coupling(...), or use Model.solve()."
        )

    # Partition rows into coupling (dualized) and block (kept).
    Ac_rows, rc_rows = [], []
    Ab_rows, rb_rows = [], []
    for vec, rhs, src in zip(lin.rows_coeff, lin.rows_rhs, lin.rows_source):
        if src in coupling_src:
            Ac_rows.append(vec)
            rc_rows.append(rhs)
        else:
            Ab_rows.append(vec)
            rb_rows.append(rhs)
    A_c = np.array(Ac_rows) if Ac_rows else np.zeros((0, n))
    r_c = np.array(rc_rows) if rc_rows else np.zeros(0)
    A_b = np.array(Ab_rows) if Ab_rows else np.zeros((0, n))
    r_b = np.array(rb_rows) if rb_rows else np.zeros(0)
    m_coup = A_c.shape[0]

    lb_all, ub_all = flat_bounds(model)
    bounds = [(float(lb_all[i]), float(ub_all[i])) for i in range(n)]
    integrality = np.zeros(n, dtype=np.int32)
    off = 0
    for v in model._variables:
        if v.var_type in (VarType.BINARY, VarType.INTEGER):
            for i in range(v.size):
                integrality[off + i] = 1
        off += v.size

    def _subproblem(lam: np.ndarray):
        """Return (L_bound, z, residual) for the relaxation at multipliers lam."""
        c_lag = lin.c + (A_c.T @ lam if m_coup else np.zeros(n))
        res = milp(
            c_lag,
            A_ub=A_b if A_b.shape[0] else None,
            b_ub=r_b if A_b.shape[0] else None,
            bounds=bounds,
            integrality=integrality,
            time_limit=max(1.0, cfg.time_limit - (time.time() - t0)),
            gap_tolerance=cfg.gap_tolerance,
        )
        if res.x is None:
            return None, None, None
        z = np.asarray(res.x, dtype=np.float64)
        # Rigorous lower bound on the relaxed subproblem objective.
        sub_lb = (
            res.bound
            if res.bound is not None
            else (res.objective if res.status == SolveStatus.OPTIMAL else None)
        )
        if sub_lb is None:
            return None, z, None
        L = float(sub_lb) - float(lam @ r_c) + lin.c_offset
        residual = (A_c @ z - r_c) if m_coup else np.zeros(0)
        return L, z, residual

    def _recover(z: np.ndarray):
        """Fix integers at z, solve the full LP, return (UB, z_full) or (None, None)."""
        fixed: dict[str, float | np.ndarray] = {}
        offp = 0
        for v in model._variables:
            if v.var_type in (VarType.BINARY, VarType.INTEGER):
                vals = np.round(z[offp : offp + v.size])
                fixed[v.name] = vals
            offp += v.size
        flb, fub = restricted_bounds(model, fixed) if fixed else (lb_all.copy(), ub_all.copy())
        rbnds = [(float(flb[i]), float(fub[i])) for i in range(n)]
        A_all = np.vstack([A_b, A_c]) if (A_b.shape[0] or A_c.shape[0]) else None
        b_all = np.concatenate([r_b, r_c]) if (A_b.shape[0] or A_c.shape[0]) else None
        res = lp(lin.c, A_ub=A_all, b_ub=b_all, bounds=rbnds)
        if res.status != SolveStatus.OPTIMAL or res.x is None:
            return None, None
        z_full = np.asarray(res.x, dtype=np.float64)
        ub = float(lin.c @ z_full) + lin.c_offset
        return ub, z_full

    # ── dual maximization ──
    lam = np.zeros(m_coup)
    best_L = -np.inf
    best_UB = np.inf
    incumbent: np.ndarray | None = None
    alpha = cfg.initial_step
    stall = 0
    # Bundle model: list of (L_k, g_k, lam_k).
    cuts: list[tuple[float, np.ndarray, np.ndarray]] = []
    status = "iteration_limit"

    for it in range(cfg.max_iterations):
        if time.time() - t0 > cfg.time_limit:
            status = "time_limit"
            break
        L, z, residual = _subproblem(lam)
        if L is None or z is None:
            # Could not certify a bound at this lambda; stop.
            break
        if L > best_L:
            best_L = L
            stall = 0
        else:
            stall += 1

        if residual is not None and it % cfg.recover_every == 0:
            ub, z_full = _recover(z)
            if ub is not None and ub < best_UB:
                best_UB = ub
                incumbent = z_full

        if np.isfinite(best_UB) and best_L > -np.inf:
            gap = (best_UB - best_L) / (abs(best_UB) + 1e-10)
            if gap <= cfg.gap_tolerance:
                status = "optimal"
                break

        if m_coup == 0 or residual is None:
            break

        # ── next multipliers ──
        if cfg.method == "subgradient":
            gnorm2 = float(residual @ residual)
            if gnorm2 < 1e-16:
                status = "optimal"
                break
            target = best_UB if np.isfinite(best_UB) else best_L + max(1.0, abs(best_L))
            step = alpha * max(target - L, 1e-9) / gnorm2
            lam = np.maximum(0.0, lam + step * residual)
            if stall >= cfg.patience:
                alpha *= 0.5
                stall = 0
        else:  # bundle / cutting-plane (Kelley)
            cuts.append((L, residual.copy(), lam.copy()))
            lam = _bundle_step(lp, cuts, m_coup)

    # Promote to optimal if bounds already met.
    if status == "iteration_limit" and np.isfinite(best_UB) and best_L > -np.inf:
        if (best_UB - best_L) / (abs(best_UB) + 1e-10) <= cfg.gap_tolerance:
            status = "optimal"

    sense_flip = 1.0 if lin.minimize else -1.0
    objective = None if not np.isfinite(best_UB) else best_UB
    bound: float | None = None if best_L == -np.inf else best_L
    reported_obj = None if objective is None else objective * sense_flip
    reported_bound = None if bound is None else bound * sense_flip
    reported_gap: float | None = None
    if reported_obj is not None and reported_bound is not None:
        reported_gap = abs(reported_obj - reported_bound) / (abs(reported_obj) + 1e-10)
    x_dict = solution_dict(model, incumbent) if incumbent is not None else None
    if x_dict is None and objective is not None:
        status = "iteration_limit"  # have a bound but no recovered primal

    return SolveResult(
        status=status,
        objective=reported_obj,
        bound=reported_bound,
        gap=reported_gap,
        x=x_dict,
        wall_time=time.time() - t0,
        node_count=0,
        gap_certified=(status == "optimal"),
    )


def _bundle_step(lp, cuts: list[tuple[float, np.ndarray, np.ndarray]], m_coup: int) -> np.ndarray:
    """Maximize the cutting-plane model of L over the multiplier box [0, λmax].

    max_{λ∈[0,λmax], η}  η   s.t.  η <= L_k + g_k^T (λ - λ_k)  for each cut k.
    Encoded as an LP minimizing -η.
    """
    ncol = m_coup + 1  # [lam..., eta]
    c = np.zeros(ncol)
    c[-1] = -1.0
    rows, rhs = [], []
    for L_k, g_k, lam_k in cuts:
        row = np.zeros(ncol)
        row[:m_coup] = -g_k
        row[-1] = 1.0
        rows.append(row)
        rhs.append(float(L_k - g_k @ lam_k))
    A_ub = np.array(rows)
    b_ub = np.array(rhs)
    bounds = [(0.0, _LAMBDA_MAX)] * m_coup + [(-_BIG, _BIG)]
    res = lp(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    if res.x is None:
        return cuts[-1][2]  # fall back to last lambda
    return np.maximum(0.0, np.asarray(res.x[:m_coup], dtype=np.float64))
