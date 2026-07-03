"""Lagrangian relaxation solver (v1: linear (mixed-integer) models).

Given (internally minimizing)

    min  c^T z   s.t.  coupling rows  A_c z <= r_c,   block rows  A_b z <= r_b,
                       z in bounds, some z integer,

dualize the coupling rows with multipliers ``λ >= 0`` to form the relaxation

    L(λ) = min_z  (c + A_c^T λ)^T z - λ^T r_c   s.t.  block rows, bounds, integrality.

For every ``λ >= 0`` this is a **valid lower bound** on the optimum (the dualized
term is ``<= 0`` at any feasible point). The Lagrangian dual ``max_{λ>=0} L(λ)``
is maximized by:

- **subgradient** ascent (Polyak step with target-level halving),
- a **level bundle** method (``method="bundle"``; de Oliveira & Sagastizábal): the
  next iterate is the projection of the stability centre onto a level set of the
  cutting-plane model, solved as a small QP. Stabilized, with a reliable
  stopping test (``L̂* - L_best`` small) that plain subgradient lacks, or
- the plain **Kelley** cutting-plane (``method="kelley"``): the unstabilized
  piecewise-linear model maximized over a box (kept for comparison; it oscillates).

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

from discopt.decomposition._linear import extract_linear, relative_gap, solution_dict
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
    backend: str = "sequential"  # per-block execution: "sequential" | "threads"
    level_gamma: float = 0.5  # level parameter for the bundle method (T2.1)
    lambda_max0: float = 1e4  # initial multiplier box; grows x10 when hit


def solve_lagrangian(
    model: Model,
    *,
    structure: DecompositionStructure | None = None,
    time_limit: float = 3600.0,
    gap_tolerance: float = 1e-4,
    max_iterations: int = 200,
    method: str = "subgradient",
    nlp_solver: str = "pounce",
    backend: str = "sequential",
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
    backend : {"sequential", "threads"}
        Execution backend for the per-block relaxed subproblem solves. The
        relaxed subproblem separates across the detected blocks; ``"threads"``
        solves them concurrently (results reduced in block order, so the dual
        bound is identical to ``"sequential"``).

    Returns
    -------
    SolveResult
        ``bound`` is the rigorous Lagrangian dual lower bound; ``objective`` is
        the best feasible value found by the recovery heuristic (if any).
    """
    from discopt.solvers.lp_backend import get_lp_solver, get_milp_solver, get_qp_solver

    cfg = config or LagrangianConfig(
        time_limit=time_limit,
        gap_tolerance=gap_tolerance,
        max_iterations=max_iterations,
        method=method,
        prefer_pounce=(nlp_solver == "pounce"),
        backend=backend,
    )
    if cfg.method not in ("subgradient", "bundle", "kelley"):
        raise ValueError(
            f"Unknown method={cfg.method!r}; choose 'subgradient', 'bundle', or 'kelley'."
        )
    t0 = time.time()

    milp = get_milp_solver(prefer_pounce=cfg.prefer_pounce)
    lp = get_lp_solver(prefer_pounce=cfg.prefer_pounce)
    # The level-bundle projection is a small QP; fall back to Kelley if no QP
    # backend is installed (T2.1).
    qp = None
    if cfg.method == "bundle":
        try:
            qp = get_qp_solver(prefer_pounce=cfg.prefer_pounce)
        except ImportError:
            logger.warning(
                "method='bundle' needs a QP backend for the level projection; "
                "none available, falling back to the Kelley cutting-plane step."
            )

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

    # ── coupling rows (dualized) ─────────────────────────────────
    # Built from the *native* rows so an equality coupling constraint keeps a
    # single **free** multiplier (``free_mask``) instead of two nonnegative ones
    # — half the dual dimension (T1.2). ``>=`` rows are flipped to ``<=`` so
    # their multiplier stays ``>= 0``.
    dense = lin.dense()
    Ac_rows, rc_rows, free_rows = [], [], []
    for i, sense in enumerate(lin.sense):
        if lin.row_source[i] not in coupling_src:
            continue
        row = dense[i]
        bi = float(lin.b[i])
        if sense == ">=":
            Ac_rows.append(-row)
            rc_rows.append(-bi)
            free_rows.append(False)
        elif sense == "==":
            Ac_rows.append(row)
            rc_rows.append(bi)
            free_rows.append(True)
        else:  # "<="
            Ac_rows.append(row)
            rc_rows.append(bi)
            free_rows.append(False)
    A_c = np.array(Ac_rows) if Ac_rows else np.zeros((0, n))
    r_c = np.array(rc_rows) if rc_rows else np.zeros(0)
    free_mask = np.array(free_rows, dtype=bool)
    m_coup = A_c.shape[0]

    lb_all, ub_all = flat_bounds(model)
    integrality = np.zeros(n, dtype=np.int32)
    off = 0
    for v in model._variables:
        if v.var_type in (VarType.BINARY, VarType.INTEGER):
            integrality[off : off + v.size] = 1
        off += v.size

    # ── block partition (the relaxed subproblem separates across blocks) ──
    # Column blocks are the connected components of the non-coupling constraint
    # graph (every variable belongs to exactly one). Each non-coupling row lies
    # entirely in one block, so the relaxed subproblem
    # ``min (c + A_c^T λ)·z s.t. block rows`` decomposes into independent
    # per-block MILPs whose bounds sum to the monolithic relaxation bound.
    flat_of: dict[str, tuple[int, int]] = {}
    _o = 0
    for v in model._variables:
        flat_of[v.name] = (_o, v.size)
        _o += v.size
    num_blocks = max(1, structure.num_blocks)
    col_blocks: list[list[int]] = [[] for _ in range(num_blocks)]
    for v in model._variables:
        b = structure.block_of_var.get(v.name, 0)
        o, sz = flat_of[v.name]
        col_blocks[b].extend(range(o, o + sz))
    col_arrays = [np.array(cb, dtype=int) for cb in col_blocks]

    # Block ``<=`` rows (equalities expanded), each restricted to its block cols.
    A_leq, b_leq, src_leq = lin.rows_leq()
    blk_A_rows: list[list[np.ndarray]] = [[] for _ in range(num_blocks)]
    blk_r_rows: list[list[float]] = [[] for _ in range(num_blocks)]
    for vec, rhs, src in zip(A_leq, b_leq, src_leq):
        if src in coupling_src:
            continue
        b = structure.block_of_constraint[src]
        if b < 0:
            continue  # references no variable
        cols = col_arrays[b]
        support = np.nonzero(np.abs(vec) > 0)[0]
        assert set(support.tolist()) <= set(cols.tolist()), "non-coupling row spans blocks"
        blk_A_rows[b].append(vec[cols])
        blk_r_rows[b].append(rhs)

    block_A = [
        (np.array(rows) if rows else np.zeros((0, col_arrays[b].size)))
        for b, rows in enumerate(blk_A_rows)
    ]
    block_r = [np.array(rr) if rr else np.zeros(0) for rr in blk_r_rows]
    block_bounds = [[(float(lb_all[j]), float(ub_all[j])) for j in cols] for cols in col_arrays]
    block_intg = [integrality[cols] for cols in col_arrays]
    # Execute biggest blocks first (straggler avoidance); results reduce in this
    # fixed order regardless of backend, so the bound is backend-independent.
    exec_order = sorted(range(num_blocks), key=lambda b: (-col_arrays[b].size, b))

    from discopt.decomposition.parallel.comm import select_backend

    comm = select_backend(cfg.backend)

    def _solve_block(b: int):
        cols = col_arrays[b]
        if cols.size == 0:
            return (0.0, np.zeros(0), cols)
        cb = c_lag_global[cols]
        A_ub = block_A[b] if block_A[b].shape[0] else None
        b_ub = block_r[b] if block_A[b].shape[0] else None
        res = milp(
            cb,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=block_bounds[b],
            integrality=block_intg[b],
            time_limit=max(1.0, cfg.time_limit - (time.time() - t0)),
            gap_tolerance=cfg.gap_tolerance,
        )
        if res.x is None:
            return None
        sub_lb = (
            res.bound
            if res.bound is not None
            else (res.objective if res.status == SolveStatus.OPTIMAL else None)
        )
        if sub_lb is None:
            return None
        return (float(sub_lb), np.asarray(res.x, dtype=np.float64), cols)

    # ``c_lag_global`` is set per call before dispatching the block solves.
    c_lag_global = lin.c.copy()

    def _subproblem(lam: np.ndarray):
        """Return (L_bound, z, residual) for the relaxation at multipliers lam.

        Solves the per-block relaxed MILPs on the configured backend and sums
        their rigorous dual bounds. Returns ``(None, ...)`` if any block cannot
        certify a bound (same contract as the monolithic version)."""
        nonlocal c_lag_global
        c_lag_global = lin.c + (A_c.T @ lam if m_coup else np.zeros(n))
        results = comm.map(exec_order, _solve_block)
        if any(res is None for res in results):
            return None, None, None
        z = np.zeros(n, dtype=np.float64)
        sub_total = 0.0
        for sub_lb, zb, cols in results:  # results are in exec_order (fixed)
            sub_total += sub_lb
            if cols.size:
                z[cols] = zb
        L = sub_total - float(lam @ r_c) + lin.c_offset
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
        A_all = A_leq if A_leq.shape[0] else None
        b_all = b_leq if A_leq.shape[0] else None
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
    # Bundle model: list of (L_k, g_k, lam_k). Level bundle also tracks a
    # stability centre and an adaptive multiplier box.
    cuts: list[tuple[float, np.ndarray, np.ndarray]] = []
    lam_center = np.zeros(m_coup)
    lam_max = cfg.lambda_max0
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
            lam_center = lam.copy()  # serious step: move the stability centre
            stall = 0
        else:
            stall += 1

        if residual is not None and it % cfg.recover_every == 0:
            ub, z_full = _recover(z)
            if ub is not None and ub < best_UB:
                best_UB = ub
                incumbent = z_full

        if np.isfinite(best_UB) and best_L > -np.inf:
            gap = relative_gap(best_UB, best_L)
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
            lam = lam + step * residual
            # Project only the inequality (nonnegative) multipliers; equality
            # coupling rows carry a free multiplier (T1.2).
            lam[~free_mask] = np.maximum(0.0, lam[~free_mask])
            if stall >= cfg.patience:
                alpha *= 0.5
                stall = 0
        elif cfg.method == "kelley" or qp is None:
            # Plain Kelley cutting-plane (explicit, or bundle without a QP backend).
            cuts.append((L, residual.copy(), lam.copy()))
            lam = _bundle_step(lp, cuts, m_coup, free_mask)
        else:  # level bundle (T2.1)
            cuts.append((L, residual.copy(), lam.copy()))
            lam, L_hat, hit_box = _level_bundle_step(
                lp, qp, cuts, m_coup, free_mask, best_L, lam_center, lam_max, cfg.level_gamma
            )
            if hit_box and lam_max < 1e12:
                lam_max *= 10.0  # adaptive box growth
            # Reliable dual stopping test: the cutting-plane upper model L̂* is
            # within tolerance of the best dual value, so the dual is maximized.
            if np.isfinite(L_hat) and L_hat - best_L <= cfg.gap_tolerance * max(1.0, abs(best_L)):
                break

    # Promote to optimal if bounds already met.
    if status == "iteration_limit" and np.isfinite(best_UB) and best_L > -np.inf:
        if relative_gap(best_UB, best_L) <= cfg.gap_tolerance:
            status = "optimal"

    sense_flip = 1.0 if lin.minimize else -1.0
    objective = None if not np.isfinite(best_UB) else best_UB
    bound: float | None = None if best_L == -np.inf else best_L
    reported_obj = None if objective is None else objective * sense_flip
    reported_bound = None if bound is None else bound * sense_flip
    reported_gap: float | None = None
    if reported_obj is not None and reported_bound is not None:
        reported_gap = abs(relative_gap(reported_obj, reported_bound))
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


def _bundle_step(
    lp,
    cuts: list[tuple[float, np.ndarray, np.ndarray]],
    m_coup: int,
    free_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Maximize the cutting-plane model of L over the multiplier box.

    max_{λ∈box, η}  η   s.t.  η <= L_k + g_k^T (λ - λ_k)  for each cut k.
    Encoded as an LP minimizing -η. Inequality-row multipliers are bounded to
    ``[0, λmax]``; equality-row (free) multipliers to ``[-λmax, λmax]``.
    """
    if free_mask is None:
        free_mask = np.zeros(m_coup, dtype=bool)
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
    bounds = [
        (-_LAMBDA_MAX, _LAMBDA_MAX) if free_mask[i] else (0.0, _LAMBDA_MAX) for i in range(m_coup)
    ] + [(-_BIG, _BIG)]
    res = lp(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    if res.x is None:
        return cuts[-1][2]  # fall back to last lambda
    lam = np.asarray(res.x[:m_coup], dtype=np.float64)
    lam[~free_mask] = np.maximum(0.0, lam[~free_mask])
    return lam


def _kelley_max(
    lp,
    cuts: list[tuple[float, np.ndarray, np.ndarray]],
    m_coup: int,
    free_mask: np.ndarray,
    lam_max: float,
) -> tuple[np.ndarray | None, float]:
    """Maximize the cutting-plane model ``L̂`` over the box; return
    ``(argmax λ, L̂*)`` (``(None, -inf)`` if the LP fails)."""
    ncol = m_coup + 1
    c = np.zeros(ncol)
    c[-1] = -1.0
    rows, rhs = [], []
    for L_k, g_k, lam_k in cuts:
        row = np.zeros(ncol)
        row[:m_coup] = -g_k
        row[-1] = 1.0
        rows.append(row)
        rhs.append(float(L_k - g_k @ lam_k))
    bounds = [
        (-lam_max, lam_max) if free_mask[i] else (0.0, lam_max) for i in range(m_coup)
    ] + [(-_BIG, _BIG)]
    res = lp(np.asarray(c), A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bounds)
    if res.x is None:
        return None, -np.inf
    return np.asarray(res.x[:m_coup], dtype=np.float64), float(res.x[m_coup])


def _level_bundle_step(
    lp,
    qp,
    cuts: list[tuple[float, np.ndarray, np.ndarray]],
    m_coup: int,
    free_mask: np.ndarray,
    best_L: float,
    lam_center: np.ndarray,
    lam_max: float,
    gamma: float,
) -> tuple[np.ndarray, float, bool]:
    """One level-bundle iterate: project the stability centre onto the level set
    ``{λ : L̂(λ) >= ℓ}`` with ``ℓ = best_L + γ(L̂* - best_L)``.

    Returns ``(next_λ, L̂*, hit_box)``. Falls back to the Kelley maximizer when no
    QP backend is available or the projection QP fails.
    """
    lam_kelley, L_hat = _kelley_max(lp, cuts, m_coup, free_mask, lam_max)
    if lam_kelley is None:
        return cuts[-1][2], -np.inf, False
    hit_box = bool(np.any(np.abs(lam_kelley) > 0.99 * lam_max))
    if qp is None or not np.isfinite(best_L):
        return lam_kelley, L_hat, hit_box

    ell = best_L + gamma * (L_hat - best_L)
    # Projection QP: min ||λ - λ_c||^2  s.t.  L_k + g_k·(λ-λ_k) >= ℓ,  λ in box.
    Q = 2.0 * np.eye(m_coup)
    c = -2.0 * np.asarray(lam_center, dtype=np.float64)
    rows, rhs = [], []
    for L_k, g_k, lam_k in cuts:
        rows.append(-g_k)  # -g_k·λ <= L_k - g_k·λ_k - ℓ
        rhs.append(float(L_k - g_k @ lam_k - ell))
    bounds = [(-lam_max, lam_max) if free_mask[i] else (0.0, lam_max) for i in range(m_coup)]
    try:
        qres = qp(Q, c, A_ub=np.array(rows), b_ub=np.array(rhs), bounds=bounds)
    except Exception:  # noqa: BLE001 - a failed projection falls back to Kelley
        return lam_kelley, L_hat, hit_box
    if qres.x is None:
        return lam_kelley, L_hat, hit_box
    lam = np.asarray(qres.x[:m_coup], dtype=np.float64)
    lam[~free_mask] = np.maximum(0.0, lam[~free_mask])
    return lam, L_hat, hit_box
