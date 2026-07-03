"""Classical Benders decomposition solver (v1: linear (MI)LP recourse).

Algorithm (minimization, internally):

    min  c_x^T x + c_y^T y
    s.t. master-only rows           A_m x <= r_m
         coupling/recourse rows     A_x x + A_y y <= r          (after <= canonicalization)
         x integer-or-continuous (first stage), y continuous (recourse)

Master:  min c_x^T x + eta   s.t. A_m x <= r_m  +  accumulated Benders cuts.
Subproblem at x̂:  min c_y^T y  s.t. A_y y <= r - A_x x̂.

- Feasible subproblem → **optimality cut** ``eta >= lam^T(r - A_x x) + bt``.
- Infeasible subproblem → **feasibility cut** ``lam_f^T(r - A_x x) + bt_f <= 0``
  from a slack-penalized always-feasible LP, excluding x̂.

Every cut is the **complete LP dual objective** as an affine function of the
master variables: the row-dual term ``lam^T(r - A_x x)`` plus the variable
reduced-cost term ``bt = sum_j rc_j·(lb_j if rc_j>0 else ub_j)``. By LP weak
duality this is a lower bound on the recourse value for *every* master point and
*any* dual-feasible ``(lam, rc)``, so the cut is always a valid global
underestimator. This is robust to two failure modes that simpler anchors have:
an interior-point recourse solve returning a slightly *suboptimal* primal
(anchoring at that primal would over-cut), and the row duals alone being an
*incomplete* dual when a variable bound is active (the ``rc`` term restores the
missing contribution). The solver therefore runs soundly on whichever LP/MILP
backend is installed, including POUNCE's interior-point duals (**no HiGHS
dependency**), and the master objective is a rigorous lower bound.

A *nonlinear* model is routed to Generalized Benders (:mod:`.gbd`), which
handles a convex-NLP recourse subproblem with KKT-multiplier cuts.
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
)
from discopt.modeling.core import (
    Model,
    SolveResult,
    VarType,
)
from discopt.solvers import SolveStatus

logger = logging.getLogger(__name__)

# A valid global lower bound on the recourse cost, used as the master ``eta``
# floor so the master is bounded before any optimality cut is generated. The
# master LB stays rigorous as long as the true recourse cost is >= this value.
_ETA_FLOOR = -1e12
_BIG = 1e20


@dataclass
class BendersConfig:
    """Configuration for the Benders solver."""

    time_limit: float = 3600.0
    gap_tolerance: float = 1e-4
    # One Benders cut is added per iteration, so 100 is far too few for real
    # instances; runtime stays bounded by ``time_limit`` regardless (T0.4).
    max_iterations: int = 500
    prefer_pounce: bool = False
    feas_tol: float = 1e-6
    eta_floor: float = _ETA_FLOOR
    # Multicut: give each independent recourse block its own eta variable and
    # cut (T1.3). Reduces to the classic single-cut solver when the recourse does
    # not separate (one block). ``backend`` runs the per-block recourse solves
    # sequentially or on a thread pool.
    multicut: bool = True
    backend: str = "sequential"


def _model_is_linear(model: Model) -> bool:
    """True iff every constraint body and the objective are linear.

    Used to route between classical Benders (linear recourse LP) and Generalized
    Benders (convex-NLP recourse).
    """
    from discopt._jax.gdp_reformulate import _is_linear
    from discopt.modeling.core import Constraint

    for c in model._constraints:
        if not isinstance(c, Constraint):
            return False
        if not _is_linear(c.body):
            return False
    if model._objective is not None and not _is_linear(model._objective.expression):
        return False
    return True


# ── Column partition ──────────────────────────────────────────


@dataclass
class _Partition:
    master_cols: np.ndarray  # int indices into flat vector (first stage)
    sub_cols: np.ndarray  # int indices (recourse)
    master_int: np.ndarray  # bool over master_cols: integer?


def _partition_columns(model: Model, structure: DecompositionStructure) -> _Partition:
    complicating = set(structure.complicating_vars)
    master_cols: list[int] = []
    sub_cols: list[int] = []
    master_int: list[bool] = []
    off = 0
    for v in model._variables:
        is_master = v.name in complicating
        is_int = v.var_type in (VarType.BINARY, VarType.INTEGER)
        for _ in range(v.size):
            if is_master:
                master_cols.append(off)
                master_int.append(is_int)
            else:
                if is_int:
                    raise NotImplementedError(
                        f"Variable {v.name!r} is integer but is in the recourse "
                        "subproblem; Benders v1 requires all integer variables to "
                        "be first-stage (mark them with model.first_stage(...))."
                    )
                sub_cols.append(off)
            off += 1
    return _Partition(
        np.array(master_cols, dtype=int),
        np.array(sub_cols, dtype=int),
        np.array(master_int, dtype=bool),
    )


# ── Solver ────────────────────────────────────────────────────


def solve_benders(
    model: Model,
    *,
    structure: DecompositionStructure | None = None,
    time_limit: float = 3600.0,
    gap_tolerance: float = 1e-4,
    max_iterations: int = 100,
    nlp_solver: str = "pounce",
    config: BendersConfig | None = None,
    **_ignored,
) -> SolveResult:
    """Solve a two-stage (mixed-integer) linear program by Benders decomposition.

    Parameters
    ----------
    model : Model
        Linear-constraint, linear-objective model. Integer variables must be
        first-stage.
    structure : DecompositionStructure, optional
        Decomposition structure; auto-detected when omitted (complicating
        variables default to the integer variables).
    time_limit, gap_tolerance, max_iterations
        Standard termination controls.

    Returns
    -------
    SolveResult
        With a rigorous lower ``bound`` (the master objective) on convergence.
    """
    from discopt.solvers.lp_backend import get_lp_solver, get_milp_solver

    cfg = config or BendersConfig(
        time_limit=time_limit,
        gap_tolerance=gap_tolerance,
        max_iterations=max_iterations,
        prefer_pounce=(nlp_solver == "pounce"),
    )
    t0 = time.time()

    # Recourse-LP dual generator. The cuts below use the *complete* LP dual
    # objective (row duals + variable reduced costs), which is a valid lower
    # bound by weak duality for any dual-feasible point — so it stays sound even
    # with POUNCE's interior-point (analytic-centre) duals and an inexact primal,
    # with no HiGHS dependency. Whichever LP backend is installed is used.
    dual_lp = get_lp_solver(prefer_pounce=cfg.prefer_pounce)
    milp = get_milp_solver(prefer_pounce=cfg.prefer_pounce)

    if structure is None:
        structure = detect_decomposition(model)

    # Nonlinear objective/constraint -> Generalized Benders (convex-NLP recourse).
    if not _model_is_linear(model):
        from discopt.decomposition.benders.gbd import solve_gbd

        return solve_gbd(
            model,
            structure=structure,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            max_iterations=max_iterations,
            nlp_solver=nlp_solver,
        )

    lin = extract_linear(model)
    part = _partition_columns(model, structure)
    mcols, scols = part.master_cols, part.sub_cols
    n_master = len(mcols)
    if n_master == 0:
        raise NotImplementedError(
            "Benders needs at least one first-stage (complicating) variable. "
            "Annotate them with model.first_stage(...), or ensure the model has "
            "integer variables (the default complicating set)."
        )

    lb_all, ub_all = flat_bounds(model)

    # Split rows into master-only and recourse (any nonzero recourse coeff).
    # The recourse LP is a ``<=`` problem, so use the ``<=``-canonical view
    # (equalities expanded, ``>=`` flipped) — identical to the classic row set.
    A_leq, b_leq, _src_leq = lin.rows_leq()
    A_m_rows, b_m_rows = [], []  # master-only, over master cols
    Ax_rows, Ay_rows, r_rows = [], [], []  # recourse: A_x (master), A_y (sub), rhs
    for vec, rhs in zip(A_leq, b_leq):
        sub_part = vec[scols] if len(scols) else np.zeros(0)
        if sub_part.size and np.any(np.abs(sub_part) > 0):
            Ax_rows.append(vec[mcols])
            Ay_rows.append(sub_part)
            r_rows.append(rhs)
        else:
            A_m_rows.append(vec[mcols])
            b_m_rows.append(rhs)

    A_m = np.array(A_m_rows) if A_m_rows else np.zeros((0, n_master))
    b_m = np.array(b_m_rows) if b_m_rows else np.zeros(0)

    c_master = lin.c[mcols]
    master_bounds = [(float(lb_all[i]), float(ub_all[i])) for i in mcols]

    # ── partition the recourse into independent blocks (T1.3 multicut) ──
    # Two sub-variables share a block iff they co-occur in a recourse row; the
    # blocks are the connected components of that graph. Each block gets its own
    # eta and its own cut, so a separable recourse converges in far fewer
    # iterations. With one block this is exactly the classic single-cut solver.
    ny = len(scols)
    if cfg.multicut and ny:
        from discopt.decomposition.graph import kernels

        row_positions = [list(np.nonzero(np.abs(ay) > 0)[0]) for ay in Ay_rows]
        sub_cliques = [pos for pos in row_positions if pos]
        block_of_pos, n_blocks = kernels.connected_components(ny, sub_cliques)
    else:
        row_positions = [list(np.nonzero(np.abs(ay) > 0)[0]) for ay in Ay_rows]
        block_of_pos = [0] * ny
        n_blocks = 1
    n_blocks = max(1, n_blocks)

    pos_of_block: list[list[int]] = [[] for _ in range(n_blocks)]
    for k in range(ny):
        pos_of_block[block_of_pos[k]].append(k)
    rows_of_block: list[list[int]] = [[] for _ in range(n_blocks)]
    for i, pos in enumerate(row_positions):
        b = block_of_pos[pos[0]] if pos else 0
        rows_of_block[b].append(i)

    class _Block:
        __slots__ = ("bid", "cols", "c_sub", "sub_bounds", "A_x", "A_y", "r")

        def __init__(self, bid: int, positions: list[int], row_idx: list[int]):
            self.bid = bid
            self.cols = np.array([scols[k] for k in positions], dtype=int)
            self.c_sub = lin.c[self.cols] if self.cols.size else np.zeros(0)
            self.sub_bounds = [(float(lb_all[j]), float(ub_all[j])) for j in self.cols]
            self.A_x = (
                np.array([Ax_rows[i] for i in row_idx]) if row_idx else np.zeros((0, n_master))
            )
            self.A_y = (
                np.array([Ay_rows[i][positions] for i in row_idx])
                if row_idx
                else np.zeros((0, len(positions)))
            )
            self.r = np.array([r_rows[i] for i in row_idx]) if row_idx else np.zeros(0)

    blocks = [_Block(b, pos_of_block[b], rows_of_block[b]) for b in range(n_blocks)]

    from discopt.decomposition.parallel.comm import select_backend

    comm = select_backend(cfg.backend)

    # Accumulated cuts, each on ``(x, eta_b)``: (block_id, coeff_x, coeff_eta, rhs).
    cut_block: list[int] = []
    cut_x: list[np.ndarray] = []
    cut_eta: list[float] = []
    cut_rhs: list[float] = []

    def _solve_master(with_eta: bool):
        n_eta = n_blocks if with_eta else 0
        ncol = n_master + n_eta
        c = np.zeros(ncol)
        c[:n_master] = c_master
        if with_eta:
            c[n_master:] = 1.0  # min c_x·x + sum_b eta_b
        rows, rhs = [], []
        if A_m.shape[0]:
            pad = np.zeros((A_m.shape[0], ncol))
            pad[:, :n_master] = A_m
            rows.append(pad)
            rhs.append(b_m)
        if with_eta and cut_x:
            cm = np.zeros((len(cut_x), ncol))
            for k in range(len(cut_x)):
                cm[k, :n_master] = cut_x[k]
                cm[k, n_master + cut_block[k]] = cut_eta[k]
            rows.append(cm)
            rhs.append(np.array(cut_rhs))
        A_ub = np.vstack(rows) if rows else None
        b_ub = np.concatenate(rhs) if rhs else None
        bounds = list(master_bounds)
        integrality = np.zeros(ncol, dtype=np.int32)
        integrality[:n_master] = part.master_int.astype(np.int32)
        if with_eta:
            bounds.extend([(cfg.eta_floor, _BIG)] * n_blocks)
        res = milp(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            integrality=integrality,
            time_limit=max(1.0, cfg.time_limit - (time.time() - t0)),
            gap_tolerance=cfg.gap_tolerance,
        )
        return res

    def _recourse_block(blk: "_Block", x_hat: np.ndarray):
        """Classify one recourse block's LP at x̂ (kinds as in the single-block
        docstring: ``opt`` / ``unbounded`` / ``infeas`` / ``recourse_fail``)."""
        A_x, A_y, r, sub_bounds, c_sub = blk.A_x, blk.A_y, blk.r, blk.sub_bounds, blk.c_sub
        rhs = r - (A_x @ x_hat if A_x.shape[0] else np.zeros(0))
        res = dual_lp(
            c_sub,
            A_ub=A_y if A_y.shape[0] else None,
            b_ub=rhs if A_y.shape[0] else None,
            bounds=sub_bounds,
        )
        if res.status == SolveStatus.OPTIMAL:
            lam = (
                np.asarray(res.dual_values, dtype=np.float64)
                if res.dual_values is not None
                else np.zeros(A_y.shape[0])
            )
            rc = (
                np.asarray(res.reduced_costs, dtype=np.float64)
                if res.reduced_costs is not None
                else None
            )
            return (
                "opt",
                float(res.objective),
                np.asarray(res.x, dtype=np.float64),
                lam,
                rc,
                sub_bounds,
            )
        if res.status == SolveStatus.UNBOUNDED:
            return "unbounded", None, None, None, None, sub_bounds
        m_rec = A_y.shape[0]
        nyb = len(blk.cols)
        Afeas = np.zeros((m_rec, nyb + m_rec))
        Afeas[:, :nyb] = A_y
        Afeas[:, nyb:] = -np.eye(m_rec)
        c_feas = np.concatenate([np.zeros(nyb), np.ones(m_rec)])
        bnds = list(sub_bounds) + [(0.0, _BIG)] * m_rec
        fres = dual_lp(c_feas, A_ub=Afeas, b_ub=rhs, bounds=bnds)
        if fres.status != SolveStatus.OPTIMAL:
            return "recourse_fail", None, None, None, None, bnds
        v = float(fres.objective) if fres.objective is not None else 0.0
        lam = (
            np.asarray(fres.dual_values, dtype=np.float64)
            if fres.dual_values is not None
            else np.zeros(m_rec)
        )
        rc = (
            np.asarray(fres.reduced_costs, dtype=np.float64)
            if fres.reduced_costs is not None
            else None
        )
        return "infeas", v, None, lam, rc, bnds

    def _recourse_all(x_hat: np.ndarray):
        """Solve every block's recourse at x̂ on the configured backend, in block
        order (deterministic regardless of backend)."""
        return comm.map(blocks, lambda blk: _recourse_block(blk, x_hat))

    def _dual_const(blk: "_Block", lam, rc, col_bounds) -> float:
        """The y-independent part of the *complete* LP dual objective,
        ``lam^T r + sum_j rc_j * (lb_j if rc_j>0 else ub_j)``.

        By LP weak duality the full dual objective ``D(y) = lam^T(r - A_x y) +
        bound_terms`` is a lower bound on the recourse value ``Q(y)`` for *every*
        master point y and *any* dual-feasible (lam, rc) — so a cut anchored at
        ``D`` is always a valid global underestimator. This is robust to two
        failure modes a naive anchor has: (1) an interior-point recourse solve
        returns a slightly *suboptimal* primal ``Q`` (anchoring there would
        over-cut), and (2) the row duals alone are an *incomplete* dual when a
        variable bound is active (the ``rc`` term restores the missing
        contribution).

        A bound at the ``_BIG`` open sentinel (or any ``|bound| >= _BIG``) is
        treated as infinite and skipped: ``np.isfinite(1e20)`` is ``True``, so a
        tiny reduced-cost noise at a sentinel-bounded column (e.g. a feasibility
        slack's ``_BIG`` upper bound) would otherwise inject a ``rc * 1e20``
        term and corrupt the cut constant.
        """
        r = blk.r
        d = float(lam @ r) if lam.size and r.size else 0.0
        if rc is not None:
            for j, rcj in enumerate(rc):
                if rcj > 0:
                    lbj = col_bounds[j][0]
                    if np.isfinite(lbj) and abs(lbj) < _BIG:
                        d += rcj * lbj
                elif rcj < 0:
                    ubj = col_bounds[j][1]
                    if np.isfinite(ubj) and abs(ubj) < _BIG:
                        d += rcj * ubj
        return d

    def _add_opt_cut(blk: "_Block", lam, rc, col_bounds):
        # Optimality cut on this block's eta = the complete LP dual objective:
        #     eta_b >= lam^T(r - A_x y) + bound_terms = const + s^T y,
        # with s = -A_x^T lam. Valid for any dual-feasible (lam, rc) by weak
        # duality.  s^T x - eta_b <= -const.
        s = -(blk.A_x.T @ lam) if blk.A_x.shape[0] else np.zeros(n_master)
        const = _dual_const(blk, lam, rc, col_bounds)
        cut_block.append(blk.bid)
        cut_x.append(s.copy())
        cut_eta.append(-1.0)
        cut_rhs.append(-const)
        return blk.bid, s, -1.0, -const

    def _add_feas_cut(blk: "_Block", lam, rc, col_bounds):
        # Feasibility cut for this block: the min-infeasibility dual objective
        #     v_D(y) = lam^T(r - A_x y) + bound_terms <= 0  (excludes ŷ).
        s = -(blk.A_x.T @ lam) if blk.A_x.shape[0] else np.zeros(n_master)
        const = _dual_const(blk, lam, rc, col_bounds)
        cut_block.append(blk.bid)
        cut_x.append(s.copy())
        cut_eta.append(0.0)
        cut_rhs.append(-const)
        return blk.bid, s, 0.0, -const

    def _cut_separates(cut, x_hat, eta_vec) -> bool:
        # A cut ``s·x + eta_coef·eta_b <= rhs`` separates (x̂, η̂_b) iff violated
        # there beyond tolerance. A cut that does not separate makes no progress.
        bid, s, eta_coef, rhs = cut
        eta_b = float(eta_vec[bid]) if bid < len(eta_vec) else -np.inf
        lhs = float(s @ x_hat) + eta_coef * eta_b
        return bool(lhs > rhs + cfg.feas_tol)

    # ── initialize: feasible master point (no eta) ──
    init = _solve_master(with_eta=False)
    if init.status == SolveStatus.INFEASIBLE:
        return SolveResult(status="infeasible", wall_time=time.time() - t0, gap_certified=True)
    if init.status != SolveStatus.OPTIMAL or init.x is None:
        return SolveResult(status="error", wall_time=time.time() - t0)
    x_hat = np.asarray(init.x[:n_master], dtype=np.float64)

    results = _recourse_all(x_hat)
    if any(r is not None and r[0] == "recourse_fail" for r in results):
        results = _recourse_all(x_hat)  # retry once
        if any(r is not None and r[0] == "recourse_fail" for r in results):
            logger.warning("Benders recourse solve failed twice at the initial point.")
            return SolveResult(status="error", wall_time=time.time() - t0)
    for r in results:
        if r[0] == "unbounded":
            return SolveResult(status="unbounded", wall_time=time.time() - t0)
    for blk, res in zip(blocks, results):
        kind = res[0]
        if kind == "opt":
            _add_opt_cut(blk, res[3], res[4], res[5])
        elif kind == "infeas":
            _add_feas_cut(blk, res[3], res[4], res[5])

    best_ub = np.inf
    incumbent_full: np.ndarray | None = None
    status = "iteration_limit"
    stall = 0  # consecutive non-separating iterations (T0.4 progress guard)

    for _it in range(cfg.max_iterations):
        if time.time() - t0 > cfg.time_limit:
            status = "time_limit"
            break
        mres = _solve_master(with_eta=True)
        if mres.status == SolveStatus.INFEASIBLE:
            status = "infeasible"
            best_ub = np.inf
            incumbent_full = None
            break
        if mres.x is None:
            status = "error"
            break
        x_hat = np.asarray(mres.x[:n_master], dtype=np.float64)
        eta_vec = np.asarray(mres.x[n_master : n_master + n_blocks], dtype=np.float64)
        lb = mres.bound if mres.bound is not None else mres.objective
        lower_bound = (float(lb) + lin.c_offset) if lb is not None else None

        results = _recourse_all(x_hat)
        if any(r is not None and r[0] == "recourse_fail" for r in results):
            results = _recourse_all(x_hat)  # retry once (T0.3)
            if any(r is not None and r[0] == "recourse_fail" for r in results):
                logger.warning("Benders recourse solve failed twice; stopping.")
                break
        for r in results:
            if r[0] == "unbounded":
                return SolveResult(status="unbounded", wall_time=time.time() - t0)

        # Incumbent: only when every block's recourse is optimal (feasible).
        if all(res[0] == "opt" for res in results):
            total = float(c_master @ x_hat + lin.c_offset)
            x_full = np.zeros(lin.n)
            x_full[mcols] = x_hat
            for blk, res in zip(blocks, results):
                total += res[1]
                if blk.cols.size:
                    x_full[blk.cols] = res[2]
            if total < best_ub:
                best_ub = total
                incumbent_full = x_full

        # Per-block cuts; track whether any of them separates the master point.
        separated = False
        for blk, res in zip(blocks, results):
            if res[0] == "opt":
                cut = _add_opt_cut(blk, res[3], res[4], res[5])
            else:  # infeas
                cut = _add_feas_cut(blk, res[3], res[4], res[5])
            if _cut_separates(cut, x_hat, eta_vec):
                separated = True

        if separated:
            stall = 0
        else:
            stall += 1
            logger.warning(
                "Benders iteration %d added no separating cut "
                "(degenerate or missing duals); stall=%d.",
                _it,
                stall,
            )
            if stall >= 2:
                break

        if np.isfinite(best_ub) and lower_bound is not None:
            gap = relative_gap(best_ub, lower_bound)
            if gap <= cfg.gap_tolerance:
                status = "optimal"
                break

    # Final bound = best master lower bound from the last master solve.
    final = _solve_master(with_eta=True)
    bound = None
    if final.status == SolveStatus.OPTIMAL:
        lb = final.bound if final.bound is not None else final.objective
        if lb is not None:
            bound = float(lb) + lin.c_offset
        # The eta floor keeps the master bounded before cuts arrive; if any
        # block's eta still sits on it, that block's recourse is not yet
        # underestimated and the master "bound" is the floor — withhold it (T0.5).
        if final.x is not None and len(final.x) >= n_master + n_blocks:
            eta_final = np.asarray(final.x[n_master : n_master + n_blocks], dtype=np.float64)
            if np.any(eta_final <= cfg.eta_floor + 1.0):
                logger.warning("Benders eta floor still active; withholding bound.")
                bound = None

    if status == "infeasible":
        return SolveResult(status="infeasible", wall_time=time.time() - t0, gap_certified=True)

    objective = None if not np.isfinite(best_ub) else best_ub
    # Promote to optimal if the final master bound already meets the incumbent.
    if status == "iteration_limit" and objective is not None and bound is not None:
        if relative_gap(objective, bound) <= cfg.gap_tolerance:
            status = "optimal"

    # Map the internal-minimization values back to the model's sense. For a
    # maximize problem (c was negated), negating swaps the incumbent (a lower
    # bound on the max) and the dual bound (an upper bound on the max) into the
    # reported objective/bound; their gap magnitude is unchanged.
    sense_flip = 1.0 if lin.minimize else -1.0
    reported_obj = None if objective is None else objective * sense_flip
    reported_bound = None if bound is None else bound * sense_flip
    reported_gap: float | None = None
    if reported_obj is not None and reported_bound is not None:
        reported_gap = abs(relative_gap(reported_obj, reported_bound))

    x_dict = solution_dict(model, incumbent_full) if incumbent_full is not None else None

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
