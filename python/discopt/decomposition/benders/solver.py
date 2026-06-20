"""Classical Benders decomposition solver (v1: linear (MI)LP recourse).

Algorithm (minimization, internally):

    min  c_x^T x + c_y^T y
    s.t. master-only rows           A_m x <= r_m
         coupling/recourse rows     A_x x + A_y y <= r          (after <= canonicalization)
         x integer-or-continuous (first stage), y continuous (recourse)

Master:  min c_x^T x + eta   s.t. A_m x <= r_m  +  accumulated Benders cuts.
Subproblem at x̂:  min c_y^T y  s.t. A_y y <= r - A_x x̂.

- Feasible subproblem → **optimality cut** ``eta >= Q(x̂) + s^T (x - x̂)`` with
  slope ``s = -A_x^T lam`` from the recourse-LP row duals ``lam``.
- Infeasible subproblem → **feasibility cut** ``v(x̂) + s^T (x - x̂) <= 0`` from a
  slack-penalized always-feasible LP, excluding x̂.

Every cut is anchored at the **primal** recourse value (``Q(x̂)`` / ``v(x̂)``),
which is exact at x̂; the row-dual slope is a subgradient of the convex LP value
function. Primal anchoring stays sound even when the recourse optimum is partly
set by variable bounds (where the row duals are an *incomplete* dual solution)
and with POUNCE's interior-point duals — so the solver runs on whichever LP/MILP
backend is installed (**no HiGHS dependency**: the POUNCE stack does it all), and
the master objective is a rigorous lower bound.

A *nonlinear* model is routed to Generalized Benders (:mod:`.gbd`), which
handles a convex-NLP recourse subproblem with KKT-multiplier cuts.
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
    max_iterations: int = 100
    prefer_pounce: bool = False
    feas_tol: float = 1e-6
    eta_floor: float = _ETA_FLOOR


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

    # Recourse-LP dual generator. POUNCE and HiGHS share the dual sign
    # convention, and the cuts below are anchored at the *dual value* so they
    # stay valid even for POUNCE's interior-point (analytic-centre) duals — no
    # HiGHS dependency. Whichever LP backend is installed is used.
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
    A_m_rows, b_m_rows = [], []  # master-only, over master cols
    Ax_rows, Ay_rows, r_rows = [], [], []  # recourse: A_x (master), A_y (sub), rhs
    for vec, rhs in zip(lin.rows_coeff, lin.rows_rhs):
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
    A_x = np.array(Ax_rows) if Ax_rows else np.zeros((0, n_master))
    A_y = np.array(Ay_rows) if Ay_rows else np.zeros((0, len(scols)))
    r = np.array(r_rows) if r_rows else np.zeros(0)

    c_master = lin.c[mcols]
    c_sub = lin.c[scols] if len(scols) else np.zeros(0)

    master_bounds = [(float(lb_all[i]), float(ub_all[i])) for i in mcols]
    sub_bounds = [(float(lb_all[i]), float(ub_all[i])) for i in scols]

    # Accumulated cuts on (x, eta): coeff_x (n_master,), coeff_eta, rhs (<=).
    cut_x: list[np.ndarray] = []
    cut_eta: list[float] = []
    cut_rhs: list[float] = []

    def _solve_master(with_eta: bool):
        ncol = n_master + (1 if with_eta else 0)
        c = np.zeros(ncol)
        c[:n_master] = c_master
        if with_eta:
            c[-1] = 1.0
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
                cm[k, -1] = cut_eta[k]
            rows.append(cm)
            rhs.append(np.array(cut_rhs))
        A_ub = np.vstack(rows) if rows else None
        b_ub = np.concatenate(rhs) if rhs else None
        bounds = list(master_bounds)
        integrality = np.zeros(ncol, dtype=np.int32)
        integrality[:n_master] = part.master_int.astype(np.int32)
        if with_eta:
            bounds.append((cfg.eta_floor, _BIG))
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

    def _recourse(x_hat: np.ndarray):
        """Return ('opt', Q, y, lam) or ('infeas', v, None, lam)."""
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
            return "opt", float(res.objective), np.asarray(res.x, dtype=np.float64), lam
        # Feasibility subproblem: min 1^T s s.t. A_y y - s <= rhs, s >= 0.
        m_rec = A_y.shape[0]
        ny = len(scols)
        Afeas = np.zeros((m_rec, ny + m_rec))
        Afeas[:, :ny] = A_y
        Afeas[:, ny:] = -np.eye(m_rec)
        c_feas = np.concatenate([np.zeros(ny), np.ones(m_rec)])
        bnds = list(sub_bounds) + [(0.0, _BIG)] * m_rec
        fres = dual_lp(c_feas, A_ub=Afeas, b_ub=rhs, bounds=bnds)
        lam = (
            np.asarray(fres.dual_values, dtype=np.float64)
            if fres.dual_values is not None
            else np.zeros(m_rec)
        )
        v = float(fres.objective) if fres.objective is not None else 0.0
        return "infeas", v, None, lam

    def _add_opt_cut(x_hat, Q, lam):
        # Optimality cut anchored at the **primal** recourse value Q(x̂) (exact),
        # with slope s = -A_x^T lam from the recourse row duals:
        #     eta >= Q(x̂) + s^T (x - x̂).
        # Primal anchoring is required for soundness. The reconstructed dual
        # value lam^T (r - A_x x̂) equals Q(x̂) only for a *complete* dual
        # solution; when the recourse optimum is partly set by variable bounds
        # the row duals lam are incomplete (their bound-multiplier counterpart is
        # omitted), so lam^T r can *exceed* Q(x̂) — e.g. POUNCE's interior-point
        # solver splits a marginal between a row and a coinciding variable bound,
        # returning a half-magnitude row dual. Anchoring at that dual value would
        # push the cut above the true value and prune the optimum. The primal
        # anchor is exact at x̂; the row-dual slope is a valid subgradient of the
        # convex (piecewise-linear) LP value function regardless of the split.
        # eta >= Q + s^T (x - x̂)  ->  s^T x - eta <= s^T x̂ - Q.
        s = -(A_x.T @ lam) if A_x.shape[0] else np.zeros(n_master)
        cut_x.append(s.copy())
        cut_eta.append(-1.0)
        cut_rhs.append(float(s @ x_hat) - Q)

    def _add_feas_cut(x_hat, v, lam):
        # Feasibility cut anchored at the primal min-infeasibility v(x̂) > 0 of
        # the slack-min subproblem, slope s = -A_x^T lam_feas:
        #     v(x̂) + s^T (x - x̂) <= 0      (drive the infeasibility to 0).
        # Primal-anchored for the same soundness reason as the optimality cut;
        # excludes x̂ since v(x̂) > 0.  s^T x <= s^T x̂ - v.
        s = -(A_x.T @ lam) if A_x.shape[0] else np.zeros(n_master)
        cut_x.append(s.copy())
        cut_eta.append(0.0)
        cut_rhs.append(float(s @ x_hat) - v)

    # ── initialize: feasible master point (no eta) ──
    init = _solve_master(with_eta=False)
    if init.status == SolveStatus.INFEASIBLE:
        return SolveResult(status="infeasible", wall_time=time.time() - t0, gap_certified=True)
    if init.status != SolveStatus.OPTIMAL or init.x is None:
        return SolveResult(status="error", wall_time=time.time() - t0)
    x_hat = np.asarray(init.x[:n_master], dtype=np.float64)

    kind, val, y_star, lam = _recourse(x_hat)
    if kind == "opt":
        _add_opt_cut(x_hat, val, lam)
    else:
        _add_feas_cut(x_hat, val, lam)

    best_ub = np.inf
    incumbent_full: np.ndarray | None = None
    status = "iteration_limit"

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
        lb = mres.bound if mres.bound is not None else mres.objective
        lower_bound = (float(lb) + lin.c_offset) if lb is not None else None

        kind, val, y_star, lam = _recourse(x_hat)
        if kind == "opt":
            full_obj = float(c_master @ x_hat + val + lin.c_offset)
            if full_obj < best_ub:
                best_ub = full_obj
                x_full = np.zeros(lin.n)
                x_full[mcols] = x_hat
                if len(scols):
                    x_full[scols] = y_star
                incumbent_full = x_full
            _add_opt_cut(x_hat, val, lam)
        else:
            _add_feas_cut(x_hat, val, lam)

        if np.isfinite(best_ub) and lower_bound is not None:
            gap = (best_ub - lower_bound) / (abs(best_ub) + 1e-10)
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

    if status == "infeasible":
        return SolveResult(status="infeasible", wall_time=time.time() - t0, gap_certified=True)

    objective = None if not np.isfinite(best_ub) else best_ub
    # Promote to optimal if the final master bound already meets the incumbent.
    if status == "iteration_limit" and objective is not None and bound is not None:
        if (objective - bound) / (abs(objective) + 1e-10) <= cfg.gap_tolerance:
            status = "optimal"

    # Map the internal-minimization values back to the model's sense. For a
    # maximize problem (c was negated), negating swaps the incumbent (a lower
    # bound on the max) and the dual bound (an upper bound on the max) into the
    # reported objective/bound; their gap magnitude is unchanged.
    sense_flip = 1.0 if lin.minimize else -1.0
    reported_obj = None if objective is None else objective * sense_flip
    reported_bound = None if bound is None else bound * sense_flip
    gap = None
    if reported_obj is not None and reported_bound is not None:
        gap = abs(reported_obj - reported_bound) / (abs(reported_obj) + 1e-10)

    x_dict = solution_dict(model, incumbent_full) if incumbent_full is not None else None

    return SolveResult(
        status=status,
        objective=reported_obj,
        bound=reported_bound,
        gap=gap,
        x=x_dict,
        wall_time=time.time() - t0,
        node_count=0,
        gap_certified=(status == "optimal"),
    )
