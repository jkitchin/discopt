"""Generalized Benders Decomposition (Geoffrion 1972): convex-NLP recourse.

Where classical Benders (``solver.py``) handles a *linear* recourse LP, GBD
handles a **convex nonlinear** recourse subproblem. The two-stage problem
(internally minimizing) is

    min  f(x, y)
    s.t. g(x, y) <= 0            (coupling / recourse constraints)
         x in X                  (complicating / first-stage; may be integer)
         y in Y                  (continuous recourse)

For a fixed first-stage point ``x̂`` the **recourse value function** is

    v(x̂) = min_y f(x̂, y)  s.t.  g(x̂, y) <= 0,

a *convex* function of ``x`` when ``f`` and every ``g_i`` are jointly convex.
Solving the recourse NLP yields the optimal recourse ``ŷ`` and KKT multipliers
``μ >= 0``. By the envelope theorem a subgradient of ``v`` at ``x̂`` is the
gradient of the Lagrangian w.r.t. the first-stage variables,

    s = ∇_x [ f(x, y) + μ^T g(x, y) ] |_(x̂, ŷ),

so the **optimality cut**

    eta >= v(x̂) + s^T (x - x̂)

is a valid global underestimator of the convex value function. The master

    min_x  eta   s.t.  master-only rows,  GBD cuts,  x integral,  eta >= floor

therefore yields a rigorous lower bound. When the recourse is *infeasible* at a
0/1 first-stage point we exclude that point with a **no-good cut** (rigorous: a
point whose recourse is infeasible cannot be part of any feasible solution).

Soundness is gated on convexity: the reported ``bound`` is valid only when the
objective and all constraints are convex (checked with
``classify_oa_cut_convexity``); otherwise the solver runs heuristically and
reports ``bound=None`` so the ``incorrect_count <= 0`` gate is never threatened.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from discopt.decomposition._linear import solution_dict
from discopt.decomposition.structure import (
    DecompositionStructure,
    detect_decomposition,
    flat_bounds,
)
from discopt.modeling.core import Model, ObjectiveSense, SolveResult, VarType
from discopt.solvers import SolveStatus

logger = logging.getLogger(__name__)

_ETA_FLOOR = -1e12
_BIG = 1e20


def _master_columns(model: Model, structure: DecompositionStructure):
    """Return (master_cols, sub_cols, master_int) flat-index partition.

    Mirrors ``solver._partition_columns`` but lives here to keep GBD
    self-contained. Integer variables must be first-stage (the recourse NLP
    needs continuous KKT multipliers).
    """
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
            elif is_int:
                raise NotImplementedError(
                    f"Variable {v.name!r} is integer but in the recourse subproblem; "
                    "GBD requires all integer variables to be first-stage "
                    "(mark them with model.first_stage(...))."
                )
            else:
                sub_cols.append(off)
            off += 1
    return (
        np.array(master_cols, dtype=int),
        np.array(sub_cols, dtype=int),
        np.array(master_int, dtype=bool),
    )


def solve_gbd(
    model: Model,
    *,
    structure: DecompositionStructure | None = None,
    time_limit: float = 3600.0,
    gap_tolerance: float = 1e-4,
    max_iterations: int = 100,
    nlp_solver: str = "pounce",
    **_ignored,
) -> SolveResult:
    """Solve a two-stage convex MINLP with nonlinear recourse by GBD.

    Parameters
    ----------
    model : Model
        Convex objective/constraints, continuous recourse, integer variables (if
        any) first-stage. Master-only constraints must be linear.
    structure : DecompositionStructure, optional
        Auto-detected when omitted (complicating vars default to integers).

    Returns
    -------
    SolveResult
        With a rigorous lower ``bound`` when the model is convex; ``bound=None``
        otherwise (heuristic mode).
    """
    from discopt._jax.convexity import classify_oa_cut_convexity
    from discopt._jax.gdp_reformulate import _extract_body_coeffs, _is_linear
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt.modeling.core import Constraint
    from discopt.solvers.lp_backend import get_milp_solver

    t0 = time.time()
    prefer_pounce = nlp_solver == "pounce"
    milp = get_milp_solver(prefer_pounce=prefer_pounce)

    if structure is None:
        structure = detect_decomposition(model)
    mcols, scols, master_int = _master_columns(model, structure)
    n_master = len(mcols)
    if n_master == 0:
        raise NotImplementedError(
            "GBD needs at least one first-stage (complicating) variable. Annotate "
            "them with model.first_stage(...), or give the model integer variables."
        )
    if len(scols) == 0:
        raise NotImplementedError(
            "GBD needs at least one recourse (second-stage) variable; this model "
            "has none. Solve it directly with Model.solve()."
        )

    evaluator = NLPEvaluator(model)
    n_vars = evaluator.n_variables
    lb_all, ub_all = flat_bounds(model)

    sense = model._objective.sense if model._objective is not None else ObjectiveSense.MINIMIZE
    sense_flip = 1.0 if sense == ObjectiveSense.MINIMIZE else -1.0

    # Convexity gate: a rigorous bound requires a convex value function.
    oa_conv = classify_oa_cut_convexity(model)
    is_convex = oa_conv.objective_is_convex and all(oa_conv.constraint_mask)

    # Master-only constraints (no recourse variable) become linear master rows;
    # constraints touching recourse variables live in the recourse NLP. A
    # master-only nonlinear constraint is unsupported in v1.
    A_m_rows: list[np.ndarray] = []
    b_m_rows: list[float] = []  # A_m x <= b_m
    sub_mask = np.zeros(n_vars, dtype=bool)
    sub_mask[scols] = True
    for c in model._constraints:
        if not isinstance(c, Constraint):
            raise NotImplementedError(
                f"GBD supports only algebraic constraints (got {type(c).__name__})."
            )
        coeffs = _extract_body_coeffs(c.body, model, n_vars) if _is_linear(c.body) else None
        if coeffs is None:
            continue  # nonlinear: handled inside the recourse NLP
        vec = np.asarray(coeffs[0], dtype=np.float64)
        if np.any(np.abs(vec[sub_mask]) > 0):
            continue  # touches a recourse variable: handled inside the recourse NLP
        # master-only linear row
        off = float(coeffs[1])
        mrow = vec[mcols]
        if c.sense == "<=":
            A_m_rows.append(mrow)
            b_m_rows.append(-off)
        elif c.sense == ">=":
            A_m_rows.append(-mrow)
            b_m_rows.append(off)
        else:  # ==
            A_m_rows.append(mrow)
            b_m_rows.append(-off)
            A_m_rows.append(-mrow)
            b_m_rows.append(off)

    A_m = np.array(A_m_rows) if A_m_rows else np.zeros((0, n_master))
    b_m = np.array(b_m_rows) if b_m_rows else np.zeros(0)

    master_bounds = [(float(lb_all[i]), float(ub_all[i])) for i in mcols]
    all_binary_master = bool(
        np.all(master_int) and all(lb_all[i] >= -1e-9 and ub_all[i] <= 1 + 1e-9 for i in mcols)
    )

    # Accumulated cuts on (x, eta): coeff_x (n_master,), coeff_eta, rhs (<=).
    cut_x: list[np.ndarray] = []
    cut_eta: list[float] = []
    cut_rhs: list[float] = []

    def _solve_master(with_eta: bool):
        ncol = n_master + (1 if with_eta else 0)
        c = np.zeros(ncol)
        if with_eta:
            c[-1] = 1.0  # min eta
        rows, rhs = [], []
        if A_m.shape[0]:
            pad = np.zeros((A_m.shape[0], ncol))
            pad[:, :n_master] = A_m
            rows.append(pad)
            rhs.append(b_m)
        if cut_x:
            cm = np.zeros((len(cut_x), ncol))
            for k in range(len(cut_x)):
                cm[k, :n_master] = cut_x[k]
                if with_eta:
                    cm[k, -1] = cut_eta[k]
            rows.append(cm)
            rhs.append(np.array(cut_rhs))
        A_ub = np.vstack(rows) if rows else None
        b_ub = np.concatenate(rhs) if rhs else None
        bounds = list(master_bounds)
        integrality = np.zeros(ncol, dtype=np.int32)
        integrality[:n_master] = master_int.astype(np.int32)
        if with_eta:
            bounds.append((_ETA_FLOOR, _BIG))
        return milp(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            integrality=integrality,
            time_limit=max(1.0, time_limit - (time.time() - t0)),
            gap_tolerance=gap_tolerance,
        )

    def _recourse(x_hat: np.ndarray):
        """Fix master vars at x̂, solve the recourse NLP.

        Returns ('opt', v, x_full, s) with v = recourse value, x_full the full
        primal, s the value-function subgradient over master cols; or
        ('infeas', None, None, None).
        """
        sub_lb = lb_all.copy()
        sub_ub = ub_all.copy()
        sub_lb[mcols] = x_hat
        sub_ub[mcols] = x_hat
        from discopt.solvers.oa import _BoundsProxy, _is_primal_feasible

        if nlp_solver == "ipopt":
            from discopt.solvers.nlp_ipopt import solve_nlp
        else:
            from discopt.solvers.nlp_pounce import solve_nlp

        proxy = _BoundsProxy(evaluator, sub_lb, sub_ub)
        x0 = np.clip(0.5 * (sub_lb + sub_ub), -1e8, 1e8)
        try:
            res = solve_nlp(proxy, x0, options={"print_level": 0, "max_iter": 300})  # type: ignore[arg-type]
        except Exception:
            return "infeas", None, None, None

        feasible = res.x is not None and (
            res.status == SolveStatus.OPTIMAL or _is_primal_feasible(evaluator, res.x)
        )
        if not feasible:
            return "infeas", None, None, None

        x_full = np.asarray(res.x, dtype=np.float64)
        v = float(evaluator.evaluate_objective(x_full))
        mu = (
            np.asarray(res.multipliers, dtype=np.float64)
            if res.multipliers is not None
            else np.zeros(evaluator.n_constraints)
        )
        grad = np.asarray(evaluator.evaluate_gradient(x_full), dtype=np.float64)
        if evaluator.n_constraints and mu.size == evaluator.n_constraints:
            jac = np.asarray(evaluator.evaluate_jacobian(x_full), dtype=np.float64)
            grad_lag = grad + jac.T @ mu
        else:
            grad_lag = grad
        s = grad_lag[mcols]
        return "opt", v, x_full, s

    def _add_opt_cut(x_hat, v, s):
        # eta >= v + s^T (x - x̂)  ->  s^T x - eta <= s^T x̂ - v
        cut_x.append(s.copy())
        cut_eta.append(-1.0)
        cut_rhs.append(float(s @ x_hat) - v)

    def _add_nogood_cut(x_hat) -> bool:
        # Exclude a 0/1 master point with infeasible recourse:
        # sum_{x̂=1}(1-x_j) + sum_{x̂=0} x_j >= 1
        #   -> sum_{x̂=1} x_j - sum_{x̂=0} x_j <= (#ones) - 1
        if not all_binary_master:
            return False
        z = np.round(x_hat).astype(int)
        a = np.where(z == 1, 1.0, -1.0)
        cut_x.append(a)
        cut_eta.append(0.0)
        cut_rhs.append(float(z.sum()) - 1.0)
        return True

    # ── initialize from a feasible master point (no eta) ──
    init = _solve_master(with_eta=False)
    if init.status == SolveStatus.INFEASIBLE:
        return SolveResult(status="infeasible", wall_time=time.time() - t0, gap_certified=True)
    if init.status != SolveStatus.OPTIMAL or init.x is None:
        return SolveResult(status="error", wall_time=time.time() - t0)
    x_hat = np.asarray(init.x[:n_master], dtype=np.float64)

    best_ub = np.inf
    incumbent_full: np.ndarray | None = None
    status = "iteration_limit"

    kind, v, x_full, s = _recourse(x_hat)
    if kind == "opt":
        _add_opt_cut(x_hat, v, s)
        best_ub = v
        incumbent_full = x_full
    elif not _add_nogood_cut(x_hat):
        raise NotImplementedError(
            "GBD recourse is infeasible at a non-binary first-stage point and no "
            "feasibility cut is available; GBD v1 supports infeasible recourse only "
            "for 0/1 first-stage variables (relatively complete recourse otherwise)."
        )

    for _it in range(max_iterations):
        if time.time() - t0 > time_limit:
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
        lower_bound = float(lb) if lb is not None else None

        kind, v, x_full, s = _recourse(x_hat)
        if kind == "opt":
            if v < best_ub:
                best_ub = v
                incumbent_full = x_full
            _add_opt_cut(x_hat, v, s)
        elif not _add_nogood_cut(x_hat):
            raise NotImplementedError(
                "GBD recourse infeasible at a non-binary first-stage point; "
                "no feasibility cut available (GBD v1)."
            )

        if np.isfinite(best_ub) and lower_bound is not None:
            gap = (best_ub - lower_bound) / (abs(best_ub) + 1e-10)
            if gap <= gap_tolerance:
                status = "optimal"
                break

    # Final master lower bound.
    bound = None
    if is_convex:
        final = _solve_master(with_eta=True)
        if final.status == SolveStatus.OPTIMAL:
            lb = final.bound if final.bound is not None else final.objective
            if lb is not None:
                bound = float(lb)

    if status == "infeasible":
        return SolveResult(status="infeasible", wall_time=time.time() - t0, gap_certified=True)

    objective = None if not np.isfinite(best_ub) else best_ub
    if status == "iteration_limit" and objective is not None and bound is not None:
        if (objective - bound) / (abs(objective) + 1e-10) <= gap_tolerance:
            status = "optimal"

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
        gap_certified=(status == "optimal" and is_convex),
    )


__all__ = ["solve_gbd"]
