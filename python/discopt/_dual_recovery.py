"""Active-set multiplier recovery (least squares), shared by the examiner and
the solver's dual reporting.

Given a point ``x``, a box, and the constraint rows, this builds the
first-order stationarity system over the ACTIVE set

    ∇f + Σ_act μ_i ∇g_i − λ_lb + λ_ub = 0,    μ ≥ 0 (inequalities), λ ≥ 0

and solves it with :func:`scipy.optimize.lsq_linear` under those sign bounds.
Integer columns are dropped: fixing the discrete variables and running
continuous KKT is exactly what omitting their columns does (the GAMS Examiner
"fix and re-check" convention).

Why this lives in its own module rather than inside the examiner
----------------------------------------------------------------
Two callers need the same arithmetic, and they must not drift apart:

* :mod:`discopt.validation.examiner` wraps it in ``CheckResult`` reporting;
* :mod:`discopt.solver` uses it to report duals against the box the USER
  DECLARED when presolve solved a tightened one (#1037). A bound the
  tightening introduced is not a bound of the user's model, so a multiplier
  sitting on it is not a multiplier of the user's model; the fit has to be
  redone against the declared active set, and this is the fit that does it.

The module is deliberately dependency-light (numpy + scipy only) so both the
validation layer and the solver can import it without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import lsq_linear

# Distance within which a point counts as sitting ON a bound / row. Matches
# ``examiner.ACTIVE_TOL``; kept here so the solver does not have to import the
# validation layer just to name a tolerance.
ACTIVE_TOL = 1e-6


def row_metadata(evaluator):
    """``(sense_arr, rhs_arr, row_labels)`` for the evaluator's flat row order.

    Vector-bodied constraints contribute ``sz`` rows that share the source
    constraint's sense and rhs.
    """
    senses: list[str] = []
    rhss: list[float] = []
    labels: list[str] = []
    for c, sz in zip(evaluator._source_constraints, evaluator._constraint_flat_sizes):
        sz = int(sz)
        senses.extend([c.sense] * sz)
        rhss.extend([float(c.rhs)] * sz)
        cname = getattr(c, "name", None) or repr(c.body)[:40]
        if sz > 1:
            labels.extend([f"{cname}[{i}]" for i in range(sz)])
        else:
            labels.append(str(cname))
    return np.asarray(senses), np.asarray(rhss, dtype=float), labels


@dataclass
class DualRecovery:
    """Outcome of an active-set multiplier fit.

    ``ok`` is False only when ``lsq_linear`` itself raised; a large residual is
    reported, not hidden, and is the caller's decision to act on (CLAUDE.md §7 —
    an instrument must not swallow the thing it is measuring).
    """

    ok: bool
    detail: str = ""
    # Index arrays, all into the FLAT variable / row vectors.
    cont_idx: np.ndarray = None  # type: ignore[assignment]
    row_select: np.ndarray = None  # type: ignore[assignment]
    lb_active: np.ndarray = None  # type: ignore[assignment]  # indices into cont_idx
    ub_active: np.ndarray = None  # type: ignore[assignment]  # indices into cont_idx
    # Fitted multipliers, in active-set order.
    mu_act: np.ndarray = None  # type: ignore[assignment]
    lam_lb_act: np.ndarray = None  # type: ignore[assignment]
    lam_ub_act: np.ndarray = None  # type: ignore[assignment]
    # Full-length vectors aligned with the evaluator's row order and the flat
    # variable order; zero off the active set.
    mu_full: Optional[np.ndarray] = None
    lam_lb_full: Optional[np.ndarray] = None
    lam_ub_full: Optional[np.ndarray] = None
    # Stationarity residual of the fit, over continuous columns.
    stat_resid: np.ndarray = None  # type: ignore[assignment]
    residual_max: float = float("inf")
    residual_norm: float = float("inf")
    # True when the active set was empty, so the "fit" is just ‖∇f‖ at x.
    empty_active_set: bool = False

    @property
    def n_active_cons(self) -> int:
        return 0 if self.row_select is None else int(self.row_select.size)

    @property
    def n_active_bounds(self) -> int:
        n_lb = 0 if self.lb_active is None else int(self.lb_active.size)
        n_ub = 0 if self.ub_active is None else int(self.ub_active.size)
        return n_lb + n_ub


def recover_multipliers(
    *,
    grad: np.ndarray,
    jac: np.ndarray,
    body: np.ndarray,
    sense_arr: np.ndarray,
    rhs_arr: np.ndarray,
    x_flat: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    is_continuous: np.ndarray,
    active_tol: float = ACTIVE_TOL,
) -> DualRecovery:
    """Fit KKT multipliers at ``x_flat`` against the box ``(lb, ub)``.

    ``lb``/``ub`` decide which bounds are active, and therefore which
    multipliers are allowed to be nonzero — this is the whole reason the caller
    gets to choose the box (#1037). Pass the DECLARED bounds to obtain duals of
    the user's model; pass a tightened box and you get duals of the tightened
    problem, which is a different problem.

    ``jac`` rows follow ``body``/``sense_arr``/``rhs_arr``; the returned
    ``mu_full`` uses the Lagrangian convention μ ≥ 0 for "<=", μ ≤ 0 for ">=".
    """
    cont_idx = np.where(is_continuous)[0]
    if cont_idx.size == 0:
        return DualRecovery(
            ok=True,
            detail="no continuous columns",
            cont_idx=cont_idx,
            row_select=np.zeros(0, dtype=int),
            lb_active=np.zeros(0, dtype=int),
            ub_active=np.zeros(0, dtype=int),
            mu_act=np.zeros(0),
            lam_lb_act=np.zeros(0),
            lam_ub_act=np.zeros(0),
            mu_full=np.zeros(jac.shape[0]) if jac.size else np.zeros(0),
            lam_lb_full=np.zeros(x_flat.size),
            lam_ub_full=np.zeros(x_flat.size),
            stat_resid=np.zeros(0),
            residual_max=0.0,
            residual_norm=0.0,
            empty_active_set=True,
        )

    grad_c = grad[cont_idx]
    jac_c = jac[:, cont_idx] if jac.size else np.empty((0, cont_idx.size))
    lb_c = lb[cont_idx]
    ub_c = ub[cont_idx]
    x_c = x_flat[cont_idx]

    if body.size:
        signed = body - rhs_arr
        is_le = sense_arr == "<="
        is_ge = sense_arr == ">="
        is_eq = sense_arr == "=="
        # Row activity is judged RELATIVE to the row's own scale, using the same
        # row_scale the examiner uses for scaled primal feasibility:
        # max(1, |rhs|, max|J_row| * max(1, |x|)). An absolute test measures the
        # row in whatever units it happens to be written in, so multiplying a
        # constraint through by 1000 -- which changes nothing about the problem
        # -- makes a binding row read as inactive: the NLP's own convergence
        # slack is scaled up with it. That empties the active set and the
        # multiplier of a genuinely binding row is lost.
        if jac.size:
            jac_scale = np.max(np.abs(jac) * np.maximum(1.0, np.abs(x_flat))[None, :], axis=1)
        else:
            jac_scale = np.zeros(body.size)
        row_scale = np.maximum(np.maximum(np.abs(rhs_arr), jac_scale), 1.0)
        near = np.abs(signed) <= active_tol * row_scale
        row_select = np.where((is_le & near) | (is_ge & near) | is_eq)[0]
    else:
        row_select = np.zeros(0, dtype=int)

    lb_active = np.where(np.isfinite(lb_c) & (x_c - lb_c <= active_tol))[0]
    ub_active = np.where(np.isfinite(ub_c) & (ub_c - x_c <= active_tol))[0]

    n_mu = int(row_select.size)
    n_llb = int(lb_active.size)
    n_lub = int(ub_active.size)

    if n_mu + n_llb + n_lub == 0:
        max_r = float(np.max(np.abs(grad_c))) if grad_c.size else 0.0
        return DualRecovery(
            ok=True,
            detail="no active set; ‖∇f‖∞ at returned x",
            cont_idx=cont_idx,
            row_select=row_select,
            lb_active=lb_active,
            ub_active=ub_active,
            mu_act=np.zeros(0),
            lam_lb_act=np.zeros(0),
            lam_ub_act=np.zeros(0),
            mu_full=np.zeros(jac.shape[0]) if jac.size else np.zeros(0),
            lam_lb_full=np.zeros(x_flat.size),
            lam_ub_full=np.zeros(x_flat.size),
            stat_resid=grad_c,
            residual_max=max_r,
            residual_norm=float(np.linalg.norm(grad_c)),
            empty_active_set=True,
        )

    cols = []
    var_lb_y: list[float] = []
    var_ub_y: list[float] = []
    if n_mu:
        sub_sense = sense_arr[row_select]
        sub_jac = jac_c[row_select, :].copy()
        flip = sub_sense == ">="
        sub_jac[flip, :] *= -1.0
        cols.append(sub_jac.T)
        for s in sub_sense:
            var_lb_y.append(-np.inf if s == "==" else 0.0)
            var_ub_y.append(np.inf)
    if n_llb:
        I_lb = np.zeros((cont_idx.size, n_llb))
        for k, j in enumerate(lb_active):
            I_lb[j, k] = -1.0
        cols.append(I_lb)
        var_lb_y.extend([0.0] * n_llb)
        var_ub_y.extend([np.inf] * n_llb)
    if n_lub:
        I_ub = np.zeros((cont_idx.size, n_lub))
        for k, j in enumerate(ub_active):
            I_ub[j, k] = 1.0
        cols.append(I_ub)
        var_lb_y.extend([0.0] * n_lub)
        var_ub_y.extend([np.inf] * n_lub)

    A = np.concatenate(cols, axis=1) if cols else np.zeros((cont_idx.size, 0))
    b = -grad_c

    try:
        y = lsq_linear(A, b, bounds=(np.asarray(var_lb_y), np.asarray(var_ub_y))).x
    except Exception as e:  # pragma: no cover - scipy failure path
        return DualRecovery(
            ok=False,
            detail=f"dual recovery failed: {e}",
            cont_idx=cont_idx,
            row_select=row_select,
            lb_active=lb_active,
            ub_active=ub_active,
            stat_resid=np.zeros(0),
        )

    stat_resid = A @ y - b
    mu_act = y[:n_mu] if n_mu else np.zeros(0)
    lam_lb_act = y[n_mu : n_mu + n_llb] if n_llb else np.zeros(0)
    lam_ub_act = y[n_mu + n_llb :] if n_lub else np.zeros(0)

    mu_full = np.zeros(jac.shape[0]) if jac.size else np.zeros(0)
    if n_mu:
        # mu_act is in "flipped to ≤ form"; un-flip ">=" rows back to original sign.
        mu_signed = mu_act.copy()
        mu_signed[sense_arr[row_select] == ">="] *= -1.0
        mu_full[row_select] = mu_signed
    lam_lb_full = np.zeros(x_flat.size)
    lam_ub_full = np.zeros(x_flat.size)
    for k, j in enumerate(lb_active):
        lam_lb_full[cont_idx[j]] = lam_lb_act[k]
    for k, j in enumerate(ub_active):
        lam_ub_full[cont_idx[j]] = lam_ub_act[k]

    return DualRecovery(
        ok=True,
        cont_idx=cont_idx,
        row_select=row_select,
        lb_active=lb_active,
        ub_active=ub_active,
        mu_act=mu_act,
        lam_lb_act=lam_lb_act,
        lam_ub_act=lam_ub_act,
        mu_full=mu_full,
        lam_lb_full=lam_lb_full,
        lam_ub_full=lam_ub_full,
        stat_resid=stat_resid,
        residual_max=float(np.max(np.abs(stat_resid))) if stat_resid.size else 0.0,
        residual_norm=float(np.linalg.norm(stat_resid)),
    )
