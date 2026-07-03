"""Feasibility-phase NLP for the (Generalized) Benders recourse subproblem.

The GBD recourse solve at a fixed first-stage point ``x̂`` can fail for two
distinct reasons that must never be conflated (see the decomposition remediation
plan, C1):

- the recourse is **genuinely infeasible** — no ``y`` satisfies ``g(x̂, y) <= 0``
  over the recourse box, so ``x̂`` cannot be part of any solution and may be
  excluded (a no-good / feasibility cut is sound); or
- the NLP solver **failed** (raised, stalled, diverged) at a point whose recourse
  is actually feasible — excluding ``x̂`` here can cut off the optimum.

The recourse NLP itself cannot tell these apart: the backend maps
"Infeasible_Problem_Detected" to ``ERROR`` (it is a *local* verdict only; see
``nlp_ipopt._IPOPT_STATUS_MAP``), indistinguishable from a numerical failure.

This module builds an **elastic phase-1 NLP** whose optimum certifies the
answer. With the recourse constraints written in ``discopt``'s row form
``cl <= g(x) <= cu`` and the master variables pinned (``lb == ub`` on the master
columns), the phase-1 problem is the epigraph of the maximum constraint
violation::

    min_{y, t}  t
    s.t.        g_i(x̂, y) - cu_i <= t     (rows with a finite upper bound)
                cl_i - g_i(x̂, y) <= t     (rows with a finite lower bound)
                t >= 0,  y in the recourse box.

Its optimum ``t*`` is the minimum achievable maximum violation:

- ``t* <= feas_tol`` → the recourse is **feasible** at ``x̂`` (a point with
  violation ``<= t*`` exists); the original solve failed for another reason.
- ``t* > feas_tol`` (and the phase-1 solve itself converged) → the recourse is
  **infeasible**; ``x̂`` may be excluded. On a *convex* model the phase-1 problem
  is convex, so this verdict is rigorous.

The returned multipliers ``mu`` on the active violation rows are what Phase 3's
Geoffrion feasibility cut consumes: for any master point ``x`` it holds that
``min_y mu^T g_raw(x, y) <= 0`` fails to hold at ``x̂`` and separates it.
"""

from __future__ import annotations

import logging

import numpy as np

from discopt.solvers import SolveStatus

logger = logging.getLogger(__name__)

_BIG = 1e20
_T_UB = 1e8  # finite box on the elastic variable keeps the IPM well-posed


class FeasibilityPhaseEvaluator:
    """Elastic phase-1 wrapper adding one violation variable ``t``.

    Wraps a base :class:`~discopt._jax.nlp_evaluator.NLPEvaluator` (with the
    master columns already pinned via its variable bounds) and presents the
    epigraph phase-1 problem described in the module docstring. Implements only
    the callback surface the NLP backends use
    (:class:`discopt.solvers.nlp_ipopt._IpoptCallbacks` plus the three scalars
    ``n_variables`` / ``n_constraints`` / ``variable_bounds``), so it can be
    handed straight to ``solve_nlp``.
    """

    def __init__(
        self,
        base,
        cl: np.ndarray,
        cu: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> None:
        self._base = base
        self._n = int(base.n_variables)
        self._lb = np.asarray(lb, dtype=np.float64).copy()
        self._ub = np.asarray(ub, dtype=np.float64).copy()
        cl = np.asarray(cl, dtype=np.float64)
        cu = np.asarray(cu, dtype=np.float64)

        # One elastic row per finite constraint bound. ``coef_sign`` multiplies
        # the raw constraint body g_i; ``const`` is the row's constant so that
        # the row value is ``coef_sign * g_i + const - t`` (all rows are <= 0).
        self._base_row: list[int] = []
        self._coef_sign: list[float] = []
        self._const: list[float] = []
        for i in range(cl.size):
            if cu[i] < _BIG:  # g_i <= cu_i  ->  g_i - cu_i - t <= 0
                self._base_row.append(i)
                self._coef_sign.append(1.0)
                self._const.append(-float(cu[i]))
            if cl[i] > -_BIG:  # g_i >= cl_i  ->  cl_i - g_i - t <= 0
                self._base_row.append(i)
                self._coef_sign.append(-1.0)
                self._const.append(float(cl[i]))
        self._m = len(self._base_row)
        self._m_base = int(cl.size)
        self._sign_arr = np.asarray(self._coef_sign, dtype=np.float64)
        self._const_arr = np.asarray(self._const, dtype=np.float64)

    # ── scalars / bounds the backend reads directly ──────────────
    @property
    def n_variables(self) -> int:
        return self._n + 1

    @property
    def n_constraints(self) -> int:
        return self._m

    @property
    def variable_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lb = np.concatenate([self._lb, [0.0]])
        ub = np.concatenate([self._ub, [_T_UB]])
        return lb, ub

    def has_sparse_structure(self) -> bool:
        # Force the dense callback path; this wrapper returns dense Jacobian and
        # Hessian only.
        return False

    def constraint_bounds(self) -> list[tuple[float, float]]:
        """Explicit ``(cl, cu)`` per elastic row: every row is ``<= 0``."""
        return [(-_BIG, 0.0)] * self._m

    # ── callback surface ─────────────────────────────────────────
    def _split(self, z: np.ndarray) -> tuple[np.ndarray, float]:
        z = np.asarray(z, dtype=np.float64)
        return z[: self._n], float(z[self._n])

    def evaluate_objective(self, z: np.ndarray) -> float:
        return float(np.asarray(z, dtype=np.float64)[self._n])

    def evaluate_gradient(self, z: np.ndarray) -> np.ndarray:
        g = np.zeros(self._n + 1, dtype=np.float64)
        g[self._n] = 1.0
        return g

    def evaluate_constraints(self, z: np.ndarray) -> np.ndarray:
        x, t = self._split(z)
        if self._m == 0:
            return np.zeros(0, dtype=np.float64)
        g = np.asarray(self._base.evaluate_constraints(x), dtype=np.float64)
        return self._sign_arr * g[self._base_row] + self._const_arr - t

    def evaluate_jacobian(self, z: np.ndarray) -> np.ndarray:
        x, _ = self._split(z)
        jac = np.zeros((self._m, self._n + 1), dtype=np.float64)
        if self._m == 0:
            return jac
        base_jac = np.asarray(self._base.evaluate_jacobian(x), dtype=np.float64)
        for k, (i, sgn) in enumerate(zip(self._base_row, self._coef_sign)):
            jac[k, : self._n] = sgn * base_jac[i, :]
            jac[k, self._n] = -1.0
        return jac

    def evaluate_lagrangian_hessian(
        self, z: np.ndarray, obj_factor: float, lambda_: np.ndarray
    ) -> np.ndarray:
        # The phase-1 objective (t) is linear, so obj_factor drops out. Each
        # elastic row's Hessian is ``coef_sign_k * H(g_{base_row_k})``; map the
        # elastic multipliers back onto the base rows with their signs.
        x, _ = self._split(z)
        H = np.zeros((self._n + 1, self._n + 1), dtype=np.float64)
        if self._m == 0:
            return H
        lam = np.asarray(lambda_, dtype=np.float64)
        base_lam = np.zeros(self._m_base, dtype=np.float64)
        for k, (i, sgn) in enumerate(zip(self._base_row, self._coef_sign)):
            base_lam[i] += sgn * lam[k]
        H[: self._n, : self._n] = np.asarray(
            self._base.evaluate_lagrangian_hessian(x, 0.0, base_lam), dtype=np.float64
        )
        return H


def certify_recourse_feasibility(
    base,
    cl: np.ndarray,
    cu: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    *,
    nlp_solver: str = "pounce",
    feas_tol: float = 1e-6,
    x0: np.ndarray | None = None,
):
    """Solve the elastic phase-1 NLP and classify the recourse at the fixed x̂.

    Returns ``(verdict, info)`` where ``verdict`` is one of:

    - ``"feasible"``   — ``t* <= feas_tol``; the recourse is feasible (the base
      solve failed for a non-infeasibility reason);
    - ``"infeasible"`` — ``t* > feas_tol`` and the phase-1 solve converged; the
      recourse is infeasible and ``x̂`` may be excluded;
    - ``"unknown"``    — the phase-1 solve itself failed to converge; no verdict
      (the caller must not exclude ``x̂``).

    ``info`` carries ``t`` (the certified max violation, when known), ``mu`` (the
    elastic-row multipliers, for Phase-3 feasibility cuts), and the constructed
    ``evaluator`` (so the cut builder can reuse its row map).
    """
    ev = FeasibilityPhaseEvaluator(base, cl, cu, lb, ub)
    if ev.n_constraints == 0:
        # No constraints on the recourse: it is trivially feasible over the box.
        return "feasible", {"t": 0.0, "mu": None, "evaluator": ev}

    if nlp_solver == "ipopt":
        from discopt.solvers.nlp_ipopt import solve_nlp
    else:
        from discopt.solvers.nlp_pounce import solve_nlp

    p_lb, p_ub = ev.variable_bounds
    if x0 is None:
        z0 = np.clip(0.5 * (p_lb + p_ub), -1e8, 1e8)
    else:
        z0 = np.concatenate([np.asarray(x0, dtype=np.float64)[: ev._n], [0.0]])
    # Seed t at the violation of the midpoint so the IPM starts feasible in t.
    try:
        c0 = ev.evaluate_constraints(z0)
        z0[ev._n] = max(0.0, float(np.max(c0 + z0[ev._n])) if c0.size else 0.0)
    except Exception:  # noqa: BLE001 - evaluation is best-effort for the seed
        pass

    try:
        res = solve_nlp(
            ev,  # type: ignore[arg-type]  # duck-typed callback surface
            z0,
            constraint_bounds=ev.constraint_bounds(),
            options={"print_level": 0, "max_iter": 300},
        )
    except Exception as exc:  # noqa: BLE001 - a failed phase-1 yields no verdict
        logger.debug("feasibility phase-1 NLP raised: %s", exc)
        return "unknown", {"t": None, "mu": None, "evaluator": ev}

    converged = res.x is not None and res.status in (
        SolveStatus.OPTIMAL,
        SolveStatus.ITERATION_LIMIT,
    )
    if not converged or res.x is None:
        return "unknown", {"t": None, "mu": None, "evaluator": ev}

    t_star = float(np.asarray(res.x, dtype=np.float64)[ev._n])
    mu = (
        np.asarray(res.multipliers, dtype=np.float64)
        if res.multipliers is not None and len(res.multipliers) == ev.n_constraints
        else None
    )
    if t_star <= feas_tol:
        return "feasible", {"t": t_star, "mu": mu, "evaluator": ev}
    # A positive certified violation only excludes a point when the phase-1
    # solve actually reached optimality (a stalled phase-1 might report a
    # spuriously high t). Require OPTIMAL for the infeasible verdict.
    if res.status == SolveStatus.OPTIMAL:
        return "infeasible", {"t": t_star, "mu": mu, "evaluator": ev}
    return "unknown", {"t": t_star, "mu": mu, "evaluator": ev}


__all__ = ["FeasibilityPhaseEvaluator", "certify_recourse_feasibility"]
