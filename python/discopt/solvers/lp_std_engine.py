"""The raw-LP engine seam under ``lp_milp_highs.solve_lp_std`` (highs-parity P1.1).

``solve_lp_std`` certifies an LP answer with a ladder of checks -- ``readback_problem``,
``feasibility_problem``, ``certify_lp_dual`` (NS bound over the declared box, the FBBT box,
exact dual correction), ``farkas_verified``, ``primal_ray_verified`` and the phase-1 proof --
that read only the standard form ``sf`` and the vectors an engine hands back (``x``, the row
dual ``y``, a dual or primal ray). Measured at P1.1's entry: none of those helpers touches
an engine object. Only the raw solve does, so only the raw solve sits behind this protocol:

* :class:`HighsEngine` -- the exact calls ``solve_lp_std`` made before the seam (same
  options, same order, rays fetched lazily), so ``engine="highs"`` is byte-for-byte the
  shipped route;
* :class:`RustEngine` -- the in-house warm-startable simplex
  (``solve_lp_warm_csc_py``), which solves ``A x = b, xl <= x <= xu`` directly. It has no
  presolve; its optimal ``dual`` is ``y = B⁻ᵀc_B``, its infeasible ``dual`` a Farkas
  candidate (either orientation; ``farkas_verified`` tries both) and its ``ray`` a primal
  ray candidate -- every one of them a *candidate* the ladder verifies, never trusted.

Engine-neutral statuses: ``optimal``, ``infeasible``, ``unbounded``, ``undecided``
(infeasible-or-unbounded), ``time_limit``, ``iteration_limit``, ``empty`` (a model with no
columns and no rows), ``error``. ``name`` keeps the engine's own status string for
provenance (``highs_status``).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import numpy as np

from discopt import _timing

_HIGHS_STATUS = {
    "kOptimal": "optimal",
    "kInfeasible": "infeasible",
    "kUnbounded": "unbounded",
    "kUnboundedOrInfeasible": "undecided",
    "kTimeLimit": "time_limit",
    "kIterationLimit": "iteration_limit",
    "kModelEmpty": "empty",
}


@dataclass
class LpRaw:
    """One engine run on one standard form, before any certificate."""

    status: str
    name: str
    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    rc: Optional[np.ndarray] = None
    #: The primal readback on an unbounded exit, when the engine says it is feasible.
    x_primal: Optional[np.ndarray] = None
    iters: float = 0.0
    run_error: bool = False
    pass_warning: bool = False
    pass_error: str = ""
    version: Optional[float] = None
    _dual_ray: Callable[[], Optional[np.ndarray]] = field(default=lambda: None, repr=False)
    _primal_ray: Callable[[], Optional[np.ndarray]] = field(default=lambda: None, repr=False)

    def dual_ray(self) -> Optional[np.ndarray]:
        return self._dual_ray()

    def primal_ray(self) -> Optional[np.ndarray]:
        return self._primal_ray()


class LpEngine(Protocol):
    name: str
    has_presolve: bool

    def solve(
        self,
        sf: Any,
        *,
        time_limit: Optional[float],
        presolve: Optional[bool],
        tight_tolerances: bool,
    ) -> LpRaw: ...


class HighsEngine:
    """Today's HiGHS calls. ``presolve=None`` leaves HiGHS's own default (as the phase-1 and
    recession LPs always did); ``tight_tolerances`` adds the 1e-7 primal/dual feasibility
    tolerances the main solve and the phase-1 LP set."""

    name = "highs"
    has_presolve = True

    def solve(self, sf, *, time_limit, presolve, tight_tolerances) -> LpRaw:
        from discopt.solvers.lp_milp_highs import (
            _new_highs,
            _pass_model,
            _ray,
            _status_name,
            _version_number,
            require_highspy,
        )

        highspy = require_highspy()
        opts: list[tuple[str, Any]] = []
        if tight_tolerances:
            opts += [("primal_feasibility_tolerance", 1e-7), ("dual_feasibility_tolerance", 1e-7)]
        opts.append(("run_crossover", "on"))
        if time_limit is not None:
            opts.append(("time_limit", float(time_limit)))
        if presolve is False:
            opts.append(("presolve", "off"))
        h = _new_highs(highspy, opts)
        version = _version_number(h)
        pass_st, pass_why = _pass_model(h, highspy, sf, integer=False)
        if pass_st == highspy.HighsStatus.kError:
            return LpRaw(
                "error", "", pass_error=pass_why or "HiGHS passModel error", version=version
            )
        run_st = h.run()
        name = _status_name(h)
        info = h.getInfo()
        raw = LpRaw(
            _HIGHS_STATUS.get(name, "error"),
            name,
            iters=float(info.simplex_iteration_count),
            run_error=run_st == highspy.HighsStatus.kError,
            pass_warning=bool(pass_why),
            version=version,
            _dual_ray=lambda: _ray(h.getDualRay),
            _primal_ray=lambda: _ray(h.getPrimalRay),
        )
        if name == "kUnknown" and _both_solutions_feasible(info, highspy):
            # #2221: highspy 1.16 reports kUnknown where 1.15 reported kOptimal for the
            # same primal/dual pair. Measured on the root LP of
            # ``test_sweep_rust_presolve_fbbt_rounding`` (``x0 + z + sum y == 2^40``,
            # ``y_i <= 2^-14``): both versions return primal and dual solution status
            # feasible with max primal/dual infeasibility 0.0 and objective 1.0, under
            # every presolve/tolerance setting; 1.15.1 labels it kOptimal, 1.16.0
            # kUnknown. The label was never the certificate: every consumer of an
            # "optimal" raw answer re-verifies the point against ``sf``
            # (``readback_problem``/``feasibility_problem``), takes its bound from the
            # NS-safe bound of the duals, which is valid for ANY dual vector, and
            # certifies only when that bound meets the objective. So the pair is handed
            # to the same checks; ``name`` keeps "kUnknown" for the diagnostics. A
            # kUnknown without both solutions stays an error.
            status = "optimal"
        else:
            status = _HIGHS_STATUS.get(name, "error")
        raw.status = status
        if status == "optimal":
            sol = h.getSolution()
            raw.x = np.asarray(sol.col_value, dtype=np.float64)
            raw.y = np.asarray(sol.row_dual, dtype=np.float64)
            raw.rc = np.asarray(sol.col_dual, dtype=np.float64)
        elif name == "kUnbounded":
            if info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible:
                raw.x_primal = np.asarray(h.getSolution().col_value, dtype=np.float64)
        return raw


def _both_solutions_feasible(info: Any, highspy: Any) -> bool:
    """Whether HiGHS holds a primal AND a dual solution it calls feasible."""
    feasible = highspy.SolutionStatus.kSolutionStatusFeasible
    return bool(info.primal_solution_status == feasible and info.dual_solution_status == feasible)


class RustEngine:
    """The in-house simplex on ``A x = b, xl <= x <= xu`` (cold start, no presolve)."""

    name = "rust"
    has_presolve = False
    max_iter = 100_000

    def solve(self, sf, *, time_limit, presolve, tight_tolerances) -> LpRaw:
        if sf.n == 0 and sf.m == 0:
            return LpRaw("empty", "rust:empty")
        t0 = time.perf_counter()
        raw = self._solve(sf.c, sf.A, sf.b, sf.xl, sf.xu, time_limit)
        if raw.name != "rust:numerical" or not _equilibrated_retry_enabled():
            return raw
        # The simplex broke down numerically ("numerical" -> ``undecided``). Measured on
        # ex14_1_7's McCormick root LP (658 rows; |b| to 2e11, coefficients 9e-11..2e8,
        # finite bounds to 2e9): the cold solve breaks down at 200 pivots with or
        # without a deadline and at tol 1e-7, HiGHS calls it optimal (0), and the same
        # solve on an exact power-of-two equilibration of it is optimal in 325 pivots
        # (original-space residual 6e-8). So retry once on that equilibration. The
        # transform is exact (x = C x', y = R y'), and nothing it returns is trusted:
        # ``solve_lp_std`` re-verifies the point, the NS bound and any ray against
        # ``sf`` itself.
        rem = None if time_limit is None else float(time_limit) - (time.perf_counter() - t0)
        if rem is not None and rem <= 0.0:
            return raw
        RUST_EQUIL_RETRY_STATS["attempts"] += 1
        R, C, As = _pow2_equilibration(sf.A)
        with np.errstate(over="ignore", invalid="ignore"):
            xl = np.where(np.isfinite(sf.xl), sf.xl / C, sf.xl)
            xu = np.where(np.isfinite(sf.xu), sf.xu / C, sf.xu)
        scaled = self._solve(C * sf.c, As, R * sf.b, xl, xu, rem)
        if scaled.name == "rust:numerical":
            return raw
        RUST_EQUIL_RETRY_STATS["decided"] += 1
        out = LpRaw(
            scaled.status,
            scaled.name + "+equilibrated",
            iters=raw.iters + scaled.iters,
            _dual_ray=lambda: _mul(R, scaled.dual_ray()),
            _primal_ray=lambda: _mul(C, scaled.primal_ray()),
        )
        if scaled.status == "optimal":
            out.x = C * scaled.x
            out.y = R * scaled.y
            out.rc = np.asarray(sf.c, dtype=np.float64) - sf.A.T @ out.y
        elif scaled.x_primal is not None:
            out.x_primal = C * scaled.x_primal
        return out

    def _solve(self, c, A, b, xl, xu, time_limit) -> LpRaw:
        from discopt._rust import solve_lp_warm_csc_py

        m, n = A.shape
        A = A.tocsc()
        kw: dict[str, Any] = {"max_iter": self.max_iter}
        if time_limit is not None:
            kw["time_limit_s"] = max(0.0, float(time_limit))
        with _timing.charge("rust"):
            status, x, _obj, iters, _cs, _bv, dual, ray = solve_lp_warm_csc_py(
                np.ascontiguousarray(c, dtype=np.float64),
                m,
                n,
                np.ascontiguousarray(A.indptr, dtype=np.int64),
                np.ascontiguousarray(A.indices, dtype=np.int64),
                np.ascontiguousarray(A.data, dtype=np.float64),
                np.ascontiguousarray(b, dtype=np.float64),
                np.ascontiguousarray(xl, dtype=np.float64),
                np.ascontiguousarray(xu, dtype=np.float64),
                None,
                None,
                **kw,
            )
        name = f"rust:{status}"
        if status == "iter_limit":
            # A deadline exit reports ``iter_limit`` short of the pivot cap (lp_simplex).
            st = "time_limit" if int(iters) < self.max_iter else "iteration_limit"
        else:
            st = {"optimal": "optimal", "infeasible": "infeasible", "unbounded": "unbounded",
                  "numerical": "undecided"}.get(status, "error")  # fmt: skip
        dual_a = np.asarray(dual, dtype=np.float64)
        ray_a = np.asarray(ray, dtype=np.float64)
        raw = LpRaw(
            st,
            name,
            iters=float(iters),
            _dual_ray=lambda: dual_a.copy() if st == "infeasible" and dual_a.size else None,
            _primal_ray=lambda: ray_a.copy() if st == "unbounded" and ray_a.size else None,
        )
        if st == "optimal":
            raw.x = np.asarray(x, dtype=np.float64)
            raw.y = dual_a.copy()
            raw.rc = np.asarray(c, dtype=np.float64) - A.T @ raw.y
        elif st == "unbounded" and np.asarray(x).shape == (n,):
            # A candidate; ``solve_lp_std`` re-checks it as a primal point before use.
            raw.x_primal = np.asarray(x, dtype=np.float64)
        return raw


#: Process-wide counts of the numerical-breakdown retry: ``attempts`` (the cold solve broke
#: down and the retry ran) and ``decided`` (the equilibrated solve did not break down).
RUST_EQUIL_RETRY_STATS: dict[str, int] = {"attempts": 0, "decided": 0}


def _equilibrated_retry_enabled() -> bool:
    """Retry a numerical breakdown of the Rust engine on an exact equilibration
    (``DISCOPT_LP_STD_EQUIL_RETRY``; ``=0`` opts out). Bound-changing only in that a
    solve that was an ``error`` can now return a verified answer; the certificate
    ladder in ``solve_lp_std`` is unchanged and still judges it against ``sf``.

    Default ON from its ON/OFF panel (2026-11-15; ``issue2115_graduation_panel.py
    --flag DISCOPT_LP_STD_EQUIL_RETRY --time-limit 45``, 74 MINLP-corpus instances, 812
    checks): 0 violations, 0 looser bounds, certified 58 -> 59 (ex14_1_7: feasible at 45 s
    with bound -341 -> optimal in 3 nodes), fired on ex14_1_7 and nvs05 only, total wall
    953 -> 911 s."""
    return os.environ.get("DISCOPT_LP_STD_EQUIL_RETRY", "1") != "0"


def _pow2_equilibration(A, iters: int = 20):
    """``(R, C, R·A·C)``: Ruiz infinity-norm row/column sweeps with every factor a
    power of two, so the scaling is exact in floating point (the same scheme as
    ``_relax.milp_relaxation.equilibrate_relaxation_lp``, on an equality form)."""
    import scipy.sparse as sp

    M = sp.csr_matrix(A, dtype=np.float64, copy=True)
    m, n = M.shape
    R = np.ones(m)
    C = np.ones(n)
    for _ in range(iters):
        rmax = abs(M).max(axis=1).toarray().ravel()
        rmax[rmax == 0.0] = 1.0
        dr = 2.0 ** np.round(np.log2(1.0 / np.sqrt(rmax)))
        M = sp.diags(dr) @ M
        R *= dr
        cmax = abs(M).max(axis=0).toarray().ravel()
        cmax[cmax == 0.0] = 1.0
        dc = 2.0 ** np.round(np.log2(1.0 / np.sqrt(cmax)))
        M = M @ sp.diags(dc)
        C *= dc
    return R, C, M.tocsc()


def _mul(s: np.ndarray, v: Optional[np.ndarray]) -> Optional[np.ndarray]:
    return None if v is None else s * v


ENGINES: dict[str, LpEngine] = {"highs": HighsEngine(), "rust": RustEngine()}


def get_engine(name: str) -> LpEngine:
    try:
        return ENGINES[name]
    except KeyError:
        raise ValueError(f"LP engine {name!r}: expected one of {sorted(ENGINES)}") from None
