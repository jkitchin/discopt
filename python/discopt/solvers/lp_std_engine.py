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

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

import numpy as np

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
        if name == "kOptimal":
            sol = h.getSolution()
            raw.x = np.asarray(sol.col_value, dtype=np.float64)
            raw.y = np.asarray(sol.row_dual, dtype=np.float64)
            raw.rc = np.asarray(sol.col_dual, dtype=np.float64)
        elif name == "kUnbounded":
            if info.primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible:
                raw.x_primal = np.asarray(h.getSolution().col_value, dtype=np.float64)
        return raw


class RustEngine:
    """The in-house simplex on ``A x = b, xl <= x <= xu`` (cold start, no presolve)."""

    name = "rust"
    has_presolve = False
    max_iter = 100_000

    def solve(self, sf, *, time_limit, presolve, tight_tolerances) -> LpRaw:
        from discopt._rust import solve_lp_warm_csc_py

        if sf.n == 0 and sf.m == 0:
            return LpRaw("empty", "rust:empty")
        A = sf.A.tocsc()
        kw: dict[str, Any] = {"max_iter": self.max_iter}
        if time_limit is not None:
            kw["time_limit_s"] = max(0.0, float(time_limit))
        status, x, _obj, iters, _cs, _bv, dual, ray = solve_lp_warm_csc_py(
            np.ascontiguousarray(sf.c, dtype=np.float64),
            sf.m,
            sf.n,
            np.ascontiguousarray(A.indptr, dtype=np.int64),
            np.ascontiguousarray(A.indices, dtype=np.int64),
            np.ascontiguousarray(A.data, dtype=np.float64),
            np.ascontiguousarray(sf.b, dtype=np.float64),
            np.ascontiguousarray(sf.xl, dtype=np.float64),
            np.ascontiguousarray(sf.xu, dtype=np.float64),
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
            raw.rc = np.asarray(sf.c, dtype=np.float64) - sf.A.T @ raw.y
        elif st == "unbounded" and np.asarray(x).shape == (sf.n,):
            # A candidate; ``solve_lp_std`` re-checks it as a primal point before use.
            raw.x_primal = np.asarray(x, dtype=np.float64)
        return raw


ENGINES: dict[str, LpEngine] = {"highs": HighsEngine(), "rust": RustEngine()}


def get_engine(name: str) -> LpEngine:
    try:
        return ENGINES[name]
    except KeyError:
        raise ValueError(f"LP engine {name!r}: expected one of {sorted(ENGINES)}") from None
