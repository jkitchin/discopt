"""The shared loop of the native GDP algorithms (#1775 G3.1, spec §5.1 and §5.4).

:class:`GDPAlgorithm` owns what every native method (LOA, GLOA, RIC, enumerate,
LBB, LD-SDA) must get right the same way:

* LB and UB in the minimization convention; ``obj_sign`` is applied only where
  the :class:`~discopt.modeling.core.SolveResult` is built (as in #1611);
* one :class:`Deadline`, whose remainder is the time limit of every master and
  subproblem call (Pyomo's GDPopt checks its limit only between subsolves);
* the solved and unresolved configuration sets (the C-35 discipline) and the
  counters and per-iteration LB/UB trace;
* incumbents verified on the original model -- its rows, bounds, integrality and
  declared logic (:func:`~discopt.validation.feasibility.verify_point` builds the
  :class:`DeclaredLogic`) -- never on a lowered model alone (#1659);
* the status rules: ``optimal`` only when the LB is valid for the method, no
  configuration is unresolved and the gap closed; ``feasible`` for any other
  verified incumbent; ``infeasible`` only on a proof with nothing unresolved;
  otherwise the budget that stopped the loop, or ``unknown``. Stopping is never
  a proof: running out of iterations or time cannot make a result ``optimal``.

A subclass implements :meth:`GDPAlgorithm.iterate` and reports through
``offer_bound``, ``offer_incumbent``, ``mark_solved``, ``mark_unresolved``,
``invalidate_bound`` and ``prove_infeasible``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Hashable
from typing import Any, Optional

import numpy as np

from discopt.modeling.core import Model, ObjectiveSense, SolveResult


class Deadline:
    """A wall-clock budget, read through :meth:`sub_limit` by every subsolve."""

    def __init__(
        self, time_limit: Optional[float], *, clock: Callable[[], float] = time.perf_counter
    ) -> None:
        if time_limit is not None and not (time_limit >= 0):
            raise ValueError(f"time_limit must be >= 0 or None, got {time_limit!r}")
        self._clock = clock
        self.start = clock()
        self.time_limit = None if time_limit is None else float(time_limit)

    def elapsed(self) -> float:
        return self._clock() - self.start

    def remaining(self) -> float:
        """Seconds left, never negative; ``inf`` without a limit."""
        if self.time_limit is None:
            return math.inf
        return max(0.0, self.time_limit - self.elapsed())

    def expired(self) -> bool:
        return self.remaining() <= 0.0

    def absolute(self) -> Optional[float]:
        """The caller's own deadline as a reading of this clock; ``None`` without one.

        For a component that takes an absolute deadline: it receives the whole of
        the caller's budget (#912's ``contract`` role), never a carved slice of it.
        """
        return None if self.time_limit is None else self.start + self.time_limit

    def sub_limit(self, cap: Optional[float] = None) -> float:
        """The time limit to give a subsolve: what is left, capped by ``cap``."""
        left = self.remaining()
        return left if cap is None else min(left, float(cap))


class GDPAlgorithm:
    """Base class of the native GDP algorithms; see the module docstring.

    Subclasses set :attr:`route` and implement :meth:`iterate`. A method whose LB
    is valid only under conditions (LOA: every outer-approximated row and the
    objective certified convex, no slacks) calls :meth:`invalidate_bound` when a
    condition fails; one with no LB at all (LD-SDA) sets ``lb_valid = False``.
    """

    route: str = "gdp"
    #: Whether the LB is valid only once every configuration is resolved. True
    #: for an LB built from configuration bounds (enumerate); False for a master
    #: relaxation that never excluded the unresolved ones (LOA, RIC), whose
    #: bound stays valid while optimality is still blocked (C-35).
    bound_needs_resolution: bool = True

    def __init__(
        self,
        model: Model,
        *,
        time_limit: Optional[float] = None,
        gap_tolerance: float = 1e-4,
        abs_gap_tolerance: float = 1e-6,
        iteration_limit: Optional[int] = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if model._objective is None:
            raise ValueError(f"{type(self).__name__}: the model has no objective.")
        if iteration_limit is not None and iteration_limit < 0:
            raise ValueError(f"iteration_limit must be >= 0 or None, got {iteration_limit!r}")
        self.model = model
        self.deadline = Deadline(time_limit, clock=clock)
        self.gap_tolerance = float(gap_tolerance)
        self.abs_gap_tolerance = float(abs_gap_tolerance)
        self.iteration_limit = iteration_limit
        self.obj_sign = -1.0 if model._objective.sense == ObjectiveSense.MAXIMIZE else 1.0
        # Minimization convention from here to _build_result.
        self.lb = -math.inf
        self.ub = math.inf
        self.lb_valid = True
        self.lb_invalid_reason = ""
        self.incumbent: Optional[np.ndarray] = None
        self.solved: set[Hashable] = set()
        self.unresolved: set[Hashable] = set()
        self.proven_infeasible = False
        self.iterations = 0
        self.n_master = 0
        self.n_subproblems = 0
        self.trace: list[tuple[int, float, float, float]] = []
        self.stop_reason = ""

    # ── reporting, for subclasses ─────────────────────────────────────────

    def offer_bound(self, lb: float) -> None:
        """A lower bound (minimization convention) from a relaxation.

        The LB only rises. A bound above a verified incumbent by more than
        rounding (``bound_inversion_tolerance``) means one of the two is wrong:
        the bound is withdrawn and nothing is certified -- never clamped into a
        certificate (#945, #1059).
        """
        lb = float(lb)
        if math.isnan(lb):
            raise ValueError("offer_bound: NaN bound")
        self.lb = max(self.lb, lb)
        self._check_crossing()

    def offer_incumbent(self, x_flat: Any) -> bool:
        """Verify ``x_flat`` on the original model; keep it when it improves UB.

        Returns whether the point was verified feasible. Its objective is the
        model's own, recomputed here, never one reported by a subsolve.
        """
        from discopt.validation.feasibility import verify_point

        x = np.asarray(x_flat, dtype=np.float64).reshape(-1)
        vr = verify_point(self.model, x, with_objective=True)
        if not vr.ok or vr.objective is None or not math.isfinite(vr.objective):
            return False
        obj = self.obj_sign * float(vr.objective)
        if obj < self.ub:
            self.ub = obj
            self.incumbent = x.copy()
            self._check_crossing()
        return True

    def _check_crossing(self) -> None:
        from discopt.solvers._gap import bound_inversion_tolerance

        if self.incumbent is None or not math.isfinite(self.lb):
            return
        if self.lb - self.ub > bound_inversion_tolerance(self.lb, self.ub, self.abs_gap_tolerance):
            self.invalidate_bound(
                f"the bound {self.lb!r} crossed the verified incumbent {self.ub!r}"
            )

    def invalidate_bound(self, reason: str) -> None:
        """The method's LB conditions failed: no bound, no ``optimal``."""
        self.lb_valid = False
        if not self.lb_invalid_reason:
            self.lb_invalid_reason = reason

    def mark_solved(self, config: Hashable) -> None:
        self.solved.add(config)
        self.unresolved.discard(config)

    def mark_unresolved(self, config: Hashable) -> None:
        """A configuration whose subproblem neither solved nor proved infeasible."""
        if config not in self.solved:
            self.unresolved.add(config)

    def prove_infeasible(self) -> None:
        """The search space is proved empty (e.g. the master is infeasible)."""
        self.proven_infeasible = True

    # ── the loop ──────────────────────────────────────────────────────────

    def iterate(self) -> bool:
        """One iteration. Return ``False`` when the search space is exhausted."""
        raise NotImplementedError

    def run(self) -> SolveResult:
        while True:
            if self.converged():
                self.stop_reason = "converged"
                break
            if self.proven_infeasible:
                self.stop_reason = "infeasible"
                break
            if self.deadline.expired():
                self.stop_reason = "time_limit"
                break
            if self.iteration_limit is not None and self.iterations >= self.iteration_limit:
                self.stop_reason = "iteration_limit"
                break
            more = self.iterate()
            self.iterations += 1
            self.trace.append((self.iterations, self.lb, self.ub, self.deadline.elapsed()))
            if not more:
                self.stop_reason = "exhausted"
                break
        return self.result()

    # ── status (spec §5.4) ───────────────────────────────────────────────

    def _published_lb(self) -> float:
        # offer_bound admits an LB up to tolerance above UB; publishing it as is
        # would report bound > objective. Clamped so bound <= incumbent exactly.
        return min(self.lb, self.ub)

    def _gap_value(self) -> float:
        """The repository's one convergence gap (``solvers._gap.optimality_gap``)."""
        from discopt.solvers._gap import optimality_gap

        return optimality_gap(self.lb, self.ub, abs_tol=self.abs_gap_tolerance, denom_floor=1e-10)

    def gap(self) -> Optional[float]:
        """The reported gap (``reported_gap``), ``None`` without a valid LB or incumbent.

        The convergence test's own arithmetic, so a gap under ``gap_tolerance``
        is one that converged.
        """
        if not self.lb_valid or self.incumbent is None or not math.isfinite(self.lb):
            return None
        return self._gap_value()

    def converged(self) -> bool:
        """Optimality proved: valid LB, nothing unresolved, gap within tolerance."""
        if self.incumbent is None or not self.lb_valid or self.unresolved:
            return False
        if not math.isfinite(self.lb):
            return False
        return self._gap_value() <= self.gap_tolerance

    def status(self) -> str:
        if self.incumbent is not None:
            if self.proven_infeasible:
                raise RuntimeError(
                    f"{type(self).__name__}: infeasibility was proved but a verified "
                    "incumbent exists; the proof is not valid."
                )
            return "optimal" if self.converged() else "feasible"
        if self.proven_infeasible and not self.unresolved:
            return "infeasible"
        if self.stop_reason in ("time_limit", "iteration_limit"):
            return self.stop_reason
        return "unknown"

    def result(self) -> SolveResult:
        status = self.status()
        # bound_valid: the LB is a valid bound. gap_certified: the result is a
        # proof -- an open gap (feasible / time_limit / iteration_limit) never is,
        # however valid its bound (SolveResult contract).
        lb_ok = self.lb_valid and not (self.bound_needs_resolution and self.unresolved)
        certified = lb_ok and not self.unresolved and status in ("optimal", "infeasible")
        bound: Optional[float] = None
        if lb_ok and math.isfinite(self.lb) and status != "infeasible":
            bound = self.obj_sign * self._published_lb()
        x = None
        objective = None
        if self.incumbent is not None:
            x = _unpack(self.model, self.incumbent)
            objective = self.obj_sign * self.ub
        stats: dict[str, Any] = {
            "route": self.route,
            "stop_reason": self.stop_reason,
            "iterations": self.iterations,
            "n_master": self.n_master,
            "n_subproblems": self.n_subproblems,
            "n_solved": len(self.solved),
            "n_unresolved": len(self.unresolved),
            "trace": [
                (k, self.obj_sign * lb, self.obj_sign * ub, t) for k, lb, ub, t in self.trace
            ],
        }
        if self.lb_invalid_reason:
            stats["lb_invalid_reason"] = self.lb_invalid_reason
        return SolveResult(
            status=status,
            objective=objective,
            bound=bound,
            gap=self.gap() if bound is not None else None,
            x=x,
            wall_time=self.deadline.elapsed(),
            gap_certified=certified,
            bound_valid=bound is not None and lb_ok,
            solver_stats=stats,
            algorithm_route=self.route,
        )


def _unpack(model: Model, x: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    offset = 0
    for v in model._variables:
        size = int(v.size)
        out[v.name] = x[offset : offset + size].reshape(v.shape)
        offset += size
    return out


__all__ = ["Deadline", "GDPAlgorithm"]
