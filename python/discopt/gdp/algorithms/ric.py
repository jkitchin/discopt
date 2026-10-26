"""Relaxation with integer cuts (#1775 G4.2, spec §5.3).

:class:`RIC` is :class:`~discopt.gdp.algorithms.loa.LOA` with the outer
approximation taken out:

* **The master is the lowered model's linear rows plus no-goods.** Dropping the
  nonlinear rows only enlarges the feasible set, so the master is a relaxation
  whatever the convexity: no OA cut, no convexity pass, no NLP-relaxation seed.
* **Subproblems are global**, and a certificate -- optimal or infeasible --
  excludes a configuration (the no-good rule). Anything less leaves it
  unresolved, blocking ``optimal``; it is excluded too, so the search moves on,
  with its bound kept honest (:meth:`RIC._global_step`). Pyomo's RIC is rigorous only "if the NLP
  subproblems are solved globally" (its docstring); here they always are.

Bound (§5.4): with a linear objective the master's dual bound is a valid LB at
every iteration. With a nonlinear one the master is a feasibility problem and
bounds nothing; the LB then comes only from exhaustion -- once every
configuration is excluded, ``min`` of their certified bounds (LOA's
``excluded_floor``) -- so the bound is never withdrawn, only absent until then.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from discopt.gdp.algorithms.loa import LOA
from discopt.modeling.core import Model


@dataclass
class _NoCuts:
    """A convexity verdict that admits no OA cut (RIC's master has none)."""

    constraint_mask: list = field(default_factory=list)
    objective_is_convex: bool = False
    truncated: int = 0


class RIC(LOA):
    """Relaxation with integer cuts; see the module docstring."""

    route = "gdpopt-ric: relaxation with integer cuts (linear master + global subproblems)"

    def __init__(self, model: Model, *, subproblem: str = "global", **kw: Any) -> None:
        if subproblem != "global":
            raise ValueError(
                "RIC's subproblems are global: a configuration leaves the master only "
                "on a certificate (subproblem must be 'global')"
            )
        #: A configuration left the master with no bound of its own (see
        #: :meth:`_global_step`): master bounds no longer cover it.
        self.lb_frozen = False
        super().__init__(model, subproblem="global", **kw)

    def _oa_convexity(self, low: Model) -> Any:
        return _NoCuts(constraint_mask=[False] * len(low._constraints))

    def _master_bounds_nothing(self) -> None:
        # A nonlinear objective leaves the master a feasibility problem: no
        # master bound is offered (``_offer_master_bound``), and the LB comes
        # from exhaustion alone, which is valid.
        pass

    def _seed_cuts(self) -> None:
        pass

    def _cuts_at(self, x: Any) -> None:
        pass

    # ── continuing past an uncertified configuration ─────────────────────

    def _global_step(self, x_master: Any, config: tuple) -> bool:
        """Certified: excluded (the no-good rule). Uncertified: excluded *too*,
        so the search moves on (with no OA cut the master would only propose it
        again), but unresolved -- ``optimal`` stays blocked -- and the bound kept
        honest: the configuration's own valid bound joins ``excluded_floor``,
        and without one the LB freezes at its last value, which still covers it.
        A configuration whose global solve found no point gets LOA's
        fixed-integer NLP for one (measured: small_batch, positioning)."""
        from discopt.gdp.algorithms.subproblem import solve_subproblem

        selection = self.space.selection_from_x(x_master)
        res = solve_subproblem(
            self.space.work,
            selection,
            self.deadline,
            mode="global",
            gap_tolerance=self.gap_tolerance,
            time_cap=self.subproblem_time_limit,
        )
        self.n_subproblems += 1
        self._step_log = f"global {res.status}"
        if res.x is not None:
            self._offer(self._lowered_point(res.x))
        bound = None if res.bound is None else self.obj_sign * float(res.bound)
        if res.status == "infeasible":
            self._exclude_certified(config, None)
            self.mark_solved(config)
            return True
        if res.status == "optimal":
            self._exclude_certified(config, bound)
            self.mark_solved(config)
            return True
        if res.x is None:
            x, rigorous = self._fixed_integer_nlp(x_master)
            self._step_log = f"global {res.status}, then {self._step_log}"
            if x is not None:
                self._offer(x)
            elif rigorous:
                # Interval arithmetic proved the fixed configuration empty: a
                # certificate, as in LOA's NLP step.
                self._exclude_certified(config, None)
                self.mark_solved(config)
                return True
        if bound is not None:
            self.excluded_floor = min(self.excluded_floor, bound)
        else:
            self.lb_frozen = True
        self.master.exclude(config)
        self.mark_unresolved(config)
        return True

    def _offer_master_bound(self, bound: Any) -> None:
        # Once a configuration left the master with no bound of its own, a later
        # master bound no longer covers it: the LB stays where it was.
        if not self.lb_frozen:
            super()._offer_master_bound(bound)

    def _master_empty(self, r: Any) -> None:
        if self.lb_frozen:
            # Every configuration is gone, one of them with no bound: neither an
            # infeasibility proof nor a new bound.
            return
        super()._master_empty(r)


def solve_ric(model: Model, **options: Any):
    """Run :class:`RIC` on ``model`` and return its ``SolveResult``."""
    return RIC(model, **options).run()


__all__ = ["RIC", "solve_ric"]
