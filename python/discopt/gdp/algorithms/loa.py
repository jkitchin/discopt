"""Logic-based outer approximation on the GDP framework (#1775 G4.1, spec §5.3).

:class:`LOA` is today's LOA route (``solvers/gdpopt_loa.py``) rebuilt on
:class:`~discopt.gdp.algorithms.base.GDPAlgorithm`, plus:

* ``init_algorithm`` -- Pyomo GDPopt's initialization strategies
  (:mod:`~discopt.gdp.algorithms.init`); default ``"no_init"``, which keeps
  today's seed (the OA cuts at the continuous NLP relaxation, which every mode
  also adds).
* ``subproblem="global"`` -- each configuration's reduced subproblem (§5.2) is
  solved by discopt itself and its certificate read: a configuration certified
  optimal or infeasible is excluded from the master, so the search closes on
  nonconvex disjuncts the OA cuts cannot represent. The default ``"nlp"`` is
  today's fixed-integer NLP on the lowered model.

Bounds (§5.4): the master is a relaxation -- OA cuts go only on rows (and an
objective) certified convex -- so its dual bound is a valid LB when the
objective is linear or certified convex. A configuration excluded because it
was certified *optimal* (not infeasible) leaves the master, so the LB is then
``min(master bound, best excluded configuration bound)``. Unresolved
configurations never leave the master: they block ``optimal``, not the bound.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

import numpy as np

from discopt.gdp.algorithms.base import GDPAlgorithm
from discopt.gdp.algorithms.init import INIT_STRATEGIES, GDPSpace, initialize
from discopt.gdp.algorithms.master import DEFAULT_MASTER_BACKEND, MasterMILP
from discopt.modeling.core import Model, SolveResult

LOA_SUBPROBLEMS = ("nlp", "global")


class LOA(GDPAlgorithm):
    """Logic-based OA; see the module docstring."""

    route = "gdpopt-loa: logic-based outer approximation (MILP master + NLP)"
    bound_needs_resolution = False

    def __init__(
        self,
        model: Model,
        *,
        time_limit: Optional[float] = None,
        gap_tolerance: float = 1e-4,
        iteration_limit: Optional[int] = 100,
        nlp_solver: str = "ipm",
        milp_solver: str = DEFAULT_MASTER_BACKEND,
        init_algorithm: str = "no_init",
        set_cover_iterlim: int = 8,
        subproblem: str = "global",
        **kw: Any,
    ) -> None:
        from discopt._relax.convexity import classify_oa_cut_convexity
        from discopt._tape_nlp_evaluator import make_evaluator
        from discopt.solvers.oa import _oa_rows

        if init_algorithm not in INIT_STRATEGIES:
            raise ValueError(f"init_algorithm must be one of {INIT_STRATEGIES}")
        if subproblem not in LOA_SUBPROBLEMS:
            raise ValueError(f"subproblem must be one of {LOA_SUBPROBLEMS}")
        self.space = GDPSpace.from_model(model)
        super().__init__(
            self.space.work,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            iteration_limit=iteration_limit,
            **kw,
        )
        self.nlp_solver = nlp_solver
        self.init_algorithm = init_algorithm
        self.set_cover_iterlim = set_cover_iterlim
        self.subproblem = subproblem
        low = self.space.lowered
        self.evaluator = make_evaluator(low)
        self.n = int(self.evaluator.n_variables)
        self.n_work = sum(int(v.size) for v in self.space.work._variables)
        # The certification pass gets half of what is left (> 100 s on kaibel
        # otherwise, leaving the master nothing); a row it does not reach gets no
        # cut, which keeps the master a relaxation. A #912 role-2 slice, so it is
        # routed through the #1116 switch: under ``deterministic`` it becomes the
        # solve's own role-1 deadline, so speed matters only at the wall.
        from discopt.solver import _role2_deadline

        left = self.deadline.remaining()
        convexity = classify_oa_cut_convexity(
            low,
            deadline=_role2_deadline(
                None if math.isinf(left) else time.perf_counter() + 0.5 * left
            ),
        )
        self.convexity_truncated = convexity.truncated
        self.oa_rows = _oa_rows(self.evaluator, low, convexity.constraint_mask)
        self.objective_certified_convex = bool(convexity.objective_is_convex)
        self.master = self._make_master(milp_solver)
        self.obj_is_linear = self.master.objective is not None
        if not self.obj_is_linear and convexity.objective_is_convex:
            self.master.objective_convex = True
        if not self.master.bound_valid:
            self.invalidate_bound("the objective is neither linear nor certified convex")
        self.lb_box, self.ub_box = self.evaluator.variable_bounds
        #: Best certified bound (min convention) of a configuration excluded as
        #: dominated: the master no longer bounds it.
        self.excluded_floor = math.inf
        self.init_report: Any = None

    # ── pieces ────────────────────────────────────────────────────────────

    def _make_master(self, milp_solver: str) -> MasterMILP:
        """The master: the lowered model's linear rows (GLOA overrides this)."""
        return self.space.master(backend=milp_solver)

    def _nlp_limit(self) -> Optional[float]:
        left = self.deadline.sub_limit()
        return None if math.isinf(left) else left

    def _cuts_at(self, x: np.ndarray) -> None:
        from discopt.solvers.gdpopt_loa import _add_oa_cuts

        _add_oa_cuts(
            self.evaluator,
            x,
            self.n,
            int(self.evaluator.n_constraints),
            self.master.cuts,
            self.master.cut_rhs,
            self.obj_is_linear,
            self.oa_rows,
            self.master.objective_convex,
        )

    def _lowered_point(self, x_work: np.ndarray) -> np.ndarray:
        """``x_work`` extended to the lowered model's columns.

        The lowering shares the work model's columns as its prefix and may add
        its own after them (the logical constraints' auxiliary binaries, which
        feed disjunction, SOS and logical rows, some of them big-M). The extension
        only feeds cut generation, and any in-box value keeps every cut valid: a
        certified-convex row's OA tangent is valid at any point, a non-convex row
        gets no cut, and the auxiliaries enter rows linearly. Their values (here
        the box point nearest 0) affect cut strength only. Previously a point was
        dropped -- neither offered nor cut -- whenever the lowering added a column.
        """
        x = np.asarray(x_work, dtype=np.float64)
        if x.shape[0] >= self.n:
            return x[: self.n]
        extra = np.clip(0.0, self.lb_box[x.shape[0] :], self.ub_box[x.shape[0] :])
        return np.concatenate([x, extra])

    def _offer(self, x_lowered: np.ndarray) -> bool:
        return self.offer_incumbent(x_lowered[: self.n_work])

    def _exclude_certified(self, config: tuple, bound_min: Optional[float]) -> None:
        if bound_min is not None:
            self.excluded_floor = min(self.excluded_floor, bound_min)
        self.master.exclude(config)

    def _offer_master_bound(self, bound: Optional[float]) -> None:
        if bound is None or not self.master.bound_valid:
            return
        self.offer_bound(min(bound, self.excluded_floor))

    # ── the initialization strategies ─────────────────────────────────────

    def _solve_selection(self, selection: list) -> Any:
        from discopt.gdp.algorithms.subproblem import solve_subproblem

        mode = "global" if self.subproblem == "global" else "local"
        res = solve_subproblem(
            self.space.work,
            selection,
            self.deadline,
            mode=mode,
            gap_tolerance=self.gap_tolerance,
            nlp_solver="ipopt" if self.nlp_solver == "ipopt" else "pounce",
        )
        self.n_subproblems += 1
        if res.x is None:
            return res
        x = self._lowered_point(res.x)
        self._offer(x)
        self._cuts_at(x)
        config = self.master.config_of(x)
        self.mark_solved(config)
        if res.certified and mode == "global":
            bound = None if res.bound is None else self.obj_sign * float(res.bound)
            self._exclude_certified(config, bound)
        return res

    def initialize(self) -> None:
        from discopt.solvers.gdpopt_loa import _solve_nlp_relaxation

        x_relax = _solve_nlp_relaxation(
            self.evaluator,
            self.lb_box,
            self.ub_box,
            self.nlp_solver,
            time_limit=self._nlp_limit(),
        )
        if x_relax is not None:
            self._cuts_at(np.asarray(x_relax, dtype=np.float64))
        if self.init_algorithm != "no_init":
            self.init_report = initialize(
                self.init_algorithm,
                self.space,
                self._solve_selection,
                self.deadline,
                master=self.master,
                iterlim=self.set_cover_iterlim,
            )

    # ── one iteration ─────────────────────────────────────────────────────

    def iterate(self) -> bool:
        r = self.master.solve(
            self.deadline.sub_limit(),
            self.gap_tolerance,
            incumbent=self.ub if self.incumbent is not None else None,
        )
        self.n_master += 1
        if r.x is None:
            # Only a rigorous INFEASIBLE verdict on a relaxation is a proof.
            if r.proven_infeasible and not math.isfinite(self.excluded_floor):
                self.prove_infeasible()
            elif r.proven_infeasible:
                # Every remaining configuration is gone: the excluded ones bound it.
                self.offer_bound(self.excluded_floor)
            return False
        self._offer_master_bound(r.bound)
        assert r.config is not None
        if r.config in self.solved:
            return False  # a re-proposal adds no cut: stop (#1611 C-11)
        if self.subproblem == "global":
            return self._global_step(r.x, r.config)
        return self._nlp_step(r.x, r.config)

    def _nlp_step(self, x_master: np.ndarray, config: tuple) -> bool:
        from discopt.solvers.gdpopt_loa import (
            _fixed_subproblem_rigorously_infeasible,
            _repair_incumbent,
            _solve_nlp_subproblem,
        )

        sub_lb = np.array(self.lb_box, dtype=np.float64)
        sub_ub = np.array(self.ub_box, dtype=np.float64)
        for idx in self.master.int_indices:
            sub_lb[idx] = sub_ub[idx] = round(float(x_master[idx]))
        x = _solve_nlp_subproblem(
            self.evaluator, sub_lb, sub_ub, self.nlp_solver, time_limit=self._nlp_limit()
        )
        self.n_subproblems += 1
        if x is not None:
            self.mark_solved(config)
            x = _repair_incumbent(self.space.lowered, self.evaluator, x)
            self._offer(np.asarray(x, dtype=np.float64))
            self._cuts_at(np.asarray(x, dtype=np.float64))
            return True
        rigorous = _fixed_subproblem_rigorously_infeasible(self.evaluator, sub_lb, sub_ub)
        self._cuts_at(x_master)
        if rigorous:
            self.master.exclude(config)
            self.mark_solved(config)
            return True
        seen = config in self.unresolved
        self.mark_unresolved(config)
        return not seen  # C-35: never a no-good on a local failure; stop on a repeat

    def _global_step(self, x_master: np.ndarray, config: tuple) -> bool:
        from discopt.gdp.algorithms.subproblem import solve_subproblem

        selection = self.space.selection_from_x(x_master)
        res = solve_subproblem(
            self.space.work,
            selection,
            self.deadline,
            mode="global",
            gap_tolerance=self.gap_tolerance,
        )
        self.n_subproblems += 1
        x = None if res.x is None else self._lowered_point(res.x)
        if x is not None:
            self._offer(x)
            self._cuts_at(x)
        if res.status == "infeasible":
            self._cuts_at(x_master)
            self._exclude_certified(config, None)
            self.mark_solved(config)
        elif res.status == "optimal":
            bound = None if res.bound is None else self.obj_sign * float(res.bound)
            self._exclude_certified(config, bound)
            self.mark_solved(config)
        elif x is not None:
            self.mark_solved(config)  # a point but no certificate: as the NLP mode
        else:
            # The global solve proved nothing and found no point: fall back to the
            # NLP mode's step for this configuration, so global mode never does
            # worse than it (measured: small_batch, positioning, where discopt
            # could not bound the reduced model but the fixed-integer NLP solves).
            return self._nlp_step(x_master, config)
        return True

    def run(self) -> "SolveResult":
        if not self.deadline.expired():
            self.initialize()
        return super().run()


def solve_loa(model: Model, **options: Any):
    """Run :class:`LOA` on ``model`` and return its ``SolveResult``."""
    return LOA(model, **options).run()


__all__ = ["LOA", "LOA_SUBPROBLEMS", "solve_loa"]
