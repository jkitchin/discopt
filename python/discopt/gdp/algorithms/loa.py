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
  nonconvex disjuncts the OA cuts cannot represent. This is the default.
  ``subproblem="nlp"`` is the fixed-integer NLP on the lowered model (the old
  ``solvers/gdpopt_loa.py`` LOA's subproblem).
* ``subproblem_time_limit`` -- each global subproblem's wall budget (#1984).
  Without it one nested ``Model.solve`` could take the whole run (measured: 849
  s of a 3600 s budget on one flowsheet configuration). The default is LBB's and
  LD-SDA's node budget, a tenth of the run within [1 s, 30 s] (#1153). A
  subproblem stopped by its cap is uncertified; when its solve kept a valid
  bound the configuration leaves the master with ``max(that bound, LB)`` in
  ``excluded_floor`` (``LOA._exclude_bounded``), so the search moves on instead
  of stopping when the master proposes it again. A configuration whose global
  solve kept neither a point nor a certificate falls back to the fixed-integer
  NLP and stays in the master; if the master proposes it again it leaves then,
  with the same floor (``LOA._exclude_reproposed``, #2082), instead of ending
  the run.

Bounds (§5.4): the master is a relaxation -- OA cuts go only on rows (and an
objective) certified convex -- so its dual bound is a valid LB when the
objective is linear or certified convex. A configuration excluded because it
was certified *optimal* (not infeasible) leaves the master, so the LB is then
``min(master bound, best excluded configuration bound)``. The same holds for
a configuration excluded uncertified with its valid bound. In NLP mode an
unresolved configuration (no point, no bound) never leaves the master; in
global mode it leaves only with a floor, and an unresolved one still blocks
``optimal``. When the run stops on a re-proposal it cannot exclude,
``stop_reason`` is ``"reproposal"`` (``"unresolved_reproposal"`` for a
configuration no step resolved), not ``"exhausted"`` (#2082).
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Optional

import numpy as np

from discopt.gdp.algorithms.base import GDPAlgorithm
from discopt.gdp.algorithms.init import INIT_STRATEGIES, GDPSpace, initialize
from discopt.gdp.algorithms.master import DEFAULT_MASTER_BACKEND, MasterMILP
from discopt.modeling.core import Model, SolveResult

LOA_SUBPROBLEMS = ("nlp", "global")

#: The ceiling of the default per-subproblem budget, as LBB's ``NODE_TIME_LIMIT``.
SUBPROBLEM_TIME_LIMIT = 30.0

# The old ``solvers/gdpopt_loa.py`` LOA logged one INFO line per iteration to this
# logger, and user code enables it by that name (#1984); the per-iteration line
# keeps it.
logger = logging.getLogger("discopt.solvers.gdpopt_loa")


def default_subproblem_time_limit(time_limit: Optional[float]) -> float:
    """A tenth of the run, at least 1 s, never more than ``SUBPROBLEM_TIME_LIMIT``.

    #1153's saturation rule (LBB's node budget, LD-SDA's subproblem budget): a
    budget that grew with the caller's would spend a bigger run on longer
    subproblems instead of more configurations.
    """
    if time_limit is None:
        return SUBPROBLEM_TIME_LIMIT
    return min(SUBPROBLEM_TIME_LIMIT, max(1.0, 0.1 * float(time_limit)))


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
        subproblem_time_limit: Optional[float] = None,
        **kw: Any,
    ) -> None:
        from discopt._tape_nlp_evaluator import make_evaluator
        from discopt.solvers.oa import _oa_rows

        if init_algorithm not in INIT_STRATEGIES:
            raise ValueError(f"init_algorithm must be one of {INIT_STRATEGIES}")
        if subproblem not in LOA_SUBPROBLEMS:
            raise ValueError(f"subproblem must be one of {LOA_SUBPROBLEMS}")
        if subproblem_time_limit is None:
            subproblem_time_limit = default_subproblem_time_limit(time_limit)
        if isinstance(subproblem_time_limit, bool) or not float(subproblem_time_limit) > 0:
            raise ValueError(f"subproblem_time_limit must be > 0, got {subproblem_time_limit!r}")
        #: Each global subproblem's wall budget (#1984).
        self.subproblem_time_limit = float(subproblem_time_limit)
        #: What the last iteration's subproblem step did, for its log line.
        self._step_log = ""
        deadline = self.start_deadline(time_limit, kw)
        self.space = GDPSpace.from_model(model)
        super().__init__(
            self.space.work,
            deadline=deadline,
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
        convexity = self._oa_convexity(low)
        self.convexity_truncated = convexity.truncated
        self.oa_rows = _oa_rows(self.evaluator, low, convexity.constraint_mask)
        self.objective_certified_convex = bool(convexity.objective_is_convex)
        self.master = self._make_master(milp_solver)
        self.obj_is_linear = self.master.objective is not None
        if not self.obj_is_linear and convexity.objective_is_convex:
            self.master.objective_convex = True
        if not self.master.bound_valid:
            self._master_bounds_nothing()
        self.lb_box, self.ub_box = self.evaluator.variable_bounds
        #: Best certified bound (min convention) of a configuration excluded as
        #: dominated: the master no longer bounds it.
        self.excluded_floor = math.inf
        #: The global subproblem's bound (minimization convention) for each
        #: configuration it left unresolved and handed to the NLP fallback
        #: (#2082), or ``None`` when it kept none; ``_exclude_reproposed`` reads it.
        self.fallback_bound: dict[tuple, Optional[float]] = {}
        self.init_report: Any = None

    # ── pieces ────────────────────────────────────────────────────────────

    def _oa_convexity(self, low: Model) -> Any:
        """Which rows (and whether the objective) may carry OA cuts."""
        from discopt._relax.convexity import classify_oa_cut_convexity
        from discopt.solver import _role2_deadline

        # The certification pass gets half of what is left (> 100 s on kaibel
        # otherwise, leaving the master nothing); a row it does not reach gets no
        # cut, which keeps the master a relaxation. A #912 role-2 slice, so it is
        # routed through the #1116 switch: under ``deterministic`` it becomes the
        # solve's own role-1 deadline, so speed matters only at the wall.
        left = self.deadline.remaining()
        return classify_oa_cut_convexity(
            low,
            deadline=_role2_deadline(
                None if math.isinf(left) else time.perf_counter() + 0.5 * left
            ),
        )

    def _master_bounds_nothing(self) -> None:
        """The master's optimum bounds nothing (no linear or certified-convex objective)."""
        self.invalidate_bound("the objective is neither linear nor certified convex")

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

    def _exclude_bounded(self, config: tuple, sub_bound: Optional[float]) -> bool:
        """Exclude an uncertified configuration whose global solve kept a valid bound (#1984).

        Left in the master, a configuration whose subproblem stopped at its cap
        (``subproblem_time_limit``) short of a certificate is proposed again
        whenever the master cannot see its cost -- a nonconvex row gets no OA
        cut -- and the re-proposal stops the run on that configuration's
        incumbent (measured: the book's flowsheet GDP, epigraph form, stopped at
        iteration 2 on 2276.14 against the optimum 1945.96). Its bound
        ``max(subproblem bound, LB)`` is a valid lower bound on its optimum --
        the LB bounds every configuration -- so the configuration leaves the
        master with that bound in ``excluded_floor``, and ``min(master bound,
        excluded_floor)`` stays a valid LB. Only a finite bound excludes: an
        infinite floor would read a later empty master as an infeasibility proof.
        Returns whether the configuration was excluded.
        """
        cands = [] if sub_bound is None else [self.obj_sign * float(sub_bound)]
        if self.lb_valid and math.isfinite(self.lb):
            cands.append(self.lb)
        bound = max(cands, default=-math.inf)
        if not math.isfinite(bound):
            return False
        self._exclude_certified(config, bound)
        return True

    def _exclude_reproposed(self, config: tuple) -> bool:
        """Exclude a configuration the global step could not resolve, now re-proposed (#2082).

        Its global subproblem stopped with neither a point nor a certificate
        (typically at ``subproblem_time_limit``) and the fixed-integer NLP step
        took over, which neither bounds it nor excludes it. When the master
        proposes it again no new cut can follow, and the #1611 C-11 rule ended
        the whole run there -- reported as ``"exhausted"`` with most of the time
        limit unused and configurations never proposed (measured: the book's
        flowsheet GDP, stopped at iteration 3 of 8 at 13.8 s of 45 s, incumbent
        2276.14 against the optimum 1948.58).

        It leaves the master as ``_exclude_bounded`` leaves it, with
        ``max(that subproblem's bound, LB)`` in ``excluded_floor``. The LB was
        just raised by the master that re-proposed it, and a valid LB bounds
        every configuration's optimum, so ``min(master bound, excluded_floor)``
        stays valid; the price is that the bound can never pass that floor.
        Only configurations of the global fallback qualify: an NLP-mode
        configuration keeps the C-11 stop. Returns whether it was excluded.
        """
        if self.subproblem != "global" or config not in self.fallback_bound:
            return False
        bound = self.fallback_bound[config]
        if not self._exclude_bounded(config, None if bound is None else self.obj_sign * bound):
            return False
        del self.fallback_bound[config]
        return True

    def _offer_master_bound(self, bound: Optional[float]) -> None:
        if bound is None or not self.master.bound_valid:
            return
        self.offer_bound(min(bound, self.excluded_floor))

    def _master_empty(self, r: Any) -> None:
        """The master returned no point (``r.proven_infeasible`` if it proved so)."""
        # Only a rigorous INFEASIBLE verdict on a relaxation is a proof.
        if r.proven_infeasible and not math.isfinite(self.excluded_floor):
            self.prove_infeasible()
        elif r.proven_infeasible:
            # Every remaining configuration is gone: the excluded ones bound it.
            self.offer_bound(self.excluded_floor)

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
            time_cap=self.subproblem_time_limit if mode == "global" else None,
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
        elif mode == "global":
            self._exclude_bounded(config, res.bound)
        return res

    def _seed_cuts(self) -> None:
        """OA cuts at the continuous NLP relaxation, before the first master."""
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

    def initialize(self) -> None:
        self._seed_cuts()
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
        k = self.iterations + 1
        r = self.master.solve(
            self.deadline.sub_limit(),
            self.gap_tolerance,
            incumbent=self.ub if self.incumbent is not None else None,
        )
        self.n_master += 1
        if r.x is None:
            self._master_empty(r)
            logger.info(
                "%s iter %d: master returned no point (%s); bound %s incumbent %s",
                self._log_name(),
                k,
                "proved infeasible" if r.proven_infeasible else "no proof",
                self._fmt(self.lb),
                self._fmt(self.ub),
            )
            return False
        self._offer_master_bound(r.bound)
        assert r.config is not None
        if r.config in self.solved:
            if self._exclude_reproposed(r.config):
                logger.info(
                    "%s iter %d: master re-proposed configuration %s, unresolved by its "
                    "global subproblem; excluded (excluded floor %s, bound %s incumbent %s)",
                    self._log_name(),
                    k,
                    self._fmt_selection(r.x),
                    self._fmt(self.excluded_floor),
                    self._fmt(self.lb),
                    self._fmt(self.ub),
                )
                return True
            logger.info(
                "%s iter %d: master re-proposed configuration %s, already solved; "
                "stopping (bound %s incumbent %s)",
                self._log_name(),
                k,
                self._fmt_selection(r.x),
                self._fmt(self.lb),
                self._fmt(self.ub),
            )
            # A re-proposal adds no cut: stop (#1611 C-11). Not an exhausted
            # search space, so not reported as one (#2082).
            self.stop_detail = "reproposal"
            return False
        t0 = time.perf_counter()
        self._step_log = ""
        if self.subproblem == "global":
            more = self._global_step(r.x, r.config)
        else:
            more = self._nlp_step(r.x, r.config)
        logger.info(
            "%s iter %d: master bound %s, configuration %s; subproblem %s in %.2f s; "
            "bound %s incumbent %s",
            self._log_name(),
            k,
            self._fmt(r.bound),
            self._fmt_selection(r.x),
            self._step_log or "?",
            time.perf_counter() - t0,
            self._fmt(self.lb),
            self._fmt(self.ub),
        )
        return more

    # ── the per-iteration log line (#1984) ────────────────────────────────

    def _log_name(self) -> str:
        return type(self).__name__

    def _fmt(self, v: Optional[float]) -> str:
        """A min-convention value in model units, for the log."""
        return "none" if v is None else f"{self.obj_sign * float(v):.8g}"

    def _fmt_selection(self, x_master: np.ndarray) -> str:
        """The disjuncts the master point selects, by name (by id when unnamed)."""
        sel = self.space.selection_from_x(x_master)
        return "[" + ", ".join(d.name or f"#{d.id}" for d in sel) + "]"

    def _fixed_integer_nlp(self, x_master: np.ndarray) -> tuple[Optional[np.ndarray], bool]:
        """The lowered model's NLP with every integer column fixed at the master's
        value: ``(repaired point or None, proved infeasible by interval arithmetic)``."""
        from discopt.solvers.gdpopt_loa import (
            _fixed_subproblem_rigorously_infeasible,
            _repair_incumbent,
            _solve_nlp_subproblem,
        )

        sub_lb = np.array(self.lb_box, dtype=np.float64)
        sub_ub = np.array(self.ub_box, dtype=np.float64)
        for idx in self.master.int_indices:
            sub_lb[idx] = sub_ub[idx] = round(float(x_master[idx]))
        if self.deadline.expired():
            # Nothing left to give the NLP: a solve launched with the 0.05 s floor
            # still ran 4-5 s on GDPlib ``stranded_gas`` past the limit (#2079).
            # No point is not a proof; the interval check below is one, and cheap.
            self._step_log = "nlp skipped (deadline expired)"
            return None, _fixed_subproblem_rigorously_infeasible(self.evaluator, sub_lb, sub_ub)
        x = _solve_nlp_subproblem(
            self.evaluator, sub_lb, sub_ub, self.nlp_solver, time_limit=self._nlp_limit()
        )
        self.n_subproblems += 1
        self._step_log = "nlp feasible" if x is not None else "nlp failed"
        if x is not None:
            x = _repair_incumbent(self.space.lowered, self.evaluator, x)
            return np.asarray(x, dtype=np.float64), False
        return None, _fixed_subproblem_rigorously_infeasible(self.evaluator, sub_lb, sub_ub)

    def _nlp_step(self, x_master: np.ndarray, config: tuple) -> bool:
        x, rigorous = self._fixed_integer_nlp(x_master)
        if x is not None:
            self.mark_solved(config)
            self._offer(x)
            self._cuts_at(x)
            return True
        self._cuts_at(x_master)
        if rigorous:
            self.master.exclude(config)
            self.mark_solved(config)
            self.fallback_bound.pop(config, None)
            return True
        seen = config in self.unresolved
        self.mark_unresolved(config)
        if seen:
            self.stop_detail = "unresolved_reproposal"
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
            time_cap=self.subproblem_time_limit,
        )
        self.n_subproblems += 1
        self._step_log = f"global {res.status}"
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
            self._exclude_bounded(config, res.bound)
        else:
            # The global solve proved nothing and found no point: fall back to the
            # NLP mode's step for this configuration, so global mode never does
            # worse than it (measured: small_batch, positioning, where discopt
            # could not bound the reduced model but the fixed-integer NLP solves).
            global_status = self._step_log
            # Remembered for ``_exclude_reproposed`` (#2082); the NLP step drops
            # it when it proves the configuration infeasible. Every attempt's
            # bound is valid, so the best one is kept.
            cands = [b for b in (self.fallback_bound.get(config),) if b is not None]
            if res.bound is not None:
                cands.append(self.obj_sign * float(res.bound))
            self.fallback_bound[config] = max(cands, default=None)
            more = self._nlp_step(x_master, config)
            self._step_log = f"{global_status}, then {self._step_log}"
            if not more and self._exclude_reproposed(config):
                # A repeat with no point from either step: leave with the floor
                # rather than end the run; it stays unresolved, so it still
                # blocks ``optimal``.
                self.stop_detail = ""
                self._step_log += ", excluded with floor"
                return True
            return more
        return True

    def run(self) -> "SolveResult":
        if not self.deadline.expired():
            self.initialize()
        return super().run()


def solve_loa(model: Model, **options: Any):
    """Run :class:`LOA` on ``model`` and return its ``SolveResult``."""
    return LOA(model, **options).run()


__all__ = [
    "LOA",
    "LOA_SUBPROBLEMS",
    "SUBPROBLEM_TIME_LIMIT",
    "default_subproblem_time_limit",
    "solve_loa",
]
