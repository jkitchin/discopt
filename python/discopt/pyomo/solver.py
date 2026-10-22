"""``SolverFactory('discopt')`` — a Pyomo plugin backed by discopt.

In-process ``.nl`` round-trip: Pyomo's NL writer emits a temporary ``.nl`` (in the
original variable space, no presolve — see ``_writer``), ``discopt.from_nl`` reads
it into a discopt ``Model`` with the *same* column/row order, ``Model.solve`` runs,
and the primal (and best-effort dual) solution is mapped back by index.

Importing this module registers the solver, so it activates via the
``pyomo.solvers`` entry point or an explicit ``import discopt.pyomo``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import tempfile
from collections.abc import Iterator
from typing import Any

from pyomo.opt import OptSolver, SolverFactory, SolverResults

from . import _mapping, _writer

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _tee_solver_log(enabled: bool) -> Iterator[None]:
    """Stream discopt's solver log to stdout for the duration of a solve.

    Pyomo's ``tee=True`` means *show me the solver's log*. discopt has no
    stdout-printing "verbose" mode: it reports progress through the ``discopt``
    logger at INFO, so teeing means attaching a stdout handler to that logger and
    lifting its level for the call, then restoring both.

    This is deliberately **not** ``Model.solve(stream=True)`` — that kwarg asks for
    an iterator of ``SolveUpdate`` instead of a ``SolveResult`` (and today raises
    ``NotImplementedError``), which is a different feature entirely. Mapping
    ``tee`` onto it made every ``tee=True`` solve return termination ``error`` with
    no solution loaded (issue #1178).

    Propagation is left alone: a caller who configured their own handlers keeps
    seeing what they asked for.
    """
    if not enabled:
        yield
        return

    discopt_logger = logging.getLogger("discopt")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    previous_level = discopt_logger.level
    # NOTSET (0) means "inherit", which for an unconfigured root is WARNING, so it
    # needs lifting too; a level already at or below INFO is left as the user set it.
    raise_level = previous_level == logging.NOTSET or previous_level > logging.INFO
    discopt_logger.addHandler(handler)
    if raise_level:
        discopt_logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        discopt_logger.removeHandler(handler)
        handler.flush()
        if raise_level:
            discopt_logger.setLevel(previous_level)


@SolverFactory.register("discopt", doc="discopt hybrid MINLP solver (in-process)")
class DiscoptSolver(OptSolver):
    """Pyomo ``OptSolver`` that solves a model in-process with discopt."""

    def __init__(self, **kwds: Any):
        kwds.setdefault("type", "discopt")
        super().__init__(**kwds)
        # discopt handles continuous, integer and nonlinear constraints natively.
        self._capabilities.linear = True
        self._capabilities.integer = True
        self._capabilities.quadratic_objective = True
        self._capabilities.quadratic_constraint = True
        self._capabilities.sos1 = False
        self._capabilities.sos2 = False

    # -- Pyomo solver contract -------------------------------------------------
    def available(self, exception_flag: bool = False) -> bool:
        try:
            import discopt.modeling  # noqa: F401

            return True
        except ImportError:
            if exception_flag:
                raise
            return False

    def version(self):
        try:
            from importlib.metadata import version

            return tuple(int(p) for p in version("discopt").split(".")[:3] if p.isdigit())
        except Exception:
            return None

    def warm_start_capable(self) -> bool:
        return True

    def solve(self, *args: Any, **kwds: Any) -> SolverResults:
        if not args:
            raise ValueError("DiscoptSolver.solve() requires a Pyomo model argument")
        model = args[0]

        tee = bool(kwds.pop("tee", False))
        load_solutions = bool(kwds.pop("load_solutions", True))
        keepfiles = bool(kwds.pop("keepfiles", False))
        warmstart = bool(kwds.pop("warmstart", False))
        timelimit = self._resolve_time_limit(
            kwds.pop("timelimit", None), kwds.pop("time_limit", None)
        )
        # Accepted for interface compatibility; neither changes what discopt computes.
        kwds.pop("symbolic_solver_labels", None)
        kwds.pop("report_timing", None)
        self._check_suffixes(kwds.pop("suffixes", None))
        call_options = dict(kwds.pop("options", {}) or {})
        call_options.update(self._options_string_to_dict(kwds.pop("options_string", "") or ""))

        # Anything left was not consumed above. Dropping it silently runs a different
        # solve from the one the caller asked for (#1559: ``time_limit=`` used to run
        # at the 3600 s default). Pyomo's own HiGHS plugins (``appsi_highs``,
        # ``highs``) raise TypeError here, and legacy ``OptSolver._presolve`` raises
        # ValueError("passed unrecognized keywords"), so refusing is the contract.
        if kwds:
            raise TypeError(
                f"DiscoptSolver.solve() got unexpected keyword argument(s): {sorted(kwds)}. "
                "Solver options go in options={...}."
            )

        # Merge persistent options (self.options) with this call's options.
        options: dict[str, Any] = dict(self.options)
        options.update(call_options)
        solve_kwargs = _mapping.translate_options(options)
        if timelimit is not None:
            existing = solve_kwargs.get("time_limit")
            if existing is not None and existing != timelimit:
                raise ValueError(
                    f"conflicting time limit: solve(timelimit/time_limit={timelimit!r}) "
                    f"but options give {existing!r}; pass it once"
                )
            solve_kwargs["time_limit"] = timelimit

        return self._solve_via_nl(
            model,
            solve_kwargs,
            load_solutions=load_solutions,
            keepfiles=keepfiles,
            warmstart=warmstart,
            tee=tee,
        )

    # -- Core flow -------------------------------------------------------------
    def _solve_via_nl(
        self,
        model: Any,
        solve_kwargs: dict[str, Any],
        *,
        load_solutions: bool,
        keepfiles: bool,
        warmstart: bool,
        tee: bool = False,
    ) -> SolverResults:
        import discopt.modeling as dm

        if keepfiles:
            ctx = None
            workdir = tempfile.mkdtemp(prefix="discopt_pyomo_")
        else:
            ctx = tempfile.TemporaryDirectory(prefix="discopt_pyomo_")
            workdir = ctx.name
        try:
            nl_path = os.path.join(workdir, "model.nl")
            cols, rows, eliminated = _writer.write_nl(model, nl_path)

            if not cols:
                # Zero-variable / constant model: nothing to round-trip.
                return self._trivial_results(model)

            discopt_model = dm.from_nl(nl_path)

            if warmstart:
                seed = self._warmstart_seed(discopt_model, cols)
                if seed:
                    solve_kwargs.setdefault("initial_solution", seed)

            # The header/footer are logged at INFO so ``tee=True`` always shows a
            # solver log: the core logs INFO during the search, but a model small
            # enough to finish in presolve can otherwise say nothing at all. Both
            # are outside the try, and format only with ``%s``, so a log line can
            # never turn a completed solve into an ``error`` result.
            with _tee_solver_log(tee):
                logger.info(
                    "discopt %s: solving %s variables, %s constraints",
                    ".".join(str(part) for part in self.version() or ()) or "unknown version",
                    len(cols),
                    len(rows),
                )
                try:
                    result = discopt_model.solve(**solve_kwargs)
                except Exception as exc:  # noqa: BLE001 - surface as a structured result
                    logger.info("discopt failed: %s", exc)
                    return self._error_results(model, f"discopt.solve failed: {exc}")
                wall = getattr(result, "wall_time", None)
                logger.info(
                    "discopt finished: status=%s objective=%s bound=%s nodes=%s wall=%s",
                    getattr(result, "status", "unknown"),
                    getattr(result, "objective", None),
                    getattr(result, "bound", None),
                    getattr(result, "node_count", None),
                    # isinstance, not a try/except: formatting must not be able to raise.
                    f"{wall:.3f}s" if isinstance(wall, (int, float)) else wall,
                )

            return self._build_results(
                model,
                discopt_model,
                result,
                cols,
                rows,
                eliminated,
                load_solutions=load_solutions,
            )
        except Exception as exc:  # noqa: BLE001
            return self._error_results(model, f"discopt pyomo bridge failed: {exc}")
        finally:
            if ctx is not None:
                ctx.cleanup()

    def _build_results(
        self, model, discopt_model, result, cols, rows, eliminated, *, load_solutions
    ) -> SolverResults:
        from pyomo.opt import SolutionStatus

        results = SolverResults()
        results.solver.name = "discopt"
        results.solver.wallclock_time = getattr(result, "wall_time", None)
        tc, ss = _mapping.termination_for(result.status, getattr(result, "termination", None))
        results.solver.termination_condition = tc
        results.solver.status = ss
        try:
            results.solver.statistics.branch_and_bound.number_of_nodes = int(
                getattr(result, "node_count", 0) or 0
            )
        except Exception as exc:  # noqa: BLE001 - a missing node count never fails a solve
            logger.debug("node-count statistic not recorded: %s: %s", type(exc).__name__, exc)

        results.problem.name = getattr(model, "name", "unknown")
        results.problem.number_of_variables = len(cols)
        results.problem.number_of_constraints = len(rows)
        sense, incumbent_field, bound_field = self._objective_sense(model)
        if sense is not None:
            results.problem.sense = sense
        # An absent incumbent leaves its field at Pyomo's default (+inf for a
        # minimize, -inf for a maximize): "no feasible point known" (#1560).
        if result.objective is not None:
            setattr(results.problem, incumbent_field, result.objective)
        # Pyomo consumers read the dual-bound field as a certificate (a gap is computed
        # from it), so only a bound discopt itself asserts to be a valid global dual
        # bound is published there; ``bound_valid`` fails closed (#1244).
        if result.bound is not None and getattr(result, "bound_valid", False):
            setattr(results.problem, bound_field, result.bound)

        has_primal = getattr(result, "x", None) is not None
        if has_primal:
            flat = _mapping.solution_flat(discopt_model, result)
            if load_solutions:
                _mapping.load_solution(cols, flat)
                self._backsub_eliminated(eliminated)
                _mapping.load_duals(model, discopt_model, result, cols, rows)
            else:
                self._attach_solution(results, cols, flat, result, SolutionStatus)
        return results

    # -- Helpers ---------------------------------------------------------------
    @staticmethod
    def _resolve_time_limit(timelimit: Any, time_limit: Any) -> Any:
        """Merge Pyomo's ``timelimit`` with discopt's ``time_limit`` spelling (#1559)."""
        if timelimit is not None and time_limit is not None and timelimit != time_limit:
            raise ValueError(
                f"conflicting time limit: timelimit={timelimit!r} and "
                f"time_limit={time_limit!r}; pass one"
            )
        return timelimit if timelimit is not None else time_limit

    @staticmethod
    def _check_suffixes(suffixes: Any) -> None:
        """Validate the legacy ``suffixes=`` request.

        discopt loads ``dual`` and ``rc`` into the model's import Suffixes of those
        names (``_mapping.load_duals``); any other suffix it cannot produce, and
        accepting the request would hand back a model silently missing it.
        """
        if not suffixes:
            return
        if isinstance(suffixes, str):
            suffixes = [suffixes]
        unsupported = sorted(set(map(str, suffixes)) - {"dual", "rc"})
        if unsupported:
            raise ValueError(
                f"DiscoptSolver cannot return suffix(es) {unsupported}; supported: ['dual', 'rc']"
            )

    @staticmethod
    def _objective_sense(model):
        """Return ``(pyomo sense, incumbent field, dual-bound field)``.

        Pyomo's convention, as written by its own HiGHS plugins
        (``contrib/solver/common/base.py``, ``contrib/appsi/base.py``): for a
        minimize the incumbent objective is ``upper_bound`` and the dual bound is
        ``lower_bound``; for a maximize the two swap (#1560).
        """
        from pyomo.core.base.objective import Objective
        from pyomo.opt import ProblemSense

        for obj in model.component_data_objects(Objective, active=True):
            if obj.sense == 1:  # minimize
                return ProblemSense.minimize, "upper_bound", "lower_bound"
            return ProblemSense.maximize, "lower_bound", "upper_bound"
        return None, "upper_bound", "lower_bound"

    @staticmethod
    def _warmstart_seed(discopt_model, cols) -> dict[str, Any]:
        seed: dict[str, Any] = {}
        for var, vdata in zip(discopt_model._variables, cols):
            if vdata.value is not None:
                seed[var.name] = float(vdata.value)
        return seed

    @staticmethod
    def _backsub_eliminated(eliminated) -> None:
        from pyomo.core.expr import value as pyo_value

        for item in eliminated:
            try:
                var, expr = item
                var.set_value(pyo_value(expr), skip_validation=True)
            except Exception as exc:  # noqa: BLE001 - one failed back-substitution is not fatal
                logger.debug(
                    "eliminated variable not back-substituted: %s: %s", type(exc).__name__, exc
                )
                continue

    @staticmethod
    def _attach_solution(results, cols, flat, result, SolutionStatus) -> None:
        soln = results.solution.add()
        soln.status = SolutionStatus.feasible
        if result.objective is not None:
            soln.objective["__default_objective__"] = {"Value": result.objective}
        for vdata, val in zip(cols, flat):
            soln.variable[vdata.name] = {"Value": float(val)}

    @staticmethod
    def _violated_constant_constraints(model) -> list[str]:
        """Names of the active constraints a zero-variable model violates.

        With no free columns every constraint body is a constant, so feasibility is
        decided here, not by a solve. The tolerance is the incumbent verifier's
        bound-keyed form, ``ABS_TOL + BOUND_REL_TOL * |bound|``
        (``discopt.validation.feasibility``): a constant row has no gradient, so the
        bound is its only scale. A body that cannot be evaluated, or evaluates to a
        non-finite number, raises; the caller turns that into an ``error`` result, so
        an unchecked constraint can never be reported ``optimal``.
        """
        import math

        from pyomo.core.base.constraint import Constraint
        from pyomo.core.expr import value as pyo_value

        from discopt.validation.feasibility import ABS_TOL, BOUND_REL_TOL

        violated: list[str] = []
        for con in model.component_data_objects(Constraint, active=True, descend_into=True):
            body = float(pyo_value(con.body))
            if not math.isfinite(body):
                raise ValueError(f"constraint {con.name!r} body evaluates to {body}")
            lb, ub = con.lb, con.ub
            if lb is not None and body < float(lb) - (ABS_TOL + BOUND_REL_TOL * abs(float(lb))):
                violated.append(con.name)
            elif ub is not None and body > float(ub) + (ABS_TOL + BOUND_REL_TOL * abs(float(ub))):
                violated.append(con.name)
        return violated

    def _trivial_results(self, model) -> SolverResults:
        from pyomo.core.expr import value as pyo_value

        violated = self._violated_constant_constraints(model)

        results = SolverResults()
        results.solver.name = "discopt"
        tc, ss = _mapping.termination_for("infeasible" if violated else "optimal")
        results.solver.termination_condition = tc
        results.solver.status = ss
        results.problem.number_of_variables = 0
        sense, incumbent_field, bound_field = self._objective_sense(model)
        if sense is not None:
            results.problem.sense = sense
        if violated:
            # Infeasible: no incumbent and no bound, so both fields stay at +-inf.
            results.solver.message = f"constant constraint(s) violated: {violated}"
            return results
        try:
            from pyomo.core.base.objective import Objective

            for obj in model.component_data_objects(Objective, active=True):
                # A constant objective is exact: it is both incumbent and bound.
                val = pyo_value(obj.expr)
                setattr(results.problem, incumbent_field, val)
                setattr(results.problem, bound_field, val)
                break
        except Exception as exc:  # noqa: BLE001 - a constant objective is reported unset
            logger.debug(
                "trivial-model objective value not recorded: %s: %s", type(exc).__name__, exc
            )
        return results

    def _error_results(self, model, message: str) -> SolverResults:
        results = SolverResults()
        results.solver.name = "discopt"
        tc, ss = _mapping.termination_for("error")
        results.solver.termination_condition = tc
        results.solver.status = ss
        results.solver.message = message
        return results
