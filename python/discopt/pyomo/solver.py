"""``SolverFactory('discopt')`` — a Pyomo plugin backed by discopt.

In-process ``.nl`` round-trip: Pyomo's NL writer emits a temporary ``.nl`` (in the
original variable space, no presolve — see ``_writer``), ``discopt.from_nl`` reads
it into a discopt ``Model`` with the *same* column/row order, ``Model.solve`` runs,
and the primal (and best-effort dual) solution is mapped back by index.

Importing this module registers the solver, so it activates via the
``pyomo.solvers`` entry point or an explicit ``import discopt.pyomo``.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

from pyomo.opt import OptSolver, SolverFactory, SolverResults

from . import _mapping, _writer


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
        timelimit = kwds.pop("timelimit", None)
        kwds.pop("symbolic_solver_labels", None)
        kwds.pop("report_timing", None)

        # Merge persistent options (self.options) with this call's options.
        options: dict[str, Any] = dict(self.options)
        options.update(kwds.pop("options", {}) or {})
        if timelimit is not None:
            options.setdefault("time_limit", timelimit)
        solve_kwargs = _mapping.translate_options(options)
        if tee:
            solve_kwargs.setdefault("stream", True)

        return self._solve_via_nl(
            model,
            solve_kwargs,
            load_solutions=load_solutions,
            keepfiles=keepfiles,
            warmstart=warmstart,
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

            try:
                result = discopt_model.solve(**solve_kwargs)
            except Exception as exc:  # noqa: BLE001 - surface as a structured result
                return self._error_results(model, f"discopt.solve failed: {exc}")

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
        tc, ss = _mapping.termination_for(result.status)
        results.solver.termination_condition = tc
        results.solver.status = ss
        try:
            results.solver.statistics.branch_and_bound.number_of_nodes = int(
                getattr(result, "node_count", 0) or 0
            )
        except Exception:
            pass

        results.problem.name = getattr(model, "name", "unknown")
        results.problem.number_of_variables = len(cols)
        results.problem.number_of_constraints = len(rows)
        sense, obj_bound_field = self._objective_sense(model)
        if sense is not None:
            results.problem.sense = sense
        if result.objective is not None:
            setattr(results.problem, obj_bound_field, result.objective)
        if result.bound is not None:
            other = "upper_bound" if obj_bound_field == "lower_bound" else "lower_bound"
            setattr(results.problem, other, result.bound)

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
    def _objective_sense(model):
        """Return ``(pyomo sense, bound-field)`` for the model's active objective."""
        from pyomo.core.base.objective import Objective
        from pyomo.opt import ProblemSense

        for obj in model.component_data_objects(Objective, active=True):
            if obj.sense == 1:  # minimize
                return ProblemSense.minimize, "lower_bound"
            return ProblemSense.maximize, "upper_bound"
        return None, "lower_bound"

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
            except Exception:
                continue

    @staticmethod
    def _attach_solution(results, cols, flat, result, SolutionStatus) -> None:
        soln = results.solution.add()
        soln.status = SolutionStatus.feasible
        if result.objective is not None:
            soln.objective["__default_objective__"] = {"Value": result.objective}
        for vdata, val in zip(cols, flat):
            soln.variable[vdata.name] = {"Value": float(val)}

    def _trivial_results(self, model) -> SolverResults:
        from pyomo.core.expr import value as pyo_value

        results = SolverResults()
        results.solver.name = "discopt"
        tc, ss = _mapping.termination_for("optimal")
        results.solver.termination_condition = tc
        results.solver.status = ss
        results.problem.number_of_variables = 0
        sense, field = self._objective_sense(model)
        if sense is not None:
            results.problem.sense = sense
        try:
            from pyomo.core.base.objective import Objective

            for obj in model.component_data_objects(Objective, active=True):
                setattr(results.problem, field, pyo_value(obj.expr))
                break
        except Exception:
            pass
        return results

    def _error_results(self, model, message: str) -> SolverResults:
        results = SolverResults()
        results.solver.name = "discopt"
        tc, ss = _mapping.termination_for("error")
        results.solver.termination_condition = tc
        results.solver.status = ss
        results.solver.message = message
        return results
