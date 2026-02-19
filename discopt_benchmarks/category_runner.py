"""Per-category benchmark orchestrator.

Runs all problems for a given category against all applicable solvers,
validates correctness, and collects performance metrics.
"""

from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime

from benchmarks.metrics import (
    BenchmarkResults,
    InstanceInfo,
    SolveResult,
    SolveStatus,
    incorrect_count,
    iteration_stats,
    shifted_geometric_mean,
    solved_count,
)
from benchmarks.problems.base import (
    TestProblem,
    get_applicable_solvers,
    get_problems,
)


class CategoryBenchmarkRunner:
    """Orchestrates benchmark runs for a single problem category.

    For each problem in the category, runs all applicable solvers and
    collects SolveResult objects into a BenchmarkResults container.
    """

    def __init__(
        self,
        category: str,
        level: str = "smoke",
        time_limit: float = 300.0,
        num_runs: int = 1,
    ):
        self.category = category
        self.level = level
        self.time_limit = time_limit
        self.num_runs = num_runs
        self.results = BenchmarkResults(
            suite=f"{category}_{level}",
            timestamp=datetime.now().isoformat(),
        )
        self._known_optima: dict[str, float] = {}

    def run(self) -> BenchmarkResults:
        """Run all problems for the category against all solvers."""
        problems = get_problems(self.category, self.level)
        if not problems:
            print(
                f"No problems found for category={self.category} "
                f"level={self.level}"
            )
            return self.results

        # Collect known optima
        for p in problems:
            if (
                p.known_optimum is not None
                and p.known_optimum != float("inf")
                and p.known_optimum != float("-inf")
            ):
                self._known_optima[p.name] = p.known_optimum

        # Register instance info
        for p in problems:
            self.results.instance_info[p.name] = InstanceInfo(
                name=p.name,
                num_variables=p.n_vars,
                num_constraints=p.n_constraints,
                problem_class=p.category,
                best_known_objective=p.known_optimum,
                source=p.source,
            )

        # Header
        solvers = get_applicable_solvers(self.category)
        total_runs = len(problems) * len(solvers)
        print(f"\n{'=' * 70}")
        print(
            f"Category Benchmark: {self.category.upper()} "
            f"({self.level})"
        )
        print(
            f"Problems: {len(problems)} | Solvers: {len(solvers)} "
            f"| Total runs: {total_runs}"
        )
        print(f"Time limit: {self.time_limit}s")
        print(f"{'=' * 70}\n")

        completed = 0
        for problem in problems:
            for solver in problem.applicable_solvers:
                result = self._run_one(problem, solver)
                self._validate(result, problem)
                self.results.add_result(result)
                completed += 1

                # Progress output
                icon = self._status_icon(result)
                t_str = (
                    f"{result.wall_time:.3f}s"
                    if result.wall_time < float("inf")
                    else "TL"
                )
                obj_str = (
                    f"obj={result.objective:.6g}"
                    if result.objective is not None
                    else ""
                )
                iter_str = (
                    f"iter={result.iterations}"
                    if result.iterations > 0
                    else ""
                )
                parts = [
                    f"  {icon} {problem.name:30s}",
                    f"{solver:8s}",
                    f"{t_str:>10s}",
                ]
                if obj_str:
                    parts.append(f"{obj_str:>20s}")
                if iter_str:
                    parts.append(iter_str)
                parts.append(f"[{completed}/{total_runs}]")
                print("  ".join(parts))

        # Summary
        self._print_summary(problems)
        return self.results

    def _run_one(
        self, problem: TestProblem, solver: str
    ) -> SolveResult:
        """Run a single problem with a single solver."""
        if solver == "highs":
            return self._run_highs(problem)
        return self._run_discopt(problem, solver)

    def _run_discopt(
        self, problem: TestProblem, solver: str
    ) -> SolveResult:
        """Run problem through discopt's model.solve()."""
        try:
            model = problem.build_fn()
            start = time.monotonic()
            result = model.solve(
                nlp_solver=solver,
                time_limit=self.time_limit,
                gap_tolerance=1e-4,
                max_nodes=100_000,
            )
            elapsed = time.monotonic() - start

            # Map status
            status_map = {
                "optimal": SolveStatus.OPTIMAL,
                "feasible": SolveStatus.FEASIBLE,
                "infeasible": SolveStatus.INFEASIBLE,
                "unbounded": SolveStatus.UNBOUNDED,
                "time_limit": SolveStatus.TIME_LIMIT,
                "node_limit": SolveStatus.TIME_LIMIT,
            }
            bench_status = status_map.get(
                result.status, SolveStatus.UNKNOWN
            )

            return SolveResult(
                instance=problem.name,
                solver=f"discopt_{solver}",
                status=bench_status,
                objective=result.objective,
                bound=getattr(result, "bound", None),
                wall_time=elapsed,
                node_count=getattr(result, "node_count", 0) or 0,
                iterations=getattr(result, "iterations", 0) or 0,
            )
        except Exception as e:
            print(f"    ERROR ({solver}): {problem.name}: {e}")
            if "--verbose" in sys.argv:
                traceback.print_exc()
            return SolveResult(
                instance=problem.name,
                solver=f"discopt_{solver}",
                status=SolveStatus.ERROR,
                wall_time=float("inf"),
            )

    def _run_highs(self, problem: TestProblem) -> SolveResult:
        """Run LP problem through HiGHS solver directly."""
        try:
            model = problem.build_fn()

            from discopt._jax.problem_classifier import (
                classify_problem,
            )

            pclass = classify_problem(model)
            if pclass.value != "lp":
                return SolveResult(
                    instance=problem.name,
                    solver="highs",
                    status=SolveStatus.ERROR,
                    wall_time=float("inf"),
                )

            # Extract LP data
            from discopt._jax.problem_classifier import (
                extract_lp_data_algebraic,
            )

            lp = extract_lp_data_algebraic(model)

            import numpy as np
            from discopt.solvers.lp_highs import solve_lp

            # Convert to HiGHS format
            c = np.asarray(lp.c)
            bounds = list(
                zip(
                    np.asarray(lp.x_l).tolist(),
                    np.asarray(lp.x_u).tolist(),
                    strict=True,
                )
            )

            a_eq = np.asarray(lp.A_eq) if lp.A_eq.size > 0 else None
            b_eq = np.asarray(lp.b_eq) if lp.b_eq.size > 0 else None

            start = time.monotonic()
            lp_result = solve_lp(
                c=c,
                A_eq=a_eq,
                b_eq=b_eq,
                bounds=bounds,
                time_limit=self.time_limit,
            )
            elapsed = time.monotonic() - start

            # Map HiGHS LPResult status to benchmark SolveStatus
            highs_status = lp_result.status.value
            if highs_status == "optimal":
                status = SolveStatus.OPTIMAL
            elif highs_status == "infeasible":
                status = SolveStatus.INFEASIBLE
            elif highs_status == "unbounded":
                status = SolveStatus.UNBOUNDED
            else:
                status = SolveStatus.ERROR

            obj_val = (
                float(lp_result.objective)
                if lp_result.objective is not None
                else None
            )
            if obj_val is not None:
                obj_val += float(lp.obj_const)

            return SolveResult(
                instance=problem.name,
                solver="highs",
                status=status,
                objective=obj_val,
                wall_time=elapsed,
                iterations=getattr(
                    lp_result, "iterations", 0
                ) or 0,
            )
        except Exception as e:
            print(f"    ERROR (highs): {problem.name}: {e}")
            if "--verbose" in sys.argv:
                traceback.print_exc()
            return SolveResult(
                instance=problem.name,
                solver="highs",
                status=SolveStatus.ERROR,
                wall_time=float("inf"),
            )

    def _validate(
        self, result: SolveResult, problem: TestProblem
    ) -> None:
        """Check result against expected status and known optimum."""
        # Check expected status
        if problem.expected_status == "infeasible":
            if result.status not in (
                SolveStatus.INFEASIBLE,
                SolveStatus.ERROR,
            ):
                print(
                    f"    WARN: {problem.name} expected infeasible, "
                    f"got {result.status.value}"
                )
            return
        if problem.expected_status == "unbounded":
            if result.status not in (
                SolveStatus.UNBOUNDED,
                SolveStatus.ERROR,
            ):
                print(
                    f"    WARN: {problem.name} expected unbounded, "
                    f"got {result.status.value}"
                )
            return

        # Check objective correctness
        if (
            result.is_solved
            and result.objective is not None
            and problem.known_optimum is not None
            and problem.known_optimum != float("inf")
            and problem.known_optimum != float("-inf")
        ):
            ref = problem.known_optimum
            diff = abs(result.objective - ref)
            tol = 1e-4 + 1e-3 * abs(ref)
            if diff > tol:
                print(
                    f"    INCORRECT: {problem.name} "
                    f"solver={result.solver} "
                    f"obj={result.objective:.8e} "
                    f"ref={ref:.8e} diff={diff:.2e}"
                )

    def _status_icon(self, result: SolveResult) -> str:
        """Status icon for progress output."""
        if result.is_solved:
            return "OK"
        if result.is_feasible:
            return "~~"
        if result.status == SolveStatus.INFEASIBLE:
            return "IF"
        if result.status == SolveStatus.UNBOUNDED:
            return "UB"
        if result.status == SolveStatus.ERROR:
            return "!!"
        return "??"

    def _print_summary(self, problems: list[TestProblem]) -> None:
        """Print summary table after all runs complete."""
        solvers = sorted(self.results.get_solvers())
        n_inst = len(problems)

        print(f"\n{'=' * 70}")
        print(
            f"Summary: {self.category.upper()} ({self.level}) "
            f"— {n_inst} problems"
        )
        print(f"{'=' * 70}")
        print(
            f"{'Solver':<18s} {'Solved':>8s} "
            f"{'Incorrect':>10s} {'SGM(s)':>10s} "
            f"{'Med Iter':>10s}"
        )
        print("-" * 60)

        for solver in solvers:
            solver_results = self.results.get_results(solver)
            n_solved = solved_count(solver_results)
            n_incorrect = incorrect_count(
                solver_results, self._known_optima
            )
            times = [
                r.wall_time
                for r in solver_results
                if r.is_solved
            ]
            sgm = shifted_geometric_mean(times)
            istats = iteration_stats(solver_results)
            med_iter = istats["median"]

            sgm_str = (
                f"{sgm:.3f}" if sgm < 1e6 else "inf"
            )
            iter_str = (
                f"{med_iter:.0f}"
                if med_iter == med_iter
                else "N/A"
            )

            print(
                f"{solver:<18s} {n_solved:>3d}/{n_inst:<4d}"
                f" {n_incorrect:>10d} {sgm_str:>10s}"
                f" {iter_str:>10s}"
            )

        print("-" * 60)

    def get_known_optima(self) -> dict[str, float]:
        """Return the known optima dict."""
        return dict(self._known_optima)
