"""Per-category benchmark orchestrator.

Runs all problems for a given category against all applicable solvers,
validates correctness, and collects performance metrics.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import suppress
from datetime import datetime
from pathlib import Path

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
        hard_timeout_grace: float | None = 2.0,
        solver_filter: list[str] | None = None,
    ):
        self.category = category
        self.level = level
        self.time_limit = time_limit
        self.num_runs = num_runs
        self.hard_timeout_grace = (
            None if hard_timeout_grace is None else max(0.0, float(hard_timeout_grace))
        )
        self.solver_filter = list(solver_filter) if solver_filter is not None else None
        self.results = BenchmarkResults(
            suite=f"{category}_{level}",
            timestamp=datetime.now().isoformat(),
        )
        self._known_optima: dict[str, float] = {}

    def run(self) -> BenchmarkResults:
        """Run all problems for the category against all solvers."""
        problems = get_problems(self.category, self.level)
        if not problems:
            print(f"No problems found for category={self.category} level={self.level}")
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
        problem_solvers = {p.name: self._solvers_for_problem(p) for p in problems}
        solvers = list(dict.fromkeys(s for slist in problem_solvers.values() for s in slist))
        total_runs = sum(len(slist) for slist in problem_solvers.values())
        print(f"\n{'=' * 70}")
        print(f"Category Benchmark: {self.category.upper()} ({self.level})")
        solver_label = ", ".join(solvers) if solvers else "none"
        print(f"Problems: {len(problems)} | Solvers: {solver_label} | Total runs: {total_runs}")
        print(f"Time limit: {self.time_limit}s")
        if self.hard_timeout_grace is not None:
            print(f"Hard timeout grace: {self.hard_timeout_grace}s")
        print(f"{'=' * 70}\n")
        if total_runs == 0:
            print("No applicable solvers selected for this category.")
            return self.results

        completed = 0
        for problem in problems:
            for solver in problem_solvers[problem.name]:
                result = self._run_one(problem, solver)
                self._validate(result, problem)
                self.results.add_result(result)
                completed += 1

                # Progress output
                icon = self._status_icon(result)
                t_str = f"{result.wall_time:.3f}s" if result.wall_time < float("inf") else "TL"
                obj_str = f"obj={result.objective:.6g}" if result.objective is not None else ""
                iter_str = f"iter={result.iterations}" if result.iterations > 0 else ""
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

    def _solvers_for_problem(self, problem: TestProblem) -> list[str]:
        """Return solvers to run for a problem, honoring explicit filters."""
        if self.solver_filter is None:
            return list(problem.applicable_solvers)

        selected = []
        for solver in self.solver_filter:
            if solver == "amp" or solver in problem.applicable_solvers:
                selected.append(solver)
        return selected

    def _run_one(self, problem: TestProblem, solver: str) -> SolveResult:
        """Run a single problem with a single solver."""
        if self.hard_timeout_grace is not None:
            return self._run_with_hard_timeout(problem, solver)
        return self._run_one_direct(problem, solver)

    def _run_one_direct(self, problem: TestProblem, solver: str) -> SolveResult:
        """Run a single problem in the current process."""
        if solver == "highs":
            return self._run_highs(problem)
        return self._run_discopt(problem, solver)

    def _run_with_hard_timeout(self, problem: TestProblem, solver: str) -> SolveResult:
        """Run a case in a worker process and kill it at the hard timeout."""
        solver_name = "highs" if solver == "highs" else f"discopt_{solver}"
        timeout = max(0.0, self.time_limit) + (self.hard_timeout_grace or 0.0)

        def print_worker_output(proc: subprocess.CompletedProcess[str]) -> None:
            if proc.stdout:
                print(proc.stdout, end="")
            if proc.stderr:
                print(proc.stderr, end="", file=sys.stderr)

        with tempfile.TemporaryDirectory(prefix="discopt_category_bench_") as tmp:
            result_path = Path(tmp) / "result.json"
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                self.category,
                self.level,
                problem.name,
                solver,
                str(self.time_limit),
                str(result_path),
            ]
            started = time.monotonic()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.communicate()
                return SolveResult(
                    instance=problem.name,
                    solver=solver_name,
                    status=SolveStatus.TIME_LIMIT,
                    wall_time=self.time_limit,
                )
            proc = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)

            elapsed = time.monotonic() - started
            if proc.returncode != 0 or not result_path.exists():
                print(f"    ERROR ({solver}): {problem.name}: worker failed")
                if "--verbose" in sys.argv:
                    print_worker_output(proc)
                return SolveResult(
                    instance=problem.name,
                    solver=solver_name,
                    status=SolveStatus.ERROR,
                    wall_time=elapsed,
                )

            try:
                result = SolveResult.from_dict(json.loads(result_path.read_text()))
            except Exception as e:
                print(f"    ERROR ({solver}): {problem.name}: invalid worker result: {e}")
                return SolveResult(
                    instance=problem.name,
                    solver=solver_name,
                    status=SolveStatus.ERROR,
                    wall_time=elapsed,
                )

        if result.wall_time > self.time_limit:
            result.wall_time = self.time_limit
            if result.status == SolveStatus.OPTIMAL and result.objective is not None:
                # The worker returned during the grace window. Keep objective and
                # bound for gap reporting, but do not count the late certificate
                # as an in-budget proof of optimality.
                result.status = SolveStatus.FEASIBLE
            elif result.status in {
                SolveStatus.OPTIMAL,
                SolveStatus.INFEASIBLE,
                SolveStatus.UNBOUNDED,
                SolveStatus.UNKNOWN,
            }:
                result.status = SolveStatus.TIME_LIMIT
        if result.status == SolveStatus.ERROR and "--verbose" in sys.argv:
            print_worker_output(proc)
        return result

    def _run_discopt(self, problem: TestProblem, solver: str) -> SolveResult:
        """Run problem through discopt's model.solve()."""
        try:
            model = problem.build_fn()
            start = time.monotonic()
            if solver == "amp":
                result = model.solve(
                    solver="amp",
                    time_limit=self.time_limit,
                    gap_tolerance=1e-4,
                )
            else:
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
            bench_status = status_map.get(result.status, SolveStatus.UNKNOWN)

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

            obj_val = float(lp_result.objective) if lp_result.objective is not None else None
            if obj_val is not None:
                obj_val += float(lp.obj_const)

            return SolveResult(
                instance=problem.name,
                solver="highs",
                status=status,
                objective=obj_val,
                wall_time=elapsed,
                iterations=getattr(lp_result, "iterations", 0) or 0,
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

    def _validate(self, result: SolveResult, problem: TestProblem) -> None:
        """Check result against expected status and known optimum."""
        # Check expected status
        if problem.expected_status == "infeasible":
            if result.status not in (
                SolveStatus.INFEASIBLE,
                SolveStatus.ERROR,
            ):
                print(f"    WARN: {problem.name} expected infeasible, got {result.status.value}")
            return
        if problem.expected_status == "unbounded":
            if result.status not in (
                SolveStatus.UNBOUNDED,
                SolveStatus.ERROR,
            ):
                print(f"    WARN: {problem.name} expected unbounded, got {result.status.value}")
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
        print(f"Summary: {self.category.upper()} ({self.level}) — {n_inst} problems")
        print(f"{'=' * 70}")
        print(
            f"{'Solver':<18s} {'Solved':>8s} {'Incorrect':>10s} {'SGM(s)':>10s} {'Med Iter':>10s}"
        )
        print("-" * 60)

        for solver in solvers:
            solver_results = self.results.get_results(solver)
            n_solved = solved_count(solver_results)
            n_incorrect = incorrect_count(solver_results, self._known_optima)
            times = [r.wall_time for r in solver_results if r.is_solved]
            sgm = shifted_geometric_mean(times)
            istats = iteration_stats(solver_results)
            med_iter = istats["median"]

            sgm_str = f"{sgm:.3f}" if sgm < 1e6 else "inf"
            iter_str = f"{med_iter:.0f}" if med_iter == med_iter else "N/A"

            print(
                f"{solver:<18s} {n_solved:>3d}/{n_inst:<4d}"
                f" {n_incorrect:>10d} {sgm_str:>10s}"
                f" {iter_str:>10s}"
            )

        print("-" * 60)

    def get_known_optima(self) -> dict[str, float]:
        """Return the known optima dict."""
        return dict(self._known_optima)


def _run_worker(argv: list[str]) -> int:
    """Worker entry point used by subprocess-backed hard timeouts."""
    if len(argv) != 6:
        print(
            "usage: category_runner.py --worker "
            "<category> <level> <problem> <solver> <time_limit> <result_path>",
            file=sys.stderr,
        )
        return 2

    category, level, problem_name, solver, time_limit_s, result_path_s = argv
    problems = get_problems(category, level)
    problem = next((p for p in problems if p.name == problem_name), None)
    if problem is None:
        print(f"unknown problem {problem_name!r} for {category}/{level}", file=sys.stderr)
        return 2

    runner = CategoryBenchmarkRunner(
        category=category,
        level=level,
        time_limit=float(time_limit_s),
        hard_timeout_grace=None,
    )
    result = runner._run_one_direct(problem, solver)
    Path(result_path_s).write_text(json.dumps(result.to_dict()), encoding="utf-8")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        raise SystemExit(_run_worker(sys.argv[2:]))
    raise SystemExit("category_runner.py is an internal module; use run_category_benchmarks.py")
