"""
CUTEst Benchmark Runner

Extends the benchmark framework to run discopt's NLP solver against CUTEst
problems, with optional head-to-head comparison against standalone Ipopt
and SciPy.

PyCUTEst is required: pip install discopt[cutest]
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from benchmarks.metrics import (
    BenchmarkResults,
    InstanceInfo,
    SolveResult,
    SolveStatus,
)


@dataclass
class CUTEstSuiteConfig:
    """Configuration for a CUTEst benchmark suite."""

    name: str
    description: str = ""
    # Classification filters (CUTEst codes)
    classification_objective: list[str] | None = None  # e.g. ["Q", "S", "O"]
    classification_constraints: list[str] | None = None  # e.g. ["U", "B"]
    # Dimension limits
    max_variables: int | None = None
    max_constraints: int | None = None
    max_instances: int | None = None
    # Solve settings
    time_limit_seconds: int = 300
    num_runs: int = 1
    # For variable-dimension problems
    sif_params: dict | None = None
    variable_dimension: bool = False
    # Problem list override (explicit names instead of classification filter)
    problem_names: list[str] | None = None


class CUTEstBenchmarkRunner:
    """
    Benchmark runner for CUTEst NLP problems.

    Workflow:
    1. Discover problems via PyCUTEst classification filters
    2. Load each problem, extract metadata
    3. Solve with discopt's NLP solver (via NLPEvaluatorFromCUTEst + Ipopt)
    4. Optionally solve with standalone Ipopt / SciPy for comparison
    5. Emit SolveResult compatible with existing metrics pipeline
    """

    def __init__(self, suite_config: CUTEstSuiteConfig) -> None:
        self.config = suite_config
        self.results = BenchmarkResults(
            suite=f"cutest_{suite_config.name}",
            timestamp=datetime.now().isoformat(),
        )
        self._problems: list[str] = []

    def discover_problems(self) -> list[str]:
        """Find CUTEst problems matching suite filters."""
        from discopt.interfaces.cutest import list_cutest_problems

        if self.config.problem_names:
            self._problems = list(self.config.problem_names)
            return self._problems

        # Build filters from classification codes
        all_problems: set[str] = set()
        obj_codes = self.config.classification_objective or [None]
        con_codes = self.config.classification_constraints or [None]

        for obj_code in obj_codes:
            for con_code in con_codes:
                found = list_cutest_problems(
                    objective=obj_code,
                    constraints=con_code,
                    max_n=self.config.max_variables,
                    max_m=self.config.max_constraints,
                    userN=True if self.config.variable_dimension else None,
                )
                all_problems.update(found)

        self._problems = sorted(all_problems)
        if self.config.max_instances and len(self._problems) > self.config.max_instances:
            self._problems = self._problems[: self.config.max_instances]

        return self._problems

    def load_problem_info(self) -> None:
        """Load metadata for all discovered problems into results."""
        from discopt.interfaces.cutest import load_cutest_problem

        for name in self._problems:
            try:
                prob = load_cutest_problem(name, sif_params=self.config.sif_params)
                info = InstanceInfo(
                    name=name,
                    num_variables=prob.n,
                    num_constraints=prob.m,
                    num_integer_vars=0,
                    num_binary_vars=0,
                    num_continuous_vars=prob.n,
                    num_nonlinear_constraints=prob.m,
                    problem_class=f"cutest_{prob.info.constraint_type}",
                    source="cutest",
                )
                self.results.instance_info[name] = info
                prob.close()
            except Exception as e:
                print(f"  Warning: Could not load {name}: {e}")

    def run_all(
        self,
        solvers: list[str] | None = None,
        verbose: bool = True,
    ) -> BenchmarkResults:
        """
        Run all discovered CUTEst problems through specified solvers.

        Args:
            solvers: List of solver names to use. Options:
                - "discopt_ipopt": discopt's Ipopt backend via CUTEst evaluator
                - "discopt_ipm": discopt's pure-JAX IPM backend
                - "scipy": SciPy minimize for comparison
                - "ipopt_standalone": direct cyipopt for comparison
                Default: ["discopt_ipopt"]
            verbose: Print progress.

        Returns:
            BenchmarkResults with all solve results.
        """
        if solvers is None:
            solvers = ["discopt_ipopt"]

        if not self._problems:
            self.discover_problems()
            self.load_problem_info()

        total = len(self._problems) * len(solvers) * self.config.num_runs
        completed = 0

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"CUTEst Benchmark: {self.config.name}")
            print(
                f"Problems: {len(self._problems)} | Solvers: {solvers} "
                f"| Runs: {self.config.num_runs} | Total: {total}"
            )
            print(f"Time limit: {self.config.time_limit_seconds}s")
            print(f"{'=' * 70}\n")

        for solver_name in solvers:
            if verbose:
                print(f"\n--- Solver: {solver_name} ---")

            for problem_name in self._problems:
                best_result = None
                for _run_idx in range(self.config.num_runs):
                    result = self._run_single(solver_name, problem_name)
                    if (
                        best_result is None
                        or (result.is_solved and not best_result.is_solved)
                        or (
                            result.is_solved
                            and best_result.is_solved
                            and result.wall_time < best_result.wall_time
                        )
                    ):
                        best_result = result
                    completed += 1

                self.results.add_result(best_result)
                if verbose:
                    status_char = "+" if best_result.is_solved else "-"
                    time_str = (
                        f"{best_result.wall_time:.2f}s"
                        if best_result.wall_time < float("inf")
                        else "TL"
                    )
                    obj_str = (
                        f"obj={best_result.objective:.6e}"
                        if best_result.objective is not None
                        else "obj=N/A"
                    )
                    print(
                        f"  {status_char} {problem_name:20s} {time_str:>10s} "
                        f"{obj_str:>20s}  [{completed}/{total}]"
                    )

        return self.results

    def _run_single(self, solver_name: str, problem_name: str) -> SolveResult:
        """Run a single solver on a single CUTEst problem."""
        dispatch = {
            "discopt_ipopt": self._run_discopt_ipopt,
            "discopt_ripopt": self._run_discopt_ripopt,
            "discopt_ipm": self._run_discopt_ipm,
            "scipy": self._run_scipy,
            "ipopt_standalone": self._run_ipopt_standalone,
        }
        runner = dispatch.get(solver_name)
        if runner is None:
            return SolveResult(
                instance=problem_name,
                solver=solver_name,
                status=SolveStatus.ERROR,
                wall_time=float("inf"),
            )
        return runner(problem_name)

    def _run_discopt_ipopt(self, problem_name: str) -> SolveResult:
        """Solve via discopt's Ipopt backend using NLPEvaluatorFromCUTEst."""
        try:
            from discopt.interfaces.cutest import load_cutest_problem
            from discopt.solvers.nlp_ipopt import solve_nlp

            prob = load_cutest_problem(problem_name, sif_params=self.config.sif_params)
            evaluator = prob.to_evaluator()

            # Build constraint bounds for Ipopt
            constraint_bounds = None
            if prob.m > 0:
                cl = prob.cl
                cu = prob.cu
                constraint_bounds = list(zip(cl.tolist(), cu.tolist(), strict=False))

            opts = {"print_level": 0, "max_iter": 3000, "tol": 1e-7}

            t0 = time.perf_counter()
            nlp_result = solve_nlp(
                evaluator, prob.x0, constraint_bounds=constraint_bounds, options=opts
            )
            wall_time = time.perf_counter() - t0

            from discopt.solvers import SolveStatus as DiscoptStatus

            status_map = {
                DiscoptStatus.OPTIMAL: SolveStatus.OPTIMAL,
                DiscoptStatus.INFEASIBLE: SolveStatus.INFEASIBLE,
                DiscoptStatus.UNBOUNDED: SolveStatus.UNBOUNDED,
                DiscoptStatus.ITERATION_LIMIT: SolveStatus.TIME_LIMIT,
                DiscoptStatus.TIME_LIMIT: SolveStatus.TIME_LIMIT,
                DiscoptStatus.ERROR: SolveStatus.ERROR,
            }

            prob.close()
            return SolveResult(
                instance=problem_name,
                solver="discopt_ipopt",
                status=status_map.get(nlp_result.status, SolveStatus.ERROR),
                objective=nlp_result.objective,
                wall_time=wall_time,
            )
        except Exception:
            return SolveResult(
                instance=problem_name,
                solver="discopt_ipopt",
                status=SolveStatus.ERROR,
                wall_time=float("inf"),
            )

    def _run_discopt_ripopt(self, problem_name: str) -> SolveResult:
        """Solve via discopt's ripopt (Rust IPM) backend using NLPEvaluatorFromCUTEst."""
        try:
            from discopt.interfaces.cutest import load_cutest_problem
            from discopt.solvers.nlp_ripopt import solve_nlp

            prob = load_cutest_problem(problem_name, sif_params=self.config.sif_params)
            evaluator = prob.to_evaluator()

            constraint_bounds = None
            if prob.m > 0:
                cl = prob.cl
                cu = prob.cu
                constraint_bounds = list(zip(cl.tolist(), cu.tolist(), strict=False))

            opts = {"print_level": 0, "max_iter": 3000, "tol": 1e-7}

            t0 = time.perf_counter()
            nlp_result = solve_nlp(
                evaluator, prob.x0, constraint_bounds=constraint_bounds, options=opts
            )
            wall_time = time.perf_counter() - t0

            from discopt.solvers import SolveStatus as DiscoptStatus

            status_map = {
                DiscoptStatus.OPTIMAL: SolveStatus.OPTIMAL,
                DiscoptStatus.INFEASIBLE: SolveStatus.INFEASIBLE,
                DiscoptStatus.UNBOUNDED: SolveStatus.UNBOUNDED,
                DiscoptStatus.ITERATION_LIMIT: SolveStatus.TIME_LIMIT,
                DiscoptStatus.TIME_LIMIT: SolveStatus.TIME_LIMIT,
                DiscoptStatus.ERROR: SolveStatus.ERROR,
            }

            prob.close()
            return SolveResult(
                instance=problem_name,
                solver="discopt_ripopt",
                status=status_map.get(nlp_result.status, SolveStatus.ERROR),
                objective=nlp_result.objective,
                wall_time=wall_time,
            )
        except Exception:
            return SolveResult(
                instance=problem_name,
                solver="discopt_ripopt",
                status=SolveStatus.ERROR,
                wall_time=float("inf"),
            )

    def _run_discopt_ipm(self, problem_name: str) -> SolveResult:
        """Solve via discopt's pure-JAX IPM backend."""
        try:
            from discopt.interfaces.cutest import load_cutest_problem

            prob = load_cutest_problem(problem_name, sif_params=self.config.sif_params)
            evaluator = prob.to_evaluator()

            x0 = prob.x0

            constraint_bounds = None
            if prob.m > 0:
                constraint_bounds = list(zip(prob.cl.tolist(), prob.cu.tolist(), strict=False))

            opts = {"print_level": 0, "max_iter": 200}

            t0 = time.perf_counter()

            from discopt._jax.ipm import solve_nlp_ipm

            nlp_result = solve_nlp_ipm(
                evaluator, x0, constraint_bounds=constraint_bounds, options=opts
            )
            wall_time = time.perf_counter() - t0

            from discopt.solvers import SolveStatus as DiscoptStatus

            status_map = {
                DiscoptStatus.OPTIMAL: SolveStatus.OPTIMAL,
                DiscoptStatus.INFEASIBLE: SolveStatus.INFEASIBLE,
                DiscoptStatus.ERROR: SolveStatus.ERROR,
            }

            prob.close()
            return SolveResult(
                instance=problem_name,
                solver="discopt_ipm",
                status=status_map.get(nlp_result.status, SolveStatus.ERROR),
                objective=nlp_result.objective,
                wall_time=wall_time,
            )
        except Exception:
            return SolveResult(
                instance=problem_name,
                solver="discopt_ipm",
                status=SolveStatus.ERROR,
                wall_time=float("inf"),
            )

    def _run_scipy(self, problem_name: str) -> SolveResult:
        """Solve via SciPy minimize for comparison baseline."""
        try:
            from discopt.interfaces.cutest import load_cutest_problem
            from scipy.optimize import minimize

            prob = load_cutest_problem(problem_name, sif_params=self.config.sif_params)
            evaluator = prob.to_evaluator()

            # Build bounds
            bounds = list(zip(prob.bl.tolist(), prob.bu.tolist(), strict=False))
            # Replace large bounds with None for scipy
            scipy_bounds = []
            for lo, hi in bounds:
                lo_val = lo if lo > -1e19 else None
                hi_val = hi if hi < 1e19 else None
                scipy_bounds.append((lo_val, hi_val))

            # Build constraints for scipy
            scipy_constraints = []
            if prob.m > 0:
                cl = prob.cl
                cu = prob.cu
                is_eq = prob.is_eq_cons

                for i in range(prob.m):
                    if is_eq[i]:
                        scipy_constraints.append(
                            {
                                "type": "eq",
                                "fun": lambda x, idx=i: float(
                                    evaluator.evaluate_constraints(x)[idx] - cl[idx]
                                ),
                            }
                        )
                    else:
                        # Inequality: cl[i] <= c(x) <= cu[i]
                        if cl[i] > -1e19:
                            scipy_constraints.append(
                                {
                                    "type": "ineq",
                                    "fun": lambda x, idx=i: float(
                                        evaluator.evaluate_constraints(x)[idx] - cl[idx]
                                    ),
                                }
                            )
                        if cu[i] < 1e19:
                            scipy_constraints.append(
                                {
                                    "type": "ineq",
                                    "fun": lambda x, idx=i: float(
                                        cu[idx] - evaluator.evaluate_constraints(x)[idx]
                                    ),
                                }
                            )

            t0 = time.perf_counter()
            result = minimize(
                evaluator.evaluate_objective,
                prob.x0,
                jac=evaluator.evaluate_gradient,
                bounds=scipy_bounds,
                constraints=scipy_constraints if scipy_constraints else (),
                method="SLSQP" if scipy_constraints else "L-BFGS-B",
                options={"maxiter": 3000, "ftol": 1e-10},
            )
            wall_time = time.perf_counter() - t0

            status = SolveStatus.OPTIMAL if result.success else SolveStatus.ERROR

            prob.close()
            return SolveResult(
                instance=problem_name,
                solver="scipy",
                status=status,
                objective=float(result.fun) if result.success else None,
                wall_time=wall_time,
            )
        except Exception:
            return SolveResult(
                instance=problem_name,
                solver="scipy",
                status=SolveStatus.ERROR,
                wall_time=float("inf"),
            )

    def _run_ipopt_standalone(self, problem_name: str) -> SolveResult:
        """Solve via standalone cyipopt (direct CUTEst callbacks, no discopt)."""
        try:
            import cyipopt
            from discopt.interfaces.cutest import load_cutest_problem

            prob = load_cutest_problem(problem_name, sif_params=self.config.sif_params)
            p = prob.problem

            class _CUTEstIpoptProblem:
                """Direct CUTEst-to-cyipopt adapter (bypasses discopt evaluator)."""

                def __init__(self, cutest_prob):
                    self._p = cutest_prob

                def objective(self, x):
                    return float(self._p.obj(x))

                def gradient(self, x):
                    _, g = self._p.obj(x, gradient=True)
                    return g

                def constraints(self, x):
                    return self._p.cons(x)

                def jacobian(self, x):
                    return self._p.jac(x).flatten()

                def jacobianstructure(self):
                    m, n = prob.m, prob.n
                    rows, cols = np.meshgrid(np.arange(m), np.arange(n), indexing="ij")
                    return (rows.flatten(), cols.flatten())

                def hessian(self, x, lagrange, obj_factor):
                    if prob.m > 0:
                        hess = obj_factor * self._p.ihess(x)
                        for i in range(prob.m):
                            hess += lagrange[i] * self._p.ihess(x, cons_index=i)
                    else:
                        hess = obj_factor * self._p.hess(x)
                    rows, cols = self.hessianstructure()
                    return hess[rows, cols]

                def hessianstructure(self):
                    return np.tril_indices(prob.n)

            callbacks = _CUTEstIpoptProblem(p)

            lb = prob.bl
            ub = prob.bu
            cl = prob.cl if prob.m > 0 else np.empty(0)
            cu = prob.cu if prob.m > 0 else np.empty(0)

            problem = cyipopt.Problem(
                n=prob.n,
                m=prob.m,
                problem_obj=callbacks,
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu,
            )
            problem.add_option("print_level", 0)
            problem.add_option("max_iter", 3000)
            problem.add_option("tol", 1e-7)

            from discopt.solvers.nlp_ipopt import _IPOPT_STATUS_MAP

            t0 = time.perf_counter()
            x, info = problem.solve(prob.x0)
            wall_time = time.perf_counter() - t0

            from discopt.solvers import SolveStatus as DiscoptStatus

            ipopt_status = _IPOPT_STATUS_MAP.get(info["status"], DiscoptStatus.ERROR)
            status_map = {
                DiscoptStatus.OPTIMAL: SolveStatus.OPTIMAL,
                DiscoptStatus.INFEASIBLE: SolveStatus.INFEASIBLE,
                DiscoptStatus.UNBOUNDED: SolveStatus.UNBOUNDED,
                DiscoptStatus.ITERATION_LIMIT: SolveStatus.TIME_LIMIT,
                DiscoptStatus.TIME_LIMIT: SolveStatus.TIME_LIMIT,
                DiscoptStatus.ERROR: SolveStatus.ERROR,
            }

            prob.close()
            return SolveResult(
                instance=problem_name,
                solver="ipopt_standalone",
                status=status_map.get(ipopt_status, SolveStatus.ERROR),
                objective=float(info["obj_val"]),
                wall_time=wall_time,
            )
        except Exception:
            return SolveResult(
                instance=problem_name,
                solver="ipopt_standalone",
                status=SolveStatus.ERROR,
                wall_time=float("inf"),
            )

    def save_results(self, path: Path | None = None) -> None:
        """Save results to JSON."""
        if path is None:
            path = Path("reports") / (
                f"cutest_{self.config.name}_{self.results.timestamp.replace(':', '-')}.json"
            )
        self.results.save(path)
        print(f"\nResults saved to: {path}")
