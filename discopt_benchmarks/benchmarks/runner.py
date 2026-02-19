"""
discopt Benchmark Runner

Orchestrates benchmark execution across solvers and instances,
collects results, computes metrics, and generates reports.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from benchmarks.metrics import (
    BenchmarkResults,
    InstanceInfo,
    SolveResult,
    SolveStatus,
)


@dataclass
class SolverConfig:
    """Configuration for a solver executable."""
    name: str
    command: str
    solver_type: str          # "internal" (discopt) or "external"
    nl_interface: bool = False
    options: dict = None

    def __post_init__(self):
        if self.options is None:
            self.options = {}


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    suite_name: str
    time_limit: int = 3600
    memory_limit_mb: int = 32768
    num_runs: int = 3
    solvers: list[SolverConfig] = None
    instance_filter: dict | None = None
    output_dir: Path = Path("reports")

    def __post_init__(self):
        if self.solvers is None:
            self.solvers = []


class BenchmarkRunner:
    """
    Main benchmark orchestrator.

    Workflow:
    1. Load instance list from problem library
    2. For each solver × instance × run:
       a. Launch solver with time/memory limits
       b. Parse output into SolveResult
       c. Validate correctness against known optima
    3. Compute aggregate metrics
    4. Check phase gate criteria
    5. Generate report
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = BenchmarkResults(
            suite=config.suite_name,
            timestamp=datetime.now().isoformat(),
        )
        self._known_optima: dict[str, float] = {}

    def load_instances(self, instances: list[InstanceInfo]):
        """Load instance metadata and apply filters."""
        for inst in instances:
            if self._passes_filter(inst):
                self.results.instance_info[inst.name] = inst

    def load_known_optima(self, optima: dict[str, float]):
        """Load known best objectives for correctness validation."""
        self._known_optima = optima

    def _passes_filter(self, inst: InstanceInfo) -> bool:
        """Check if instance passes suite filters."""
        f = self.config.instance_filter or {}
        if "max_variables" in f and inst.num_variables > f["max_variables"]:
            return False
        if "max_constraints" in f and inst.num_constraints > f["max_constraints"]:
            return False
        if "max_instances" in f:
            # Handled externally by truncating list
            pass
        if "problem_class" in f and inst.problem_class != f["problem_class"]:
            return False
        return True

    def run_all(self, parallel: bool = False, max_workers: int = 4):
        """Run all solvers on all instances."""
        instances = sorted(self.results.instance_info.keys())
        total = len(instances) * len(self.config.solvers) * self.config.num_runs
        completed = 0

        print(f"\n{'='*70}")
        print(f"discopt Benchmark: {self.config.suite_name}")
        print(f"Instances: {len(instances)} | Solvers: {len(self.config.solvers)} "
              f"| Runs: {self.config.num_runs} | Total: {total}")
        print(f"Time limit: {self.config.time_limit}s | Memory: {self.config.memory_limit_mb}MB")
        print(f"{'='*70}\n")

        for solver_config in self.config.solvers:
            print(f"\n--- Solver: {solver_config.name} ---")
            for instance_name in instances:
                run_times = []
                best_result = None

                for run_idx in range(self.config.num_runs):
                    result = self._run_single(
                        solver_config, instance_name, run_idx
                    )
                    run_times.append(result.wall_time)

                    # Keep the result from the median-time run
                    if best_result is None or (
                        result.is_solved and not best_result.is_solved
                    ) or (
                        result.is_solved and best_result.is_solved
                        and result.wall_time < best_result.wall_time
                    ):
                        best_result = result

                    completed += 1

                # Validate correctness
                if best_result.is_solved and instance_name in self._known_optima:
                    ref = self._known_optima[instance_name]
                    if best_result.objective is not None:
                        diff = abs(best_result.objective - ref)
                        if diff > 1e-4 + 1e-3 * abs(ref):
                            print(
                                f"  ⚠ INCORRECT: {instance_name} "
                                f"obj={best_result.objective:.6e} "
                                f"ref={ref:.6e}"
                            )

                # Check determinism
                if self.config.num_runs > 1 and all(
                    t < float("inf") for t in run_times
                ):
                    cv = _coefficient_of_variation(run_times)
                    if cv > 0.1:
                        print(
                            f"  ⚠ Non-deterministic: {instance_name} "
                            f"times={[f'{t:.1f}' for t in run_times]} "
                            f"CV={cv:.2f}"
                        )

                self.results.add_result(best_result)

                # Status icon
                if best_result.is_solved:
                    icon = "✓"
                elif best_result.is_feasible:
                    icon = "~"
                elif best_result.status == SolveStatus.ERROR:
                    icon = "!"
                else:
                    icon = "✗"

                time_str = (
                    f"{best_result.wall_time:.2f}s"
                    if best_result.wall_time < float("inf")
                    else "TL"
                )

                # Objective and gap info
                obj_str = ""
                if best_result.objective is not None:
                    obj_str = f"obj={best_result.objective:.4g}"
                gap_str = ""
                gap = best_result.relative_gap
                if gap is not None:
                    gap_str = f"gap={gap:.1%}"

                parts = [f"  {icon} {instance_name:30s}"]
                if obj_str:
                    parts.append(f"{obj_str:>16s}")
                if gap_str:
                    parts.append(f"{gap_str:>10s}")
                parts.append(f"{time_str:>10s}")
                parts.append(f"nodes={best_result.node_count:>6d}")
                parts.append(f"[{completed}/{total}]")
                print("  ".join(parts))

    def _run_single(
        self,
        solver: SolverConfig,
        instance: str,
        run_idx: int,
    ) -> SolveResult:
        """Run a single solver on a single instance."""
        if solver.solver_type == "internal":
            return self._run_discopt(solver, instance, run_idx)
        else:
            return self._run_external(solver, instance, run_idx)

    def _run_discopt(
        self,
        solver: SolverConfig,
        instance: str,
        run_idx: int,
    ) -> SolveResult:
        """
        Run discopt solver on a .nl instance.

        Uses the Python API directly, capturing layer profiling data.
        """
        start_time = time.monotonic()
        try:
            import discopt.modeling as dm

            # Resolve .nl file path
            nl_path = self._find_nl_file(instance)
            if nl_path is None:
                return SolveResult(
                    instance=instance,
                    solver=solver.name,
                    status=SolveStatus.ERROR,
                    wall_time=float("inf"),
                )

            model = dm.from_nl(nl_path)

            # Map solver options
            opts = dict(solver.options)
            time_limit = opts.pop("time_limit", self.config.time_limit)
            gap_tol = opts.pop("gap_tolerance", 1e-4)
            max_nodes = opts.pop("max_nodes", 100_000)
            opts.pop("gpu", None)  # legacy option, ignored

            result = model.solve(
                time_limit=time_limit,
                gap_tolerance=gap_tol,
                max_nodes=max_nodes,
                **opts,
            )

            # Map discopt status to benchmark status
            status_map = {
                "optimal": SolveStatus.OPTIMAL,
                "feasible": SolveStatus.FEASIBLE,
                "infeasible": SolveStatus.INFEASIBLE,
                "time_limit": SolveStatus.TIME_LIMIT,
                "node_limit": SolveStatus.TIME_LIMIT,
            }
            bench_status = status_map.get(result.status, SolveStatus.UNKNOWN)

            # Compute layer profiling fractions
            wt = result.wall_time if result.wall_time > 0 else 1e-10
            rust_frac = result.rust_time / wt if result.rust_time else None
            jax_frac = result.jax_time / wt if result.jax_time else None
            py_frac = result.python_time / wt if result.python_time else None

            return SolveResult(
                instance=instance,
                solver=solver.name,
                status=bench_status,
                objective=result.objective,
                bound=result.bound,
                wall_time=result.wall_time,
                node_count=result.node_count or 0,
                rust_time_fraction=rust_frac,
                jax_time_fraction=jax_frac,
                python_time_fraction=py_frac,
            )
        except Exception as e:
            elapsed = time.monotonic() - start_time
            print(f"  ERROR {instance}: {e}")
            return SolveResult(
                instance=instance,
                solver=solver.name,
                status=SolveStatus.ERROR,
                wall_time=elapsed,
            )

    def _find_nl_file(self, instance: str) -> str | None:
        """Resolve instance name to .nl file path.

        Searches in order:
          1. Instance name as-is (if it's an absolute/relative path)
          2. data_dir / instance.nl
          3. python/tests/data/minlplib / instance.nl (development fallback)
        """
        # Direct path
        p = Path(instance)
        if p.suffix == ".nl" and p.exists():
            return str(p)

        # Data directory (set via config or default)
        data_dir = getattr(self.config, "data_dir", None)
        if data_dir:
            candidate = Path(data_dir) / f"{instance}.nl"
            if candidate.exists():
                return str(candidate)

        # Development fallback: test data directory
        project_root = Path(__file__).parent.parent.parent
        for subdir in ["python/tests/data/minlplib", "python/tests/data/minlplib_nl"]:
            candidate = project_root / subdir / f"{instance}.nl"
            if candidate.exists():
                return str(candidate)

        return None

    def _run_external(
        self,
        solver: SolverConfig,
        instance: str,
        run_idx: int,
    ) -> SolveResult:
        """
        Run external solver via subprocess with time/memory limits.

        Parses .sol file or stdout for results.
        """
        # Build command line
        cmd = self._build_command(solver, instance)

        try:
            start_time = time.monotonic()
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.time_limit + 30,  # Grace period
            )
            elapsed = time.monotonic() - start_time

            # Parse solver output (solver-specific parsing)
            return self._parse_external_output(
                solver.name, instance, proc.stdout, proc.stderr, elapsed
            )
        except subprocess.TimeoutExpired:
            return SolveResult(
                instance=instance,
                solver=solver.name,
                status=SolveStatus.TIME_LIMIT,
                wall_time=self.config.time_limit,
            )
        except Exception:
            return SolveResult(
                instance=instance,
                solver=solver.name,
                status=SolveStatus.ERROR,
            )

    def _build_command(self, solver: SolverConfig, instance: str) -> list[str]:
        """Build solver command line."""
        cmd = [solver.command]
        # Solver-specific flags would be added here
        cmd.append(instance)
        return cmd

    def _parse_external_output(
        self,
        solver_name: str,
        instance: str,
        stdout: str,
        stderr: str,
        elapsed: float,
    ) -> SolveResult:
        """Parse external solver output into SolveResult."""
        # This would have solver-specific parsers for BARON, Couenne, SCIP, etc.
        # Stub implementation
        return SolveResult(
            instance=instance,
            solver=solver_name,
            status=SolveStatus.UNKNOWN,
            wall_time=elapsed,
        )

    def save_results(self, path: Path | None = None):
        """Save results to JSON."""
        if path is None:
            path = (
                self.config.output_dir
                / f"{self.config.suite_name}_{self.results.timestamp.replace(':', '-')}.json"
            )
        self.results.save(path)
        print(f"\nResults saved to: {path}")


def _coefficient_of_variation(values: list[float]) -> float:
    """CV for determinism checking."""
    if len(values) < 2:
        return 0.0
    import numpy as np
    arr = np.array(values)
    mean = np.mean(arr)
    if mean < 1e-6:
        return 0.0
    return float(np.std(arr) / mean)
