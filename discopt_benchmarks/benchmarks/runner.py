"""
discopt Benchmark Runner

Orchestrates benchmark execution across solvers and instances,
collects results, computes metrics, and generates reports.
"""

from __future__ import annotations

import re
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


def _downsample_trajectory(points: list, max_points: int) -> list:
    """Downsample a bound trajectory to at most ``max_points`` entries.

    Preserves the first and last points and takes a uniform stride over the
    interior, so the time axis stays monotone and the endpoints (root bound,
    final bound) are always retained.
    """
    n = len(points)
    if max_points <= 0 or n <= max_points:
        return list(points)
    # Uniform indices across [0, n-1], inclusive of both endpoints.
    idx = sorted({round(i * (n - 1) / (max_points - 1)) for i in range(max_points)})
    return [points[i] for i in idx]


@dataclass
class SolverConfig:
    """Configuration for a solver executable."""

    name: str
    command: str
    solver_type: str  # "internal" (discopt) or "external"
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
    # Opt-in bound-trajectory recording (cert:T0.2). Default OFF: attaching a
    # node_callback disables discopt's GP fast path (solver.py auto-GP probe),
    # which would change node_count on geometric-program instances — so the
    # default (and every bound-neutrality baseline) runs without it.
    record_trajectory: bool = False
    trajectory_max_points: int = 500

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
        return not ("problem_class" in f and inst.problem_class != f["problem_class"])

    def run_all(self, parallel: bool = False, max_workers: int = 4):
        """Run all solvers on all instances."""
        instances = sorted(self.results.instance_info.keys())
        total = len(instances) * len(self.config.solvers) * self.config.num_runs
        completed = 0

        print(f"\n{'=' * 70}")
        print(f"discopt Benchmark: {self.config.suite_name}")
        print(
            f"Instances: {len(instances)} | Solvers: {len(self.config.solvers)} "
            f"| Runs: {self.config.num_runs} | Total: {total}"
        )
        print(f"Time limit: {self.config.time_limit}s | Memory: {self.config.memory_limit_mb}MB")
        print(f"{'=' * 70}\n")

        for solver_config in self.config.solvers:
            print(f"\n--- Solver: {solver_config.name} ---")
            for instance_name in instances:
                run_times = []
                best_result = None

                for run_idx in range(self.config.num_runs):
                    result = self._run_single(solver_config, instance_name, run_idx)
                    run_times.append(result.wall_time)

                    # Keep the result from the median-time run
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
                if self.config.num_runs > 1 and all(t < float("inf") for t in run_times):
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

        A daemon-thread watchdog joins with ``time_limit + grace`` so a
        single hung instance (e.g. a JAX while_loop that ignored the
        wall clock, see issue #80) cannot hold the whole sweep hostage.
        The aborted thread is left to drain in the background; the
        returned result reflects the time-limit verdict.
        """
        import threading

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

            # Opt-in bound-trajectory recorder (cert:T0.2). Records
            # (t, node, bound, incumbent) per B&B iteration into a list, then
            # downsamples to <= trajectory_max_points after the solve. Off by
            # default so the standard (bound-neutral) path is unchanged.
            _traj: list[list] = []
            if getattr(self.config, "record_trajectory", False):

                def _traj_cb(ctx, _model, _sink=_traj) -> None:
                    try:
                        _sink.append(
                            [
                                float(ctx.elapsed_time),
                                int(ctx.node_count),
                                (None if ctx.best_bound is None else float(ctx.best_bound)),
                                (None if ctx.incumbent_obj is None else float(ctx.incumbent_obj)),
                            ]
                        )
                    except Exception:
                        pass

                opts["node_callback"] = _traj_cb

            box: dict = {}

            def _do_solve() -> None:
                try:
                    box["result"] = model.solve(
                        time_limit=time_limit,
                        gap_tolerance=gap_tol,
                        max_nodes=max_nodes,
                        **opts,
                    )
                except BaseException as e:  # propagate to caller
                    box["error"] = e

            grace = 30.0  # mirrors the external-solver path
            worker = threading.Thread(target=_do_solve, daemon=True)
            worker.start()
            worker.join(timeout=float(time_limit) + grace)
            if worker.is_alive():
                # Watchdog tripped: a JAX-compiled loop or other in-process
                # call ignored time_limit. Surface a TIME_LIMIT result and
                # let the daemon thread drain in the background.
                elapsed = time.monotonic() - start_time
                print(
                    f"  WATCHDOG {instance}: in-process solve exceeded "
                    f"time_limit+{int(grace)}s, abandoning thread"
                )
                return SolveResult(
                    instance=instance,
                    solver=solver.name,
                    status=SolveStatus.TIME_LIMIT,
                    wall_time=elapsed,
                )
            if "error" in box:
                raise box["error"]
            result = box["result"]

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

            traj = (
                _downsample_trajectory(
                    _traj, getattr(self.config, "trajectory_max_points", 500)
                )
                if _traj
                else None
            )

            return SolveResult(
                instance=instance,
                solver=solver.name,
                status=bench_status,
                objective=result.objective,
                bound=result.bound,
                wall_time=result.wall_time,
                node_count=result.node_count or 0,
                root_gap=getattr(result, "root_gap", None),
                root_time=getattr(result, "root_time", None),
                trajectory=traj,
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
        """Build a solver command line for the .nl-interface external solvers.

        SCIP is driven through its batch (``-c``) interface so we can set the
        time limit and dump the solution/statistics; the AMPL-ASL solvers
        (Couenne, BARON, Bonmin, HiGHS-ASL) read the ``.nl`` directly and report
        on stdout. The subprocess wrapper in :meth:`_run_external` enforces a
        hard wall-clock timeout regardless, so an internal limit is best-effort.
        """
        nl = self._find_nl_file(instance) or instance
        cmd = [solver.command]
        name = solver.name.lower()
        time_limit = int(self.config.time_limit)

        if name.startswith("scip"):
            cmd += [
                "-c",
                f"set limits time {time_limit}",
                "-c",
                f"read {nl}",
                "-c",
                "optimize",
                "-c",
                "display solution",
                "-c",
                "display statistics",
                "-c",
                "quit",
            ]
        else:
            # Couenne / BARON / Bonmin / HiGHS-ASL: solve the .nl directly.
            cmd.append(nl)
        return cmd

    @staticmethod
    def _nl_is_maximize(nl_path: str) -> bool:
        """Read the .nl objective sense from the ``O`` segment.

        ``O<k> 0`` is minimize, ``O<k> 1`` is maximize. Couenne reports its
        Lower/Upper bounds in internal-minimization sense, so we need this to
        recover the original-sense objective.
        """
        # The ``O`` header is ASCII, but some vendored ``.nl`` files carry
        # non-UTF-8 bytes elsewhere (binary-format payloads, latin-1 comments).
        # Read tolerantly so a stray byte can't crash the whole sweep
        # (UnicodeDecodeError under the default utf-8 codec, observed mid-sweep).
        try:
            with open(nl_path, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith("O"):
                        parts = line.split()
                        return len(parts) >= 2 and parts[1].strip() == "1"
        except OSError:
            pass
        return False

    @staticmethod
    def _first_float(pattern: str, text: str) -> float | None:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if not m:
            return None
        try:
            val = float(m.group(1))
        except ValueError:
            return None
        # AMPL/solvers use ~1e20/1e50 as +/-infinity sentinels for "no bound".
        return None if abs(val) >= 1e19 else val

    @staticmethod
    def _first_int(pattern: str, text: str) -> int:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if not m:
            return 0
        try:
            return int(m.group(1))
        except ValueError:
            return 0

    def _parse_external_output(
        self,
        solver_name: str,
        instance: str,
        stdout: str,
        stderr: str,
        elapsed: float,
    ) -> SolveResult:
        """Parse external-solver stdout into a :class:`SolveResult`."""
        name = solver_name.lower()
        if name.startswith("scip"):
            return self._parse_scip(instance, solver_name, stdout, elapsed)
        if name.startswith("highs"):
            return self._parse_highs(instance, solver_name, stdout, elapsed)
        # Couenne / BARON / Bonmin and other AMPL-ASL solvers.
        return self._parse_ampl_solver(instance, solver_name, stdout, elapsed)

    def _parse_scip(
        self, instance: str, solver_name: str, stdout: str, elapsed: float
    ) -> SolveResult:
        status = SolveStatus.UNKNOWN
        m = re.search(r"SCIP Status\s*:\s*(.+)", stdout)
        line = m.group(1).lower() if m else ""
        if "optimal solution found" in line:
            status = SolveStatus.OPTIMAL
        elif "infeasible" in line and "unbounded" not in line:
            status = SolveStatus.INFEASIBLE
        elif "unbounded" in line:
            status = SolveStatus.UNBOUNDED
        elif "time limit" in line:
            status = SolveStatus.TIME_LIMIT
        elif "memory limit" in line:
            status = SolveStatus.MEMORY_LIMIT

        primal = self._first_float(r"Primal Bound\s*:\s*([+\-0-9.eE]+)", stdout)
        dual = self._first_float(r"Dual Bound\s*:\s*([+\-0-9.eE]+)", stdout)
        nodes = self._first_int(r"Solving Nodes\s*:\s*([0-9]+)", stdout)
        return SolveResult(
            instance=instance,
            solver=solver_name,
            status=status,
            objective=primal,
            bound=dual,
            wall_time=elapsed,
            node_count=nodes,
        )

    def _parse_highs(
        self, instance: str, solver_name: str, stdout: str, elapsed: float
    ) -> SolveResult:
        # ASL build prints e.g. "HiGHS 1.11.0: optimal solution; objective 6"
        status = SolveStatus.UNKNOWN
        low = stdout.lower()
        if "optimal" in low:
            status = SolveStatus.OPTIMAL
        elif "infeasible" in low:
            status = SolveStatus.INFEASIBLE
        elif "unbounded" in low:
            status = SolveStatus.UNBOUNDED
        elif "time limit" in low or "time_limit" in low:
            status = SolveStatus.TIME_LIMIT
        objective = self._first_float(r"objective\s+([+\-0-9.eE]+)", stdout)
        return SolveResult(
            instance=instance,
            solver=solver_name,
            status=status,
            objective=objective,
            wall_time=elapsed,
        )

    def _parse_ampl_solver(
        self, instance: str, solver_name: str, stdout: str, elapsed: float
    ) -> SolveResult:
        """Parse Couenne (and similar COIN AMPL solvers) stdout.

        Couenne prints a ``<solver>: <Status>`` summary plus ``Lower bound:`` /
        ``Upper bound:`` lines in *internal-minimization* sense; we un-negate
        them for maximization models so the reported objective is original-sense.
        """
        status = SolveStatus.UNKNOWN
        low = stdout.lower()
        m = re.search(
            r"^\s*\w+:\s*(optimal|infeasible|unbounded|\w+)",
            stdout,
            re.MULTILINE | re.IGNORECASE,
        )
        verdict = m.group(1).lower() if m else ""
        if verdict == "optimal" or "optimal" in low:
            status = SolveStatus.OPTIMAL
        elif "infeasible" in low:
            status = SolveStatus.INFEASIBLE
        elif "unbounded" in low:
            status = SolveStatus.UNBOUNDED
        elif verdict in {"stopped", "limit"} or "time limit" in low:
            status = SolveStatus.TIME_LIMIT

        upper = self._first_float(r"Upper bound:\s*([+\-0-9.eE]+)", stdout)
        lower = self._first_float(r"Lower bound:\s*([+\-0-9.eE]+)", stdout)
        nodes = self._first_int(r"Branch-and-bound nodes:\s*([0-9]+)", stdout)

        objective, bound = upper, lower
        nl = self._find_nl_file(instance)
        if nl and self._nl_is_maximize(nl):
            # internal min g = -f: f_best = -upper_internal, bound = -lower_internal
            objective = None if upper is None else -upper
            bound = None if lower is None else -lower
        return SolveResult(
            instance=instance,
            solver=solver_name,
            status=status,
            objective=objective,
            bound=bound,
            wall_time=elapsed,
            node_count=nodes,
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
