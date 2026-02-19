"""
discopt Benchmark Metrics

Computes all standard optimization solver benchmarking metrics:
- Solve counts, shifted geometric mean times
- Dolan-Moré performance profiles
- Root gap analysis, node counts, throughput
- Layer profiling (Rust vs JAX vs Python overhead)
- Regression detection with statistical tests
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np


class SolveStatus(Enum):
    """Standardized solve status across all solvers."""
    OPTIMAL = "optimal"                 # Proven global optimum found
    FEASIBLE = "feasible"               # Feasible solution found, not proven optimal
    INFEASIBLE = "infeasible"           # Proven infeasible
    UNBOUNDED = "unbounded"             # Proven unbounded
    TIME_LIMIT = "time_limit"           # Hit time limit
    MEMORY_LIMIT = "memory_limit"       # Hit memory limit
    NUMERICAL_ERROR = "numerical_error" # Solver failed numerically
    ERROR = "error"                     # Other solver error
    UNKNOWN = "unknown"


@dataclass
class SolveResult:
    """Result from a single solver run on a single instance."""
    instance: str
    solver: str
    status: SolveStatus
    objective: Optional[float] = None       # Best objective found
    bound: Optional[float] = None           # Best dual bound (lower bound for min)
    wall_time: float = float("inf")         # Wall-clock seconds
    node_count: int = 0
    root_gap: Optional[float] = None        # (UB - LB_root) / |UB| at root
    root_time: Optional[float] = None       # Time spent at root node

    # Layer profiling (discopt-specific)
    rust_time_fraction: Optional[float] = None
    jax_time_fraction: Optional[float] = None
    python_time_fraction: Optional[float] = None
    batch_sizes: Optional[list[int]] = None  # Node batch sizes during solve

    # NLP/LP subsolver stats
    nlp_solves: int = 0
    lp_solves: int = 0
    nlp_time: float = 0.0
    lp_time: float = 0.0

    # Iteration count (NLP solver iterations or B&B iterations)
    iterations: int = 0

    @property
    def is_solved(self) -> bool:
        return self.status == SolveStatus.OPTIMAL

    @property
    def is_feasible(self) -> bool:
        return self.status in (SolveStatus.OPTIMAL, SolveStatus.FEASIBLE)

    @property
    def is_incorrect(self) -> bool:
        """True if solver claims optimal but objective disagrees with known best."""
        return False  # Set externally after comparison

    @property
    def relative_gap(self) -> Optional[float]:
        if self.objective is None or self.bound is None:
            return None
        if abs(self.objective) < 1e-10:
            return abs(self.objective - self.bound)
        return abs(self.objective - self.bound) / max(abs(self.objective), 1e-10)

    @property
    def nodes_per_second(self) -> Optional[float]:
        if self.wall_time > 0 and self.node_count > 0:
            return self.node_count / self.wall_time
        return None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SolveResult":
        d = dict(d)
        d["status"] = SolveStatus(d["status"])
        return cls(**d)


@dataclass
class InstanceInfo:
    """Metadata about a benchmark instance."""
    name: str
    num_variables: int = 0
    num_constraints: int = 0
    num_integer_vars: int = 0
    num_binary_vars: int = 0
    num_continuous_vars: int = 0
    num_nonlinear_constraints: int = 0
    problem_class: str = "unknown"      # pooling, qcqp, signomial, etc.
    best_known_objective: Optional[float] = None
    is_convex: Optional[bool] = None
    source: str = "minlplib"


@dataclass
class BenchmarkResults:
    """Collected results from a benchmark run."""
    suite: str
    timestamp: str
    solver_results: dict[str, list[SolveResult]] = field(default_factory=dict)
    instance_info: dict[str, InstanceInfo] = field(default_factory=dict)

    def add_result(self, result: SolveResult):
        if result.solver not in self.solver_results:
            self.solver_results[result.solver] = []
        self.solver_results[result.solver].append(result)

    def get_results(self, solver: str) -> list[SolveResult]:
        return self.solver_results.get(solver, [])

    def get_solvers(self) -> list[str]:
        return list(self.solver_results.keys())

    def get_instances(self) -> list[str]:
        instances = set()
        for results in self.solver_results.values():
            for r in results:
                instances.add(r.instance)
        return sorted(instances)

    def save(self, path: Path):
        data = {
            "suite": self.suite,
            "timestamp": self.timestamp,
            "solver_results": {
                s: [r.to_dict() for r in results]
                for s, results in self.solver_results.items()
            },
            "instance_info": {
                name: asdict(info)
                for name, info in self.instance_info.items()
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "BenchmarkResults":
        with open(path) as f:
            data = json.load(f)
        br = cls(suite=data["suite"], timestamp=data["timestamp"])
        for solver, results in data["solver_results"].items():
            for r in results:
                br.add_result(SolveResult.from_dict(r))
        for name, info in data.get("instance_info", {}).items():
            br.instance_info[name] = InstanceInfo(**info)
        return br


# ─────────────────────────────────────────────────────────────
# METRIC FUNCTIONS
# ─────────────────────────────────────────────────────────────

def solved_count(results: list[SolveResult]) -> int:
    """Number of instances solved to proven global optimality."""
    return sum(1 for r in results if r.is_solved)


def solved_count_by_size(
    results: list[SolveResult],
    instance_info: dict[str, InstanceInfo],
    max_vars: int,
) -> int:
    """Solved count filtered by max variable count."""
    return sum(
        1 for r in results
        if r.is_solved
        and r.instance in instance_info
        and instance_info[r.instance].num_variables <= max_vars
    )


def incorrect_count(
    results: list[SolveResult],
    reference: dict[str, float],
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-3,
) -> int:
    """
    Count instances where solver claims optimal but disagrees with reference.

    This is the most critical metric — must always be 0 for release.
    """
    count = 0
    for r in results:
        if not r.is_solved or r.objective is None:
            continue
        if r.instance not in reference:
            continue
        ref = reference[r.instance]
        if abs(r.objective - ref) > abs_tol + rel_tol * abs(ref):
            count += 1
            warnings.warn(
                f"INCORRECT: {r.instance} solver={r.solver} "
                f"obj={r.objective:.8e} ref={ref:.8e} "
                f"diff={abs(r.objective - ref):.2e}"
            )
    return count


def proved_optimal_count(
    results: list[SolveResult],
    gap_tolerance: float = 1e-4,
) -> int:
    """Instances with OPTIMAL status and relative gap within tolerance."""
    count = 0
    for r in results:
        if r.status != SolveStatus.OPTIMAL:
            continue
        gap = r.relative_gap
        if gap is not None and gap <= gap_tolerance:
            count += 1
        elif gap is None and r.is_solved:
            # No bound info but solver claims optimal
            count += 1
    return count


def proved_optimal_rate(
    results: list[SolveResult],
    known_optima: dict[str, float],
    gap_tolerance: float = 1e-4,
) -> float:
    """Fraction of instances with known optima that are proved optimal."""
    with_known = [r for r in results if r.instance in known_optima]
    if not with_known:
        return 0.0
    return proved_optimal_count(with_known, gap_tolerance) / len(with_known)


def final_gap_stats(
    results: list[SolveResult],
) -> dict[str, float]:
    """Mean and median final relative gap across solved/feasible instances."""
    gaps = []
    for r in results:
        gap = r.relative_gap
        if gap is not None and r.is_feasible:
            gaps.append(gap)
    if not gaps:
        return {"mean": float("nan"), "median": float("nan"), "count": 0}
    arr = np.array(gaps)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": len(gaps),
    }


def bound_quality_rate(
    results: list[SolveResult],
    known_optima: dict[str, float],
    threshold: float = 0.99,
) -> float:
    """Fraction of instances where final bound >= threshold * known optimum."""
    valid = 0
    total = 0
    for r in results:
        if r.instance not in known_optima or r.bound is None:
            continue
        opt = known_optima[r.instance]
        total += 1
        # For minimization: bound should be close to (or above) optimum
        if abs(opt) < 1e-10:
            if abs(r.bound - opt) < 1e-6:
                valid += 1
        elif r.bound >= threshold * opt:
            valid += 1
    return valid / max(total, 1)


def shifted_geometric_mean(
    times: list[float],
    shift: float = 1.0,
) -> float:
    """
    Shifted geometric mean of solve times.

    Standard metric for solver benchmarking (Mittelmann).
    shift=1.0 is conventional to handle near-zero times.
    Only includes instances where time is finite (i.e., solved).
    """
    valid = [t for t in times if math.isfinite(t) and t >= 0]
    if not valid:
        return float("inf")
    log_sum = sum(math.log(t + shift) for t in valid)
    return math.exp(log_sum / len(valid)) - shift


def geometric_mean_ratio(
    results_a: list[SolveResult],
    results_b: list[SolveResult],
    shift: float = 1.0,
) -> float:
    """
    Ratio of shifted geometric mean times: solver_a / solver_b.

    Only computed over instances solved by BOTH solvers.
    ratio < 1.0 means solver_a is faster.
    """
    times_a = {r.instance: r.wall_time for r in results_a if r.is_solved}
    times_b = {r.instance: r.wall_time for r in results_b if r.is_solved}
    common = set(times_a.keys()) & set(times_b.keys())
    if not common:
        return float("inf")
    ta = [times_a[i] for i in common]
    tb = [times_b[i] for i in common]
    return shifted_geometric_mean(ta, shift) / max(shifted_geometric_mean(tb, shift), 1e-10)


def performance_profile(
    benchmark: BenchmarkResults,
    tau_max: float = 100.0,
    tau_steps: int = 1000,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Dolan-Moré performance profiles.

    For each solver, computes the fraction of problems solved within
    factor tau of the best solver time.

    Returns dict mapping solver name to (tau_values, fraction_solved).
    """
    instances = benchmark.get_instances()
    solvers = benchmark.get_solvers()

    # Build time matrix: solvers x instances
    times = {}
    for solver in solvers:
        solver_times = {}
        for r in benchmark.get_results(solver):
            if r.is_solved:
                solver_times[r.instance] = r.wall_time
        times[solver] = solver_times

    # For each instance, find best time across all solvers
    best_times = {}
    for inst in instances:
        inst_times = [
            times[s][inst] for s in solvers
            if inst in times[s]
        ]
        if inst_times:
            best_times[inst] = max(min(inst_times), 1e-6)  # Floor at 1μs

    tau_values = np.logspace(0, np.log10(tau_max), tau_steps)
    profiles = {}

    for solver in solvers:
        fractions = []
        n_total = len(instances)
        for tau in tau_values:
            n_within = 0
            for inst in instances:
                if inst in times[solver] and inst in best_times:
                    ratio = times[solver][inst] / best_times[inst]
                    if ratio <= tau:
                        n_within += 1
            fractions.append(n_within / max(n_total, 1))
        profiles[solver] = (tau_values, np.array(fractions))

    return profiles


def root_gap_analysis(
    results: list[SolveResult],
) -> dict[str, float]:
    """Aggregate statistics on root node gap."""
    gaps = [r.root_gap for r in results if r.root_gap is not None]
    if not gaps:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan")}
    arr = np.array(gaps)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
        "count": len(gaps),
    }


def root_gap_ratio(
    results_a: list[SolveResult],
    results_b: list[SolveResult],
) -> float:
    """Mean ratio of root gaps: solver_a / solver_b on common instances."""
    gaps_a = {r.instance: r.root_gap for r in results_a if r.root_gap is not None}
    gaps_b = {r.instance: r.root_gap for r in results_b if r.root_gap is not None}
    common = set(gaps_a.keys()) & set(gaps_b.keys())
    if not common:
        return float("nan")
    ratios = []
    for inst in common:
        if gaps_b[inst] > 1e-10:
            ratios.append(gaps_a[inst] / gaps_b[inst])
    return float(np.mean(ratios)) if ratios else float("nan")


def node_count_reduction(
    results_new: list[SolveResult],
    results_baseline: list[SolveResult],
) -> float:
    """
    Fraction reduction in node count: 1 - (new_nodes / baseline_nodes).

    Positive means new solver uses fewer nodes.
    Computed over commonly-solved instances.
    """
    nodes_new = {r.instance: r.node_count for r in results_new if r.is_solved and r.node_count > 0}
    nodes_base = {r.instance: r.node_count for r in results_baseline if r.is_solved and r.node_count > 0}
    common = set(nodes_new.keys()) & set(nodes_base.keys())
    if not common:
        return 0.0
    ratios = [nodes_new[i] / max(nodes_base[i], 1) for i in common]
    return 1.0 - float(np.mean(ratios))


def layer_profiling_summary(results: list[SolveResult]) -> dict[str, float]:
    """
    Aggregate Rust/JAX/Python time fractions across runs.

    This is discopt-specific: measures where time is spent
    across the hybrid architecture layers.
    """
    rust_fracs = [r.rust_time_fraction for r in results if r.rust_time_fraction is not None]
    jax_fracs = [r.jax_time_fraction for r in results if r.jax_time_fraction is not None]
    py_fracs = [r.python_time_fraction for r in results if r.python_time_fraction is not None]

    return {
        "mean_rust_fraction": float(np.mean(rust_fracs)) if rust_fracs else float("nan"),
        "mean_jax_fraction": float(np.mean(jax_fracs)) if jax_fracs else float("nan"),
        "mean_python_fraction": float(np.mean(py_fracs)) if py_fracs else float("nan"),
        "max_python_fraction": float(np.max(py_fracs)) if py_fracs else float("nan"),
    }


def gpu_vs_cpu_speedup(
    gpu_results: list[SolveResult],
    cpu_results: list[SolveResult],
) -> dict[str, float]:
    """Compare GPU-enabled vs CPU-only discopt runs."""
    gpu_times = {r.instance: r.wall_time for r in gpu_results if r.is_solved}
    cpu_times = {r.instance: r.wall_time for r in cpu_results if r.is_solved}
    common = set(gpu_times.keys()) & set(cpu_times.keys())
    if not common:
        return {"mean_speedup": float("nan"), "median_speedup": float("nan")}
    speedups = [cpu_times[i] / max(gpu_times[i], 1e-6) for i in common]
    return {
        "mean_speedup": float(np.mean(speedups)),
        "median_speedup": float(np.median(speedups)),
        "max_speedup": float(np.max(speedups)),
        "min_speedup": float(np.min(speedups)),
        "count": len(speedups),
    }


def subsolver_pass_rate(
    results: list[SolveResult],
    reference_objectives: dict[str, float],
    abs_tol: float = 1e-6,
) -> float:
    """Fraction of subsolver instances matching reference solution."""
    if not results:
        return 0.0
    correct = 0
    total = 0
    for r in results:
        if r.instance not in reference_objectives:
            continue
        total += 1
        if r.is_solved and r.objective is not None:
            ref = reference_objectives[r.instance]
            if abs(r.objective - ref) <= abs_tol + 1e-4 * abs(ref):
                correct += 1
    return correct / max(total, 1)


def problem_classes_beating(
    results_a: list[SolveResult],
    results_b: list[SolveResult],
    instance_info: dict[str, InstanceInfo],
    min_instances_per_class: int = 5,
) -> list[str]:
    """
    Problem classes where solver_a has lower geomean time than solver_b.

    Requires at least min_instances_per_class commonly-solved instances.
    """
    # Group by problem class
    classes: dict[str, list[str]] = {}
    for name, info in instance_info.items():
        pc = info.problem_class
        if pc not in classes:
            classes[pc] = []
        classes[pc].append(name)

    times_a = {r.instance: r.wall_time for r in results_a if r.is_solved}
    times_b = {r.instance: r.wall_time for r in results_b if r.is_solved}

    beating = []
    for pc, instances in classes.items():
        common = [i for i in instances if i in times_a and i in times_b]
        if len(common) < min_instances_per_class:
            continue
        gm_a = shifted_geometric_mean([times_a[i] for i in common])
        gm_b = shifted_geometric_mean([times_b[i] for i in common])
        if gm_a < gm_b:
            beating.append(pc)

    return beating


def speedup_table(
    benchmark: BenchmarkResults,
    shift: float = 1.0,
) -> dict[tuple[str, str], float]:
    """Pairwise SGM speedup ratios between all solvers.

    Returns dict mapping (solver_a, solver_b) -> SGM(a) / SGM(b).
    Ratio < 1.0 means solver_a is faster.
    """
    solvers = benchmark.get_solvers()
    table = {}
    for sa in solvers:
        for sb in solvers:
            if sa == sb:
                table[(sa, sb)] = 1.0
            else:
                ratio = geometric_mean_ratio(
                    benchmark.get_results(sa),
                    benchmark.get_results(sb),
                    shift=shift,
                )
                table[(sa, sb)] = ratio
    return table


def iteration_stats(
    results: list[SolveResult],
) -> dict[str, float]:
    """Mean and median iteration counts for solved instances."""
    iters = [r.iterations for r in results if r.is_solved and r.iterations > 0]
    if not iters:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "min": 0,
            "max": 0,
            "count": 0,
        }
    arr = np.array(iters)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "count": len(iters),
    }


# ─────────────────────────────────────────────────────────────
# REGRESSION DETECTION
# ─────────────────────────────────────────────────────────────

def detect_regressions(
    current: list[SolveResult],
    baseline: list[SolveResult],
    time_regression_threshold: float = 1.3,   # 30% slower triggers alert
    count_regression_threshold: int = 2,       # Solving 2+ fewer triggers alert
) -> list[dict]:
    """
    Detect performance regressions relative to a baseline run.

    Returns list of regression alerts with details.
    """
    alerts = []

    # Check solve count
    current_solved = solved_count(current)
    baseline_solved = solved_count(baseline)
    if current_solved < baseline_solved - count_regression_threshold:
        alerts.append({
            "type": "solve_count_regression",
            "severity": "high",
            "message": f"Solve count dropped: {baseline_solved} → {current_solved} "
                       f"(Δ = {current_solved - baseline_solved})",
            "baseline": baseline_solved,
            "current": current_solved,
        })

    # Check per-instance time regressions
    current_times = {r.instance: r.wall_time for r in current if r.is_solved}
    baseline_times = {r.instance: r.wall_time for r in baseline if r.is_solved}
    common = set(current_times.keys()) & set(baseline_times.keys())

    time_regressions = []
    for inst in common:
        ratio = current_times[inst] / max(baseline_times[inst], 1e-3)
        if ratio > time_regression_threshold:
            time_regressions.append({
                "instance": inst,
                "baseline_time": baseline_times[inst],
                "current_time": current_times[inst],
                "ratio": ratio,
            })

    if time_regressions:
        # Sort by severity
        time_regressions.sort(key=lambda x: x["ratio"], reverse=True)
        alerts.append({
            "type": "time_regression",
            "severity": "medium",
            "message": f"{len(time_regressions)} instances are >{time_regression_threshold:.0%}x "
                       f"slower (worst: {time_regressions[0]['instance']} at "
                       f"{time_regressions[0]['ratio']:.1f}x)",
            "instances": time_regressions[:10],  # Top 10 worst
        })

    # Check for new incorrect results
    instances_now_unsolved = []
    for inst in baseline_times:
        if inst not in current_times:
            instances_now_unsolved.append(inst)
    if instances_now_unsolved:
        alerts.append({
            "type": "lost_instances",
            "severity": "high",
            "message": f"{len(instances_now_unsolved)} previously-solved instances "
                       f"are no longer solved",
            "instances": instances_now_unsolved[:20],
        })

    return alerts


# ─────────────────────────────────────────────────────────────
# PHASE GATE EVALUATION
# ─────────────────────────────────────────────────────────────

@dataclass
class GateCriterionResult:
    """Result of evaluating a single phase gate criterion."""
    name: str
    target: float
    actual: float
    passed: bool
    direction: str  # "min" or "max" — whether actual must be >= or <= target
    description: str = ""


def evaluate_phase_gate(
    gate_name: str,
    benchmark: BenchmarkResults,
    gate_config: dict,
    reference_solvers: Optional[dict[str, list[SolveResult]]] = None,
    known_optima: Optional[dict[str, float]] = None,
) -> tuple[bool, list[GateCriterionResult]]:
    """
    Evaluate all criteria for a phase gate.

    Returns (all_passed, list_of_criterion_results).
    """
    criteria_results = []
    all_passed = True

    discopt_results = benchmark.get_results("discopt")
    instance_info = benchmark.instance_info

    for crit_name, crit_config in gate_config.get("criteria", {}).items():
        actual = float("nan")
        target = crit_config.get("min", crit_config.get("max", 0))
        direction = "min" if "min" in crit_config else "max"

        metric = crit_config.get("metric", "")

        # Dispatch to metric functions
        if metric == "solved_count":
            actual = solved_count(discopt_results)
        elif metric.startswith("solved_count_le") and metric.endswith("var"):
            max_v = int(metric.replace("solved_count_le", "").replace("var", ""))
            actual = solved_count_by_size(discopt_results, instance_info, max_v)
        elif metric == "convergence_rate" or metric == "pass_rate":
            if known_optima:
                actual = subsolver_pass_rate(discopt_results, known_optima)
        elif metric == "incorrect_count":
            if known_optima:
                actual = incorrect_count(discopt_results, known_optima)
        elif metric.startswith("geomean_ratio_vs_"):
            ref_solver = metric.replace("geomean_ratio_vs_", "")
            if reference_solvers and ref_solver in reference_solvers:
                actual = geometric_mean_ratio(discopt_results, reference_solvers[ref_solver])
        elif metric.startswith("root_gap_ratio_vs_"):
            ref_solver = metric.replace("root_gap_ratio_vs_", "")
            if reference_solvers and ref_solver in reference_solvers:
                actual = root_gap_ratio(discopt_results, reference_solvers[ref_solver])
        elif metric == "gpu_vs_cpu_speedup":
            cpu_results = benchmark.get_results("discopt_cpu")
            if cpu_results:
                stats = gpu_vs_cpu_speedup(discopt_results, cpu_results)
                actual = stats["median_speedup"]
        elif metric == "median_nodes_per_second":
            nps = [r.nodes_per_second for r in discopt_results if r.nodes_per_second is not None]
            actual = float(np.median(nps)) if nps else 0.0
        elif metric == "python_orchestration_fraction" or metric == "rust_tree_overhead_fraction":
            profile = layer_profiling_summary(discopt_results)
            if metric == "python_orchestration_fraction":
                actual = profile["mean_python_fraction"]
            else:
                actual = profile["mean_rust_fraction"]
        elif metric == "relaxation_validity_rate":
            # All relaxation lower bounds must be valid (≤ known optimum)
            if known_optima:
                valid = 0
                total = 0
                for r in discopt_results:
                    if r.bound is not None and r.instance in known_optima:
                        total += 1
                        if r.bound <= known_optima[r.instance] + 1e-6:
                            valid += 1
                actual = valid / max(total, 1)
        elif metric == "node_reduction_vs_classical":
            baseline = benchmark.get_results("discopt_classical_branching")
            if baseline:
                actual = node_count_reduction(discopt_results, baseline)
        elif metric == "problem_classes_beating_baron":
            if reference_solvers and "baron" in reference_solvers:
                beating = problem_classes_beating(
                    discopt_results, reference_solvers["baron"], instance_info
                )
                actual = len(beating)
        elif metric.startswith("solve_count_ratio_vs_"):
            ref_solver = metric.replace("solve_count_ratio_vs_", "")
            if reference_solvers and ref_solver in reference_solvers:
                ref_solved = solved_count(reference_solvers[ref_solver])
                our_solved = solved_count(discopt_results)
                actual = our_solved / max(ref_solved, 1)

        # Evaluate pass/fail
        if direction == "min":
            passed = actual >= target
        else:
            passed = actual <= target

        if math.isnan(actual):
            passed = False

        if not passed:
            all_passed = False

        criteria_results.append(GateCriterionResult(
            name=crit_name,
            target=target,
            actual=actual,
            passed=passed,
            direction=direction,
        ))

    return all_passed, criteria_results
