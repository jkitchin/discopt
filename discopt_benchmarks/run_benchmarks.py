#!/usr/bin/env python3
"""
discopt Benchmark CLI

Usage:
    # Run smoke tests (quick sanity check)
    python run_benchmarks.py --suite smoke

    # Run phase gate check
    python run_benchmarks.py --gate phase1

    # Run comparison benchmark with report
    python run_benchmarks.py --suite comparison --solvers discopt,baron --report

    # Run nightly regression check
    python run_benchmarks.py --suite nightly --baseline reports/latest.json

    # List available suites
    python run_benchmarks.py --list-suites
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.metrics import BenchmarkResults, evaluate_phase_gate


def main():
    parser = argparse.ArgumentParser(
        description="discopt Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--suite", type=str, default="smoke",
        help="Benchmark suite to run (smoke, phase1, phase2, phase3, full, nightly, comparison)"
    )
    parser.add_argument(
        "--gate", type=str, default=None,
        help="Evaluate phase gate criteria (phase1, phase2, phase3, phase4)"
    )
    parser.add_argument(
        "--solvers", type=str, default="discopt",
        help="Comma-separated list of solvers to benchmark"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate markdown report after benchmarking"
    )
    parser.add_argument(
        "--baseline", type=str, default=None,
        help="Path to baseline results JSON for regression detection"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for results JSON"
    )
    parser.add_argument(
        "--list-suites", action="store_true",
        help="List available benchmark suites"
    )
    parser.add_argument(
        "--ci", action="store_true",
        help="CI mode: compact JSON output for pipeline integration"
    )
    parser.add_argument(
        "--save-baseline", action="store_true",
        help="Save current results as baseline for future regression detection"
    )
    parser.add_argument(
        "--compare-baseline", action="store_true",
        help="Compare against saved baseline (auto-detected if available)"
    )
    parser.add_argument(
        "--history", action="store_true",
        help="Print historical trend table for the suite"
    )

    args = parser.parse_args()

    if args.list_suites:
        _list_suites()
        return

    if args.history:
        from utils.history import print_trend
        print_trend(args.suite)
        return

    if args.gate:
        _run_gate_check(args)
        return

    _run_benchmark(args)


def _list_suites():
    """List available benchmark suites."""
    suites = {
        "smoke":         "Quick sanity check (10 instances, 60s limit)",
        "phase1":        "Phase 1 validation (small instances, ≤10 vars)",
        "phase2":        "Phase 2 validation (medium instances, ≤50 vars)",
        "phase3":        "Phase 3 validation (large instances, ≤100 vars)",
        "full":          "Complete MINLPLib (~1700 instances)",
        "comparison":    "Head-to-head solver comparison (curated set)",
        "nightly":       "Nightly CI regression suite (100 instances)",
        "global_opt":    "Global optimality verification (known optima)",
        "lp":            "Pure LP instances",
        "qp":            "Quadratic programming instances",
        "convex_nlp":    "Convex NLP instances",
        "nonconvex_nlp": "Non-convex NLP instances",
        "lp_netlib":     "Rust LP solver vs Netlib",
        "nlp_cutest":    "Hybrid NLP solver vs CUTEst",
        "pooling":       "GPU-amenable pooling problems",
        "gpu_scaling":   "GPU batching scalability measurement",
    }
    print("\nAvailable benchmark suites:\n")
    for name, desc in suites.items():
        print(f"  {name:15s}  {desc}")
    print()


def _run_gate_check(args):
    """Run phase gate evaluation."""
    print(f"\n{'='*60}")
    print(f"Phase Gate Check: {args.gate}")
    print(f"{'='*60}\n")

    # Load latest results
    results_path = Path(args.output) if args.output else _find_latest_results(args.gate)
    if not results_path or not results_path.exists():
        print(f"ERROR: No results found for gate '{args.gate}'.")
        print(f"Run benchmarks first: python run_benchmarks.py --suite {args.gate}")
        sys.exit(1)

    benchmark = BenchmarkResults.load(results_path)

    # Load gate config
    gate_config = _load_gate_config(args.gate)
    if not gate_config:
        print(f"ERROR: No gate configuration found for '{args.gate}'")
        sys.exit(1)

    all_passed, criteria = evaluate_phase_gate(
        args.gate, benchmark, gate_config
    )

    # Display results
    print(f"{'Criterion':<40s} {'Target':>12s} {'Actual':>12s} {'Status':>8s}")
    print("-" * 75)
    for c in criteria:
        target_str = f"{'≥' if c.direction == 'min' else '≤'} {c.target}"
        actual_str = f"{c.actual:.4f}" if isinstance(c.actual, float) else str(c.actual)
        status = "✅ PASS" if c.passed else "🔴 FAIL"
        print(f"{c.name:<40s} {target_str:>12s} {actual_str:>12s} {status:>8s}")

    print()
    if all_passed:
        print("✅ ALL CRITERIA PASSED — proceed to next phase")
        sys.exit(0)
    else:
        failed = [c for c in criteria if not c.passed]
        print(f"🔴 {len(failed)} CRITERIA FAILED — do not proceed")
        sys.exit(1)


def _run_benchmark(args):
    """Run a benchmark suite."""
    from benchmarks.runner import BenchmarkConfig, BenchmarkRunner, SolverConfig

    print("\ndiscopt Benchmark Runner")
    print(f"Suite: {args.suite}")
    print(f"Solvers: {args.solvers}")
    print(f"Time: {datetime.now().isoformat()}")
    print()

    # Load suite config
    suite_config = _load_suite_config(args.suite)
    time_limit = suite_config.get("time_limit_seconds", 3600) if suite_config else 3600

    # Build solver configs
    solver_names = [s.strip() for s in args.solvers.split(",")]
    solver_configs = []
    for name in solver_names:
        solver_toml = _load_solver_config(name)
        if solver_toml:
            solver_configs.append(SolverConfig(
                name=name,
                command=solver_toml.get("command", name),
                solver_type=solver_toml.get("type", "internal"),
                nl_interface=solver_toml.get("nl_interface", False),
                options=solver_toml.get("options", {}),
            ))
        else:
            solver_configs.append(SolverConfig(
                name=name, command=name, solver_type="internal",
            ))

    config = BenchmarkConfig(
        suite_name=args.suite,
        time_limit=time_limit,
        num_runs=1,
        solvers=solver_configs,
    )

    runner = BenchmarkRunner(config)

    # Load instances from available .nl files
    instances, known_optima = _load_minlplib_instances(suite_config)
    runner.load_instances(instances)
    runner.load_known_optima(known_optima)

    # Run
    runner.run_all()

    # Save results
    output_path = Path(args.output) if args.output else None
    runner.save_results(output_path)

    # Print summary table
    results = runner.results
    from benchmarks.metrics import (
        final_gap_stats,
        incorrect_count,
        proved_optimal_count,
        shifted_geometric_mean,
        solved_count,
    )

    n_instances = len(results.get_instances())
    print(f"\nSuite: {args.suite} ({n_instances} instances, {time_limit}s limit)")
    print("-" * 70)
    print(f"{'Solver':<15s} {'Solved':>8s} {'Proved':>8s} {'Incorrect':>10s} "
          f"{'SGM(s)':>10s} {'Med Gap':>10s}")
    print("-" * 70)
    for solver_name in results.get_solvers():
        solver_results = results.get_results(solver_name)
        n_solved = solved_count(solver_results)
        n_proved = proved_optimal_count(solver_results)
        n_incorrect = incorrect_count(solver_results, known_optima)
        times = [r.wall_time for r in solver_results if r.is_solved]
        sgm = shifted_geometric_mean(times)
        gap_stats = final_gap_stats(solver_results)
        med_gap = gap_stats["median"]

        sgm_str = f"{sgm:.2f}" if sgm < 1e6 else "inf"
        gap_str = f"{med_gap:.2%}" if med_gap == med_gap else "N/A"
        solved_str = f"{n_solved}/{n_instances}"
        proved_str = f"{n_proved}/{n_instances}"
        print(
            f"{solver_name:<15s} {solved_str:>8s} {proved_str:>8s}"
            f" {n_incorrect:>10d} {sgm_str:>10s} {gap_str:>10s}"
        )
    print("-" * 70)

    # Append to history
    try:
        from utils.history import append_to_history
        history_path = append_to_history(args.suite, results)
        print(f"History updated: {history_path}")
    except Exception:
        pass

    # Save baseline if requested
    if args.save_baseline:
        baseline_dir = Path("reports/baselines")
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_path = baseline_dir / f"{args.suite}_baseline.json"
        results.save(baseline_path)
        print(f"Baseline saved: {baseline_path}")

    # Compare against baseline
    baseline_path_for_compare = None
    if args.compare_baseline or args.baseline:
        if args.baseline:
            baseline_path_for_compare = Path(args.baseline)
        else:
            candidate = Path("reports/baselines") / f"{args.suite}_baseline.json"
            if candidate.exists():
                baseline_path_for_compare = candidate

    if baseline_path_for_compare and baseline_path_for_compare.exists():
        from benchmarks.metrics import BenchmarkResults, detect_regressions
        baseline = BenchmarkResults.load(baseline_path_for_compare)
        for solver_name in results.get_solvers():
            alerts = detect_regressions(
                results.get_results(solver_name),
                baseline.get_results(solver_name),
            )
            if alerts:
                print(f"\nRegressions vs baseline ({solver_name}):")
                for alert in alerts:
                    severity = alert["severity"].upper()
                    print(f"  [{severity}] {alert['message']}")
            else:
                print(f"\nNo regressions vs baseline ({solver_name})")

    # Generate report if requested
    if args.report:
        try:
            from utils.reporting import generate_report
            report_path = Path("reports") / f"{args.suite}_report.md"
            generate_report(results, known_optima=known_optima, output_path=report_path)
            print(f"\nReport: {report_path}")
        except ImportError:
            print("\nReport generation requires utils.reporting module")


def _find_latest_results(suite: str) -> Path | None:
    """Find the most recent results file for a suite."""
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return None
    candidates = sorted(reports_dir.glob(f"{suite}_*.json"), reverse=True)
    return candidates[0] if candidates else None


def _load_toml_config() -> dict:
    """Load the benchmarks.toml configuration."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    config_path = Path(__file__).parent / "config" / "benchmarks.toml"
    if not config_path.exists():
        return {}
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def _load_gate_config(gate_name: str) -> dict | None:
    """Load gate configuration from TOML config."""
    config = _load_toml_config()
    return config.get("gates", {}).get(gate_name)


def _load_suite_config(suite_name: str) -> dict | None:
    """Load suite configuration from TOML config."""
    config = _load_toml_config()
    return config.get("suites", {}).get(suite_name)


def _load_solver_config(solver_name: str) -> dict | None:
    """Load solver configuration from TOML config."""
    config = _load_toml_config()
    return config.get("solvers", {}).get(solver_name)


def _load_instance_classes() -> dict[str, str]:
    """Load instance classification from config/instance_classes.toml."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    config_path = Path(__file__).parent / "config" / "instance_classes.toml"
    if not config_path.exists():
        return {}
    with open(config_path, "rb") as f:
        data = tomllib.load(f)
    return data.get("instances", {})


def _load_minlplib_instances(
    suite_config: dict | None,
) -> tuple[list, dict[str, float]]:
    """Load MINLPLib instances from the test data directory.

    Returns (instances, known_optima) where instances is a list of InstanceInfo
    and known_optima maps instance name to expected objective.
    """
    from benchmarks.metrics import InstanceInfo

    # Known optima from MINLPLib (verified by BARON/ANTIGONE/SCIP)
    known_optima_map = {
        "ex1221": 7.66718007,
        "ex1225": 31.0,
        "ex1226": -17.0,
        "st_e13": 2.0,
        "st_e15": 7.66718007,
        "st_e27": 2.0,
        "st_e38": 7197.72714900,
        "st_e40": 30.41421350,
        "nvs01": 12.46966882,
        "nvs02": 5.96418452,
        "nvs03": 16.0,
        "nvs04": 0.72,
        "nvs05": 5.47093411,
        "nvs06": 1.77031250,
        "nvs07": 4.0,
        "nvs08": 23.44972735,
        "nvs10": -310.80,
        "nvs11": -431.0,
        "nvs12": -481.20,
        "nvs14": -40358.15477,
        "nvs15": 1.0,
        "nvs16": 0.70312500,
        "nvs21": -5.68478250,
        "prob03": 10.0,
        "prob06": 1.17712434,
        "prob10": 3.44550379,
        "gear": 0.0,
        "gear3": 0.0,
        "gear4": 1.64342847,
        "chance": 29.89437816,
        "dispatch": 3155.28792700,
        "meanvar": 5.24339907,
        "alan": 2.9250,
    }

    # Load instance classifications
    instance_classes = _load_instance_classes()

    # Find .nl files
    project_root = Path(__file__).parent.parent
    nl_dirs = [
        project_root / "python" / "tests" / "data" / "minlplib",
        project_root / "python" / "tests" / "data" / "minlplib_nl",
    ]

    found_instances = {}
    for nl_dir in nl_dirs:
        if not nl_dir.exists():
            continue
        for nl_file in sorted(nl_dir.glob("*.nl")):
            name = nl_file.stem
            if name not in found_instances:
                found_instances[name] = nl_file

    # Apply suite filters
    max_vars = suite_config.get("max_variables", 10000) if suite_config else 10000
    max_instances = suite_config.get("max_instances", 10000) if suite_config else 10000
    problem_class_filter = suite_config.get("problem_class") if suite_config else None
    instance_list_inline = suite_config.get("instance_list_inline") if suite_config else None

    # If inline instance list specified, only use those
    if instance_list_inline:
        filtered = {}
        for name in instance_list_inline:
            if name in found_instances:
                filtered[name] = found_instances[name]
        found_instances = filtered

    instances = []
    for name, nl_path in sorted(found_instances.items()):
        # Try to get variable count from parsing
        try:
            from discopt._rust import parse_nl_file
            parsed = parse_nl_file(str(nl_path))
            n_vars = parsed.n_vars
            n_cons = parsed.n_constraints
        except Exception:
            n_vars = 0
            n_cons = 0

        if n_vars > max_vars:
            continue

        # Determine problem class
        prob_class = instance_classes.get(name, "unknown")

        # Filter by problem class if specified
        if problem_class_filter and prob_class != problem_class_filter:
            continue

        instances.append(InstanceInfo(
            name=name,
            num_variables=n_vars,
            num_constraints=n_cons,
            best_known_objective=known_optima_map.get(name),
            problem_class=prob_class,
            source="minlplib",
        ))

        if len(instances) >= max_instances:
            break

    return instances, known_optima_map


if __name__ == "__main__":
    main()
