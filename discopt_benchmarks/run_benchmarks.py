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
import json
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
    parser.add_argument(
        "--subprocess", action="store_true",
        help="Use the subprocess-isolated ScaledRunner (resumable, parallel). "
             "Recommended for the full MINLPLib suite."
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (only with --subprocess)"
    )
    parser.add_argument(
        "--mem-limit-mb", type=int, default=8192,
        help="Per-instance address-space cap in MB (only with --subprocess)"
    )
    parser.add_argument(
        "--scaled-out-dir", type=str, default=None,
        help="Output directory for the ScaledRunner (default: reports/<suite>_<timestamp>/)"
    )
    parser.add_argument(
        "--fetch", action="store_true",
        help="Fetch MINLPLib into the local cache before running"
    )
    parser.add_argument(
        "--minlplib-cache", type=str, default=None,
        help="Override the MINLPLib cache dir (defaults to ~/.cache/discopt/minlplib)"
    )
    parser.add_argument(
        "--minlplib-version", type=str, default="current",
        help="MINLPLib snapshot version tag (default: current)"
    )
    parser.add_argument(
        "--use-cache", action="store_true",
        help="Load instances from the MINLPLib cache directory rather than vendored test data"
    )
    parser.add_argument(
        "--pin-baseline", action="store_true",
        help="After the run, write current results to baselines/<suite>.json"
    )
    parser.add_argument(
        "--strict-time", action="store_true",
        help="In gate mode, treat >1.5x per-instance time regression as failure"
    )
    parser.add_argument(
        "--report-format", type=str, default=None,
        choices=[None, "default", "minlplib"],
        help="Report style (default = legacy markdown, minlplib = MINLPLib-style per-class table)"
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
    tiers = {
        "small":   "~60 stratified MINLPLib instances, 60s/inst   (~30 min on 8 workers)",
        "medium":  "~250 stratified MINLPLib instances, 300s/inst (~2 h on 8 workers)",
        "full":    "All MINLPLib instances, 600s/inst             (~12-30 h on 8 workers)",
    }
    other = {
        "smoke":         "Quick sanity check (10 instances, 60s limit)",
        "phase1":        "Phase 1 validation (small instances, ≤10 vars)",
        "phase2":        "Phase 2 validation (medium instances, ≤50 vars)",
        "phase3":        "Phase 3 validation (large instances, ≤100 vars)",
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
    print("\nStratified MINLPLib tiers (preferred for the --gate workflow):\n")
    for name, desc in tiers.items():
        print(f"  {name:15s}  {desc}")
    print("\nOther suites:\n")
    for name, desc in other.items():
        print(f"  {name:15s}  {desc}")
    print()


def _run_gate_check(args):
    """Run phase gate evaluation.

    Two modes:

      1. ``--gate <suite>`` with a pinned baseline at
         ``baselines/<suite>.json`` → baseline regression check
         (solved-count + incorrect-count, optionally time).
      2. ``--gate phaseN`` with TOML criteria in benchmarks.toml → phase gate
         criterion evaluation (legacy behavior).

    Mode 1 takes precedence when a baseline file exists.
    """
    from utils.baseline import (
        baseline_path as _bpath,
        compare_to_baseline,
        format_gate_report,
        load_baseline,
        save_gate_report,
    )

    print(f"\n{'='*60}")
    print(f"Gate Check: {args.gate}")
    print(f"{'='*60}\n")

    results_path = Path(args.output) if args.output else _find_latest_results(args.gate)
    if not results_path or not results_path.exists():
        print(f"ERROR: No results found for gate '{args.gate}'.")
        print(f"Run benchmarks first: python run_benchmarks.py --suite {args.gate}")
        sys.exit(1)

    benchmark = BenchmarkResults.load(results_path)

    # Mode 1: baseline regression gate
    baseline = load_baseline(args.gate)
    if baseline is not None:
        # Pull MINLPLib known optima from the cached index if available.
        known_optima = _known_optima_for_gate(args)
        report = compare_to_baseline(
            current=benchmark,
            baseline=baseline,
            suite=args.gate,
            solver="discopt",
            known_optima=known_optima,
            strict_time=args.strict_time,
        )
        print(format_gate_report(report))
        out = Path("reports") / f"{args.gate}_gate_report.json"
        save_gate_report(report, out)
        print(f"\nGate report saved: {out}")
        print(f"Baseline: {_bpath(args.gate)}")
        sys.exit(0 if report.passed else 1)

    # Mode 2: TOML phase gate criteria
    gate_config = _load_gate_config(args.gate)
    if not gate_config:
        print(f"ERROR: No baseline at baselines/{args.gate}.json AND no gate "
              f"configuration in benchmarks.toml for '{args.gate}'")
        sys.exit(1)

    all_passed, criteria = evaluate_phase_gate(
        args.gate, benchmark, gate_config, known_optima=_known_optima_for_gate(args)
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


def _known_optima_for_gate(args) -> dict[str, float]:
    """Best-effort known optima for gate correctness checks.

    Merges two self-contained sources with the (optional) MINLPLib cache:
      * the committed certification oracle file
        ``docs/dev/data/cert-optima.json`` (global50 BARON optima + perf-panel
        oracles) — always available, no external fetch (cert:T0.5);
      * the MINLPLib cache index, when present, which the cache values override
        the committed file with (fresher, larger).
    """
    optima: dict[str, float] = {}
    # Committed oracle file (repo-relative), so the cert0 gate is evaluable
    # without a MINLPLib checkout.
    cert_optima = (
        Path(__file__).resolve().parent.parent / "docs" / "dev" / "data" / "cert-optima.json"
    )
    try:
        if cert_optima.exists():
            with open(cert_optima) as fh:
                optima.update({k: float(v) for k, v in json.load(fh).items()})
    except Exception:
        pass
    try:
        from scripts.fetch_minlplib import get_cache_dir, get_instancedata_path
        from utils.minlplib_data import known_optima_from_index, load_instance_data

        cache = Path(args.minlplib_cache).expanduser() if args.minlplib_cache else get_cache_dir()
        csv = get_instancedata_path(cache, args.minlplib_version)
        if csv.exists():
            index = load_instance_data(csv)
            optima.update(known_optima_from_index(index))
    except Exception:
        pass
    return optima


def _load_minlplib_from_cache(
    args,
    suite_config: dict | None,
) -> tuple[list, dict[str, float]]:
    """Load instances + known optima from the fetched MINLPLib cache.

    Mirrors _load_minlplib_instances but uses the cache directory and the
    full instancedata.csv index instead of the vendored ~30 .nl files.
    """
    from scripts.fetch_minlplib import get_cache_dir, get_instancedata_path, get_nl_dir
    from utils.minlplib_data import known_optima_from_index, load_instance_data
    from benchmarks.metrics import InstanceInfo

    cache = Path(args.minlplib_cache).expanduser() if args.minlplib_cache else get_cache_dir()
    nl_dir = get_nl_dir(cache, args.minlplib_version)
    csv = get_instancedata_path(cache, args.minlplib_version)

    if not nl_dir.is_dir():
        print(f"ERROR: MINLPLib cache not populated at {nl_dir}. Run with --fetch first.",
              file=sys.stderr)
        sys.exit(1)

    index = load_instance_data(csv) if csv.exists() else {}
    known_optima = known_optima_from_index(index)

    max_vars = suite_config.get("max_variables", 10**9) if suite_config else 10**9
    max_cons = suite_config.get("max_constraints", 10**9) if suite_config else 10**9
    max_instances = suite_config.get("max_instances", 10**9) if suite_config else 10**9
    problem_class_filter = suite_config.get("problem_class") if suite_config else None
    inline = suite_config.get("instance_list_inline") if suite_config else None
    list_file = suite_config.get("instance_list") if suite_config else None

    instances: list[InstanceInfo] = []
    nl_files = sorted(nl_dir.glob("*.nl"))
    allowed = _read_instance_list(list_file)
    if inline:
        allowed = (allowed or set()) | set(inline)
    if allowed is not None:
        nl_files = [p for p in nl_files if p.stem in allowed]
        # Preserve list-file ordering: emit instances in the order they appear,
        # falling back to alphabetical for anything not in the list.
        order = _instance_list_order(list_file)
        if order:
            rank = {n: i for i, n in enumerate(order)}
            nl_files.sort(key=lambda p: rank.get(p.stem, len(order)))

    for nl in nl_files:
        name = nl.stem
        meta = index.get(name)
        n_vars = meta.n_vars if meta else 0
        n_cons = meta.n_constraints if meta else 0
        cat = meta.category_bucket if meta else "unknown"

        if n_vars and n_vars > max_vars:
            continue
        if n_cons and n_cons > max_cons:
            continue
        if problem_class_filter and cat.lower() != problem_class_filter.lower():
            continue

        instances.append(InstanceInfo(
            name=name,
            num_variables=n_vars,
            num_constraints=n_cons,
            best_known_objective=known_optima.get(name),
            problem_class=cat.lower(),
            source="minlplib",
        ))
        if len(instances) >= max_instances:
            break

    return instances, known_optima


def _run_scaled(args, suite_config: dict, instances: list, known_optima: dict[str, float]) -> Path:
    """Run the subprocess-isolated ScaledRunner and return the consolidated JSON path."""
    from benchmarks.scaled_runner import ScaledConfig, ScaledRunner
    from scripts.fetch_minlplib import get_cache_dir, get_nl_dir

    time_limit = suite_config.get("time_limit_seconds", 3600) if suite_config else 3600

    out_dir = Path(args.scaled_out_dir) if args.scaled_out_dir else (
        Path("reports") / f"{args.suite}_{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}"
    )

    cache = Path(args.minlplib_cache).expanduser() if args.minlplib_cache else get_cache_dir()
    nl_dir = get_nl_dir(cache, args.minlplib_version)

    # Build (name, nl_path) pairs from the InstanceInfo list.
    instance_specs: list[tuple[str, Path]] = []
    info_by_name = {inst.name: inst for inst in instances}
    for inst in instances:
        nl_path = nl_dir / f"{inst.name}.nl"
        if not nl_path.exists():
            # Fallback to vendored data
            for candidate in [
                Path("python/tests/data/minlplib_nl") / f"{inst.name}.nl",
                Path("python/tests/data/minlplib") / f"{inst.name}.nl",
            ]:
                if candidate.exists():
                    nl_path = candidate
                    break
        if nl_path.exists():
            instance_specs.append((inst.name, nl_path))
        else:
            print(f"  skip {inst.name}: no .nl file found", file=sys.stderr)

    cfg = ScaledConfig(
        suite_name=args.suite,
        out_dir=out_dir,
        time_limit=time_limit,
        mem_limit_mb=args.mem_limit_mb,
        n_workers=args.workers,
        solver_name="discopt",
        solver_options={},
    )
    runner = ScaledRunner(cfg)
    runner.run(instance_specs, instance_info=info_by_name)
    consolidated = runner.save_consolidated()
    print(f"\nConsolidated results: {consolidated}")

    # Append history entry as the in-process path does.
    try:
        from utils.history import append_to_history
        results = runner.collect()
        hp = append_to_history(args.suite, results)
        print(f"History updated: {hp}")
    except Exception as e:  # noqa: BLE001
        print(f"history update skipped: {e}")

    # Pin baseline if requested
    if args.pin_baseline:
        from utils.baseline import save_baseline
        results = runner.collect()
        bp = save_baseline(results, args.suite)
        print(f"Baseline pinned: {bp}")

    # Reports
    if args.report or args.report_format:
        _emit_reports(args, runner.collect(), known_optima)

    return consolidated


def _emit_reports(args, results: BenchmarkResults, known_optima: dict[str, float]) -> None:
    fmt = args.report_format or "default"
    if fmt == "minlplib":
        from utils.reporting import generate_minlplib_report
        from utils.minlplib_data import load_instance_data
        from scripts.fetch_minlplib import get_cache_dir, get_instancedata_path

        cache = Path(args.minlplib_cache).expanduser() if args.minlplib_cache else get_cache_dir()
        csv = get_instancedata_path(cache, args.minlplib_version)
        index = load_instance_data(csv) if csv.exists() else {}
        out = Path("reports") / f"{args.suite}_minlplib_report.md"
        generate_minlplib_report(results, index, output_path=out)
    else:
        from utils.reporting import generate_report
        out = Path("reports") / f"{args.suite}_report.md"
        generate_report(results, known_optima=known_optima, output_path=out)


def _run_benchmark(args):
    """Run a benchmark suite."""
    from benchmarks.runner import BenchmarkConfig, BenchmarkRunner, SolverConfig

    # Fetch MINLPLib first if requested
    if args.fetch:
        from scripts.fetch_minlplib import fetch, get_cache_dir
        cache = Path(args.minlplib_cache).expanduser() if args.minlplib_cache else get_cache_dir()
        fetch(cache_dir=cache, version=args.minlplib_version)

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

    # Load instances either from the MINLPLib cache or the vendored test data.
    if args.use_cache:
        instances, known_optima = _load_minlplib_from_cache(args, suite_config)
    else:
        instances, known_optima = _load_minlplib_instances(suite_config)

    # Subprocess-isolated, resumable path (recommended for full MINLPLib).
    if args.subprocess:
        _run_scaled(args, suite_config or {}, instances, known_optima)
        return

    runner = BenchmarkRunner(config)
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

    # Pin baseline (writes to checked-in baselines/ directory)
    if args.pin_baseline:
        from utils.baseline import save_baseline
        bp = save_baseline(results, args.suite)
        print(f"Pinned baseline: {bp}")

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
    if args.report or args.report_format:
        try:
            _emit_reports(args, results, known_optima)
        except ImportError as e:
            print(f"\nReport generation failed: {e}")


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


def _read_instance_list(path: str | Path | None) -> set[str] | None:
    """Parse an instance-list file (one name per line; '#'-comments ignored).

    Returns None if no path given (meaning "no filter"), or a set of names
    if the file exists. A non-existent path returns an empty set so the caller
    sees "filtered to nothing" rather than silently running everything.
    """
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).parent / p
    if not p.exists():
        print(f"WARN: instance_list {p} not found", file=sys.stderr)
        return set()
    names: set[str] = set()
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.add(line)
    return names


def _instance_list_order(path: str | Path | None) -> list[str]:
    """Same as _read_instance_list but preserves file order."""
    if not path:
        return []
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).parent / p
    if not p.exists():
        return []
    out: list[str] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


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
