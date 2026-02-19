"""
Benchmark Report Generator

Produces markdown reports with tables, metrics, and phase gate results.
Designed for both human consumption and CI/CD integration.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from benchmarks.metrics import (
    BenchmarkResults,
    GateCriterionResult,
    SolveResult,
    bound_quality_rate,
    evaluate_phase_gate,
    final_gap_stats,
    gpu_vs_cpu_speedup,
    incorrect_count,
    iteration_stats,
    layer_profiling_summary,
    proved_optimal_count,
    proved_optimal_rate,
    root_gap_analysis,
    shifted_geometric_mean,
    solved_count,
    solved_count_by_size,
    speedup_table,
    geometric_mean_ratio,
)


def generate_report(
    benchmark: BenchmarkResults,
    gate_name: Optional[str] = None,
    gate_config: Optional[dict] = None,
    reference_solvers: Optional[dict[str, list[SolveResult]]] = None,
    known_optima: Optional[dict[str, float]] = None,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate a comprehensive markdown benchmark report.

    Returns the report as a string and optionally writes to file.
    """
    lines = []
    solvers = benchmark.get_solvers()
    instances = benchmark.get_instances()

    # ── Header ──
    lines.append(f"# discopt Benchmark Report: {benchmark.suite}")
    lines.append(f"")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Benchmark timestamp:** {benchmark.timestamp}")
    lines.append(f"**Instances:** {len(instances)}")
    lines.append(f"**Solvers:** {', '.join(solvers)}")
    lines.append("")

    # ── Summary Table ──
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | " + " | ".join(solvers) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(solvers)) + "|")

    # Solved count
    row = "| **Solved (global opt)** |"
    for s in solvers:
        row += f" {solved_count(benchmark.get_results(s))} |"
    lines.append(row)

    # Solved by size categories
    for max_v in [10, 30, 50, 100]:
        row = f"| Solved (≤{max_v} vars) |"
        for s in solvers:
            row += f" {solved_count_by_size(benchmark.get_results(s), benchmark.instance_info, max_v)} |"
        lines.append(row)

    # Geometric mean time
    row = "| **Geom. mean time (s)** |"
    for s in solvers:
        results = benchmark.get_results(s)
        times = [r.wall_time for r in results if r.is_solved]
        gm = shifted_geometric_mean(times)
        row += f" {gm:.2f} |"
    lines.append(row)

    # Incorrect count
    if known_optima:
        row = "| **Incorrect results** |"
        for s in solvers:
            ic = incorrect_count(benchmark.get_results(s), known_optima)
            marker = "🔴" if ic > 0 else "✅"
            row += f" {marker} {ic} |"
        lines.append(row)

    lines.append("")

    # ── Pairwise Comparisons ──
    if len(solvers) >= 2 and "discopt" in solvers:
        lines.append("## Pairwise Comparison (discopt vs Others)")
        lines.append("")
        lines.append("| Comparison | Geom. Mean Ratio | discopt Faster | Other Faster | Common Solved |")
        lines.append("|------------|------------------|-----------------|--------------|---------------|")

        jax_results = benchmark.get_results("discopt")
        jax_times = {r.instance: r.wall_time for r in jax_results if r.is_solved}

        for s in solvers:
            if s == "discopt":
                continue
            other_results = benchmark.get_results(s)
            ratio = geometric_mean_ratio(jax_results, other_results)

            other_times = {r.instance: r.wall_time for r in other_results if r.is_solved}
            common = set(jax_times.keys()) & set(other_times.keys())
            jax_faster = sum(1 for i in common if jax_times[i] < other_times[i])
            other_faster = sum(1 for i in common if jax_times[i] > other_times[i])

            ratio_str = f"{ratio:.2f}x" if ratio < 100 else ">100x"
            lines.append(
                f"| vs {s} | {ratio_str} | {jax_faster} | {other_faster} | {len(common)} |"
            )
        lines.append("")

    # ── Global Optimality ──
    if known_optima:
        lines.append("## Global Optimality")
        lines.append("")
        lines.append("| Metric | " + " | ".join(solvers) + " |")
        lines.append("|--------|" + "|".join(["--------"] * len(solvers)) + "|")

        row = "| Proved optimal |"
        for s in solvers:
            row += f" {proved_optimal_count(benchmark.get_results(s))} |"
        lines.append(row)

        row = "| Proved optimal rate |"
        for s in solvers:
            rate = proved_optimal_rate(benchmark.get_results(s), known_optima)
            row += f" {rate:.1%} |"
        lines.append(row)

        for label, key in [("Mean final gap", "mean"), ("Median final gap", "median")]:
            row = f"| {label} |"
            for s in solvers:
                stats = final_gap_stats(benchmark.get_results(s))
                val = stats[key]
                val_str = f"{val:.2%}" if not np.isnan(val) else "N/A"
                row += f" {val_str} |"
            lines.append(row)

        row = "| Bound quality rate |"
        for s in solvers:
            bqr = bound_quality_rate(benchmark.get_results(s), known_optima)
            row += f" {bqr:.1%} |"
        lines.append(row)

        lines.append("")

    # ── Per-Class Breakdown ──
    classes: dict[str, list[str]] = {}
    for name, info in benchmark.instance_info.items():
        pc = info.problem_class
        if pc and pc != "unknown":
            if pc not in classes:
                classes[pc] = []
            classes[pc].append(name)

    if classes:
        lines.append("## Results by Problem Class")
        lines.append("")
        lines.append(
            "| Class | Instances | "
            + " | ".join(f"{s} solved" for s in solvers)
            + " |"
        )
        lines.append(
            "|-------|-----------|"
            + "|".join(["--------"] * len(solvers))
            + "|"
        )

        for pc in sorted(classes.keys()):
            pc_set = set(classes[pc])
            row = f"| {pc} | {len(pc_set)} |"
            for s in solvers:
                s_solved = sum(
                    1
                    for r in benchmark.get_results(s)
                    if r.is_solved and r.instance in pc_set
                )
                row += f" {s_solved}/{len(pc_set)} |"
            lines.append(row)
        lines.append("")

    # ── Root Gap Analysis ──
    lines.append("## Root Gap Analysis")
    lines.append("")
    lines.append("| Solver | Mean Root Gap | Median Root Gap | Max Root Gap |")
    lines.append("|--------|-------------- |-----------------|-------------|")

    for s in solvers:
        rga = root_gap_analysis(benchmark.get_results(s))
        lines.append(
            f"| {s} | {rga['mean']:.4f} | {rga['median']:.4f} | {rga['max']:.4f} |"
        )
    lines.append("")

    # ── Layer Profiling (discopt-specific) ──
    if "discopt" in solvers:
        jax_results = benchmark.get_results("discopt")
        profile = layer_profiling_summary(jax_results)
        if not np.isnan(profile["mean_rust_fraction"]):
            lines.append("## Layer Profiling (discopt)")
            lines.append("")
            lines.append("| Layer | Mean Time Fraction | Target |")
            lines.append("|-------|-------------------|--------|")
            lines.append(f"| Rust (tree, LP, sparse LA) | {profile['mean_rust_fraction']:.1%} | — |")
            lines.append(f"| JAX (relaxations, autodiff, NLP) | {profile['mean_jax_fraction']:.1%} | — |")
            lines.append(f"| Python orchestration | {profile['mean_python_fraction']:.1%} | <5% |")
            lines.append(f"| Max Python overhead | {profile['max_python_fraction']:.1%} | <10% |")
            lines.append("")

    # ── GPU vs CPU ──
    if "discopt" in solvers and "discopt_cpu" in solvers:
        gpu_stats = gpu_vs_cpu_speedup(
            benchmark.get_results("discopt"),
            benchmark.get_results("discopt_cpu"),
        )
        lines.append("## GPU vs CPU Comparison")
        lines.append("")
        lines.append(f"- **Mean speedup:** {gpu_stats['mean_speedup']:.1f}x")
        lines.append(f"- **Median speedup:** {gpu_stats['median_speedup']:.1f}x")
        lines.append(f"- **Max speedup:** {gpu_stats['max_speedup']:.1f}x")
        lines.append(f"- **Min speedup:** {gpu_stats['min_speedup']:.2f}x")
        lines.append(f"- **Instances compared:** {gpu_stats['count']}")
        lines.append("")

    # ── Node Throughput ──
    if "discopt" in solvers:
        jax_results = benchmark.get_results("discopt")
        nps = [
            r.nodes_per_second for r in jax_results
            if r.nodes_per_second is not None and r.nodes_per_second > 0
        ]
        if nps:
            lines.append("## Node Throughput")
            lines.append("")
            lines.append(f"- **Mean:** {np.mean(nps):.0f} nodes/sec")
            lines.append(f"- **Median:** {np.median(nps):.0f} nodes/sec")
            lines.append(f"- **Min:** {np.min(nps):.0f} nodes/sec")
            lines.append(f"- **Max:** {np.max(nps):.0f} nodes/sec")
            lines.append("")

    # ── Phase Gate Evaluation ──
    if gate_name and gate_config:
        lines.append(f"## Phase Gate: {gate_name}")
        lines.append("")

        all_passed, criteria = evaluate_phase_gate(
            gate_name, benchmark, gate_config,
            reference_solvers=reference_solvers,
            known_optima=known_optima,
        )

        status = "✅ PASSED" if all_passed else "🔴 FAILED"
        lines.append(f"**Overall: {status}**")
        lines.append("")
        lines.append("| Criterion | Target | Actual | Status |")
        lines.append("|-----------|--------|--------|--------|")

        for c in criteria:
            if c.direction == "min":
                target_str = f"≥ {c.target}"
            else:
                target_str = f"≤ {c.target}"

            if np.isnan(c.actual):
                actual_str = "N/A"
            elif isinstance(c.actual, float) and c.actual == int(c.actual):
                actual_str = f"{int(c.actual)}"
            elif isinstance(c.actual, float):
                actual_str = f"{c.actual:.4f}"
            else:
                actual_str = str(c.actual)

            status_mark = "✅" if c.passed else "🔴"
            lines.append(
                f"| {c.name} | {target_str} | {actual_str} | {status_mark} |"
            )
        lines.append("")

    # ── Instance-Level Results (Top 20 hardest) ──
    if "discopt" in solvers:
        lines.append("## Hardest Instances (by solve time)")
        lines.append("")
        jax_results = benchmark.get_results("discopt")
        solved_results = sorted(
            [r for r in jax_results if r.is_solved],
            key=lambda r: r.wall_time,
            reverse=True,
        )[:20]

        if solved_results:
            lines.append("| Instance | Time (s) | Nodes | Root Gap | Vars |")
            lines.append("|----------|----------|-------|----------|------|")
            for r in solved_results:
                info = benchmark.instance_info.get(r.instance)
                nvars = info.num_variables if info else "?"
                rg = f"{r.root_gap:.4f}" if r.root_gap is not None else "—"
                lines.append(
                    f"| {r.instance} | {r.wall_time:.1f} | {r.node_count} | {rg} | {nvars} |"
                )
            lines.append("")

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"Report written to: {output_path}")

    return report


def generate_ci_summary(
    benchmark: BenchmarkResults,
    baseline: Optional[BenchmarkResults] = None,
    known_optima: Optional[dict[str, float]] = None,
) -> dict:
    """
    Generate a compact CI-friendly summary (JSON).

    Suitable for GitHub Actions annotations or CI dashboards.
    """
    jax_results = benchmark.get_results("discopt")

    summary = {
        "suite": benchmark.suite,
        "timestamp": benchmark.timestamp,
        "solved_count": solved_count(jax_results),
        "total_instances": len(benchmark.get_instances()),
        "incorrect_count": (
            incorrect_count(jax_results, known_optima) if known_optima else None
        ),
    }

    # Add size breakdowns
    for max_v in [10, 30, 50, 100]:
        summary[f"solved_le{max_v}var"] = solved_count_by_size(
            jax_results, benchmark.instance_info, max_v
        )

    # Geomean time
    times = [r.wall_time for r in jax_results if r.is_solved]
    summary["geomean_time"] = shifted_geometric_mean(times)

    # Layer profiling
    profile = layer_profiling_summary(jax_results)
    summary["python_overhead"] = profile["mean_python_fraction"]

    # Regression check
    if baseline:
        from benchmarks.metrics import detect_regressions
        baseline_results = baseline.get_results("discopt")
        regressions = detect_regressions(jax_results, baseline_results)
        summary["regressions"] = len(regressions)
        summary["regression_details"] = regressions
    else:
        summary["regressions"] = None

    return summary


# ─────────────────────────────────────────────────────────────
# CUTEst Performance Profile Generation
# ─────────────────────────────────────────────────────────────


def generate_cutest_report(
    benchmark: BenchmarkResults,
    known_optima: Optional[dict[str, float]] = None,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate a CUTEst-specific benchmark report with performance profiles.

    Includes:
    - Summary table with solve rates per solver
    - Dolan-Moré performance profiles (text-based)
    - Results stratified by problem class (unconstrained / bound / general)
    - Shifted geometric mean time ratios
    """
    from benchmarks.metrics import performance_profile

    lines = []
    solvers = benchmark.get_solvers()
    instances = benchmark.get_instances()

    # ── Header ──
    lines.append(f"# CUTEst Benchmark Report: {benchmark.suite}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Instances:** {len(instances)}")
    lines.append(f"**Solvers:** {', '.join(solvers)}")
    lines.append("")

    # ── Summary Table ──
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | " + " | ".join(solvers) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(solvers)) + "|")

    # Solve count
    row = "| **Solved** |"
    for s in solvers:
        row += f" {solved_count(benchmark.get_results(s))} / {len(instances)} |"
    lines.append(row)

    # Solve rate
    row = "| **Solve rate** |"
    for s in solvers:
        rate = solved_count(benchmark.get_results(s)) / max(len(instances), 1)
        row += f" {rate:.1%} |"
    lines.append(row)

    # Geometric mean time
    row = "| **Geom. mean time (s)** |"
    for s in solvers:
        results = benchmark.get_results(s)
        times = [r.wall_time for r in results if r.is_solved]
        gm = shifted_geometric_mean(times)
        row += f" {gm:.3f} |"
    lines.append(row)

    # Incorrect count
    if known_optima:
        row = "| **Incorrect** |"
        for s in solvers:
            ic = incorrect_count(benchmark.get_results(s), known_optima)
            marker = "FAIL" if ic > 0 else "OK"
            row += f" {marker} ({ic}) |"
        lines.append(row)
    lines.append("")

    # ── Pairwise Comparisons ──
    if len(solvers) >= 2:
        lines.append("## Pairwise Comparisons")
        lines.append("")
        lines.append(
            "| Solver A | Solver B | Geom. Mean Ratio | A Faster | B Faster | Common |"
        )
        lines.append(
            "|----------|----------|------------------|----------|----------|--------|"
        )

        for i, sa in enumerate(solvers):
            for sb in solvers[i + 1 :]:
                ra = benchmark.get_results(sa)
                rb = benchmark.get_results(sb)
                ratio = geometric_mean_ratio(ra, rb)

                ta = {r.instance: r.wall_time for r in ra if r.is_solved}
                tb = {r.instance: r.wall_time for r in rb if r.is_solved}
                common = set(ta.keys()) & set(tb.keys())
                a_faster = sum(1 for inst in common if ta[inst] < tb[inst])
                b_faster = sum(1 for inst in common if ta[inst] > tb[inst])

                ratio_str = f"{ratio:.3f}x" if ratio < 100 else ">100x"
                lines.append(
                    f"| {sa} | {sb} | {ratio_str} | {a_faster} | {b_faster} | {len(common)} |"
                )
        lines.append("")

    # ── Stratified by Problem Class ──
    classes: dict[str, list[str]] = {}
    for name, info in benchmark.instance_info.items():
        pc = info.problem_class
        if pc not in classes:
            classes[pc] = []
        classes[pc].append(name)

    if classes:
        lines.append("## Results by Problem Class")
        lines.append("")
        lines.append(
            "| Class | Instances | " + " | ".join(f"{s} solved" for s in solvers) + " |"
        )
        lines.append(
            "|-------|-----------|" + "|".join(["--------"] * len(solvers)) + "|"
        )

        for pc in sorted(classes.keys()):
            pc_instances = set(classes[pc])
            row = f"| {pc} | {len(pc_instances)} |"
            for s in solvers:
                s_solved = sum(
                    1
                    for r in benchmark.get_results(s)
                    if r.is_solved and r.instance in pc_instances
                )
                row += f" {s_solved} |"
            lines.append(row)
        lines.append("")

    # ── Performance Profile (text-based) ──
    profiles = performance_profile(benchmark, tau_max=100.0, tau_steps=20)
    if profiles:
        lines.append("## Performance Profile (Dolan-Moré)")
        lines.append("")
        lines.append(
            "Fraction of problems solved within factor tau of the best solver."
        )
        lines.append("")

        # Table format
        first_solver = next(iter(profiles))
        tau_values = profiles[first_solver][0]
        header = "| tau |"
        sep = "|-----|"
        for s in profiles:
            header += f" {s} |"
            sep += "--------|"
        lines.append(header)
        lines.append(sep)

        for j, tau in enumerate(tau_values):
            row = f"| {tau:.1f} |"
            for s in profiles:
                frac = profiles[s][1][j]
                row += f" {frac:.2f} |"
            lines.append(row)
        lines.append("")

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"CUTEst report written to: {output_path}")

    return report


def generate_cutest_performance_profile_data(
    benchmark: BenchmarkResults,
) -> dict[str, dict]:
    """
    Generate performance profile data for CUTEst results in JSON-serializable format.

    Returns dict with:
      - "profiles": {solver: {"tau": [...], "fraction": [...]}}
      - "stratified": {class: {solver: {"solved": int, "total": int, "geomean": float}}}
    """
    from benchmarks.metrics import performance_profile

    profiles = performance_profile(benchmark, tau_max=100.0, tau_steps=100)

    result = {"profiles": {}, "stratified": {}}

    for solver, (tau, frac) in profiles.items():
        result["profiles"][solver] = {
            "tau": tau.tolist(),
            "fraction": frac.tolist(),
        }

    # Stratified results
    classes: dict[str, list[str]] = {}
    for name, info in benchmark.instance_info.items():
        pc = info.problem_class
        if pc not in classes:
            classes[pc] = []
        classes[pc].append(name)

    for pc, pc_instances in classes.items():
        pc_set = set(pc_instances)
        result["stratified"][pc] = {}
        for solver in benchmark.get_solvers():
            solver_results = benchmark.get_results(solver)
            pc_results = [r for r in solver_results if r.instance in pc_set]
            pc_solved = [r for r in pc_results if r.is_solved]
            times = [r.wall_time for r in pc_solved]

            result["stratified"][pc][solver] = {
                "solved": len(pc_solved),
                "total": len(pc_results),
                "geomean_time": shifted_geometric_mean(times),
            }

    return result


# ─────────────────────────────────────────────────────────────
# Per-Category Benchmark Report
# ─────────────────────────────────────────────────────────────


def generate_category_report(
    benchmark: BenchmarkResults,
    category: str,
    level: str = "smoke",
    known_optima: Optional[dict[str, float]] = None,
    output_path: Optional[Path] = None,
) -> str:
    """Generate a markdown report for a single category benchmark run.

    Includes summary table, speedup matrix, correctness validation,
    per-problem detail, and gap analysis (for global_opt).
    """
    lines = []
    solvers = benchmark.get_solvers()
    instances = benchmark.get_instances()

    # ── Header ──
    lines.append(
        f"# Category Benchmark: {category.upper()} ({level})"
    )
    lines.append("")
    lines.append(
        f"**Generated:** "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append(f"**Category:** {category}")
    lines.append(f"**Level:** {level}")
    lines.append(f"**Instances:** {len(instances)}")
    lines.append(f"**Solvers:** {', '.join(solvers)}")
    lines.append("")

    # ── Summary Table ──
    lines.append("## Summary")
    lines.append("")
    header = "| Metric |"
    sep = "|--------|"
    for s in solvers:
        header += f" {s} |"
        sep += "--------|"
    lines.append(header)
    lines.append(sep)

    # Solved count
    row = "| **Solved** |"
    for s in solvers:
        n = solved_count(benchmark.get_results(s))
        row += f" {n}/{len(instances)} |"
    lines.append(row)

    # Incorrect count
    if known_optima:
        row = "| **Incorrect** |"
        for s in solvers:
            ic = incorrect_count(
                benchmark.get_results(s), known_optima
            )
            marker = "FAIL" if ic > 0 else "OK"
            row += f" {marker} ({ic}) |"
        lines.append(row)

    # SGM time
    row = "| **SGM time (s)** |"
    for s in solvers:
        results = benchmark.get_results(s)
        times = [r.wall_time for r in results if r.is_solved]
        gm = shifted_geometric_mean(times)
        gm_str = f"{gm:.3f}" if gm < 1e6 else "inf"
        row += f" {gm_str} |"
    lines.append(row)

    # Median iterations
    row = "| **Med. iterations** |"
    for s in solvers:
        istats = iteration_stats(benchmark.get_results(s))
        med = istats["median"]
        med_str = f"{med:.0f}" if not np.isnan(med) else "N/A"
        row += f" {med_str} |"
    lines.append(row)

    lines.append("")

    # ── Speedup Matrix ──
    if len(solvers) >= 2:
        lines.append("## Speedup Matrix (SGM ratio)")
        lines.append("")
        lines.append(
            "Values < 1.0 mean the row solver is faster "
            "than the column solver."
        )
        lines.append("")
        header = "| |"
        sep = "|---|"
        for s in solvers:
            header += f" {s} |"
            sep += "---|"
        lines.append(header)
        lines.append(sep)

        table = speedup_table(benchmark)
        for sa in solvers:
            row = f"| **{sa}** |"
            for sb in solvers:
                ratio = table.get((sa, sb), float("nan"))
                if sa == sb:
                    row += " — |"
                elif ratio < 100:
                    row += f" {ratio:.2f}x |"
                else:
                    row += " >100x |"
            lines.append(row)
        lines.append("")

    # ── Correctness Validation ──
    if known_optima:
        lines.append("## Correctness Validation")
        lines.append("")
        total_incorrect = 0
        for s in solvers:
            ic = incorrect_count(
                benchmark.get_results(s), known_optima
            )
            total_incorrect += ic
        if total_incorrect == 0:
            lines.append(
                "All solvers produced correct results "
                "(0 incorrect across all solvers)."
            )
        else:
            lines.append(
                f"**WARNING:** {total_incorrect} incorrect "
                f"result(s) detected!"
            )
        lines.append("")

    # ── Per-Problem Detail ──
    lines.append("## Per-Problem Results")
    lines.append("")
    header = "| Instance |"
    sep = "|----------|"
    for s in solvers:
        header += f" {s} status | {s} time | {s} obj |"
        sep += "---|---|---|"
    lines.append(header)
    lines.append(sep)

    for inst in instances:
        row = f"| {inst} |"
        for s in solvers:
            results = benchmark.get_results(s)
            match = [r for r in results if r.instance == inst]
            if match:
                r = match[0]
                st = r.status.value[:3].upper()
                t = (
                    f"{r.wall_time:.2f}s"
                    if r.wall_time < float("inf")
                    else "TL"
                )
                obj = (
                    f"{r.objective:.4g}"
                    if r.objective is not None
                    else "—"
                )
                row += f" {st} | {t} | {obj} |"
            else:
                row += " — | — | — |"
        lines.append(row)
    lines.append("")

    # ── Gap Analysis (global_opt only) ──
    if category == "global_opt" and known_optima:
        lines.append("## Gap Analysis")
        lines.append("")
        for s in solvers:
            results = benchmark.get_results(s)
            n_proved = proved_optimal_count(results)
            gap_stats = final_gap_stats(results)
            bqr = bound_quality_rate(results, known_optima)
            lines.append(f"### {s}")
            lines.append(f"- Proved optimal: {n_proved}")
            mean_g = gap_stats["mean"]
            med_g = gap_stats["median"]
            lines.append(
                f"- Mean gap: "
                f"{'N/A' if np.isnan(mean_g) else f'{mean_g:.4%}'}"
            )
            lines.append(
                f"- Median gap: "
                f"{'N/A' if np.isnan(med_g) else f'{med_g:.4%}'}"
            )
            lines.append(f"- Bound quality rate: {bqr:.1%}")
            lines.append("")

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        print(f"Category report written to: {output_path}")

    return report
