#!/usr/bin/env python3
"""Per-category benchmark CLI.

Usage:
    # Run LP smoke benchmarks with report
    python run_category_benchmarks.py --category lp --level smoke --report

    # Run all categories at smoke level with HTML dashboard
    python run_category_benchmarks.py --category all --level smoke --html

    # Run QP full benchmarks
    python run_category_benchmarks.py --category qp --level full --report --html

    # List available categories
    python run_category_benchmarks.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).parent))

from benchmarks.problems.base import get_all_categories


def main():
    parser = argparse.ArgumentParser(
        description="Per-category benchmark runner for discopt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--category",
        type=str,
        default="lp",
        help=("Category to benchmark: lp|qp|milp|miqp|minlp|global_opt|all (default: lp)"),
    )
    parser.add_argument(
        "--level",
        type=str,
        default="smoke",
        choices=["smoke", "full"],
        help="Benchmark level (default: smoke)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate markdown report",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML dashboard",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: results/)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=300.0,
        help="Per-problem time limit in seconds (default: 300)",
    )
    parser.add_argument(
        "--solvers",
        type=str,
        default=None,
        help=(
            "Comma-separated solver filter. Supports ipm,ripopt,ipopt,highs,amp. "
            "Use --solvers amp for AMP-specific category benchmarks."
        ),
    )
    parser.add_argument(
        "--hard-timeout-grace",
        type=float,
        default=2.0,
        help=(
            "Extra seconds before forcibly killing a per-problem worker "
            "after --time-limit (default: 2)"
        ),
    )
    parser.add_argument(
        "--no-hard-timeout",
        action="store_true",
        help="Run category cases in-process without subprocess hard timeouts",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available categories",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full tracebacks on errors",
    )

    args = parser.parse_args()

    if args.list:
        _list_categories()
        return

    solver_filter = _parse_solver_filter(args.solvers)
    categories = get_all_categories() if args.category == "all" else [args.category]

    # Validate categories
    valid = set(get_all_categories())
    for cat in categories:
        if cat not in valid:
            print(f"ERROR: Unknown category '{cat}'. Valid: {', '.join(sorted(valid))}")
            sys.exit(1)

    # Run each category
    all_results = []
    for cat in categories:
        results = _run_category(
            cat,
            args.level,
            args.time_limit,
            args.output,
            args.report,
            args.html,
            None if args.no_hard_timeout else args.hard_timeout_grace,
            solver_filter,
        )
        all_results.append((cat, results))

    # Combined summary for --category all
    if len(categories) > 1:
        _print_combined_summary(all_results, args.level)


def _list_categories():
    """List available benchmark categories."""
    categories = {
        "lp": "Linear programming (LP) — ipm, ripopt, ipopt, highs",
        "qp": "Quadratic programming (QP) — ipm, ripopt, ipopt",
        "milp": "Mixed-integer LP (MILP) — ipm, ripopt, ipopt",
        "miqp": "Mixed-integer QP (MIQP) — ipm, ripopt, ipopt",
        "minlp": "Mixed-integer NLP (MINLP) — ipm, ripopt, ipopt; use --solvers amp for AMP",
        "global_opt": "Global optimization — ipm, ripopt, ipopt; use --solvers amp for AMP",
    }
    print("\nAvailable benchmark categories:\n")
    for name, desc in categories.items():
        print(f"  {name:12s}  {desc}")
    print("\n  all          Run all categories sequentially")
    print()


def _parse_solver_filter(raw: str | None) -> list[str] | None:
    """Parse and validate a comma-separated solver filter."""
    if raw is None:
        return None
    solvers = [part.strip() for part in raw.split(",") if part.strip()]
    if not solvers:
        return None

    valid = {"ipm", "ripopt", "ipopt", "highs", "amp"}
    invalid = [solver for solver in solvers if solver not in valid]
    if invalid:
        print(f"ERROR: Unknown solver(s): {', '.join(invalid)}. Valid: {', '.join(sorted(valid))}")
        sys.exit(1)
    return list(dict.fromkeys(solvers))


def _run_category(
    category: str,
    level: str,
    time_limit: float,
    output_dir: str | None,
    generate_report: bool,
    generate_html: bool,
    hard_timeout_grace: float | None,
    solver_filter: list[str] | None,
):
    """Run benchmarks for a single category."""
    from category_runner import CategoryBenchmarkRunner

    runner = CategoryBenchmarkRunner(
        category=category,
        level=level,
        time_limit=time_limit,
        hard_timeout_grace=hard_timeout_grace,
        solver_filter=solver_filter,
    )
    results = runner.run()

    # Determine output directory
    out_dir = Path(output_dir) if output_dir else Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON results
    ts = results.timestamp.replace(":", "-")
    json_path = out_dir / f"{category}_{level}_{ts}.json"
    results.save(json_path)
    print(f"\nResults saved to: {json_path}")

    known_optima = runner.get_known_optima()

    # Generate markdown report
    if generate_report:
        try:
            from utils.reporting import generate_category_report

            report_path = out_dir / f"{category}_{level}_{ts}.md"
            generate_category_report(
                results,
                category=category,
                level=level,
                known_optima=known_optima,
                output_path=report_path,
            )
        except Exception as e:
            print(f"Report generation failed: {e}")

    # Generate HTML dashboard
    if generate_html:
        try:
            from report_html import generate_html_dashboard

            html_path = out_dir / f"{category}_{level}_{ts}.html"
            generate_html_dashboard(
                results,
                category=category,
                level=level,
                known_optima=known_optima,
                output_path=html_path,
            )
        except Exception as e:
            print(f"HTML dashboard generation failed: {e}")

    return results


def _print_combined_summary(
    all_results: list[tuple[str, object]],
    level: str,
):
    """Print combined summary across all categories."""
    from benchmarks.metrics import (
        incorrect_count,
        shifted_geometric_mean,
        solved_count,
    )

    print(f"\n{'=' * 70}")
    print(f"Combined Summary — All Categories ({level})")
    print(f"{'=' * 70}")

    print(f"{'Category':<12s} {'Solver':<18s} {'Solved':>8s} {'Incorrect':>10s} {'SGM(s)':>10s}")
    print("-" * 62)

    total_incorrect = 0
    for cat, results in all_results:
        for solver in sorted(results.get_solvers()):
            solver_results = results.get_results(solver)
            n_inst = len({r.instance for r in solver_results})
            n_solved = solved_count(solver_results)

            # Build known optima from instance_info
            known = {}
            for name, info in results.instance_info.items():
                if info.best_known_objective is not None:
                    known[name] = info.best_known_objective

            n_inc = incorrect_count(solver_results, known)
            total_incorrect += n_inc

            times = [r.wall_time for r in solver_results if r.is_solved]
            sgm = shifted_geometric_mean(times)
            sgm_str = f"{sgm:.3f}" if sgm < 1e6 else "inf"

            print(
                f"{cat:<12s} {solver:<18s} {n_solved:>3d}/{n_inst:<4d} {n_inc:>10d} {sgm_str:>10s}"
            )

    print("-" * 62)
    if total_incorrect == 0:
        print("All results correct (0 incorrect total)")
    else:
        print(f"WARNING: {total_incorrect} incorrect result(s)!")


if __name__ == "__main__":
    main()
