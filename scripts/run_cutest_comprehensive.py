#!/usr/bin/env python
"""
Comprehensive CUTEst benchmark: discopt_ipopt vs discopt_ripopt.

Runs all available CUTEst problems (filtered by size) through both solvers
and prints a summary comparison table.

Usage:
    export CUTEST=/tmp/cutest_install/local
    export SIFDECODE=/tmp/cutest_install/local
    export MASTSIF=/tmp/cutest_install/sif
    export PYCUTEST_CACHE=/tmp/cutest_cache
    python scripts/run_cutest_comprehensive.py
"""

from __future__ import annotations

import os
import signal
import sys
import time

import numpy as np


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Wall time limit exceeded")


# Set JAX to CPU + float64
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "discopt_benchmarks"))


def solve_with_discopt_ipopt(prob, evaluator):
    """Solve using discopt's Ipopt backend."""
    from discopt.solvers.nlp_ipopt import solve_nlp

    constraint_bounds = None
    if prob.m > 0:
        constraint_bounds = list(zip(prob.cl.tolist(), prob.cu.tolist(), strict=False))

    return solve_nlp(
        evaluator,
        prob.x0,
        constraint_bounds=constraint_bounds,
        options={"print_level": 0, "max_iter": 3000, "tol": 1e-7},
    )


def solve_with_ripopt(prob, evaluator):
    """Solve using discopt's ripopt (Rust IPM) backend."""
    from discopt.solvers.nlp_ripopt import solve_nlp

    constraint_bounds = None
    if prob.m > 0:
        constraint_bounds = list(zip(prob.cl.tolist(), prob.cu.tolist(), strict=False))

    # Use same settings as Ipopt for fair comparison
    return solve_nlp(
        evaluator,
        prob.x0,
        constraint_bounds=constraint_bounds,
        options={"print_level": 0, "max_iter": 3000, "tol": 1e-7},
    )


# Wall-time limit per solver per problem (seconds)
WALL_TIME_LIMIT = 60.0


def run_benchmark(problem_names, label="benchmark"):
    """Run both solvers on a list of CUTEst problems."""
    from discopt.interfaces.cutest import load_cutest_problem

    results = []
    total = len(problem_names)

    print(f"\n{'=' * 90}")
    print(f"  CUTEst Benchmark: {label} ({total} problems)")
    print("  Solvers: discopt_ipopt, discopt_ripopt")
    print(f"{'=' * 90}")
    print(
        f"  {'Problem':<20s} {'n':>4s} {'m':>4s} │ "
        f"{'Ipopt Status':<12s} {'Time':>8s} {'Obj':>14s} │ "
        f"{'Ripopt Status':<12s} {'Time':>8s} {'Obj':>14s}"
    )
    sep = (
        f"  {'─' * 20} {'─' * 4} {'─' * 4} ┼ "
        f"{'─' * 12} {'─' * 8} {'─' * 14} ┼ "
        f"{'─' * 12} {'─' * 8} {'─' * 14}"
    )
    print(sep)

    for i, name in enumerate(problem_names, 1):
        row = {"name": name, "n": 0, "m": 0}

        try:
            prob = load_cutest_problem(name)
            evaluator = prob.to_evaluator()
            row["n"] = prob.n
            row["m"] = prob.m
        except Exception:
            row["ipopt_status"] = "LOAD_ERR"
            row["ripopt_status"] = "LOAD_ERR"
            row["ipopt_time"] = float("inf")
            row["ripopt_time"] = float("inf")
            row["ipopt_obj"] = None
            row["ripopt_obj"] = None
            results.append(row)
            print(
                f"  {name:<20s} {'?':>4s} {'?':>4s} │ "
                f"{'LOAD_ERR':<12s} {'--':>8s} {'--':>14s} │ "
                f"{'LOAD_ERR':<12s} {'--':>8s} {'--':>14s}"
                f"  [{i}/{total}]"
            )
            continue

        # --- Ipopt ---
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(int(WALL_TIME_LIMIT))
            t0 = time.perf_counter()
            r_ipopt = solve_with_discopt_ipopt(prob, evaluator)
            t_ipopt = time.perf_counter() - t0
            signal.alarm(0)
            row["ipopt_status"] = r_ipopt.status.value
            row["ipopt_time"] = t_ipopt
            row["ipopt_obj"] = r_ipopt.objective
        except TimeoutError:
            signal.alarm(0)
            row["ipopt_status"] = "time_limit"
            row["ipopt_time"] = WALL_TIME_LIMIT
            row["ipopt_obj"] = None
        except Exception:
            signal.alarm(0)
            row["ipopt_status"] = "ERROR"
            row["ipopt_time"] = float("inf")
            row["ipopt_obj"] = None

        # --- Ripopt ---
        # Note: SIGALRM can't be used with ripopt (Rust panics on signal
        # interrupting a callback). Use max_iter to limit runtime instead.
        try:
            t0 = time.perf_counter()
            r_ripopt = solve_with_ripopt(prob, evaluator)
            t_ripopt = time.perf_counter() - t0
            row["ripopt_status"] = r_ripopt.status.value
            row["ripopt_time"] = t_ripopt
            row["ripopt_obj"] = r_ripopt.objective
        except Exception:
            row["ripopt_status"] = "ERROR"
            row["ripopt_time"] = float("inf")
            row["ripopt_obj"] = None

        prob.close()
        results.append(row)

        # Print row
        ipopt_t = f"{row['ipopt_time']:.3f}s" if row["ipopt_time"] < 999 else "TL"
        ripopt_t = f"{row['ripopt_time']:.3f}s" if row["ripopt_time"] < 999 else "TL"
        ipopt_o = f"{row['ipopt_obj']:.6e}" if row["ipopt_obj"] is not None else "--"
        ripopt_o = f"{row['ripopt_obj']:.6e}" if row["ripopt_obj"] is not None else "--"

        print(
            f"  {name:<20s} {row['n']:>4d} {row['m']:>4d} │ "
            f"{row['ipopt_status']:<12s} {ipopt_t:>8s} {ipopt_o:>14s} │ "
            f"{row['ripopt_status']:<12s} {ripopt_t:>8s} {ripopt_o:>14s}  [{i}/{total}]"
        )

    return results


def print_summary(results, label=""):
    """Print summary statistics."""
    ipopt_solved = sum(1 for r in results if r.get("ipopt_status") == "optimal")
    ripopt_solved = sum(1 for r in results if r.get("ripopt_status") == "optimal")
    ipopt_errors = sum(1 for r in results if r.get("ipopt_status") in ("ERROR", "LOAD_ERR"))
    ripopt_errors = sum(1 for r in results if r.get("ripopt_status") in ("ERROR", "LOAD_ERR"))
    total = len(results)

    # Compute times for commonly-solved problems
    common_solved = [
        r
        for r in results
        if r.get("ipopt_status") == "optimal" and r.get("ripopt_status") == "optimal"
    ]

    print(f"\n{'=' * 70}")
    print(f"  Summary: {label}")
    print(f"{'=' * 70}")
    print(f"  Total problems:      {total}")
    print(
        f"  Ipopt  solved:       {ipopt_solved}/{total} ({100 * ipopt_solved / max(total, 1):.1f}%)"
    )
    ripopt_pct = 100 * ripopt_solved / max(total, 1)
    print(f"  Ripopt solved:       {ripopt_solved}/{total} ({ripopt_pct:.1f}%)")
    print(f"  Ipopt  errors:       {ipopt_errors}")
    print(f"  Ripopt errors:       {ripopt_errors}")
    print(f"  Both solved:         {len(common_solved)}")

    if common_solved:
        ipopt_times = [r["ipopt_time"] for r in common_solved]
        ripopt_times = [r["ripopt_time"] for r in common_solved]

        import math

        def sgm(times, shift=1.0):
            log_sum = sum(math.log(t + shift) for t in times)
            return math.exp(log_sum / len(times)) - shift

        sgm_ipopt = sgm(ipopt_times)
        sgm_ripopt = sgm(ripopt_times)

        print(f"\n  Timing (commonly-solved, n={len(common_solved)}):")
        print(f"    Ipopt  mean:       {np.mean(ipopt_times):.4f}s")
        print(f"    Ripopt mean:       {np.mean(ripopt_times):.4f}s")
        print(f"    Ipopt  median:     {np.median(ipopt_times):.4f}s")
        print(f"    Ripopt median:     {np.median(ripopt_times):.4f}s")
        print(f"    Ipopt  SGM:        {sgm_ipopt:.4f}s")
        print(f"    Ripopt SGM:        {sgm_ripopt:.4f}s")
        print(f"    SGM ratio (R/I):   {sgm_ripopt / max(sgm_ipopt, 1e-10):.3f}x")

        # Count wins
        ipopt_faster = sum(1 for r in common_solved if r["ipopt_time"] < r["ripopt_time"])
        ripopt_faster = len(common_solved) - ipopt_faster
        print(f"\n    Ipopt faster:      {ipopt_faster}")
        print(f"    Ripopt faster:     {ripopt_faster}")

        # Check objective agreement
        disagree = 0
        for r in common_solved:
            if r["ipopt_obj"] is not None and r["ripopt_obj"] is not None:
                diff = abs(r["ipopt_obj"] - r["ripopt_obj"])
                tol = 1e-4 + 1e-3 * max(abs(r["ipopt_obj"]), abs(r["ripopt_obj"]))
                if diff > tol:
                    disagree += 1
                    print(
                        f"    DISAGREE: {r['name']} ipopt={r['ipopt_obj']:.8e} "
                        f"ripopt={r['ripopt_obj']:.8e} diff={diff:.2e}"
                    )
        print(f"\n    Objective agreement: {len(common_solved) - disagree}/{len(common_solved)}")

    # Problems solved by only one solver
    only_ipopt = [
        r
        for r in results
        if r.get("ipopt_status") == "optimal" and r.get("ripopt_status") != "optimal"
    ]
    only_ripopt = [
        r
        for r in results
        if r.get("ripopt_status") == "optimal" and r.get("ipopt_status") != "optimal"
    ]
    if only_ipopt:
        print(f"\n  Solved by Ipopt only ({len(only_ipopt)}):")
        for r in only_ipopt[:20]:
            print(f"    {r['name']} (n={r['n']}, m={r['m']}, ripopt={r.get('ripopt_status')})")
    if only_ripopt:
        print(f"\n  Solved by Ripopt only ({len(only_ripopt)}):")
        for r in only_ripopt[:20]:
            print(f"    {r['name']} (n={r['n']}, m={r['m']}, ipopt={r.get('ipopt_status')})")

    print(f"\n{'=' * 70}")


def discover_problems(max_n=100, max_m=None):
    """Discover CUTEst problems matching size filters."""
    import pycutest

    constraint_types = ["unconstrained", "bounds", "linear", "quadratic", "other"]
    all_names = set()

    for ct in constraint_types:
        try:
            names = pycutest.find_problems(constraints=ct)
            for name in names:
                try:
                    props = pycutest.problem_properties(name)
                    n = props.get("n", 0)
                    m = props.get("m", 0)
                    if max_n is not None and n > max_n:
                        continue
                    if max_m is not None and m > max_m:
                        continue
                    all_names.add(name)
                except Exception:
                    pass
        except Exception:
            pass

    return sorted(all_names)


if __name__ == "__main__":
    # Force unbuffered output
    sys.stdout.reconfigure(line_buffering=True)

    import argparse

    parser = argparse.ArgumentParser(description="CUTEst comprehensive benchmark")
    parser.add_argument("--max-n", type=int, default=100, help="Max variables (default: 100)")
    parser.add_argument(
        "--max-m", type=int, default=None, help="Max constraints (default: no limit)"
    )
    parser.add_argument("--smoke", action="store_true", help="Run smoke test only (10 problems)")
    parser.add_argument("--problems", nargs="*", help="Specific problem names to run")
    args = parser.parse_args()

    if args.smoke:
        problem_names = [
            "ROSENBR",
            "BEALE",
            "BROWNAL",
            "DENSCHNA",
            "HILBERTA",
            "HS35",
            "HS71",
            "HS100",
            "HS106",
            "PENALTY1",
        ]
        label = "Smoke Test"
    elif args.problems:
        problem_names = args.problems
        label = "Custom"
    else:
        print(f"Discovering CUTEst problems with n <= {args.max_n}...")
        problem_names = discover_problems(max_n=args.max_n, max_m=args.max_m)
        label = f"Comprehensive (n <= {args.max_n})"

    print(f"Found {len(problem_names)} problems")
    results = run_benchmark(problem_names, label=label)
    print_summary(results, label=label)
