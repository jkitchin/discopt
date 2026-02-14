#!/usr/bin/env python
"""
CUTEst benchmark: JAX IPM (callback-based) vs cyipopt.

Compares discopt's pure-JAX interior point method (via ipm_callbacks.py)
against cyipopt on CUTEst problems with n <= 100.

Usage:
    export CUTEST=/tmp/cutest_install/local
    export SIFDECODE=/tmp/cutest_install/local
    export MASTSIF=/tmp/cutest_install/sif
    export PYCUTEST_CACHE=/tmp/cutest_cache
    python scripts/run_cutest_ipm_vs_ipopt.py --smoke
    python scripts/run_cutest_ipm_vs_ipopt.py --max-n 100
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

# Auto-detect CUTEst installation
_CUTEST_ENV = Path.home() / ".local" / "cutest" / "env.sh"
if "CUTEST" not in os.environ and _CUTEST_ENV.exists():
    for line in _CUTEST_ENV.read_text().splitlines():
        line = line.strip()
        if line.startswith("export ") and "=" in line:
            kv = line[len("export ") :]
            key, _, val = kv.partition("=")
            # Expand $VAR and ${VAR:-} references to already-set env vars
            val = val.strip('"')
            for evar in os.environ:
                val = val.replace(f"${evar}", os.environ[evar])
                val = val.replace(f"${{{evar}}}", os.environ[evar])
                val = val.replace(f"${{{evar}:-}}", os.environ.get(evar, ""))
            # Remove any remaining ${...:-} patterns (empty fallbacks)
            import re

            val = re.sub(r"\$\{[^}]*:-\}", "", val)
            val = re.sub(r"\$\{[^}]*\}", "", val)
            os.environ[key] = val

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

WALL_TIME_LIMIT = 60.0


def _solve_worker(solver_name, problem_name, queue):
    """Worker that runs a solver in a subprocess."""
    try:
        from discopt.interfaces.cutest import load_cutest_problem

        prob = load_cutest_problem(problem_name)
        evaluator = prob.to_evaluator()

        constraint_bounds = None
        if prob.m > 0:
            constraint_bounds = list(zip(prob.cl.tolist(), prob.cu.tolist(), strict=False))

        t0 = time.perf_counter()

        if solver_name == "ipopt":
            from discopt.solvers.nlp_ipopt import solve_nlp

            result = solve_nlp(
                evaluator,
                prob.x0,
                constraint_bounds=constraint_bounds,
                options={"print_level": 0, "max_iter": 3000, "tol": 1e-7},
            )
        else:
            from discopt._jax.ipm_callbacks import solve_nlp_ipm_callbacks

            result = solve_nlp_ipm_callbacks(
                evaluator,
                prob.x0,
                constraint_bounds=constraint_bounds,
                options={"max_iter": 3000, "tol": 1e-7, "acceptable_tol": 1e-6},
            )

        elapsed = time.perf_counter() - t0
        prob.close()
        queue.put(("ok", result.status.value, elapsed, result.objective, result.iterations))
    except Exception as e:
        queue.put(("error", str(e)))


def _run_solver_with_timeout(solver_name, problem_name, timeout):
    """Run a solver in a subprocess with a hard wall-time limit."""
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()
    proc = ctx.Process(target=_solve_worker, args=(solver_name, problem_name, queue))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        proc.join(timeout=5)
        return "time_limit", timeout, None, 0

    if not queue.empty():
        msg = queue.get_nowait()
        if msg[0] == "ok":
            _, status, elapsed, obj, iters = msg
            return status, elapsed, obj, iters
        else:
            return "ERROR", float("inf"), None, 0

    return "ERROR", float("inf"), None, 0


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
                    m_val = props.get("m", 0)
                    if max_n is not None and n > max_n:
                        continue
                    if max_m is not None and m_val > max_m:
                        continue
                    all_names.add(name)
                except Exception:
                    pass
        except Exception:
            pass

    return sorted(all_names)


def sgm(times, shift=1.0):
    """Shifted geometric mean."""
    log_sum = sum(math.log(t + shift) for t in times)
    return math.exp(log_sum / len(times)) - shift


def run_benchmark(problem_names, label="benchmark"):
    """Run both solvers on a list of CUTEst problems."""
    from discopt.interfaces.cutest import load_cutest_problem

    results = []
    total = len(problem_names)

    print(f"\n{'=' * 100}")
    print(f"  CUTEst Benchmark: {label} ({total} problems)")
    print("  Solvers: ipopt (cyipopt), ipm (JAX callback-based IPM)")
    print(f"{'=' * 100}")
    print(f"  {'Problem':<20s} {'n':>4s} {'m':>4s} │ {'Ipopt':^35s} │ {'JAX IPM':^35s}")
    print(
        f"  {'':20s} {'':4s} {'':4s} │ "
        f"{'Status':<12s} {'Time':>8s} {'Obj':>14s} │ "
        f"{'Status':<12s} {'Time':>8s} {'Obj':>14s}"
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
            row["n"] = prob.n
            row["m"] = prob.m
            prob.close()
        except Exception:
            row["ipopt_status"] = "LOAD_ERR"
            row["ipm_status"] = "LOAD_ERR"
            row["ipopt_time"] = float("inf")
            row["ipm_time"] = float("inf")
            row["ipopt_obj"] = None
            row["ipm_obj"] = None
            row["ipopt_iters"] = 0
            row["ipm_iters"] = 0
            results.append(row)
            print(
                f"  {name:<20s} {'?':>4s} {'?':>4s} │ "
                f"{'LOAD_ERR':<12s} {'--':>8s} {'--':>14s} │ "
                f"{'LOAD_ERR':<12s} {'--':>8s} {'--':>14s}"
                f"  [{i}/{total}]"
            )
            continue

        # --- Ipopt ---
        status, elapsed, obj, iters = _run_solver_with_timeout("ipopt", name, WALL_TIME_LIMIT)
        row["ipopt_status"] = status
        row["ipopt_time"] = elapsed
        row["ipopt_obj"] = obj
        row["ipopt_iters"] = iters

        # --- JAX IPM ---
        status, elapsed, obj, iters = _run_solver_with_timeout("ipm", name, WALL_TIME_LIMIT)
        row["ipm_status"] = status
        row["ipm_time"] = elapsed
        row["ipm_obj"] = obj
        row["ipm_iters"] = iters

        results.append(row)

        ipopt_t = f"{row['ipopt_time']:.3f}s" if row["ipopt_time"] < 999 else "TL"
        ipm_t = f"{row['ipm_time']:.3f}s" if row["ipm_time"] < 999 else "TL"
        ipopt_o = f"{row['ipopt_obj']:.6e}" if row["ipopt_obj"] is not None else "--"
        ipm_o = f"{row['ipm_obj']:.6e}" if row["ipm_obj"] is not None else "--"

        print(
            f"  {name:<20s} {row['n']:>4d} {row['m']:>4d} │ "
            f"{row['ipopt_status']:<12s} {ipopt_t:>8s} {ipopt_o:>14s} │ "
            f"{row['ipm_status']:<12s} {ipm_t:>8s} {ipm_o:>14s}"
            f"  [{i}/{total}]"
        )

    return results


def print_summary(results, label=""):
    """Print summary statistics."""
    ipopt_solved = sum(1 for r in results if r.get("ipopt_status") == "optimal")
    ipm_solved = sum(1 for r in results if r.get("ipm_status") == "optimal")
    ipopt_errors = sum(1 for r in results if r.get("ipopt_status") in ("ERROR", "LOAD_ERR"))
    ipm_errors = sum(1 for r in results if r.get("ipm_status") in ("ERROR", "LOAD_ERR"))
    total = len(results)

    common_solved = [
        r
        for r in results
        if r.get("ipopt_status") == "optimal" and r.get("ipm_status") == "optimal"
    ]

    print(f"\n{'=' * 70}")
    print(f"  Summary: {label}")
    print(f"{'=' * 70}")
    print(f"  Total problems:      {total}")
    print(
        f"  Ipopt  solved:       {ipopt_solved}/{total} ({100 * ipopt_solved / max(total, 1):.1f}%)"
    )
    print(f"  JAX IPM solved:      {ipm_solved}/{total} ({100 * ipm_solved / max(total, 1):.1f}%)")
    print(f"  Ipopt  errors:       {ipopt_errors}")
    print(f"  JAX IPM errors:      {ipm_errors}")
    print(f"  Both solved:         {len(common_solved)}")

    if common_solved:
        ipopt_times = [r["ipopt_time"] for r in common_solved]
        ipm_times = [r["ipm_time"] for r in common_solved]

        sgm_ipopt = sgm(ipopt_times)
        sgm_ipm = sgm(ipm_times)

        print(f"\n  Timing (commonly-solved, n={len(common_solved)}):")
        print(f"    Ipopt   mean:      {np.mean(ipopt_times):.4f}s")
        print(f"    JAX IPM mean:      {np.mean(ipm_times):.4f}s")
        print(f"    Ipopt   median:    {np.median(ipopt_times):.4f}s")
        print(f"    JAX IPM median:    {np.median(ipm_times):.4f}s")
        print(f"    Ipopt   SGM:       {sgm_ipopt:.4f}s")
        print(f"    JAX IPM SGM:       {sgm_ipm:.4f}s")
        print(f"    SGM ratio (IPM/I): {sgm_ipm / max(sgm_ipopt, 1e-10):.3f}x")

        ipopt_faster = sum(1 for r in common_solved if r["ipopt_time"] < r["ipm_time"])
        ipm_faster = len(common_solved) - ipopt_faster
        print(f"\n    Ipopt faster:      {ipopt_faster}")
        print(f"    JAX IPM faster:    {ipm_faster}")

        # Objective agreement
        disagree = 0
        for r in common_solved:
            if r["ipopt_obj"] is not None and r["ipm_obj"] is not None:
                diff = abs(r["ipopt_obj"] - r["ipm_obj"])
                tol = max(1e-4, 1e-3 * max(abs(r["ipopt_obj"]), abs(r["ipm_obj"])))
                if diff > tol:
                    disagree += 1
                    print(
                        f"    DISAGREE: {r['name']} ipopt={r['ipopt_obj']:.8e} "
                        f"ipm={r['ipm_obj']:.8e} diff={diff:.2e}"
                    )
        print(f"\n    Objective agreement: {len(common_solved) - disagree}/{len(common_solved)}")

    # Problems solved by only one solver
    only_ipopt = [
        r
        for r in results
        if r.get("ipopt_status") == "optimal" and r.get("ipm_status") != "optimal"
    ]
    only_ipm = [
        r
        for r in results
        if r.get("ipm_status") == "optimal" and r.get("ipopt_status") != "optimal"
    ]
    if only_ipopt:
        print(f"\n  Solved by Ipopt only ({len(only_ipopt)}):")
        for r in only_ipopt[:20]:
            print(f"    {r['name']} (n={r['n']}, m={r['m']}, ipm={r.get('ipm_status')})")
    if only_ipm:
        print(f"\n  Solved by JAX IPM only ({len(only_ipm)}):")
        for r in only_ipm[:20]:
            print(f"    {r['name']} (n={r['n']}, m={r['m']}, ipopt={r.get('ipopt_status')})")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)

    import argparse

    parser = argparse.ArgumentParser(description="CUTEst benchmark: JAX IPM vs cyipopt")
    parser.add_argument("--max-n", type=int, default=100, help="Max variables (default: 100)")
    parser.add_argument(
        "--max-m", type=int, default=None, help="Max constraints (default: no limit)"
    )
    parser.add_argument("--smoke", action="store_true", help="Run smoke test only (10 problems)")
    parser.add_argument("--problems", nargs="*", help="Specific problem names")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON to this path")
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
        label = "Smoke Test (IPM vs Ipopt)"
    elif args.problems:
        problem_names = args.problems
        label = "Custom (IPM vs Ipopt)"
    else:
        print(f"Discovering CUTEst problems with n <= {args.max_n}...")
        problem_names = discover_problems(max_n=args.max_n, max_m=args.max_m)
        label = f"Comprehensive (n <= {args.max_n}, IPM vs Ipopt)"

    print(f"Found {len(problem_names)} problems")
    results = run_benchmark(problem_names, label=label)
    print_summary(results, label=label)

    if args.output:

        def _sanitize(v):
            if v is None:
                return None
            if isinstance(v, float) and (v == float("inf") or v != v):
                return None
            return v

        out = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "max_n": args.max_n,
            "max_m": args.max_m,
            "n_problems": len(results),
            "results": [{k: _sanitize(v) for k, v in r.items()} for r in results],
        }
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nResults saved to {args.output}")
