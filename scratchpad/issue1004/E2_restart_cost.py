"""#1004 E2 — is a *re*-start on an already-built fixed-integer sub-NLP cheaper?

This is the issue's one escape hatch. Its budget argument (spreading one start over
many configurations weakly dominates concentrating k starts on few) assumes every
start costs the same; it flips sign only if starts 2..k on an already-built
sub-problem are materially cheaper than start 1.

Protocol, per model and per known-feasible configuration:

* pin the integers to the configuration,
* run ``--starts`` sub-NLP solves from *different* starts back to back,
* record each solve's wall time individually,
* report ``t1`` vs ``mean(t2..tk)`` with a standard deviation (CLAUDE.md §9),
* and repeat the whole thing ``--reps`` times, **interleaving** configurations
  rather than running one configuration's k starts in a burst, so a drifting
  machine cannot masquerade as a first-solve penalty.

An "A/B interleave" here means alternating which position in the sequence is being
timed; the control for "start 1 is special" is a *repeat* sequence where the same
sub-problem is solved again from the same start (``--same-start``), which isolates
any caching in the evaluator or the backend from genuine start-dependence.

CLAUDE.md §6: the executed-solve counter is printed and a zero count exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "discopt_benchmarks"))

from discopt._relax.primal_heuristics import (  # noqa: E402
    _generate_starts,
    _get_integer_mask,
    _get_variable_bounds,
    _scan_one_hot_rows,
    cached_evaluator,
    one_hot_config_dive,
    subnlp,
)
from gdp_loader import load_gdplib  # noqa: E402

MODELS = ["small_batch", "cstr", "spectralog", "batch_processing", "syngas", "gdp_col"]


def _backend():
    from discopt.solvers.nlp_backend import get_nlp_solver

    return get_nlp_solver("auto")


def _relaxation_point(evaluator, backend, lb, ub):
    x0 = 0.5 * (np.clip(lb, -1e6, 1e6) + np.clip(ub, -1e6, 1e6))
    res = backend(evaluator, x0, options={"print_level": 0, "max_iter": 500})
    x = res.x
    return (
        np.clip(np.asarray(x, dtype=np.float64), lb, ub) if x is not None else np.clip(x0, lb, ub)
    )


def probe_model(name, *, dive_seconds, starts, reps, n_configs, seed, same_start):
    model = load_gdplib(name)
    evaluator = cached_evaluator(model)
    backend = _backend()
    int_mask = _get_integer_mask(model)
    lb, ub = _get_variable_bounds(model)
    int_idx = np.nonzero(int_mask)[0].tolist()
    groups = _scan_one_hot_rows(model, int_mask, int(int_mask.size))
    rng = np.random.default_rng(seed)
    x_relax = _relaxation_point(evaluator, backend, lb, ub)

    dive_pts = one_hot_config_dive(
        model,
        x_relax,
        backend=backend,
        evaluator=evaluator,
        deadline=time.perf_counter() + dive_seconds,
    )
    configs: list[tuple[int, ...]] = []
    seen = set()
    for x, _ in dive_pts:
        key = tuple(int(round(float(x[j]))) for j in int_idx)
        if key not in seen:
            seen.add(key)
            configs.append(key)
        if len(configs) >= n_configs:
            break
    if not configs:
        print(f"[{name}] no feasible configuration found in {dive_seconds}s — skipped", flush=True)
        return None

    cont = ~int_mask
    zero_template = x_relax.copy()
    zero_template[cont] = np.clip(0.0, lb, ub)[cont]

    # Pre-draw the start pool once so the timing loop does no sampling work.
    pool = _generate_starts(lb, ub, starts * max(reps, 1), rng)

    first: list[float] = []
    later: list[float] = []
    solves = 0
    # Interleaved: rep-major, so every configuration contributes to every rep and a
    # machine that drifts over the run drifts across both arms equally.
    for rep in range(reps):
        for ci, cfg in enumerate(configs):
            for k in range(starts):
                if k == 0 or same_start:
                    tmpl = zero_template
                else:
                    tmpl = pool[(rep * starts + k) % len(pool)]
                x0 = np.array(tmpl, dtype=np.float64, copy=True)
                for j, v in zip(int_idx, cfg):
                    x0[j] = float(v)
                x0 = np.clip(x0, lb, ub)
                t0 = time.perf_counter()
                subnlp(model, x0, backend=backend, evaluator=evaluator)
                dt = time.perf_counter() - t0
                solves += 1
                (first if k == 0 else later).append(dt)
            _ = ci

    def stat(xs):
        if not xs:
            return (float("nan"), float("nan"))
        return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)

    m1, s1 = stat(first)
    mk, sk = stat(later)
    ratio = mk / m1 if m1 > 0 else float("nan")
    rec = {
        "model": name,
        "configs": len(configs),
        "solves": solves,
        "same_start": same_start,
        "t_first_mean": m1,
        "t_first_sd": s1,
        "t_first_n": len(first),
        "t_later_mean": mk,
        "t_later_sd": sk,
        "t_later_n": len(later),
        "ratio_later_over_first": ratio,
    }
    print(
        f"[{name}] {len(configs)} config(s), {solves} solves | "
        f"start 1: {m1 * 1000:.1f} ± {s1 * 1000:.1f} ms (n={len(first)}) | "
        f"starts 2..{starts}: {mk * 1000:.1f} ± {sk * 1000:.1f} ms (n={len(later)}) | "
        f"ratio {ratio:.3f}",
        flush=True,
    )
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--dive-seconds", type=float, default=30.0)
    ap.add_argument("--starts", type=int, default=5)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--configs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--same-start",
        action="store_true",
        help="control: repeat the SAME start, isolating caching from start-dependence",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import os

    print(f"load before run: {os.getloadavg()}", flush=True)
    records = [
        r
        for r in (
            probe_model(
                name,
                dive_seconds=args.dive_seconds,
                starts=args.starts,
                reps=args.reps,
                n_configs=args.configs,
                seed=args.seed,
                same_start=args.same_start,
            )
            for name in args.models
        )
        if r is not None
    ]
    print(f"load after run: {os.getloadavg()}", flush=True)

    total = sum(r["solves"] for r in records)
    print("\n" + "=" * 84)
    print(
        f"{'model':20s} {'solves':>7s} {'start1 (ms)':>16s} {'starts2..k (ms)':>18s} {'ratio':>8s}"
    )
    print("-" * 84)
    for r in records:
        print(
            f"{r['model']:20s} {r['solves']:>7d} "
            f"{r['t_first_mean'] * 1000:>9.1f} ± {r['t_first_sd'] * 1000:<5.1f} "
            f"{r['t_later_mean'] * 1000:>11.1f} ± {r['t_later_sd'] * 1000:<5.1f} "
            f"{r['ratio_later_over_first']:>8.3f}"
        )
    print("-" * 84)
    print(f"TOTAL executed sub-NLP solves: {total}")
    print("=" * 84)

    if args.out:
        Path(args.out).write_text(json.dumps(records, indent=2))
        print(f"wrote {args.out}")
    if total == 0:
        print("PROBE VACUOUS: zero sub-NLP solves executed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
