"""#1004 E1 — per-model detection rate of the constructor's feasibility test.

The issue reports that with the integers pinned to a *known-feasible* configuration
on ``syngas``, only 12 of 67 starts (B1) and 2 of 6 (#993 C2) produced a feasible
point, and concludes the constructor "rejects ~80% of genuinely feasible
configurations". Both measurements are over a *start family*. The constructor does
not draw from a start family. At a fixed configuration its start is

    zero_start = clip(0, lb, ub) on every continuous slot, the configuration on
    the integer slots

which is fully determined by the model and the configuration — ``x_relax``
contributes nothing but the integer slots, and the plan overwrites those. So the
number that decides the issue is not "detection over random starts", it is "does
the constructor's own deterministic start detect a feasible configuration".

Design (the bias trap this probe had to be rebuilt around): ``one_hot_config_dive``
accepts a configuration by trying **the zero-continuous start first**
(``primal_heuristics.py``, the dive's completion block), so a witness set drawn
from the dive is enriched with exactly the configurations arm Z can already solve.
Instead:

* build a **configuration pool** from three sources — the dive's returned
  configurations (biased toward Z, reported separately and never pooled with the
  rest), uniformly sampled valid configurations, and one-group-flip neighbours of
  both (unbiased with respect to every arm);
* on each pooled configuration run every arm — **Z** the constructor's exact
  zero-continuous start, **X** the relaxation point's continuous slots, **R**
  ``--random-starts`` stratified random starts;
* call a configuration *genuinely feasible* when **any** arm returned a
  constraint-verified point. The oracle is the union, so it is symmetric in the
  arms and no arm is scored on a set it selected.

CLAUDE.md §6: every arm keeps an executed-test counter; the script prints the total
and exits non-zero when it is zero. Exceptions are not swallowed (§7).
"""

from __future__ import annotations

import argparse
import json
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

MODELS = [
    "jobshop",
    "ex1_linan_2023",
    "positioning",
    "small_batch",
    "cstr",
    "spectralog",
    "methanol",
    "batch_processing",
    "syngas",
    "water_network",
    "gdp_col",
    "modprodnet",
]


def _backend():
    from discopt.solvers.nlp_backend import get_nlp_solver

    return get_nlp_solver("auto")


def _relaxation_point(evaluator, backend, lb, ub):
    """A root relaxation point: one NLP solve over the box with integrality dropped."""
    x0 = 0.5 * (np.clip(lb, -1e6, 1e6) + np.clip(ub, -1e6, 1e6))
    res = backend(evaluator, x0, options={"print_level": 0, "max_iter": 500})
    x = res.x
    return (
        np.clip(np.asarray(x, dtype=np.float64), lb, ub) if x is not None else np.clip(x0, lb, ub)
    )


def _seed_from(config, int_idx, cont_template, lb, ub):
    seed = np.array(cont_template, dtype=np.float64, copy=True)
    for j, v in zip(int_idx, config):
        seed[j] = float(v)
    return np.clip(seed, lb, ub)


def _random_valid_config(groups, residual, rng, int_idx):
    """A configuration satisfying every ``sum_k y_k == 1`` row by construction."""
    bits = dict.fromkeys(int_idx, 0)
    for g in groups:
        bits[g[int(rng.integers(len(g)))]] = 1
    for j in residual:
        bits[j] = int(rng.integers(2))
    return tuple(bits[j] for j in int_idx)


def _flip_neighbour(config, groups, int_idx, rng):
    """One group re-pointed to a different disjunct: still a valid configuration."""
    pos = {j: i for i, j in enumerate(int_idx)}
    flippable = [g for g in groups if len(g) >= 2]
    if not flippable:
        return None
    g = flippable[int(rng.integers(len(flippable)))]
    bits = list(config)
    for j in g:
        bits[pos[j]] = 0
    bits[pos[g[int(rng.integers(len(g)))]]] = 1
    out = tuple(bits)
    return None if out == config else out


def probe_model(name, *, dive_seconds, random_starts, pool_random, pool_neighbours, seed):
    t_load = time.time()
    model = load_gdplib(name)
    evaluator = cached_evaluator(model)
    backend = _backend()
    int_mask = _get_integer_mask(model)
    lb, ub = _get_variable_bounds(model)
    int_idx = np.nonzero(int_mask)[0].tolist()
    groups = _scan_one_hot_rows(model, int_mask, int(int_mask.size))
    covered = {j for g in groups for j in g}
    residual = [j for j in int_idx if j not in covered and lb[j] >= -1e-9 and ub[j] <= 1.0 + 1e-9]
    rng = np.random.default_rng(seed)

    x_relax = _relaxation_point(evaluator, backend, lb, ub)
    cont = ~int_mask
    zero_template = x_relax.copy()
    zero_template[cont] = np.clip(0.0, lb, ub)[cont]
    relax_template = x_relax.copy()

    print(
        f"[{name}] loaded in {time.time() - t_load:.1f}s: {int(int_mask.size)} vars, "
        f"{len(int_idx)} int, {len(groups)} groups, {len(residual)} residual binaries",
        flush=True,
    )

    # ── Configuration pool ───────────────────────────────────────────────────
    pool: dict[tuple[int, ...], str] = {}
    t0 = time.time()
    dive_pts = one_hot_config_dive(
        model,
        x_relax,
        backend=backend,
        evaluator=evaluator,
        deadline=time.perf_counter() + dive_seconds,
    )
    t_dive = time.time() - t0
    for x, _obj in dive_pts:
        pool.setdefault(tuple(int(round(float(x[j]))) for j in int_idx), "dive")
    for _ in range(pool_random):
        pool.setdefault(_random_valid_config(groups, residual, rng, int_idx), "random")
    bases = list(pool)
    for _ in range(pool_neighbours):
        if not bases:
            break
        base = bases[int(rng.integers(len(bases)))]
        nb = _flip_neighbour(base, groups, int_idx, rng)
        if nb is not None:
            pool.setdefault(nb, "neighbour")

    print(
        f"[{name}] pool {len(pool)} configs "
        f"(dive {t_dive:.1f}s -> {len(dive_pts)} pts / "
        f"{sum(1 for s in pool.values() if s == 'dive')} distinct, "
        f"{sum(1 for s in pool.values() if s == 'random')} random, "
        f"{sum(1 for s in pool.values() if s == 'neighbour')} neighbour)",
        flush=True,
    )

    # ── Arms, run on every pooled configuration ──────────────────────────────
    tests = 0
    rows = []
    for cfg, source in pool.items():
        hit_z = (
            subnlp(
                model,
                _seed_from(cfg, int_idx, zero_template, lb, ub),
                backend=backend,
                evaluator=evaluator,
            )
            is not None
        )
        tests += 1
        hit_x = (
            subnlp(
                model,
                _seed_from(cfg, int_idx, relax_template, lb, ub),
                backend=backend,
                evaluator=evaluator,
            )
            is not None
        )
        tests += 1
        rand_hits = 0
        if random_starts > 0:
            starts = _generate_starts(lb, ub, random_starts, rng)
            for i in range(random_starts):
                if (
                    subnlp(
                        model,
                        _seed_from(cfg, int_idx, starts[i], lb, ub),
                        backend=backend,
                        evaluator=evaluator,
                    )
                    is not None
                ):
                    rand_hits += 1
                tests += 1
        rows.append(
            {
                "source": source,
                "zero": bool(hit_z),
                "relax": bool(hit_x),
                "random_hits": rand_hits,
                "random_tests": random_starts,
                "feasible": bool(hit_z or hit_x or rand_hits),
            }
        )

    def summarise(subset):
        feas = [r for r in subset if r["feasible"]]
        # Feasibility oracle that never consults arm Z: whether some *other* start
        # produced a verified point. Scoring Z on this subset removes the "Z picked
        # its own test set" circularity for the oracle (pool membership is handled
        # by keeping the dive-derived configs in their own table).
        nonz = [r for r in subset if r["random_hits"] or r["relax"]]
        return {
            "configs": len(subset),
            "feasible": len(feas),
            "zero_hits": sum(1 for r in feas if r["zero"]),
            "relax_hits": sum(1 for r in feas if r["relax"]),
            "random_hits": sum(r["random_hits"] for r in feas),
            "random_tests": sum(r["random_tests"] for r in feas),
            "nonz_feasible": len(nonz),
            "zero_hits_on_nonz": sum(1 for r in nonz if r["zero"]),
            "z_only": sum(1 for r in feas if r["zero"] and not (r["random_hits"] or r["relax"])),
            "z_missed": sum(1 for r in feas if not r["zero"]),
        }

    unbiased = [r for r in rows if r["source"] != "dive"]
    rec = {
        "model": name,
        "n_vars": int(int_mask.size),
        "n_int": len(int_idx),
        "n_groups": len(groups),
        "n_residual": len(residual),
        "dive_points": len(dive_pts),
        "dive_seconds": round(t_dive, 2),
        "tests": tests,
        "all": summarise(rows),
        "unbiased": summarise(unbiased),
        "dive_pool": summarise([r for r in rows if r["source"] == "dive"]),
    }
    u = rec["unbiased"]
    print(
        f"[{name}] unbiased pool: {u['feasible']}/{u['configs']} feasible | "
        f"zero {u['zero_hits']}/{u['feasible']} | relax {u['relax_hits']}/{u['feasible']} | "
        f"random {u['random_hits']}/{u['random_tests']} | {tests} tests",
        flush=True,
    )
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--dive-seconds", type=float, default=30.0)
    ap.add_argument("--random-starts", type=int, default=8)
    ap.add_argument("--pool-random", type=int, default=20)
    ap.add_argument("--pool-neighbours", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    records = []
    for name in args.models:
        records.append(
            probe_model(
                name,
                dive_seconds=args.dive_seconds,
                random_starts=args.random_starts,
                pool_random=args.pool_random,
                pool_neighbours=args.pool_neighbours,
                seed=args.seed,
            )
        )
        # Write after every model: a crash on model 12 must not throw away 11
        # models of measurement (this probe lost a full panel to a RecursionError
        # in the .nl expression compiler on ``modprodnet``).
        if args.out:
            Path(args.out).write_text(json.dumps(records, indent=2))

    def pct(h, t):
        return f"{100.0 * h / t:.0f}%" if t else "n/a"

    for label, key in (
        ("UNBIASED POOL (random + neighbour configs)", "unbiased"),
        ("DIVE-DERIVED POOL (biased toward the zero start)", "dive_pool"),
    ):
        print("\n" + "=" * 86)
        print(label)
        print(
            f"{'model':20s} {'cfgs':>5s} {'feas':>5s} {'zero':>12s} {'relax':>12s} {'random':>16s}"
        )
        print("-" * 86)
        for r in records:
            s = r[key]
            z = f"{s['zero_hits']}/{s['feasible']} {pct(s['zero_hits'], s['feasible'])}"
            x = f"{s['relax_hits']}/{s['feasible']} {pct(s['relax_hits'], s['feasible'])}"
            rr = (
                f"{s['random_hits']}/{s['random_tests']} {pct(s['random_hits'], s['random_tests'])}"
            )
            print(
                f"{r['model']:20s} {s['configs']:>5d} {s['feasible']:>5d} {z:>12s} {x:>12s} {rr:>16s}"
            )
        tz = sum(r[key]["zero_hits"] for r in records)
        tf = sum(r[key]["feasible"] for r in records)
        tx = sum(r[key]["relax_hits"] for r in records)
        rh = sum(r[key]["random_hits"] for r in records)
        rt = sum(r[key]["random_tests"] for r in records)
        nz = sum(r[key]["nonz_feasible"] for r in records)
        nzz = sum(r[key]["zero_hits_on_nonz"] for r in records)
        print("-" * 86)
        print(
            f"{'TOTAL':20s} {sum(r[key]['configs'] for r in records):>5d} {tf:>5d} "
            f"{tz}/{tf} {pct(tz, tf):>5s}   {tx}/{tf} {pct(tx, tf):>5s}   "
            f"{rh}/{rt} {pct(rh, rt):>5s}"
        )
        print(
            f"  zero-start detection on configurations proven feasible WITHOUT it: "
            f"{nzz}/{nz} {pct(nzz, nz)}  |  "
            f"found only by the zero start: {sum(r[key]['z_only'] for r in records)}  |  "
            f"missed by the zero start: {sum(r[key]['z_missed'] for r in records)}"
        )
    total = sum(r["tests"] for r in records)
    print("=" * 86)
    print(f"TOTAL executed feasibility tests: {total}")

    if args.out:
        Path(args.out).write_text(json.dumps(records, indent=2))
        print(f"wrote {args.out}")
    if total == 0:
        print("PROBE VACUOUS: zero feasibility tests executed", file=sys.stderr)
        return 1
    return 0


def _run_with_deep_stack():
    """Run ``main`` on a thread with a large stack and recursion limit.

    ``cached_evaluator`` lowers the .nl expression DAG recursively, and
    ``modprodnet`` (488 vars, one long product chain) blows the default 1000-frame
    limit inside ``_nl_expr_compiler._lower``. Raising the limit alone segfaults —
    the C stack goes first — so the thread also gets a 512 MB stack. This is a
    *probe-side* workaround for reaching the model, deliberately not a library
    change; the underlying depth limit is out of scope for #1004.
    """
    import threading

    sys.setrecursionlimit(200_000)
    threading.stack_size(512 * 1024 * 1024)
    rc: list[int] = []
    t = threading.Thread(target=lambda: rc.append(main()))
    t.start()
    t.join()
    return rc[0] if rc else 1


if __name__ == "__main__":
    raise SystemExit(_run_with_deep_stack())
