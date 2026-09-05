#!/usr/bin/env python
"""#1180 -- what one OBBT/node probe LP actually costs, and how much of it is native.

The §1.3 table this issue supersedes recorded the Rust LP as **3.4 %** of nvs05's
wall ("the node LP is nothing"). The post-tape layer split (deliverable 1) puts
``discopt._rust.solve_lp_warm_csc_py`` at roughly half of it. This probe answers
the follow-up that decides what to build: is that half *native simplex time* --
in which case the lever is the LP itself or the number of probes -- or is it
Python marshaling around the binding, the #764 finding on tanksize (4.25 ms
in-loop vs 1.28 ms pure-binding, i.e. ~70 % marshaling)?

Method: wrap the binding, count calls and accumulate in-loop wall during a real
solve, capture one representative argument tuple, then replay THAT call against
the raw binding with the arrays already built. The difference between in-loop
ms/call and replay ms/call is everything Python does per probe. The replay is
interleaved over rounds with a spread reported (CLAUDE.md §9).

Nothing is monkeypatched after the solve returns; the replay uses captured
arrays and cannot affect a bound.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

ASSERTS = {"n": 0}


def check(cond: bool, msg: str) -> None:
    ASSERTS["n"] += 1
    if not cond:
        raise AssertionError(msg)


def run(nl: str, time_limit: float, replay_reps: int, rounds: int) -> dict:
    import discopt._rust as R
    from discopt.modeling.core import from_nl

    orig = R.solve_lp_warm_csc_py
    stats = {"n": 0, "wall": 0.0, "by_m": {}, "warm_basis_given": 0, "captured": None}

    def wrapped(*a, **k):
        stats["n"] += 1
        m = int(a[1]) if len(a) > 1 and np.isscalar(a[1]) else -1
        # The warm basis is whichever argument comes in as an int array or None;
        # counted rather than assumed so "warm-started" is a measurement.
        if any(x is not None and isinstance(x, np.ndarray) and x.dtype.kind == "i" and
               x.size and x.size in (m, m + 1) for x in a):
            stats["warm_basis_given"] += 1
        t = time.perf_counter()
        try:
            return orig(*a, **k)
        finally:
            dt = time.perf_counter() - t
            stats["wall"] += dt
            b = stats["by_m"].setdefault(m, {"n": 0, "wall": 0.0})
            b["n"] += 1
            b["wall"] += dt
            if stats["captured"] is None and m > 0:
                stats["captured"] = (a, dict(k))

    R.solve_lp_warm_csc_py = wrapped
    try:
        model = from_nl(nl)
        t0 = time.perf_counter()
        result = model.solve(time_limit=time_limit, gap_tolerance=1e-4)
        wall = time.perf_counter() - t0
    finally:
        R.solve_lp_warm_csc_py = orig

    check(stats["n"] > 0, f"{nl}: the LP binding was never called -- probe measured nothing")
    check(stats["captured"] is not None, f"{nl}: no LP call was captured for replay")

    a, k = stats["captured"]
    shapes = [
        (i, type(x).__name__, getattr(x, "shape", None), getattr(x, "dtype", None))
        for i, x in enumerate(a)
    ]
    per_round = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(replay_reps):
            orig(*a, **k)
        per_round.append((time.perf_counter() - t0) / replay_reps)
    replay_ms = statistics.median(per_round) * 1e3
    replay_sd = (statistics.stdev(per_round) * 1e3) if rounds > 1 else 0.0
    in_loop_ms = stats["wall"] / stats["n"] * 1e3

    return {
        "instance": os.path.basename(nl),
        "wall_s": wall,
        "nodes": int(result.node_count),
        "lp_calls": stats["n"],
        "lp_calls_per_node": stats["n"] / max(result.node_count, 1),
        "lp_wall_s": stats["wall"],
        "lp_share_of_wall_pct": 100.0 * stats["wall"] / wall,
        "in_loop_ms_per_call": in_loop_ms,
        "replay_ms_per_call": replay_ms,
        "replay_sd_ms": replay_sd,
        "python_overhead_ms_per_call": in_loop_ms - replay_ms,
        "python_overhead_pct_of_call": 100.0 * (in_loop_ms - replay_ms) / in_loop_ms,
        "warm_basis_given": stats["warm_basis_given"],
        "captured_arg_shapes": [[i, t, list(s) if s else None, str(d)] for i, t, s, d in shapes],
        "by_lp_rows": {
            str(m): {"n": v["n"], "ms_per_call": v["wall"] / v["n"] * 1e3}
            for m, v in sorted(stats["by_m"].items(), key=lambda kv: -kv[1]["wall"])
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", default="nvs05")
    ap.add_argument("--nl-dir", default="python/tests/data/minlplib_nl")
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--replay-reps", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import discopt

    check(
        os.path.abspath(discopt.__file__).startswith(os.path.abspath("python/discopt")),
        f"discopt imported from {discopt.__file__}, not the worktree under test",
    )

    records = []
    for name in [s.strip() for s in args.instances.split(",") if s.strip()]:
        nl = os.path.join(args.nl_dir, f"{name}.nl")
        check(os.path.exists(nl), f"missing instance {nl}")
        print(f"[{name}] solving ...", flush=True)
        rec = run(nl, args.time_limit, args.replay_reps, args.rounds)
        records.append(rec)
        print(
            f"[{name}] {rec['lp_calls']} LP calls ({rec['lp_calls_per_node']:.0f}/node), "
            f"{rec['lp_share_of_wall_pct']:.0f}% of wall; in-loop "
            f"{rec['in_loop_ms_per_call']:.3f} ms/call vs pure-binding replay "
            f"{rec['replay_ms_per_call']:.3f} ms (sd {rec['replay_sd_ms']:.3f}) "
            f"-> Python overhead {rec['python_overhead_pct_of_call']:.0f}%",
            flush=True,
        )

    out = {
        "probe": "issue1180_probe_lp_cost",
        "discopt_file": discopt.__file__,
        "records": records,
        "executed_assertions": ASSERTS["n"],
    }
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(json.dumps(out, indent=1))
    print(f"\nexecuted assertions: {ASSERTS['n']}")
    if ASSERTS["n"] == 0:
        print("PROBE MEASURED NOTHING", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
