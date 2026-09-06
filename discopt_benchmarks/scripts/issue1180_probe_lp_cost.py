#!/usr/bin/env python
"""#1180 -- what one OBBT/node probe LP costs, and how much of it is native.

The §1.3 table this issue supersedes recorded the Rust LP as **3.4 %** of nvs05's
wall ("the node LP is nothing"). The post-tape layer split (deliverable 1) puts
``discopt._rust.solve_lp_warm_csc_py`` at roughly half of that instance. This
probe answers the follow-up that decides what to build: is that half *native
simplex time* -- in which case the lever is the probe count, not the plumbing --
or Python marshaling around the binding, which is what #764 measured on tanksize
in 2026-07 (4.25 ms in-loop vs 1.28 ms pure-binding, i.e. ~70 % marshaling)?

Method: two nested timers in the SAME run, which is the only way the difference
means anything.

* ``_PersistentProbeLP.solve`` -- the whole probe: objective assembly, the
  standard-form concatenations, the warm-basis marshaling, the binding call.
* ``discopt._rust.solve_lp_warm_csc_py`` -- the binding call alone.

Python-per-probe is then the difference, measured on the same probes rather than
inferred by replaying a captured call against a different warm state. (An earlier
version of this probe did exactly that and produced a *negative* overhead: the
captured call cold-starts while the in-loop population is warm-started, so it
compared two different LPs. The two-level timer has no such failure mode.)

Warm-basis usage is counted rather than assumed, so "warm-started" is a
measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ASSERTS = {"n": 0}


def check(cond: bool, msg: str) -> None:
    ASSERTS["n"] += 1
    if not cond:
        raise AssertionError(msg)


def run(nl: str, time_limit: float) -> dict:
    from discopt import _rust
    from discopt._relax import obbt
    from discopt.modeling.core import from_nl

    orig_binding = _rust.solve_lp_warm_csc_py
    orig_probe = obbt._PersistentProbeLP.solve

    st = {
        "native_n": 0,
        "native_wall": 0.0,
        "probe_n": 0,
        "probe_wall": 0.0,
        "probe_warm": 0,
        "probe_native_n": 0,
        "probe_native_wall": 0.0,
        "in_probe": False,
        "rows": {},
    }

    def binding(*a, **k):
        st["native_n"] += 1
        t = time.perf_counter()
        try:
            return orig_binding(*a, **k)
        finally:
            dt = time.perf_counter() - t
            st["native_wall"] += dt
            if st["in_probe"]:
                st["probe_native_n"] += 1
                st["probe_native_wall"] += dt
            m = int(a[1]) if len(a) > 1 else -1
            b = st["rows"].setdefault(m, {"n": 0, "wall": 0.0})
            b["n"] += 1
            b["wall"] += dt

    def probe(self, c, lb_arr, ub_arr, warm_basis):
        st["probe_n"] += 1
        if warm_basis is not None:
            st["probe_warm"] += 1
        st["in_probe"] = True
        t = time.perf_counter()
        try:
            return orig_probe(self, c, lb_arr, ub_arr, warm_basis)
        finally:
            st["probe_wall"] += time.perf_counter() - t
            st["in_probe"] = False

    _rust.solve_lp_warm_csc_py = binding
    obbt._PersistentProbeLP.solve = probe
    try:
        model = from_nl(nl)
        t0 = time.perf_counter()
        result = model.solve(time_limit=time_limit, gap_tolerance=1e-4)
        wall = time.perf_counter() - t0
    finally:
        _rust.solve_lp_warm_csc_py = orig_binding
        obbt._PersistentProbeLP.solve = orig_probe

    check(st["native_n"] > 0, f"{nl}: the LP binding was never called -- probe measured nothing")

    py_wall = st["probe_wall"] - st["probe_native_wall"]
    return {
        "instance": os.path.basename(nl),
        "wall_s": wall,
        "nodes": int(result.node_count),
        "binding_calls": st["native_n"],
        "binding_wall_s": st["native_wall"],
        "binding_share_of_wall_pct": 100.0 * st["native_wall"] / wall,
        "binding_ms_per_call": st["native_wall"] / st["native_n"] * 1e3,
        "obbt_probes": st["probe_n"],
        "obbt_probes_per_node": st["probe_n"] / max(result.node_count, 1),
        "obbt_probe_wall_s": st["probe_wall"],
        "obbt_probe_share_of_wall_pct": 100.0 * st["probe_wall"] / wall,
        "obbt_probe_ms_per_probe": (st["probe_wall"] / st["probe_n"] * 1e3)
        if st["probe_n"]
        else None,
        "obbt_native_ms_per_probe": (
            (st["probe_native_wall"] / st["probe_n"] * 1e3) if st["probe_n"] else None
        ),
        "obbt_python_ms_per_probe": (py_wall / st["probe_n"] * 1e3) if st["probe_n"] else None,
        "obbt_python_pct_of_probe": (
            100.0 * py_wall / st["probe_wall"] if st["probe_wall"] > 0 else None
        ),
        "obbt_warm_basis_pct": (
            100.0 * st["probe_warm"] / st["probe_n"] if st["probe_n"] else None
        ),
        "binding_calls_inside_probes": st["probe_native_n"],
        "by_lp_rows": {
            str(m): {"n": v["n"], "ms_per_call": v["wall"] / v["n"] * 1e3}
            for m, v in sorted(st["rows"].items(), key=lambda kv: -kv[1]["wall"])[:6]
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", default="nvs05")
    ap.add_argument("--nl-dir", default="python/tests/data/minlplib_nl")
    ap.add_argument("--time-limit", type=float, default=20.0)
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
        rec = run(nl, args.time_limit)
        records.append(rec)
        print(
            f"[{name}] binding {rec['binding_calls']} calls, "
            f"{rec['binding_share_of_wall_pct']:.0f}% of wall, "
            f"{rec['binding_ms_per_call']:.3f} ms/call",
            flush=True,
        )
        if rec["obbt_probes"]:
            print(
                f"[{name}] OBBT probes {rec['obbt_probes']} "
                f"({rec['obbt_probes_per_node']:.0f}/node, "
                f"{rec['obbt_warm_basis_pct']:.0f}% warm-started): "
                f"{rec['obbt_probe_ms_per_probe']:.3f} ms/probe = "
                f"{rec['obbt_native_ms_per_probe']:.3f} native + "
                f"{rec['obbt_python_ms_per_probe']:.3f} Python "
                f"({rec['obbt_python_pct_of_probe']:.0f}% Python)",
                flush=True,
            )
        else:
            print(f"[{name}] no OBBT probe LPs on this instance", flush=True)

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
