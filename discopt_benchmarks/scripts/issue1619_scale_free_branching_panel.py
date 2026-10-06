"""#1619 D-11 units panel: ``DISCOPT_SCALE_FREE_BRANCHING`` OFF vs ON.

The native spatial kernel picks its branch term by absolute McCormick gap and the
operand by absolute width, so the same model written in other units takes a
different tree (15,745 vs 593 nodes on the issue's plant). This panel solves
pooling instances -- three seeded two-pool plants and Haverly 1-3 -- in the p-,
q- and pq-formulations with flows in 1, 10 and 100 barrels, OFF and ON back to
back per row. The unit change is exact (prices and costs scale with it), so every
row of one instance has the same optimum in model units: the unit-1 OFF certified
objective is the oracle for its siblings.

Reports per arm: false certificates (certified off the oracle, bound past it),
certificates lost, total nodes / wall, and the node-count spread across units
(max/min over the three units) -- the quantity the flag exists to shrink. The
corpus half of the §5 gate is ``recentre_graduation_panel.py --flag
DISCOPT_SCALE_FREE_BRANCHING``. Prints the executed-comparison count; exits
non-zero when it is zero (CLAUDE.md §6).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from _pooling.pool18 import build_pool, haverly, in_units, plant  # noqa: E402

FLAG = "DISCOPT_SCALE_FREE_BRANCHING"
UNITS = (1, 10, 100)


def instances():
    out = [(f"plant{s}", lambda s=s: plant(seed=s)) for s in (5, 6, 7)]
    out += [(f"haverly{c}", lambda c=c: haverly(c)) for c in (1, 2, 3)]
    return out


def solve(data, form, unit, on, tl):
    os.environ[FLAG] = "1" if on else "0"
    m, _ = build_pool(in_units(data, unit), form)
    t0 = time.perf_counter()
    r = m.solve(time_limit=tl)
    return r, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=60.0)
    args = ap.parse_args()
    import discopt

    print("discopt", discopt.__file__, "load", os.getloadavg(), flush=True)
    tally = {a: dict(false=0, cert=0, nodes=0, wall=0.0) for a in ("OFF", "ON")}
    spread = {"OFF": [], "ON": []}
    lost = gained = compared = 0
    for name, make in instances():
        for form in ("p", "q", "pq"):
            oracle = None
            nodes = {"OFF": [], "ON": []}
            for unit in UNITS:
                res = {}
                for arm, on in (("OFF", False), ("ON", True)):
                    r, wall = solve(make(), form, unit, on, args.time_limit)
                    res[arm] = r
                    t = tally[arm]
                    t["nodes"] += int(r.node_count or 0)
                    t["wall"] += wall
                    t["cert"] += bool(r.gap_certified)
                    nodes[arm].append(max(1, int(r.node_count or 0)))
                    if oracle is None and arm == "OFF" and unit == 1 and r.gap_certified:
                        oracle = float(r.objective)
                    if oracle is not None:
                        tol = 1e-4 * (1 + abs(oracle))
                        if r.gap_certified and abs(r.objective - oracle) > tol:
                            t["false"] += 1
                            print(f"FALSE {arm} {name}/{form}/u{unit}: {r.objective} vs {oracle}")
                        # maximize: a valid bound is >= the optimum
                        if r.bound is not None and r.bound < oracle - tol:
                            t["false"] += 1
                            print(f"FALSE {arm} {name}/{form}/u{unit}: bound {r.bound} < {oracle}")
                    print(
                        f"{name:9s} {form:2s} u{unit:<3d} {arm:3s} {r.status:10s} "
                        f"obj={r.objective} n={r.node_count} wall={wall:.1f} "
                        f"route={(r.algorithm_route or '')[:20]}",
                        flush=True,
                    )
                compared += 1
                c0, c1 = res["OFF"].gap_certified, res["ON"].gap_certified
                lost += bool(c0 and not c1)
                gained += bool(c1 and not c0)
            for arm in ("OFF", "ON"):
                spread[arm].append(max(nodes[arm]) / min(nodes[arm]))
    for arm, t in tally.items():
        s = sorted(spread[arm])
        print(
            f"{arm}: false={t['false']} certified={t['cert']} nodes={t['nodes']} "
            f"wall={t['wall']:.1f}s unit-spread median={s[len(s) // 2]:.2f} max={s[-1]:.2f}",
            flush=True,
        )
    print(f"compared={compared} lost={lost} gained={gained}", flush=True)
    return 0 if compared else 1


if __name__ == "__main__":
    sys.exit(main())
