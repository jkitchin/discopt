"""What the #1009 node-LP deadline is worth, as a function of node-LP size.

#1014 threaded the tree's live deadline into the node LP (`node_lp_opts`) on the
argument that it is sound by construction. It shipped without a measurement, and the
instance the issue was opened for turned out to overrun for an unrelated reason (the
Python incremental fast path, fixed separately in #1015), so nothing on record says
what the kernel-side fix actually bounds.

This measures it. The quantity that matters is the *overshoot*: the tree checks its
clock only BETWEEN nodes, so without a per-LP deadline the wall-clock floor of any
solve is the cost of one node LP, however small the budget.

Both arms run against the SAME (fixed) code, so this is reproducible on `main` with
no flag and no revert:

* **uncapped** — `time_limit_s=None, max_nodes=1`: one node LP, run to completion.
  This is exactly what the pre-#1014 path spent on node 1 regardless of the budget.
* **capped** — `time_limit_s=BUDGET`: the same LP under the deadline.

The uncapped time is therefore the wall the legacy path could not go below; the
capped time is what it costs now. Holding BUDGET fixed and growing the relaxation
shows the overshoot is unbounded in LP size rather than a property of one instance
(CLAUDE.md §2 — the driver is a general dense-bilinear model, not a named instance).

Usage:  python -u scripts/spatial_kernel_lp_deadline_scaling.py [n ...]
        SCALE_BUDGET=1.0 SCALE_REPS=3 python -u scripts/... 50 70 90 120

Prints per-size progress unbuffered (§10) and an executed-comparison count, exiting
non-zero if it is zero (§6).
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import discopt
import discopt.modeling as dm
from discopt import _rust
from discopt._relax.spatial_producer import build_spatial_kernel_spec

BUDGET = float(os.environ.get("SCALE_BUDGET", "1.0"))
REPS = int(os.environ.get("SCALE_REPS", "3"))


def dense_bilinear(n: int):
    """``min sum_{i<j} x_i x_j  s.t.  sum x >= n/2,  x in [0,1]^n``.

    ``n(n-1)/2`` lifted bilinear terms over a trivial box: the node LP grows
    quadratically while the search stays a single root node, which isolates the cost
    of one LP — the quantity the overshoot is bounded by. Global minimum ``C(n/2, 2)``
    in closed form (``sum_{i<j} x_i x_j = ((sum x)^2 - sum x^2)/2``, minimized at
    ``sum x = n/2`` with ``n/2`` variables at 1), so the soundness check has an oracle.
    """
    m = dm.Model()
    xs = [m.continuous(f"x{i}", lb=0.0, ub=1.0) for i in range(n)]
    m.constraint(dm.RangeSet(1), lambda _k: sum(xs) >= n / 2.0, name="half", fast=False)
    terms = [xs[i] * xs[j] for i in range(n) for j in range(i + 1, n)]
    # Balanced reduction: a left-deep `sum()` chain exceeds the canonicalizer's
    # recursion limit long before the LP is big enough to matter here.
    while len(terms) > 1:
        terms = [
            terms[k] + terms[k + 1] if k + 1 < len(terms) else terms[k]
            for k in range(0, len(terms), 2)
        ]
    m.minimize(terms[0])
    return m


def optimum(n: int) -> float:
    half = n // 2
    return half * (half - 1) / 2.0


def main(sizes: list[int]) -> int:
    # §8: say which code produced these numbers.
    print(f"# discopt.__file__ = {discopt.__file__}", flush=True)
    print(f"# _rust.__file__   = {_rust.__file__}", flush=True)
    print(f"# budget={BUDGET}s reps={REPS}", flush=True)
    print(
        f"{'n':>4} {'terms':>6} {'cols':>6} | {'uncapped (1 node)':>22} | "
        f"{'capped':>22} | bound <= opt",
        flush=True,
    )

    compared = 0
    for size in sizes:
        spec = build_spatial_kernel_spec(dense_bilinear(size))
        if spec is None:
            print(f"{size:>4} SKIP (producer declined)", flush=True)
            continue
        for key in [k for k in spec if k.startswith("meta_")]:
            spec.pop(key)

        uncapped, capped = [], []
        sound = True
        for rep in range(REPS):
            # Alternate the order so a first-run penalty cannot land on one arm
            # (CLAUDE.md §9 — interleaved, not sequential).
            order = ("uncapped", "capped") if rep % 2 == 0 else ("capped", "uncapped")
            for arm in order:
                t0 = time.perf_counter()
                if arm == "uncapped":
                    res = _rust.solve_spatial_tree_py(**spec, time_limit_s=None, max_nodes=1)
                    uncapped.append(time.perf_counter() - t0)
                else:
                    res = _rust.solve_spatial_tree_py(**spec, time_limit_s=BUDGET, max_nodes=10**9)
                    capped.append(time.perf_counter() - t0)
                compared += 1
                # Soundness rides along on every solve: a cut LP must never lift the
                # dual bound above the true optimum.
                if res["bound"] > optimum(size) + 1e-6 * (1.0 + optimum(size)):
                    sound = False
                    print(
                        f"!! UNSOUND: n={size} {arm} bound={res['bound']} "
                        f"> optimum {optimum(size)}",
                        flush=True,
                    )

        u_med, c_med = statistics.median(uncapped), statistics.median(capped)
        u_sd = statistics.stdev(uncapped) if REPS > 1 else 0.0
        c_sd = statistics.stdev(capped) if REPS > 1 else 0.0
        print(
            f"{size:>4} {size * (size - 1) // 2:>6} {spec['n_cols']:>6} | "
            f"{u_med:7.2f}s sd {u_sd:4.2f} {u_med / BUDGET:5.1f}x | "
            f"{c_med:7.2f}s sd {c_sd:4.2f} {c_med / BUDGET:5.1f}x | "
            f"{'ok' if sound else 'VIOLATED'}",
            flush=True,
        )

    print(f"# executed kernel solves: {compared}", flush=True)
    if compared == 0:
        print("PROBE FIRED ZERO SOLVES", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main([int(a) for a in sys.argv[1:]] or [50, 70, 90, 120]))
