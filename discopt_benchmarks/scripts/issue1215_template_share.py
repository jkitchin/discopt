"""GATE 3: construction's share of an actual solve, on real corpus instances.

Gate 1 gives per-element construction wall at corpus row counts.  This asks the
only question that makes that number mean anything: what fraction of the work a
user actually waits for is it?  A lever that removes 80% of 0.3% of the wall is
not a lever.

Uses the largest in-repo instances (the most favourable case for a construction
lever -- the small ones are even more lopsided).
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import issue1215_template_entry as template_probe  # noqa: E402


def n_rows_of(path: str) -> int:
    with open(path, "rb") as fh:
        head = fh.read(4096).decode("latin-1")
    return int(head.splitlines()[1].split()[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="python/tests/data/minlplib_nl/*.nl")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--time-limit", type=float, default=30.0)
    args = ap.parse_args()

    print(f"# discopt: {template_probe.assert_loaded_discopt()[1]}")
    print(f"# load at start: {os.getloadavg()[0]:.2f}")

    import discopt.modeling as dm
    from discopt.modeling import Model
    from discopt.modeling.core import from_nl

    files = sorted(glob.glob(args.corpus), key=lambda p: -n_rows_of(p))[: args.top]
    if not files:
        raise SystemExit(f"GATE 3 found ZERO instances in {args.corpus!r}")

    # Measured per-element construction cost for a family of n rows, so the
    # share is computed from a MEASUREMENT at that size, not an extrapolation.
    def construct_ms(n: int) -> float:
        if n <= 0:
            return 0.0
        samples = []
        for _ in range(3):
            m = Model()
            idx = m.set("I", list(range(n)))
            x = m.continuous("x", over=idx, lb=0.5, ub=4.0)
            y = m.continuous("y", over=idx, lb=0.5, ub=4.0)
            t0 = time.perf_counter()
            m.constraint(idx, lambda i, x=x, y=y: x[i] * y[i] + dm.exp(x[i]) <= 4.0)
            samples.append(time.perf_counter() - t0)
        return min(samples) * 1e3

    print(f"\n{'instance':<20} {'rows':>6} {'build_ms':>9} {'solve_s':>9} {'share%':>8}  status")
    n_done = 0
    shares = []
    for path in files:
        name = os.path.basename(path)[:-3]
        n = n_rows_of(path)
        b_ms = construct_ms(n)
        m = from_nl(path)
        t0 = time.perf_counter()
        res = m.solve(time_limit=args.time_limit)
        solve_s = time.perf_counter() - t0
        share = 100.0 * (b_ms / 1e3) / solve_s if solve_s > 0 else float("nan")
        shares.append(share)
        n_done += 1
        print(f"{name:<20} {n:>6} {b_ms:>9.3f} {solve_s:>9.3f} {share:>8.3f}  {res.status}")

    if n_done == 0:
        raise SystemExit("GATE 3 executed ZERO instances")
    print(f"\nmedian construction share of solve wall: {statistics.median(shares):.3f}%")
    print(f"[gate3] instances measured: {n_done}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
