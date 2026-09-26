"""Measure the four ``Model.piecewise`` encodings against each other (#1482).

Evidence for the default ``method=``. Every encoding is exact, so the choice is
purely about speed; this panel measures it rather than asserting it.

Two model classes, both built from seeded random nonconvex tables so no instance
is hand-picked:

* ``milp``  -- K separable PWL costs, one coupling row ``sum x_k == D``,
  ``min sum f_k(x_k)`` (pure MILP; routed to the LP/MILP backend).
* ``minlp`` -- the same inputs, but ``min sum (f_k(x_k) - t_k)^2`` (nonconvex
  in ``y`` only through the PWL; solved by spatial B&B).

For each (class, K, n, seed) the four methods run *interleaved* (method order
rotated per instance) so load drift hits all of them alike. Each run's
certified objective is cross-checked against the other methods -- a method
disagreeing with the others is a correctness failure and aborts the panel.

Usage::

    python -u discopt_benchmarks/scripts/piecewise_method_panel.py [--seeds 3]

Prints one line per run and a summary (median / mean / sd wall, shifted
geometric mean, median nodes) per method. Exits non-zero if no run executed.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import discopt.modeling as dm
import numpy as np
from discopt.modeling._piecewise import PIECEWISE_METHODS


def _table(rng, n):
    b = np.linspace(0.0, 10.0, n)
    v = np.cumsum(rng.uniform(-3.0, 4.0, n))  # nonconvex, mostly increasing
    return b, v - v.min()


def _build(kind, n_fn, n, seed, method):
    rng = np.random.default_rng(seed)
    m = dm.Model(f"{kind}_{n_fn}_{n}_{seed}")
    x = m.continuous("x", shape=(n_fn,), lb=0.0, ub=10.0)
    ys = []
    targets = rng.uniform(2.0, 12.0, n_fn)
    for k in range(n_fn):
        b, v = _table(rng, n)
        ys.append(m.piecewise(x[k], b, v, method=method, name=f"f{k}"))
    m.subject_to(dm.sum(x) == 4.0 * n_fn)
    if kind == "milp":
        m.minimize(sum(ys))
    else:
        m.minimize(sum((ys[k] - targets[k]) ** 2 for k in range(n_fn)))
    return m


def _sgm(ts, shift=1.0):
    return float(np.exp(np.mean(np.log(np.asarray(ts) + shift))) - shift)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument(
        "--methods",
        default=",".join(PIECEWISE_METHODS),
        help="comma-separated subset of methods to compare",
    )
    args = ap.parse_args(argv)
    methods = tuple(mt.strip() for mt in args.methods.split(",") if mt.strip())
    unknown = set(methods) - set(PIECEWISE_METHODS)
    if unknown:
        ap.error(f"unknown method(s): {sorted(unknown)}")

    configs = [
        ("milp", 8, 9),
        ("milp", 8, 17),
        ("milp", 8, 33),
        ("minlp", 4, 9),
        ("minlp", 4, 17),
    ]
    runs: dict[str, list[tuple[float, int]]] = {mt: [] for mt in methods}
    executed = 0
    rot = 0
    for kind, n_fn, n in configs:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            order = list(methods[rot:] + methods[:rot])
            rot = (rot + 1) % len(methods)
            objs = {}
            for method in order:
                m = _build(kind, n_fn, n, seed, method)
                t0 = time.perf_counter()
                r = m.solve(time_limit=args.time_limit)
                wall = time.perf_counter() - t0
                nodes = int(r.node_count or 0)
                print(
                    f"{kind:5s} K={n_fn} n={n:2d} seed={seed} {method:13s} "
                    f"status={r.status:10s} cert={r.gap_certified!s:5s} "
                    f"obj={r.objective!s:>22s} wall={wall:7.3f}s nodes={nodes}",
                    flush=True,
                )
                if r.status != "optimal" or not r.gap_certified:
                    # Not a failure of soundness, but a run that did not finish is
                    # charged the full limit so a slow method cannot hide.
                    wall = max(wall, args.time_limit)
                else:
                    objs[method] = float(r.objective)
                runs[method].append((wall, nodes))
                executed += 1
            if len(objs) >= 2:
                ref = next(iter(objs.values()))
                for o in objs.values():
                    if abs(o - ref) > 1e-5 * max(1.0, abs(ref)):
                        print(f"OBJECTIVE DISAGREEMENT: {objs}", flush=True)
                        return 2

    print("\nsummary (wall seconds; unfinished runs charged the time limit)")
    print(
        f"{'method':13s} {'runs':>4s} {'median':>8s} {'mean':>8s} {'sd':>8s} {'sgm':>8s} "
        f"{'med nodes':>9s}"
    )
    for method, rs in runs.items():
        ts = [w for w, _ in rs]
        ns = [nd for _, nd in rs]
        print(
            f"{method:13s} {len(rs):4d} {statistics.median(ts):8.3f} {statistics.mean(ts):8.3f} "
            f"{statistics.stdev(ts) if len(ts) > 1 else 0.0:8.3f} {_sgm(ts):8.3f} "
            f"{statistics.median(ns):9.1f}"
        )
    print(f"\nexecuted runs: {executed}")
    return 0 if executed else 1


if __name__ == "__main__":
    sys.exit(main())
