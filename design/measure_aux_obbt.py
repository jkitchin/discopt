"""Decision-gate measurement for issue #208 (OBBT-on-auxiliaries).

The McCormick relaxation bakes its envelope rows over the *build-time* auxiliary
(lifted product/ratio) bounds and never regenerates them, so any OBBT tightening
of an aux column is discarded — the "80% of the work thrown away" gap. Before
investing in the (soundness-sensitive) envelope-rebuild, issue #208 prescribes a
cheap measurement: *how much would the aux columns tighten if OBBT ran on them?*

This script builds the McCormick relaxation of each vendored ``.nl`` instance,
runs OBBT over the aux columns only (via
:func:`discopt._jax.obbt.measure_discarded_aux_tightening`), and prints the
per-instance discarded tightening plus an aggregate verdict. It touches nothing
on the solve path — it is a pure diagnostic.

Usage::

    python design/measure_aux_obbt.py [--cutoff] [--limit N] [--glob PATTERN]

``--cutoff`` first solves each instance (short time limit) and feeds the incumbent
objective as an optimality-based cutoff (the regime where aux tightening is
largest). Default is structural-only (no incumbent), the conservative measurement.
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import discopt.modeling as dm
import numpy as np
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.obbt import measure_discarded_aux_tightening

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "python", "tests", "data", "minlplib_nl")


def _maybe_cutoff(model, enable: bool, time_limit: float = 8.0):
    if not enable:
        return None
    try:
        res = model.solve(time_limit=time_limit, gap_tolerance=1e-3)
        return None if res.objective is None else float(res.objective)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", action="store_true", help="feed a solved incumbent cutoff")
    ap.add_argument("--limit", type=int, default=0, help="max instances (0 = all)")
    ap.add_argument("--glob", default="*.nl", help="filename glob within the corpus dir")
    ap.add_argument("--time-per-lp", type=float, default=0.2)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(CORPUS_DIR, args.glob)))
    if args.limit:
        files = files[: args.limit]

    widths = [26, 5, 6, 7, 7, 5, 5]
    cols = ("instance", "#aux", "#tght", "mean%", "max%", "lp", "t(s)")
    print(" ".join(f"{c:>{w}}" for c, w in zip(cols, widths)))
    print("-" * 72)

    rows = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            model = dm.from_nl(path)
        except Exception as e:
            print(f"{name:28s}  load-failed: {type(e).__name__}")
            continue
        lb, ub = flat_variable_bounds(model)
        cutoff = _maybe_cutoff(model, args.cutoff)
        t0 = time.perf_counter()
        rep = measure_discarded_aux_tightening(
            model, lb, ub, incumbent_cutoff=cutoff, time_limit_per_lp=args.time_per_lp
        )
        dt = time.perf_counter() - t0
        if rep is None:
            print(f"{name:28s}  (no relaxable aux columns)")
            continue
        rows.append(rep)
        print(
            f"{name:28s} {rep.n_aux:5d} {rep.n_aux_tightened:6d} "
            f"{100 * rep.mean_rel_reduction:7.1f} {100 * rep.max_rel_reduction:7.1f} "
            f"{rep.n_lp_solves:5d} {dt:6.1f}"
        )

    print("-" * 72)
    if not rows:
        print("no instances with aux columns measured")
        return
    n_inst = len(rows)
    big = [r for r in rows if r.mean_rel_reduction > 0.10]
    any_tight = [r for r in rows if r.n_aux_tightened > 0]
    mean_of_means = float(np.mean([r.mean_rel_reduction for r in rows]))
    print(
        f"instances with aux: {n_inst}   "
        f"any aux tightened: {len(any_tight)}   "
        f"mean aux reduction > 10%: {len(big)}"
    )
    print(f"corpus mean of per-instance mean aux reduction: {100 * mean_of_means:.1f}%")
    print(
        "VERDICT: "
        + (
            "part-2 envelope rebuild looks worthwhile (widespread aux tightening)"
            if len(big) >= max(3, n_inst // 4)
            else "park part 2 — aux tightening is sparse / small on this corpus"
        )
    )


if __name__ == "__main__":
    main()
