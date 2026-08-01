#!/usr/bin/env python
"""Issue #911 follow-on probe — WHERE does a convex-kernel attempt spend its wall?

The #911 fix deducts the attempt from the caller's budget, which bounds the total at
``time_limit`` only if the attempt itself respects the budget it was handed. The
BEFORE panel says it sometimes does not: ``clay0304hfsg`` at a 30 s budget ran
**272.76 s** end to end with the kernel ON, against 31.19 s with it OFF. Deducting a
242 s attempt from a 30 s budget cannot bring that inside 30 s — so before doing
anything about it, this probe measures which phase of the attempt overruns:

* ``build_convex_spec`` — the convexity classification (no deadline of its own);
* ``solve_convex_tree`` — the native tree, handed ``time_limit_s=budget``;
* ``_incumbent_is_feasible`` — the #779 pristine-model verification, JAX-compiled.

Each phase is timed separately and printed with its overrun against the budget, so
the answer is a measurement rather than a reading of the code. Exits non-zero if it
timed nothing (CLAUDE.md §6).

Usage::

    python -u discopt_benchmarks/scripts/issue911_attempt_phase_probe.py \
        --instances clay0304hfsg,clay0305hfsg --budget 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CORPUS = _REPO_ROOT / "python" / "tests" / "data" / "minlplib_nl"
_SNAPSHOT = Path(os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"))


def _find(stem: str) -> Path | None:
    for d in (_CORPUS, _SNAPSHOT):
        p = d / f"{stem}.nl"
        if p.exists():
            return p
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--budget", type=float, default=30.0)
    args = ap.parse_args()

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ["DISCOPT_CONVEX_KERNEL"] = "1"

    import numpy as np
    from discopt.modeling.core import from_nl
    from discopt.solvers import _convex_kernel as ck

    timed = 0
    print(f"budget = {args.budget:g} s\n", flush=True)
    for stem in [s.strip() for s in args.instances.split(",") if s.strip()]:
        nl = _find(stem)
        if nl is None:
            print(f"{stem}: MISSING", flush=True)
            continue
        m = from_nl(str(nl))

        t = time.perf_counter()
        spec = ck.build_convex_spec(m)
        t_spec = time.perf_counter() - t
        if spec is None:
            print(f"{stem:20s} not eligible (spec {t_spec:.2f}s)", flush=True)
            continue

        t = time.perf_counter()
        r = ck.solve_convex_tree(spec, time_limit_s=args.budget, gap_tol=1e-4)
        t_tree = time.perf_counter() - t

        t_verify = float("nan")
        inc_x = np.asarray(r["incumbent_x"], float)
        if inc_x.size:
            _, x_flat = ck._unflatten(m, inc_x)
            t = time.perf_counter()
            ck._incumbent_is_feasible(m, x_flat)
            t_verify = time.perf_counter() - t

        total = t_spec + t_tree + (0.0 if t_verify != t_verify else t_verify)
        timed += 1
        print(
            f"{stem:20s} spec={t_spec:7.2f}s tree={t_tree:7.2f}s verify={t_verify:7.2f}s "
            f"total={total:7.2f}s  over-budget={total - args.budget:+7.2f}s  "
            f"status={r['status']} nodes={r['node_count']}",
            flush=True,
        )

    print(f"\nEXECUTED PHASE TIMINGS: {timed}", flush=True)
    if timed == 0:
        print("probe measured NOTHING", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
