#!/usr/bin/env python3
"""#2011 census: where does ``DISCOPT_KERNEL_FREE_LINEAR`` change the native kernel's
spec decision?

For every instance of the population (the ``scip_wins_screen`` population: MINLPLib
``.nl`` instances with a proven status and ``nvars``/``ncons`` under the caps, plus
every in-repo corpus ``.nl``), one subprocess runs ``Model.solve`` with the producer
wrapped: at the first ``build_spatial_kernel_spec`` call -- with the bounds the solver
actually passes, after presolve -- it builds the spec with the flag OFF and ON,
records both decisions and the free columns, and ends the solve. Instances that never
reach the producer are recorded as such; the flag cannot touch them.

Classes: ``admitted`` (OFF declines, ON builds: the flag changes the engine),
``free_factor`` (an infinite bound, still declined ON: a free factor or a free lifted
column), ``same`` (no infinite bound; OFF and ON must agree -- checked), ``no_spec``.

    python -u discopt_benchmarks/scripts/issue2011_free_linear_census.py --out <dir> \\
        --max-vars 200 --max-cons 200 --workers 3

Prints per-instance progress and an EXECUTED count; exits non-zero when nothing was
censused or an OFF/ON disagreement appears without an infinite bound.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scip_wins_screen as screen  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


class _Stop(Exception):
    pass


def _census_one(nl: str) -> None:
    import discopt._relax.spatial_producer as spp
    import discopt.modeling as dm
    import numpy as np

    real = spp.build_spatial_kernel_spec
    rec: dict = {"reached": False}

    def wrapped(model, bounds=None, deadline=None):
        lb = np.asarray(bounds[0]) if bounds is not None else None
        ub = np.asarray(bounds[1]) if bounds is not None else None
        n_inf = int((~np.isfinite(lb)).sum() + (~np.isfinite(ub)).sum()) if lb is not None else -1
        n_free_cols = int((~(np.isfinite(lb) & np.isfinite(ub))).sum()) if lb is not None else -1
        os.environ["DISCOPT_KERNEL_FREE_LINEAR"] = "0"
        off = real(model, bounds=bounds)
        os.environ["DISCOPT_KERNEL_FREE_LINEAR"] = "1"
        on = real(model, bounds=bounds)
        big = None
        off_big = None
        if off is not None:
            g = np.concatenate([off["global_lo"], off["global_hi"]])
            off_big = int((~np.isfinite(g) | (np.abs(g) >= 1e20)).sum())
        if on is not None:
            g = np.concatenate([on["global_lo"], on["global_hi"]])
            big = int((np.abs(g) >= 1e20).sum())
            assert np.isfinite(g).all(), "inf reached a spec"
        rec.update(
            reached=True,
            n_inf_bounds=n_inf,
            n_free_cols=n_free_cols,
            off_built=off is not None,
            on_built=on is not None,
            on_sentinel_bounds=big,
            off_infinite_bounds=off_big,
        )
        raise _Stop

    spp.build_spatial_kernel_spec = wrapped
    t0 = time.perf_counter()
    try:
        dm.from_nl(nl).solve(time_limit=8)
    except _Stop:
        pass
    rec["wall"] = time.perf_counter() - t0
    print("RESULT " + json.dumps(rec), flush=True)
    os._exit(0)


def classify(r: dict) -> str:
    if not r.get("reached"):
        return "no_spec"
    if r["on_built"] and not r["off_built"]:
        return "admitted"
    if r["n_free_cols"] > 0 and not r["on_built"]:
        return "free_factor"
    return "same"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-vars", type=int, default=200)
    ap.add_argument("--max-cons", type=int, default=200)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--_census", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args._census:
        _census_one(args._census)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (p["name"], None) for p in screen.build_panel(args.out, args.max_vars, args.max_cons, None)
    ]
    names = {n for n, _ in jobs}
    for p in sorted((REPO / "python/tests/data/minlplib_nl").glob("*.nl")):
        if p.stem not in names:
            jobs.append((p.stem, str(p)))
    out_file = args.out / "census.jsonl"
    done = set()
    if out_file.exists():
        done = {json.loads(ln)["name"] for ln in out_file.open()}
    jobs = [j for j in jobs if j[0] not in done]
    print(f"{len(jobs)} instances to census", flush=True)

    def one(job):
        name, path = job
        nl = path or str(
            screen._fetch(f"{screen.MINLPLIB}/nl/{name}.nl", args.out / "nl" / f"{name}.nl")
        )
        try:
            p = subprocess.run(
                [sys.executable, __file__, "--out", str(args.out), "--_census", nl],
                capture_output=True,
                text=True,
                timeout=120,
            )
            lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
            res = json.loads(lines[-1][7:]) if lines else {"crash": p.stderr[-800:]}
        except subprocess.TimeoutExpired:
            res = {"timeout": True}
        res["name"] = name
        return res

    executed = 0
    with ThreadPoolExecutor(args.workers) as ex, out_file.open("a") as fh:
        for res in ex.map(one, jobs):
            executed += 1
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            print(f"[{executed}/{len(jobs)}] {res['name']:28s} {classify(res)}", flush=True)
    rows = [json.loads(ln) for ln in out_file.open()]
    counts: dict[str, list[str]] = {}
    bad = []
    for r in rows:
        c = classify(r) if "crash" not in r and "timeout" not in r else "error"
        counts.setdefault(c, []).append(r["name"])
        if c == "same" and r.get("reached") and r["off_built"] != r["on_built"]:
            bad.append(r["name"])
    for c, ns in sorted(counts.items()):
        shown = sorted(ns) if c in ("admitted", "free_factor", "error") else ""
        print(f"{c:12s} {len(ns):4d} {shown}")
    print(f"EXECUTED {executed}; censused {len(rows)}; OFF/ON disagreements: {bad}")
    if not rows or bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
