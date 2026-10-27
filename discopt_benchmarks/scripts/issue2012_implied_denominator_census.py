#!/usr/bin/env python3
"""#2012 census: where can ``DISCOPT_IMPLIED_DENOMINATOR`` act?

The flag narrows the base of a negative power only when the base is an affine sum
whose BOX interval holds 0 (``_Builder.denominator_bounds``). A node box is a
sub-box of the declared box, so a base whose interval over the declared box
excludes 0 excludes it at every node too: an instance where no base qualifies at
the declared box is untouched by the flag. One subprocess per instance builds the
uniform relaxation at the declared box with the flag ON and counts the qualifying
bases (``candidates``) and those the linear rows narrowed (``narrowed``).

    python -u discopt_benchmarks/scripts/issue2012_implied_denominator_census.py \\
        --out <dir> --max-vars 200 --max-cons 200 --workers 3

Prints per-instance progress and an EXECUTED count; exits non-zero when nothing
was censused.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scip_wins_screen as screen  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _census_one(nl: str) -> None:
    os.environ["DISCOPT_IMPLIED_DENOMINATOR"] = "1"
    # Deep polynomial objectives (autocorr, graphpart) recurse past the default
    # limit in a standalone build; ``Model.solve`` raises it the same way.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 50_000))
    import discopt._relax.uniform_relax as ur
    import discopt.modeling as dm

    counts = {"candidates": 0, "narrowed": 0}
    real = ur._Builder.denominator_bounds
    real_bounds = ur._Builder.bounds

    def wrapped(self, node):
        lo, hi = real_bounds(self, node)
        if lo <= 0.0 <= hi and node.kind == "sum" and ur.is_affine(node):
            counts["candidates"] += 1
        out = real(self, node)
        if out != (lo, hi):
            counts["narrowed"] += 1
        return out

    ur._Builder.denominator_bounds = wrapped
    rel = ur.build_uniform_relaxation(dm.from_nl(nl))
    counts["objective_bound_valid"] = bool(rel.model._objective_bound_valid)
    print("RESULT " + json.dumps(counts), flush=True)
    os._exit(0)


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
    out_file = args.out / "census2012.jsonl"
    done = {json.loads(ln)["name"] for ln in out_file.open()} if out_file.exists() else set()
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
            print(
                f"[{executed}/{len(jobs)}] {res['name']:28s} "
                f"cand={res.get('candidates')} narrowed={res.get('narrowed')}",
                flush=True,
            )
    rows = [json.loads(ln) for ln in out_file.open()]
    cand = sorted(r["name"] for r in rows if r.get("candidates"))
    narrowed = sorted(r["name"] for r in rows if r.get("narrowed"))
    errors = sorted(r["name"] for r in rows if "crash" in r or "timeout" in r)
    print(f"candidates {len(cand)}: {cand}")
    print(f"narrowed {len(narrowed)}: {narrowed}")
    print(f"errors {len(errors)}: {errors}")
    print(f"EXECUTED {executed}; censused {len(rows)}")
    if not rows:
        sys.exit(1)


if __name__ == "__main__":
    main()
