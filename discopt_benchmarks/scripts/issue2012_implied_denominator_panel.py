#!/usr/bin/env python3
"""#2012 panel: ``DISCOPT_IMPLIED_DENOMINATOR`` OFF vs ON (CLAUDE.md section 5).

The population is every instance the census (``issue2012_implied_denominator_
census.py``) finds with a qualifying denominator; the flag cannot act anywhere
else. Each solve runs in its own subprocess, the arm order alternates per
repetition, and per solve the checks are:

* the dual bound is no better than the best known primal value in
  ``minlplib.solu`` (``=opt=`` or ``=best=``) -- valid for open instances too,
  since the optimum lies between them;
* a claimed optimum agrees with ``=opt=`` where one exists;
* every incumbent passes ``verify_point`` on a freshly parsed model, and its
  reported objective is its own.

    python -u discopt_benchmarks/scripts/issue2012_implied_denominator_panel.py \\
        --nl-dir <dir with .nl> --solu <minlplib.solu> --names 4stufen,beuster,minlphi \\
        --time-limit 60 --reps 2

Prints per-solve rows, a summary and the executed-check count; exits non-zero on a
failed check or when nothing was checked (CLAUDE.md section 6).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

FLAG = "DISCOPT_IMPLIED_DENOMINATOR"


def _solve_one(nl: str, tl: float) -> None:
    import numpy as np
    from discopt.modeling.core import from_nl
    from discopt.validation.feasibility import verify_point

    t0 = time.perf_counter()
    r = from_nl(nl).solve(time_limit=tl)
    wall = time.perf_counter() - t0
    verified = None
    vobj = None
    if r.x is not None and r.objective is not None:
        fresh = from_nl(nl)
        flat = np.concatenate(
            [np.ravel(np.asarray(r.x[v.name], dtype=np.float64)) for v in fresh._variables]
        )
        vr = verify_point(fresh, flat, with_objective=True)
        verified = bool(vr.ok)
        vobj = getattr(vr, "objective", None)

    def f(v):
        return None if v is None or not math.isfinite(float(v)) else float(v)

    print(
        "RESULT "
        + json.dumps(
            {
                "status": str(r.status),
                "cert": bool(r.gap_certified),
                "objective": f(r.objective),
                "bound": f(r.bound),
                "nodes": r.node_count,
                "wall": wall,
                "verified": verified,
                "verified_obj": f(vobj),
            }
        ),
        flush=True,
    )
    os._exit(0)


def _solu(path: Path) -> dict:
    ref: dict = {}
    for ln in path.read_text().splitlines():
        p = ln.split()
        if len(p) >= 3 and p[0] in ("=opt=", "=best="):
            ref[p[1]] = (p[0], float(p[2]))
    return ref


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--_solve":
        assert os.environ.get(FLAG) in ("0", "1"), os.environ.get(FLAG)
        _solve_one(sys.argv[2], float(sys.argv[3]))
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--nl-dir", type=Path, required=True)
    ap.add_argument("--solu", type=Path, required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    ref = _solu(a.solu)
    checks = 0
    fails: list[str] = []
    rows = []
    for name in a.names.split(","):
        kind, val = ref[name]
        for rep in range(a.reps):
            for arm in ("0", "1") if rep % 2 == 0 else ("1", "0"):
                p = subprocess.run(
                    [sys.executable, __file__, "--_solve", str(a.nl_dir / f"{name}.nl"), str(a.time_limit)],
                    capture_output=True,
                    text=True,
                    timeout=a.time_limit + 180,
                    env=dict(os.environ, **{FLAG: arm}),
                )
                lines = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT ")]
                if not lines:
                    fails.append(f"{name} arm={arm}: crash {p.stderr[-600:]}")
                    continue
                r = json.loads(lines[-1][7:])
                r.update(name=name, arm=arm)
                rows.append(r)
                print(json.dumps(r), flush=True)
                tol = 1e-6 * (1.0 + abs(val))
                if r["bound"] is not None:
                    checks += 1
                    if r["bound"] > val + tol:
                        fails.append(f"{name} arm={arm}: bound {r['bound']} above {kind} {val}")
                if r["cert"] and kind == "=opt=":
                    checks += 1
                    if abs(r["objective"] - val) > 1e-4 * (1.0 + abs(val)):
                        fails.append(f"{name} arm={arm}: certified {r['objective']} vs {val}")
                if r["verified"] is not None:
                    checks += 1
                    if not r["verified"]:
                        fails.append(f"{name} arm={arm}: incumbent fails verify_point")
                    elif r["verified_obj"] is not None and abs(
                        r["verified_obj"] - r["objective"]
                    ) > 1e-6 * (1.0 + abs(r["objective"])):
                        fails.append(f"{name} arm={arm}: objective is not its incumbent's")
    print("\nsummary (per arm: certified, total wall, bounds)")
    for name in a.names.split(","):
        for arm in ("0", "1"):
            rs = [r for r in rows if r["name"] == name and r["arm"] == arm]
            print(
                f"  {name:10s} {'ON ' if arm == '1' else 'OFF'} cert {sum(r['cert'] for r in rs)}/"
                f"{len(rs)} wall {sum(r['wall'] for r in rs):7.1f} s bounds "
                f"{[r['bound'] for r in rs]} nodes {[r['nodes'] for r in rs]}"
            )
    print(f"EXECUTED {checks} checks, {len(fails)} failures")
    for f_ in fails:
        print("FAIL", f_)
    sys.exit(1 if fails or not checks else 0)


if __name__ == "__main__":
    main()
