#!/usr/bin/env python3
"""#2021 section 2 entry experiment: does switching off HiGHS's sub-MIPs cut the
lp_nlp_bb master's wall on slay without loosening a bound?

No library edit: arm C wraps ``highspy.Highs.passModel`` process-wide so every
HiGHS instance the solve creates (each master and the #1634 cross-solve) also gets
``mip_heuristic_run_rins = mip_heuristic_run_rens = False``. Arms:

* ``A`` -- ``DISCOPT_CONVEX_MINLP_ROUTE=0`` (the route-off reference of the
  issue's "within 2x of the route-off arm");
* ``B`` -- route on (the default);
* ``C`` -- route on, sub-MIPs off.

Each solve runs in its own subprocess; the arm order rotates per (instance, rep)
so the arms are interleaved (CLAUDE.md section 9). Every incumbent is verified on
a freshly parsed model. Prints per-solve progress and an EXECUTED count, and exits
non-zero when nothing ran or when arm C's patch never fired on a routed solve.

A baseline-only instrument: it asserts ``solve_milp_with_lazy_cuts`` has no
``highs_options`` parameter yet, so it refuses to run on a tree that has the flag.
Its result is recorded in performance-plan section 103.7.

    python -u discopt_benchmarks/scripts/issue2021_submip_probe.py --out <dir> \\
        --time-limit 60 --instances slay04m,slay05m --reps 1 [--reps-for slay07m:3]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scip_wins_screen as screen  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OPTS = {"mip_heuristic_run_rins": False, "mip_heuristic_run_rens": False}


def _solve(nl: str, tl: float, arm: str) -> None:
    import discopt
    import highspy
    import numpy as np
    from discopt.modeling.core import from_nl
    from discopt.solvers import milp_highs
    from discopt.validation.feasibility import verify_point

    assert discopt.__file__.startswith(str(REPO)), discopt.__file__
    # CLAUDE.md section 8: the entry experiment runs on the code WITHOUT the flag.
    import inspect

    assert "highs_options" not in inspect.signature(milp_highs.solve_milp_with_lazy_cuts).parameters
    patched = [0]
    if arm == "C":
        orig = highspy.Highs.passModel

        def _pass(self, *a, **k):
            for key, val in OPTS.items():
                if self.setOptionValue(key, val) != highspy.HighsStatus.kOk:
                    raise RuntimeError(f"HiGHS rejected {key}")
            patched[0] += 1
            return orig(self, *a, **k)

        highspy.Highs.passModel = _pass
    masters: list[dict] = []
    orig_lazy = milp_highs.solve_milp_with_lazy_cuts

    def _lazy(*a, **k):
        res = orig_lazy(*a, **k)
        cs = res.callback_stats or {}
        masters.append(
            {
                "restarts": cs.get("restarts"),
                "nodes": res.node_count,
                "bound": res.bound,
                "wall": res.wall_time,
                "cross": cs.get("presolve_cross_check"),
            }
        )
        return res

    milp_highs.solve_milp_with_lazy_cuts = _lazy
    t0 = time.perf_counter()
    r = from_nl(nl).solve(time_limit=tl, gap_tolerance=1e-4)
    wall = time.perf_counter() - t0
    bad = None
    if r.x is not None and r.objective is not None:
        fresh = from_nl(nl)
        flat = np.concatenate(
            [np.ravel(np.asarray(r.x[v.name], dtype=np.float64)) for v in fresh._variables]
        )
        bad = not verify_point(fresh, flat).ok

    def f(v):
        return None if v is None or not math.isfinite(float(v)) else float(v)

    print(
        "RESULT "
        + json.dumps(
            {
                "status": str(r.status),
                "gap_certified": bool(r.gap_certified),
                "objective": f(r.objective),
                "bound": f(r.bound),
                "nodes": r.node_count,
                "wall": wall,
                "route": getattr(r, "algorithm_route", None),
                "verify_failed": bad,
                "patched_pass_model": patched[0],
                "masters": masters,
            },
            default=str,
        ),
        flush=True,
    )


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--_solve":
        _solve(sys.argv[2], float(sys.argv[3]), sys.argv[4])
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--instances", required=True)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--reps-for", default="", help="name:reps,name:reps")
    a = ap.parse_args()
    names = [s for s in a.instances.split(",") if s]
    reps = dict.fromkeys(names, a.reps)
    for tok in filter(None, a.reps_for.split(",")):
        n, k = tok.split(":")
        reps[n] = int(k)
    panel = {p["name"]: p for p in screen.build_panel(a.out, 10**9, 10**9, set(names))}
    out = a.out / f"submip_tl{int(a.time_limit)}.jsonl"
    executed = 0
    c_routed_unpatched = 0
    order = ["A", "B", "C"]
    k = 0
    with out.open("a") as fh:
        for rep in range(max(reps.values())):
            for name in names:
                if rep >= reps[name]:
                    continue
                nl = screen._fetch(f"{screen.MINLPLIB}/nl/{name}.nl", a.out / "nl" / f"{name}.nl")
                arms = order[k % 3 :] + order[: k % 3]
                k += 1
                for arm in arms:
                    env = dict(os.environ)
                    if arm == "A":
                        env["DISCOPT_CONVEX_MINLP_ROUTE"] = "0"
                    t0 = time.perf_counter()
                    proc = subprocess.run(
                        [
                            sys.executable,
                            "-u",
                            __file__,
                            "--_solve",
                            str(nl),
                            str(a.time_limit),
                            arm,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=a.time_limit + 180,
                        env=env,
                    )
                    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
                    if not lines:
                        res = {"status": "CRASH", "rc": proc.returncode, "err": proc.stderr[-2000:]}
                    else:
                        res = json.loads(lines[-1][7:])
                    res.update(name=name, arm=arm, rep=rep, opt=panel[name]["opt"])
                    res["proc_wall"] = time.perf_counter() - t0
                    if arm == "C" and res.get("masters") and not res.get("patched_pass_model"):
                        c_routed_unpatched += 1
                    fh.write(json.dumps(res) + "\n")
                    fh.flush()
                    executed += 1
                    ms = res.get("masters") or []
                    print(
                        f"[{executed}] {name:10s} rep={rep} arm={arm} {res['status']:22s} "
                        f"cert={res.get('gap_certified')} obj={res.get('objective')} "
                        f"bound={res.get('bound')} wall={res.get('wall')} nodes={res.get('nodes')} "
                        f"restarts={[m.get('restarts') for m in ms]} "
                        f"patched={res.get('patched_pass_model')} "
                        f"verify_failed={res.get('verify_failed')}",
                        flush=True,
                    )
    print(f"EXECUTED {executed}; arm-C routed solves with an unfired patch: {c_routed_unpatched}")
    if executed == 0 or c_routed_unpatched:
        sys.exit(1)


if __name__ == "__main__":
    main()
