#!/usr/bin/env python3
"""#1917: lowering panel and soundness audit for ``DISCOPT_MBIGM_LP_COMPONENTS``.

The flag changes how each mbigm big-M LP is solved (over the components the row's
body touches), so its first-order effect is on the *lowering*: its wall time and
its M values. ``issue1917_gdp_flag_panel.py --phase solve`` cannot isolate that,
because ``time_limit`` is charged from the start of ``Model.solve`` and an
unsolved instance spends any lowering time saved on more search. This script
measures the lowering directly. The pre-registered rule that reads its rows is
``docs/dev/data/1917-flag-panel/DISCOPT_MBIGM_LP_COMPONENTS.prereg.md``.

``--phase lower``
    For each instance and rep, lower with ``create_using("gdp.mbigm")`` in a fresh
    subprocess per arm (``=0`` OFF, ``=1`` ON), arms interleaved and their order
    alternated by instance and rep. Each child records the lowering wall, every
    ``_compute_big_m_lp`` return in call order, the exact-LP statuses, and how
    many of those returns fell back to the interval M (``interval_fallback``). A child
    that exceeds ``--timeout`` is killed and recorded as ``killed``.
``--phase audit``
    One ON lowering per instance in which every OPTIMAL component sub-LP is
    re-solved as the *full* LP by HiGHS (``scipy.optimize.linprog``). The sub-LP
    is a relaxation, so its minimum must not exceed the full minimum; a violation
    is an unsound M. Every non-OPTIMAL HiGHS answer is recorded, never skipped.
    At most ``--audit-cap`` sub-LPs per instance are re-solved, in call order;
    the rest are counted in ``skipped_cap``, never silently dropped.

The child also records ``discopt.__file__`` and asserts the flag code is loaded
(CLAUDE.md §8).

    python -u discopt_benchmarks/scripts/issue1917_mbigm_components_lowering.py \\
        --phase lower --reps 2 --out <L>.jsonl
    python -u discopt_benchmarks/scripts/issue1917_mbigm_components_lowering.py \\
        --phase audit --out <A>.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
BENCH = HERE.parents[1]
REPO = HERE.parents[2]
FLAG = "DISCOPT_MBIGM_LP_COMPONENTS"
THREAD_ENVS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

CHILD = r"""
import json, os, sys, time, collections
phase, name, cap = sys.argv[1], sys.argv[2], int(sys.argv[3])
import numpy as np
import discopt, discopt.transformations
import discopt._relax.gdp_reformulate as G
import discopt.solvers.lp_backend as LB
assert hasattr(G, "_component_sub_lp") and hasattr(G, "_lp_box_memo"), "flag code absent"
from benchmarks import gdplib_runner as gr
from discopt.pyomo.gdp import convert_pyomo_gdp
specs = [s for s in gr.discover_models(include=[name]) if s.name == name]
assert len(specs) == 1, (name, [s.name for s in specs])
model = convert_pyomo_gdp(specs[0].builder()).model
out = {"discopt": discopt.__file__}
ms = []
real_bm = G._compute_big_m_lp
real_interval = G._compute_big_m
depth = [0]
fallback = [0]
def interval(*a, **k):
    if depth[0]:
        fallback[0] += 1
    return real_interval(*a, **k)
def bm(*a, **k):
    depth[0] += 1
    try:
        r = real_bm(*a, **k)
    finally:
        depth[0] -= 1
    ms.append(float(r))
    return r
G._compute_big_m_lp = bm
G._compute_big_m = interval
solver = LB.get_exact_lp_solver()
assert solver is not None, "no exact LP oracle"
status = collections.Counter()
audit = {
    "checked": 0,
    "skipped_cap": 0,
    "violations": [],
    "highs_status": collections.Counter(),
    "worst": None,
}
last = {}
if phase == "audit":
    real_sub = G._component_sub_lp
    def sub(lp_data, obj, bl):
        res = real_sub(lp_data, obj, bl)
        if res[1] is not lp_data[0] or res[3] is not lp_data[2]:
            last["full"] = (obj, lp_data[0], lp_data[1], lp_data[2], lp_data[3], bl)
        return res
    G._component_sub_lp = sub
def lp(**kw):
    r = solver(**kw)
    status[str(r.status)] += 1
    full = last.pop("full", None)
    if full is not None and str(r.status) == "SolveStatus.OPTIMAL":
        if sum(audit["highs_status"].values()) >= cap:
            audit["skipped_cap"] += 1
            return r
        from scipy.optimize import linprog
        c, Au, bu, Ae, be, bl = full
        h = linprog(c, A_ub=Au, b_ub=bu, A_eq=Ae, b_eq=be, bounds=bl, method="highs")
        audit["highs_status"][str(h.status)] += 1
        if h.status == 0:
            audit["checked"] += 1
            d = (float(r.objective) - float(h.fun)) / (1.0 + abs(float(h.fun)))
            audit["worst"] = d if audit["worst"] is None else max(audit["worst"], d)
            if d > 1e-9:
                audit["violations"].append([float(r.objective), float(h.fun)])
    return r
LB.get_exact_lp_solver = lambda: lp
t0 = time.perf_counter()
discopt.transformations.create_using("gdp.mbigm", model)
out["lower_s"] = round(time.perf_counter() - t0, 3)
out["n_m"] = len(ms)
out["m"] = [float(x).hex() for x in ms]
out["lp_status"] = dict(status)
out["interval_fallback"] = fallback[0]
if phase == "audit":
    audit["highs_status"] = dict(audit["highs_status"])
    out["audit"] = audit
print("@@RESULT " + json.dumps(out), flush=True)
sys.stdout.flush()
os._exit(0)
"""


def _names(include: str) -> list[str]:
    sys.path.insert(0, str(BENCH))
    from benchmarks import gdplib_runner as gr

    names = list(gr.GDP_PARITY)
    if include:
        keep = set(include.split(","))
        names = [n for n in names if n in keep]
    return names


def _env(on: bool) -> dict:
    env = dict(os.environ)
    env[FLAG] = "1" if on else "0"
    env["PYTHONHASHSEED"] = "0"
    for k in THREAD_ENVS:
        env[k] = "1"
    env["PYTHONPATH"] = f"{REPO / 'python'}{os.pathsep}{BENCH}"
    return env


def _child(phase: str, name: str, on: bool, timeout: float, cap: int) -> dict:
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            [sys.executable, "-u", "-c", CHILD, phase, name, str(cap)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_env(on),
            cwd=str(REPO),
        )
    except subprocess.TimeoutExpired:
        return {"status": "killed", "elapsed": round(time.perf_counter() - t0, 2)}
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("@@RESULT ")]
    if not lines:
        tail = (p.stderr or p.stdout).strip().splitlines()[-3:]
        return {"status": "error", "error": " | ".join(tail)[:400], "returncode": p.returncode}
    rec = json.loads(lines[-1][len("@@RESULT ") :])
    rec["status"] = "lowered"
    rec["elapsed"] = round(time.perf_counter() - t0, 2)
    return rec


def _done(out: Path) -> set:
    if not out.exists():
        return set()
    rows = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
    return {(r["phase"], r["instance"], r["arm"], r["rep"]) for r in rows}


def _wait_for_load(max_load: float | None) -> float:
    load = os.getloadavg()[0]
    waited = 0
    while max_load is not None and load > max_load:
        if waited % 300 == 0:
            print(f"   load gate: load1 {load:.1f} > {max_load:g}, waiting", flush=True)
        time.sleep(30)
        waited += 30
        load = os.getloadavg()[0]
    return load


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("lower", "audit"))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--max-load", type=float, default=None)
    ap.add_argument("--audit-cap", type=int, default=2000)
    ap.add_argument("--include", default="")
    args = ap.parse_args(argv)
    names = _names(args.include)
    done = _done(args.out)
    reps = range(args.reps) if args.phase == "lower" else range(1)
    executed = 0
    for rep in reps:
        for i, name in enumerate(names):
            if args.phase == "lower":
                order = (False, True) if (rep + i) % 2 == 0 else (True, False)
            else:
                order = (True,)
            for on in order:
                arm = "on" if on else "off"
                if (args.phase, name, arm, rep) in done:
                    continue
                load = _wait_for_load(args.max_load)
                rec = _child(args.phase, name, on, args.timeout, args.audit_cap)
                rec.update(
                    phase=args.phase,
                    flag=FLAG,
                    instance=name,
                    arm=arm,
                    rep=rep,
                    load1=round(load, 2),
                    timeout=args.timeout,
                )
                with args.out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                executed += 1
                extra = ""
                if "audit" in rec:
                    a = rec["audit"]
                    extra = (
                        f" checked={a['checked']} capped={a['skipped_cap']} "
                        f"viol={len(a['violations'])}"
                    )
                print(
                    f"{args.phase} rep{rep} {name:34s} {arm:3s} {rec['status']:8s} "
                    f"lower_s={rec.get('lower_s')} n_m={rec.get('n_m')} "
                    f"load={rec['load1']}{extra}",
                    flush=True,
                )
    print(f"executed {executed} children", flush=True)
    return 0 if executed or done else 2


if __name__ == "__main__":
    sys.exit(main())
