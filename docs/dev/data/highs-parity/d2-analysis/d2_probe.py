"""D2 analysis probe: one Rust-route solve per (instance, env) in a fresh subprocess.

Records every driver call (status / nodes / obj / bound / wall / lp iters) through a spy
on ``discopt._rust.solve_milp_csc_py`` and, with ``DISCOPT_PROFILE=1``, the phase tables
the driver dumps to stderr (one table per driver call). Prints an executed count and
exits non-zero when nothing ran (CLAUDE.md §6). Read-only: nothing in the tree changes.

Usage: python d2_probe.py --out results.jsonl --time-limit 20 --jobs 2 JOB [JOB ...]
  JOB = name[:KEY=VAL,KEY=VAL]   (extra env for that solve)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path("/home/user/discopt")
SCRIPTS = REPO / "discopt_benchmarks" / "scripts"
MARKER = "d2-probe-2026-11-06"

CHILD = r"""
import json, os, sys, tempfile, time
sys.path.insert(0, sys.argv[2])
import lp_milp_loader as L
import discopt
assert discopt.__file__.startswith("/home/user/discopt/python/"), discopt.__file__
from discopt import _rust
from discopt.modeling.core import from_nl
inst = L.read_instance(sys.argv[1])
with tempfile.TemporaryDirectory() as td:
    nl = os.path.join(td, "m.nl"); perm = L.write_nl(inst, nl); m = from_nl(nl)
calls = []
real = _rust.solve_milp_csc_py
def spy(*a, **kw):
    t0 = time.perf_counter()
    out = real(*a, **kw)
    calls.append({"status": out[0], "obj": float(out[2]), "bound": float(out[3]),
                  "nodes": int(out[4]), "iters": int(out[5]), "wall": time.perf_counter() - t0,
                  "time_limit_s": kw.get("time_limit_s"), "max_nodes": int(a[12]),
                  "max_open_nodes": kw.get("max_open_nodes"),
                  "seeded": kw.get("initial_incumbent") is not None,
                  "kw": sorted(k for k in kw if k not in ("initial_incumbent", "debug_hook"))})
    return out
_rust.solve_milp_csc_py = spy
t0 = time.perf_counter()
r = m.solve(time_limit=float(sys.argv[3]), gap_tolerance=float(sys.argv[4]))
wall = time.perf_counter() - t0
import numpy as np
viol = None
if r.x is not None:
    x = np.empty(inst.n)
    for k in range(inst.n):
        x[perm[k]] = float(np.asarray(r.x[f"x{k}"], float).ravel()[0])
    viol = float(L.dense_max_violation(inst, x))
print("RESULT " + json.dumps({"status": r.status, "objective": r.objective, "bound": r.bound,
    "gap_certified": bool(r.gap_certified), "nodes": int(r.node_count), "wall": wall,
    "termination": None if r.termination is None else str(r.termination),
    "route": r.algorithm_route, "max_viol": viol, "calls": calls,
    "n": inst.n, "m": inst.m, "n_int": int(inst.is_int.sum())}))
"""

PHASE = re.compile(r"^\s+(\w+)\s+(\d+) calls\s+([\d.]+) ms$")
CTR = re.compile(r"^\s+(\w+)\s+(\d+)$")


def instance_path(name: str) -> Path:
    for p in (REPO / "ref" / "miplib2017" / f"{name}.mps.gz",
              REPO / "ref" / "HiGHS" / "check" / "instances" / f"{name}.mps",
              REPO / "ref" / "HiGHS" / "check" / "instances" / f"{name}.lp"):
        if p.exists():
            return p
    raise FileNotFoundError(name)


def oracle(name: str) -> tuple[float | None, bool]:
    for mf in (REPO / "discopt_benchmarks/data/miplib/panel.json",
               REPO / "discopt_benchmarks/data/lp_milp/manifest.json"):
        for r in json.loads(mf.read_text())["members"]:
            if r["name"] == name:
                return r.get("oracle_objective"), False
    return None, False


def run_job(job: str, time_limit: float, gap: float) -> dict:
    name, _, envs = job.partition(":")
    env = dict(os.environ)
    env["DISCOPT_LP_MILP_BACKEND"] = "rust"
    env["DISCOPT_PROFILE"] = "1"
    extra = {}
    for kv in filter(None, envs.split(",")):
        k, _, v = kv.partition("=")
        env[k] = v
        extra[k] = v
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-u", "-c", CHILD, str(instance_path(name)), str(SCRIPTS),
                        str(time_limit), str(gap)], env=env, capture_output=True, text=True,
                       timeout=time_limit * 6 + 120)
    row = {"name": name, "env": extra, "marker": MARKER, "wall_total": time.perf_counter() - t0,
           "rc": p.returncode}
    res = None
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            res = json.loads(line[7:])
    if res is None:
        row["error"] = (p.stderr[-3000:] + p.stdout[-1000:])
        return row
    row.update(res)
    # Profile tables: one per driver call, in order.
    tables, cur = [], None
    for line in p.stderr.splitlines():
        if line.startswith("--- MILP phase profile"):
            cur = {"phases": {}, "counters": {}}
            tables.append(cur)
        elif cur is not None:
            mp = PHASE.match(line)
            if mp:
                cur["phases"][mp.group(1)] = (int(mp.group(2)), float(mp.group(3)))
                continue
            mc = CTR.match(line)
            if mc and int(mc.group(2)) != 0:
                cur["counters"][mc.group(1)] = int(mc.group(2))
    row["profile"] = tables
    opt, _ = oracle(name)
    row["opt"] = opt
    if opt is not None:
        ob, bd = res["objective"], res["bound"]
        row["primal_gap"] = None if ob is None else abs(ob - opt) / (1 + abs(opt))
        row["dual_gap"] = None if bd is None else abs(opt - bd) / (1 + abs(opt))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--gap", type=float, default=1e-4)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("job", nargs="+")
    a = ap.parse_args()
    executed = 0
    with open(a.out, "a") as f, ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for row in ex.map(lambda j: run_job(j, a.time_limit, a.gap), a.job):
            f.write(json.dumps(row) + "\n")
            f.flush()
            executed += 1
            s = row.get("status", "ERR")
            print(f"{row['name']:22s} {row['env']} -> {s} cert={row.get('gap_certified')} "
                  f"nodes={row.get('nodes')} wall={row.get('wall', 0):.1f} "
                  f"p={row.get('primal_gap')} d={row.get('dual_gap')} "
                  f"calls={[(c['status'], c['nodes'], round(c['wall'], 1)) for c in row.get('calls', [])]}"
                  + (f"\n   ERROR {row['error'][-600:]}" if 'error' in row else ''), flush=True)
    print(f"executed {executed}")
    return 0 if executed else 1


if __name__ == "__main__":
    sys.exit(main())
