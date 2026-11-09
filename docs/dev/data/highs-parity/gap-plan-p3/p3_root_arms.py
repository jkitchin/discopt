"""Gap plan P3: the route's root bound with the driver's cut families switched per arm.

Patches ``discopt._rust.solve_milp_csc_py`` to override cut kwargs (the
``d2_probe.py`` spy pattern), solves root only (``max_nodes=2``) in a fresh process,
and reports the first driver call's bound. Arms mirror the SCIP root ablation:
  none -- root_cuts=0, cut_rounds=0 (the root LP after the driver's presolve)
  gmi  -- the route's default root loop (GMI + covers)
  cmir -- c-MIR only (gmi_cuts=False, cmir_cuts=True), same budget
  both -- GMI + c-MIR
Prints an executed count; exits non-zero on 0 (CLAUDE.md §6).
Usage: python p3_root_arms.py --instances a,b [--arms none,gmi,cmir,both] --out X.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
SCRIPTS = REPO / "discopt_benchmarks" / "scripts"
sys.path.insert(0, str(SCRIPTS))
import p42_rens_panel as R  # noqa: E402, N812

ARMS = {
    "none": {"root_cuts": 0, "cut_rounds": 0},
    "gmi": {},
    "cmir": {"gmi_cuts": False, "cmir_cuts": True},
    "both": {"cmir_cuts": True},
}

CHILD = r"""
import json, os, sys, tempfile
sys.path.insert(0, sys.argv[2])
import lp_milp_loader as L
import discopt
assert discopt.__file__.startswith(sys.argv[4]), discopt.__file__
from discopt import _rust
from discopt.modeling.core import from_nl
over = json.loads(sys.argv[3])
inst = L.read_instance(sys.argv[1])
with tempfile.TemporaryDirectory() as td:
    nl = os.path.join(td, "m.nl"); L.write_nl(inst, nl); m = from_nl(nl)
calls = []
real = _rust.solve_milp_csc_py
def spy(*a, **kw):
    kw.update(over)
    out = real(*a, **kw)
    calls.append({"status": out[0], "bound": float(out[3]), "nodes": int(out[4])})
    return out
_rust.solve_milp_csc_py = spy
r = m.solve(time_limit=120.0, gap_tolerance=1e-4, max_nodes=2)
print("RESULT " + json.dumps({"calls": calls, "bound": r.bound}))
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--arms", default="none,gmi,cmir,both")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pop = {m["name"]: m for m in R.population()}
    env = dict(os.environ, DISCOPT_LP_MILP_BACKEND="rust", PYTHONUNBUFFERED="1")
    env.update(dict(kv.split("=", 1) for kv in a.env))
    executed = 0
    with open(a.out, "w") as fh:
        for nm in a.instances.split(","):
            mem = pop[nm]
            opt = mem["oracle_objective"]
            for arm in a.arms.split(","):
                p = subprocess.run(
                    [sys.executable, "-c", CHILD, str(REPO / mem["path"]), str(SCRIPTS),
                     json.dumps(ARMS[arm]), str(REPO / "python")],
                    capture_output=True, text=True, env=env, timeout=900,
                )  # fmt: skip
                line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
                if p.returncode != 0 or not line:
                    raise RuntimeError(f"{nm}/{arm}: rc={p.returncode}\n{p.stderr[-2000:]}")
                res = json.loads(line[-1][7:])
                b = res["calls"][0]["bound"] if res["calls"] else None
                gap = None if b is None else 100.0 * abs(opt - b) / max(1.0, abs(opt))
                fh.write(json.dumps({"name": nm, "arm": arm, "opt": opt, "root_bound": b,
                                     "root_gap_pct": gap, **res}) + "\n")  # fmt: skip
                fh.flush()
                executed += 1
                print(f"{nm:22} {arm:5} root_bound={b} gap={gap}", flush=True)
    print(f"executed: {executed}", flush=True)
    return 0 if executed else 1


if __name__ == "__main__":
    sys.exit(main())
