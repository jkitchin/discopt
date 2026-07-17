"""#282 entry experiment: does the improver-LNS spend starve the syn/rsyn tree?

H1: on syn/rsyn, the improver-role heuristics (RINS + local branching) consume the
    bulk of the NLP-BB budget and still leave the incumbent far from the optimum;
    capping their contingent converts budget into nodes and a tighter dual bound.

A = default (DISCOPT_HEUR_QUOT=0.5). B = DISCOPT_HEUR_QUOT=0 (improvers blocked once
an incumbent exists; note _improver_allowed is never gated *before* the first
incumbent, so B still secures one).

KILL H1 if B does not materially increase node_count AND tighten the dual bound,
or if B's incumbent regresses (heuristics were in fact paying for themselves).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

CONVEX = ["rsyn0805m", "rsyn0810m", "rsyn0815m", "syn40m"]
NONCONVEX = ["syn30hfsg", "syn40hfsg", "syn15m02hfsg"]

CHILD = r"""
import os, json, sys
os.environ.setdefault("JAX_PLATFORMS","cpu"); os.environ.setdefault("JAX_ENABLE_X64","1")
from pathlib import Path
from discopt.modeling.core import from_nl
name, budget = sys.argv[1], float(sys.argv[2])
nl = Path.home()/"Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"/(name+".nl")
m = from_nl(str(nl))
r = m.solve(time_limit=budget, gap_tolerance=1e-4)
def f(x):
    try:
        v=float(x)
        return v if abs(v)<1e29 else None
    except (TypeError, ValueError):
        return None
print("@@@"+json.dumps({
  "instance":name,"status":r.status,"objective":f(r.objective),"bound":f(r.bound),
  "node_count":r.node_count,"root_time":f(r.root_time),"root_bound":f(r.root_bound),
  "wall":f(r.wall_time),"jax":f(r.jax_time),"python":f(r.python_time),
  "gap_certified":bool(r.gap_certified),
}))
"""


def run(name, budget, quot):
    env = dict(os.environ)
    env["DISCOPT_HEUR_QUOT"] = str(quot)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env["JAX_ENABLE_X64"] = "1"
    p = subprocess.run(
        [sys.executable, "-c", CHILD, name, str(budget)],
        capture_output=True,
        text=True,
        env=env,
        timeout=budget + 300,
    )
    for line in p.stdout.splitlines():
        if line.startswith("@@@"):
            return json.loads(line[3:])
    return {"instance": name, "error": (p.stderr or p.stdout)[-400:]}


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    names = CONVEX + NONCONVEX
    rows = []
    for name in names:
        for arm, quot in (("A_default", "0.5"), ("B_noimprover", "0")):
            r = run(name, budget, quot)
            r["arm"] = arm
            r["budget"] = budget
            rows.append(r)
            print(
                f"{name:14s} {arm:13s} obj={r.get('objective')} bound={r.get('bound')} "
                f"nodes={r.get('node_count')} cert={r.get('gap_certified')}",
                flush=True,
            )
    out = Path(__file__).with_name(f"heur_ab_{int(budget)}s.json")
    out.write_text(json.dumps(rows, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
