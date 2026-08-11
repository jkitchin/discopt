"""#966 entry experiment 4: yield the round instead of declining it.

Three arms on the same instance/budget, each a fresh process:

* ``base``    -- flag off (the reference bound/incumbent/nodes);
* ``decline`` -- every round declined (``expected_build_cost`` -> 1e9), i.e. the
  flag's current behaviour taken to its limit;
* ``yield``   -- every round *yielded* instead: the cold build is fully truncated
  (``build_deadline`` already spent, the #694 anytime mechanism) and the
  separation chain is skipped, so the round costs ~a base-LP solve and still
  returns a valid weaker bound AND an LP vertex for branching.

Hypothesis: the ``decline`` arm's damage (nvs05: bound 3.514 -> 0.684, incumbent
8.73 -> 523.69, nodes 39 -> 1, and 16 s of the budget left unused) is caused by
the round banking *nothing* -- no bound and, just as important, no LP point for
the spatial brancher and the primal heuristics.  A yielded round banks both at a
fraction of the cost.

Kill criterion: if ``yield`` is no better than ``decline`` on bound/incumbent, a
yielded round is not a usable floor and the fix must look elsewhere.

Counted output (CLAUDE.md §6); exits non-zero if no arm triple ran.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

WORKER = r"""
import json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
import discopt
from discopt._relax import mccormick_lp as mc
from discopt.modeling.core import from_nl

assert "/python/discopt/" in discopt.__file__, discopt.__file__
assert "round_deadline" in mc.MccormickLPRelaxer.solve_at_node.__code__.co_varnames

path, budget, arm = sys.argv[1], float(sys.argv[2]), sys.argv[3]
N = {"checks": 0, "rounds": 0, "yielded": 0}

_orig_expected = mc.MccormickLPRelaxer.expected_build_cost
_orig_solve = mc.MccormickLPRelaxer.solve_at_node

def _expected(self):
    N["checks"] += 1
    return 1e9 if arm == "decline" else _orig_expected(self)

def _solve(self, node_lb, node_ub, time_limit=None, **kw):
    if kw.get("round_deadline") is not None:
        N["rounds"] += 1
        if arm == "yield":
            N["yielded"] += 1
            kw["separate"] = False
            kw["build_deadline"] = time.perf_counter()
    return _orig_solve(self, node_lb, node_ub, time_limit, **kw)

mc.MccormickLPRelaxer.expected_build_cost = _expected
mc.MccormickLPRelaxer.solve_at_node = _solve

m = from_nl(path)
t0 = time.perf_counter()
r = m.solve(time_limit=budget)
print(json.dumps({
    "arm": arm,
    "wall": round(time.perf_counter() - t0, 2),
    "status": r.status,
    "bound": r.bound,
    "objective": r.objective,
    "gap_certified": bool(r.gap_certified),
    "node_count": int(r.node_count or 0),
    "checks": N["checks"], "rounds": N["rounds"], "yielded": N["yielded"],
}))
"""

worker = os.path.join(HERE, "_issue966_yield_worker.py")
with open(worker, "w") as fh:
    fh.write(WORKER)


def run(inst: str, budget: float, arm: str) -> dict:
    env = dict(os.environ)
    env["DISCOPT_NODE_ROUND_BUDGET"] = "0" if arm == "base" else "1"
    env["DISCOPT_LP_WARM_DEADLINE"] = os.environ.get("DISCOPT_LP_WARM_DEADLINE", "0")
    env["DISCOPT_HESS_COMPILE_GATE"] = "0"
    out = subprocess.run(
        [
            sys.executable,
            "-u",
            worker,
            f"python/tests/data/minlplib_nl/{inst}.nl",
            str(budget),
            arm,
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


budget = float(os.environ.get("PROBE_BUDGET", "20"))
triples = 0
for inst in sys.argv[1:]:
    # Interleaved within the instance (CLAUDE.md §9), one process per cell.
    rows = [run(inst, budget, arm) for arm in ("base", "decline", "yield")]
    triples += 1
    for r in rows:
        print(
            f"{inst:12s} {r['arm']:8s} bound={r['bound']!r:24s} obj={r['objective']!r:22s} "
            f"cert={r['gap_certified']!s:5s} nodes={r['node_count']:5d} wall={r['wall']:6.2f} "
            f"rounds={r['rounds']} yielded={r['yielded']} status={r['status']}",
            flush=True,
        )

print(f"ARM_TRIPLES_EXECUTED={triples}")
if triples == 0:
    sys.exit(2)
