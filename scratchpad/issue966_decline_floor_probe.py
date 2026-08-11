"""#966 entry experiment 3: what does a DECLINED round cost the certificate?

The round-budget flag's decline branch is documented as "the node stays open on
its valid parent bound".  That is true only when the node's slot does not
already hold the failure sentinel (``_INFEASIBILITY_SENTINEL`` = 1e30, written
when the per-node NLP failed/diverged).  When it does, the declined round is the
only thing that would have replaced the sentinel with a real bound, so the node
is left *fathomed without proof* -- ``_node_decertifies`` fires and the whole
run's dual bound is discarded (the panel's contvar ``183632.766 -> None``).

This probe forces EVERY round to be declined (``expected_build_cost`` -> 1e9,
which is what the admission check reads) and compares against the same solve
with the flag off.

Kill criterion: if the forced-decline arm still reports a finite CERTIFIED bound
on every instance, a decline does not cost the certificate and this hypothesis
is falsified.

Counted output; exits non-zero if no arm pair ran (CLAUDE.md §6).
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

path, budget, force = sys.argv[1], float(sys.argv[2]), sys.argv[3] == "1"

DECLINED = {"checks": 0, "rounds": 0}
_orig_expected = mc.MccormickLPRelaxer.expected_build_cost
_orig_solve = mc.MccormickLPRelaxer.solve_at_node

def _expected(self):
    DECLINED["checks"] += 1
    return 1e9 if force else _orig_expected(self)

def _solve(self, *a, **kw):
    if kw.get("round_deadline") is not None:
        DECLINED["rounds"] += 1
    return _orig_solve(self, *a, **kw)

mc.MccormickLPRelaxer.expected_build_cost = _expected
mc.MccormickLPRelaxer.solve_at_node = _solve

m = from_nl(path)
t0 = time.perf_counter()
r = m.solve(time_limit=budget)
print(json.dumps({
    "wall": round(time.perf_counter() - t0, 2),
    "status": r.status,
    "bound": r.bound,
    "objective": r.objective,
    "gap_certified": bool(r.gap_certified),
    "node_count": int(r.node_count or 0),
    "admission_checks": DECLINED["checks"],
    "rounds_run": DECLINED["rounds"],
    "declines": DECLINED["checks"] - DECLINED["rounds"],
}))
"""

worker_path = os.path.join(HERE, "_issue966_decline_worker.py")
with open(worker_path, "w") as fh:
    fh.write(WORKER)


def run(inst: str, budget: float, flag: str, force: str) -> dict:
    env = dict(os.environ)
    env["DISCOPT_NODE_ROUND_BUDGET"] = flag
    env["DISCOPT_LP_WARM_DEADLINE"] = os.environ.get("DISCOPT_LP_WARM_DEADLINE", "0")
    env["DISCOPT_HESS_COMPILE_GATE"] = "0"
    out = subprocess.run(
        [
            sys.executable,
            "-u",
            worker_path,
            f"python/tests/data/minlplib_nl/{inst}.nl",
            str(budget),
            force,
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


budget = float(os.environ.get("PROBE_BUDGET", "20"))
pairs = 0
for inst in sys.argv[1:]:
    base = run(inst, budget, "0", "0")
    forced = run(inst, budget, "1", "1")
    pairs += 1
    print(
        f"{inst:12s} base   bound={base['bound']!r} cert={base['gap_certified']} "
        f"nodes={base['node_count']} wall={base['wall']}\n"
        f"{'':12s} forced bound={forced['bound']!r} cert={forced['gap_certified']} "
        f"nodes={forced['node_count']} wall={forced['wall']} "
        f"declines={forced['declines']}/{forced['admission_checks']}",
        flush=True,
    )

print(f"ARM_PAIRS_EXECUTED={pairs}")
if pairs == 0:
    sys.exit(2)
