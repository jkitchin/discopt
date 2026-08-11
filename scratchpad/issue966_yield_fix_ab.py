"""#966 verification: the shipped yield-mode fix, A/B against the pre-fix decline.

Three arms per instance, one process each, interleaved (CLAUDE.md §9):

* ``base``    -- ``DISCOPT_NODE_ROUND_BUDGET=0`` (the reference);
* ``decline`` -- the flag ON with yield mode monkeypatched back OFF (i.e. the
  merged pre-fix behaviour: a short grant skips the round entirely);
* ``yield``   -- the flag ON as shipped.

Every arm forces the admission check to fire on every round
(``expected_build_cost`` -> 1e9) so the branch under test is exercised
deterministically instead of only in the budget's last seconds -- this is a
mechanism differential, not a wall-time panel.

Counted output (§6): the yield arm must report a non-zero yield count, else the
probe exits non-zero rather than reporting a vacuous pass.

Honest limit of the ``decline`` arm: it emulates the merged behaviour by returning
a bound-less result from ``solve_at_node`` rather than by the node loop's ``continue``,
so the node still passes through ``_yield_keeps_node_open``. The arm therefore
isolates the *banking* half of the fix (bound + LP point) and already carries its
certificate half -- which makes the measured yield-vs-decline gap a LOWER bound on
the fix's effect, never an inflated one.
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
assert "yield_round" in mc.MccormickLPRelaxer.solve_at_node.__code__.co_varnames, \
    "#966 yield-mode marker absent from solve_at_node"

path, budget, arm = sys.argv[1], float(sys.argv[2]), sys.argv[3]
N = {"checks": 0, "rounds": 0}
RELAXERS = {}

_orig_expected = mc.MccormickLPRelaxer.expected_build_cost
_orig_solve = mc.MccormickLPRelaxer.solve_at_node

def _expected(self):
    N["checks"] += 1
    RELAXERS[id(self)] = self
    return 1e9 if arm != "base" else _orig_expected(self)

def _solve(self, node_lb, node_ub, time_limit=None, **kw):
    if kw.get("round_deadline") is not None:
        N["rounds"] += 1
    if arm == "decline" and kw.pop("yield_round", False):
        # Pre-fix behaviour: a grant too short for a full round skipped it.
        return mc.MccormickLPResult(status="time_limit")
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
    "checks": N["checks"], "rounds": N["rounds"],
    "yield_rounds": sum(getattr(x, "yield_rounds", 0) for x in RELAXERS.values()),
}))
"""

worker = os.path.join(HERE, "_issue966_yield_fix_worker.py")
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
yields_seen = 0
for inst in sys.argv[1:]:
    rows = [run(inst, budget, arm) for arm in ("base", "decline", "yield")]
    triples += 1
    for r in rows:
        if r["arm"] == "yield":
            yields_seen += r["yield_rounds"]
        print(
            f"{inst:12s} {r['arm']:8s} bound={r['bound']!r:24s} obj={r['objective']!r:22s} "
            f"cert={r['gap_certified']!s:5s} nodes={r['node_count']:5d} wall={r['wall']:6.2f} "
            f"rounds={r['rounds']} yield_rounds={r['yield_rounds']} status={r['status']}",
            flush=True,
        )

print(f"ARM_TRIPLES_EXECUTED={triples} YIELD_ROUNDS_OBSERVED={yields_seen}")
if triples == 0 or yields_seen == 0:
    sys.exit(2)
