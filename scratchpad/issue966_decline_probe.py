"""#966 entry experiment: what does the round-admission check actually decline?

Hypothesis (H1). ``DISCOPT_NODE_ROUND_BUDGET``'s admission check compares the
round's remaining grant against ``expected_build_cost()`` -- an EMA of the COLD
``build_uniform_relaxation`` wall -- for *every* node, including the nodes the
relaxer would serve from its incremental fast path (patch + warm start, orders
of magnitude cheaper than a cold build).  If most node rounds take the fast
path, the check is estimating a cost the round would not have paid, so it
declines rounds that were cheap, losing their bounds for nothing.  That is the
mechanism behind the measured casctanks collapse (2.9098 -> -56.5001, §14b).

Kill criterion: if fewer than 50% of the node rounds in the OFF arm take the
incremental fast path -- i.e. rounds really are cold builds -- H1 is falsified
and over-declining is not an estimator-fidelity problem.

Instrumentation (CLAUDE.md §6: every count is printed; zero counts exit
non-zero).  Declines are derived exactly: under the flag each node-loop
iteration calls ``expected_build_cost()`` once and then either declines or calls
``solve_at_node(round_deadline=...)``, so

    declines = admission_checks - rounds_with_round_deadline

No exception is swallowed anywhere in this file (§7).

Usage:  python -u issue966_decline_probe.py <instance.nl> <budget_s> <0|1>
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

path = sys.argv[1]
budget = float(sys.argv[2])
round_budget = sys.argv[3]
os.environ["DISCOPT_NODE_ROUND_BUDGET"] = round_budget
os.environ.setdefault("DISCOPT_LP_WARM_DEADLINE", "0")
os.environ.setdefault("DISCOPT_HESS_COMPILE_GATE", "0")

import discopt  # noqa: E402
from discopt._relax import mccormick_lp as _mc  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

# §8: prove the tree under test carries the mechanism being probed.
assert "/python/discopt/" in discopt.__file__, discopt.__file__
assert "round_deadline" in _mc.MccormickLPRelaxer.solve_at_node.__code__.co_varnames
assert hasattr(_mc.MccormickLPRelaxer, "expected_build_cost")

STATS = {
    "admission_checks": 0,
    "rounds": 0,
    "rounds_with_round_deadline": 0,
    "fast_path_calls": 0,
    "fast_path_hits": 0,
    "ema_at_check": [],
    "round_walls": [],  # (wall, fast_hit, status, bound)
}

_orig_expected = _mc.MccormickLPRelaxer.expected_build_cost
_orig_solve = _mc.MccormickLPRelaxer.solve_at_node
_orig_fast = _mc.MccormickLPRelaxer._try_incremental_node


def _expected(self):
    v = _orig_expected(self)
    STATS["admission_checks"] += 1
    STATS["ema_at_check"].append(v)
    return v


_FAST_HIT = {"v": False}


def _fast(self, *a, **kw):
    r = _orig_fast(self, *a, **kw)
    STATS["fast_path_calls"] += 1
    _FAST_HIT["v"] = r is not None
    if r is not None:
        STATS["fast_path_hits"] += 1
    return r


def _solve(self, node_lb, node_ub, time_limit=None, **kw):
    _FAST_HIT["v"] = False
    t0 = time.perf_counter()
    res = _orig_solve(self, node_lb, node_ub, time_limit, **kw)
    wall = time.perf_counter() - t0
    STATS["rounds"] += 1
    if kw.get("round_deadline") is not None:
        STATS["rounds_with_round_deadline"] += 1
    STATS["round_walls"].append((round(wall, 4), _FAST_HIT["v"], res.status, res.lower_bound))
    return res


_mc.MccormickLPRelaxer.expected_build_cost = _expected
_mc.MccormickLPRelaxer.solve_at_node = _solve
_mc.MccormickLPRelaxer._try_incremental_node = _fast

m = from_nl(path)
t0 = time.perf_counter()
r = m.solve(time_limit=budget)
wall = time.perf_counter() - t0

walls = STATS["round_walls"]
fast_rounds = sum(1 for w in walls if w[1])
declines = STATS["admission_checks"] - STATS["rounds_with_round_deadline"]
out = {
    "instance": path.split("/")[-1].removesuffix(".nl"),
    "budget": budget,
    "DISCOPT_NODE_ROUND_BUDGET": round_budget,
    "wall": round(wall, 2),
    "status": r.status,
    "bound": r.bound,
    "objective": r.objective,
    "node_count": int(r.node_count or 0),
    "admission_checks": STATS["admission_checks"],
    "rounds": STATS["rounds"],
    "rounds_with_round_deadline": STATS["rounds_with_round_deadline"],
    "declines": declines,
    "fast_path_calls": STATS["fast_path_calls"],
    "fast_path_hits": STATS["fast_path_hits"],
    "rounds_taking_fast_path": fast_rounds,
    "fast_round_fraction": (fast_rounds / len(walls)) if walls else None,
    "build_ema_last": STATS["ema_at_check"][-1] if STATS["ema_at_check"] else None,
    "round_wall_total": round(sum(w[0] for w in walls), 3),
    "round_wall_max": max((w[0] for w in walls), default=None),
    "round_walls_head": walls[:12],
    "round_walls_tail": walls[-6:],
}
print(json.dumps(out, indent=2))

# §6: the probe must prove it measured something.
if STATS["rounds"] == 0:
    print("PROBE MEASURED NOTHING: zero solve_at_node rounds", file=sys.stderr)
    sys.exit(2)
if round_budget == "1" and STATS["admission_checks"] == 0:
    print("PROBE MEASURED NOTHING: flag ON but zero admission checks", file=sys.stderr)
    sys.exit(3)
print(
    f"COUNTS_EXECUTED rounds={STATS['rounds']} admission_checks={STATS['admission_checks']} "
    f"declines={declines} fast_hits={STATS['fast_path_hits']}"
)
