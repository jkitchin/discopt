"""Where does the contvar bound go when BOTH #928 and #966 clamps are on?

§14b's verdict names the suspect as an *interaction*: "LP yields on its deadline"
x "round yields on its deadline" compounding into a node result that carries no
adoptable bound. This probe instruments every bound-producing seam of one solve
and prints what each one returned, so the loss has a call site rather than a
theory.

Usage:  DISCOPT_LP_WARM_DEADLINE=1 DISCOPT_NODE_ROUND_BUDGET=1 \
        DISCOPT_HESS_COMPILE_GATE=1 python -u issue928_round_floor_probe.py contvar 20

Prints OBSERVATIONS_EXECUTED and exits non-zero when it observed nothing
(CLAUDE.md §6). Nothing is wrapped in a bare ``except`` (§7).
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt  # noqa: E402
from discopt import solver as _solver  # noqa: E402
from discopt._relax import mccormick_lp as _mc  # noqa: E402
from discopt._relax import milp_relaxation as _mr  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

assert "/python/discopt/" in discopt.__file__, discopt.__file__
assert "round_deadline" in _mc.MccormickLPRelaxer.solve_at_node.__code__.co_varnames

ROOT = Path(__file__).resolve().parents[1]
name, budget = sys.argv[1], float(sys.argv[2])

fired = Counter()
t0 = time.perf_counter()


def _stamp() -> float:
    return time.perf_counter() - t0


_orig_node = _mc.MccormickLPRelaxer.solve_at_node


def _node(self, lb, ub, time_limit=None, **kw):
    t = time.perf_counter()
    r = _orig_node(self, lb, ub, time_limit, **kw)
    fired["solve_at_node"] += 1
    rd = kw.get("round_deadline")
    print(
        f"[{_stamp():6.2f}] solve_at_node tl={time_limit} "
        f"round_left={None if rd is None else round(rd - t, 3)} "
        f"-> status={r.status} lb={r.lower_bound} wall={time.perf_counter() - t:.2f}",
        flush=True,
    )
    return r


_orig_lp = _mr.MilpRelaxationModel.solve


def _lp(self, time_limit=None, gap_tolerance=1e-4, backend="auto", **kw):
    t = time.perf_counter()
    r = _orig_lp(self, time_limit, gap_tolerance, backend, **kw)
    fired["milp_solve"] += 1
    fired[f"milp_solve:{backend}:{r.status}"] += 1
    if r.status != "optimal":
        print(
            f"[{_stamp():6.2f}]   lp.solve backend={backend} tl={time_limit} "
            f"-> status={r.status} bound={r.bound} wall={time.perf_counter() - t:.2f}",
            flush=True,
        )
    return r


_orig_fb = _solver._root_relaxation_lower_bound


def _fb(*a, **kw):
    t = time.perf_counter()
    r = _orig_fb(*a, **kw)
    fired["root_fallback"] += 1
    print(
        f"[{_stamp():6.2f}] ROOT FALLBACK -> {r} (wall={time.perf_counter() - t:.2f})",
        flush=True,
    )
    return r


_mc.MccormickLPRelaxer.solve_at_node = _node
_mr.MilpRelaxationModel.solve = _lp
_solver._root_relaxation_lower_bound = _fb

m = from_nl(str(ROOT / "python/tests/data/minlplib_nl" / f"{name}.nl"))
res = m.solve(time_limit=budget)
print(
    f"\nRESULT status={res.status} bound={res.bound} obj={res.objective} "
    f"nodes={res.node_count} wall={res.wall_time:.1f}"
)
print("counters:", dict(fired))
print(f"OBSERVATIONS_EXECUTED={fired['solve_at_node'] + fired['milp_solve']}")
raise SystemExit(0 if (fired["solve_at_node"] + fired["milp_solve"]) else 1)
