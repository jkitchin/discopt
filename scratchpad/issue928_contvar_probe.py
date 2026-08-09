"""#928: where does contvar's sporadic ON-arm blowup spend its time?

Spies on solve_lp_warm_std (per-call time_limit vs wall — a call whose wall
exceeds its passed time_limit by >1s is a DEADLINE VIOLATION) and on
MilpRelaxationModel.solve (caller-level budgets). Prints per-call lines only for
slow calls, plus a summary. CLAUDE.md §6: exits non-zero if no LP call was seen.

Usage: python issue928_contvar_probe.py <instance> <budget>
"""

import os
import sys
import time

os.environ["DISCOPT_LP_WARM_DEADLINE"] = "1"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt  # noqa: E402
import discopt.solvers.milp_simplex as MS  # noqa: E402
from discopt._jax.deadline import deadline_scope  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402
from discopt.solver import solve_model  # noqa: E402

assert "/home/user/discopt/python/discopt/" in discopt.__file__, discopt.__file__

name, budget = sys.argv[1], float(sys.argv[2])

calls = []
violations = 0
orig = MS.solve_lp_warm_std


def spy(c, A_ub, b_ub, bounds, in_basis=None, *, return_cert=False, time_limit=None):
    global violations
    t0 = time.perf_counter()
    out = orig(c, A_ub, b_ub, bounds, in_basis, return_cert=return_cert, time_limit=time_limit)
    wall = time.perf_counter() - t0
    calls.append((wall, time_limit, in_basis is not None))
    if time_limit is not None and wall > time_limit + 1.0:
        violations += 1
        print(
            f"DEADLINE VIOLATION: wall={wall:.2f}s vs time_limit={time_limit:.2f}s "
            f"warm={in_basis is not None}",
            flush=True,
        )
    elif wall > 2.0:
        print(f"slow call: wall={wall:.2f}s time_limit={time_limit} warm={in_basis is not None}", flush=True)
    return out


MS.solve_lp_warm_std = spy

m = from_nl(f"python/tests/data/minlplib_nl/{name}.nl")
t0 = time.perf_counter()
with deadline_scope(budget):
    r = solve_model(m, time_limit=budget)
wall = time.perf_counter() - t0
tot = sum(w for w, _, _ in calls)
print(
    f"{name}: wall={wall:.1f}s (budget {budget}) lp_calls={len(calls)} "
    f"lp_time={tot:.1f}s violations={violations} bound={r.bound}"
)
if not calls:
    print("PROBE FIRED NOTHING", file=sys.stderr)
    sys.exit(1)
