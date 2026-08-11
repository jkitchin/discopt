"""#928: hunt the sporadic ON-arm blowup with call-level walls.

Spies MilpRelaxationModel.solve (relaxation-level) and prints any call whose wall
exceeds 2 s together with its time_limit, plus a per-run summary line. Rerun by
the driver until a blowup (wall > 1.8x budget) is caught.

Usage: python issue928_blowup_hunt.py <instance> <budget>
"""

import os
import sys
import time

os.environ["DISCOPT_LP_WARM_DEADLINE"] = "1"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt  # noqa: E402
import discopt._relax.milp_relaxation as MR  # noqa: E402
from discopt._relax.deadline import deadline_scope  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402
from discopt.solver import solve_model  # noqa: E402

assert "/home/user/discopt/python/discopt/" in discopt.__file__, discopt.__file__

name, budget = sys.argv[1], float(sys.argv[2])

calls = []
orig = MR.MilpRelaxationModel.solve


def spy(self, time_limit=None, gap_tolerance=1e-4, backend="auto", *, want_marginals=False):
    t0 = time.perf_counter()
    out = orig(
        self,
        time_limit=time_limit,
        gap_tolerance=gap_tolerance,
        backend=backend,
        want_marginals=want_marginals,
    )
    wall = time.perf_counter() - t0
    calls.append(wall)
    if wall > 2.0 or (time_limit is not None and wall > time_limit + 1.0):
        tag = "VIOLATION" if (time_limit is not None and wall > time_limit + 1.0) else "slow"
        print(
            f"  {tag}: MilpRelaxationModel.solve wall={wall:.2f}s "
            f"time_limit={time_limit} backend={backend} status={out.status}",
            flush=True,
        )
    return out


MR.MilpRelaxationModel.solve = spy

m = from_nl(f"python/tests/data/minlplib_nl/{name}.nl")
t0 = time.perf_counter()
with deadline_scope(budget):
    r = solve_model(m, time_limit=budget)
wall = time.perf_counter() - t0
print(
    f"RUN: wall={wall:.1f}s budget={budget} relax_calls={len(calls)} "
    f"relax_time={sum(calls):.1f}s bound={r.bound}",
    flush=True,
)
if not calls:
    print("PROBE FIRED NOTHING", file=sys.stderr)
    sys.exit(1)
sys.exit(2 if wall > 1.8 * budget else 0)
