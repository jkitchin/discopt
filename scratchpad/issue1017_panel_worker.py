"""#1017 Farkas-margin A/B panel worker; runs on EITHER tree.

CLAUDE.md §8: the tree under test is asserted in BOTH directions before any
measurement. The change is in the Rust extension, so the marker is a *runtime*
one — the crafted feasible cancellation LP of `test_1017_farkas_cancellation_margin`
run through `solve_lp_py`: the pre-fix margin fathoms it (`infeasible`), the fixed
margin does not. A baseline run that silently loaded the fixed extension (or the
reverse) fails here instead of producing a bogus "identical" verdict.

Usage: python issue1017_panel_worker.py <path.nl> <time_limit> <expect_marker:0|1>
Emits one JSON line on stdout.
"""

import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
from discopt._rust import profile_counters_py, profile_reset_py, solve_lp_py  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

path, budget, expect = sys.argv[1], float(sys.argv[2]), sys.argv[3]

# --- runtime marker -------------------------------------------------------
b = np.array([1e16] + [1.5] * 8 + [-1e16, -12.0])
m = b.size
n = m - 1
a = np.zeros((m, n))
for j in range(n):
    a[j, j] = 1.0
    a[j + 1, j] = -1.0
w = 4.0 * (1e16 + 12.0)
status, _x, _obj, _it = solve_lp_py(np.zeros(n), a, b, np.full(n, -w), np.full(n, w))
fixed = status != "infeasible"
assert fixed == (expect == "1"), (
    f"marker mismatch: cancellation LP -> {status!r} (fixed={fixed}), "
    f"expected fixed={expect == '1'}"
)

profile_reset_py()
model = from_nl(path)
t0 = time.perf_counter()
r = model.solve(time_limit=budget)
wall = time.perf_counter() - t0
ctr = profile_counters_py()
print(
    json.dumps(
        {
            "instance": path.split("/")[-1].removesuffix(".nl"),
            "wall": wall,
            "status": r.status,
            "objective": r.objective,
            "bound": r.bound,
            "node_count": int(r.node_count or 0),
            "gap_certified": bool(r.gap_certified),
            "farkas_reject_margin": int(ctr.get("FarkasRejectMargin", 0)),
            "farkas_reject_open": int(ctr.get("FarkasRejectOpen", 0)),
            "farkas_reject_cancellation": int(ctr.get("FarkasRejectCancellation", 0)),
            "lp_infeasible": int(ctr.get("LpVerdictInfeasible", 0)),
            "warm_infeasible": int(ctr.get("WarmVerdictInfeasible", 0)),
        }
    ),
    flush=True,
)
