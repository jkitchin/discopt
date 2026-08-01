"""Neutrality-check worker that runs on EITHER tree.

CLAUDE.md §8: the marker is asserted in both directions — ``expect_marker=1``
requires ``_extend_budget_for_incumbent`` to be present, ``0`` requires it absent —
so a baseline run that silently imported the changed tree fails instead of
producing a bogus "identical" verdict.

Usage: python issue917_neutral_worker.py <path.nl> <time_limit> <expect_marker:0|1>
"""

import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt  # noqa: E402
from discopt import solver as _solver  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

path, budget, expect = sys.argv[1], float(sys.argv[2]), sys.argv[3]
assert "/python/discopt/" in discopt.__file__, discopt.__file__
present = hasattr(_solver, "_extend_budget_for_incumbent")
assert present == (expect == "1"), (
    f"marker mismatch: _extend_budget_for_incumbent present={present}, expected={expect == '1'}"
)

m = from_nl(path)
t0 = time.perf_counter()
r = m.solve(time_limit=budget)
print(
    json.dumps(
        {
            "instance": path.split("/")[-1].removesuffix(".nl"),
            "wall": time.perf_counter() - t0,
            "status": r.status,
            "objective": r.objective,
            "bound": r.bound,
            "node_count": int(r.node_count or 0),
            "gap_certified": bool(r.gap_certified),
        }
    )
)
