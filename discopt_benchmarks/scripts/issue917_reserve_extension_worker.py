"""#917 panel worker: one instance, one budget, one process.

Reads ``DISCOPT_LP_SPATIAL_RESERVE_EXTENSION`` from the environment (the panel sets
it) and reports the full result record as JSON on stdout. Exceptions propagate:
a broken arm must fail the panel, never read as a clean run (CLAUDE.md §7).
"""

from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt  # noqa: E402
from discopt import solver as _solver  # noqa: E402
from discopt.modeling.core import ObjectiveSense, from_nl  # noqa: E402

# CLAUDE.md §8: prove which tree is under test before measuring anything with it.
assert "/python/discopt/" in discopt.__file__, discopt.__file__
assert hasattr(_solver, "_extend_budget_for_incumbent"), (
    "#917 marker '_extend_budget_for_incumbent' absent — this is not the tree under test"
)

path, budget = sys.argv[1], float(sys.argv[2])
flag = os.environ.get("DISCOPT_LP_SPATIAL_RESERVE_EXTENSION", "0")

m = from_nl(path)
sense = "max" if m._objective.sense == ObjectiveSense.MAXIMIZE else "min"

t0 = time.perf_counter()
r = m.solve(time_limit=budget)
wall = time.perf_counter() - t0

print(
    json.dumps(
        {
            "instance": path.split("/")[-1].removesuffix(".nl"),
            "flag": flag,
            "budget": budget,
            "sense": sense,
            "wall": wall,
            "status": r.status,
            "objective": r.objective,
            "bound": r.bound,
            "gap": r.gap,
            "gap_certified": bool(r.gap_certified),
            "node_count": int(r.node_count or 0),
            "incumbent_verification_failed": bool(
                getattr(r, "incumbent_verification_failed", False)
            ),
            "extension_s": (r.solver_stats or {}).get("budget/incumbent_extension_s"),
        }
    )
)
