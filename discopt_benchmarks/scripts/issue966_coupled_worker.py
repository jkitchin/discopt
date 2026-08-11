"""#966/#928 coupled panel worker: one instance, one budget, one process.

Reads the three coupled flags from the environment (the panel sets all three
explicitly on every arm, so an inherited value can never leak into a cell) and
reports the full result record as JSON on stdout. Exceptions propagate: a broken
arm must fail the panel, never read as a clean run (CLAUDE.md §7).
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
from discopt._relax import mccormick_lp as _mc  # noqa: E402
from discopt.modeling.core import ObjectiveSense, from_nl  # noqa: E402
from discopt.solvers import milp_simplex as _ms  # noqa: E402

# CLAUDE.md §8: prove which tree is under test before measuring anything with it.
# One marker per merged PR whose behaviour this panel is scoring; a run against a
# tree missing any of them is not the experiment and must die, not report.
assert "/python/discopt/" in discopt.__file__, discopt.__file__
assert hasattr(_solver, "_extend_budget_for_incumbent"), "#917 marker absent"
assert hasattr(_ms, "_dual_start_slack_basis"), "#928 marker '_dual_start_slack_basis' absent"
assert "round_deadline" in _mc.MccormickLPRelaxer.solve_at_node.__code__.co_varnames, (
    "#966 marker 'round_deadline' absent from solve_at_node"
)
assert "yield_round" in _mc.MccormickLPRelaxer.solve_at_node.__code__.co_varnames, (
    "#966 yield-mode marker 'yield_round' absent from solve_at_node — this tree "
    "predates the fix that makes a short-granted round yield instead of skip"
)

FLAGS = ("DISCOPT_LP_WARM_DEADLINE", "DISCOPT_NODE_ROUND_BUDGET", "DISCOPT_HESS_COMPILE_GATE")
for _f in FLAGS:
    assert _f in os.environ, f"{_f} not set by the panel — an arm must never inherit a default"

path, budget = sys.argv[1], float(sys.argv[2])

m = from_nl(path)
sense = "max" if m._objective.sense == ObjectiveSense.MAXIMIZE else "min"

t0 = time.perf_counter()
r = m.solve(time_limit=budget)
wall = time.perf_counter() - t0

print(
    json.dumps(
        {
            "instance": path.split("/")[-1].removesuffix(".nl"),
            "flags": {f: os.environ[f] for f in FLAGS},
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
        }
    )
)
