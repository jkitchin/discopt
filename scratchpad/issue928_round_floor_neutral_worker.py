"""Neutrality worker for the #928 round-cut-short floor. Runs on EITHER tree.

CLAUDE.md §8: the marker is asserted in BOTH directions — ``expect_marker=1``
requires ``_cut_short_floor`` to be a local of
``MccormickLPRelaxer._solve_at_node_impl``, ``0`` requires it absent — so a
baseline run that silently imported the changed tree fails instead of producing a
bogus "identical" verdict. Every flag this change interacts with is set to its
shipped default explicitly, so no cell can inherit one from the environment.

Usage: python issue928_round_floor_neutral_worker.py <path.nl> <limit> <0|1>
"""

import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt  # noqa: E402
from discopt._relax.mccormick_lp import MccormickLPRelaxer  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

path, budget, expect = sys.argv[1], float(sys.argv[2]), sys.argv[3]
assert "/python/discopt/" in discopt.__file__, discopt.__file__
present = "_cut_short_floor" in MccormickLPRelaxer._solve_at_node_impl.__code__.co_varnames
assert present == (expect == "1"), (
    f"marker mismatch: _cut_short_floor present={present}, expected={expect == '1'}"
)
for _flag in ("DISCOPT_LP_WARM_DEADLINE", "DISCOPT_NODE_ROUND_BUDGET", "DISCOPT_HESS_COMPILE_GATE"):
    assert os.environ.get(_flag) == "0", f"{_flag} must be pinned OFF for the neutrality arm"

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
