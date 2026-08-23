"""#1116 verification: does ``deterministic=True`` actually reproduce?

Usage: python -u deterministic_verify.py <stem> <max_nodes> <reps> <0|1>

Runs the same solve N times in one process with ``deterministic`` set to the
given value and reports whether (nodes, bound, objective, status) is bit-stable.
This is the acceptance test for the fix: the arm with the flag ON must be STABLE
on an instance whose OFF arm is measurably not.

Asserts the flag reached the solver (``_role2_deadline`` / ``_role2_horizon``
firing counts, split by whether they suppressed a clock) and exits non-zero when
the role-2 helpers never fired at all -- otherwise "it reproduced" could just mean
"the code under test was never entered" (CLAUDE.md §6, §8).
"""

import json
import sys

import discopt
import discopt.solver as S
from discopt.modeling.core import from_nl

print(f"discopt.__file__={discopt.__file__}", flush=True)

stem, max_nodes, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
det = bool(int(sys.argv[4]))

fired = {"deadline_suppressed": 0, "deadline_kept": 0, "horizon_suppressed": 0, "horizon_kept": 0}

_rd, _rh = S._role2_deadline, S._role2_horizon


def _spy_deadline(d):
    r = _rd(d)
    fired["deadline_suppressed" if r is None else "deadline_kept"] += 1
    return r


def _spy_horizon(x):
    r = _rh(x)
    fired["horizon_suppressed" if r == float("inf") else "horizon_kept"] += 1
    return r


S._role2_deadline = _spy_deadline
S._role2_horizon = _spy_horizon

NL = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/{}.nl"
rows = []
for rep in range(reps):
    r = from_nl(NL.format(stem)).solve(max_nodes=max_nodes, deterministic=det)
    row = {
        "rep": rep,
        "deterministic": det,
        "nodes": int(r.node_count or 0),
        "bound": repr(float(r.bound)) if r.bound is not None else None,
        "objective": repr(float(r.objective)) if r.objective is not None else None,
        "status": r.status,
        "role2": dict(fired),
    }
    rows.append(row)
    print(json.dumps(row), flush=True)

comparisons = 0
varies = 0
for key in ("nodes", "bound", "objective", "status"):
    distinct = sorted({repr(x[key]) for x in rows})
    comparisons += len(rows) - 1
    varies += len(distinct) > 1
    print(
        f"{key:10s} {'STABLE' if len(distinct) == 1 else 'VARIES'} "
        f"distinct={len(distinct)} {distinct}",
        flush=True,
    )

print(f"ARM deterministic={det}: {'REPRODUCES' if varies == 0 else 'STILL VARIES'}", flush=True)
print(f"comparisons={comparisons} role2_firings={fired}", flush=True)
if comparisons == 0 or sum(fired.values()) == 0:
    print("PROBE FIRED NOTHING", flush=True)
    sys.exit(2)
