"""Bound-neutrality evidence for the #1160 guards on the `.nl` corpus.

Two measurements, both during REAL solves (a claim about which code runs cannot
be made from a parsed model -- a relaxation pass constructs nodes mid-solve):

1. every ``sum_is_full_reduction`` call is counted, and every call that returned
   False (the only way any guard changes behaviour) is counted separately;
2. every ``SumExpression`` node handed to a guarded walker is counted.

Prints the counts and the per-instance (status, objective, bound, node_count) so
the run can be diffed against the same script on the baseline tree. Exits
non-zero if it made no observation at all.
"""

import glob
import json
import os
import sys

import discopt._relax.scalarize as scalarize

CALLS = [0]
REFUSALS = [0]
GUARDED = ["absent"]

# CLAUDE.md 8: the arm must say WHICH tree it loaded. The baseline tree has no
# `sum_is_full_reduction`; asserting the marker absent there is what stops a
# mixed arm (editable install + pkgutil path) from being reported as a baseline.
print("discopt.scalarize file:", scalarize.__file__)
if hasattr(scalarize, "sum_is_full_reduction"):
    GUARDED[0] = "present"
    _real = scalarize.sum_is_full_reduction

    def counting(expr):
        CALLS[0] += 1
        out = _real(expr)
        if not out:
            REFUSALS[0] += 1
        return out

    scalarize.sum_is_full_reduction = counting
    for mod in (
        "discopt._relax.problem_classifier",
        "discopt._relax.milp_relaxation",
        "discopt._relax.obbt",
        "discopt._relax.canonical_expr",
        "discopt._relax.dependent_vars",
        "discopt._relax.convexity.linear_context",
    ):
        __import__(mod)
        m = sys.modules[mod]
        if hasattr(m, "sum_is_full_reduction"):
            m.sum_is_full_reduction = counting
print("GUARD MARKER:", GUARDED[0])

import discopt.modeling.core as dm  # noqa: E402
from discopt.modeling import from_nl  # noqa: E402

# Instrument self-test (CLAUDE.md 6): a counter that reports 0 because the patch
# never took hold is indistinguishable from a counter that reports 0 because the
# branch never runs. Solve one Python-API axis-sum model first and require the
# counters to MOVE; then reset them for the `.nl` measurement.
if GUARDED[0] == "present":
    _probe = dm.Model("selftest")
    _A = _probe.continuous("A", shape=(2, 3), lb=0, ub=1)
    _probe.subject_to(dm.sum(_A, axis=1) <= 2)
    _probe.minimize(-dm.sum(_A))
    _probe.solve(time_limit=30, gap_tolerance=1e-6)
    print(f"SELFTEST calls={CALLS[0]} refusals={REFUSALS[0]}")
    if CALLS[0] == 0 or REFUSALS[0] == 0:
        sys.exit("instrument is dead: the axis-sum self-test moved no counter")
    CALLS[0] = 0
    REFUSALS[0] = 0

TIME_LIMIT = float(os.environ.get("NL_TIME_LIMIT", "10"))
MAX_NODES = int(os.environ.get("NL_MAX_NODES", "3000"))

files = sorted(glob.glob("python/tests/data/minlplib_nl/*.nl"))
limit = int(os.environ.get("NL_LIMIT", "0"))
if limit:
    files = files[:limit]
only = os.environ.get("NL_ONLY")
if only:
    wanted = {n.strip() for n in only.split(",")}
    files = [f for f in files if os.path.basename(f)[:-3] in wanted]

rows = {}
for path in files:
    name = os.path.basename(path)[:-3]
    try:
        model = from_nl(path)
        r = model.solve(time_limit=TIME_LIMIT, gap_tolerance=1e-6, max_nodes=MAX_NODES)
        rows[name] = {
            "status": r.status,
            "objective": r.objective,
            "bound": r.bound,
            "node_count": getattr(r, "node_count", None),
        }
    except Exception as exc:  # a load/solve failure is data, not something to hide
        rows[name] = {"error": f"{type(exc).__name__}: {exc}"}
    print(f"{name:24s} {rows[name]}", flush=True)

print(f"INSTANCES: {len(rows)}")
print(f"GUARD_CALLS: {CALLS[0]}")
print(f"GUARD_REFUSALS (axis-reduced sums seen): {REFUSALS[0]}")
out = os.environ.get("NL_OUT")
if out:
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1, sort_keys=True)
if not rows:
    sys.exit("probe solved nothing")
