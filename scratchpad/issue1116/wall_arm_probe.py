"""#1116 E6: WHICH wall seam causes the drift? (bisection of E4)

E4 neutralized three things at once and the solve became bit-reproducible:
  clock  - time.perf_counter -> deterministic counter
  lp     - MilpRelaxationModel.solve(time_limit=...) -> None
  nlp    - max_wall_time stripped from the POUNCE options at both seams

That is a confounded result: it proves "a wall clock is the cause" but not which
one. This probe enables an arbitrary SUBSET so the arms can be run singly.

Usage: python -u wall_arm_probe.py <stem> <max_nodes> <reps> <arms>
  arms: comma-separated subset of {clock,lp,nlp}, or "none" for the baseline.

Kill criterion per arm: if an arm reproduces (all reps bit-identical) the seams
it patches are sufficient; if it varies they are not the whole story.

Each patch counts its firings; the probe exits non-zero if a requested patch
never fired (CLAUDE.md §6) so a mis-targeted patch cannot masquerade as
"this seam is not the cause". No exception is swallowed (§7). Module identity is
printed (§8) and per-rep progress is flushed (§10).
"""

import json
import sys
import time

stem, max_nodes, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
arms = set() if sys.argv[4] == "none" else set(sys.argv[4].split(","))
assert arms <= {"clock", "lp", "nlp"}, f"unknown arm in {arms}"

_real_perf_counter = time.perf_counter
_clock = {"n": 0, "t": 0.0}

if "clock" in arms:

    def _fake():
        _clock["n"] += 1
        _clock["t"] += 1e-4
        return _clock["t"]

    time.perf_counter = _fake  # type: ignore[assignment]
    time.monotonic = _fake  # type: ignore[assignment]

import discopt  # noqa: E402
import pounce  # noqa: E402
from discopt._relax.milp_relaxation import MilpRelaxationModel  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402
from discopt.solvers import nlp_pounce  # noqa: E402

print(f"discopt.__file__={discopt.__file__} arms={sorted(arms)}", flush=True)

counts = {"lp": 0, "nlp_single": 0, "nlp_batch": 0}

if "lp" in arms:
    _real_milp_solve = MilpRelaxationModel.solve

    def _milp_solve(self, time_limit=None, *a, **kw):
        counts["lp"] += 1
        return _real_milp_solve(self, None, *a, **kw)

    MilpRelaxationModel.solve = _milp_solve  # type: ignore[method-assign]

if "nlp" in arms:

    def _strip(d):
        return {k: v for k, v in d.items() if k != "max_wall_time"}

    _real_solve_nlp = nlp_pounce.solve_nlp

    def _solve_nlp(*a, **kw):
        counts["nlp_single"] += 1
        if len(a) >= 4 and isinstance(a[3], dict):
            a = a[:3] + (_strip(a[3]),) + a[4:]
        if isinstance(kw.get("options"), dict):
            kw["options"] = _strip(kw["options"])
        return _real_solve_nlp(*a, **kw)

    nlp_pounce.solve_nlp = _solve_nlp

    _real_batch = pounce.solve_nlp_batch

    def _batch(*a, **kw):
        counts["nlp_batch"] += 1
        if isinstance(kw.get("options"), dict):
            kw["options"] = _strip(kw["options"])
        return _real_batch(*a, **kw)

    pounce.solve_nlp_batch = _batch

NL = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/{}.nl"
rows = []
for rep in range(reps):
    model = from_nl(NL.format(stem))
    t0 = _real_perf_counter()
    r = model.solve(max_nodes=max_nodes)
    row = {
        "rep": rep,
        "nodes": int(r.node_count or 0),
        "bound": repr(float(r.bound)) if r.bound is not None else None,
        "objective": repr(float(r.objective)) if r.objective is not None else None,
        "status": r.status,
        "wall": round(_real_perf_counter() - t0, 2),
        "counts": dict(counts),
        "clock_reads": _clock["n"],
    }
    rows.append(row)
    print(json.dumps(row), flush=True)

comparisons = 0
varies = 0
for key in ("nodes", "bound", "objective", "status"):
    distinct = sorted({repr(r[key]) for r in rows})
    comparisons += len(rows) - 1
    varies += len(distinct) > 1
    print(
        f"{key:10s} {'STABLE' if len(distinct) == 1 else 'VARIES'} "
        f"distinct={len(distinct)} {distinct}",
        flush=True,
    )

print(
    f"ARM {sorted(arms) or ['none']}: {'REPRODUCES' if varies == 0 else 'STILL VARIES'}", flush=True
)
print(f"comparisons={comparisons} patch_counts={counts} clock_reads={_clock['n']}", flush=True)

missing = (
    [a for a in arms if a == "lp" and counts["lp"] == 0]
    + [a for a in arms if a == "nlp" and counts["nlp_single"] + counts["nlp_batch"] == 0]
    + [a for a in arms if a == "clock" and _clock["n"] == 0]
)
if comparisons == 0 or missing:
    print(f"PROBE FIRED NOTHING (missing={missing})", flush=True)
    sys.exit(2)
