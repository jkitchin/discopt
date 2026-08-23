"""#1116 E4: is the run-to-run drift caused by WALL-CLOCK-derived budgets?

Neutralizes every wall-derived budget the Python layer can reach, then runs the
same reproduction as ``repro_probe.py``:

  1. ``time.perf_counter`` -> a deterministic counter (fixed step per call), so
     every budget the solver DERIVES from the clock is a pure function of the
     call sequence, not of machine speed.
  2. ``MilpRelaxationModel.solve(time_limit=...)`` -> ``None`` (the LP seam into the
     Rust simplex; the simplex stays bounded by its iteration cap).
  3. ``nlp_pounce.solve_nlp`` opts / ``pounce.solve_nlp_batch`` options ->
     ``max_wall_time`` removed (the two seams through which every NLP wall clamp
     reaches POUNCE, which measures REAL wall internally and so is immune to 1).

Every patch counts its firings and the probe exits non-zero if the patch set as a
whole never fired (CLAUDE.md §6) -- a patch that silently matched nothing would
turn "the clock is not the cause" into an unfalsifiable pass. Nothing is wrapped
in try/except (§7).

Usage: python -u no_wall_probe.py <instance-stem> <max_nodes> <reps>
"""

import json
import sys
import time

# ---- 1. deterministic clock -------------------------------------------------
_CLOCK_STEP = 1e-4
_clock = {"t": 0.0, "n": 0}


def _fake_perf_counter() -> float:
    _clock["n"] += 1
    _clock["t"] += _CLOCK_STEP
    return _clock["t"]


_real_perf_counter = time.perf_counter
time.perf_counter = _fake_perf_counter  # type: ignore[assignment]
time.monotonic = _fake_perf_counter  # type: ignore[assignment]

# Prove which tree we loaded (CLAUDE.md §8).
import discopt  # noqa: E402
import pounce  # noqa: E402
from discopt._relax.milp_relaxation import MilpRelaxationModel  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402
from discopt.solvers import nlp_pounce  # noqa: E402

print(f"discopt.__file__={discopt.__file__}", flush=True)

counts = {"milp_solve": 0, "solve_nlp": 0, "batch": 0, "batch_qp": 0}

# ---- 2. LP seam -------------------------------------------------------------
_real_milp_solve = MilpRelaxationModel.solve


def _milp_solve(self, time_limit=None, *a, **kw):
    counts["milp_solve"] += 1
    return _real_milp_solve(self, None, *a, **kw)


MilpRelaxationModel.solve = _milp_solve  # type: ignore[method-assign]

# ---- 3. NLP seams -----------------------------------------------------------
_real_solve_nlp = nlp_pounce.solve_nlp


def _strip(d):
    return {k: v for k, v in d.items() if k != "max_wall_time"}


def _solve_nlp(*a, **kw):
    counts["solve_nlp"] += 1
    # ``options`` is the 4th positional parameter; callers use both forms.
    if len(a) >= 4 and isinstance(a[3], dict):
        a = a[:3] + (_strip(a[3]),) + a[4:]
    opts = kw.get("options")
    if isinstance(opts, dict):
        kw["options"] = _strip(opts)
    return _real_solve_nlp(*a, **kw)


nlp_pounce.solve_nlp = _solve_nlp

_real_batch = pounce.solve_nlp_batch


def _batch(*a, **kw):
    counts["batch"] += 1
    opts = kw.get("options")
    if isinstance(opts, dict) and "max_wall_time" in opts:
        kw["options"] = {k: v for k, v in opts.items() if k != "max_wall_time"}
    return _real_batch(*a, **kw)


pounce.solve_nlp_batch = _batch

if hasattr(pounce, "solve_qp_batch"):
    _real_qp_batch = pounce.solve_qp_batch

    def _qp_batch(*a, **kw):
        counts["batch_qp"] += 1
        opts = kw.get("options")
        if isinstance(opts, dict) and "max_wall_time" in opts:
            kw["options"] = {k: v for k, v in opts.items() if k != "max_wall_time"}
        return _real_qp_batch(*a, **kw)

    pounce.solve_qp_batch = _qp_batch

NL = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/{}.nl"
stem, max_nodes, reps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

rows = []
for rep in range(reps):
    model = from_nl(NL.format(stem))
    t0 = _real_perf_counter()
    r = model.solve(max_nodes=max_nodes)
    wall = _real_perf_counter() - t0
    row = {
        "rep": rep,
        "nodes": int(r.node_count or 0),
        "bound": float(r.bound) if r.bound is not None else None,
        "objective": float(r.objective) if r.objective is not None else None,
        "status": r.status,
        "wall": round(wall, 2),
        "clock_calls": _clock["n"],
        "patch_counts": dict(counts),
    }
    rows.append(row)
    print(json.dumps(row), flush=True)

comparisons = 0
for key in ("nodes", "bound", "objective", "status"):
    vals = [r[key] for r in rows]
    distinct = sorted({repr(v) for v in vals})
    comparisons += len(vals) - 1
    verdict = "STABLE" if len(distinct) == 1 else "VARIES"
    print(f"{key:10s} {verdict}  distinct={len(distinct)}  {distinct}", flush=True)

fired = sum(counts.values())
print(f"patch_firings={counts} total={fired}", flush=True)
print(f"comparisons={comparisons}", flush=True)
if comparisons == 0 or fired == 0:
    print("PROBE FIRED NOTHING", flush=True)
    sys.exit(2)
