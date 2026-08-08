"""#945 replication: are the panel's time-limited diffs signal or noise?

The corpus panel's differing instances are almost all runs that hit the 20 s
budget, where the answer depends on how far the search got. A single run cannot
separate an arm effect from that, so this repeats each instance N times per arm,
INTERLEAVED (pre, post, pre, post, ...), and reports the spread. §9: a claim about
a difference needs a spread, not two numbers.

Reports, per instance and arm: the set of statuses seen, the incumbent range, the
bound range, and the node-count range.

This script COLLECTS; it does not judge. Its first version also printed a
verdict column, and that column was wrong — see the RETRACTION at the top of
``panel_replicate_analyze.py``. Run that on the JSON this writes.

§6: prints an executed-run count and exits non-zero if it is zero.

Usage:  python -u scratchpad/issue945/panel_replicate.py out.json [reps] [time_limit]
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.solver as SOLVER  # noqa: E402
import discopt.solvers.gdpopt_loa as LOA  # noqa: E402
import discopt.solvers.nlp_pounce as NLPP  # noqa: E402
import discopt.solvers.oa as OA  # noqa: E402
from discopt.modeling import from_nl  # noqa: E402
from discopt.solvers import pounce_option_defaults  # noqa: E402

# §8 markers, tracking the CONTRACT rather than a line of code: #945 requests the
# incumbent options at the point-consuming call sites and leaves the NLP backend
# neutral. An earlier version named the superseded contract and refused to run once
# the branch rescoped — the guard working, not a bug.
for _fn in (SOLVER._solve_continuous, OA._solve_nlp_attempt, LOA._solve_nlp_subproblem):
    assert "pounce_incumbent_options()" in inspect.getsource(_fn), (
        f"post-#945 marker absent in {_fn.__name__} — nothing to replicate"
    )
assert "pounce_incumbent_options()" not in inspect.getsource(NLPP.solve_nlp), (
    "bound_relax_factor leaked back into the NLP backend default"
)

# Same two-seam arm reconstruction as panel_corpus.set_arm: bound_relax_factor is
# not a backend default post-#945, so neutralizing only the backend would leave the
# incumbent requests live and mislabel the arm.
_PRE_BACKEND = {"print_level": 0}
_REAL = pounce_option_defaults
_REAL_INCUMBENT = SOLVER.pounce_incumbent_options
_CONSUMERS = (SOLVER, OA, LOA)

ROOT = "python/tests/data/minlplib_nl"

# Every instance the final corpus panel reported as differing. The set deliberately
# includes movers in BOTH directions — tls2 (panel: certification LOST) and
# syn05hfsg (panel: certification GAINED) — because a replication harness that only
# ever chases regressions cannot demonstrate it is capable of seeing anything.
TARGETS = ["chance", "nvs05", "syn05hfsg", "tanksize", "tls2"]


def set_arm(arm: str) -> None:
    pre = arm == "pre"
    NLPP.pounce_option_defaults = (lambda: dict(_PRE_BACKEND)) if pre else _REAL
    for mod in _CONSUMERS:
        mod.pounce_incumbent_options = (lambda: {}) if pre else _REAL_INCUMBENT
        if hasattr(mod, "pounce_option_defaults"):
            mod.pounce_option_defaults = (lambda: dict(_PRE_BACKEND)) if pre else _REAL


def run(path: str, arm: str, tl: float) -> dict:
    set_arm(arm)
    t0 = time.perf_counter()
    try:
        model = from_nl(path)
        maximize = model._objective is not None and str(model._objective.sense).endswith("MAXIMIZE")
        r = model.solve(time_limit=tl)
    except Exception as exc:  # recorded, never swallowed (§7)
        return {"error": f"{type(exc).__name__}: {exc}", "wall": time.perf_counter() - t0}
    return {
        "maximize": maximize,
        "status": r.status,
        "objective": r.objective,
        "bound": r.bound,
        "node_count": r.node_count,
        "wall": time.perf_counter() - t0,
    }


def _rng(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return (min(vals), max(vals))


def _fmt(rg, prec=10):
    if rg is None:
        return "—"
    lo, hi = rg
    if lo == hi:
        return f"{lo:.{prec}g}"
    return f"[{lo:.{prec}g}, {hi:.{prec}g}]"


def main(out: str, reps: int, tl: float) -> int:
    results: dict = {}
    runs = 0
    for name in TARGETS:
        p = os.path.join(ROOT, f"{name}.nl")
        results[name] = {"pre": [], "post": []}
        for _ in range(reps):
            for arm in ("pre", "post"):  # interleaved, not two sequential blocks
                r = run(p, arm, tl)
                results[name][arm].append(r)
                runs += 1
                print(
                    f"  {name:10s} {arm:5s} status={str(r.get('status')):11s} "
                    f"obj={r.get('objective')!r} bound={r.get('bound')!r} "
                    f"n={r.get('node_count')} wall={r.get('wall', 0):.1f}s",
                    flush=True,
                )
    set_arm("post")

    json.dump({"results": results, "reps": reps, "time_limit": tl}, open(out, "w"), indent=1)

    print(f"\n{'instance':11s} {'arm':5s} {'statuses':24s} {'objective':>26s} {'bound':>26s} {'nodes':>14s}")
    print("-" * 112)
    for name in TARGETS:
        rngs = {}
        for arm in ("pre", "post"):
            rs = results[name][arm]
            st = sorted({str(r.get("status")) for r in rs})
            o = _rng([r.get("objective") for r in rs])
            b = _rng([r.get("bound") for r in rs])
            n = _rng([r.get("node_count") for r in rs])
            rngs[arm] = (st, o, b, n)
            print(f"{name:11s} {arm:5s} {','.join(st):24s} {_fmt(o):>26s} {_fmt(b):>26s} {_fmt(n,6):>14s}")
        # No verdict here. A bare interval-overlap test with no tolerance called
        # every instance an "arm effect", including one whose arms differ by 2e-9
        # relative — a criterion that fires on everything separates nothing.
        # `panel_replicate_analyze.py` does the judging, against the saved JSON.

    print(f"\nEXECUTED_RUNS={runs}  reps={reps}  time_limit={tl}")
    print(f"Now run: python -u scratchpad/issue945/panel_replicate_analyze.py {out}")
    if runs == 0:
        print("REPLICATION RAN NOTHING", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    o = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    tl = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
    sys.exit(main(o, reps, tl))
