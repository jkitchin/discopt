"""#945 replication: are the panel's time-limited diffs signal or noise?

The corpus panel's differing instances are almost all runs that hit the 20 s
budget, where the answer depends on how far the search got. A single run cannot
separate an arm effect from that, so this repeats each instance N times per arm,
INTERLEAVED (pre, post, pre, post, ...), and reports the spread. §9: a claim about
a difference needs a spread, not two numbers.

Reports, per instance and arm: the set of statuses seen, the incumbent range, the
bound range, and the node-count range. A difference is an arm effect only when the
arms' ranges do not overlap.

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
import discopt.solvers.nlp_pounce as NLPP  # noqa: E402
from discopt.modeling import from_nl  # noqa: E402
from discopt.solvers import pounce_option_defaults  # noqa: E402

assert "opts = pounce_option_defaults()" in inspect.getsource(NLPP.solve_nlp), (
    "post-#945 marker absent — nothing to replicate"
)

_PRE_ARM = {"print_level": 0}
_REAL = pounce_option_defaults
_ORIG_BATCH = SOLVER.pounce_option_defaults

ROOT = "python/tests/data/minlplib_nl"

# Every instance the corpus panel reported as differing on status, objective or
# bound in a way that is not pure last-digit drift, plus tspn05 (where the post
# arm's bound was BETTER) as a control — if replication says everything is noise,
# a control that moved the other way is what shows the harness can see anything.
TARGETS = ["tls2", "tspn12", "nvs05", "tspn05", "syn05hfsg", "tanksize"]


def set_arm(arm: str) -> None:
    fn = (lambda: dict(_PRE_ARM)) if arm == "pre" else None
    NLPP.pounce_option_defaults = fn or _REAL
    SOLVER.pounce_option_defaults = fn or _ORIG_BATCH


def run(path: str, arm: str, tl: float) -> dict:
    set_arm(arm)
    t0 = time.perf_counter()
    try:
        r = from_nl(path).solve(time_limit=tl)
    except Exception as exc:  # recorded, never swallowed (§7)
        return {"error": f"{type(exc).__name__}: {exc}", "wall": time.perf_counter() - t0}
    return {
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
    verdicts = []
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
        # Overlapping ranges => the panel's single-run difference was within
        # this instance's own run-to-run spread, i.e. not an arm effect.
        def overlap(x, y):
            if x is None or y is None:
                return x is None and y is None
            return not (x[1] < y[0] or y[1] < x[0])

        same_status = set(rngs["pre"][0]) == set(rngs["post"][0])
        verdict = (
            "arm effect"
            if not (same_status and overlap(rngs["pre"][1], rngs["post"][1])
                    and overlap(rngs["pre"][2], rngs["post"][2]))
            else "within noise"
        )
        verdicts.append((name, verdict))
        print(f"{'':11s} {'':5s} -> {verdict}")

    print(f"\nEXECUTED_RUNS={runs}  reps={reps}  time_limit={tl}")
    for n, v in verdicts:
        print(f"  VERDICT {n}: {v}")
    if runs == 0:
        print("REPLICATION RAN NOTHING", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    o = sys.argv[1]
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    tl = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
    sys.exit(main(o, reps, tl))
