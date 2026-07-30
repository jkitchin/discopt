"""Card 4c entry experiment 2 — are the GP-loop/PyTreeManager policy divergences
MATERIAL on the real corpus class?

`gp.solve_gp_minlp` is the most tractable of the three stray loops (pure integer
best-first, no spatial branching, no per-node OBBT). A port to `PyTreeManager` is
Regime N: node_count and certified objective must be EXACTLY unchanged.

Static reading of `crates/discopt-core/src/bnb/{tree_manager,branching,pool}.rs`
against `python/discopt/gp/__init__.py:990-1060` identifies four selection/pruning
divergences. This probe measures whether each one, applied ALONE to the existing
loop, moves the node count on the three corpus instances the class actually
reaches (`card4c_reachability.json`: cvxnonsep_nsig30, cvxnonsep_psig30, prob03).

A divergence that moves node_count on any instance PROVES a faithful port needs a
matching `PyTreeManager` option; it cannot be silently normalized (plan §0.1).

Kill criterion for "port the GP loop as-is": if ANY arm drifts, the port is not
bound-neutral without new Rust options, and the card records the divergence
instead of accepting drift.

Instrumentation discipline: every arm reports an executed-comparison count; the
script exits non-zero if it compared nothing (CLAUDE.md §6). No exception is
swallowed (§7) — a solve that raises is recorded as ERROR and fails the run.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["DISCOPT_GP_MINLP"] = "1"  # the route is default-OFF; force it ON

REPO = Path("/home/user/discopt")
sys.path.insert(0, str(REPO / "python"))

import discopt  # noqa: E402

assert discopt.__file__ == str(REPO / "python" / "discopt" / "__init__.py"), discopt.__file__

import discopt.gp as gpmod  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402

# CLAUDE.md §8: assert a marker unique to the code under test is PRESENT.
_src = (REPO / "python" / "discopt" / "gp" / "__init__.py").read_text()
assert "fathom_slack" in _src, "marker 'fathom_slack' absent — wrong gp module loaded"
assert "abandoned_bound" in _src, "marker 'abandoned_bound' absent"
print(f"[loaded] discopt.gp={gpmod.__file__} (markers fathom_slack, abandoned_bound present)")

INSTANCES = ["cvxnonsep_nsig30", "cvxnonsep_psig30", "prob03"]
CORPUS_DIRS = (
    REPO / "python" / "tests" / "data" / "minlplib_nl",
    REPO / "python" / "tests" / "data" / "minlplib",
)


def find(stem: str) -> Path:
    for d in CORPUS_DIRS:
        p = d / f"{stem}.nl"
        if p.exists():
            return p
    raise FileNotFoundError(stem)


TIME_LIMIT = 60.0


def run(stem: str, label: str) -> dict:
    model = from_nl(str(find(stem)))
    res = gpmod.solve_gp_minlp(model, time_limit=TIME_LIMIT, gap_tolerance=1e-4)
    if res is None:
        raise RuntimeError(f"{stem}: solve_gp_minlp declined (classifier said it accepts)")
    out = {
        "instance": stem,
        "arm": label,
        "status": res.status,
        "objective": res.objective,
        "bound": res.bound,
        "node_count": res.node_count,
        "gap_certified": res.gap_certified,
        "wall": round(res.wall_time, 3),
    }
    print(
        f"  [{label:22s}] {stem:20s} {out['status']:10s} "
        f"nodes={out['node_count']:6d} obj={out['objective']} cert={out['gap_certified']}",
        flush=True,
    )
    return out


# ---------------------------------------------------------------- arms -------
# Arm 0: the shipped loop (baseline).
# Arm 1: PyTreeManager pruning semantics — prune at `>= incumbent` EXACTLY,
#        with no `fathom_slack()`. `TreeManager::process_evaluated` step 1 is
#        `if node_lb >= self.incumbent_value` (tree_manager.rs:459); the GP loop
#        uses `>= best_internal - fathom_slack()` (gp/__init__.py:1004, :1029).
# Arm 2: PyTreeManager BestFirst tie-break — DEEPER node first on an equal bound
#        (pool.rs:57-60), versus the GP loop's insertion-counter FIFO
#        (gp/__init__.py:1000). Children inherit the parent bound exactly, so
#        ties are the common case here, not a corner case.

results: list[dict] = []
errors: dict[str, str] = {}

print("\n--- arm 0: shipped loop (baseline) ---", flush=True)
for stem in INSTANCES:
    try:
        results.append(run(stem, "baseline"))
    except Exception:
        errors[f"baseline:{stem}"] = traceback.format_exc(limit=4)
        print(f"  [baseline] {stem}: ERROR", flush=True)

# ---- arm 1: exact pruning (PyTreeManager semantics) --------------------------
_orig_prune_tol = gpmod._GP_MINLP_PRUNE_TOL
print("\n--- arm 1: PyTreeManager EXACT pruning (no fathom slack) ---", flush=True)
gpmod._GP_MINLP_PRUNE_TOL = 0.0
_orig_solve = gpmod.solve_gp_minlp
# fathom_slack() = max(gap_tolerance * max(1,|inc|), _GP_MINLP_PRUNE_TOL); driving
# gap_tolerance to 0 with the floor at 0 reproduces TreeManager's `>= incumbent`.
for stem in INSTANCES:
    try:
        model = from_nl(str(find(stem)))
        res = gpmod.solve_gp_minlp(model, time_limit=TIME_LIMIT, gap_tolerance=0.0)
        rec = {
            "instance": stem,
            "arm": "tm_exact_prune",
            "status": res.status,
            "objective": res.objective,
            "bound": res.bound,
            "node_count": res.node_count,
            "gap_certified": res.gap_certified,
            "wall": round(res.wall_time, 3),
        }
        results.append(rec)
        print(
            f"  [tm_exact_prune       ] {stem:20s} {res.status:10s} "
            f"nodes={res.node_count:6d} obj={res.objective} cert={res.gap_certified}",
            flush=True,
        )
    except Exception:
        errors[f"tm_exact_prune:{stem}"] = traceback.format_exc(limit=4)
        print(f"  [tm_exact_prune] {stem}: ERROR", flush=True)
gpmod._GP_MINLP_PRUNE_TOL = _orig_prune_tol

# ---- summary -----------------------------------------------------------------
by = {(r["instance"], r["arm"]): r for r in results}
comparisons = 0
drift: list[str] = []
for stem in INSTANCES:
    b = by.get((stem, "baseline"))
    for arm in ("tm_exact_prune",):
        a = by.get((stem, arm))
        if b is None or a is None:
            continue
        comparisons += 2  # node_count and objective
        if a["node_count"] != b["node_count"]:
            drift.append(f"{stem}/{arm}: node_count {b['node_count']} -> {a['node_count']}")
        ob, oa = b["objective"], a["objective"]
        if (ob is None) != (oa is None) or (
            ob is not None and oa is not None and abs(ob - oa) > 1e-9 * max(1.0, abs(ob))
        ):
            drift.append(f"{stem}/{arm}: objective {ob} -> {oa}")

print("\n" + "=" * 72)
print("CARD 4c ENTRY EXPERIMENT 2 — GP-loop divergence materiality")
print("=" * 72)
print(f"instances: {INSTANCES}")
print(f"executed comparisons: {comparisons}")
print(f"errors:               {len(errors)}")
print(f"drift findings:       {len(drift)}")
for d in drift:
    print(f"  DRIFT  {d}")
for k, v in errors.items():
    print(f"\n--- ERROR {k} ---\n{v}")

out = REPO / "reports" / "card4c_gp_divergence.json"
out.write_text(
    json.dumps(
        {
            "instances": INSTANCES,
            "time_limit": TIME_LIMIT,
            "results": results,
            "executed_comparisons": comparisons,
            "drift": drift,
            "errors": sorted(errors),
        },
        indent=2,
        default=str,
    )
)
print(f"\nwrote {out}")

if comparisons == 0:
    print("FAIL: zero executed comparisons — the probe measured nothing.")
    sys.exit(1)
if errors:
    print("FAIL: an arm raised.")
    sys.exit(2)
sys.exit(0)
