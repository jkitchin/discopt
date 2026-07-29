#!/usr/bin/env python
"""Card 3e entry experiment: does seeding the root FBBT actually compose?

**Hypothesis (from Card 3b, measured — not invented here).** `FbbtPass::run` calls
`fbbt_with_cutoff_until(&ctx.model, …)`, which seeds `var_bounds` from
`model.variables` — the *declared* box — and never reads `ctx.bounds`. The
orchestrator deliberately never writes tightened bounds back into `ctx.model`, so the
wired-in `fbbt` pass re-derives the same box on every sweep and can never propagate
from what `eliminate`, `simplify`, `implied_bounds`, `coefficient_strengthening` or
`probing` just proved. Card 3b measured the consequence: at `max_iterations=1`,
`[implied_bounds, X]` beats `intersect(X@1, implied_bounds@1)` on **0** bounds for
`fbbt` across 7/7 instances and on **48** for `fbbt_fp`
(`reports/card3b_fbbt_vs_fbbt_fp.json`).

**What this probe tests, and its kill criterion.** Running the *production* root
presolve with `DISCOPT_FBBT_SEED` on vs off over the real corpus, does the seeded arm
end with a strictly tighter box? **Kill criterion: if the seeded arm tightens ZERO
bounds corpus-wide, the mechanism is a no-op on real instances and Card 3e ships as a
measured negative** — exactly the outcome Card 2a and Card 3b recorded, and the
outcome the #727 RLT lesson says a synthetic-only validation would have missed.

**Soundness assertion, run on every instance.** Seeding can only shrink the initial
box and `backward_propagate` only tightens, so the seeded box must be *contained* in
the unseeded one. Any bound where the seeded arm is LOOSER is a defect, is counted,
and makes the probe exit non-zero. This is not a formality: it is the only automatic
check that the seed is a valid box rather than a stale one.

Prints an executed-comparison count and exits non-zero when it is zero (CLAUDE.md §6).
Exceptions are never swallowed (§7). The loaded module and a version marker are
asserted before any measurement (§8).

Usage::

    python -u discopt_benchmarks/scripts/card3e_fbbt_seed_entry.py
    python -u discopt_benchmarks/scripts/card3e_fbbt_seed_entry.py --subset 12
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "discopt_benchmarks") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "discopt_benchmarks"))
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))

from scripts.panel_baseline import _short_sha, corpus_instances, instance_path  # noqa: E402

_REPORTS_DIR = _REPO_ROOT / "reports"
_TIGHTEN_TOL = 1e-9


#: Root-presolve budget, in ms. Matches what production gives the pass on a 45 s
#: solve (`solver/__init__.py`: `min(max(0.25*time_limit, 2.0), 30.0)`), so the
#: comparison is of the pass as it actually runs. Unbounded is not an option: with
#: `time_limit_ms=0` this probe sat on `carton7` past a 300 s child timeout — the
#: #863 overrun, reproduced.
_PRESOLVE_BUDGET_MS = 11250


def _presolve_bounds(instance: str, seed: bool):
    """Root presolve on one instance; returns (lo, hi, stats-ish dict)."""
    import numpy as np
    from discopt._jax.presolve_pipeline import run_root_presolve
    from discopt._rust import model_to_repr
    from discopt.modeling.core import from_nl

    model = from_nl(str(instance_path(instance)))
    repr_ = model_to_repr(model, getattr(model, "_builder", None))
    t0 = time.perf_counter()
    _new, stats = run_root_presolve(
        repr_, fbbt_seed_from_ctx=seed, time_limit_ms=_PRESOLVE_BUDGET_MS
    )
    wall = time.perf_counter() - t0
    fb = stats.get("fbbt")
    if fb is None:
        return None, None, {"wall": wall, "no_fbbt_stats": True}
    return (
        np.asarray(fb["lb"], dtype=float),
        np.asarray(fb["ub"], dtype=float),
        {
            "wall": wall,
            "iterations": stats.get("iterations"),
            "terminated_by": stats.get("terminated_by"),
        },
    )


def _run_child(instance: str) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    import discopt
    import numpy as np
    from discopt._jax import presolve_pipeline

    out: dict = {"instance": instance, "discopt_file": discopt.__file__}
    # CLAUDE.md §8: a marker unique to the version under test. Its absence means the
    # child imported a pre-Card-3e discopt and every number below is from the wrong
    # tree. Asserted, not merely recorded.
    import inspect

    sig = inspect.signature(presolve_pipeline.run_root_presolve)
    if "fbbt_seed_from_ctx" not in sig.parameters:
        out["error"] = "MARKER ABSENT: run_root_presolve has no fbbt_seed_from_ctx"
        print("RESULT_JSON " + json.dumps(out), flush=True)
        return 2
    out["marker_ok"] = True

    lo0, hi0, s0 = _presolve_bounds(instance, seed=False)
    lo1, hi1, s1 = _presolve_bounds(instance, seed=True)
    out["off"] = s0
    out["on"] = s1
    # A row where the two arms stopped for different reasons is budget-dependent, not
    # a measurement of the seed. Flagged rather than silently averaged in.
    out["terminated_differs"] = s0.get("terminated_by") != s1.get("terminated_by")
    if lo0 is None or lo1 is None:
        out["comparable"] = False
        print("RESULT_JSON " + json.dumps(out), flush=True)
        return 0
    if lo0.shape != lo1.shape or hi0.shape != hi1.shape:
        out["comparable"] = False
        out["shape_mismatch"] = [list(lo0.shape), list(lo1.shape)]
        print("RESULT_JSON " + json.dumps(out), flush=True)
        return 0

    out["comparable"] = True
    out["n_bounds"] = int(2 * lo0.size)
    finite = np.isfinite(lo0) & np.isfinite(lo1)
    tighter_lo = int(np.sum(finite & (lo1 > lo0 + _TIGHTEN_TOL)))
    looser_lo = int(np.sum(finite & (lo1 < lo0 - _TIGHTEN_TOL)))
    finite_hi = np.isfinite(hi0) & np.isfinite(hi1)
    tighter_hi = int(np.sum(finite_hi & (hi1 < hi0 - _TIGHTEN_TOL)))
    looser_hi = int(np.sum(finite_hi & (hi1 > hi0 + _TIGHTEN_TOL)))
    # An infinite bound becoming finite is the strongest possible tightening and the
    # finite masks above would have hidden it.
    tighter_lo += int(np.sum(~np.isfinite(lo0) & np.isfinite(lo1)))
    tighter_hi += int(np.sum(~np.isfinite(hi0) & np.isfinite(hi1)))
    looser_lo += int(np.sum(np.isfinite(lo0) & ~np.isfinite(lo1)))
    looser_hi += int(np.sum(np.isfinite(hi0) & ~np.isfinite(hi1)))

    out["tightened"] = tighter_lo + tighter_hi
    out["loosened"] = looser_lo + looser_hi
    if out["loosened"]:
        idx = np.flatnonzero(finite & (lo1 < lo0 - _TIGHTEN_TOL))[:3]
        out["loosened_examples"] = [
            {"var": int(i), "off_lo": float(lo0[i]), "on_lo": float(lo1[i])} for i in idx
        ]
    print("RESULT_JSON " + json.dumps(out), flush=True)
    return 0


def _child(instance: str) -> dict:
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--one", instance]
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64="1")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300.0, env=env)
    except subprocess.TimeoutExpired:
        return {"instance": instance, "error": "child_timeout"}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON ") :])
    return {"instance": instance, "error": "child_no_result", "stderr_tail": proc.stderr[-500:]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--one", metavar="INSTANCE")
    ap.add_argument("--subset", type=int, default=0)
    args = ap.parse_args()
    if args.one:
        return _run_child(args.one)

    instances = sorted(corpus_instances())
    if args.subset:
        instances = instances[: args.subset]
    print(f"instances: {len(instances)}", flush=True)
    print(f"load at start: {os.getloadavg()}", flush=True)

    rows: list[dict] = []
    t0 = time.perf_counter()
    for i, inst in enumerate(instances, 1):
        row = _child(inst)
        rows.append(row)
        print(
            f"[{i}/{len(instances)}] {inst:<24} comparable={row.get('comparable')} "
            f"tightened={row.get('tightened')} loosened={row.get('loosened')} "
            f"{row.get('error', '')}",
            flush=True,
        )
    wall = time.perf_counter() - t0

    comparable = [r for r in rows if r.get("comparable")]
    executed = sum(int(r.get("n_bounds", 0)) for r in comparable)
    helped = [r for r in comparable if r.get("tightened", 0) > 0]
    loosened = [r for r in comparable if r.get("loosened", 0) > 0]
    total_tightened = sum(int(r.get("tightened", 0)) for r in comparable)

    print("\n## VERDICT", flush=True)
    print(f"  EXECUTED BOUND COMPARISONS : {executed}", flush=True)
    print(f"  comparable instances       : {len(comparable)} of {len(rows)}", flush=True)
    print(f"  instances tightened by seed: {len(helped)}", flush=True)
    for r in sorted(helped, key=lambda r: -r["tightened"]):
        print(f"      {r['instance']:<24} +{r['tightened']} bounds", flush=True)
    print(f"  TOTAL bounds tightened     : {total_tightened}", flush=True)
    budget_dep = [r for r in comparable if r.get("terminated_differs")]
    print(
        f"  budget-dependent rows (arms stopped differently): {len(budget_dep)} -> "
        f"{[r['instance'] for r in budget_dep]}",
        flush=True,
    )
    print(f"  SOUNDNESS: instances loosened : {len(loosened)}", flush=True)
    for r in loosened:
        print(f"      {r['instance']:<24} {r.get('loosened_examples')}", flush=True)
    print(f"  wall {wall:.1f}s  load at end {os.getloadavg()}", flush=True)

    _REPORTS_DIR.mkdir(exist_ok=True)
    path = _REPORTS_DIR / f"card3e_fbbt_seed_entry_{_short_sha()}.json"
    path.write_text(
        json.dumps(
            {
                "executed_bound_comparisons": executed,
                "total_tightened": total_tightened,
                "instances_helped": [r["instance"] for r in helped],
                "instances_loosened": [r["instance"] for r in loosened],
                "wall": wall,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"  artifact: {path}", flush=True)

    if executed == 0:
        print("FAIL: zero executed comparisons — the probe measured nothing", flush=True)
        return 1
    if loosened:
        print("FAIL: the seeded arm LOOSENED a bound — the seed is not a valid box", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
