#!/usr/bin/env python
"""Card 3e root-cause probe: why does the seeded root presolve call `casctanks` empty?

**What Card 3e left open.** `DISCOPT_FBBT_SEED` tightens 129 bounds on 9 of 119
instances but the seeded arm terminates `Infeasible` at orchestrator iteration 2 on
`casctanks`, identically at 11.25 s / 60 s / 300 s. Two hypotheses were named:

(a) `ctx.bounds` is an invalid seed for `ctx.model` — an orchestrator bookkeeping bug,
    hence a live defect in shipped code independent of the flag; or
(b) the composed box is genuinely empty to within `FEAS_TOL` — some pass's tightening
    is itself unsound.

**The distinguishing experiment** (written into the Card 3e block): run the FBBT
kernel from the *unseeded* orchestrator's own final box — the box that run already
certified non-empty across 16 sweeps. `PyModelRepr.in_tree_presolve` takes an
explicit `(node_lb, node_ub)` box and patches the model with it before running
`fbbt_with_cutoff`, so it *is* "the kernel, from a supplied box", and it is the
production per-node kernel rather than a bespoke harness.

**Third possibility this probe exists to catch, and the one it found.** Neither (a)
nor (b): the *emptiness test*. `orchestrator::any_empty` and
`bnb::in_tree_presolve` both use `Interval::is_empty()` — strict `lo > hi`, **zero
tolerance** — while `fbbt.rs`, `fbbt_fp.rs` and `probing.rs` all use
`is_empty_beyond(FEAS_TOL)`, whose own doc-comment says the strict form "mistakes
that numerical noise for infeasibility". So this probe does not merely report
"empty / not empty": it reports the **magnitude** of every crossing, split at
`FEAS_TOL`, which is what separates a genuine infeasibility from a rounding artifact.

Per-instance metrics (both arms):

- `terminated_by`, `iterations`
- `min_slack`   : min over blocks of `hi - lo` (negative ⇒ crossed)
- `n_noise`     : blocks with `0 < lo - hi <= FEAS_TOL`  — rounding artifacts
- `n_genuine`   : blocks with `lo - hi > FEAS_TOL`        — real infeasibility
- `E1`          : `in_tree_presolve` re-run from the OFF arm's own final box —
                  `infeasible` flag plus the worst crossing it produces
- `E1_declared` : control — the same kernel from the model's *declared* box

Every number is a comparison; the probe prints an executed-comparison count and
exits non-zero when it is zero (CLAUDE.md §6). Exceptions are never swallowed (§7).
The loaded module and a marker unique to the version under test are asserted before
any measurement (§8) — and the marker asserted here is the *compiled* one
(`PyModelRepr.presolve` accepting `fbbt_seed_from_ctx`), not merely the Python
signature: a stale `.so` with a fresh `presolve_pipeline.py` passes a
signature-only marker and measures the wrong tree.

Usage::

    python -u discopt_benchmarks/scripts/card3e_infeasible_root_cause.py
    python -u discopt_benchmarks/scripts/card3e_infeasible_root_cause.py --only casctanks
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

#: Presolve budget in ms — the same 11.25 s the Card 3e entry probe used, which is
#: what production gives the pass on a 45 s solve. Kept identical so this probe's
#: OFF arm is directly comparable to `reports/card3e_fbbt_seed_entry_*.json`.
_PRESOLVE_BUDGET_MS = 11250

#: `discopt_core::presolve::fbbt::FEAS_TOL`. The split point between "numerical
#: noise" and "genuine infeasibility"; the value is asserted against the Rust
#: constant's documented value rather than invented here.
_FEAS_TOL = 1e-6

_CHILD_TIMEOUT_S = 300


def _classify(lo, hi):
    """Crossing census for one bound vector pair."""
    import numpy as np

    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    slack = hi - lo
    finite = np.isfinite(slack)
    cross = lo - hi  # > 0 ⇔ empty
    n_noise = int(((cross > 0.0) & (cross <= _FEAS_TOL)).sum())
    n_genuine = int((cross > _FEAS_TOL).sum())
    worst_idx = int(np.argmax(np.where(np.isfinite(cross), cross, -np.inf)))
    return {
        "min_slack": float(slack[finite].min()) if finite.any() else None,
        "n_noise": n_noise,
        "n_genuine": n_genuine,
        "worst_cross": float(cross[worst_idx]),
        "worst_idx": worst_idx,
        "worst_lo": float(lo[worst_idx]),
        "worst_hi": float(hi[worst_idx]),
        "n_blocks": int(lo.size),
    }


def _run_child(instance: str) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    import discopt
    import numpy as np
    from discopt._jax.presolve_pipeline import run_root_presolve
    from discopt._rust import model_to_repr
    from discopt.modeling.core import from_nl

    out: dict = {"instance": instance, "discopt_file": discopt.__file__}

    # ── CLAUDE.md §8: assert the COMPILED marker, not the Python signature ──
    # A stale `.so` alongside a fresh `presolve_pipeline.py` satisfies a
    # signature-only check and then measures a tree without Card 3e in it.
    probe_model = from_nl(str(instance_path(instance)))
    probe_repr = model_to_repr(probe_model, getattr(probe_model, "_builder", None))
    try:
        probe_repr.presolve(passes=["fbbt"], max_iterations=1, fbbt_seed_from_ctx=True)
    except TypeError as exc:
        out["error"] = f"MARKER ABSENT (stale .so): {exc}"
        print("RESULT_JSON " + json.dumps(out), flush=True)
        return 2
    out["marker_ok"] = True

    comparisons = 0

    def arm(seed: bool):
        model = from_nl(str(instance_path(instance)))
        repr_ = model_to_repr(model, getattr(model, "_builder", None))
        t0 = time.perf_counter()
        new_repr, stats = run_root_presolve(
            repr_, fbbt_seed_from_ctx=seed, time_limit_ms=_PRESOLVE_BUDGET_MS
        )
        wall = time.perf_counter() - t0
        fb = stats.get("fbbt")
        if fb is None:
            return None, {"wall": wall, "no_fbbt_stats": True}
        lo = np.asarray(fb["lb"], dtype=float)
        hi = np.asarray(fb["ub"], dtype=float)
        rec = {
            "wall": wall,
            "iterations": stats.get("iterations"),
            "terminated_by": stats.get("terminated_by"),
            "n_vars_after": int(new_repr.n_vars),
        }
        rec.update(_classify(lo, hi))
        # Which pass ran last — on an `Infeasible` termination the orchestrator
        # breaks immediately after pushing that pass's delta, so the last entry
        # names the pass that emptied the box.
        deltas = stats.get("deltas") or []
        if deltas:
            rec["last_pass"] = deltas[-1]["pass_name"]
            rec["last_pass_iter"] = int(deltas[-1]["pass_iter"])
        return (new_repr, lo, hi), rec

    off, off_rec = arm(False)
    comparisons += off_rec.get("n_blocks", 0)
    out["off"] = off_rec
    on, on_rec = arm(True)
    comparisons += on_rec.get("n_blocks", 0)
    out["on"] = on_rec

    # ── E1: the distinguishing experiment ──────────────────────────────────
    # The FBBT kernel, from the UNSEEDED orchestrator's own final box, on the
    # unseeded orchestrator's own final model. If that box is a valid seed the
    # kernel cannot call it empty; if it does, the emptiness is not attributable
    # to the seed mechanism at all.
    if off is not None:
        new_repr, lo, hi = off
        n = int(new_repr.n_var_blocks)
        if lo.size == n:
            delta = new_repr.in_tree_presolve(
                lo.tolist(), hi.tolist(), node_depth=0, depth_stride=1, probing=False
            )
            e1 = {
                "infeasible": bool(delta["infeasible"]),
                "ran": bool(delta["ran"]),
                "bounds_tightened": int(delta["bounds_tightened"]),
            }
            e1.update(_classify(np.asarray(delta["lb"]), np.asarray(delta["ub"])))
            out["E1_from_off_final_box"] = e1
            comparisons += n

            # Control: the same kernel from the model's DECLARED box. Isolates
            # "the kernel dislikes this model" from "the kernel dislikes this box".
            dlo = np.asarray([np.min(new_repr.var_lb(i)) for i in range(n)], dtype=float)
            dhi = np.asarray([np.max(new_repr.var_ub(i)) for i in range(n)], dtype=float)
            dctl = new_repr.in_tree_presolve(
                dlo.tolist(), dhi.tolist(), node_depth=0, depth_stride=1, probing=False
            )
            c = {
                "infeasible": bool(dctl["infeasible"]),
                "bounds_tightened": int(dctl["bounds_tightened"]),
            }
            c.update(_classify(np.asarray(dctl["lb"]), np.asarray(dctl["ub"])))
            out["E1_declared_control"] = c
            comparisons += n

    out["comparisons"] = comparisons
    print("RESULT_JSON " + json.dumps(out), flush=True)
    return 0


def _parent(args) -> int:
    instances = corpus_instances()
    if args.only:
        wanted = set(args.only.split(","))
        instances = [i for i in instances if i in wanted]
        missing = wanted - set(instances)
        if missing:
            raise SystemExit(f"unknown instance(s): {sorted(missing)}")
    if args.subset:
        instances = instances[: args.subset]

    load_start = os.getloadavg()[0]
    print(f"# instances: {len(instances)}   load(1m) at start: {load_start:.2f}", flush=True)
    print(f"# presolve budget: {_PRESOLVE_BUDGET_MS} ms   FEAS_TOL: {_FEAS_TOL:g}", flush=True)

    rows: list[dict] = []
    load_peak = load_start
    t_all = time.perf_counter()
    for k, inst in enumerate(instances, 1):
        proc = subprocess.run(
            [sys.executable, "-u", str(Path(__file__).resolve()), "--child", inst],
            capture_output=True,
            text=True,
            timeout=_CHILD_TIMEOUT_S,
            cwd=str(_REPO_ROOT),
        )
        row: dict = {"instance": inst, "rc": proc.returncode}
        for line in proc.stdout.splitlines():
            if line.startswith("RESULT_JSON "):
                row.update(json.loads(line[len("RESULT_JSON ") :]))
        if proc.returncode != 0 and "error" not in row:
            row["error"] = (proc.stderr or "")[-800:]
        rows.append(row)
        load_peak = max(load_peak, os.getloadavg()[0])

        off = row.get("off", {})
        on = row.get("on", {})
        e1 = row.get("E1_from_off_final_box", {})
        print(
            f"[{k:3d}/{len(instances)}] {inst:24s} rc={row['rc']} "
            f"OFF[{off.get('terminated_by')},it={off.get('iterations')},"
            f"noise={off.get('n_noise')},real={off.get('n_genuine')},"
            f"minslack={off.get('min_slack')}] "
            f"ON[{on.get('terminated_by')},it={on.get('iterations')},"
            f"noise={on.get('n_noise')},real={on.get('n_genuine')},"
            f"last={on.get('last_pass')}] "
            f"E1[infeas={e1.get('infeasible')},cross={e1.get('worst_cross')}]",
            flush=True,
        )

    wall = time.perf_counter() - t_all

    # ── VERDICT ────────────────────────────────────────────────────────────
    executed = sum(int(r.get("comparisons", 0)) for r in rows)
    ok = [r for r in rows if r.get("marker_ok") and "off" in r and "on" in r]
    off_infeas = [r for r in ok if r["off"].get("terminated_by") == "Infeasible"]
    on_infeas = [r for r in ok if r["on"].get("terminated_by") == "Infeasible"]
    off_noise = [r for r in ok if r["off"].get("n_noise", 0) > 0]
    on_noise = [r for r in ok if r["on"].get("n_noise", 0) > 0]
    off_real = [r for r in ok if r["off"].get("n_genuine", 0) > 0]
    on_real = [r for r in ok if r["on"].get("n_genuine", 0) > 0]
    e1_infeas = [r for r in ok if r.get("E1_from_off_final_box", {}).get("infeasible")]
    ctl_infeas = [r for r in ok if r.get("E1_declared_control", {}).get("infeasible")]

    print("\n## VERDICT", flush=True)
    print(f"  EXECUTED BOUND COMPARISONS : {executed}", flush=True)
    print(f"  comparable instances       : {len(ok)} of {len(rows)}", flush=True)
    print(f"  wall {wall:.1f}s   load {load_start:.2f} -> {load_peak:.2f}", flush=True)
    off_names = [r["instance"] for r in off_infeas]
    on_names = [r["instance"] for r in on_infeas]
    print(f"  OFF terminated Infeasible  : {len(off_infeas)}  {off_names}")
    print(f"  ON  terminated Infeasible  : {len(on_infeas)}  {on_names}")
    print("  --- crossing census, split at FEAS_TOL ---", flush=True)
    print(f"  OFF instances w/ NOISE crossings (0 < lo-hi <= 1e-6) : {len(off_noise)}")
    for r in off_noise:
        d = r["off"]
        print(
            f"      {r['instance']:24s} n={d['n_noise']} worst={d['worst_cross']:.6e} "
            f"var{d['worst_idx']} [{d['worst_lo']!r}, {d['worst_hi']!r}]"
        )
    print(f"  OFF instances w/ GENUINE crossings (lo-hi > 1e-6)    : {len(off_real)}")
    for r in off_real:
        d = r["off"]
        print(f"      {r['instance']:24s} n={d['n_genuine']} worst={d['worst_cross']:.6e}")
    print(f"  ON  instances w/ NOISE crossings                     : {len(on_noise)}")
    for r in on_noise:
        d = r["on"]
        print(
            f"      {r['instance']:24s} n={d['n_noise']} worst={d['worst_cross']:.6e} "
            f"var{d['worst_idx']} [{d['worst_lo']!r}, {d['worst_hi']!r}] "
            f"last_pass={d.get('last_pass')}@it{d.get('last_pass_iter')}"
        )
    print(f"  ON  instances w/ GENUINE crossings                   : {len(on_real)}")
    for r in on_real:
        d = r["on"]
        print(f"      {r['instance']:24s} n={d['n_genuine']} worst={d['worst_cross']:.6e}")
    print("  --- E1: FBBT kernel from the UNSEEDED arm's own certified final box ---")
    print(f"  E1 declared the certified box EMPTY on : {len(e1_infeas)} instance(s)")
    for r in e1_infeas:
        d = r["E1_from_off_final_box"]
        print(
            f"      {r['instance']:24s} worst_cross={d['worst_cross']:.6e} "
            f"noise={d['n_noise']} genuine={d['n_genuine']}"
        )
    print(f"  E1 declared the DECLARED box empty on  : {len(ctl_infeas)} instance(s) (control)")
    for r in ctl_infeas:
        print(f"      {r['instance']:24s} worst={r['E1_declared_control']['worst_cross']:.6e}")

    bad = [r for r in rows if r.get("rc", 1) != 0]
    if bad:
        print(f"  children with rc != 0 : {len(bad)} {[r['instance'] for r in bad]}")

    artifact = _REPORTS_DIR / f"card3e_infeasible_root_cause_{_short_sha()}.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "schema": "card3e_infeasible_root_cause/1",
                "git_sha": _short_sha(),
                "budget_ms": _PRESOLVE_BUDGET_MS,
                "feas_tol": _FEAS_TOL,
                "executed": executed,
                "load_start": load_start,
                "load_peak": load_peak,
                "total_wall_seconds": wall,
                "instrument_marker": "compiled presolve(fbbt_seed_from_ctx=) accepted",
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nartifact: {artifact}", flush=True)

    # CLAUDE.md §6: a probe that compared nothing is a failed probe, not a pass.
    if executed == 0:
        print("FAIL: zero executed comparisons — the probe measured nothing.", flush=True)
        return 1
    if bad:
        print(f"FAIL: {len(bad)} child(ren) exited non-zero.", flush=True)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--child", metavar="INSTANCE", default=None)
    ap.add_argument("--only", default=None, help="comma-separated instance names")
    ap.add_argument("--subset", type=int, default=0)
    args = ap.parse_args()
    if args.child:
        return _run_child(args.child)
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
