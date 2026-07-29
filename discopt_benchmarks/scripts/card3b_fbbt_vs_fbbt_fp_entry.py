"""Card 3b **entry experiment** — `fbbt` (sweep) vs `fbbt_fp` (watch-list).

Consolidation plan Card 3b. ``crates/discopt-core/src/presolve/fbbt_fp.rs`` claims
in its own header to *supersede* the wired-in sweep FBBT in ``fbbt.rs``:

    The existing iterative-with-cap FBBT in `fbbt.rs` (B4 reference) visits every
    constraint on every sweep. That is wasteful when only a small part of the model
    has changed since the last sweep, and oscillates in the tail of convergence.

That claim has never been measured, and the pass has never been enabled on any
solve path.  Phase 1 Card 1b therefore parked the file instead of deleting it,
with a header comment naming this card.  This is the measurement.

## The two arms

Both passes are run through the **same** orchestrator entry point
(``PyModelRepr.presolve``), alone, from the **same** starting repr, with the same
tolerance, iterated until the orchestrator reports no further progress.  Running
them alone is the point: bundling either with ``eliminate``/``simplify`` would
measure the bundle, and any bound difference would be unattributable.

* arm ``fbbt``     — ``passes=["fbbt"]``,             ``fbbt_max_iter`` sweeps per invocation
* arm ``fbbt_fp``  — ``passes=["fbbt_fixed_point"]``, work-queue to an empty queue

## What decides the card

1. **Fixpoint equality.** Both are FBBT over the same DAG with the same
   forward/backward kernels (``fbbt_fp`` literally imports ``forward_propagate`` /
   ``backward_propagate`` from ``fbbt``), so on any instance where both report a
   converged (non-capped) run their fixpoints must agree.  **A bound difference is
   a bug in one of them, not a tuning knob** — the card says investigate and
   report before proceeding, do not average over it.  This probe classifies every
   disagreement by direction (which arm is tighter) and by whether either arm hit
   its work/iteration cap, because a capped arm being looser is an *expected*
   truncation rather than a soundness divergence.
2. **Wall time to that fixpoint**, interleaved A/B/A/B within one process across
   ``--replicates`` rounds, reported with a standard deviation (CLAUDE.md §9).

## Instrumentation contract

Per CLAUDE.md §6 the probe prints an executed **bound-comparison count** and exits
non-zero when it is zero — a run that compared nothing must not read as a pass.
Per §7 nothing around a measurement swallows an exception: a child that fails
records the exception and the parent counts it as an error, it does not silently
drop the instance.  Per §8 each child asserts and reports ``discopt.__file__`` and
``discopt._rust.__file__`` before measuring.

Usage::

    python -u discopt_benchmarks/scripts/card3b_fbbt_vs_fbbt_fp_entry.py
    python -u discopt_benchmarks/scripts/card3b_fbbt_vs_fbbt_fp_entry.py --subset 20
    python -u discopt_benchmarks/scripts/card3b_fbbt_vs_fbbt_fp_entry.py --replicates 5
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

_CHILD_TIMEOUT = 600.0

#: Relative tolerance for calling two bounds equal. Both arms carry the same
#: outward-rounded interval arithmetic, so a genuine fixpoint match is exact or
#: near-exact; this only absorbs the last-ulp differences a different *visit
#: order* can leave behind in floating-point accumulation.
_REL_TOL = 1e-12
_ABS_TOL = 1e-12


def _close(a: float, b: float) -> bool:
    if a == b:
        return True
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return False
    return abs(a - b) <= max(_ABS_TOL, _REL_TOL * max(abs(a), abs(b)))


# --------------------------------------------------------------------------
# Child: one instance, both arms, interleaved replicates.
# --------------------------------------------------------------------------


def _run_child(instance: str, replicates: int) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    import discopt  # noqa: PLC0415
    from discopt._rust import model_to_repr  # noqa: PLC0415
    from discopt.modeling.core import from_nl  # noqa: PLC0415

    from scripts.panel_baseline import instance_path  # noqa: PLC0415

    rust_mod = sys.modules["discopt._rust"]
    out: dict = {
        "instance": instance,
        "discopt_file": discopt.__file__,
        "rust_file": getattr(rust_mod, "__file__", None),
        "replicates": replicates,
    }
    # CLAUDE.md §8: prove which code is loaded before measuring anything.
    assert str(_REPO_ROOT) in str(discopt.__file__), (
        f"discopt loaded from outside the repo: {discopt.__file__}"
    )

    try:
        model = from_nl(str(instance_path(instance)))
        repr_ = model_to_repr(model, getattr(model, "_builder", None))
        out["n_var_blocks"] = int(repr_.n_var_blocks)

        def _arm(pass_name: str) -> dict:
            t0 = time.perf_counter()
            _new, stats = repr_.presolve(
                passes=[pass_name],
                max_iterations=16,
                time_limit_ms=0,
                fbbt_max_iter=20,
                fbbt_tol=1e-8,
            )
            wall = time.perf_counter() - t0
            return {
                "wall": wall,
                "lo": [float(v) for v in stats["bounds_lo"]],
                "hi": [float(v) for v in stats["bounds_hi"]],
                "iterations": int(stats["iterations"]),
                "terminated_by": str(stats["terminated_by"]),
                "work_units": sum(int(d.get("work_units", 0) or 0) for d in stats["deltas"]),
                "bounds_tightened": sum(int(d.get("bounds_tightened", 0)) for d in stats["deltas"]),
            }

        walls: dict[str, list[float]] = {"fbbt": [], "fbbt_fp": []}
        last: dict[str, dict] = {}
        # Interleaved A/B/A/B, not all-A-then-all-B (CLAUDE.md §9).
        for _ in range(replicates):
            for name, pass_name in (("fbbt", "fbbt"), ("fbbt_fp", "fbbt_fixed_point")):
                res = _arm(pass_name)
                walls[name].append(res["wall"])
                last[name] = res

        a, b = last["fbbt"], last["fbbt_fp"]
        out["fbbt"] = {k: v for k, v in a.items() if k not in ("lo", "hi")}
        out["fbbt_fp"] = {k: v for k, v in b.items() if k not in ("lo", "hi")}
        out["walls"] = walls

        # ---- fixpoint comparison -----------------------------------------
        n = min(len(a["lo"]), len(b["lo"]))
        out["n_blocks_compared"] = n
        out["len_mismatch"] = len(a["lo"]) != len(b["lo"])
        comparisons = 0
        diffs: list[dict] = []
        fp_tighter = fbbt_tighter = incomparable = 0
        for i in range(n):
            for side, av, bv in (("lo", a["lo"][i], b["lo"][i]), ("hi", a["hi"][i], b["hi"][i])):
                comparisons += 1
                if _close(av, bv):
                    continue
                # `lo` tighter means larger; `hi` tighter means smaller.
                if side == "lo":
                    tighter = "fbbt_fp" if bv > av else "fbbt"
                else:
                    tighter = "fbbt_fp" if bv < av else "fbbt"
                if tighter == "fbbt_fp":
                    fp_tighter += 1
                else:
                    fbbt_tighter += 1
                if len(diffs) < 25:
                    diffs.append(
                        {"block": i, "side": side, "fbbt": av, "fbbt_fp": bv, "tighter": tighter}
                    )
        out["comparisons"] = comparisons
        out["diff_count"] = fp_tighter + fbbt_tighter
        out["fp_tighter"] = fp_tighter
        out["fbbt_tighter"] = fbbt_tighter
        out["incomparable"] = incomparable
        out["diffs"] = diffs
    except Exception as exc:  # recorded, never swallowed (CLAUDE.md §7)
        import traceback

        out["error"] = repr(exc)
        out["traceback"] = traceback.format_exc()

    print("RESULT_JSON " + json.dumps(out), flush=True)
    return 0


# --------------------------------------------------------------------------
# Parent
# --------------------------------------------------------------------------


def _run_one(instance: str, replicates: int) -> dict:
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--solve",
        instance,
        "--replicates",
        str(replicates),
    ]
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("JAX_ENABLE_X64", "1")
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_CHILD_TIMEOUT,
            env=env,
            cwd=str(_BENCH_ROOT),
        )
    except subprocess.TimeoutExpired:
        return {"instance": instance, "error": "child_timeout"}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON ") :])
    return {
        "instance": instance,
        "error": "no result line",
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr[-2000:],
    }


def _stats(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return (float("nan"), float("nan"))
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return (m, 0.0)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return (m, math.sqrt(var))


# --------------------------------------------------------------------------
# Diagnosis mode: WHY the two fixpoints differ. Run after the A/B, in-process.
# --------------------------------------------------------------------------

_DIAG_INSTANCES = ("4stufen", "util", "hda", "st_e31", "beuster", "gbd", "st_e03")


def _diagnose() -> int:
    """Attribute the fixpoint gap to a mechanism, with executed check counts.

    Three hypotheses, each with its own falsifiable check:

    H1 — **non-composition.** ``FbbtPass::run`` calls
    ``fbbt_with_cutoff_until(&ctx.model, ...)``, which seeds ``var_bounds`` from
    ``model.variables`` (the *declared* box) and never reads ``ctx.bounds``; the
    adapter then intersects.  ``FbbtFixedPointPass`` propagates in place from
    ``ctx.bounds``.  The orchestrator deliberately never writes tightened bounds
    back into ``ctx.model`` (documented at ``orchestrator.rs``, for LP-dual
    validity), so ``fbbt`` structurally cannot see another pass's tightenings.
    Check: at ``max_iterations=1`` — so no pass can re-run and launder the
    composition — ``[prior, X]`` must beat ``intersect(X@1, prior@1)`` iff ``X``
    composes.

    H2 — **fixpoint order-dependence.** FBBT has a unique greatest fixpoint only
    on monotone/linear systems.  On the cyclic nonconvex DAGs in this corpus the
    reachable fixpoint depends on visit order, so *neither* pass dominates and
    both boxes are valid outer approximations.  Check: run them in both orders
    and show each composition beats both singletons.

    H3 — **no termination at the claimed fixed point.** ``fbbt_fp``'s header
    claims it "terminates the moment the queue is empty — that's the true fixed
    point, no ``max_iter`` artefact".  Check: report ``bounds_tightened`` per
    orchestrator sweep and ``terminated_by``.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    import discopt  # noqa: PLC0415
    from discopt._rust import model_to_repr  # noqa: PLC0415
    from discopt.modeling.core import from_nl  # noqa: PLC0415

    from scripts.panel_baseline import instance_path  # noqa: PLC0415

    assert str(_REPO_ROOT) in str(discopt.__file__), discopt.__file__
    print(f"[diag] discopt={discopt.__file__}")
    print(f"[diag] _rust={sys.modules['discopt._rust'].__file__}")

    def run(inst, passes, iters):
        m = from_nl(str(instance_path(inst)))
        r = model_to_repr(m, getattr(m, "_builder", None))
        _n, st = r.presolve(passes=passes, max_iterations=iters, fbbt_max_iter=20, fbbt_tol=1e-8)
        return (
            [float(v) for v in st["bounds_lo"]],
            [float(v) for v in st["bounds_hi"]],
            str(st["terminated_by"]),
            [(str(d["pass_name"]), int(d["bounds_tightened"])) for d in st["deltas"]],
        )

    executed = 0
    print("\n=== H1: does the pass COMPOSE with a prior pass? (1 sweep, no re-runs) ===")
    composed = {"fbbt": 0, "fbbt_fixed_point": 0}
    for inst in _DIAG_INSTANCES:
        for pass_name in ("fbbt", "fbbt_fixed_point"):
            alone = run(inst, [pass_name], 1)
            prior = run(inst, ["implied_bounds"], 1)
            comb = run(inst, ["implied_bounds", pass_name], 1)
            n = len(alone[0])
            ilo = [max(alone[0][i], prior[0][i]) for i in range(n)]
            ihi = [min(alone[1][i], prior[1][i]) for i in range(n)]
            beyond = sum(
                1 for i in range(n) if comb[0][i] > ilo[i] + 1e-9 or comb[1][i] < ihi[i] - 1e-9
            )
            composed[pass_name] += beyond
            executed += 1
            print(f"  {inst:9s} {pass_name:17s}: composed on {beyond:3d} bounds")
    print(f"  TOTAL composed-beyond-intersection bounds: {composed}")
    print(f"  [executed composition checks: {executed}]")

    print("\n=== H2: order-dependence — neither fixpoint dominates ===")
    for inst in ("st_e03",):
        rows = [
            ("fbbt", run(inst, ["fbbt"], 16)),
            ("fbbt_fp", run(inst, ["fbbt_fixed_point"], 16)),
            ("fbbt->fp", run(inst, ["fbbt", "fbbt_fixed_point"], 16)),
            ("fp->fbbt", run(inst, ["fbbt_fixed_point", "fbbt"], 16)),
        ]
        for nm, (lo, _hi, term, _d) in rows:
            executed += 1
            print(f"  {inst} {nm:10s} [{term:12s}] block2.lo={lo[2]!r} block4.lo={lo[4]!r}")

    print("\n=== H3: does fbbt_fp actually terminate at its fixed point? ===")
    for inst in ("4stufen", "util"):
        lo, hi, term, deltas = run(inst, ["fbbt_fixed_point"], 16)
        lo1, hi1, _t1, _d1 = run(inst, ["fbbt_fixed_point"], 1)
        same = sum(1 for i in range(len(lo)) if lo[i] == lo1[i] and hi[i] == hi1[i])
        executed += 2
        per_sweep = [b for _, b in deltas]
        print(f"  {inst}: terminated_by={term}; per-sweep bounds_tightened={per_sweep}")
        print(f"      blocks identical between max_iterations 1 and 16: {same}/{len(lo)}")

    print(f"\n[diag] EXECUTED CHECKS: {executed}")
    if executed == 0:
        print("FAIL: the diagnosis executed nothing")
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solve", metavar="INSTANCE")
    ap.add_argument("--diagnose", action="store_true", help="attribute the fixpoint gap")
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--subset", type=int, default=0, help="first N instances only")
    ap.add_argument("--out", default=str(_REPO_ROOT / "reports" / "card3b_fbbt_vs_fbbt_fp.json"))
    args = ap.parse_args()

    if args.solve:
        return _run_child(args.solve, args.replicates)
    if args.diagnose:
        return _diagnose()

    from scripts.panel_baseline import corpus_instances  # noqa: PLC0415

    instances = list(corpus_instances())
    if args.subset:
        instances = instances[: args.subset]

    print(f"[card3b] {len(instances)} instances, {args.replicates} interleaved replicates each")
    print(f"[card3b] load at start: {os.getloadavg()}")

    rows: list[dict] = []
    t0 = time.perf_counter()
    for k, inst in enumerate(instances, 1):
        row = _run_one(inst, args.replicates)
        rows.append(row)
        if "error" in row:
            print(f"[{k}/{len(instances)}] {inst}: ERROR {row['error']}", flush=True)
            continue
        fm, fs = _stats(row["walls"]["fbbt"])
        pm, ps = _stats(row["walls"]["fbbt_fp"])
        print(
            f"[{k}/{len(instances)}] {inst}: cmp={row['comparisons']} "
            f"diff={row['diff_count']} (fp_tighter={row['fp_tighter']}, "
            f"fbbt_tighter={row['fbbt_tighter']})  "
            f"fbbt {fm * 1e3:.2f}±{fs * 1e3:.2f}ms [{row['fbbt']['terminated_by']}]  "
            f"fbbt_fp {pm * 1e3:.2f}±{ps * 1e3:.2f}ms [{row['fbbt_fp']['terminated_by']}]",
            flush=True,
        )

    wall = time.perf_counter() - t0
    ok = [r for r in rows if "error" not in r]
    errs = [r for r in rows if "error" in r]
    total_cmp = sum(r["comparisons"] for r in ok)
    total_diff = sum(r["diff_count"] for r in ok)
    disagreeing = [r for r in ok if r["diff_count"] > 0]
    fp_t = sum(r["fp_tighter"] for r in ok)
    fb_t = sum(r["fbbt_tighter"] for r in ok)

    # Timing aggregate over instances where BOTH arms converged (a capped arm's
    # wall is a budget, not a fixpoint cost, so it does not belong in the ratio).
    converged = [
        r
        for r in ok
        if r["fbbt"]["terminated_by"] == "NoProgress"
        and r["fbbt_fp"]["terminated_by"] == "NoProgress"
    ]
    tot_fbbt = sum(_stats(r["walls"]["fbbt"])[0] for r in converged)
    tot_fp = sum(_stats(r["walls"]["fbbt_fp"])[0] for r in converged)

    print("\n================ CARD 3b ENTRY EXPERIMENT ================")
    print(f"instances: {len(rows)} ({len(ok)} ok, {len(errs)} errored)")
    print(f"EXECUTED BOUND COMPARISONS: {total_cmp}")
    print(f"instances with a fixpoint disagreement: {len(disagreeing)} / {len(ok)}")
    print(f"disagreeing bounds: {total_diff}  (fbbt_fp tighter {fp_t}, fbbt tighter {fb_t})")
    print(f"both-converged instances: {len(converged)} / {len(ok)}")
    print(f"summed mean wall  fbbt={tot_fbbt:.3f}s  fbbt_fp={tot_fp:.3f}s")
    if tot_fbbt > 0:
        print(f"fbbt_fp / fbbt wall ratio (converged set): {tot_fp / tot_fbbt:.3f}")
    print(f"harness wall: {wall:.1f}s   load at end: {os.getloadavg()}")

    if disagreeing:
        print("\n--- fixpoint disagreements (first 10 instances) ---")
        for r in disagreeing[:10]:
            print(
                f"  {r['instance']}: {r['diff_count']} bounds; "
                f"fbbt[{r['fbbt']['terminated_by']}, iters={r['fbbt']['iterations']}] "
                f"fbbt_fp[{r['fbbt_fp']['terminated_by']}, "
                f"visits={r['fbbt_fp']['work_units']}]"
            )
            for d in r["diffs"][:4]:
                print(
                    f"      block {d['block']}.{d['side']}: fbbt={d['fbbt']!r} "
                    f"fbbt_fp={d['fbbt_fp']!r} -> {d['tighter']} tighter"
                )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "rows": rows,
                "summary": {
                    "n_instances": len(rows),
                    "n_ok": len(ok),
                    "n_errored": len(errs),
                    "executed_comparisons": total_cmp,
                    "disagreeing_instances": len(disagreeing),
                    "disagreeing_bounds": total_diff,
                    "fp_tighter": fp_t,
                    "fbbt_tighter": fb_t,
                    "converged_instances": len(converged),
                    "wall_fbbt": tot_fbbt,
                    "wall_fbbt_fp": tot_fp,
                    "replicates": args.replicates,
                },
            },
            indent=2,
        )
    )
    print(f"\nwrote {out_path}")

    # CLAUDE.md §6: zero executed comparisons is a FAILURE, not a pass.
    if total_cmp == 0:
        print("FAIL: zero bound comparisons executed — the probe measured nothing")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
