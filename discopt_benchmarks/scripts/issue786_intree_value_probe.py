#!/usr/bin/env python
"""Issue #786 entry experiment — is IN-TREE cutting worth building?

Hypothesis (#786): separating GMI/c-MIR at DESCENDANT node boxes (over their
FBBT-tightened boxes), on top of the inherited root cuts, tightens the node
dual bound enough to reduce nodes-to-certify and close a real fraction of the
discopt->SCIP wall-clock gap on the convex synthesis family.

Prior (#781): discopt's separators SATURATE after <=3 rounds; the root-cut
stage closes 75-93% of the ROOT spread but was HELD (primal starvation) and
does not reach SCIP wall-clock. So there is a real chance in-tree cutting adds
little beyond (inherited root cuts + child FBBT) — a FRONTIER (#786 option b).

This probe measures exactly that delta, out-of-solver, on the real panel:

  1. Build the root LP (linear rows + OA tangents), run the #781 root cut loop
     (GMI + c-MIR + cover under pooled top-K selection) to convergence ->
     root cuts + root LP vertex x_root.
  2. Branch on each of the top-K fractional integers (both directions) ->
     child boxes; FBBT-tighten each child box.
  3. Per child, compute TWO bounds (excess over the known optimum):
       BASE = child LP with (inherited root cuts + child FBBT + OA reconverged),
              NO new separation — what the current root-cut stage already gives
              at a descendant node (the cuts ride in every node relaxation).
       CUT  = BASE + a child-box separation loop (GMI/c-MIR/cover over the
              child box) — what IN-TREE cutting would add.
  4. Report the tightening delta = BASE_excess - CUT_excess, and as a fraction
     of the child's remaining gap to the optimum.

Kill criterion: if median child tightening closes < 10% of the child's
remaining gap across the panel, in-tree cutting is FALSIFIED as a lever here
(record the frontier per #786 option b). If it closes materially, in-tree
cutting is the lever and the in-solver build is justified.

THROWAWAY probe — no shipped solver code.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["DISCOPT_COEF_TIGHTEN"] = "1"

import numpy as np

# Reuse the validated #781 machinery (RootModel, HiGHS LP, GMI, pool, selection).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discopt._jax.cmir_cuts import separate_cmir  # noqa: E402
from discopt._jax.cover_cuts import separate_cover_cuts  # noqa: E402
from issue781_cutmgmt_probe import (  # noqa: E402
    PANEL,
    CutPool,
    RootModel,
    select_cuts,
    separate_gmi,
    solve_lp_highs,
)

ROUNDS = 20
OA_TOL = 1e-6
CUT_VIOL_TOL = 1e-6
BRANCH_VARS = 3  # top-K fractional integers to branch on
KILL_BAR = 10.0  # % of child remaining gap that in-tree cutting must close


def excess(bound, opt):
    return (bound - opt) / abs(opt) * 100.0


def child_fbbt(rm, lo, hi):
    """FBBT the child box using the root linear rows (+ eq both ways)."""
    lo = lo.copy()
    hi = hi.copy()
    rows_a = list(rm.A_le) + list(rm.A_eq) + list(-rm.A_eq)
    rows_b = list(rm.b_le) + list(rm.b_eq) + list(-rm.b_eq)
    for _ in range(20):
        changed = False
        for a, b in zip(rows_a, rows_b):
            nz = np.where(np.abs(a) > 1e-12)[0]
            for j in nz:
                aj = a[j]
                rest, ok = 0.0, True
                for k in nz:
                    if k == j:
                        continue
                    v = lo[k] if a[k] > 0 else hi[k]
                    if not np.isfinite(v):
                        ok = False
                        break
                    rest += a[k] * v
                if not ok:
                    continue
                bound = (b - rest) / aj
                if aj > 0 and bound < hi[j] - 1e-9:
                    hi[j] = bound
                    changed = True
                elif aj < 0 and bound > lo[j] + 1e-9:
                    lo[j] = bound
                    changed = True
        if not changed:
            break
    return np.maximum(lo, rm.lb_sep), np.minimum(hi, rm.ub_sep)


def solve_over_box(rm, lo, hi, cuts_a, cuts_b, separate, is_int, lb_s, ub_s):
    """Solve the LP over [lo,hi] with OA reconverged + inherited cuts; optionally
    run a child-box separation loop. Returns the final bound (model sense)."""
    # temporarily install the child box on a shallow copy of rm's bounds
    saved_lb, saved_ub = rm.lb, rm.ub
    rm.lb, rm.ub = lo, hi
    try:
        ca = list(cuts_a)
        cb = list(cuts_b)
        pool = CutPool()

        def oa_converge():
            obj, x, duals, h, _nle = solve_lp_highs(rm, ca, cb)
            for _ in range(60):
                if x is None:
                    break
                added = 0
                for i, v in rm.nonlinear_violation(x).items():
                    if v > OA_TOL:
                        t = rm.oa_tangent(i, x)
                        if t is not None:
                            ca.append(t[0])
                            cb.append(t[1])
                            added += 1
                if added == 0:
                    break
                obj, x, duals, h, _nle = solve_lp_highs(rm, ca, cb)
            return obj, x, duals, h

        obj, x, duals, h = oa_converge()
        if not separate or x is None:
            return obj
        for _ in range(ROUNDS):
            if x is None:
                break
            a_all = np.vstack([rm.A_le] + ([np.array(ca)] if ca else []))
            b_all = np.concatenate([rm.b_le] + ([np.array(cb)] if cb else []))
            cands = list(
                separate_cmir(a_all, b_all, x, lb_s, ub_s, is_int, max_cuts=24, duals=duals)
            )
            for cover, rhs in separate_cover_cuts(a_all, b_all, x, rm.is_bin, max_cuts=32):
                arr = np.zeros(rm.n)
                arr[list(cover)] = 1.0
                cands.append((arr, float(rhs)))
            if h is not None:
                cands += separate_gmi(rm, h, x, a_all, b_all, rm.A_eq.shape[0])
            for arr, r in cands:
                arr = np.asarray(arr, float)
                if arr @ x - float(r) > CUT_VIOL_TOL:
                    pool.offer(arr, float(r))
            chosen = select_cuts(pool.violated(x), x)
            if not chosen:
                break
            for arr, r in chosen:
                ca.append(arr)
                cb.append(r)
            obj, x, duals, h = oa_converge()
        return obj
    finally:
        rm.lb, rm.ub = saved_lb, saved_ub


def root_cut_loop(rm, is_int, lb_s, ub_s):
    """Run the #781 root cut loop; return (root cuts, root LP vertex, root bound)."""
    ca, cb = [], []
    pool = CutPool()

    def oa_converge():
        obj, x, duals, h, _nle = solve_lp_highs(rm, ca, cb)
        for _ in range(60):
            if x is None:
                break
            added = 0
            for i, v in rm.nonlinear_violation(x).items():
                if v > OA_TOL:
                    t = rm.oa_tangent(i, x)
                    if t is not None:
                        ca.append(t[0])
                        cb.append(t[1])
                        added += 1
            if added == 0:
                break
            obj, x, duals, h, _nle = solve_lp_highs(rm, ca, cb)
        return obj, x, duals, h

    obj, x, duals, h = oa_converge()
    for _ in range(ROUNDS):
        if x is None:
            break
        a_all = np.vstack([rm.A_le] + ([np.array(ca)] if ca else []))
        b_all = np.concatenate([rm.b_le] + ([np.array(cb)] if cb else []))
        cands = list(separate_cmir(a_all, b_all, x, lb_s, ub_s, is_int, max_cuts=24, duals=duals))
        for cover, rhs in separate_cover_cuts(a_all, b_all, x, rm.is_bin, max_cuts=32):
            arr = np.zeros(rm.n)
            arr[list(cover)] = 1.0
            cands.append((arr, float(rhs)))
        if h is not None:
            cands += separate_gmi(rm, h, x, a_all, b_all, rm.A_eq.shape[0])
        for arr, r in cands:
            arr = np.asarray(arr, float)
            if arr @ x - float(r) > CUT_VIOL_TOL:
                pool.offer(arr, float(r))
        chosen = select_cuts(pool.violated(x), x)
        if not chosen:
            break
        for arr, r in chosen:
            ca.append(arr)
            cb.append(r)
        obj, x, duals, h = oa_converge()
    return ca, cb, x, obj


def main():
    results = {}
    for name, info in PANEL.items():
        opt = info["opt"]
        print(f"\n===== {name} opt={opt} =====", flush=True)
        rm = RootModel(name)
        is_int = rm.is_int
        lb_s = np.where(np.isfinite(rm.lb_sep), rm.lb_sep, 0.0)
        ub_s = np.where(
            np.isfinite(np.minimum(rm.ub, rm.ub_sep)), np.minimum(rm.ub, rm.ub_sep), 1e5
        )

        ca, cb, x_root, b_root = root_cut_loop(rm, is_int, lb_s, ub_s)
        print(f"  root bound excess {excess(b_root, opt):.2f}%  ({len(ca)} rows)", flush=True)
        if x_root is None:
            continue

        # top-K fractional integers at the root LP vertex
        fr = [
            (abs(x_root[j] - round(x_root[j])), j)
            for j in range(rm.n)
            if is_int[j] and 1e-6 < x_root[j] - np.floor(x_root[j]) < 1 - 1e-6
        ]
        fr.sort(reverse=True)
        branch_js = [j for _f, j in fr[:BRANCH_VARS]]
        print(f"  fractional integers: {len(fr)}; branching on {branch_js}", flush=True)

        child_recs = []
        for j in branch_js:
            for direction in ("down", "up"):
                lo = rm.lb.copy()
                hi = rm.ub.copy()
                if direction == "down":
                    hi[j] = np.floor(x_root[j])
                else:
                    lo[j] = np.ceil(x_root[j])
                lo, hi = child_fbbt(rm, lo, hi)
                if np.any(lo > hi + 1e-9):
                    continue  # infeasible child
                t0 = time.time()
                b_base = solve_over_box(rm, lo, hi, ca, cb, False, is_int, lb_s, ub_s)
                b_cut = solve_over_box(rm, lo, hi, ca, cb, True, is_int, lb_s, ub_s)
                if b_base is None or b_cut is None:
                    continue
                e_base = excess(b_base, opt)
                e_cut = excess(b_cut, opt)
                remaining = max(e_base, 1e-9)  # child's gap to opt (excess %)
                closed = (e_base - e_cut) / remaining * 100.0 if remaining > 1e-9 else 0.0
                sound = e_cut >= -1e-4  # bound must stay >= opt (maximize panel)
                child_recs.append(
                    dict(
                        var=int(j),
                        dir=direction,
                        base=e_base,
                        cut=e_cut,
                        closed_pct=closed,
                        sound=bool(sound),
                        secs=time.time() - t0,
                    )
                )
                print(
                    f"    x{j} {direction:4s}: base +{e_base:.2f}% -> cut +{e_cut:.2f}%  "
                    f"closes {closed:.1f}% of child gap  sound={sound}",
                    flush=True,
                )
        if child_recs:
            closes = sorted(r["closed_pct"] for r in child_recs)
            med = closes[len(closes) // 2]
            allsound = all(r["sound"] for r in child_recs)
            results[name] = dict(
                root_excess=excess(b_root, opt),
                n_children=len(child_recs),
                median_child_closed_pct=med,
                all_sound=allsound,
                children=child_recs,
            )
            print(
                f"  MEDIAN child in-tree tightening: {med:.1f}% of child gap "
                f"({len(child_recs)} children, sound={allsound})",
                flush=True,
            )

    print("\n===== VERDICT (#786 in-tree value) =====")
    survivor = False
    for name, r in results.items():
        med = r["median_child_closed_pct"]
        print(f"  {name}: median child tightening {med:.1f}% of child gap (kill bar {KILL_BAR}%)")
        if med >= KILL_BAR:
            survivor = True
    print(f"\n  in-tree cutting SURVIVOR (median >= {KILL_BAR}% on >=1 instance): {survivor}")
    _go = "GO: build in-solver in-tree separation"
    _frontier = "FRONTIER (#786 option b): in-tree adds little beyond root cuts + child FBBT"
    print(f"  -> {_go if survivor else _frontier}")
    results["_verdict"] = {"survivor": survivor, "kill_bar_pct": KILL_BAR}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.normpath(os.path.join(here, "..", "results", "issue786"))
    os.makedirs(outdir, exist_ok=True)
    outp = f"{outdir}/intree_value_probe_{stamp}.json"
    with open(outp, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\n  wrote {outp}")


if __name__ == "__main__":
    main()
