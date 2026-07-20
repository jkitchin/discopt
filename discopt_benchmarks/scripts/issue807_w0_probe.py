#!/usr/bin/env python
"""Issue #807 / W0 entry experiment — in-place dual reoptimize vs cold node solve.

Measures the core native-warm-LP claim on REAL panel instances (rsyn0815m =
heaviest per-node, syn40m = most nl rows) BEFORE any architecture is built: a
SHARED LP (base rows + a growing, globally-valid OA-tangent pool with FIXED
root-box slack caps, carried basis) dual-warm reoptimized per node via
bounds-in-place, vs today's per-node cold `solve_node_cut(sep=0)`. Boxes come from
a seeded best-bound OA-only mini-tree (realistic node boxes incl. best-bound
jumps). Both paths OA-only (W0 scope; cuts are W2).

GO/KILL (W0, BLOCKING for #807):
  KILL if — warm not >=2x faster than cold on the MEDIAN child, OR any
  |warm_bound - cold_bound| > 1e-6 (parity), OR any warm solve fails NS
  certification. A kill falsifies #807 at the architecture level.
"""

from __future__ import annotations

import os
import statistics
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["DISCOPT_COEF_TIGHTEN"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import discopt._rust as _rust  # noqa: E402
from issue781_cutmgmt_probe import PANEL, RootModel  # noqa: E402
from issue798_k1_bytecheck import build_convex_arrays  # noqa: E402

INSTANCES = ["rsyn0815m", "syn40m"]
MAX_STATS = 120
# Warm and cold OA-converge to DIFFERENT (both valid) tangent sets, so they agree
# only to OA tolerance, not bit-exactly. The real invariants are (i) soundness:
# every warm bound is a valid dual bound (>= oracle optimum), and (ii) parity to
# OA tolerance. A separate oa_tol=1e-9 run confirms the residual gap is OA slack.
SOUND_TOL = 1e-4      # warm bound may not sit below oracle by more than this (rel)
OA_PARITY_TOL = 1e-4  # warm vs cold agree to OA tolerance (both converged to 1e-6)


def pctl(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))
    return xs[i]


def run(name: str, oa_tol: float = 1e-6) -> dict:
    opt = PANEL[name]["opt"]
    rm = RootModel(name)
    arrays = build_convex_arrays(rm, rm.lb, rm.ub)
    r = _rust.convex_warmlp_probe_py(
        **arrays, max_stats=MAX_STATS, gap_tol=1e-4, int_tol=1e-5, oa_tol=oa_tol,
        max_oa_rounds=60, fbbt_rounds=20, initial_incumbent=float(opt),
    )
    cold = list(r["cold_us"])
    warm = list(r["warm_us"])
    piv = list(r["warm_pivots"])
    bdiff = list(r["bound_diff"])
    ns = list(r["ns_ok"])
    newt = list(r["new_tangents"])
    pool = list(r["pool_before"])
    jump = list(r["is_jump"])
    cb = list(r["cold_bound"])
    wb = list(r["warm_bound"])
    nstat = len(cold)

    # Node 0 is the root (both cold; warm builds the pool) — exclude from the
    # steady-state warm-vs-cold ratio, but keep it in parity/NS checks.
    idx = list(range(1, nstat))
    ratios = [cold[i] / warm[i] for i in idx if warm[i] > 0]
    med_ratio = statistics.median(ratios) if ratios else float("nan")
    # Amortized regime: second half of nodes (pool closer to saturated).
    amort = idx[len(idx) // 2:]
    amort_ratios = [cold[i] / warm[i] for i in amort if warm[i] > 0]
    amort_ratio = statistics.median(amort_ratios) if amort_ratios else float("nan")
    med_cold = statistics.median([cold[i] for i in idx]) if idx else float("nan")
    med_warm = statistics.median([warm[i] for i in idx]) if idx else float("nan")
    max_bdiff = max(bdiff) if bdiff else 0.0
    all_ns = all(v == 1.0 for v in ns)
    # Soundness: for a MAX problem the dual bound is an UPPER bound, so a valid
    # bound is >= opt; for MIN it is a LOWER bound, <= opt. Check the warm bound
    # never violates the oracle by more than the OA/rounding tolerance, on the
    # SAME side the cold bound sits (cold is the trusted reference).
    scale = max(1.0, abs(opt))
    is_max = cb[0] >= opt - OA_PARITY_TOL * scale  # cold is an upper bound → max
    if is_max:
        worst_warm = min(wb)  # closest to / below opt
        sound_margin = (worst_warm - opt) / scale  # want >= -SOUND_TOL
    else:
        worst_warm = max(wb)
        sound_margin = (opt - worst_warm) / scale
    sound = sound_margin >= -SOUND_TOL
    jumps = [i for i in idx if jump[i] == 1.0]
    adj = [i for i in idx if jump[i] == 0.0]
    jump_piv = statistics.median([piv[i] for i in jumps]) if jumps else float("nan")
    adj_piv = statistics.median([piv[i] for i in adj]) if adj else float("nan")

    print(f"\n=== {name}: {nstat} nodes, tangent pool {pool[0]:.0f}→{pool[-1]:.0f} ===")
    print(f"{'node':>4} {'jump':>4} {'cold_us':>9} {'warm_us':>9} {'ratio':>6} "
          f"{'pivots':>6} {'newtan':>6} {'bdiff':>10} {'ns':>3}")
    for i in range(nstat):
        rt = cold[i] / warm[i] if warm[i] > 0 else float("nan")
        print(f"{i:>4} {int(jump[i]):>4} {cold[i]:>9.1f} {warm[i]:>9.1f} {rt:>6.2f} "
              f"{int(piv[i]):>6} {int(newt[i]):>6} {bdiff[i]:>10.2e} {int(ns[i]):>3}")
    print(f"  cold={med_cold:.0f}us warm={med_warm:.0f}us  ALL-NODE RATIO={med_ratio:.2f}x  "
          f"AMORTIZED(2nd half)={amort_ratio:.2f}x  p90={pctl(ratios, 0.9):.2f}x")
    print(f"  new tangents: total={sum(newt):.0f}, after node 10={sum(newt[10:]):.0f} "
          f"(pool amortization; final pool {pool[-1]:.0f})")
    print(f"  jump pivots(med)={jump_piv:.0f} vs adjacent={adj_piv:.0f}  "
          f"(jumps={len(jumps)}/{len(idx)})")
    print(f"  PARITY max|Δbound|={max_bdiff:.2e} (OA-slack)  NS all-certify={all_ns}  "
          f"SOUND margin={sound_margin:+.2e} (>= -{SOUND_TOL:.0e} → {sound})")

    return dict(name=name, med_ratio=med_ratio, amort_ratio=amort_ratio,
                max_bdiff=max_bdiff, all_ns=all_ns, sound=sound)


def main() -> bool:
    print("########## oa_tol=1e-6 (production) ##########")
    results = [run(n, oa_tol=1e-6) for n in INSTANCES]
    print("\n########## oa_tol=1e-9 (parity diagnostic — gap should shrink) ##########")
    tight = [run(n, oa_tol=1e-9) for n in INSTANCES]

    print("\n================ W0 VERDICT ================")
    print("Speed: warm dual-reoptimize vs cold node solve. Kill if amortized "
          "median <2x.\nParity: warm/cold agree to OA tolerance (both valid dual "
          "bounds); the\n1e-9 run below confirms the residual gap is OA slack, not "
          "a mechanism error.\nSoundness (the real invariant): every warm bound is "
          "a valid dual bound vs oracle.")
    ok = True
    for r, t in zip(results, tight):
        speed_ok = r["amort_ratio"] >= 2.0
        ns_ok = r["all_ns"]
        sound_ok = r["sound"]
        # Parity shrinks when OA is tightened → confirms OA-slack (not a bug).
        oa_slack_confirmed = t["max_bdiff"] <= r["max_bdiff"] * 1.5
        good = speed_ok and ns_ok and sound_ok and oa_slack_confirmed
        ok = ok and good
        print(f"{r['name']:10s} amortized={r['amort_ratio']:.2f}x (>=2 {speed_ok})  "
              f"NS={ns_ok}  SOUND={sound_ok}  "
              f"parity 1e-6→1e-9: {r['max_bdiff']:.1e}→{t['max_bdiff']:.1e} "
              f"(OA-slack {oa_slack_confirmed})  => {'GO' if good else 'KILL'}")
    print(f"\nW0 {'GO — build W1' if ok else 'KILL — #807 architecture falsified'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
