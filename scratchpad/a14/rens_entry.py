"""ENTRY EXPERIMENT for A14 (RENS), run BEFORE any implementation (CLAUDE.md 4).

WHAT THE MEASUREMENT SAYS SO FAR
gapsplit at 20s: median primal share 91.1% over 15 unsolved instances; the 60s
rerun put it at 91.8%, so the verdict is not a time-limit artifact. Sharper: of
the 14 instances that HAD an incumbent at 20s, *not one* improved it at 60s,
across 2-5x more nodes, while the dual bound improved on 10/15.

WHY: off the root discopt's only primal heuristic is plain nearest-rounding
(`try_rounding_csc`, no re-solve). The one heuristic that repairs continuous
variables (`try_dive_repair`) is root-only by default (DIVE_STRIDE_DEFAULT = 0)
and is additionally hard-gated `!has_incumbent`. So once ANY incumbent exists
nothing can improve it but a node LP landing integral by luck.

WHY RENS AND NOT MORE RINS: A12 built RINS and it measured neutral-or-harmful,
and this data says why. RINS fixes the integer variables on which the incumbent
and the LP relaxation AGREE -- it is an *improvement* heuristic anchored to the
incumbent it starts from. Here the incumbents are 20-140% off the optimum
(mik ~30%, beavma ~55%, neos17 ~140%), so that anchor is the problem, not the
starting point. RENS (Berthold) ignores the incumbent entirely: it fixes every
integer variable already integral in the LP RELAXATION and restricts each
fractional one to {floor, ceil}, then solves the sub-MIP. It needs no incumbent
at all -- which also covers neos-2624317-amur, which has none.

THE TEST (no Rust required; this can kill A14 before a line is written):
per instance, solve the pure LP relaxation, build the RENS box from it, and run
discopt on that box for the same 20s. Compare against the incumbent the full 20s
solve reached (gapsplit.json).

PRE-REGISTERED KILL CRITERION (written before the run):
    HIT RATE  : RENS strictly improves the 20s incumbent on >= 5 of 15 (~1/3,
                the rate Berthold reports). <= 3 of 15 -> A14/RENS IS DEAD and
                is not built.
    MAGNITUDE : over the improved instances, median reduction of the primal gap
                (obj - optimum) relative to (incumbent20 - optimum) >= 25%.
    BOTH must hold to justify implementing RENS. 4 of 15 is the inconclusive
    band: report it and rank against alternatives, do not build on it.

CORRECTNESS GATE: the RENS box is a RESTRICTION of the feasible set, so its
objective can never beat the reference optimum. Any objective below it is a bug
in the box construction (or in the solver), and no reading from this run stands.
This is the one gate that is not allowed to be soft.
"""
import math, os, sys, time, json, statistics
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import discopt
from discopt._rust import solve_milp_py
from loader import read_mps, to_engine

HERE = os.path.dirname(os.path.abspath(__file__))
TL = 20.0
BASE = dict(gap_tol=1e-4, max_nodes=10_000_000, gmi_cuts=True, root_cut_prune=True,
            time_limit_s=TL, root_cuts=500, cut_rounds=50, cut_select=True,
            root_cut_time_s=max(0.5, 0.5 * TL))
INTTOL = 1e-6

print("discopt:", discopt.__file__, flush=True)
print("load at start:", os.popen("uptime").read().strip(), flush=True)
prev = {r["name"]: r for r in json.load(open(os.path.join(HERE, "gapsplit.json")))}
names = open(os.path.join(HERE, "unsolved20.txt")).read().split(",")
PANEL = {p["name"]: p for p in json.load(open(os.path.join(HERE, "panel.json")))}

rows, cert = [], []
print(f"\n{'instance':<20}{'fix%':>7}{'free':>7}{'RENS st':>10}{'RENS obj':>13}"
      f"{'inc@20':>13}{'optimum':>13}{'better?':>9}{'gapcut%':>9}", flush=True)
for nm in names:
    f = os.path.join(HERE, "mps", nm + ".mps.gz")
    d = read_mps(f)
    try:
        c, A, b, lo, up, ic, ns, off, _ = to_engine(d)
    finally:
        os.unlink(d["mps"])

    # 1. LP relaxation: same engine, no integer columns.
    empty = np.zeros(0, dtype=np.int64)
    st0, x0, o0, _, _, _ = solve_milp_py(c, A, b, lo, up, empty, ns, off, **BASE)
    if st0 != "optimal" or x0 is None:
        print(f"{nm:<20}{'LP relaxation did not solve: ' + str(st0):>60}", flush=True)
        rows.append(dict(name=nm, skipped=f"lp_{st0}"))
        continue
    x0 = np.asarray(x0, float)

    # 2. RENS box: fix the already-integral integer columns, restrict the rest
    #    to {floor, ceil}. Continuous columns are untouched.
    l2, u2 = lo.copy(), up.copy()
    nfix = 0
    for j in ic:
        v = x0[j]
        r = round(v)
        if abs(v - r) <= INTTOL:
            l2[j] = u2[j] = r
            nfix += 1
        else:
            l2[j] = max(lo[j], math.floor(v))
            u2[j] = min(up[j], math.ceil(v))
    nint = len(ic)
    free = nint - nfix

    # 3. Solve the sub-MIP on that box.
    t0 = time.perf_counter()
    st, x, obj, bnd, nodes, _ = solve_milp_py(c, A, b, l2, u2, ic, ns, off, **BASE)
    wall = time.perf_counter() - t0

    ref = PANEL[nm].get("opt_min")
    inc20 = prev[nm]["obj"] if prev[nm]["has_inc"] else None
    has = st in {"optimal", "feasible", "node_limit"} and obj is not None \
        and math.isfinite(obj) and abs(obj) < 1e19
    # CORRECTNESS: a restriction can never beat the true optimum.
    if has and ref is not None and obj < ref - 1e-4 * (1 + abs(ref)):
        cert.append((nm, obj, ref))
    better = has and inc20 is not None and obj < inc20 - 1e-9 * (1 + abs(inc20))
    if has and inc20 is None:
        better = True                      # found a point where there was none
    gapcut = None
    if has and inc20 is not None and ref is not None:
        base = inc20 - ref
        if base > 1e-9 * (1 + abs(ref)):
            gapcut = (base - (obj - ref)) / base
    rows.append(dict(name=nm, fixfrac=(nfix / nint if nint else None), free=free,
                     st=st, obj=(obj if has else None), inc20=inc20, ref=ref,
                     better=bool(better), gapcut=gapcut, nodes=nodes, wall=wall))
    g = lambda v, w=13: f"{v:{w}.6g}" if v is not None else f"{'-':>{w}}"
    print(f"{nm:<20}{100*nfix/nint if nint else 0:7.1f}{free:7d}{st:>10}"
          f"{g(obj if has else None)}{g(inc20)}{g(ref)}"
          f"{('YES' if better else 'no'):>9}"
          f"{(f'{100*gapcut:9.1f}' if gapcut is not None else '        -')}", flush=True)

print(f"\nINSTANCES RUN: {len(rows)}")
scored = [r for r in rows if not r.get("skipped")]
print(f"SCORED (the comparison count): {len(scored)}")
if not scored:
    sys.exit("VACUOUS: nothing was compared")
if cert:
    for nm, got, ref in cert:
        print(f"  CERT VIOLATION {nm}: RENS objective {got} beats optimum {ref}")
    sys.exit("CERTIFICATE VIOLATION -- the RENS box is not a restriction; no reading stands")

hits = [r for r in scored if r["better"]]
print(f"\nHIT RATE: RENS improved the 20s incumbent on {len(hits)}/{len(scored)}")
cuts = [r["gapcut"] for r in hits if r["gapcut"] is not None]
medcut = statistics.median(cuts) if cuts else None
print(f"MEDIAN PRIMAL-GAP REDUCTION over improved instances: "
      f"{(f'{100*medcut:.1f}%' if medcut is not None else 'n/a')} (n={len(cuts)})")
ff = [r["fixfrac"] for r in scored if r["fixfrac"] is not None]
print(f"MEDIAN LP-INTEGRAL FIXING RATE: {100*statistics.median(ff):.1f}%")

if len(hits) <= 3:
    v = "DEAD -- RENS does not improve the incumbent broadly; A14/RENS is NOT built"
elif len(hits) >= 5 and medcut is not None and medcut >= 0.25:
    v = "SURVIVES -- both pre-registered bars met; implement A14/RENS"
else:
    v = ("INCONCLUSIVE -- in the 4/15 band or magnitude short; rank against "
         "alternatives, do not build on this alone")
print(f"\nVERDICT: {v}")
print("load at end:", os.popen("uptime").read().strip())
json.dump(rows, open(os.path.join(HERE, "rens_entry.json"), "w"), indent=1)
