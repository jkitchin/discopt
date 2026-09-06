"""TIME-LIMIT ROBUSTNESS RERUN of gapsplit.py -- is the PRIMAL verdict an artifact?

gapsplit.py at TL=20s returned a median primal share of 91.1% over 15 unsolved
instances and therefore "PRIMAL is the lever". A short time limit can manufacture
that answer on its own: heuristics need wall time to fire, so a starved run can
look primal-bound when a longer one would not. Before a whole track is re-ordered
on that number it has to survive a bigger budget.

PRE-REGISTERED KILL CRITERION (written before this run, CLAUDE.md 4):
    Rerun ONLY the 15 instances unsolved at 20s, at TL=60s (3x the budget).
    median primal share < 0.60  -> the 20s verdict is a TIME-LIMIT ARTIFACT;
                                   Track B2 is NOT promoted on this evidence.
    median primal share >= 0.60 -> the verdict is robust to a 3x budget and the
                                   primal side is the real residual.
Instances that reach optimality at 60s leave the scored set, exactly as at 20s.
The correctness gate, the shares-sum-to-1 check and the executed-comparison count
are inherited unchanged from gapsplit.py.
"""
import math
import os, sys, time, json, statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import discopt
from discopt._rust import solve_milp_py
from loader import read_mps, to_engine

HERE = os.path.dirname(os.path.abspath(__file__))
TL = float(os.environ["GS_TL"])
BASE = dict(gap_tol=1e-4, max_nodes=10_000_000, gmi_cuts=True, root_cut_prune=True,
            time_limit_s=TL, root_cuts=500, cut_rounds=50, cut_select=True,
            root_cut_time_s=max(0.5, 0.5 * TL))
D_STATUSES = {"optimal", "feasible", "infeasible", "unbounded", "node_limit"}
HAS_INCUMBENT = {"optimal", "feasible", "node_limit"}

print("discopt:", discopt.__file__, flush=True)
print("load at start:", os.popen("uptime").read().strip(), flush=True)
PANEL = json.load(open(os.path.join(HERE, "panel.json")))
ONLY = set(open(os.path.join(HERE, "unsolved20.txt")).read().split(","))
PANEL = [p for p in PANEL if p["name"] in ONLY]
assert len(PANEL) == len(ONLY), f"panel/name mismatch: {len(PANEL)} vs {len(ONLY)}"
print(f"TL={TL}s over {len(PANEL)} instances unsolved at 20s", flush=True)

rows, cert = [], []
print(f"\n{'instance':<20}{'status':>10}{'incumbent':>14}{'bound':>14}{'optimum':>14}"
      f"{'primal%':>9}{'dual%':>8}", flush=True)
for p in PANEL:
    f = os.path.join(HERE, "mps", p["name"] + ".mps.gz")
    if not os.path.exists(f):
        continue
    d = read_mps(f)
    try:
        c, A, b, lo, up, ic, ns, off, _ = to_engine(d)
    finally:
        os.unlink(d["mps"])
    t0 = time.perf_counter()
    st, x, obj, bound, nodes, iters = solve_milp_py(c, A, b, lo, up, ic, ns, off, **BASE)
    wall = time.perf_counter() - t0
    if st not in D_STATUSES:
        sys.exit(f"UNKNOWN discopt status {st!r} on {p['name']}")
    ref = p.get("opt_min")
    tol = lambda v: 1e-6 * (1 + abs(v))
    if ref is not None:
        if bound is not None and bound > ref + tol(ref):
            cert.append((p["name"], "bound above optimum", bound, ref))
        if st in HAS_INCUMBENT and obj is not None and obj < ref - 1e-4 * (1 + abs(ref)):
            cert.append((p["name"], "incumbent below optimum", obj, ref))

    solved = (st == "optimal")
    # `obj is not None` is NOT enough: a `node_limit` run with no feasible point
    # reports obj=inf, which passes a None-check and then poisons the whole
    # scoring path with nan (see the nan-guard note below).
    has_inc = (st in HAS_INCUMBENT and obj is not None
               and math.isfinite(obj) and abs(obj) < 1e19)
    pshare = dshare = None
    if not solved and ref is not None:
        if not has_inc:
            pshare, dshare = 1.0, 0.0          # no incumbent: the primal side is the whole failure
        elif bound is not None:
            span = obj - bound
            if span > tol(ref):
                pshare = (obj - ref) / span
                dshare = (ref - bound) / span
    rows.append(dict(name=p["name"], status=st, solved=solved, has_inc=has_inc,
                     obj=obj, bound=bound, ref=ref, nodes=nodes, wall=wall,
                     pshare=pshare, dshare=dshare))
    fmt = lambda v: f"{v:14.6g}" if v is not None else f"{'-':>14}"
    print(f"{p['name']:<20}{st:>10}{fmt(obj if has_inc else None)}{fmt(bound)}{fmt(ref)}"
          f"{(f'{100*pshare:8.1f}' if pshare is not None else '       -')}"
          f"{(f'{100*dshare:7.1f}' if dshare is not None else '      -')}", flush=True)

print(f"\nINSTANCES RUN: {len(rows)}")
if not rows:
    sys.exit("VACUOUS: no instance ran")
if cert:
    for nm, what, got, ref in cert:
        print(f"  CERT VIOLATION {nm}: {what} ({got} vs optimum {ref})")
    sys.exit("CERTIFICATE VIOLATION -- no performance reading from this run is valid")

unsolved = [r for r in rows if not r["solved"]]
scored = [r for r in unsolved if r["pshare"] is not None]
noinc = [r for r in unsolved if not r["has_inc"]]
print(f"UNSOLVED: {len(unsolved)}   SCORED (the comparison count): {len(scored)}"
      f"   of which NO INCUMBENT AT ALL: {len(noinc)}")
if noinc:
    print("  no incumbent at the time limit: " + ", ".join(r["name"] for r in noinc))
if not scored:
    sys.exit("VACUOUS: no unsolved instance could be scored; nothing was compared")

# The two shares partition the open gap by construction; check it rather than trust it.
# Written as `not (… <= tol)` and NOT as `… > tol` on purpose. Every comparison
# against nan is False, so the `>` form SILENTLY PASSES on exactly the input this
# guard exists to catch -- which is what happened on the 2026-09-05 run:
# enlight_hard and neos-2624317-amur returned obj=inf, has_inc was wrongly True,
# pshare came out nan, `abs(nan) > 1e-6` was False, the guard let it through, and
# statistics.median then sorted a list containing nan (undefined order). The
# median was unchanged by luck; the guard was defeated regardless. (CLAUDE.md 6.)
for r in scored:
    s = r["pshare"] + r["dshare"]
    if not (abs(s - 1.0) <= 1e-6):
        sys.exit(f"WIRING BUG: shares on {r['name']} sum to {s}, not 1")
    if not (math.isfinite(r["pshare"]) and math.isfinite(r["dshare"])):
        sys.exit(f"WIRING BUG: non-finite share on {r['name']}: "
                 f"pshare={r['pshare']} dshare={r['dshare']}")

med = statistics.median(r["pshare"] for r in scored)
print(f"\nMEDIAN PRIMAL SHARE over the {len(scored)} unsolved instances: {100*med:.1f}%")
print(f"MEDIAN DUAL   SHARE: {100*(1-med):.1f}%")
if med >= 0.60:
    v = "PRIMAL is the lever -- build heuristics (Track B2) before branching work"
elif med <= 0.20:
    v = "DUAL is the lever -- build branching/bounding before heuristics"
else:
    v = "BOTH are live -- report the split and rank by cost; do not guess a winner"
print(f"\nVERDICT: {v}")
print("load at end:", os.popen("uptime").read().strip())
json.dump(rows, open(os.path.join(HERE, "gapsplit_tl.json"), "w"), indent=1)
