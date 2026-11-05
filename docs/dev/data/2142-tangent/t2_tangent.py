"""#2142 item 3, tangent entry experiment, step 2 (fixed boxes, no tree).

For each captured cstr big-M node (t1_capture_lp.py) with a finite node bound, on the
node's post-propagation box:

  ours0   the kernel's own node LP, solved by the kernel (`_rust.solve_spatial_node_py`
          = `solve_spatial_node_with_cuts`, no cuts) -- must reproduce the tree's node
          bound (reconstruction check);
  oursT   ours0 + LP-point tangent rounds: for every lifted convex univariate term
          (here: the `Monomial p=2` squares, the spec's only convex univariate kind)
          whose aux sits below f(x*) at the LP point, add the global tangent at x*
          as a cut and re-solve, until no violation or MAX_ROUNDS;
  scip    SCIP's root dual bound on the same box (LP + constraint enforcement;
          separators, presolve, OBBT, redcost, heuristics, symmetry off) -- the
          p5_relax.py arm. Extra SCIP params on the command line (attribution arms).

All bounds are the kernel's Neumaier-Shcherbina safe bounds. Gap closure over the
boxes where SCIP is tighter by > 1e-4: sum(oursT - ours0) / sum(scip - ours0).
Prints comparison counts (CLAUDE.md §6); nothing is caught (§7).

  DISCOPT_TREE=<wt> PYTHONPATH=<wt>/python PROPDIR=<2142-propagation dir> \\
      python -u t2_tangent.py LP.npz [--no-tangents] [scip-param=value ...]
"""
import math, os, sys
import numpy as np

import discopt._rust as R

assert R.__file__.startswith(os.environ["DISCOPT_TREE"]), R.__file__
assert hasattr(R, "solve_spatial_node_py"), "marker absent: build without solve_spatial_node_py"
sys.path.insert(0, os.environ["PROPDIR"])
import scip_prop  # noqa: E402

MAX_ROUNDS = 20
VIOL_TOL = 1e-7
args = sys.argv[2:]
NO_TAN = "--no-tangents" in args
args = [a for a in args if a != "--no-tangents"]
d = dict(np.load(sys.argv[1]))
extra = dict(kv.split("=", 1) for kv in args)
TAG = ",".join(args) or "default"
SPEC = [k for k in d if not k.startswith("rec_")]
kw = {k: (int(d[k]) if d[k].ndim == 0 else d[k]) for k in SPEC}
params = {"lp/solvefreq": 1, "numerics/boundstreps": 1e-9}
for k, v in extra.items():
    params[k] = (v == "True") if v in ("True", "False") else (int(v) if v.lstrip("-").isdigit() else float(v))
sp = scip_prop.ScipProp(kw, params)
n_cols = int(d["n_cols"])
terms = [(int(d["term_i"][k]), int(d["term_out"][k]), int(d["term_p"][k]))
         for k in range(len(d["term_kind"])) if int(d["term_kind"][k]) == 1]
assert terms and all(p == 2 for _, _, p in terms), "only squares handled (the cstr spec)"
assert set(int(k) for k in d["term_kind"]) == {1}, "spec has non-monomial terms: extend the separator"


def rec(k, b):
    p = d["rec_%s_ptr" % k]
    return d["rec_%s_cat" % k][p[b]:p[b + 1]]


n_retry = [0]


def node(lo, hi, cuts):
    ptr = np.cumsum([0] + [len(c[0]) for c in cuts]).astype(np.int64)
    cols = np.array([j for c in cuts for j in c[0]], dtype=np.int64)
    co = np.array([v for c in cuts for v in c[1]], dtype=float)
    rhs = np.array([c[2] for c in cuts], dtype=float)
    r = R.solve_spatial_node_py(**kw, lo=lo, hi=hi, cut_ptr=ptr, cut_cols=cols, cut_coeffs=co, cut_rhs=rhs)
    if not (r["status"] == "infeasible" or (r["status"] == "optimal" and math.isfinite(r["bound"]))):
        # the cold primal leaves an uncertified/undecided LP here; the tree's nodes are
        # warm-started, so retry the same LP with the dual simplex from the slack basis
        n_retry[0] += 1
        r = R.solve_spatial_node_py(**kw, lo=lo, hi=hi, cut_ptr=ptr, cut_cols=cols, cut_coeffs=co,
                                    cut_rhs=rhs, cold_dual_start=True)
    return r


rows = []
n_cmp = n_recon = n_recon_ok = 0
n_viol_checks = 0
n_undecided = [0]
n_retry = [0]
for b in range(len(d["rec_id"])):
    if not (math.isfinite(d["rec_bound"][b]) and d["rec_status"][b] == "optimal"):
        continue
    lo, hi = rec("lo", b), rec("hi", b)
    r0 = node(lo, hi, [])
    n_recon += 1
    ok = r0["status"] == "optimal" and abs(r0["obj"] - d["rec_obj"][b]) <= 1e-6 * max(1.0, abs(r0["obj"]))
    n_recon_ok += ok
    cuts, r, rounds, best = [], r0, 0, r0["bound"]
    while not NO_TAN and rounds < MAX_ROUNDS and r["status"] == "optimal":
        x = np.asarray(r["x"])[:n_cols]
        new = []
        for i, s, _ in terms:
            x0 = float(x[i])
            n_viol_checks += 1
            if x[s] < x0 * x0 - VIOL_TOL * max(1.0, x0 * x0):
                new.append(([i, s], [2.0 * x0, -1.0], x0 * x0))  # s >= 2 x0 x - x0^2
        if not new:
            break
        r = node(lo, hi, cuts + new)
        if r["status"] not in ("optimal", "infeasible"):
            # the tree's separation_loop drops an undecided round's cuts and stops
            n_undecided[0] += 1
            break
        cuts += new
        rounds += 1
        best = math.inf if r["status"] == "infeasible" else max(best, r["bound"])
    sp.m.freeTransform()
    sst, _, _ = sp.propagate(lo, hi, None)
    sc = sp.m.getDualbound() if sst != "infeasible" else math.inf
    n_cmp += 1
    rows.append((int(d["rec_id"][b]), float(d["rec_bound"][b]), r0["bound"], best, sc, len(cuts), rounds))

ids, kb, o0, oT, sc, nc, nr = (np.array(c, dtype=float) for c in zip(*rows))
fin = np.isfinite(sc)
print(f"[{TAG}] boxes {len(rows)}; kernel-LP re-solve reproduces the tree's node obj (1e-6 rel): {n_recon_ok}/{n_recon}")
print(f"[{TAG}] max |re-solved safe bound - tree node bound|: {np.max(np.abs(kb - o0)):.2e}")
print(f"[{TAG}] tangent violation checks executed: {n_viol_checks}; undecided (dropped) rounds: {n_undecided[0]}; dual-simplex retries: {n_retry[0]}")
gap0 = sc - o0
sel = fin & (gap0 > 1e-4)
print(f"[{TAG}] SCIP infeasible: {int((~fin).sum())}; SCIP tighter than ours by >1e-4: {int(sel.sum())}; "
      f"ours tighter by >1e-4: {int((fin & (gap0 < -1e-4)).sum())}")
print(f"[{TAG}] boxes with >=1 violated tangent: {int((nc > 0).sum())}; cuts total {int(nc.sum())}; max rounds {int(nr.max())}")
fT = np.isfinite(oT)
gain = np.where(fT, oT - o0, np.nan)
print(f"[{TAG}] tangent gain (finite): mean {np.nanmean(gain):.5f} max {np.nanmax(gain):.5f}; >1e-4 on {int((gain > 1e-4).sum())}; "
      f"tangents prove infeasible: {int((~fT).sum())}")
if sel.any():
    capped = np.minimum(np.where(fT, oT, sc), sc)  # an infeasible proof closes the whole box gap
    num = float(np.sum(capped[sel] - o0[sel]))
    den = float(np.sum(sc[sel] - o0[sel]))
    print(f"[{TAG}] GAP CLOSURE on the {int(sel.sum())} SCIP-tighter boxes: {num:.5f} / {den:.5f} = {num / den:.1%}")
    print(f"[{TAG}]   after tangents SCIP still tighter by >1e-4 on {int(((sc - np.where(fT, oT, np.inf))[sel] > 1e-4).sum())} of them")
over = fin & fT & (oT > sc + 1e-6)
print(f"[{TAG}] boxes where the tangent bound exceeds SCIP's by >1e-6: {int(over.sum())}")
for q in sorted(range(len(rows)), key=lambda q: -(sc[q] - o0[q]) if fin[q] else 0)[:10]:
    print(f"[{TAG}]   node {int(ids[q]):6d} ours0 {o0[q]:.4f} tangents {oT[q]:.4f} scip {sc[q]:.4f} cuts {int(nc[q])} rounds {int(nr[q])}")
np.savez(os.path.splitext(sys.argv[1])[0] + f".t2.{TAG.replace('/', '_')}{'.notan' if NO_TAN else ''}.npz",
         ids=ids, kb=kb, o0=o0, oT=oT, sc=sc, nc=nc)
print(f"[{TAG}] executed comparisons: {n_cmp}")
sys.exit(0 if n_cmp and n_recon_ok else 1)
