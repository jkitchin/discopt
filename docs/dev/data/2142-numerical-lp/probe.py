"""#2142 numerical-verdict probe: one deterministic GDPlib cstr solve with the kernel's
node_observer, reconstructing which nodes hold the global bound and re-solving every node LP
that ended undecided (numerical / iter_limit / unbounded) or uncertified (optimal, safe bound
-inf) with HiGHS and SoPlex.

  DISCOPT_PROFILE=1 PYTHONHASHSEED=0 PYTHONPATH=/home/user/discopt/python \
      python -u probe.py ARM MAX_NODES

ARM as in ../2142-exact-hull/arm.py (homog | bigm | ...). Asserts the loaded extension is this
tree's build (CLAUDE.md §8), lets every exception propagate (§7) and prints an
executed-assertion count (§6)."""
import collections
import os
import sys
import time

import numpy as np

import discopt
import discopt._rust as R

assert discopt.__file__.startswith("/home/user/discopt/python"), discopt.__file__
assert b"DISCOPT_INT_ROUND_EXACT" not in open(R.__file__, "rb").read(), "stale _rust (§114 flag)"
n_assert = 2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "2142-exact-hull"))
import exact_hull_proto as P  # noqa: E402
from discopt.pyomo.gdp import convert_pyomo_gdp  # noqa: E402
import gdplib.cstr as cstr  # noqa: E402

arm = sys.argv[1]
max_nodes = int(sys.argv[2])
method = {"hull": "hull", "bigm": "big-m"}.get(arm, "hull")
if arm in ("ratio", "homog", "lifted", "both"):
    P.install(arm)
os.environ.pop("DISCOPT_HULL_EXACT_PERSPECTIVE", None)

UNDECIDED = ("numerical", "iter_limit", "unbounded")
calls, recs, dumps, fidelity = [], {}, {}, set()
_o = R.solve_spatial_tree_py


CK = ("LpVerdictNumerical", "LpVerdictOptimal", "LpVerdictInfeasible", "LpDenseRetries",
      "LpInfeasUncertified", "DualProofGlobalLps")
last_ctr = {}


def obs(rec):
    nid = int(rec["id"])
    own = float(rec["bound"])
    st = str(rec["status"])
    # Profile-counter deltas since the previous observed node: this node's LP attempts plus
    # the previous infeasible node's dual-proof global LP (which runs after its observation).
    now = dict(R.profile_counters_py())
    d = {k: int(now.get(k, 0)) - int(last_ctr.get(k, 0)) for k in CK}
    last_ctr.update(now)
    recs[nid] = dict(parent=rec["parent"], status=st, own=own, obj=rec["obj"],
                     n_pivots=int(rec["n_pivots"]), lp_s=float(rec["lp_s"]), d=d)
    uncert = st == "optimal" and not np.isfinite(own)
    # The first few decided nodes are dumped too: HiGHS must reproduce their objective, which
    # proves the LP export is faithful before any failed LP is judged by it (§6).
    if st in UNDECIDED or uncert or (st == "optimal" and len(fidelity) < 8 and fidelity.add(nid) is None):
        dumps[nid] = {k: np.array(rec[k]) for k in
                      ("col_ptr", "row_idx", "vals", "b", "l", "u", "c", "lo", "hi")}
        dumps[nid]["m"], dumps[nid]["n"] = int(rec["m"]), int(rec["n"])


def _w(**k):
    if int(k.get("max_nodes", 0)) > 1 and "node_observer" not in k:
        k["node_observer"] = obs
    t = time.perf_counter()
    r = _o(**k)
    calls.append(dict(max_nodes=k.get("max_nodes"), status=r["status"], bound=r["bound"],
                      incumbent=r["incumbent"], nodes=r["node_count"],
                      undecided=r.get("n_undecided"), uncertified=r.get("n_uncertified"),
                      prop_retries=r.get("n_prop_retries"),
                      prop_retry_decided=r.get("n_prop_retry_decided"),
                      n_lp_solves=r.get("n_lp_solves"), n_cuts_added=r.get("n_cuts_added"),
                      n_cols=int(k["n_cols"]), wall=round(time.perf_counter() - t, 1)))
    return r


R.solve_spatial_tree_py = _w
m = convert_pyomo_gdp(cstr.build_model()).model
t0 = time.perf_counter()
res = m.solve(gdp_method=method, time_limit=7200.0, deterministic=True, max_nodes=max_nodes)
wall = time.perf_counter() - t0
print(f"ARM={arm} METHOD={method} MAX_NODES={max_nodes} status={res.status} obj={res.objective} "
      f"bound={res.bound!r} nodes={res.node_count} wall={wall:.1f}s", flush=True)
for c in calls:
    print("  kernel:", c, flush=True)
main = [c for c in calls if c["max_nodes"] > 1]
assert len(main) == 1, main; n_assert += 1
main = main[0]
ctr = dict(R.profile_counters_py())
keys = ("LpVerdictOptimal", "LpVerdictInfeasible", "LpVerdictNumerical", "LpVerdictIterLimit",
        "LpInfeasUncertified", "LpDenseRetries", "LpDenseRetryRescues", "WarmVerdictNumerical",
        "FarkasRejectMargin", "DualProofGlobalLps", "DualProofLearned")
print("counters:", {k: ctr.get(k) for k in keys})

# ---- node-level accounting ------------------------------------------------------------
st = collections.Counter(r["status"] for r in recs.values())
print("node status histogram:", dict(st), "| nodes with an LP:", len(recs),
      "| propagation-fathomed (no LP):", main["nodes"] - len(recs))
n_und = sum(st[s] for s in UNDECIDED)
n_unc = sum(1 for r in recs.values() if r["status"] == "optimal" and not np.isfinite(r["own"]))
print(f"undecided nodes: {n_und} (kernel n_undecided={main['undecided']}); "
      f"uncertified optimal nodes: {n_unc} (kernel n_uncertified={main['uncertified']})")
assert n_und == main["undecided"], (n_und, main["undecided"]); n_assert += 1
assert n_unc == main["uncertified"], (n_unc, main["uncertified"]); n_assert += 1
assert len(dumps) == n_und + n_unc + len(fidelity), (len(dumps), n_und, n_unc, len(fidelity))
n_assert += 1

# Effective bound = max(own finite bound, parent's effective bound), as the kernel does.
# Nodes popped and fathomed by propagation never reach the observer; their children do not
# exist, so they cannot hold the frontier. Node ids are pop order, parents precede children.
eff, depth, kids = {}, {}, collections.defaultdict(int)
for nid in sorted(recs):
    r = recs[nid]
    p = r["parent"]
    pb = eff.get(p, -np.inf) if p is not None else -np.inf
    own = r["own"] if r["status"] == "optimal" else -np.inf
    eff[nid] = max(own, pb)
    depth[nid] = depth[p] + 1 if p is not None else 0
    if p is not None:
        kids[p] += 1
# Open frontier: every branched node pushed two children; a node with < 2 OBSERVED children
# either has an open child carrying eff[node] or had a child popped and fathomed by
# propagation / dual proofs before any LP (those never reach the observer). The kernel pops
# best-bound first, so every open node has pb >= the reported bound; a candidate with
# eff < bound therefore accounts for (2 - kids) propagation-fathomed children -- the §6
# self-check is that this deficit fits the kernel's count, and that no candidate sits below
# the bound without that explanation.
gb, tol = main["bound"], 1e-9 * max(1.0, abs(main["bound"]))
cands = [nid for nid, r in recs.items() if r["status"] != "infeasible" and kids[nid] < 2]
assert cands, "no frontier candidates"; n_assert += 1
n_pf = main["nodes"] - len(recs)
deficit = sum(2 - kids[n] for n in cands if eff[n] < gb - tol)
print(f"candidates {len(cands)}, min eff {min(eff[n] for n in cands)!r}, kernel bound {gb!r}; "
      f"unseen children below the bound {deficit} <= propagation-fathomed {n_pf}")
assert deficit <= n_pf, (deficit, n_pf); n_assert += 1
assert min(eff[n] for n in cands) <= gb + tol, "no candidate at or below the kernel bound"
n_assert += 1
holders = [n for n in cands if abs(eff[n] - gb) <= tol]
assert holders, "no node reproduces the kernel bound"; n_assert += 1
print(f"bound-holding nodes (eff == global bound, open children): {len(holders)}")
h_st = collections.Counter(recs[n]["status"] for n in holders)
h_inh = sum(1 for n in holders if recs[n]["own"] < eff[n] - 1e-12)
print(f"  holders by status: {dict(h_st)}; holders whose bound is INHERITED (own < eff): {h_inh}")
h_und = sum(1 for n in holders if recs[n]["status"] in UNDECIDED)
h_unc = sum(1 for n in holders if recs[n]["status"] == "optimal" and not np.isfinite(recs[n]["own"]))
print(f"  holders that are undecided: {h_und}; uncertified: {h_unc}")


def chain_has(nid, pred):
    while nid is not None:
        if pred(recs[nid]):
            return True
        nid = recs[nid]["parent"]
    return False


h_anc = sum(1 for n in holders if chain_has(
    n, lambda r: r["status"] in UNDECIDED or (r["status"] == "optimal" and not np.isfinite(r["own"]))))
print(f"  holders with an undecided/uncertified node anywhere in their ancestry: {h_anc}")
if dumps:
    ds = [depth[n] for n in dumps]
    print(f"dumped nodes depth: min {min(ds)} median {int(np.median(ds))} max {max(ds)} "
          f"(all nodes: max depth {max(depth.values())})")
# Where the LpVerdictNumerical attempts went (per-node counter deltas, see obs()).
ids = sorted(recs)
prev_inf = {ids[i]: (i > 0 and recs[ids[i - 1]]["status"] == "infeasible") for i in range(len(ids))}
num_nodes = [n for n in ids if recs[n]["d"]["LpVerdictNumerical"] > 0]
by_st = collections.Counter(recs[n]["status"] for n in num_nodes)
after_inf = sum(1 for n in num_nodes if prev_inf[n])
tot_num = sum(recs[n]["d"]["LpVerdictNumerical"] for n in ids)
print(f"nodes whose counter window holds a Numerical attempt: {len(num_nodes)} / {len(ids)} "
      f"(attempts in windows: {tot_num}); by final node status {dict(by_st)}; "
      f"windows that also hold the previous infeasible node's dual-proof LP: {after_inf}")
dist = collections.Counter(recs[n]["d"]["LpVerdictNumerical"] for n in num_nodes)
print(f"  Numerical attempts per such window: {dict(sorted(dist.items()))}")
dense = sum(recs[n]["d"]["LpDenseRetries"] for n in ids)
print(f"  dense retries in windows: {dense}; nodes with a Numerical window that still ended "
      f"optimal with a FINITE safe bound: "
      f"{sum(1 for n in num_nodes if recs[n]['status'] == 'optimal' and np.isfinite(recs[n]['own']))}")
piv_num = sum(recs[n]["n_pivots"] for n in num_nodes)
piv_all = sum(r["n_pivots"] for r in recs.values())
print(f"pivots in Numerical-window nodes: {piv_num} of {piv_all} ({100 * piv_num / max(1, piv_all):.1f}%)")
piv_dump = sum(recs[n]["n_pivots"] for n in dumps)
print(f"pivots: all nodes {piv_all}, dumped nodes {piv_dump}")

# ---- robust re-solve of every dumped LP ----------------------------------------------
n_cmp = 0
if dumps:
    import highspy
    from pyscipopt import LP as SLP

    def highs_solve(d):
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = d["n"], d["m"]
        lp.col_cost_ = d["c"].astype(float)
        lp.col_lower_, lp.col_upper_ = d["l"].astype(float), d["u"].astype(float)
        lp.row_lower_ = lp.row_upper_ = d["b"].astype(float)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = d["col_ptr"].astype(np.int32)
        lp.a_matrix_.index_ = d["row_idx"].astype(np.int32)
        lp.a_matrix_.value_ = d["vals"].astype(float)
        h.passModel(lp)
        h.run()
        s = h.modelStatusToString(h.getModelStatus())
        return s, (h.getInfo().objective_function_value if s == "Optimal" else None)

    def soplex_solve(d):
        lp = SLP("node")
        inf = lp.infinity()
        bs = list(map(float, d["b"]))
        lp.addRows([[] for _ in range(d["m"])], bs, bs)  # rows first: columns cite them
        cp, ri, va = d["col_ptr"], d["row_idx"], d["vals"]
        entries = [[(int(ri[t]), float(va[t])) for t in range(cp[j], cp[j + 1])]
                   for j in range(d["n"])]
        lb = [float(v) if v > -1e20 else -inf for v in d["l"]]
        ub = [float(v) if v < 1e20 else inf for v in d["u"]]
        lp.addCols(entries, list(map(float, d["c"])), lb, ub)
        lp.solve()
        if lp.isOptimal():
            return "Optimal", lp.getObjVal()
        return ("PrimalInfeasible" if not lp.isPrimalFeasible() else "Other"), None

    out = collections.Counter()
    gains = []
    for nid, d in sorted(dumps.items()):
        r = recs[nid]
        inherited = eff[r["parent"]] if r["parent"] is not None else -np.inf
        hs, ho = highs_solve(d)
        ss, so = soplex_solve(d)
        n_cmp += 1
        if nid in fidelity:
            assert hs == "Optimal" and abs(ho - r["obj"]) <= 1e-6 * (1 + abs(ho)), (nid, hs, ho, r["obj"])
            assert ss == "Optimal" and abs(so - r["obj"]) <= 1e-6 * (1 + abs(so)), (nid, ss, so, r["obj"])
            n_assert += 2
            continue
        out[(r["status"], hs, ss)] += 1
        if ho is not None:
            gains.append(ho - inherited)
        if n_cmp <= 20 or nid in holders:
            print(f"  node {nid} d={depth[nid]} inhouse={r['status']} obj={r['obj']!r} "
                  f"inherited={inherited!r} | HiGHS {hs} {ho!r} | SoPlex {ss} {so!r}"
                  f"{'  <-- HOLDER' if nid in holders else ''}")
    print("re-solve outcomes (inhouse, HiGHS, SoPlex):", dict(out))
    if gains:
        g = np.array(gains)
        print(f"HiGHS obj - inherited bound over decided LPs: n={len(g)} min {g.min():.4g} "
              f"median {np.median(g):.4g} max {g.max():.4g}; >1e-6: {(g > 1e-6).sum()}")
    assert n_cmp == len(dumps); n_assert += 1
    print(f"export fidelity: {len(fidelity)} decided nodes reproduced by HiGHS and SoPlex")
    np.savez_compressed(os.path.join(HERE, f"dumps_{arm}_{max_nodes}.npz"),
                        ids=np.array(sorted(dumps)), **{f"{k}_{n}": v for n, d in dumps.items()
                                                        for k, v in d.items()})

assert res.bound is None or res.bound <= 3.0620073 + 1e-6, "bound above the oracle"; n_assert += 1
if res.objective is not None:
    assert res.objective >= 3.0620073 - 1e-5, "incumbent below the oracle"; n_assert += 1
print(f"RESULT arm={arm} nodes={max_nodes} bound={main['bound']!r} undecided={n_und} "
      f"uncertified={n_unc} holders={len(holders)} holders_und={h_und} holders_unc={h_unc} "
      f"holders_anc={h_anc} resolved={n_cmp} numerical_ctr={ctr.get('LpVerdictNumerical')}")
print(f"executed assertions: {n_assert}")
sys.exit(0 if n_assert else 1)
