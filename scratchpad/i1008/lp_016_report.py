"""LP-level acceptance report for the feral 0.16.0 bump (#1008).

The 52-instance certifying panel answers soundness but cannot answer whether
0.16.0 helps: `docs/dev/baron-gap-plan.md` §1.3 puts node-LP at 0.06% of panel
wall on those families, so a fill improvement is invisible there by construction
while the rounding reshuffle it also causes is fully visible. This runs the
captured relaxation LPs instead, where the LU *is* the work.

Fill (`LuFactorNnz / LuBasisNnz`) and the factorization counters are exact
integers and load-independent, so they carry the claim. Wall is printed but the
two arms ran concurrently and finished at very different times, so their
contention patterns differ — per CLAUDE.md §9 no timing claim rests on it here.

The robustness column is the one to read alongside fill: `cold_fallback` counts
warm dual solves that fell off onto the cold primal path. §18b/§18c measured that
path returning false `Unbounded`/`Numerical` verdicts, so a bump that pushes more
LPs onto it is trading certificate risk for speed even when every objective in
this run happens to land on the HiGHS optimum.

§6: prints executed comparison counts and exits non-zero if any is zero.
§7: nothing is caught.
"""

from __future__ import annotations

import json
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
ATOL, RTOL = 1e-6, 1e-4


def load(n):
    return {json.loads(x)["tag"]: json.loads(x) for x in open(os.path.join(D, n)) if x.strip()}


A, B = load("lp_0151.jsonl"), load("lp_0160.jsonl")
K = sorted(set(A) & set(B))
assert K, "the two arms share no LPs"
print(f"# arms: 0.15.1={len(A)} 0.16.0={len(B)}; comparable on {len(K)}\n")

print("─── correctness vs HiGHS (zero slack, §1) ───")
bad, n_obj = [], 0
for k in K:
    for nm, r in (("0.15.1", A[k]), ("0.16.0", B[k])):
        if r["ref"] is None or r["status"] != "optimal":
            continue
        n_obj += 1
        if abs(r["obj"] - r["ref"]) > ATOL + RTOL * abs(r["ref"]):
            bad.append((k, nm, r["obj"], r["ref"]))
for k, nm, o, r in bad:
    print(f"  FAIL {k:24s} [{nm}] obj={o!r} ref={r!r}")
if not bad:
    print(f"  clean — {n_obj} arm x LP objectives at the HiGHS optimum")

print("\n─── status changes ───")
sc = [(k, A[k]["status"], B[k]["status"]) for k in K if A[k]["status"] != B[k]["status"]]
print(f"  {len(sc)} of {len(K)}")
for k, x, y in sc:
    print(f"    {k:24s} {x} -> {y}")

print("\n─── robustness: cold primal fallbacks ───")
cf = [(k, A[k]["cold_fallback"], B[k]["cold_fallback"]) for k in K]
gained = [(k, x, y) for k, x, y in cf if y > x]
lost = [(k, x, y) for k, x, y in cf if y < x]
print(f"  total {sum(x for _, x, _ in cf)} -> {sum(y for _, _, y in cf)}")
print(f"  LPs that GAINED a fallback: {len(gained)}")
for k, x, y in gained:
    a, b = A[k], B[k]
    print(
        f"    {k:24s} {x}->{y}  facs {a['facs']}->{b['facs']}  "
        f"p2 {a['p2']}->{b['p2']}  wall {a['wall']:.2f}->{b['wall']:.2f}"
    )
print(f"  LPs that LOST one: {len(lost)}")
for k, x, y in lost:
    print(f"    {k:24s} {x}->{y}")


def fill(r):
    return r["factor_nnz"] / r["basis_nnz"] if r["basis_nnz"] else float("nan")


print("\n─── fill and factor work ───")
print(f"{'LP':<24}{'fill 0.15.1':>12}{'fill 0.16.0':>12}{'facs':>14}{'fnnz %':>9}")
nb = nw = ns = 0
for k in K:
    a, b = A[k], B[k]
    d = (b["factor_nnz"] / a["factor_nnz"] - 1) * 100 if a["factor_nnz"] else float("nan")
    nb += d < -1
    nw += d > 1
    ns += abs(d) <= 1
    print(
        f"{k:<24}{fill(a):>11.2f}x{fill(b):>11.2f}x"
        f"{a['facs']:>7}/{b['facs']:<6}{d:>8.1f}%"
    )

ta = sum(A[k]["factor_nnz"] for k in K)
tb = sum(B[k]["factor_nnz"] for k in K)
ba = sum(A[k]["basis_nnz"] for k in K)
bb = sum(B[k]["basis_nnz"] for k in K)
print("\n─── totals ───")
print(f"  aggregate fill   {ta / ba:.2f}x -> {tb / bb:.2f}x")
print(f"  factor nnz       {ta:,} -> {tb:,}  ({(tb / ta - 1) * 100:+.1f}%)")
print(f"  factorizations   {sum(A[k]['facs'] for k in K):,} -> {sum(B[k]['facs'] for k in K):,}")
print(f"  per-LP factor nnz: better {nb}, worse {nw}, within 1% {ns}")
print(f"  wall {sum(A[k]['wall'] for k in K):.1f}s -> {sum(B[k]['wall'] for k in K):.1f}s"
      f"   (NOT a timing claim, §9: arms ran concurrently, unequal contention)")

print(f"\nexecuted comparisons: objectives {n_obj}, LP rows {len(K)}")
assert n_obj and K
sys.exit(1 if bad else 0)
