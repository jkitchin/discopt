"""Compare refactorization-interval arms. Counters are exact; wall is directional
only (the R1 panel was running concurrently — CLAUDE.md §9)."""
import json, sys, os

TOL_ABS, TOL_REL = 1e-6, 1e-4
arms = {}
for iv in ("48", "100", "200", "500"):
    p = f"scratchpad/i1008/refac_{iv}.jsonl"
    if os.path.exists(p):
        arms[iv] = {json.loads(l)["tag"]: json.loads(l) for l in open(p)}
base = arms["48"]
tags = sorted(set.intersection(*[set(a) for a in arms.values()]))
assert tags, "arms share no LP"

n_cmp = 0
bad = []
print(f"{'interval':>9}{'facs':>8}{'factor_nnz':>14}{'basis_nnz':>12}{'cap':>7}{'ftfail':>8}{'wall_s':>9}")
for iv, a in arms.items():
    facs = sum(a[t]["facs"] for t in tags)
    fnnz = sum(a[t]["factor_nnz"] for t in tags)
    bnnz = sum(a[t]["basis_nnz"] for t in tags)
    cap = sum(a[t]["dual_refac_cap"] for t in tags)
    ft = sum(a[t]["dual_refac_ft"] for t in tags)
    wall = sum(a[t]["wall"] for t in tags)
    print(f"{iv:>9}{facs:>8}{fnnz:>14}{bnnz:>12}{cap:>7}{ft:>8}{wall:>9.2f}")

# Correctness, zero slack: every arm must agree with HiGHS on every LP.
for iv, a in arms.items():
    for t in tags:
        r = a[t]
        n_cmp += 1
        if r["ref"] is None:
            continue
        tol = max(TOL_ABS, TOL_REL * abs(r["ref"]))
        if r["status"] != "optimal":
            bad.append((iv, t, f"status={r['status']}"))
        elif abs(r["obj"] - r["ref"]) > tol:
            bad.append((iv, t, f"obj={r['obj']:.9g} vs HiGHS {r['ref']:.9g}"))

b48, f48 = sum(base[t]["facs"] for t in tags), sum(base[t]["factor_nnz"] for t in tags)
if "500" in arms:
    b5 = sum(arms["500"][t]["facs"] for t in tags)
    f5 = sum(arms["500"][t]["factor_nnz"] for t in tags)
    print(f"\n48 -> 500: factorizations {b48} -> {b5} ({(1-b5/b48)*100:.1f}% fewer), "
          f"factor nnz {f48/1e6:.2f}M -> {f5/1e6:.2f}M ({(1-f5/f48)*100:.1f}% less)")
print(f"fill ratio at 48: {f48/sum(base[t]['basis_nnz'] for t in tags):.2f}x")
print(f"\nLPs: {len(tags)}   arms: {sorted(arms)}")
print(f"correctness deviations: {len(bad)}")
for x in bad:
    print("   ", x)
print(f"\nexecuted: comparisons={n_cmp}")
sys.exit(1 if not n_cmp else 0)
