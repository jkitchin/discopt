"""#2142: derive SCIP-style aggregations from the homog kernel spec with no SCIP input.
1. root FBBT (the kernel's own propagate_spec_py, 15 rounds) -> fixed columns;
2. drop fixed columns from every fixed row (their contribution -> rhs); a row of support
   {j,k} whose negation is also a row is a doubleton EQUALITY a x_j + b x_k = c  ->
   aggregate x_j := (c - b x_k)/a  (choose j as the column of larger index, as SCIP does
   for aux-free ties here; a column aggregated once is never a representative);
3. iterate to closure (an aggregation can expose new parallel pairs).
Compares the result with SCIP's presolve list (presolved_homog.cip). Saves aggr.npz
(j, k, a, b meaning x_j = a*x_k + b) and prints an executed-assertion count."""
import re, sys
import numpy as np
import discopt._rust as R
assert R.__file__.startswith("/home/user/discopt/python"), R.__file__
n_assert = 1
SPEC = "/home/user/discopt/docs/dev/data/2142-exact-hull/scip/cstr_homog_spec.npz"
d = dict(np.load(SPEC))
n = int(d["n_cols"]); n_orig = int(d["n_orig"]); integ = d["integrality"].astype(bool)
lo0, hi0 = d["global_lo"].astype(float), d["global_hi"].astype(float)
kw = {k: d[k] for k in ("n_cols","n_orig","c","integrality","global_lo","global_hi","fixed_row_ptr","fixed_cols","fixed_coeffs","fixed_rhs","term_kind","term_i","term_j","term_out","term_p","term_coeff","term_cst","blf_w","blf_a_ptr","blf_a_cols","blf_a_coeffs","blf_a_const","blf_b_ptr","blf_b_cols","blf_b_coeffs","blf_b_const","obbt_candidates")}
for k in ("n_cols","n_orig"): kw[k] = int(kw[k])
for k in list(kw):
    if isinstance(kw[k], np.ndarray):
        kw[k] = kw[k].astype(np.int64) if kw[k].dtype.kind in "iub" else kw[k].astype(float)
live, lo, hi = R.propagate_spec_py(**kw, lo=lo0.copy(), hi=hi0.copy(), cutoff=None, rounds=15)
lo, hi = np.asarray(lo), np.asarray(hi)
assert live; n_assert += 1
fixedv = {j: 0.5 * (lo[j] + hi[j]) for j in range(n) if hi[j] - lo[j] <= 1e-7}
print("root FBBT fixed:", len(fixedv))
ptr, cols, coefs, rhs = d["fixed_row_ptr"], d["fixed_cols"], d["fixed_coeffs"], d["fixed_rhs"]
rows = []
for r in range(len(rhs)):
    cc = cols[ptr[r]:ptr[r+1]].tolist(); aa = coefs[ptr[r]:ptr[r+1]].tolist()
    rows.append((dict(zip(cc, aa)), float(rhs[r])))
# substitution map: j -> (k, a, b) with x_j = a x_k + b ; fixed: j -> (None, 0, v)
sub = {j: (None, 0.0, v) for j, v in fixedv.items()}
def reduce_row(row, b):
    out = {}
    for j, a in row.items():
        if j in sub:
            k, s, t = sub[j]
            b -= a * t
            if k is not None:
                out[k] = out.get(k, 0.0) + a * s
        else:
            out[j] = out.get(j, 0.0) + a
    return {j: a for j, a in out.items() if abs(a) > 1e-12}, b
n_rounds = 0
while True:
    n_rounds += 1
    red = [reduce_row(r, b) for r, b in rows]
    index = {}
    for i, (r, b) in enumerate(red):
        if len(r) == 2:
            key = tuple(sorted(r))
            index.setdefault(key, []).append(i)
    new = 0
    for key, idx in index.items():
        if len(idx) < 2: continue
        j, k = key
        if j in sub or k in sub: continue
        for i1 in idx:
            r1, b1 = red[i1]
            for i2 in idx:
                if i2 == i1: continue
                r2, b2 = red[i2]
                # r2 == -r1 (scaled)?
                s = -r2[j] / r1[j]
                if s <= 0: continue
                if abs(r2[k] + s * r1[k]) <= 1e-9 * (1 + abs(r1[k])) and abs(b2 + s * b1) <= 1e-9 * (1 + abs(b1)):
                    # equality r1: a_j x_j + a_k x_k = b1. aggregate the larger-index column
                    # (keeps the kernel's lower-index 'original' representative), unless it is
                    # integer and the other is continuous.
                    jj, kk = (max(j, k), min(j, k))
                    if integ[jj] and not integ[kk]: jj, kk = kk, jj
                    a_j, a_k = r1[jj], r1[kk]
                    sub[jj] = (kk, -a_k / a_j, b1 / a_j)
                    new += 1
                    break
            if jj in sub if len(idx) else False: break
    if new == 0: break
    print(f"  round {n_rounds}: +{new} aggregations")
# resolve chains: x_j = a x_k + b with k itself aggregated
def resolve(j):
    k, a, b = sub[j]
    while k is not None and k in sub:
        k2, a2, b2 = sub[k]
        k, a, b = k2, a * a2, a * b2 + b
    return k, a, b
agg = {j: resolve(j) for j in sub if sub[j][0] is not None}
print(f"derived aggregations: {len(agg)} (plus {len(fixedv)} fixed); representatives {len(set(k for k,_,_ in agg.values()))}")
# compare with SCIP
txt = open("presolved_homog.cip").read()
scip = {}
for mm in re.finditer(r"\] <t_x(\d+)>:.*?(aggregated|negated): (.*)", txt):
    scip[int(mm.group(1))] = mm.group(3)
single = {j: v for j, v in scip.items() if v.count("<t_x") == 1}
both = set(agg) & set(scip); only_us = set(agg) - set(scip); only_scip = set(scip) - set(agg)
print(f"SCIP: {len(scip)} aggregated/negated ({len(single)} single-column). overlap {len(both)}, only derived {len(only_us)}, only SCIP {len(only_scip)}")
print("  only derived:", sorted(only_us)[:30]); print("  only SCIP:", [(j, scip[j]) for j in sorted(only_scip)][:30])
# check a few derived ones against SCIP's formula
mism = 0
for j in sorted(both):
    k, a, b = agg[j]
    m2 = re.match(r"\s*([+-]?[\d.e-]+)?\s*([+-]?[\d.e-]*)<t_x(\d+)>", scip[j].replace("1 - <", "+1 -1<").replace("+1<", "+1.0<") if "negated" not in scip[j] else "+1 -1<t_x%s>" % re.search(r"<t_x(\d+)>", scip[j]).group(1))
    if m2 and int(m2.group(3)) != k: mism += 1
print("derived vs SCIP representative mismatches (chains may differ):", mism)
np.savez("aggr.npz", j=np.array(list(agg), dtype=np.int64), k=np.array([agg[j][0] for j in agg], dtype=np.int64),
         a=np.array([agg[j][1] for j in agg]), b=np.array([agg[j][2] for j in agg]),
         fixed_j=np.array(list(fixedv), dtype=np.int64), fixed_v=np.array(list(fixedv.values())))
assert len(agg) > 0; n_assert += 1
print(f"executed assertions: {n_assert}")
