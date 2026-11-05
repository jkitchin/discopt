"""#2142: map every ORIGINAL spec row through SCIP's aggregation maps (active columns only) and
classify it against the translated (presolved) spec: empty, implied by the translated box
(max activity <= rhs), parallel to a translated row (same support, proportional coefs), other.
Saves the mapped non-empty rows to rows_mapped.npz for the 'pre+rows' arm."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import translate as T
d = dict(np.load(T.SPEC))
spec0 = {k: (int(v) if k in ("n_cols", "n_orig") else v) for k, v in d.items() if not k.startswith("meta_")}
spec, expand, compress, info, amap = T.build_spec(spec0)
n0 = spec0["n_cols"]
lo, hi = spec["global_lo"], spec["global_hi"]
def sub(j):
    return amap[j] if j in amap else ({j: 1.0}, 0.0)
def rows_of(s):
    p, c, a, b = s["fixed_row_ptr"], s["fixed_cols"], s["fixed_coeffs"], s["fixed_rhs"]
    return [({int(c[q]): float(a[q]) for q in range(p[r], p[r + 1])}, float(b[r])) for r in range(len(b))]
def key(row):
    ks = tuple(sorted(row)); s = max(abs(v) for v in row.values())
    return ks, tuple(round(row[k] / s, 9) for k in ks)
trans = rows_of(spec); tkeys = {key(r) for r, _ in trans if r}
cnt = {"empty": 0, "implied_by_box": 0, "parallel_to_presolved": 0, "other": 0}
mapped = []
for row, b in rows_of(spec0):
    out, rhs = {}, b
    for j, a in row.items():
        r2, c2 = sub(j); rhs -= a * c2
        for k, v in r2.items(): out[k] = out.get(k, 0.0) + a * v
    out = {k: v for k, v in out.items() if abs(v) > 1e-12}
    if not out:
        assert rhs >= -1e-6, rhs; cnt["empty"] += 1; continue
    mx = sum(v * (hi[k] if v > 0 else lo[k]) for k, v in out.items())
    if mx <= rhs + 1e-9: cnt["implied_by_box"] += 1
    elif key(out) in tkeys: cnt["parallel_to_presolved"] += 1
    else: cnt["other"] += 1
    mapped.append((out, rhs))
print("original rows:", len(rows_of(spec0)), "translated rows:", len(trans), "census:", cnt)
ptr, cols, co, rh = [0], [], [], []
for r, b in mapped:
    cols += list(r); co += list(r.values()); rh.append(b); ptr.append(len(cols))
np.savez(os.path.join(HERE, "rows_mapped.npz"), ptr=ptr, cols=cols, coefs=co, rhs=rh)
assert sum(cnt.values()) == len(rows_of(spec0)); print("executed assertions: 1")
