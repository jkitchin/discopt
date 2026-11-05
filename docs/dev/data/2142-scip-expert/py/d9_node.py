"""#2142 scipx d9: one concrete node (id from argv) of discopt's 20k cstr big-M tree: converge the kernel's own
FBBT (100000 rounds), then test each dual proof row alone on that converged box (min activity vs rhs) and
report what it deduces. Prints executed-comparison count (§6)."""
import sys, math
import numpy as np
import discopt._rust as R
d = dict(np.load(sys.argv[1])); d1 = np.load(sys.argv[2]); P = np.load(sys.argv[3]); nid = int(sys.argv[4])
names = "yC1 yB1 yC2 yB2 yC3 yB3 yC4 yB4 yC5 yB5 yR1 yNR1 yR2 yNR2 yR3 yNR3 yR4 yNR4 yR5 yNR5".split()
for n in range(1, 6):
    names += [f"rA{n}", f"Q{n}", f"FA{n}", f"FB{n}", f"rB{n}", f"c{n}", f"V{n}", f"FRA{n}", f"FRB{n}", f"QFR{n}"]
names += "RA RB QR PA PB QP".split() + [f"aux{j}" for j in range(76, 105)]
kw = {k: (int(d[k]) if d[k].ndim == 0 else d[k]) for k in d if not k.startswith("meta_")}
k = int(np.where(d1["id"] == nid)[0][0])
cut = float(d1["cutoff"][k]); cut = cut if math.isfinite(cut) else None
ok, lo, hi = R.propagate_spec_py(**kw, lo=d1["lo"][k], hi=d1["hi"][k], cutoff=cut, rounds=100000)
lo, hi = np.asarray(lo), np.asarray(hi)
print(f"node {nid}: status {d1['status'][k]} tree bound {d1['bound'][k]:.4f} cutoff {cut}; converged FBBT nonempty={ok}")
interesting = [j for j in range(20) if hi[j] - lo[j] > 0.5] + [j for j in range(20, 76) if names[j].startswith(("Q", "QFR", "QP", "V"))]
print("  box: " + ", ".join(f"{names[j]}=[{lo[j]:.4g},{hi[j]:.4g}]" for j in interesting))
n_cmp = 0
for r in range(len(P["rhs"])):
    c = P["cols"][P["ptr"][r]:P["ptr"][r + 1]]; a = P["coefs"][P["ptr"][r]:P["ptr"][r + 1]]
    minact = float(np.where(a > 0, a * lo[c], a * hi[c]).sum()); n_cmp += 1
    if minact > P["rhs"][r] + 1e-7:
        print(f"  proof {r} (learned at node {P['node'][r]}) alone proves the box EMPTY: min activity {minact:.5g} > rhs {P['rhs'][r]:.5g}")
        print("     " + " ".join(f"{v:+.4g}*{names[j]}[{lo[j]:.4g},{hi[j]:.4g}]" for j, v in zip(c, a)))
        break
# --- minimal subset of proofs whose propagation (with the spec) empties the converged box ---
def with_rows(sel):
    kp = dict(kw); ptr = list(d["fixed_row_ptr"]); cols = list(d["fixed_cols"]); co = list(d["fixed_coeffs"]); rhs = list(d["fixed_rhs"])
    for r in sel:
        cols += list(P["cols"][P["ptr"][r]:P["ptr"][r + 1]]); co += list(P["coefs"][P["ptr"][r]:P["ptr"][r + 1]]); rhs.append(float(P["rhs"][r])); ptr.append(len(cols))
    kp.update(fixed_row_ptr=np.array(ptr, dtype=np.int64), fixed_cols=np.array(cols, dtype=np.int64), fixed_coeffs=np.array(co), fixed_rhs=np.array(rhs))
    return R.propagate_spec_py(**kp, lo=lo, hi=hi, cutoff=cut, rounds=100000)
allr = list(range(len(P["rhs"])))
okall = with_rows(allr)[0]
print("all 95 proofs + converged FBBT: nonempty =", okall)
if not okall:
    sel = allr[:]
    for r in allr:
        trial = [q for q in sel if q != r]
        if not with_rows(trial)[0]:
            sel = trial
    print("minimal subset:", sel)
    for r in sel:
        c = P["cols"][P["ptr"][r]:P["ptr"][r + 1]]; a = P["coefs"][P["ptr"][r]:P["ptr"][r + 1]]
        print(f"  proof {r} (learned at node {P['node'][r]}): " + " ".join(f"{v:+.4g}*{names[j]}" for j, v in zip(c, a)) + f" <= {P['rhs'][r]:.5g}")
    for r in sel:
        ok1, l1, h1 = with_rows([r]); l1, h1 = np.asarray(l1), np.asarray(h1)
        ch = [j for j in range(len(lo)) if (l1[j] > lo[j] + 1e-7 or h1[j] < hi[j] - 1e-7)] if ok1 else []
        print(f"  proof {r} alone: nonempty={ok1}; tightens " + ", ".join(f"{names[j]}:[{lo[j]:.4g},{hi[j]:.4g}]->[{l1[j]:.4g},{h1[j]:.4g}]" for j in ch[:8]))

print(f"executed comparisons: {n_cmp}")
sys.exit(0 if n_cmp else 1)
