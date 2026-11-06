"""Attribute the kernel-vs-reference residual: reference FBBT with (PAD, tol) = kernel's
(1e-9, 1e-9) vs (1e-12, 1e-12), and the kernel at 15 / 100000 rounds. Prints counts (§6)."""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
WT = os.environ["DISCOPT_TREE"]  # the tree under test (CLAUDE.md §8)
import discopt._rust as R
assert R.__file__.startswith(WT) and hasattr(R, "propagate_spec_py")
import fbbt_ref
d = dict(np.load(sys.argv[1]))
KEYS = [k for k in d if not k.startswith("rec_") and k != "prop_rounds"]
kw = {k: (int(d[k]) if d[k].ndim == 0 else d[k]) for k in KEYS}
spec = fbbt_ref.Spec(kw)
def loose(alo, ahi, blo, bhi):
    return ((alo < blo - 1e-6 * (1 + abs(blo))) | (ahi > bhi + 1e-6 * (1 + abs(bhi)))) & np.isfinite(blo) & np.isfinite(bhi)
c = dict(boxes=0, k_vs_ref12=0, k_vs_ref9=0, ref9_vs_ref12=0, kinf_vs_ref9=0, ref9_vs_kinf=0)
worst = (1.0, None)
for b in range(len(d["rec_id"])):
    plo, phi, cut = d["rec_pre_lo"][b], d["rec_pre_hi"][b], float(d["rec_cutoff"][b])
    lk, klo, khi = R.propagate_spec_py(**kw, lo=plo, hi=phi, cutoff=cut, rounds=100000)
    fbbt_ref.PAD = 1e-12; l12, r12lo, r12hi, n12 = fbbt_ref.fbbt(spec, plo, phi, cut, tol=1e-12)
    fbbt_ref.PAD = 1e-9; l9, r9lo, r9hi, n9 = fbbt_ref.fbbt(spec, plo, phi, cut, tol=1e-9)
    if not (lk and l12 and l9): continue
    c["boxes"] += 1
    c["k_vs_ref12"] += int(loose(klo, khi, r12lo, r12hi).any())
    c["k_vs_ref9"] += int(loose(klo, khi, r9lo, r9hi).any())
    c["kinf_vs_ref9"] += int(loose(klo, khi, r9lo, r9hi).sum())
    c["ref9_vs_kinf"] += int(loose(r9lo, r9hi, klo, khi).sum())
    c["ref9_vs_ref12"] += int(loose(r9lo, r9hi, r12lo, r12hi).any())
    wk = khi - klo; wr = r12hi - r12lo
    with np.errstate(divide="ignore", invalid="ignore"):
        rat = np.where(wk > 1e-9, wr / wk, 1.0)
    j = int(np.nanargmin(rat))
    if rat[j] < worst[0]:
        worst = (float(rat[j]), dict(box=b, id=int(d["rec_id"][b]), col=j, k=(klo[j], khi[j]), r12=(r12lo[j], r12hi[j]), r9=(r9lo[j], r9hi[j]), n12=n12, n9=n9))
print(c); print("worst width ratio ref12/kernel", worst)
print("executed comparisons:", c["boxes"]); sys.exit(0 if c["boxes"] else 1)
