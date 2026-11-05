"""#2142 §114 step 3 (diagnosis): the kernel's own propagation (propagate_spec_py, the tree's
fixpoint) on each captured box as captured, and with the integer columns' bounds rounded to
integers BEFORE the call ([ceil(lo-1e-6), floor(hi+1e-6)], the kernel's own step-5 rule).
H4: the kernel's step-5 integer rounding is suppressed when the rounded bound is within EPS=1e-9
of the current one (`nh < hi - EPS`), so a binary left at [0, 1e-9] keeps its eps-perspective
w = 1/((1-1e-8) y + 1e-8) at [9.09e7, 1e8] instead of the point 1e8, and the box is not refuted.
Kill of H4: pre-rounding empties < 50 % of the 91 boxes c10 refutes by forward evaluation.
Also counts binaries sitting at a sub-EPS sliver ([k, k+1e-9] or [k-1e-9, k]).
  python -u c11_kernel_snap.py BOXES.npz [rounds]"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import specio  # noqa: E402
import discopt._rust as R  # noqa: E402

print("rust", R.__file__, "DISCOPT_INT_ROUND_EXACT=", os.environ.get("DISCOPT_INT_ROUND_EXACT", ""))
d, kw = specio.load(sys.argv[1])
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else int(d["prop_rounds"])
integ = np.asarray(kw["integrality"]).astype(bool)
nb = len(d["rec_id"])
n = 0
cnt = {"asis": 0, "snapped": 0}
sliv = 0
for b in range(nb):
    lo, hi = d["rec_pre_lo"][b].copy(), d["rec_pre_hi"][b].copy()
    s = integ & (((hi - np.floor(hi + 1e-6)) > 0) | ((np.ceil(lo - 1e-6) - lo) > 0))
    sliv += int(s.sum())
    cut = float(d["rec_cutoff"][b]); cut = cut if math.isfinite(cut) else None
    for tag in ("asis", "snapped"):
        l, h = lo.copy(), hi.copy()
        if tag == "snapped":
            l[integ] = np.ceil(l[integ] - 1e-6); h[integ] = np.floor(h[integ] + 1e-6)
        ok, _, _ = R.propagate_spec_py(**kw, lo=np.ascontiguousarray(l), hi=np.ascontiguousarray(h),
                                       cutoff=None, rounds=rounds)
        cnt[tag] += (not ok); n += 1
print(f"rounds {rounds}, no cutoff: kernel-empty as captured {cnt['asis']}/{nb}, "
      f"with integer bounds pre-rounded {cnt['snapped']}/{nb}")
print(f"integer columns with a non-integral bound across boxes: {sliv}")
print(f"executed propagations: {n}")
sys.exit(0 if n else 1)
