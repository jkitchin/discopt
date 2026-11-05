"""#2142 §114 step 1c: is SCIP's emptiness on the 94 captured boxes a sliver/tolerance effect?
Widen every continuous column of each captured pre-propagation box by an absolute eps (clipped to
the global box; integer columns rounded to their integral range) and re-run the §113 SCIP root
propagation with the box's cutoff.  H2: emptiness disappears at eps <= 1e-6 (i.e. it lives in the
kernel's ~1e-9 slivers).  Kill of H2: still all empty at eps = 1e-6.
  python -u c3_widen.py BOXES.npz"""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "2142-propagation"))
import specio, scip_prop  # noqa: E402

d, kw = specio.load(sys.argv[1])
gl, gh = kw["global_lo"], kw["global_hi"]
integ = np.asarray(kw["integrality"]).astype(bool)
sp = scip_prop.ScipProp(kw)
nb = len(d["rec_id"])
n = 0
for eps in (0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
    empty = 0
    for b in range(nb):
        lo = np.maximum(gl, d["rec_pre_lo"][b] - eps)
        hi = np.minimum(gh, d["rec_pre_hi"][b] + eps)
        lo[integ] = np.ceil(d["rec_pre_lo"][b][integ] - 1e-6)
        hi[integ] = np.floor(d["rec_pre_hi"][b][integ] + 1e-6)
        cut = float(d["rec_cutoff"][b])
        st, L, _ = sp.propagate(lo, hi, cut if math.isfinite(cut) else None)
        empty += L is None
        n += 1
    print(f"eps={eps:g}: SCIP-empty {empty}/{nb}", flush=True)
print(f"executed propagations: {n}")
sys.exit(0 if n else 1)
