"""#2142 §114 step 1b: instrument check. SCIP root propagation (2142-propagation/scip_prop.py, the
§113 instrument, BASE_PARAMS) must NOT call a box empty when it contains a verified feasible point.
Boxes: x* +- delta * (global width, clipped to global box, unbounded cols use |x*|+1), binaries
fixed at x*, for several deltas; no cutoff and cutoff = 3.07 (above the optimum).
Hypothesis H1 (instrument artifact): SCIP calls some such box empty.  Kill of H1: none empty.
  python -u c2_instrument.py BOXES.npz XSTAR.npz"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "2142-propagation"))
import specio, scip_prop  # noqa: E402

d, kw = specio.load(sys.argv[1])
x = np.load(sys.argv[2])["x"]
r = specio.residuals(kw, x)
assert r["max"] < 1e-5
gl, gh = kw["global_lo"], kw["global_hi"]
w = np.where(np.isfinite(gh - gl) & (gh < 1e19) & (gl > -1e19), gh - gl, np.abs(x) + 1.0)
integ = np.asarray(kw["integrality"]).astype(bool)
sp = scip_prop.ScipProp(kw)
n = 0
bad = 0
for cut in (None, 3.07):
    for delta in (1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 0.0):
        lo = np.maximum(gl, x - delta * w)
        hi = np.minimum(gh, x + delta * w)
        lo[integ] = np.round(x[integ]); hi[integ] = np.round(x[integ])
        assert np.all(lo <= x) and np.all(x <= hi)
        st, L, H = sp.propagate(lo, hi, cut)
        n += 1
        empty = L is None
        bad += empty
        cont = None if empty else bool(np.all(L - 1e-6 <= x) and np.all(x <= H + 1e-6))
        print(f"cutoff={cut} delta={delta:g}: status={st} empty={empty} x*_in_result_box={cont}", flush=True)
print(f"boxes containing x* called empty: {bad} of {n}")
print(f"executed propagations: {n}")
sys.exit(0 if n else 1)
