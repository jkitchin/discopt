"""#2142 Part B: SCIP 6.3 root propagation (2142-propagation/scip_prop.py, BASE_PARAMS) of every
captured box.  python -u b3_scip.py BOXES.npz OUT.npz"""

import math, os, sys
import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "2142-propagation")
)
import scip_prop  # noqa: E402

d = dict(np.load(sys.argv[1]))
SPEC_KEYS = (
    "n_cols n_orig c integrality global_lo global_hi fixed_row_ptr fixed_cols fixed_coeffs "
    "fixed_rhs term_kind term_i term_j term_out term_p term_coeff term_cst blf_w blf_a_ptr "
    "blf_a_cols blf_a_coeffs blf_a_const blf_b_ptr blf_b_cols blf_b_coeffs blf_b_const "
    "obbt_candidates"
).split()
kw = {k: (int(d[k]) if d[k].ndim == 0 else d[k]) for k in SPEC_KEYS}
sp = scip_prop.ScipProp(kw)
n = kw["n_cols"]
L, H, live = [], [], []
for b in range(len(d["rec_id"])):
    cut = float(d["rec_cutoff"][b])
    st, lo, hi = sp.propagate(
        d["rec_pre_lo"][b], d["rec_pre_hi"][b], cut if math.isfinite(cut) else None
    )
    live.append(lo is not None)
    L.append(lo if lo is not None else np.full(n, np.nan))
    H.append(hi if hi is not None else np.full(n, np.nan))
    if (b + 1) % 20 == 0:
        print(f"[{b + 1}] empty so far {live.count(False)}", flush=True)
np.savez(sys.argv[2], lo=np.stack(L), hi=np.stack(H), live=np.array(live))
print("boxes", len(live), "SCIP-empty", live.count(False))
print(f"executed propagations: {len(live)}")
sys.exit(0 if live else 1)
