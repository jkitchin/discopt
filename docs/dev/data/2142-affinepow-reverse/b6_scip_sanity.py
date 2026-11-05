"""#2142 Part B: sanity of the SCIP propagation instrument on the hull spec: the global box and
the first captured boxes with no cutoff and with a cutoff ABOVE the optimum (3.07); a box that
contains a feasible point must never be called empty without a cutoff.
  python -u b6_scip_sanity.py BOXES.npz"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "2142-propagation"))
import scip_prop  # noqa: E402
d = dict(np.load(sys.argv[1]))
SPEC_KEYS = ("n_cols n_orig c integrality global_lo global_hi fixed_row_ptr fixed_cols fixed_coeffs "
             "fixed_rhs term_kind term_i term_j term_out term_p term_coeff term_cst blf_w blf_a_ptr "
             "blf_a_cols blf_a_coeffs blf_a_const blf_b_ptr blf_b_cols blf_b_coeffs blf_b_const "
             "obbt_candidates").split()
kw = {k: (int(d[k]) if d[k].ndim == 0 else d[k]) for k in SPEC_KEYS}
sp = scip_prop.ScipProp(kw)
n = 0
for cut in (None, 3.07, float(d["rec_cutoff"][0])):
    st, lo, _ = sp.propagate(d["global_lo"], d["global_hi"], cut)
    print(f"global box cutoff={cut}: status={st} empty={lo is None}"); n += 1
    res = [sp.propagate(d["rec_pre_lo"][b], d["rec_pre_hi"][b], cut)[1] is None for b in range(10)]
    print(f"  first 10 captured boxes cutoff={cut}: empty {sum(res)}/10"); n += 10
print(f"executed propagations: {n}")
