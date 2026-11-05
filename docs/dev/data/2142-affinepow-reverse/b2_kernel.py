"""#2142 Part B: kernel propagation (propagate_spec_py) of every captured box, at the tree's
round cap and at 100000 rounds, under whatever DISCOPT_AFFINE_POW_REVERSE the process has
(a Rust OnceLock: one setting per process).
  DISCOPT_TREE=<repo>/python [DISCOPT_AFFINE_POW_REVERSE=1] python -u b2_kernel.py BOXES.npz OUT.npz"""

import math, os, sys
import numpy as np

WT = os.environ["DISCOPT_TREE"]
import discopt._rust as R  # noqa: E402

assert R.__file__.startswith(WT), R.__file__
# CLAUDE.md §8: the build under test carries the Part B flag.
assert b"DISCOPT_AFFINE_POW_REVERSE" in open(R.__file__, "rb").read(), (
    "stale _rust: no reverse step"
)
d = dict(np.load(sys.argv[1]))
SPEC_KEYS = (
    "n_cols n_orig c integrality global_lo global_hi fixed_row_ptr fixed_cols fixed_coeffs "
    "fixed_rhs term_kind term_i term_j term_out term_p term_coeff term_cst blf_w blf_a_ptr "
    "blf_a_cols blf_a_coeffs blf_a_const blf_b_ptr blf_b_cols blf_b_coeffs blf_b_const "
    "obbt_candidates"
).split()
kw = {k: (int(d[k]) if d[k].ndim == 0 else d[k]) for k in SPEC_KEYS}
rounds = int(d["prop_rounds"])
out = {}
n_runs = 0
for tag, r in (("r15", rounds), ("rinf", 100000)):
    L, H, live = [], [], []
    for b in range(len(d["rec_id"])):
        cut = float(d["rec_cutoff"][b])
        cut = cut if math.isfinite(cut) else None
        ok, lo, hi = R.propagate_spec_py(
            **kw,
            lo=np.ascontiguousarray(d["rec_pre_lo"][b]),
            hi=np.ascontiguousarray(d["rec_pre_hi"][b]),
            cutoff=cut,
            rounds=r,
        )
        L.append(np.asarray(lo))
        H.append(np.asarray(hi))
        live.append(bool(ok))
        n_runs += 1
    out[f"{tag}_lo"], out[f"{tag}_hi"], out[f"{tag}_live"] = (
        np.stack(L),
        np.stack(H),
        np.array(live),
    )
    print(tag, "rounds", r, "boxes", len(live), "empty", int((~np.array(live)).sum()), flush=True)
out["reverse"] = np.array(os.environ.get("DISCOPT_AFFINE_POW_REVERSE", ""))
np.savez(sys.argv[2], **out)
print(f"executed propagations: {n_runs}")
sys.exit(0 if n_runs else 1)
