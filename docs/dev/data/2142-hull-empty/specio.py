"""#2142 §114: shared helpers -- load a captured kernel spec and evaluate every spec row
INDEPENDENTLY of SCIP (plain numpy), so a point can be checked against the exact lifted system."""
import numpy as np

SPEC_KEYS = (
    "n_cols n_orig c integrality global_lo global_hi fixed_row_ptr fixed_cols fixed_coeffs "
    "fixed_rhs term_kind term_i term_j term_out term_p term_coeff term_cst blf_w blf_a_ptr "
    "blf_a_cols blf_a_coeffs blf_a_const blf_b_ptr blf_b_cols blf_b_coeffs blf_b_const "
    "obbt_candidates"
).split()


def load(path):
    d = dict(np.load(path))
    kw = {k: (int(d[k]) if d[k].ndim == 0 else d[k]) for k in SPEC_KEYS}
    return d, kw


def residuals(kw, x):
    """Returns dict of max violations: rows (A x - rhs)+, terms |x_w - f|, blf |x_w - A B|,
    integrality, and per-item arrays."""
    x = np.asarray(x, float)
    ptr, cols, co, rhs = kw["fixed_row_ptr"], kw["fixed_cols"], kw["fixed_coeffs"], kw["fixed_rhs"]
    act = np.array([co[ptr[r]:ptr[r + 1]] @ x[cols[ptr[r]:ptr[r + 1]]] for r in range(len(rhs))])
    row_v = np.maximum(act - rhs, 0.0)
    tv = []
    for k in range(len(kw["term_kind"])):
        kind, i, j, w, p = (int(kw[q][k]) for q in ("term_kind", "term_i", "term_j", "term_out", "term_p"))
        a, b = float(kw["term_coeff"][k]), float(kw["term_cst"][k])
        if kind == 0:
            f = x[i] * x[j]
        elif kind == 1:
            f = x[i] ** p
        elif kind == 2:
            f = (a * x[i] + b) ** 2
        elif kind == 3:
            f = np.sqrt(a * x[i] + b)
        elif kind == 4:
            f = (a * x[i] + b) ** p
        else:
            raise ValueError(kind)
        tv.append(abs(x[w] - f) / max(1.0, abs(f)))
    bv = []
    for k in range(len(kw["blf_w"])):
        s = slice(int(kw["blf_a_ptr"][k]), int(kw["blf_a_ptr"][k + 1]))
        A = kw["blf_a_const"][k] + kw["blf_a_coeffs"][s] @ x[kw["blf_a_cols"][s]]
        s = slice(int(kw["blf_b_ptr"][k]), int(kw["blf_b_ptr"][k + 1]))
        B = kw["blf_b_const"][k] + kw["blf_b_coeffs"][s] @ x[kw["blf_b_cols"][s]]
        bv.append(abs(x[int(kw["blf_w"][k])] - A * B) / max(1.0, abs(A * B)))
    integ = np.asarray(kw["integrality"]).astype(bool)
    iv = np.abs(x[integ] - np.round(x[integ])) if integ.any() else np.zeros(0)
    bnd = np.maximum(np.maximum(kw["global_lo"] - x, x - kw["global_hi"]), 0.0)
    return {
        "row": row_v, "term": np.array(tv), "blf": np.array(bv), "int": iv, "bound": bnd,
        "max": max(row_v.max(initial=0), max(tv, default=0), max(bv, default=0),
                   iv.max(initial=0), bnd.max(initial=0)),
    }
