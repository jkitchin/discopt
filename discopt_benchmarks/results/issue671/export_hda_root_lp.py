"""Issue #671 entry experiment — export hda's root McCormick LP.

Builds the McCormick LP relaxation for hda at its ROOT bound box and saves the
raw system (A_ub, b_ub, bounds, c, sense=min) so the exact/high-precision
experiment is reproducible without re-running discopt's relaxation builder.

Target shape from the diagnosis: 2974 rows x 1138 columns.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import scipy.sparse as sp

import discopt.modeling as dm
from discopt._jax.discretization import DiscretizationState
from discopt._jax.milp_relaxation import build_milp_relaxation
from discopt._jax.mccormick_lp import classify_nonlinear_terms

HERE = os.path.dirname(os.path.abspath(__file__))
HDA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
    "python", "tests", "data", "minlplib", "hda.nl",
)


def main() -> None:
    print(f"loading {HDA}")
    model = dm.from_nl(HDA)
    terms = classify_nonlinear_terms(model)

    n = len(model._variables)
    root_lb = np.array([float(np.min(v.lb)) for v in model._variables])
    root_ub = np.array([float(np.max(v.ub)) for v in model._variables])
    print(f"model: {n} structural variables (before lifting)")

    relax, info = build_milp_relaxation(
        model,
        terms,
        DiscretizationState(),
        bound_override=(root_lb, root_ub),
        skip_separable_floor=True,
        skip_convex_lift=True,
    )

    A = sp.csr_matrix(relax._A_ub, dtype=np.float64)
    A.sort_indices()
    b = np.asarray(relax._b_ub, dtype=np.float64).ravel()
    bounds = np.asarray(relax._bounds, dtype=np.float64)  # (ncol, 2): [lb, ub]
    c = np.asarray(relax._c, dtype=np.float64).ravel()

    print(f"A_ub shape = {A.shape}  nnz = {A.nnz}")
    print(f"b_ub shape = {b.shape}")
    print(f"bounds shape = {bounds.shape}")
    print(f"c shape = {c.shape}, nnz(c) = {int(np.count_nonzero(c))}")
    print(f"objective_bound_valid = {relax._objective_bound_valid}")

    ad = np.abs(A.data)
    ad = ad[ad > 0]
    print(f"|A| range: min {ad.min():.3e}  max {ad.max():.3e}  spread {ad.max()/ad.min():.3e}")

    out = os.path.join(HERE, "hda_root_lp.npz")
    np.savez_compressed(
        out,
        A_data=A.data, A_indices=A.indices, A_indptr=A.indptr, A_shape=np.array(A.shape),
        b_ub=b,
        bounds=bounds,
        c=c,
        sense=np.array([0]),  # 0 = minimize
    )
    print(f"saved {out}  ({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()
