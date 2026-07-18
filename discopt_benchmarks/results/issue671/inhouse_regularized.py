"""Does the IN-HOUSE simplex (not HiGHS) recover a good dual on hda's root LP via
tau-regularization? The issue forbids an external-solver rescue, so the good dual
must come from feral. Plain solve reproduces the numerical false-fail; the
tau-perturbed solves probe whether feral can certify the well-conditioned
neighbour and hand back a dual tight enough for a rigorous NS bound."""

from __future__ import annotations

import math
import os

import numpy as np
import scipy.sparse as sp

from discopt._rust import solve_lp_warm_csc_py

HERE = os.path.dirname(os.path.abspath(__file__))
import importlib.util

spec = importlib.util.spec_from_file_location(
    "rbe", os.path.join(HERE, "refined_bound_experiment.py")
)
rbe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rbe)


def load():
    d = np.load(os.path.join(HERE, "hda_root_lp.npz"))
    shape = tuple(int(x) for x in d["A_shape"])
    A = sp.csr_matrix((d["A_data"], d["A_indices"], d["A_indptr"]), shape=shape)
    return A, d["b_ub"], d["bounds"], d["c"]


def solve_inhouse(A, b, bounds, c, tau):
    """Standard form [A|I] x = b+tau, slacks in [0,inf). Returns (status, dual)."""
    m, n = A.shape
    A_std = sp.hstack([A, sp.identity(m, format="csc")], format="csc").tocsc()
    lb = bounds[:, 0].astype(float)
    ub = bounds[:, 1].astype(float)
    lb = np.where(np.isfinite(lb), lb, -1e20)
    ub = np.where(np.isfinite(ub), ub, 1e20)
    lb_std = np.concatenate([lb, np.zeros(m)])
    ub_std = np.concatenate([ub, np.full(m, 1e20)])
    c_std = np.concatenate([c.astype(float), np.zeros(m)])
    b_vec = (b + tau).astype(float)
    status, x_full, obj, iters, cs, bv, dual, ray = solve_lp_warm_csc_py(
        np.ascontiguousarray(c_std),
        m,
        n + m,
        np.ascontiguousarray(A_std.indptr, dtype=np.int64),
        np.ascontiguousarray(A_std.indices, dtype=np.int64),
        np.ascontiguousarray(A_std.data, dtype=np.float64),
        np.ascontiguousarray(b_vec),
        np.ascontiguousarray(lb_std),
        np.ascontiguousarray(ub_std),
        None,
        None,
    )
    return status, obj, np.asarray(dual) if dual is not None and len(dual) else None


def main():
    A, b, bounds, c = load()
    m, n = A.shape
    lb = np.where(np.isfinite(bounds[:, 0]), bounds[:, 0], -np.inf).astype(float)
    ub = np.where(np.isfinite(bounds[:, 1]), bounds[:, 1], np.inf).astype(float)
    lb, ub = rbe.fbbt_tighten(A.tocsc(), b, lb, ub)

    print(f"hda root LP {m}x{n}; in-house feral simplex, standard form [A|I]")
    print(f"{'tau':>10} {'status':>10} {'lp_obj':>16} {'g(y) sound LB':>18} {'finite':>7}")
    best = -math.inf
    for tau in [0.0, 1e-3, 3e-3, 5e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1, 2e-1]:
        status, obj, dual = solve_inhouse(A, b, bounds, c, tau)
        gstr, fin = "n/a", "-"
        if dual is not None:
            g, finite = rbe.ns_safe_bound(dual, A, b, c, lb, ub)
            gstr = f"{g:.4f}" if math.isfinite(g) else "-inf"
            fin = str(finite)
            if finite:
                best = max(best, g)
        objs = f"{obj:.4f}" if obj is not None and math.isfinite(obj) else "n/a"
        print(f"{tau:>10.0e} {status:>10} {objs:>16} {gstr:>18} {fin:>7}")
    print(f"\nBEST in-house sound lower bound = {best:.4f}")
    print(f"candidate A -1.80e10  |  true root LP ~ -6.468e4")
    if math.isfinite(best):
        print(f"SOUND (<= -6.468e4): {best <= -6.468e4 + 1e-3*6.468e4}")
        print(f"TIGHT (>> -1.8e10):  {best > -1e6}")


if __name__ == "__main__":
    main()
