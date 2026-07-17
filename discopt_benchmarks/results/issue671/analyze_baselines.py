"""Issue #671 — float64 baselines + rank/presolve structure on hda's root LP.

Reproduces the false-infeasible verdicts and measures how small an exact-arithmetic
core would be after sound presolve (fixed vars / singleton rows / empty rows-cols).
"""

from __future__ import annotations

import os
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    d = np.load(os.path.join(HERE, "hda_root_lp.npz"))
    shape = tuple(int(x) for x in d["A_shape"])
    A = sp.csr_matrix((d["A_data"], d["A_indices"], d["A_indptr"]), shape=shape)
    return A, d["b_ub"], d["bounds"], d["c"]


def float64_baselines(A, b, bounds, c):
    lb = bounds[:, 0].copy()
    ub = bounds[:, 1].copy()
    lb[~np.isfinite(lb)] = None if False else -np.inf
    bnds = list(zip(np.where(np.isfinite(bounds[:, 0]), bounds[:, 0], -np.inf),
                    np.where(np.isfinite(bounds[:, 1]), bounds[:, 1], np.inf)))
    for meth in ("highs", "highs-ds", "highs-ipm"):
        t0 = time.time()
        try:
            r = linprog(c, A_ub=A, b_ub=b, bounds=bnds, method=meth)
            print(f"  [{meth:9s}] status={r.status} ({r.message.strip()[:40]!r}) "
                  f"obj={r.fun if r.fun is not None else None}  {time.time()-t0:.1f}s")
        except Exception as e:  # noqa: BLE001
            print(f"  [{meth:9s}] EXC {e}  {time.time()-t0:.1f}s")


def elastic_phase1(A, b, bounds):
    """min sum of constraint violations s: A x - s <= b, s>=0. Feasible iff min=0."""
    m, n = A.shape
    Aec = sp.hstack([A, -sp.identity(m, format="csr")], format="csr")
    c2 = np.concatenate([np.zeros(n), np.ones(m)])
    bnds = list(zip(np.where(np.isfinite(bounds[:, 0]), bounds[:, 0], -np.inf),
                    np.where(np.isfinite(bounds[:, 1]), bounds[:, 1], np.inf)))
    bnds += [(0, np.inf)] * m
    t0 = time.time()
    r = linprog(c2, A_ub=Aec, b_ub=b, bounds=bnds, method="highs")
    print(f"  elastic phase-1 (float64 highs): status={r.status} "
          f"min_total_violation={r.fun if r.fun is not None else None:.3e}  {time.time()-t0:.1f}s")
    return r


def rank_structure(A):
    Ad = A.toarray()
    # scale optimally-ish via simple geometric mean per row/col is not needed for rank
    s = np.linalg.svd(Ad, compute_uv=False)
    smax = s[0]
    rank_1e14 = int(np.sum(s > smax * 1e-14))
    rank_1e7 = int(np.sum(s > smax * 1e-7))
    tiny = int(np.sum(s < smax * 1e-7))
    print(f"  SVD: sigma_max={smax:.3e} sigma_min(nz)={s[s>0].min():.3e} "
          f"cols={A.shape[1]} rank(1e-14)={rank_1e14} rank(1e-7)={rank_1e7} #tiny(<1e-7)={tiny}")


def presolve_reduce(A, b, bounds):
    """Sound exact-style presolve (float64 here just to MEASURE the irreducible core):
    repeatedly drop zero rows/cols, fix singleton-row implied bounds, remove fixed vars.
    Reports how many rows/cols survive — the size an exact solver would face."""
    A = A.tolil().copy()
    b = b.copy().astype(float)
    lo = bounds[:, 0].astype(float).copy()
    hi = bounds[:, 1].astype(float).copy()
    m, n = A.shape
    row_alive = np.ones(m, bool)
    col_alive = np.ones(n, bool)
    changed = True
    rounds = 0
    while changed and rounds < 50:
        changed = False
        rounds += 1
        Acsr = A.tocsr()
        for i in range(m):
            if not row_alive[i]:
                continue
            r = Acsr.getrow(i)
            idx = [j for j in r.indices if col_alive[j]]
            if len(idx) == 0:
                # 0 <= b[i]; infeasible if b<0
                row_alive[i] = False
                changed = True
            elif len(idx) == 1:
                j = idx[0]
                a = Acsr[i, j]
                # a*x_j <= b  ->  bound on x_j
                if a > 0:
                    hi[j] = min(hi[j], b[i] / a)
                else:
                    lo[j] = max(lo[j], b[i] / a)
                row_alive[i] = False
                changed = True
        # drop empty columns / fixed vars
        Acsc = A.tocsc()
        for j in range(n):
            if not col_alive[j]:
                continue
            col = Acsc.getcol(j)
            rows = [i for i in col.indices if row_alive[i]]
            if len(rows) == 0 or lo[j] == hi[j]:
                col_alive[j] = False
                changed = True
    nr, nc = int(row_alive.sum()), int(col_alive.sum())
    print(f"  presolve core after {rounds} rounds: {nr} rows x {nc} cols "
          f"(from {m}x{n}); nnz~{A.tocsr()[np.ix_(np.where(row_alive)[0], np.where(col_alive)[0])].nnz}")
    return nr, nc


def main():
    A, b, bounds, c = load()
    print(f"LP: {A.shape[0]} rows x {A.shape[1]} cols, nnz={A.nnz}, c_nnz={int(np.count_nonzero(c))}")
    print("\n[1] float64 objective-LP baselines (expect false-infeasible / numerical):")
    float64_baselines(A, b, bounds, c)
    print("\n[2] float64 elastic phase-1 feasibility:")
    elastic_phase1(A, b, bounds)
    print("\n[3] rank structure (raw matrix SVD):")
    rank_structure(A)
    print("\n[4] sound-presolve irreducible core size:")
    presolve_reduce(A, b, bounds)


if __name__ == "__main__":
    main()
