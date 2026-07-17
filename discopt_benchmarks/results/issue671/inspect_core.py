"""Issue #671 — characterize the 2-row float64 infeasibility core exactly."""
from __future__ import annotations

import os
from fractions import Fraction

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    d = np.load(os.path.join(HERE, "hda_root_lp.npz"))
    shape = tuple(int(x) for x in d["A_shape"])
    A = sp.csr_matrix((d["A_data"], d["A_indices"], d["A_indptr"]), shape=shape)
    return A, d["b_ub"], d["bounds"], d["c"]


def _bnds(bounds, cols):
    lo, hi = bounds[cols, 0], bounds[cols, 1]
    return list(zip(np.where(np.isfinite(lo), lo, -np.inf),
                    np.where(np.isfinite(hi), hi, np.inf)))


def infeas(A, b, bounds, rows, cols):
    subA = A[rows][:, cols]
    r = linprog(np.zeros(len(cols)), A_ub=subA, b_ub=b[rows],
                bounds=_bnds(bounds, cols), method="highs")
    return r.status == 2


def find_core(A, b, bounds, c):
    # re-derive the 2-row core deterministically: scan pairs among rows that
    # HiGHS flags. Reuse the deletion filter quickly.
    m = A.shape[0]
    rows = list(range(m))
    bnds = list(zip(np.where(np.isfinite(bounds[:, 0]), bounds[:, 0], -np.inf),
                    np.where(np.isfinite(bounds[:, 1]), bounds[:, 1], np.inf)))

    def inf(rws):
        r = linprog(c, A_ub=A[rws], b_ub=b[rws], bounds=bnds, method="highs")
        return r.status == 2

    while len(rows) > 60:
        h = rows[: len(rows) // 2]
        o = rows[len(rows) // 2:]
        if inf(h):
            rows = h
        elif inf(o):
            rows = o
        else:
            break
    i = 0
    while i < len(rows):
        t = rows[:i] + rows[i + 1:]
        if t and inf(t):
            rows = t
        else:
            i += 1
    return rows


def main():
    A, b, bounds, c = load()
    rows = find_core(A, b, bounds, c)
    subA = A[rows]
    cols = sorted(set(subA.indices.tolist()))
    print(f"core rows = {rows}")
    print(f"core cols = {cols}")
    col_names = None
    colf = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
                        "python", "tests", "data", "minlplib", "hda.col")
    if os.path.exists(colf):
        with open(colf) as fh:
            col_names = [ln.strip() for ln in fh]
    dense = A[rows][:, cols].toarray()
    print("\nCore block  A[rows, cols] x <= b:")
    for ri, i in enumerate(rows):
        terms = "  ".join(f"{dense[ri,k]:+.6e}*x[{cols[k]}]" for k in range(len(cols))
                          if dense[ri, k] != 0)
        print(f"  row {i}: {terms}   <=  {b[i]:.6e}")
    print("\nColumn bounds:")
    for k, jc in enumerate(cols):
        nm = f" ({col_names[jc]})" if col_names and jc < len(col_names) else ""
        print(f"  x[{jc}]{nm}: [{bounds[jc,0]:.6e}, {bounds[jc,1]:.6e}]")

    # exact feasibility witness for the 2-row block
    print("\nExact-rational witness search (Fraction):")
    F = Fraction
    lo = [F(float(bounds[j, 0])) for j in cols]
    hi = [F(float(bounds[j, 1])) if np.isfinite(bounds[j, 1]) else None for j in cols]
    # try x = lo (all at lower bound) and check exactly
    for label, pt in [("all-lb", [lo[k] for k in range(len(cols))]),
                      ("all-ub", [hi[k] if hi[k] is not None else lo[k] for k in range(len(cols))])]:
        ok = True
        maxv = F(0)
        for ri, i in enumerate(rows):
            lhs = F(0)
            for k in range(len(cols)):
                lhs += F(float(dense[ri, k])) * pt[k]
            v = lhs - F(float(b[i]))
            maxv = max(maxv, v)
            if v > 0:
                ok = False
        print(f"  point={label}: feasible={ok}  max(Ax-b)={float(maxv):.3e}")


if __name__ == "__main__":
    main()
