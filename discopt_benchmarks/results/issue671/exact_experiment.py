"""Issue #671 entry experiment — does higher-precision / exact arithmetic resolve
hda's false-infeasible root McCormick LP and yield a tight dual bound?

Three measurements:

  (E1) tau-relaxation homotopy (float64): min c^T x s.t. A x <= b + tau, bounds.
       As tau -> 0+ the objective increases toward the TRUE LP optimum from below.
       Tells us the BOUND MAGNITUDE the tight relaxation implies (near -5964? or
       near candidate-A's loose -1.80e10?), without needing an exact full solve.

  (E2) float64 "infeasibility core" (deletion filter) -> a SMALL subsystem that
       float64 HiGHS still declares infeasible. Then solve THAT block in EXACT
       rational arithmetic (fractions.Fraction, dense two-phase simplex, Bland).
       If exact says FEASIBLE with a witness, the float64 infeasible verdict on
       hda is proven to be a PRECISION ARTIFACT (H, false-infeasible half).

  (E3) exact-rational feasibility witness for a moderate reduced block carrying
       the ill-conditioning, plus its exact optimum (bound half, reduced scale).
"""

from __future__ import annotations

import os
import time
from fractions import Fraction

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    d = np.load(os.path.join(HERE, "hda_root_lp.npz"))
    shape = tuple(int(x) for x in d["A_shape"])
    A = sp.csr_matrix((d["A_data"], d["A_indices"], d["A_indptr"]), shape=shape)
    bounds = d["bounds"]
    return A, d["b_ub"], bounds, d["c"]


def _bnds(bounds):
    return list(zip(np.where(np.isfinite(bounds[:, 0]), bounds[:, 0], -np.inf),
                    np.where(np.isfinite(bounds[:, 1]), bounds[:, 1], np.inf)))


# ----------------------------------------------------------------------------- E1
def tau_homotopy(A, b, bounds, c):
    print("\n[E1] tau-relaxation homotopy (float64 HiGHS): min c^Tx s.t. Ax <= b+tau")
    bnds = _bnds(bounds)
    print(f"     {'tau':>10} {'status':>8} {'objective':>18}")
    for tau in [1e-2, 1e-4, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 0.0]:
        r = linprog(c, A_ub=A, b_ub=b + tau, bounds=bnds, method="highs")
        obj = f"{r.fun:.6f}" if r.fun is not None else "None"
        st = {0: "optimal", 2: "infeas", 3: "unbounded", 4: "numerical"}.get(r.status, str(r.status))
        print(f"     {tau:>10.0e} {st:>8} {obj:>18}")


# ----------------------------------------------------------------------------- E2 helpers
def float64_infeasible(A, b, bounds, c, rows):
    sub = A[rows]
    r = linprog(c, A_ub=sub, b_ub=b[rows], bounds=_bnds(bounds), method="highs")
    return r.status == 2  # infeasible


def deletion_filter(A, b, bounds, c, max_keep=400):
    """Greedy deletion filter: shrink the row set while float64 still reports
    infeasible, yielding a small float64-infeasibility core."""
    m = A.shape[0]
    rows = list(range(m))
    if not float64_infeasible(A, b, bounds, c, rows):
        print("     (full system not float64-infeasible on this obj?!)")
        return rows
    # coarse block reduction first (halving) to get under max_keep quickly
    rng = np.random.default_rng(0)
    while len(rows) > max_keep:
        half = rows[: len(rows) // 2]
        if float64_infeasible(A, b, bounds, c, half):
            rows = half
        else:
            other = rows[len(rows) // 2:]
            if float64_infeasible(A, b, bounds, c, other):
                rows = other
            else:
                # neither half alone infeasible; shuffle and retry a random subset
                sub = list(rng.choice(rows, size=max(max_keep, len(rows) * 3 // 4), replace=False))
                if float64_infeasible(A, b, bounds, c, sub):
                    rows = sub
                else:
                    break
    # one-by-one deletion pass
    i = 0
    while i < len(rows):
        trial = rows[:i] + rows[i + 1:]
        if trial and float64_infeasible(A, b, bounds, c, trial):
            rows = trial
        else:
            i += 1
    return rows


# --------------------------------------------------- exact dense two-phase simplex
def exact_lp_feasible(A_rows, b_vec, lo, hi):
    """Decide feasibility of {lo<=x<=hi, A x <= b} EXACTLY via a Phase-1
    dense simplex in fractions.Fraction. Returns (feasible, witness_or_None).

    Formulation (bounded-variable-free): shift x = lo + y, y>=0, y<=hi-lo.
    Add slacks s>=0 for A y <= b' and upper-bound rows y<=ub'. Phase-1 minimises
    the sum of artificials; feasible iff min == 0. Dense, exact, Bland's rule."""
    m = len(A_rows)
    n = len(lo)
    F = Fraction
    lo = [F(x).limit_denominator(10**15) if not np.isfinite(x) else F(x) for x in lo]
    # Build inequality rows in y-space:  A y <= b - A lo ; and y <= hi-lo (finite ub)
    G = []  # each row: (coeffs dict col->Fraction, rhs Fraction)
    for (cols, coefs), bi in zip(A_rows, b_vec):
        rhs = F(bi)
        d = {}
        for j, a in zip(cols, coefs):
            af = F(a)
            d[j] = d.get(j, F(0)) + af
            rhs -= af * lo[j]
        G.append((d, rhs))
    for j in range(n):
        if np.isfinite(hi[j]):
            G.append(({j: F(1)}, F(hi[j]) - lo[j]))
    # Standard form: G y + s = rhs, s>=0, y>=0. If rhs<0 flip row & add artificial.
    mm = len(G)
    # columns: y (n), slack (mm), artificial (added as needed)
    ncol = n + mm
    T = [[F(0)] * (ncol) for _ in range(mm)]
    rhs = [F(0)] * mm
    basis = [0] * mm
    art_cols = []
    for i, (d, r) in enumerate(G):
        row = T[i]
        sign = F(1)
        if r < 0:
            sign = F(-1)
        for j, v in d.items():
            row[j] = sign * v
        row[n + i] = sign  # slack (becomes -1 if flipped -> needs artificial)
        rhs[i] = sign * r
        if sign > 0:
            basis[i] = n + i  # slack basic
        else:
            # slack coeff is -1, need artificial
            ac = ncol + len(art_cols)
            art_cols.append(ac)
            basis[i] = ac
    if art_cols:
        for row in T:
            row.extend([F(0)] * len(art_cols))
        for k, ac in enumerate(art_cols):
            # find its row
            pass
        # set artificial identity columns
        ai = 0
        for i, (d, r) in enumerate(G):
            if basis[i] >= ncol:
                T[i][basis[i]] = F(1)
                ai += 1
        ncol += len(art_cols)
    # Phase-1 objective: minimise sum of artificials
    obj = [F(0)] * ncol
    for ac in art_cols:
        obj[ac] = F(1)

    def simplex(obj):
        # reduced costs z_j - c_j ; we minimise, use Bland
        iters = 0
        while True:
            iters += 1
            # compute reduced costs: c_j - c_B^T (B^{-1} A_j). We keep T as current
            # tableau already in basis form (each basis col is unit). Reduced cost:
            cb = [obj[basis[i]] for i in range(mm)]
            # cost row
            red = obj[:]
            for j in range(ncol):
                s = F(0)
                for i in range(mm):
                    s += cb[i] * T[i][j]
                red[j] = obj[j] - s
            # entering: most negative? Bland: smallest index with red<0
            enter = -1
            for j in range(ncol):
                if red[j] < 0:
                    enter = j
                    break
            if enter < 0:
                return iters
            # ratio test
            leave = -1
            best = None
            for i in range(mm):
                aij = T[i][enter]
                if aij > 0:
                    ratio = rhs[i] / aij
                    if best is None or ratio < best or (ratio == best and basis[i] < basis[leave]):
                        best = ratio
                        leave = i
            if leave < 0:
                return -1  # unbounded (shouldn't happen in phase-1)
            piv = T[leave][enter]
            T[leave] = [v / piv for v in T[leave]]
            rhs[leave] = rhs[leave] / piv
            for i in range(mm):
                if i == leave:
                    continue
                f = T[i][enter]
                if f != 0:
                    T[i] = [a - f * b for a, b in zip(T[i], T[leave])]
                    rhs[i] = rhs[i] - f * rhs[leave]
            basis[leave] = enter

    if not art_cols:
        # slacks already give a feasible basis
        y = [F(0)] * n
        return True, [float(lo[j] + y[j]) for j in range(n)]
    simplex(obj)
    p1 = F(0)
    for i in range(mm):
        if basis[i] in art_cols:
            p1 += rhs[i]
    feasible = (p1 == 0)
    witness = None
    if feasible:
        yval = [F(0)] * ncol
        for i in range(mm):
            if basis[i] < ncol:
                yval[basis[i]] = rhs[i]
        witness = [float(lo[j] + yval[j]) for j in range(n)]
    return feasible, witness


def csr_rows_as_lists(A, rows, cols=None):
    sub = A[rows]
    if cols is not None:
        sub = sub[:, cols]
    sub = sub.tocsr()
    out = []
    for i in range(sub.shape[0]):
        s, e = sub.indptr[i], sub.indptr[i + 1]
        out.append((list(sub.indices[s:e]), list(sub.data[s:e])))
    return out


def e2_infeasibility_core(A, b, bounds, c):
    print("\n[E2] float64 infeasibility core -> exact-rational feasibility")
    t0 = time.time()
    rows = deletion_filter(A, b, bounds, c, max_keep=250)
    print(f"     float64-infeasible core: {len(rows)} rows  ({time.time()-t0:.1f}s)")
    # columns touched by the core
    sub = A[rows]
    cols = sorted(set(sub.indices.tolist()))
    # remap
    colmap = {c_: k for k, c_ in enumerate(cols)}
    sub_rows = []
    for i in range(sub.shape[0]):
        s, e = sub.indptr[i], sub.indptr[i + 1]
        cc = [colmap[j] for j in sub.indices[s:e]]
        sub_rows.append((cc, list(sub.data[s:e])))
    lo = bounds[cols, 0]
    hi = bounds[cols, 1]
    print(f"     core touches {len(cols)} columns")
    # verify float64 still infeasible on the projected block (obj=0 feasibility)
    subA = A[rows][:, cols]
    r64 = linprog(np.zeros(len(cols)), A_ub=subA, b_ub=b[rows],
                  bounds=list(zip(np.where(np.isfinite(lo), lo, -np.inf),
                                  np.where(np.isfinite(hi), hi, np.inf))),
                  method="highs")
    print(f"     float64 HiGHS on core block: status={r64.status} "
          f"({'infeasible' if r64.status==2 else 'FEASIBLE/other'})")
    t1 = time.time()
    feas, wit = exact_lp_feasible(sub_rows, list(b[rows]), lo, hi)
    print(f"     EXACT rational feasibility of core block: "
          f"{'FEASIBLE' if feas else 'INFEASIBLE'}  ({time.time()-t1:.1f}s)")
    if feas and wit is not None:
        xw = np.array(wit)
        viol = (subA @ xw) - b[rows]
        print(f"     exact witness max Ax-b violation (float check) = {viol.max():.3e}")
    return rows, cols, feas


def main():
    A, b, bounds, c = load()
    print(f"LP: {A.shape[0]}x{A.shape[1]} nnz={A.nnz}")
    tau_homotopy(A, b, bounds, c)
    e2_infeasibility_core(A, b, bounds, c)


if __name__ == "__main__":
    main()
