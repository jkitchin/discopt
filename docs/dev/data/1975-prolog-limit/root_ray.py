"""#1975 class census (a): root McCormick LP status over the FBBT box and, when it is
uncertified, a negative-cost recession ray verified in rationals. One JSON line."""
import json, sys, time
from fractions import Fraction
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import discopt.modeling as dm
from discopt._relax import mccormick_lp as ML
from discopt._relax.milp_relaxation import MilpRelaxationModel
from discopt.tightening import fbbt_box

name, path = sys.argv[1], sys.argv[2]
out = {"name": name}
t0 = time.time()


def row_val(A, i, d):
    r = A.getrow(i)
    return sum((Fraction(float(a)) * d[j] for j, a in zip(r.indices, r.data)), Fraction(0))


def exact_ray(A, c, lo, hi):
    """Float LP over the recession cone, box-normalised |d_j| <= 1 on open sides;
    then the vertex in rationals with active-set repair. Returns (d, cost) or reason."""
    n = A.shape[1]
    bnds = []
    for j in range(n):
        l_ = 0.0 if np.isfinite(lo[j]) else -1.0
        u_ = 0.0 if np.isfinite(hi[j]) else 1.0
        bnds.append((l_, u_))
    if all(b == (0.0, 0.0) for b in bnds):
        return None, "no open column"
    lp = linprog(c, A_ub=A, b_ub=np.zeros(A.shape[0]), bounds=bnds, method="highs-ds")
    if lp.status != 0:
        return None, f"cone LP status {lp.status}"
    x = lp.x
    if lp.fun < -1e-7 or lp.fun > 1e-9:  # clear in floating point: no rational step
        if lp.fun > 1e-9:
            return None, f"float cost {lp.fun:.3g}: no negative-cost ray"
        d = [Fraction(float(v)) for v in x]
        return (d, Fraction(float(lp.fun))), f"float cost {lp.fun:.3g} (float verdict)"
    Ad = A.toarray()
    cand = []
    for j in range(n):  # columns at a bound of the normalisation box (incl. 0)
        for v in {bnds[j][0], bnds[j][1], 0.0}:
            if abs(x[j] - v) <= 1e-12:
                e = np.zeros(n); e[j] = 1.0; cand.append((e, v)); break
    slack = sorted((abs(Ad[i] @ x) / (1 + np.abs(Ad[i]).sum()), i) for i in range(A.shape[0]))
    cand += [(Ad[i], 0.0) for s_, i in slack if s_ <= 1e-9]
    forced = []
    for _ in range(A.shape[0] + 1):
        basis = []
        for e, h in forced + cand:
            if np.linalg.matrix_rank(np.array([r for r, _ in basis] + [e])) > len(basis):
                basis.append((e, h))
            if len(basis) == n:
                break
        if len(basis) < n:
            return None, f"float cost {lp.fun:.3g}; active set does not fix a vertex"
        M = [[Fraction(float(v)) for v in r] + [Fraction(float(h))] for r, h in basis]
        for col in range(n):
            p = next(k for k in range(col, n) if M[k][col] != 0)
            M[col], M[p] = M[p], M[col]
            inv = 1 / M[col][col]; M[col] = [v * inv for v in M[col]]
            for k in range(n):
                if k != col and M[k][col] != 0:
                    f = M[k][col]; M[k] = [a - f * b for a, b in zip(M[k], M[col])]
        d = [M[i][n] for i in range(n)]
        vals = [row_val(A, i, d) for i in range(A.shape[0])]
        worst = int(np.argmax([float(v) for v in vals]))
        if vals[worst] <= 0:
            break
        forced.append((Ad[worst], 0.0))
    else:
        return None, f"float cost {lp.fun:.3g}; repair did not converge"
    for j in range(n):  # cone membership, exactly
        if (np.isfinite(lo[j]) and d[j] < 0) or (np.isfinite(hi[j]) and d[j] > 0):
            return None, f"float cost {lp.fun:.3g}; rational vertex leaves the cone"
    cost = sum((Fraction(float(cj)) * dj for cj, dj in zip(c, d)), Fraction(0))
    return (d, cost), f"float cost {lp.fun:.3g}"


try:
    milps = []
    real = MilpRelaxationModel.solve
    def spy(self, *a, **k):
        milps.append(self); return real(self, *a, **k)
    MilpRelaxationModel.solve = spy
    m = dm.from_nl(path)
    bt = fbbt_box(m)
    lb, ub = np.asarray(bt.lb, float), np.asarray(bt.ub, float)
    res = ML.MccormickLPRelaxer(m).solve_at_node(lb, ub)
    out.update(n=len(lb), root_status=res.status, root_lb=res.lower_bound, n_lps=len(milps))
    if milps and (res.lower_bound is None or res.status != "optimal"):
        Mm = milps[0]
        A = sp.csr_matrix(Mm._A_ub)
        c = np.asarray(Mm._c, float)
        lo = [float(a) for a, _ in Mm._bounds]; hi = [float(b) for _, b in Mm._bounds]
        r, why = exact_ray(A, c, lo, hi)
        out["ray_note"] = why
        if r is not None:
            d, cost = r
            out["ray_cost"] = float(cost)
            out["ray_neg"] = bool(cost < 0)
            out["ray_open_orig"] = [j for j in range(len(lb)) if d[j] != 0]
            out["ray_dir"] = {j: float(d[j]) for j in range(len(lb)) if d[j] != 0}
except Exception as e:
    out["error"] = f"{type(e).__name__}: {e}"[:300]
out["wall"] = time.time() - t0
print(json.dumps(out), flush=True)
