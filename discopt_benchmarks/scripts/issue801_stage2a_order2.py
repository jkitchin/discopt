"""#801 Stage 2a — tighter trilinear hulls via sparse order-2 moment on the stars.

#801 names "tighter trilinear hulls on the RLT-2 lift"; #764 recorded level-2 RLT
inert (0.8382→0.8382) but left one caveat open: its 213 trilinear terms used
recursive *trilinear McCormick*, and a tighter trilinear hull "could in principle
bind."

The dominating test of that caveat is a **sparse order-2 moment (Lasserre)
relaxation on each shared-variable star** (x0:{6,9,12}, x1:{7,10,13}, x2:{8,11,14},
x16:{15,17}). The order-2 moment matrix's PSD constraint relaxes the degree-3/4
monomials (the trilinear terms x_i x_j x_k) *tighter than recursive McCormick* and
keeps them mutually consistent — so it dominates both the literal trilinear-hull
refinement and the recorded recursive-McCormick RLT-2. If x17's bound does not move
under it, the trilinear/higher-order polyhedral+moment route is closed.

Built on the scaled [0,1] moment LP + rigorous PSD cutting plane (HiGHS per round;
every bound is a converged, valid lower bound). Baseline order-1 moment SDP was
inert (0.8401, Stage 2b). Kill: order-2 star bound on x17 gains < +0.005 / 0.8382.

Run: ``python discopt_benchmarks/scripts/issue801_stage2a_order2.py``
"""

from __future__ import annotations

import json
import os
from itertools import combinations_with_replacement

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import issue801_stage2b_moment_sdp as S  # noqa: E402
import issue801_stage2b_psd_cuts as P  # noqa: E402
import issue801_stage2b_scaled as SC  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from discopt._rust import model_to_repr  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402
from discopt.tightening import fbbt_box  # noqa: E402
from issue801_root_probe import NL, OBJ_VAR, ORACLE, RESULTS  # noqa: E402
from scipy.optimize import linprog  # noqa: E402

BASELINE = 0.8382369708575385
STARS = [(0, 6, 9, 12), (1, 7, 10, 13), (2, 8, 11, 14), (15, 16, 17)]


def star_basis(S_vars):
    """Degree-<=2 monomial basis over the star: [(),] + degree1 + degree2 (sorted tuples)."""
    b = [()]
    b += [(i,) for i in S_vars]
    b += [tuple(sorted(p)) for p in combinations_with_replacement(sorted(S_vars), 2)]
    return b


def mono(t):
    return tuple(sorted(t))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    model = from_nl(NL)
    repr_ = model_to_repr(model)
    bt = fbbt_box(model)
    lb, ub = bt.lb, bt.ub
    S.repr_sense = {k: repr_.constraint_sense(k) for k in range(repr_.n_constraints)}
    S.repr_rhs = {k: repr_.constraint_rhs(k) for k in range(repr_.n_constraints)}
    n, quad, lin, pairs, _ = S.build_qcqp(repr_, lb, ub)
    quad_s, lin_s, L, s = SC.scale_forms(n, quad, lin, lb, ub)
    L17, s17 = float(L[OBJ_VAR]), float(s[OBJ_VAR])

    # Base scaled moment LP (columns x + all X_ij), McCormick box + constraints.
    lb01, ub01 = np.zeros(n), np.ones(n)
    c_obj, A_rows, b_rows, bounds, idxX, ncol_base = P.build_full_moment_lp(
        n, quad_s, lin_s, lb01, ub01
    )

    # Extra columns for degree-3/4 monomials from every star's order-2 matrix.
    bases = {tuple(st): star_basis(st) for st in STARS}
    extra = {}  # monomial(len>=3) -> column
    col = ncol_base
    for st, bas in bases.items():
        for a in range(len(bas)):
            for b in range(a, len(bas)):
                mu = mono(bas[a] + bas[b])
                if len(mu) >= 3 and mu not in extra:
                    extra[mu] = col
                    col += 1
    ncol = col
    # extend objective and bounds (scaled monomials all in [0,1])
    c_obj = np.concatenate([c_obj, np.zeros(ncol - ncol_base)])
    bounds = list(bounds) + [(0.0, 1.0)] * (ncol - ncol_base)

    def col_of(mu):
        mu = mono(mu)
        if len(mu) == 0:
            return None
        if len(mu) == 1:
            return mu[0]
        if len(mu) == 2:
            return idxX[(mu[0], mu[1])]
        return extra[mu]

    def value(mu, z):
        c = col_of(mu)
        return 1.0 if c is None else float(z[c])

    def build_M2(bas, z):
        d = len(bas)
        M = np.zeros((d, d))
        for a in range(d):
            for b in range(a, d):
                M[a, b] = M[b, a] = value(bas[a] + bas[b], z)
        return M

    def psd_cut_order2(bas, v):
        """Row for vᵀ M2 v >= 0 over monomial columns.  Returns (row, rhs) as <=."""
        d = len(bas)
        coef = {}
        const = 0.0
        for a in range(d):
            for b in range(d):
                mu = mono(bas[a] + bas[b])
                c = col_of(mu)
                w = v[a] * v[b]
                if c is None:
                    const += w
                else:
                    coef[c] = coef.get(c, 0.0) + w
        r = np.zeros(ncol)
        for c, w in coef.items():
            r[c] = w
        # r·z + const >= 0  ->  -r·z <= const
        return -r, float(const)

    # Pad base moment-LP rows (width ncol_base) to the extended width ncol.
    pad = ncol - ncol_base
    A = [np.concatenate([np.asarray(r, dtype=np.float64), np.zeros(pad)]) for r in A_rows]
    b = list(b_rows)
    traj = []
    for rnd in range(300):
        Asp = sp.csr_matrix(np.array(A))
        res = linprog(c_obj, A_ub=Asp, b_ub=np.array(b), bounds=bounds, method="highs")
        if res.status != 0:
            traj.append({"round": rnd, "status": int(res.status)})
            break
        t17 = float(res.fun)
        z = res.x
        # base order-1 moment matrix PSD
        Mb = P.moment_matrix(z[:n], z, n, idxX)
        wb, Vb = np.linalg.eigh(Mb)
        worst = float(wb[0])
        added = 0
        for e in range(min(6, Mb.shape[0])):
            if wb[e] < -1e-8:
                r, rhs = P.psd_cut(Vb[:, e], n, idxX, ncol)
                A.append(r); b.append(rhs); added += 1
        # order-2 star matrices PSD
        star_worst = 0.0
        for st, bas in bases.items():
            M2 = build_M2(bas, z)
            w2, V2 = np.linalg.eigh(M2)
            star_worst = min(star_worst, float(w2[0]))
            for e in range(min(4, M2.shape[0])):
                if w2[e] < -1e-8:
                    r, rhs = psd_cut_order2(bas, V2[:, e])
                    A.append(r); b.append(rhs); added += 1
        traj.append({"round": rnd, "t17": t17, "x17": L17 + s17 * t17,
                     "base_lambda_min": worst, "star_lambda_min": star_worst,
                     "rows": len(A), "added": added})
        if rnd % 10 == 0 or added == 0:
            print(f"round {rnd:3d}  x17={L17 + s17 * t17:.6f}  base_lmin={worst:.2e} "
                  f"star_lmin={star_worst:.2e} rows={len(A)} added={added}")
        if added == 0:
            break

    final = next((e for e in reversed(traj) if e.get("x17") is not None), None)
    x17 = None if final is None else final["x17"]
    gain = None if x17 is None else x17 - BASELINE
    converged = bool(final and final.get("base_lambda_min", -1) >= -1e-7
                     and final.get("star_lambda_min", -1) >= -1e-7)
    verdict = "KILL (order-2 star moment inert — trilinear/higher-order route closed)"
    if gain is not None and gain >= 0.005:
        verdict = "GO-CANDIDATE (order-2 moves root — Stage 3)"
    out = {
        "method": "sparse order-2 moment (Lasserre) on shared-variable stars, scaled, PSD cutting-plane",
        "stars": [list(st) for st in STARS],
        "baseline": BASELINE,
        "order1_moment_sdp": 0.8401,
        "oracle": ORACLE,
        "baron_root": 0.955,
        "order2_x17_bound": x17,
        "gain_over_baseline": gain,
        "converged_psd": converged,
        "rounds": len(traj),
        "n_extra_monomial_cols": ncol - ncol_base,
        "kill_threshold": 0.005,
        "verdict": verdict,
        "trajectory_tail": traj[-4:],
    }
    print("\n" + json.dumps({k: out[k] for k in
          ["order2_x17_bound", "gain_over_baseline", "converged_psd", "rounds", "verdict"]},
          indent=2, default=str))
    with open(os.path.join(RESULTS, "stage2a_order2.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
