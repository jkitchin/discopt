"""#801 Stage 2b (rigorous confirmation) — PSD cutting-plane on the moment LP.

SCS gave an *inaccurate* (max_iters) dense-moment-SDP objective of ~1.14 — not
trustworthy per CLAUDE.md §4. This confirms/refutes it with a **rigorous** solver:
each round solves the full moment LP with HiGHS (converges exactly), reconstructs
the moment matrix M, and adds the most-violated PSD eigenvector cut
``vᵀ M v ≥ 0`` (a valid linear inequality, since a feasible M must be PSD). The LP
bound rises monotonically toward the SDP optimum, and every intermediate value is
a valid lower bound produced by a converged solve.

If the bound climbs to ~1.14 → the SDP genuinely moves the root (real GO signal,
→ Stage 3 rigor+generality). If it stalls near 0.84 → the SCS number was an
artifact → KILL.

Run: ``python discopt_benchmarks/scripts/issue801_stage2b_psd_cuts.py``
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import issue801_stage2b_moment_sdp as S  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from discopt._rust import model_to_repr  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402
from discopt.tightening import fbbt_box  # noqa: E402
from issue801_root_probe import NL, OBJ_VAR, ORACLE, RESULTS  # noqa: E402
from scipy.optimize import linprog  # noqa: E402

BASELINE = 0.8382369708575385


def build_full_moment_lp(n, quad, lin, lb, ub):
    """Columns [x(0..n-1) | X_ij for ALL i<=j]. McCormick box on every entry."""
    idxX = {}
    col = n
    for i in range(n):
        for j in range(i, n):
            idxX[(i, j)] = col
            col += 1
    ncol = col
    A, b = [], []

    def row():
        return np.zeros(ncol)

    def add_le(r, rhs):
        A.append(r); b.append(rhs)

    def add(r, rhs, sense):
        if sense == "<=":
            add_le(r, rhs)
        elif sense == ">=":
            add_le(-r, -rhs)
        else:
            add_le(r, rhs); add_le(-r, -rhs)

    for i in range(n):
        li, ui = float(lb[i]), float(ub[i])
        # square McCormick X_ii = x_i^2
        r = row(); r[idxX[(i, i)]] = -1.0; r[i] += 2 * li; add_le(r, li * li)
        r = row(); r[idxX[(i, i)]] = -1.0; r[i] += 2 * ui; add_le(r, ui * ui)
        r = row(); r[idxX[(i, i)]] = 1.0; r[i] += -(li + ui); add_le(r, -li * ui)
        for j in range(i + 1, n):
            lj, uj = float(lb[j]), float(ub[j])
            c = idxX[(i, j)]
            r = row(); r[c] = -1.0; r[j] += li; r[i] += lj; add_le(r, li * lj)
            r = row(); r[c] = -1.0; r[j] += ui; r[i] += uj; add_le(r, ui * uj)
            r = row(); r[c] = 1.0; r[j] += -ui; r[i] += -lj; add_le(r, -ui * lj)
            r = row(); r[c] = 1.0; r[j] += -li; r[i] += -uj; add_le(r, -li * uj)

    for k, (q, a, c) in quad.items():
        r = row()
        for (i, j), qij in q.items():
            key = (i, j) if i <= j else (j, i)
            r[idxX[key]] += qij
        r[:n] += a
        add(r, float(S.repr_rhs[k]) - c, str(S.repr_sense[k]))
    for k, (a, c) in lin.items():
        r = row()
        r[:n] += a
        add(r, float(S.repr_rhs[k]) - c, str(S.repr_sense[k]))

    c_obj = np.zeros(ncol)
    c_obj[OBJ_VAR] = 1.0
    bounds = [(float(lb[i]), float(ub[i])) for i in range(n)]
    for i in range(n):
        for j in range(i, n):
            vals = [lb[i] * lb[j], lb[i] * ub[j], ub[i] * lb[j], ub[i] * ub[j]]
            bounds.append((min(vals), max(vals)))
    return c_obj, A, b, bounds, idxX, ncol


def moment_matrix(x, X_of, n, idxX):
    dim = n + 1
    M = np.zeros((dim, dim))
    M[0, 0] = 1.0
    for i in range(n):
        M[0, 1 + i] = M[1 + i, 0] = x[i]
    for (i, j), col in idxX.items():
        M[1 + i, 1 + j] = M[1 + j, 1 + i] = X_of[col]
    return M


def psd_cut(v, n, idxX, ncol):
    """Linear row for vᵀ M v >= 0 over columns [x | X].  Returns (row, rhs) as <=."""
    # vᵀMv = v0² + 2 v0 Σ v_{1+i} x_i + Σ_{i,j} v_{1+i} v_{1+j} X_ij  >= 0
    r = np.zeros(ncol)
    v0 = v[0]
    for i in range(n):
        r[i] += 2.0 * v0 * v[1 + i]
    for (i, j), col in idxX.items():
        coef = v[1 + i] * v[1 + j] * (1.0 if i == j else 2.0)
        r[col] += coef
    # r·z >= -v0²  ->  -r·z <= v0²
    return -r, float(v0 * v0)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    model = from_nl(NL)
    repr_ = model_to_repr(model)
    bt = fbbt_box(model)
    lb, ub = bt.lb, bt.ub
    S.repr_sense = {k: repr_.constraint_sense(k) for k in range(repr_.n_constraints)}
    S.repr_rhs = {k: repr_.constraint_rhs(k) for k in range(repr_.n_constraints)}
    n, quad, lin, pairs, _ = S.build_qcqp(repr_, lb, ub)

    c_obj, A_rows, b_rows, bounds, idxX, ncol = build_full_moment_lp(n, quad, lin, lb, ub)
    A = list(A_rows)
    b = list(b_rows)

    dim = n + 1
    traj = []
    n_base_rows = len(A)
    for rnd in range(200):
        Asp = sp.csr_matrix(np.array(A))
        res = linprog(c_obj, A_ub=Asp, b_ub=np.array(b), bounds=bounds, method="highs")
        if res.status != 0:
            traj.append({"round": rnd, "status": int(res.status), "bound": None})
            break
        bound = float(res.fun)
        z = res.x
        M = moment_matrix(z[:n], z, n, idxX)
        w, Vs = np.linalg.eigh(M)
        lam_min = float(w[0])
        traj.append({"round": rnd, "bound": bound, "lambda_min": lam_min, "n_rows": len(A)})
        print(f"round {rnd:3d}  bound={bound:.6f}  lambda_min(M)={lam_min:.3e}  rows={len(A)}")
        if lam_min >= -1e-7:
            break
        # add the most-violated eigenvector cut (and a few more negative ones)
        added = 0
        for e in range(min(4, dim)):
            if w[e] < -1e-7:
                r, rhs = psd_cut(Vs[:, e], n, idxX, ncol)
                A.append(r); b.append(rhs); added += 1
        if added == 0:
            break

    final = traj[-1]["bound"] if traj and traj[-1].get("bound") is not None else None
    gain = None if final is None else final - BASELINE
    out = {
        "method": "rigorous PSD cutting-plane on full moment LP (HiGHS per round)",
        "baseline": BASELINE,
        "mccormick_full_moment_lp_round0": traj[0]["bound"] if traj else None,
        "final_bound": final,
        "gain_over_baseline": gain,
        "oracle": ORACLE,
        "rounds": len(traj),
        "converged_psd": bool(traj and traj[-1].get("lambda_min", -1) >= -1e-7),
        "trajectory": traj,
        "kill_threshold": 0.005,
        "verdict": (
            "KILL (SDP inert)" if (gain is not None and gain < 0.005)
            else "GO-CANDIDATE (SDP moves root — verify generality + rigor in Stage 3)"
        ),
    }
    print("\n" + json.dumps({k: out[k] for k in
          ["mccormick_full_moment_lp_round0", "final_bound", "gain_over_baseline",
           "converged_psd", "rounds", "verdict"]}, indent=2))
    with open(os.path.join(RESULTS, "stage2b_psd_cuts.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
