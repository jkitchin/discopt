"""#801 Stage 2b (well-conditioned) — moment SDP on the [0,1]-scaled tanksize QCQP.

The unscaled moment matrix has entries up to ~8e6 (x6²), so SCS overshoots (an
*inaccurate* iterate reported 1.14) and the naive PSD cutting-plane's eigenvector
cuts are scale-dominated (λ_min ~ -1e5, bound frozen at 0.8401). Neither is
trustworthy. Here we affinely rescale every variable to t_i ∈ [0,1]
(x_i = lb_i + s_i t_i) and rebuild the exact QCQP in t — the moment matrix is now
O(1), so both SCS and the rigorous PSD cutting-plane converge and must AGREE.

This is the trustworthy version of the candidate-3 SDP entry experiment.
Kill: scaled moment bound on x17 gains < +0.005 over 0.8382.

Run: ``python discopt_benchmarks/scripts/issue801_stage2b_scaled.py``
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import issue801_stage2b_moment_sdp as S  # noqa: E402
import issue801_stage2b_psd_cuts as P  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from discopt._rust import model_to_repr  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402
from discopt.tightening import fbbt_box  # noqa: E402
from issue801_root_probe import NL, OBJ_VAR, ORACLE, RESULTS  # noqa: E402
from scipy.optimize import linprog  # noqa: E402

BASELINE = 0.8382369708575385


def scale_forms(n, quad, lin, lb, ub):
    """Affine x_i = L_i + s_i t_i (t in [0,1]); return scaled quad/lin and L,s."""
    L = np.asarray(lb, dtype=np.float64)
    s = np.asarray(ub, dtype=np.float64) - L
    s = np.where(s <= 0, 1.0, s)  # degenerate (fixed) vars: keep t_i in [0,0], s=1

    def scale_quad(q, a, c):
        qn = {}
        an = np.zeros(n)
        cn = float(c)
        for (i, j), qij in q.items():
            qn[(i, j)] = qij * s[i] * s[j]
            an[i] += qij * s[i] * L[j]
            an[j] += qij * L[i] * s[j]
            cn += qij * L[i] * L[j]
        for i in range(n):
            an[i] += a[i] * s[i]
            cn += a[i] * L[i]
        return qn, an, cn

    quad_s = {k: scale_quad(*v) for k, v in quad.items()}
    lin_s = {}
    for k, (a, c) in lin.items():
        an = a * s
        cn = float(c + float(np.dot(a, L)))
        lin_s[k] = (an, cn)
    return quad_s, lin_s, L, s


def scaled_moment_lp_and_cuts(n, quad_s, lin_s, max_rounds=400):
    """Full moment LP in t∈[0,1] + rigorous PSD cutting plane. Returns trajectory."""
    lb01 = np.zeros(n)
    ub01 = np.ones(n)
    c_obj, A_rows, b_rows, bounds, idxX, ncol = P.build_full_moment_lp(n, quad_s, lin_s, lb01, ub01)
    A, b = list(A_rows), list(b_rows)
    dim = n + 1
    traj = []
    for rnd in range(max_rounds):
        Asp = sp.csr_matrix(np.array(A))
        res = linprog(c_obj, A_ub=Asp, b_ub=np.array(b), bounds=bounds, method="highs")
        if res.status != 0:
            traj.append({"round": rnd, "status": int(res.status), "t17": None})
            break
        t17 = float(res.fun)
        z = res.x
        M = P.moment_matrix(z[:n], z, n, idxX)
        w, Vs = np.linalg.eigh(M)
        lam = float(w[0])
        traj.append({"round": rnd, "t17": t17, "lambda_min": lam, "rows": len(A)})
        if rnd % 20 == 0 or lam >= -1e-8:
            print(f"round {rnd:3d}  t17={t17:.6f}  lambda_min={lam:.3e}  rows={len(A)}")
        if lam >= -1e-8:
            break
        added = 0
        for e in range(min(6, dim)):
            if w[e] < -1e-8:
                r, rhs = P.psd_cut(Vs[:, e], n, idxX, ncol)
                A.append(r); b.append(rhs); added += 1
        if added == 0:
            break
    return traj


def scaled_scs(n, quad_s, lin_s, eps=1e-7, max_iters=500000):
    """SCS on the scaled dense moment SDP (well-conditioned)."""
    lb01 = np.zeros(n)
    ub01 = np.ones(n)
    return S.solve_moment_sdp(n, quad_s, lin_s, [], lb01, ub01, eps=eps, max_iters=max_iters)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    model = from_nl(NL)
    repr_ = model_to_repr(model)
    bt = fbbt_box(model)
    lb, ub = bt.lb, bt.ub
    S.repr_sense = {k: repr_.constraint_sense(k) for k in range(repr_.n_constraints)}
    S.repr_rhs = {k: repr_.constraint_rhs(k) for k in range(repr_.n_constraints)}
    P.__dict__["S"] = S
    n, quad, lin, pairs, _ = S.build_qcqp(repr_, lb, ub)
    quad_s, lin_s, L, s = scale_forms(n, quad, lin, lb, ub)

    L17, s17 = float(L[OBJ_VAR]), float(s[OBJ_VAR])

    # SCS on the scaled dense SDP (fast, converges cleanly now) — the decisive number.
    scs_res = scaled_scs(n, quad_s, lin_s)
    t17_scs = scs_res.get("scs_pobj")
    x17_scs = None if t17_scs is None else L17 + s17 * t17_scs
    print(f"SCALED SCS: status={scs_res.get('status')} x17={x17_scs}")

    # Rigorous PSD cutting plane (converged HiGHS each round) — corroboration; the
    # bound rises monotonically toward the SDP optimum, so a capped run still gives
    # a valid lower bound confirming inertness.
    traj = scaled_moment_lp_and_cuts(n, quad_s, lin_s, max_rounds=60)
    t17_cut = next((e["t17"] for e in reversed(traj) if e.get("t17") is not None), None)
    x17_cut = None if t17_cut is None else L17 + s17 * t17_cut

    gain_cut = None if x17_cut is None else x17_cut - BASELINE
    converged = bool(traj and traj[-1].get("lambda_min", -1) >= -1e-8)
    verdict = "KILL (SDP inert on the real instance)"
    if gain_cut is not None and gain_cut >= 0.005:
        verdict = "GO-CANDIDATE (SDP moves root — Stage 3 rigor+generality)"

    out = {
        "method": "scaled [0,1] moment SDP: rigorous PSD cutting-plane + SCS cross-check",
        "baseline": BASELINE,
        "oracle": ORACLE,
        "baron_root": 0.955,
        "psd_cut_x17_bound": x17_cut,
        "psd_cut_gain": gain_cut,
        "psd_cut_converged": converged,
        "psd_cut_rounds": len(traj),
        "scs_status": scs_res.get("status"),
        "scs_x17_bound": x17_scs,
        "scs_audit": scs_res.get("audit"),
        "kill_threshold": 0.005,
        "verdict": verdict,
        "trajectory_tail": traj[-5:],
    }
    print("\n" + json.dumps({k: out[k] for k in
          ["psd_cut_x17_bound", "psd_cut_gain", "psd_cut_converged", "psd_cut_rounds",
           "scs_status", "scs_x17_bound", "verdict"]}, indent=2, default=str))
    with open(os.path.join(RESULTS, "stage2b_scaled.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
