"""#801 Stage 2a (converged confirmation) — order-2 star moment SDP via SCS.

The Stage-2a cutting plane gives a rigorous *lower* bound on the order-2 star
relaxation (frozen at 0.8401). To match the rigor of Stage 2b (a converged
upper-side solve), this assembles the order-2 relaxation directly for SCS:

  * one PSD block = the base moment matrix M=[[1,x'],[x,X]] (dim n+1), carrying
    all degree-<=2 moments + the model's McCormick/quadratic/linear rows;
  * one PSD block per shared-variable star = its order-2 moment matrix over the
    degree-<=2 monomial basis (dim 10-15), carrying degree-3/4 moments;
  * linking equalities tie each star block's degree-<=2 entries to the base
    block and enforce intra-block moment consistency. The stars are
    variable-disjoint, so there is no cross-star coupling.

Scaled to [0,1] so SCS converges. If SCS converges with x17 ~ 0.8401, the order-2
(trilinear/higher-order) route is confirmed inert. Kill: gain < +0.005 / 0.8382.

Run: ``python discopt_benchmarks/scripts/issue801_stage2a_scs.py``
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import issue801_stage2b_moment_sdp as S  # noqa: E402
import issue801_stage2b_scaled as SC  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from discopt._rust import model_to_repr  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402
from discopt.tightening import fbbt_box  # noqa: E402
from issue801_root_probe import NL, OBJ_VAR, ORACLE, RESULTS  # noqa: E402
from issue801_stage2a_order2 import STARS, mono, star_basis  # noqa: E402

BASELINE = 0.8382369708575385
_SQRT2 = float(np.sqrt(2.0))


def _svec(d, i, j):
    c, r = (i, j) if i <= j else (j, i)
    return c * d - (c * (c - 1)) // 2 + (r - c)


def solve_order2_scs(n, quad_s, lin_s, stars, eps=1e-7, max_iters=1_000_000):
    import scs

    # Blocks: base (dim n+1), then one per star.
    base_dim = n + 1
    blocks = [("base", None, base_dim)]
    sbases = {}
    for st in stars:
        bas = star_basis(st)
        sbases[st] = bas
        blocks.append(("star", st, len(bas)))
    offs, off = [], 0
    for (_, _, d) in blocks:
        offs.append(off)
        off += d * (d + 1) // 2
    m = off

    def term(bi, i, j, w):
        d = blocks[bi][2]
        gc = offs[bi] + _svec(d, i, j)
        return gc, (w if i == j else w / _SQRT2)

    eq_r, eq_c, eq_d, eq_b = [], [], [], []
    in_r, in_c, in_d, in_b = [], [], [], []

    def push(rows, cols, data, rhsl, entries, rhs, scale=1.0):
        r = len(rhsl)
        for (gc, coef) in entries:
            rows.append(r)
            cols.append(gc)
            data.append(coef * scale)
        rhsl.append(rhs * scale)

    def eq(entries, rhs):
        push(eq_r, eq_c, eq_d, eq_b, entries, rhs)

    def le(entries, rhs, scale=1.0):
        push(in_r, in_c, in_d, in_b, entries, rhs, scale)

    Vb = lambda i: 1 + i  # noqa: E731

    # base M00 = 1
    eq([term(0, 0, 0, 1.0)], 1.0)
    # x box (scaled [0,1]) on base
    for i in range(n):
        le([term(0, 0, Vb(i), -1.0)], 0.0)  # -x_i <= 0
        le([term(0, 0, Vb(i), 1.0)], 1.0)  # x_i <= 1
    # McCormick on base squares X_ii = x_i^2 over [0,1]
    for i in range(n):
        le([term(0, Vb(i), Vb(i), -1.0), term(0, 0, Vb(i), 0.0)], 0.0)  # X_ii>=0 (l=0)
        le([term(0, Vb(i), Vb(i), -1.0), term(0, 0, Vb(i), 2.0)], 1.0)  # X_ii>=2x-1
        le([term(0, Vb(i), Vb(i), 1.0), term(0, 0, Vb(i), -1.0)], 0.0)  # X_ii<=x
    # McCormick on base cross X_ij over [0,1]
    for i in range(n):
        for j in range(i + 1, n):
            le([term(0, Vb(i), Vb(j), -1.0)], 0.0)  # X>=0
            le([term(0, Vb(i), Vb(j), -1.0), term(0, 0, Vb(j), 1.0), term(0, 0, Vb(i), 1.0)], 1.0)  # X>=x_i+x_j-1
            le([term(0, Vb(i), Vb(j), 1.0), term(0, 0, Vb(j), -1.0)], 0.0)  # X<=x_j
            le([term(0, Vb(i), Vb(j), 1.0), term(0, 0, Vb(i), -1.0)], 0.0)  # X<=x_i
    # quadratic constraints on base
    for k, (q, a, c) in quad_s.items():
        ent = [term(0, Vb(i), Vb(j), qij) for (i, j), qij in q.items()]
        ent += [term(0, 0, Vb(i), float(a[i])) for i in range(n) if abs(a[i]) > 0]
        rhs = float(S.repr_rhs[k]) - c
        s = str(S.repr_sense[k])
        if s == "==":
            eq(ent, rhs)
        elif s == "<=":
            le(ent, rhs)
        else:
            le(ent, rhs, -1.0)
    # linear constraints on base
    for k, (a, c) in lin_s.items():
        ent = [term(0, 0, Vb(i), float(a[i])) for i in range(n) if abs(a[i]) > 0]
        rhs = float(S.repr_rhs[k]) - c
        s = str(S.repr_sense[k])
        if s == "==":
            eq(ent, rhs)
        elif s == "<=":
            le(ent, rhs)
        else:
            le(ent, rhs, -1.0)

    # star linking
    for bi, (kind, st, d) in enumerate(blocks):
        if kind != "star":
            continue
        bas = sbases[st]
        first = {}
        for a in range(len(bas)):
            for b in range(a, len(bas)):
                mu = mono(bas[a] + bas[b])
                if mu not in first:
                    first[mu] = (a, b)
                    # link representative to base for deg<=2
                    if len(mu) == 0:
                        eq([term(bi, a, b, 1.0)], 1.0)
                    elif len(mu) == 1:
                        eq([term(bi, a, b, 1.0), term(0, 0, Vb(mu[0]), -1.0)], 0.0)
                    elif len(mu) == 2:
                        eq([term(bi, a, b, 1.0), term(0, Vb(mu[0]), Vb(mu[1]), -1.0)], 0.0)
                    # deg>=3: representative is a free moment (bounded by PSD + [0,1])
                    if len(mu) >= 3:
                        le([term(bi, a, b, 1.0)], 1.0)
                        le([term(bi, a, b, -1.0)], 0.0)
                else:
                    a0, b0 = first[mu]
                    eq([term(bi, a, b, 1.0), term(bi, a0, b0, -1.0)], 0.0)

    c_svec = np.zeros(m)
    gc, coef = term(0, 0, Vb(OBJ_VAR), 1.0)
    c_svec[gc] = coef

    A_eq = sp.csr_matrix((eq_d, (eq_r, eq_c)), shape=(len(eq_b), m))
    A_in = sp.csr_matrix((in_d, (in_r, in_c)), shape=(len(in_b), m))
    A = sp.vstack([A_eq, A_in, -sp.identity(m, format="csr")], format="csc")
    b = np.concatenate([np.array(eq_b), np.array(in_b), np.zeros(m)])
    cone = {"z": len(eq_b), "l": len(in_b), "s": [d for (_, _, d) in blocks]}
    settings = {"verbose": False, "eps_abs": eps, "eps_rel": eps, "max_iters": max_iters}
    sol = scs.SCS({"A": A, "b": b, "c": c_svec}, cone, **settings).solve()
    info = sol.get("info", {})
    u = sol.get("x")
    t17 = None
    if u is not None:
        t17 = float(np.asarray(u)[gc] / coef)
    return {"status": str(info.get("status", "")).lower(), "pobj": info.get("pobj"),
            "t17": t17, "n_eq": len(eq_b), "n_in": len(in_b), "cones": cone["s"]}


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

    res = solve_order2_scs(n, quad_s, lin_s, STARS)
    t17 = res["pobj"]
    x17 = None if t17 is None else L17 + s17 * float(t17)
    gain = None if x17 is None else x17 - BASELINE
    verdict = "KILL (order-2 star moment inert; trilinear/higher-order route closed)"
    if gain is not None and gain >= 0.005:
        verdict = "GO-CANDIDATE (order-2 moves root — Stage 3)"
    out = {
        "method": "order-2 star moment SDP via SCS (base 48x48 PSD + per-star order-2 PSD blocks)",
        "stars": [list(st) for st in STARS],
        "baseline": BASELINE,
        "order1_moment_sdp": 0.8401,
        "scs_status": res["status"],
        "order2_scs_x17": x17,
        "gain_over_baseline": gain,
        "cones": res["cones"],
        "n_eq": res["n_eq"],
        "n_in": res["n_in"],
        "oracle": ORACLE,
        "baron_root": 0.955,
        "kill_threshold": 0.005,
        "verdict": verdict,
    }
    print(json.dumps(out, indent=2, default=str))
    with open(os.path.join(RESULTS, "stage2a_scs.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
