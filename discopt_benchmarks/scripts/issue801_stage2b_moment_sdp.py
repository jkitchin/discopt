"""#801 Stage 2b — dense/block moment (Shor) SDP on the tanksize continuous core.

Entry experiment (CLAUDE.md §4). The #764 candidate 3, never run: the unmodeled
tightness is the *joint* constraint among products sharing a variable; a moment
relaxation over the shared-variable stars may capture second-order info that
per-term McCormick and diagonal Shor (0.840, near-inert) miss.

We build the **self-contained order-1 moment (Shor) relaxation** of tanksize's own
QCQP structure (the AVM engine buries the 11 continuous-core products in opaque
intermediate columns, so we cannot reuse its lift). The dense moment matrix over
all 47 vars *dominates* any block/star version (block PSD ⊆ dense PSD), so if the
dense Shor is inert, every block version is too.

Construction:
  * extract every quadratic constraint's exact (Q, a, c) by finite differences;
  * lift each product pair (i,j) to X_ij with the 4 McCormick envelope rows;
  * rewrite each quadratic row linearly in (x, X); keep all linear rows;
  * drop the single sqrt equality (con 18) — x15 stays free in its FBBT box, a
    valid relaxation (validated: the McCormick-only LP still reproduces 0.8382);
  * add PSD on M = [[1, x'], [x, X]] (or its clique sub-blocks), solved by SCS.

FAITHFULNESS GATE: the McCormick-only QCQP-LP (no PSD) must reproduce 0.8382, else
the extraction is wrong — STOP. The SDP bound is then compared to 0.8382.

Kill: root gain < +0.005 over 0.8382 → candidate 3 dead (SDP route closed).

Run: ``python discopt_benchmarks/scripts/issue801_stage2b_moment_sdp.py``
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from discopt._rust import model_to_repr  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402
from discopt.tightening import fbbt_box  # noqa: E402
from issue801_root_probe import NL, OBJ_VAR, RESULTS  # noqa: E402
from scipy.optimize import linprog  # noqa: E402

BASELINE = 0.8382369708575385
SQRT_CON = 18  # the only non-quadratic (sqrt) constraint


def extract_quadratic(repr_, k, n):
    """Exact (q{(i,j):coef, i<j}, a[n], c) for a quadratic constraint via FD."""
    z = np.zeros(n)
    c = repr_.evaluate_constraint(k, z)
    ge, gme = np.zeros(n), np.zeros(n)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = 1.0
        ge[i] = repr_.evaluate_constraint(k, ei)
        ei[i] = -1.0
        gme[i] = repr_.evaluate_constraint(k, ei)
    a = (ge - gme) / 2.0
    qii = (ge + gme - 2.0 * c) / 2.0  # should be ~0 (no squares)
    q = {}
    max_qii = float(np.max(np.abs(qii)))
    # Scan all i<j pairs. NB: a product-only variable (appears solely in bilinear
    # terms, no linear part) has g(e_i)==c and a_i==0 — it is NOT absent, so we must
    # not skip on first-order signature. n=47 → ~1081 pairs/constraint, cheap.
    for i in range(n):
        for j in range(i + 1, n):
            eij = np.zeros(n)
            eij[i] = 1.0
            eij[j] = 1.0
            g = repr_.evaluate_constraint(k, eij)
            qij = g - ge[i] - ge[j] + c
            if abs(qij) > 1e-12:
                q[(i, j)] = float(qij)
    return q, a, float(c), max_qii


def build_qcqp(repr_, lb, ub):
    n = repr_.n_vars
    quad = {}
    pairs = set()
    max_qii_all = 0.0
    for k in range(repr_.n_constraints):
        if repr_.is_constraint_linear(k) or k == SQRT_CON:
            continue
        q, a, c, mqii = extract_quadratic(repr_, k, n)
        quad[k] = (q, a, c)
        pairs.update(q.keys())
        max_qii_all = max(max_qii_all, mqii)
    lin = {}
    for k in range(repr_.n_constraints):
        if not repr_.is_constraint_linear(k):
            continue
        z = np.zeros(n)
        c = repr_.evaluate_constraint(k, z)
        a = np.zeros(n)
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            a[i] = repr_.evaluate_constraint(k, ei) - c
        lin[k] = (a, float(c))
    return n, quad, lin, sorted(pairs), max_qii_all


def _mccormick_rows(i, j, li, lj, ui, uj, col_x, col_X, ncol):
    """4 McCormick rows for X_ij over [li,ui]x[lj,uj], returned as (row, rhs) <=."""
    rows = []
    # X >= li*xj + lj*xi - li*lj  ->  -X + li*xj + lj*xi <= li*lj
    r = np.zeros(ncol); r[col_X] = -1.0; r[col_x[j]] += li; r[col_x[i]] += lj
    rows.append((r, li * lj))
    # X >= ui*xj + uj*xi - ui*uj
    r = np.zeros(ncol); r[col_X] = -1.0; r[col_x[j]] += ui; r[col_x[i]] += uj
    rows.append((r, ui * uj))
    # X <= ui*xj + lj*xi - ui*lj  ->  X - ui*xj - lj*xi <= -ui*lj
    r = np.zeros(ncol); r[col_X] = 1.0; r[col_x[j]] -= ui; r[col_x[i]] -= lj
    rows.append((r, -ui * lj))
    # X <= li*xj + uj*xi - li*uj
    r = np.zeros(ncol); r[col_X] = 1.0; r[col_x[j]] -= li; r[col_x[i]] -= uj
    rows.append((r, -li * uj))
    return rows


def assemble_lp(n, quad, lin, pairs, lb, ub):
    """McCormick QCQP-LP: columns [x(0..n-1) | X_ij for pairs]. Returns c,A_ub,b_ub,bnds,colX."""
    col_x = list(range(n))
    colX = {p: n + t for t, p in enumerate(pairs)}
    ncol = n + len(pairs)
    A_rows, b_rows = [], []

    def add(row, rhs, sense):
        if sense == "<=":
            A_rows.append(row); b_rows.append(rhs)
        elif sense == ">=":
            A_rows.append(-row); b_rows.append(-rhs)
        else:  # ==
            A_rows.append(row); b_rows.append(rhs)
            A_rows.append(-row); b_rows.append(-rhs)

    # McCormick envelope rows for every lifted product.
    for (i, j) in pairs:
        for r, rhs in _mccormick_rows(i, j, lb[i], lb[j], ub[i], ub[j], col_x, colX[(i, j)], ncol):
            A_rows.append(r); b_rows.append(rhs)

    # Quadratic constraints -> linear in (x, X).
    for k, (q, a, c) in quad.items():
        row = np.zeros(ncol)
        for (i, j), qij in q.items():
            row[colX[(i, j)]] += qij
        row[: n] += a
        sense = str(repr_sense[k])
        rhs = float(repr_rhs[k]) - c
        add(row, rhs, sense)

    # Linear constraints.
    for k, (a, c) in lin.items():
        row = np.zeros(ncol)
        row[: n] += a
        sense = str(repr_sense[k])
        rhs = float(repr_rhs[k]) - c
        add(row, rhs, sense)

    c_obj = np.zeros(ncol)
    c_obj[OBJ_VAR] = 1.0
    bounds = [(float(lb[i]), float(ub[i])) for i in range(n)]
    # X_ij bounds from the box product range (McCormick rows are the real constraint)
    for (i, j) in pairs:
        vals = [lb[i] * lb[j], lb[i] * ub[j], ub[i] * lb[j], ub[i] * ub[j]]
        bounds.append((min(vals), max(vals)))
    A = sp.csr_matrix(np.array(A_rows)) if A_rows else None
    b = np.array(b_rows)
    return c_obj, A, b, bounds, colX, ncol


# module-level handles filled in main (used by assemble_lp)
repr_sense = {}
repr_rhs = {}

_SQRT2 = float(np.sqrt(2.0))


def _svec_index(d, i, j):
    c, r = (i, j) if i <= j else (j, i)
    return c * d - (c * (c - 1)) // 2 + (r - c)


class _Accum:
    """Accumulate rows `sum w*M_ij (<= or =) rhs` in SCS svec coordinates."""

    def __init__(self, d):
        self.d = d
        self.rows, self.cols, self.data, self.rhs = [], [], [], []

    def add(self, entries, rhs, scale=1.0):
        r = len(self.rhs)
        for i, j, w in entries:
            w = w * scale
            self.rows.append(r)
            self.cols.append(_svec_index(self.d, i, j))
            self.data.append(w if i == j else w / _SQRT2)
        self.rhs.append(rhs * scale)

    def build(self, m):
        A = sp.csr_matrix((self.data, (self.rows, self.cols)), shape=(len(self.rhs), m))
        A.sort_indices()
        return A, np.asarray(self.rhs, dtype=np.float64)


def solve_moment_sdp(n, quad, lin, pairs, lb, ub, clique_blocks=None, eps=1e-6, max_iters=200000):
    """Dense (or block) order-1 moment (Shor) SDP over M=[[1,x'],[x,X]], min x17.

    Reuses the problem's exact quadratic forms; adds McCormick box on all moment
    entries + PSD on the full matrix (dense) or on clique sub-blocks (block).
    Returns the SCS primal objective (the SDP relaxation bound on x17).
    """
    import scs

    dim = n + 1
    m = dim * (dim + 1) // 2
    V = lambda i: 1 + i  # noqa: E731  moment index of variable i

    eq, ineq = _Accum(dim), _Accum(dim)
    # M_00 = 1
    eq.add([(0, 0, 1.0)], 1.0)
    # x box: lb_i <= M[0,V(i)] <= ub_i
    for i in range(n):
        ineq.add([(0, V(i), -1.0)], -float(lb[i]))  # -x_i <= -lb_i
        ineq.add([(0, V(i), 1.0)], float(ub[i]))  # x_i <= ub_i
    # McCormick on diagonal squares X_ii = x_i^2 (convex): 2 tangents (lower) + secant (upper)
    for i in range(n):
        li, ui = float(lb[i]), float(ub[i])
        # X_ii >= 2 li x_i - li^2  -> -X_ii + 2li x_i <= li^2
        ineq.add([(V(i), V(i), -1.0), (0, V(i), 2 * li)], li * li)
        ineq.add([(V(i), V(i), -1.0), (0, V(i), 2 * ui)], ui * ui)
        # X_ii <= (li+ui) x_i - li ui
        ineq.add([(V(i), V(i), 1.0), (0, V(i), -(li + ui))], -li * ui)
    # McCormick on all cross pairs i<j (dense: every pair; block: only within a clique)
    if clique_blocks is None:
        cross = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        cross = sorted({(min(a, b), max(a, b)) for blk in clique_blocks for a in blk for b in blk if a < b})
    for (i, j) in cross:
        li, lj, ui, uj = float(lb[i]), float(lb[j]), float(ub[i]), float(ub[j])
        # X_ij >= li xj + lj xi - li lj
        ineq.add([(V(i), V(j), -1.0), (0, V(j), li), (0, V(i), lj)], li * lj)
        ineq.add([(V(i), V(j), -1.0), (0, V(j), ui), (0, V(i), uj)], ui * uj)
        # X_ij <= ui xj + lj xi - ui lj
        ineq.add([(V(i), V(j), 1.0), (0, V(j), -ui), (0, V(i), -lj)], -ui * lj)
        ineq.add([(V(i), V(j), 1.0), (0, V(j), -li), (0, V(i), -uj)], -li * uj)
    # Quadratic constraints (X_ij = M[V(i),V(j)], x_i = M[0,V(i)])
    for k, (q, a, c) in quad.items():
        ent = [(V(i), V(j), qij) for (i, j), qij in q.items()]
        ent += [(0, V(i), float(a[i])) for i in range(n) if abs(a[i]) > 0]
        rhs = float(repr_rhs[k]) - c
        s = str(repr_sense[k])
        if s == "==":
            eq.add(ent, rhs)
        elif s == "<=":
            ineq.add(ent, rhs)
        else:  # >=
            ineq.add(ent, rhs, scale=-1.0)
    # Linear constraints
    for k, (a, c) in lin.items():
        ent = [(0, V(i), float(a[i])) for i in range(n) if abs(a[i]) > 0]
        rhs = float(repr_rhs[k]) - c
        s = str(repr_sense[k])
        if s == "==":
            eq.add(ent, rhs)
        elif s == "<=":
            ineq.add(ent, rhs)
        else:
            ineq.add(ent, rhs, scale=-1.0)

    A_eq, b_eq = eq.build(m)
    A_in, h = ineq.build(m)
    c_svec = np.zeros(m)
    c_svec[_svec_index(dim, 0, V(OBJ_VAR))] = 1.0 / _SQRT2  # objective = x17 = M[0,V(17)] (off-diag)

    A = sp.vstack([A_eq, A_in, -sp.identity(m, format="csr")], format="csc")
    b = np.concatenate([b_eq, h, np.zeros(m)])
    cone = {"z": int(A_eq.shape[0]), "l": int(A_in.shape[0]), "s": [dim]}
    settings = {"verbose": False, "eps_abs": eps, "eps_rel": eps, "max_iters": max_iters}
    sol = scs.SCS({"A": A, "b": b, "c": c_svec}, cone, **settings).solve()
    info = sol.get("info", {})
    status = str(info.get("status", "")).lower()
    pobj = info.get("pobj")
    u = sol.get("x")
    x17 = None
    audit = {}
    if u is not None:
        u = np.asarray(u, dtype=np.float64)
        x17 = float(u[_svec_index(dim, 0, V(OBJ_VAR))] / _SQRT2)
        # Feasibility audit of the returned primal moment matrix.
        M = np.zeros((dim, dim))
        for cc in range(dim):
            for rr in range(cc, dim):
                val = float(u[_svec_index(dim, rr, cc)])
                M[rr, cc] = M[cc, rr] = val if rr == cc else val / _SQRT2
        lam_min = float(np.linalg.eigvalsh(M)[0])
        eq_res = float(np.max(np.abs(A_eq @ u - b_eq))) if A_eq.shape[0] else 0.0
        in_res = float(np.max((A_in @ u - h).clip(min=0.0))) if A_in.shape[0] else 0.0
        audit = {
            "M_lambda_min": lam_min,
            "M00": float(M[0, 0]),
            "max_eq_residual": eq_res,
            "max_ineq_violation": in_res,
        }
    return {
        "status": status,
        "scs_pobj": None if pobj is None else float(pobj),
        "x17_from_primal": x17,
        "dim": dim,
        "n_eq": int(A_eq.shape[0]),
        "n_ineq": int(A_in.shape[0]),
        "audit": audit,
    }


def main():
    os.makedirs(RESULTS, exist_ok=True)
    model = from_nl(NL)
    repr_ = model_to_repr(model)
    bt = fbbt_box(model)
    lb, ub = bt.lb, bt.ub
    n = repr_.n_vars
    global repr_sense, repr_rhs
    repr_sense = {k: repr_.constraint_sense(k) for k in range(repr_.n_constraints)}
    repr_rhs = {k: repr_.constraint_rhs(k) for k in range(repr_.n_constraints)}

    n, quad, lin, pairs, max_qii = build_qcqp(repr_, lb, ub)
    print(f"quad cons={len(quad)} lin cons={len(lin)} product pairs={len(pairs)} max|q_ii|={max_qii:.2e}")

    c_obj, A, b, bounds, colX, ncol = assemble_lp(n, quad, lin, pairs, lb, ub)
    res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    mc_lp = float(res.fun) if res.fun is not None else None
    print(f"McCormick QCQP-LP bound = {mc_lp} (status {res.status})")
    faithful = mc_lp is not None and abs(mc_lp - BASELINE) < 5e-3
    out = {
        "n_quad_cons": len(quad),
        "n_lin_cons": len(lin),
        "n_product_pairs": len(pairs),
        "max_abs_q_ii": max_qii,
        "mccormick_qcqp_lp": mc_lp,
        "baseline": BASELINE,
        "faithful": bool(faithful),
    }
    print(json.dumps(out, indent=2))
    with open(os.path.join(RESULTS, "stage2b_faithfulness.json"), "w") as f:
        json.dump(out, f, indent=2)
    if not faithful:
        raise SystemExit(
            f"STOP: McCormick QCQP-LP {mc_lp} != baseline {BASELINE}; extraction unfaithful "
            "(likely the dropped sqrt con 18 matters). Add the sqrt relaxation before the SDP."
        )
    print("\nFaithfulness gate PASSED — extraction reproduces 0.8382. Solving moment SDP...")

    # Dense order-1 moment (Shor) SDP — dominates every block/star version.
    sdp = solve_moment_sdp(n, quad, lin, pairs, lb, ub)
    sdp_bound = sdp["scs_pobj"]
    print(f"Dense moment SDP: status={sdp['status']} pobj={sdp_bound} x17={sdp['x17_from_primal']} "
          f"(dim={sdp['dim']}, eq={sdp['n_eq']}, ineq={sdp['n_ineq']})")

    gain = None if sdp_bound is None else sdp_bound - BASELINE
    verdict = "KILL"
    if gain is not None and gain >= 0.005:
        verdict = "PROMISING (compute rigorous safe bound before any GO)"
    out.update({
        "mccormick_qcqp_lp": mc_lp,
        "dense_moment_sdp_pobj": sdp_bound,
        "dense_moment_sdp_status": sdp["status"],
        "sdp_gain_over_baseline": gain,
        "kill_threshold": 0.005,
        "verdict": verdict,
    })
    print(json.dumps({k: out[k] for k in
                      ["mccormick_qcqp_lp", "dense_moment_sdp_pobj", "sdp_gain_over_baseline", "verdict"]},
                     indent=2))
    with open(os.path.join(RESULTS, "stage2b_moment_sdp.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
