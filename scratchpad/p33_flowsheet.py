"""P3.3 — the flowsheet entry experiment (the P3 go/no-go).

MAiNGO-parity plan §5-P3.3: build a small flowsheet-style model as sequential
``CustomCall`` unit functions (hidden internal intermediates) with a few degrees of
freedom + an integer choice, and solve it two ways with the SAME bounding engine and
the SAME spatial-B&B driver:

  (a) REDUCED    — CustomCall units; the internal intermediates are hidden, so B&B
                   branches ONLY on the true DOF (the reduced-space McCormick relaxation
                   is traced through MCBox — P3.1).
  (b) FLATTENED  — the honest full-space formulation of the exact same model: every
                   intermediate (the exp factors e_i and the carryover flows c_i) is an
                   explicit variable with a defining equality, so B&B branches on the
                   full lifted variable set.

Only the variable set differs; the relaxation engine (``reduced_mccormick_lp_bound``,
i.e. Kelley cuts on the MCBox relaxation solved as an LP) and the driver are identical.
The node-count ratio therefore isolates the DOF-tree-size effect — exactly the P3 claim.

GATE (plan §5-P3.3): reduced tree dramatically smaller (MAiNGO regime ≥5× fewer nodes)
AND both certify the same optimum.  KILL: if the reduced tree is not smaller even on
this favorable case, STOP P3 (the CustomCall-global capability may still ship for
expressiveness, but the DOF-tree-size claim is dead).

Run:  PYTHONPATH=python DISCOPT_REDUCED_LP_BACKEND=scipy python scratchpad/p33_flowsheet.py
"""

from __future__ import annotations

import itertools
from heapq import heappop, heappush

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import discopt.modeling as dm  # noqa: E402
from discopt._jax.mcbox import MCBox  # noqa: E402
from discopt._jax.mccormick_subgradient import reduced_mccormick_lp_bound  # noqa: E402

# ------------------------------------------------------------------ flowsheet data
A = [0.8, 0.6, 0.5]  # per-unit rate coefficients (k_i = a_i * T_i)
W = [0.15, 0.15, 0.15]  # per-unit energy weights
DMIN = 0.55  # minimum total production (constraint)
TLB, TUB = 0.2, 2.0  # reactor-temperature DOF box
ZLB, ZUB = 0, 2  # discrete feed-level choice: F0 = 1 + 0.5*z


def _mexp(x):
    """exp that works on floats/jax arrays AND on MCBox (dispatch)."""
    return x.exp() if isinstance(x, MCBox) else jnp.exp(x)


def unit(c_in, T, a):
    """One reactor stage: carryover flow out = c_in * exp(-a*T).

    Internal (hidden) intermediates in the CustomCall body: k = a*T, e = exp(-k).
    Returns the scalar carryover flow c_out."""
    return c_in * _mexp(-a * T)


# ------------------------------------------------------------------ true objective
def true_eval(T, z):
    """Objective of the true (un-relaxed) model at DOF (T[3], z). numpy scalars."""
    F0 = 1.0 + 0.5 * z
    c0 = F0
    c1 = unit(c0, T[0], A[0])
    c2 = unit(c1, T[1], A[1])
    c3 = unit(c2, T[2], A[2])
    reacted = F0 - c3
    cs = [c0, c1, c2, c3]
    energy = sum(W[i] * T[i] * (cs[i] - cs[i + 1]) for i in range(3))
    return float(-reacted + energy), float(reacted)


# ------------------------------------------------------------------ models
def build_reduced():
    """CustomCall flowsheet — DOF only (T1,T2,T3 continuous, z integer)."""
    m = dm.Model()
    T = m.continuous("T", 3, lb=[TLB] * 3, ub=[TUB] * 3)
    z = m.integer("z", 1, lb=[ZLB], ub=[ZUB])
    F0 = 1.0 + 0.5 * z[0]
    u = [dm.custom(lambda c, t, a=A[i]: unit(c, t, a), name=f"unit{i + 1}") for i in range(3)]
    c0 = F0
    c1 = u[0](c0, T[0])
    c2 = u[1](c1, T[1])
    c3 = u[2](c2, T[2])
    cs = [c0, c1, c2, c3]
    reacted = F0 - c3
    energy = sum(W[i] * T[i] * (cs[i] - cs[i + 1]) for i in range(3))
    m.minimize(-reacted + energy)
    m.subject_to(reacted >= DMIN)
    # branch set = all DOF; box order matches m's variable layout [T0,T1,T2,z]
    lb = [TLB, TLB, TLB, float(ZLB)]
    ub = [TUB, TUB, TUB, float(ZUB)]
    int_idx = {3}
    branch_idx = [0, 1, 2, 3]
    return m, lb, ub, int_idx, branch_idx


def _exp_bounds(a):
    # e = exp(-a*T), T in [TLB,TUB]; exp decreasing in T -> [exp(-a*TUB), exp(-a*TLB)]
    return float(np.exp(-a * TUB)), float(np.exp(-a * TLB))


def build_flattened():
    """Honest full-space formulation: e_i and c_i are explicit variables."""
    m = dm.Model()
    T = m.continuous("T", 3, lb=[TLB] * 3, ub=[TUB] * 3)
    z = m.integer("z", 1, lb=[ZLB], ub=[ZUB])
    elb = [_exp_bounds(A[i])[0] for i in range(3)]
    eub = [_exp_bounds(A[i])[1] for i in range(3)]
    e = m.continuous("e", 3, lb=elb, ub=eub)
    F0lb, F0ub = 1.0 + 0.5 * ZLB, 1.0 + 0.5 * ZUB
    # c_i interval propagation: c_i = c_{i-1} * e_i
    clb = [F0lb]
    cub = [F0ub]
    for i in range(3):
        clb.append(clb[-1] * elb[i])
        cub.append(cub[-1] * eub[i])
    c = m.continuous("c", 3, lb=clb[1:], ub=cub[1:])  # c1,c2,c3
    F0 = 1.0 + 0.5 * z[0]
    cs = [F0, c[0], c[1], c[2]]
    # defining equalities
    for i in range(3):
        m.subject_to(e[i] - dm.exp(-A[i] * T[i]) == 0)
    m.subject_to(c[0] - F0 * e[0] == 0)
    m.subject_to(c[1] - c[0] * e[1] == 0)
    m.subject_to(c[2] - c[1] * e[2] == 0)
    reacted = F0 - c[2]
    energy = sum(W[i] * T[i] * (cs[i] - cs[i + 1]) for i in range(3))
    m.minimize(-reacted + energy)
    m.subject_to(reacted >= DMIN)
    # variable layout: [T0,T1,T2, z, e0,e1,e2, c0,c1,c2]
    lb = [TLB, TLB, TLB, float(ZLB)] + elb + clb[1:]
    ub = [TUB, TUB, TUB, float(ZUB)] + eub + cub[1:]
    int_idx = {3}
    branch_idx = list(range(10))  # full-space: branch on the whole lifted set
    return m, lb, ub, int_idx, branch_idx


# ------------------------------------------------------------------ spatial B&B
XTOL = 1e-3  # continuous spatial box tolerance


MAX_ROUNDS = 25


def node_bound(model, nlb, nub):
    rb = reduced_mccormick_lp_bound(model, nlb, nub, max_rounds=MAX_ROUNDS, tol=1e-7)
    if rb.status == "infeasible":
        return np.inf
    if rb.status in ("unbounded", "unsupported"):
        return -np.inf
    return rb.bound


def solve_bb(model, lb, ub, int_idx, branch_idx, incumbent, tol, node_cap, tag=""):
    """Best-first spatial B&B; returns (certified_lb, nodes, certified?)."""
    import sys

    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    cnt = itertools.count()
    root = node_bound(model, lb, ub)
    heap = [(root, next(cnt), lb.copy(), ub.copy())]
    nodes = 0
    global_lb = root
    while heap:
        if nodes and nodes % 500 == 0:
            print(f"  [{tag}] nodes={nodes} open={len(heap)} lb={global_lb:.4f}", file=sys.stderr)
        b, _, nlb, nub = heappop(heap)
        global_lb = b  # best-first: smallest open bound is the global LB
        if b >= incumbent - tol:
            return b, nodes, True  # all open nodes >= UB - tol => certified
        nodes += 1
        if nodes > node_cap:
            return b, nodes, False
        # pick branch variable: widest eligible box
        best_i, best_w = -1, 0.0
        for i in branch_idx:
            w = nub[i] - nlb[i]
            if i in int_idx:
                if nub[i] - nlb[i] < 1.0 - 1e-9:  # integer already fixed
                    continue
            else:
                if w <= XTOL:
                    continue
            if w > best_w:
                best_w, best_i = w, i
        if best_i < 0:
            # cannot refine further; this leaf's bound stands as a valid LB for its box
            continue
        i = best_i
        if i in int_idx:
            mid = np.floor(0.5 * (nlb[i] + nub[i]))
            splits = [(nlb[i], mid), (mid + 1.0, nub[i])]
        else:
            mid = 0.5 * (nlb[i] + nub[i])
            splits = [(nlb[i], mid), (mid, nub[i])]
        for lo_i, hi_i in splits:
            if hi_i < lo_i - 1e-12:
                continue
            clb, cub = nlb.copy(), nub.copy()
            clb[i], cub[i] = lo_i, hi_i
            cb = node_bound(model, clb, cub)
            if cb >= incumbent - tol:
                continue  # child fathomed on bound
            heappush(heap, (cb, next(cnt), clb, cub))
    return global_lb, nodes, True  # queue emptied: everything fathomed


# ------------------------------------------------------------------ driver
def find_optimum():
    """Dense DOF scan + integer sweep for the true global optimum (feasible).

    Vectorized in pure numpy (no jax dispatch) so the 3-D grid scan is fast."""
    grid = np.linspace(TLB, TUB, 81)
    T0, T1, T2 = np.meshgrid(grid, grid, grid, indexing="ij")
    best = (np.inf, None)
    for z in range(ZLB, ZUB + 1):
        F0 = 1.0 + 0.5 * z
        c0 = F0
        c1 = c0 * np.exp(-A[0] * T0)
        c2 = c1 * np.exp(-A[1] * T1)
        c3 = c2 * np.exp(-A[2] * T2)
        reacted = F0 - c3
        energy = W[0] * T0 * (c0 - c1) + W[1] * T1 * (c1 - c2) + W[2] * T2 * (c2 - c3)
        obj = -reacted + energy
        obj = np.where(reacted >= DMIN - 1e-9, obj, np.inf)
        k = int(np.argmin(obj))
        if obj.flat[k] < best[0]:
            i0, i1, i2 = np.unravel_index(k, obj.shape)
            best = (float(obj.flat[k]), (float(T0[i0, i1, i2]),
                    float(T1[i0, i1, i2]), float(T2[i0, i1, i2]), z))
    return best


def main():
    opt, arg = find_optimum()
    print(f"true global optimum  obj = {opt:.6f}   at (T,z) = {arg}")
    tol = 1e-2 * (abs(opt) + 1.0)  # 1% optimality gap (identical for both runs)
    UB = opt + 1e-7  # shared incumbent (isolate relaxation/branching from heuristics)

    results = {}
    for name, builder in [("reduced", build_reduced), ("flattened", build_flattened)]:
        m, lb, ub, int_idx, branch_idx = builder()
        rootb = node_bound(m, lb, ub)
        clb, nodes, cert = solve_bb(
            m, lb, ub, int_idx, branch_idx, UB, tol, node_cap=50000, tag=name
        )
        results[name] = dict(
            nvars=len(lb), nbranch=len(branch_idx), root=rootb, lb=clb, nodes=nodes, cert=cert
        )
        print(
            f"[{name:9s}] vars={len(lb):2d} branch={len(branch_idx):2d}  "
            f"root_bound={rootb:.5f}  proved_lb={clb:.5f}  nodes={nodes:6d}  certified={cert}"
        )

    r, f = results["reduced"], results["flattened"]
    same_opt = abs(r["lb"] - f["lb"]) < 5 * tol and r["cert"] and f["cert"]
    ratio = f["nodes"] / max(r["nodes"], 1)
    print("\n================ P3.3 RESULT ================")
    print(f"reduced nodes   = {r['nodes']}")
    print(f"flattened nodes = {f['nodes']}")
    print(f"node ratio (flattened / reduced) = {ratio:.2f}x   (gate: >= 5x)")
    print(f"both certify same optimum? {same_opt}  "
          f"(reduced_lb={r['lb']:.5f}, flattened_lb={f['lb']:.5f}, opt={opt:.5f})")
    verdict = "GO" if (ratio >= 5.0 and same_opt) else (
        "KILL" if (r["nodes"] >= f["nodes"] and same_opt) else "PARTIAL")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
