#!/usr/bin/env python3
"""#2014 entry experiment: EXACT {0,1/2}-Chvatal-Gomory separation at discopt's root LP.

The parked heuristic zero-half separator (PR #427, GF(2) elimination) finds no cut
on sporttournament14, whose root vertex has every binary at 1/2. To tell "the
heuristic misses cuts" from "there are none", this script solves the separation
problem exactly (Caprara & Fischetti 1996) as a small MILP over the root
relaxation's rows: pick rows ``z_i`` and bound inequalities ``t_j`` so that every
integer column's combined coefficient is even, every continuous column's is zero,
the combined right-hand side is odd, and the total slack at the LP point is
minimal. A violated {0,1/2} cut exists iff that minimum is < 1 (violation
``(1 - min) / 2``). Each round adds the cut, re-solves the LP, and repeats.

The box is the one the solver hands the native-kernel producer (after presolve);
``--kernel-build`` builds the relaxation with the kernel's options (RLT rows on).
``--int-aux`` treats the aux column of a product of two integer originals as
integer, which is sound: it equals that product at every integer point.

    python -u discopt_benchmarks/scripts/issue2014_zerohalf_exact_entry.py \\
        <file.nl> [--rounds 40] [--int-aux] [--kernel-build]

Prints the bound per round and an EXECUTED count of separation MILPs; exits
non-zero when none ran (CLAUDE.md section 6).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp


class _StopError(Exception):
    pass


def _solver_box(model):
    """The (lb, ub) the solver passes the kernel producer; the declared box if none."""
    import discopt._relax.spatial_producer as spp
    from discopt._relax.model_utils import flat_variable_bounds

    real = spp.build_spatial_kernel_spec
    box: dict = {}

    def capture(m, bounds=None, deadline=None):
        box["lb"], box["ub"] = np.asarray(bounds[0]).copy(), np.asarray(bounds[1]).copy()
        raise _StopError

    spp.build_spatial_kernel_spec = capture
    try:
        model.solve(time_limit=30)
    except _StopError:
        pass
    finally:
        spp.build_spatial_kernel_spec = real
    if "lb" in box:
        return box["lb"], box["ub"], "solver"
    lb, ub = flat_variable_bounds(model)
    return lb, ub, "declared"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("nl")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--int-aux", action="store_true")
    ap.add_argument("--kernel-build", action="store_true")
    a = ap.parse_args()

    import discopt.modeling as dm
    from discopt._relax.uniform_relax import build_uniform_relaxation
    from discopt.modeling.core import VarType
    from discopt.solver_tuning import current as tuning

    lb, ub, src = _solver_box(dm.from_nl(a.nl))
    m = dm.from_nl(a.nl)
    kw = (
        dict(
            rlt_lineq=bool(tuning().rlt_lineq),
            rlt_ineq=bool(tuning().rlt_ineq),
            skip_separable_floor=True,
            skip_convex_lift=True,
        )
        if a.kernel_build
        else {}
    )
    rel = build_uniform_relaxation(m, box=(lb, ub), **kw)
    M = rel.model
    c = np.asarray(M._c, float)
    n = c.size
    A = sp.csr_matrix(M._A_ub).toarray()
    b = np.asarray(M._b_ub, float).ravel()
    # Equalities live in ``_A_ub`` as row pairs; read ``_A_eq`` only if present.
    a_eq = getattr(M, "_A_eq", None)
    has_eq = a_eq is not None and sp.csr_matrix(a_eq).shape[0] > 0
    Aeq = sp.csr_matrix(a_eq).toarray() if has_eq else np.zeros((0, n))
    beq = np.asarray(M._b_eq, float).ravel() if has_eq else np.zeros(0)
    bnds = np.asarray(M._bounds, float)
    integ = np.zeros(n, bool)
    off = 0
    for v in m._variables:
        if v.var_type in (VarType.BINARY, VarType.INTEGER):
            integ[off : off + int(v.size)] = True
        off += int(v.size)
    if a.int_aux:
        for (i, j), w in rel.bilinear_map.items():
            if integ[i] and integ[j]:
                integ[w] = True
    sgn = float(getattr(rel, "obj_sense_sign", 1.0))
    offc = float(getattr(rel, "obj_offset", 0.0))
    R = np.vstack([A, Aeq, -Aeq])
    rb = np.concatenate([b, beq, -beq])
    intdata = np.all(np.abs(R - np.round(R)) < 1e-9, axis=1) & (np.abs(rb - np.round(rb)) < 1e-9)
    ci = np.flatnonzero(integ)
    cc = np.flatnonzero(~integ)
    ni = len(ci)
    lo_ok = integ & np.isfinite(bnds[:, 0])
    hi_ok = integ & np.isfinite(bnds[:, 1])
    lo_ok &= (
        np.abs(np.where(lo_ok, bnds[:, 0], 0) - np.round(np.where(lo_ok, bnds[:, 0], 0))) < 1e-9
    )
    hi_ok &= (
        np.abs(np.where(hi_ok, bnds[:, 1], 0) - np.round(np.where(hi_ok, bnds[:, 1], 0))) < 1e-9
    )
    print(f"box from {src}; {R.shape[0]} rows, {n} cols, {ni} integer", flush=True)

    cuts_a: list = []
    cuts_b: list = []
    executed = 0
    root = None
    for rnd in range(a.rounds + 1):
        A2 = np.vstack([A, *cuts_a]) if cuts_a else A
        b2 = np.concatenate([b, *cuts_b]) if cuts_b else b
        r = linprog(
            c,
            A_ub=A2,
            b_ub=b2,
            A_eq=Aeq if has_eq else None,
            b_eq=beq if has_eq else None,
            bounds=list(map(tuple, bnds)),
            method="highs-ds",
        )
        x = r.x
        bound = sgn * (r.fun + offc)
        root = bound if root is None else root
        half = int(np.sum(integ & (np.abs(x - 0.5) < 1e-6)))
        print(f"round {rnd}: LP bound {bound:.6g} ({half} integer columns at 1/2)", flush=True)
        if rnd == a.rounds:
            break
        slack = rb - R @ x
        rows = np.flatnonzero(intdata & (slack < 1 - 1e-9))
        nz = len(rows)
        N = nz + 3 * ni + 1
        cost = np.zeros(N)
        cost[:nz] = np.maximum(slack[rows], 0.0)
        cost[nz : nz + ni] = np.where(lo_ok[ci], x[ci] - bnds[ci, 0], 1e6)
        cost[nz + ni : nz + 2 * ni] = np.where(hi_ok[ci], bnds[ci, 1] - x[ci], 1e6)
        Rr, br = np.round(R[rows]), np.round(rb[rows])
        Mi = np.zeros((ni, N))
        Mi[:, :nz] = Rr[:, ci].T
        Mi[np.arange(ni), nz + np.arange(ni)] = -1.0
        Mi[np.arange(ni), nz + ni + np.arange(ni)] = 1.0
        Mi[np.arange(ni), nz + 2 * ni + np.arange(ni)] = -2.0
        cons = [LinearConstraint(Mi, 0, 0)]
        if len(cc):
            Mc = np.zeros((len(cc), N))
            Mc[:, :nz] = Rr[:, cc].T
            cons.append(LinearConstraint(Mc, 0, 0))
        lov = np.where(lo_ok[ci], bnds[ci, 0], 0.0)
        hiv = np.where(hi_ok[ci], bnds[ci, 1], 0.0)
        Mr = np.zeros((1, N))
        Mr[0, :nz] = br
        Mr[0, nz : nz + ni] = -lov
        Mr[0, nz + ni : nz + 2 * ni] = hiv
        Mr[0, -1] = -2.0
        cons.append(LinearConstraint(Mr, 1, 1))
        cons.append(LinearConstraint(np.r_[np.ones(nz), np.zeros(N - nz)][None, :], 1, np.inf))
        big = 1e6
        bl = np.r_[np.zeros(nz + 2 * ni), -big * np.ones(ni + 1)]
        bh = np.r_[np.ones(nz + 2 * ni), big * np.ones(ni + 1)]
        res = milp(cost, constraints=cons, integrality=np.ones(N), bounds=Bounds(bl, bh))
        executed += 1
        if res.x is None or res.fun >= 1 - 1e-6:
            print(f"  no violated {{0,1/2}} cut (min total slack {getattr(res, 'fun', None)})")
            break
        z = np.round(res.x[:nz])
        tl = np.round(res.x[nz : nz + ni])
        th = np.round(res.x[nz + ni : nz + 2 * ni])
        comb = z @ R[rows]
        comb[ci] += th - tl
        rhs = z @ rb[rows] + th @ hiv - tl @ lov
        coef = np.round(comb / 2.0)
        assert np.allclose(2.0 * coef, comb), "parity violated: not a {0,1/2} combination"
        crhs = np.floor(rhs / 2.0 + 1e-9)
        print(f"  cut: |S|={int(z.sum())} violation {coef @ x - crhs:.4g}", flush=True)
        cuts_a.append(coef[None, :])
        cuts_b.append(np.array([crhs]))
    print("RESULT " + json.dumps({"root": root, "final": bound, "cuts": len(cuts_a)}))
    print(f"EXECUTED {executed} separation MILPs")
    sys.exit(0 if executed else 1)


if __name__ == "__main__":
    main()
