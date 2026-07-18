"""hda certification entry experiment: log-lift vs float64-intractable-row filter.

Follow-up to the #671 tight-bound work. Goal: find the lever that lets hda's node
LPs solve CLEANLY at tau=0 in the in-house simplex (the keystone for certifying
hda: clean node LPs -> warm starts -> throughput -> branch-and-reduce can close
the McCormick gap).

Hypothesis H (log-lift): hda's ill-conditioned McCormick rows come from
strictly-positive Arrhenius/monomial structure; replacing them with exact
log-domain rows yields a well-conditioned LP solvable at tau=0 with bound
>= -6.47e4.

Kill criterion: coverage — if the wide rows are mostly NOT 2-term ratio form
(rhs=0, opposite signs, both vars lb>0), the exact lift cannot absorb them.

RESULT (2026-07-18):
  * H's log-lift half is FALSIFIED ON COVERAGE: of 130 wide rows, only 4 are
    exact-liftable 2-term ratio rows (50 of the 60 two-term rows have rhs!=0,
    66 rows have 3 terms, 4 have 4; 60/154 touched columns have lb<=0).
  * BUT the measurement found the far simpler sufficient lever: the 130 wide
    rows contribute ZERO tightness at the root box. Dropping them (sound by
    construction: fewer relaxation rows = superset = still a valid outer
    approximation):
      - coefficient spread 2.837e26 -> 3.5e11,
      - HiGHS at tau=0: OPTIMAL, obj -64675.24919969549,
      - in-house feral at tau=0: OPTIMAL, obj -64675.24919969546,
      - Neumaier-Shcherbina safe bound from feral's own dual: -64675.2494
        (rigorous, finite, sound: <= opt -5964.53),
    i.e. exactly the tau-homotopy limit (-64675.25, E1) — no tightness lost —
    and the false-infeasible is gone WITHOUT perturbation, sweep, hardening, or
    external solver.

CONFIRMED lever: a build-time "float64-intractable row" filter (skip emitting
relaxation rows whose per-row coefficient spread exceeds what float64 can
resolve at the LP feasibility tolerance). Sound by superset; tightness impact
is instance-dependent and must be gated by the corpus differential panel
(a wide row MAY carry tightness elsewhere). On hda it is a pure win: the rows
poisoned every float64 engine and bought nothing.

Reproduces: python rowfilter_entry_experiment.py   (needs discopt._rust built)
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))

# Per-row classification thresholds (see doc: a principled production criterion
# should be term-magnitude-based; the coefficient-ratio proxy suffices here).
WIDE_RATIO = 1e6
WIDE_ABS_HI = 1e8
WIDE_ABS_LO = 1e-8


def load():
    d = np.load(os.path.join(HERE, "hda_root_lp.npz"))
    shape = tuple(int(x) for x in d["A_shape"])
    A = sp.csr_matrix((d["A_data"], d["A_indices"], d["A_indptr"]), shape=shape)
    return A, d["b_ub"].astype(float), d["bounds"], d["c"].astype(float)


def wide_rows(A):
    out = []
    for i in range(A.shape[0]):
        s, e = A.indptr[i], A.indptr[i + 1]
        if s == e:
            continue
        a = np.abs(A.data[s:e])
        if a.max() / a.min() > WIDE_RATIO or a.max() > WIDE_ABS_HI or a.min() < WIDE_ABS_LO:
            out.append(i)
    return np.array(out, dtype=int)


def coverage_audit(A, b, bounds, w):
    """The log-lift coverage question (H's kill criterion)."""
    lb = bounds[:, 0]
    nnzs = np.diff(A.indptr)[w]
    import collections

    hist = dict(collections.Counter(nnzs.tolist()))
    two = w[nnzs == 2]
    liftable = rhs_nz = same_sign = blocked = 0
    for i in two:
        s, e = A.indptr[i], A.indptr[i + 1]
        a = A.data[s:e]
        cols = A.indices[s:e]
        if abs(b[i]) > 0:
            rhs_nz += 1
        elif a[0] * a[1] >= 0:
            same_sign += 1
        elif all(lb[j] > 0 for j in cols):
            liftable += 1
        else:
            blocked += 1
    print(f"[coverage] wide rows {len(w)}; nnz hist {hist}")
    print(
        f"[coverage] 2-term: liftable {liftable}, rhs!=0 {rhs_nz}, "
        f"same-sign {same_sign}, lb<=0-blocked {blocked}"
    )
    print(f"[coverage] VERDICT: exact log-lift covers {liftable}/{len(w)} wide rows -> "
          f"{'FALSIFIED on coverage' if liftable < len(w) // 2 else 'viable'}")


def drop_experiment(A, b, bounds, c):
    """The row-filter measurement: drop wide rows, solve at tau=0, certify."""
    w = wide_rows(A)
    keep = np.setdiff1d(np.arange(A.shape[0]), w)
    Ak = A[keep].tocsr()
    bk = b[keep]
    ad = np.abs(Ak.data)
    print(f"[filter] dropped {len(w)} rows -> {Ak.shape[0]}; spread {ad.max()/ad.min():.3e}")

    lb = np.where(np.isfinite(bounds[:, 0]), bounds[:, 0], -np.inf)
    ub = np.where(np.isfinite(bounds[:, 1]), bounds[:, 1], np.inf)
    r = linprog(c, A_ub=Ak, b_ub=bk, bounds=list(zip(lb, ub)), method="highs")
    st = {0: "optimal", 2: "infeasible", 3: "unbounded", 4: "numerical"}.get(r.status, r.status)
    print(f"[filter] HiGHS tau=0: {st} obj={r.fun}")

    # In-house feral at tau=0, standard form [A|I].
    from discopt._rust import solve_lp_warm_csc_py

    mk, n = Ak.shape
    A_std = sp.hstack([Ak, sp.identity(mk, format="csc")], format="csc").tocsc()
    lbs = np.where(np.isfinite(bounds[:, 0]), bounds[:, 0], -1e20)
    ubs = np.where(np.isfinite(bounds[:, 1]), bounds[:, 1], 1e20)
    lb_std = np.concatenate([lbs, np.zeros(mk)])
    ub_std = np.concatenate([ubs, np.full(mk, 1e20)])
    c_std = np.concatenate([c, np.zeros(mk)])
    status, _x, obj, _it, _cs, _bv, dual, _ray = solve_lp_warm_csc_py(
        np.ascontiguousarray(c_std),
        mk,
        n + mk,
        np.ascontiguousarray(A_std.indptr, dtype=np.int64),
        np.ascontiguousarray(A_std.indices, dtype=np.int64),
        np.ascontiguousarray(A_std.data, dtype=np.float64),
        np.ascontiguousarray(bk),
        np.ascontiguousarray(lb_std),
        np.ascontiguousarray(ub_std),
        None,
        None,
    )
    print(f"[filter] in-house feral tau=0: {status} obj={obj}")

    # Rigorous NS certificate from feral's own dual (FBBT-tightened boxes).
    spec = importlib.util.spec_from_file_location(
        "rbe", os.path.join(HERE, "refined_bound_experiment.py")
    )
    rbe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rbe)
    if dual is not None and len(dual):
        lbf, ubf = rbe.fbbt_tighten(Ak.tocsc(), bk, lb.copy(), ub.copy())
        g, finite = rbe.ns_safe_bound(np.asarray(dual), Ak, bk, c, lbf, ubf)
        print(f"[filter] NS safe bound g(y) = {g:.4f} finite={finite} "
              f"sound={g <= -5964.53}")


def main():
    A, b, bounds, c = load()
    w = wide_rows(A)
    coverage_audit(A, b, bounds, w)
    drop_experiment(A, b, bounds, c)


if __name__ == "__main__":
    main()
