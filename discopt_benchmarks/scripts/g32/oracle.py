"""Shared reference + verdict logic for the G3.2 replay probes (#2136).

The first version of the replays trusted HiGHS's *status* as ground truth. On the
clay0203 GDP master HiGHS returned "Optimal" 54489.4 while Rust's OFF arm returned an
independently verified point at 41907.5, so the row was flagged against the wrong side
(HiGHS's false optimum on that master is #2069). It also never checked that a Rust
``optimal`` had a bound within gap of its objective: the same OFF arm reported
``optimal`` with bound -752904 against objective 41907 (a real driver defect, fixed in
#2136 by withdrawing the certificate when the returned point's objective disagrees
with the tree's incumbent value).

This module replaces both rules:

1. The primal reference is the best **independently verified** point across every arm
   *and* HiGHS's returned point. No solver's status is trusted.
2. An arm reporting ``optimal`` must (a) not be beaten by a verified point beyond the
   gap tolerance and (b) carry a bound within gap of its own objective.

HiGHS disagreements are recorded as ``notes``, never as BAD rows against an arm.
``judge`` is pure, so ``test_oracle.py`` exercises it on synthetic rows.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

INF = 1e20  # the Rust LP layer's infinity sentinel
VIOL_TOL = 1e-6


def _tol(v: float) -> float:
    return 1e-6 * max(1.0, abs(v))


def highs_ref(d: dict, time_limit: float = 120.0) -> dict:
    """Solve the captured standard form with HiGHS (1 thread). Returns status,
    objective, dual bound and the structural part of HiGHS's point (or None)."""
    import highspy

    n, m = d["n"], d["m"]
    A = sp.csc_matrix((d["vals"], d["row_idx"], d["col_ptr"]), shape=(m, n))
    lb = np.where(d["lb"] <= -INF, -highspy.kHighsInf, d["lb"])
    ub = np.where(d["ub"] >= INF, highspy.kHighsInf, d["ub"])
    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = m
    lp.col_cost_ = d["c"]
    lp.col_lower_ = lb
    lp.col_upper_ = ub
    lp.row_lower_ = d["b"]
    lp.row_upper_ = d["b"]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr
    lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    integ = [highspy.HighsVarType.kContinuous] * n
    for j in d["integer_cols"]:
        integ[int(j)] = highspy.HighsVarType.kInteger
    lp.integrality_ = integ
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", time_limit)
    h.setOptionValue("mip_rel_gap", 1e-9)
    h.setOptionValue("mip_abs_gap", 1e-9)
    h.passModel(lp)
    h.run()
    st = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()
    oc = d.get("obj_const", 0.0)
    x = None
    if info.primal_solution_status >= 2:  # kSolutionStatusFeasible
        x = np.asarray(h.getSolution().col_value, float)
    return {
        "status": st,
        "obj": info.objective_function_value + oc,
        "bound": info.mip_dual_bound + oc,
        "x": x,
    }


def verify(d: dict, x) -> float:
    """Worst violation of the structural point ``x`` on the ORIGINAL captured rows:
    column bounds, integrality, and each row's slack range (every logical column of
    the standard form is a singleton). Independent of every solver."""
    ns, n, m = d["n_struct"], d["n"], d["m"]
    A = sp.csc_matrix((d["vals"], d["row_idx"], d["col_ptr"]), shape=(m, n))
    As, Al = A[:, :ns], A[:, ns:].tocsc()
    xs = np.asarray(x, float)[:ns]
    lo, hi = d["lb"][:ns], d["ub"][:ns]
    worst = max(
        np.max(np.maximum(lo - xs, 0), initial=0),
        np.max(np.maximum(xs - hi, 0), initial=0),
    )
    idx = np.asarray(d["integer_cols"], int)
    idx = idx[idx < ns]
    worst = max(worst, np.max(np.abs(xs[idx] - np.round(xs[idx])), initial=0))
    r = d["b"] - As @ xs  # must equal Al s for s in the logical box
    smin = np.zeros(m)
    smax = np.zeros(m)
    for k in range(n - ns):
        s, e = Al.indptr[k], Al.indptr[k + 1]
        assert e - s == 1, "non-standard logical structure"
        i, a = Al.indices[s], Al.data[s]
        lk, uk = d["lb"][ns + k], d["ub"][ns + k]
        v = sorted([a * lk, a * uk])
        smin[i] += v[0]
        smax[i] += v[1]
    viol = np.maximum(smin - r, 0) + np.maximum(r - smax, 0)
    scale = 1.0 + np.abs(d["b"])
    return float(max(worst, float(np.max(viol / scale, initial=0))))


def objective(d: dict, x) -> float:
    """``cᵀx + obj_const`` over the structural columns (slacks carry no cost)."""
    ns = d["n_struct"]
    return float(np.dot(d["c"][:ns], np.asarray(x, float)[:ns]) + d.get("obj_const", 0.0))


def judge(highs: dict, arms: dict, gap_tol: float, abs_gap_tol: float | None = None) -> dict:
    """Verdict for one replayed instance.

    ``highs``: ``{"status", "obj", "viol"}`` where ``viol`` is ``verify`` of HiGHS's
    point (None when it returned none) and ``obj`` is that point's recomputed value.
    ``arms``: ``{name: {"status", "obj", "bound", "viol", "root"}}`` with ``viol`` None
    when the arm returned no point, ``root`` the arm's root bound (or None).

    Returns ``{"best_primal", "best_from", "bad": [...], "notes": [...], "checks": n}``.
    ``checks`` counts the comparisons actually executed (CLAUDE.md §6).
    """
    checks = 0
    verified: list[tuple[float, str]] = []
    for name, a in arms.items():
        if a["viol"] is not None and np.isfinite(a["obj"]) and a["viol"] <= VIOL_TOL:
            verified.append((a["obj"], name))
    if highs.get("viol") is not None and np.isfinite(highs["obj"]) and highs["viol"] <= VIOL_TOL:
        verified.append((highs["obj"], "highs"))
    best, best_from = min(verified) if verified else (np.inf, None)

    bad: list[str] = []
    notes: list[str] = []
    for name, a in arms.items():
        checks += 1
        if a["viol"] is not None and a["viol"] > VIOL_TOL:
            bad.append(f"{name} returned point viol {a['viol']:.3g}")
        if np.isfinite(best):
            checks += 1
            if a["bound"] > best + _tol(best):
                bad.append(f"{name} bound {a['bound']} > verified primal {best} ({best_from})")
            if a.get("root") is not None:
                checks += 1
                if a["root"] > best + _tol(best):
                    bad.append(f"{name} root {a['root']} > verified primal {best} ({best_from})")
            if a["status"] == "infeasible":
                bad.append(f"{name} infeasible but {best_from} has a verified point {best}")
        if a["status"] == "optimal":
            checks += 1
            obj, bd = a["obj"], a["bound"]
            allow = gap_tol * max(1.0, abs(obj)) + _tol(obj)
            if abs_gap_tol is not None:
                allow = max(allow, abs_gap_tol + _tol(obj))
            if not (np.isfinite(obj) and np.isfinite(bd)) or obj - bd > allow:
                bad.append(f"{name} optimal with open gap: obj {obj} bound {bd}")
            if np.isfinite(best) and obj > best + allow:
                bad.append(f"{name} false optimal {obj}: {best_from} has verified {best}")
    if np.isfinite(best):
        checks += 1
        hst, hobj = highs["status"], highs["obj"]
        allow = gap_tol * max(1.0, abs(best)) + _tol(best)
        if hst == "Optimal" and hobj > best + allow:
            notes.append(f"highs false optimum {hobj}: {best_from} has verified {best} (#2069)")
        if hst == "Infeasible":
            notes.append(f"highs infeasible but {best_from} has verified {best}")
    return {"best_primal": best, "best_from": best_from, "bad": bad, "notes": notes, "checks": checks}
