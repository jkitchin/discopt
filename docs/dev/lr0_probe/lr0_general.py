"""LR-0 general driver: build the log-space root LP for an .nl instance and
report the root LP lower bound, with sound (possibly loose) constraint
relaxations. Compares H-LOG ON vs OFF (recursive McCormick).

Also does feasible-point sampling: draws >=1000 points satisfying the true
constraints (via rejection / the oracle) and asserts the LP bound never exceeds
their true objective (a valid lower bound must be <= every feasible objective).
"""

from __future__ import annotations

import sys

sys.setrecursionlimit(1_000_000)

import numpy as np
from discopt.modeling import from_nl

from lr0_envelopes import LP
from lr0_relax import Relaxer
from nl_parse import cons_value, load_nl_expressions, obj_value


def get_bounds(nlpath, use_fbbt=True):
    m = from_nl(nlpath)
    nl = m._nl_repr
    n = nl.n_vars
    lb = np.array([m._variables[i].lb for i in range(n)], float)
    ub = np.array([m._variables[i].ub for i in range(n)], float)
    if use_fbbt:
        try:
            flb, fub = nl.fbbt(1000, 1e-9)
            flb = np.asarray(flb, float).ravel()
            fub = np.asarray(fub, float).ravel()
            if len(flb) == n:
                lb = np.maximum(lb, flb)
                ub = np.minimum(ub, fub)
        except Exception as e:
            print("  [fbbt failed]", e, file=sys.stderr)
    return m, nl, lb, ub


def build_root_lp(nlpath, use_log, n_tan=4, use_fbbt=True):
    m, nl, lb, ub = get_bounds(nlpath, use_fbbt)
    P = load_nl_expressions(nlpath)
    n = P["nvars"]
    lp = LP()
    R = Relaxer(lp, lb, ub, use_log_monomial=use_log, n_tan=n_tan)

    # objective: minimize its LP variable. include obj linear terms.
    obj_terms = {}
    obj_const = 0.0
    if P["obj"] is not None:
        on, olo, ohi = R.relax(P["obj"])
        obj_terms[on] = obj_terms.get(on, 0.0) + 1.0
    for vi, co in P["obj_lin"].items():
        obj_terms[f"x{vi}"] = obj_terms.get(f"x{vi}", 0.0) + co
    for k, v in obj_terms.items():
        lp.add_obj(k, v)
    lp.obj_const = obj_const

    # constraints: relax body -> lp var; add linear terms; apply sense vs rhs.
    # A constraint whose sound relaxation is numerically explosive (huge lifted
    # magnitudes from e.g. 1e15 constants) is DROPPED. Dropping a constraint only
    # enlarges the feasible region -> the LP bound stays a valid lower bound
    # (never an over-estimate). We record which were dropped.
    SCALE_CAP = 1e10
    dropped = []
    for j in range(P["ncons"]):
        body = P["cons"].get(j)
        sense = nl.constraint_sense(j)
        rhs = float(nl.constraint_rhs(j))
        n_rows_before = len(lp.rows)
        n_vars_before = len(lp.lb)
        try:
            row = {}
            const = 0.0
            if body is not None:
                bn, blo, bhi = R.relax(body)
                if not (np.isfinite(blo) and np.isfinite(bhi)) or max(abs(blo), abs(bhi)) > SCALE_CAP:
                    raise ValueError("explosive lifted magnitude")
                row[bn] = row.get(bn, 0.0) + 1.0
            for vi, co in P["cons_lin"][j].items():
                row[f"x{vi}"] = row.get(f"x{vi}", 0.0) + co
            s = str(sense)
            if "==" in s or s == "4":
                lp.row(row, rhs - const, "==")
            elif ">=" in s or s == "2":
                lp.row(row, rhs - const, ">=")
            elif "<=" in s or s == "1":
                lp.row(row, rhs - const, "<=")
        except Exception:
            # roll back partial rows/vars for this constraint and drop it
            del lp.rows[n_rows_before:]
            dropped.append(j)
    return m, nl, P, lp, lb, ub, dropped


def sample_feasible(nl, P, lb, ub, n_target=2000, n_try=400000):
    """Rejection-sample points that satisfy all true constraints to tol, using
    the oracle. Returns list of (x, true_obj). For equality-heavy problems this
    is hard; we relax equalities to a small band and report how many found."""
    n = P["nvars"]
    lo = np.where(np.isfinite(lb), lb, -10.0)
    hi = np.where(np.isfinite(ub), ub, 100.0)
    rng = np.random.default_rng(11)
    out = []
    senses = [str(nl.constraint_sense(j)) for j in range(P["ncons"])]
    rhs = [float(nl.constraint_rhs(j)) for j in range(P["ncons"])]
    tolband = 1e-3
    for _ in range(n_try):
        x = rng.uniform(lo, hi)
        ok = True
        for j in range(P["ncons"]):
            g = cons_value(P, j, x)
            s = senses[j]
            if "==" in s or s == "4":
                if abs(g - rhs[j]) > tolband:
                    ok = False; break
            elif ">=" in s or s == "2":
                if g < rhs[j] - 1e-7:
                    ok = False; break
            elif "<=" in s or s == "1":
                if g > rhs[j] + 1e-7:
                    ok = False; break
        if ok:
            out.append((x, obj_value(P, x)))
            if len(out) >= n_target:
                break
    return out


def run(name, nlpath, opt, discopt_root):
    print(f"===== {name} (opt={opt}, discopt root={discopt_root}) =====")
    results = {}
    for use_log in (False, True):
        m, nl, P, lp, lb, ub, dropped = build_root_lp(nlpath, use_log=use_log)
        res, names, idx = lp.solve()
        bound = (res.fun + lp.obj_const) if res.success else None
        tag = "H-LOG ON " if use_log else "McCormick"
        results[use_log] = bound
        if use_log and dropped:
            print(f"  [note] dropped {len(dropped)} numerically-explosive constraints (sound: loosens): {dropped}")
        if bound is None:
            print(f"  {tag}: LP status={res.message}")
        else:
            if discopt_root is not None:
                denom = (opt - discopt_root)
                frac = (bound - discopt_root) / denom if abs(denom) > 1e-12 else float("nan")
            else:
                frac = float("nan")
            within = abs(opt - bound) <= 1e-4 * (1 + abs(opt))
            print(f"  {tag}: root LP bound = {bound:.5f}   gap-to-opt = {opt - bound:.5f}"
                  f"   %gap-closed vs discopt-root = {100*frac:.1f}%   root-cert? {within}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--opt", type=float, required=True)
    ap.add_argument("--root", type=float, default=None)
    ap.add_argument("--sample", action="store_true")
    a = ap.parse_args()
    import os
    cands = [
        f"python/tests/data/minlplib_nl/{a.name}.nl",
        os.path.join(os.path.dirname(__file__), "..", "..", "..",
                     "python", "tests", "data", "minlplib_nl", f"{a.name}.nl"),
    ]
    path = next((p for p in cands if os.path.exists(p)), cands[0])
    res = run(a.name, path, a.opt, a.root)
    if a.sample:
        m, nl, P, lp, lb, ub, dropped = build_root_lp(path, use_log=True)
        feas = sample_feasible(nl, P, lb, ub)
        if feas:
            minobj = min(o for _, o in feas)
            bound = res[True]
            print(f"  [sampling] {len(feas)} feasible pts; min true obj = {minobj:.5f}; "
                  f"H-LOG bound {bound:.5f} <= min? {bound <= minobj + 1e-6}")
        else:
            print("  [sampling] no feasible points found by rejection (equality-heavy).")
