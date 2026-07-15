"""G5 (baron-gap) diagnosis probe — heatexch_gen1 LMTD pole-excluded children.

PF4 §3/§6 (sota-proof-plan) found the LMTD atoms ``w = (a-b)/log(a/(eps+b))``
(eps=1e-6, a,b in [10,+inf)) have a pole on the line ``a = eps+b`` that lies
INSIDE the raw box, so the ``GM <= LMTD <= AM`` envelope is UNSOUND over the whole
box (AM/GM cut feasible points near the pole).  This probe tests the only sound
route PF4 §6 left open: relax the term over pole-EXCLUDED sub-boxes.

We branch ONCE by hand on the ``a = eps+b`` pole line and measure the two
children's root bounds WITH the AM over-estimator ``w <= (a+b)/2`` (the decisive
direction: area cost ~ 1/LMTD is minimised by driving w large, so an UPPER cut on
w raises the dual bound).  Every AM configuration is feasible-point sampled inside
its child box FIRST; a bound is reported only when the sampler shows 0 cuts
(CLAUDE.md: never measure with an unsound cut).

Kill criterion (baron-gap-plan.md §7 task 2): children improve gen1's 38,184 root
bound by < 10%.

Reproduce::

    cd <worktree>
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=$PWD/python \
      python discopt_benchmarks/scripts/g5_heatexch_pole_children_probe.py
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_NL = _REPO / "python" / "tests" / "data" / "minlplib_nl" / "heatexch_gen1.nl"
_EPS = 1e-6
_ORACLE_DUAL = 100552.19  # minlplib.solu =bestdual= heatexch_gen1
_ORACLE_OPT = 154895.93   # =best=
_BASELINE = 38183.53      # uniform root bound at the raw box (PF4 §1: 38,184)
_TAIL = re.compile(r"\*\* -1\) \* \(\(0 \+ (x\d+)\) \+ \(-1 \* (x\d+)\)\)\)$")


def _build_ctx(model, flat_lb, flat_ub):
    from discopt._jax.uniform_relax import LinForm, _Builder
    from discopt._jax.canonical_expr import canonicalize
    from discopt.modeling.core import ObjectiveSense

    dag = canonicalize(model)
    ctx = _Builder(model, flat_lb, flat_ub, track_aux_exprs=True)
    sign = 1.0
    obj_lin = LinForm()
    if dag.objective is not None:
        obj_lin = ctx.rep(dag.objective)
        if model._objective is not None and model._objective.sense == ObjectiveSense.MAXIMIZE:
            sign = -1.0
            obj_lin = obj_lin.scaled(-1.0)
    for con, cnode in zip(model._constraints, dag.constraints):
        lc = ctx.rep(cnode)
        rhs = float(con.rhs)
        if con.sense == "<=":
            ctx.add_row(lc.coeffs, rhs - lc.const)
        elif con.sense == ">=":
            ctx.add_row(lc.scaled(-1.0).coeffs, -rhs + lc.const)
        elif con.sense == "==":
            ctx.add_row(lc.coeffs, rhs - lc.const)
            ctx.add_row(lc.scaled(-1.0).coeffs, -rhs + lc.const)
    return ctx, obj_lin, sign


def _lmtd_terms(ctx, name2idx):
    """[(w_col, a_col, b_col, a_name, b_name)] for each LMTD output aux."""
    out = []
    for j, e in ctx.aux_expr.items():
        s = str(e)
        if s.startswith("((log") and "log(" in s:
            m = _TAIL.search(s)
            if m:
                a, b = m.group(1), m.group(2)
                out.append((j, name2idx[a], name2idx[b], a, b))
    out.sort()
    return out


def _lmtd(a, b):
    """Exact ε-inclusive log-mean the model contains."""
    denom = np.log(a / (_EPS + b))
    # removable only in the limit; away from the pole this is finite
    return np.where(np.abs(denom) < 1e-15, np.nan, (a - b) / denom)


def main() -> None:
    from discopt.modeling.core import from_nl
    from discopt._jax.model_utils import flat_variable_bounds
    import scipy.sparse as sp
    from scipy.optimize import linprog

    warnings.simplefilter("ignore")
    model = from_nl(str(_NL))
    flat_lb, flat_ub = flat_variable_bounds(model)
    name2idx = {v.name: i for i, v in enumerate(model._variables)}
    ctx, obj_lin, sign = _build_ctx(model, flat_lb, flat_ub)
    terms = _lmtd_terms(ctx, name2idx)
    n = len(ctx.col_lb)
    print(f"heatexch_gen1: {len(model._variables)} vars, {len(terms)} LMTD terms, "
          f"{n} LP cols, {len(ctx.rows)} rows")
    print(f"baseline uniform root bound = {_BASELINE}  (oracle dual {_ORACLE_DUAL}, "
          f"opt {_ORACLE_OPT}; gap {100*(1-_BASELINE/_ORACLE_DUAL):.1f}% to dual)")

    base_rows = list(ctx.rows)
    c = np.zeros(n)
    for j, cf in obj_lin.coeffs.items():
        c[j] += cf

    def solve(extra_rows, want_x=False):
        rows = base_rows + extra_rows
        data, ri, ci = [], [], []
        b = np.zeros(len(rows))
        for i, (co, rhs) in enumerate(rows):
            b[i] = rhs
            for j, cf in co.items():
                data.append(cf); ri.append(i); ci.append(j)
        A = sp.csr_matrix((data, (ri, ci)), shape=(len(rows), n))
        bnds = [(ctx.col_lb[j] if math.isfinite(ctx.col_lb[j]) else None,
                 ctx.col_ub[j] if math.isfinite(ctx.col_ub[j]) else None) for j in range(n)]
        r = linprog(c, A_ub=A, b_ub=b, bounds=bnds, method="highs")
        bound = None if r.fun is None else sign * r.fun
        return (r.status, bound, r.x) if want_x else (r.status, bound)

    def am_row(w, a, b):
        # w <= 0.5 a + 0.5 b   ->   w - 0.5 a - 0.5 b <= 0
        return ({w: 1.0, a: -0.5, b: -0.5}, 0.0)

    def branch_pos(a, b, delta):
        # a - b >= delta  ->  b - a <= -delta
        return ({b: 1.0, a: -1.0}, -delta)

    def branch_neg(a, b, delta):
        # a - b <= -delta  ->  a - b <= -delta
        return ({a: 1.0, b: -1.0}, -delta)

    # -- soundness sampler: LMTD(a,b) <= AM over a child's (a,b) region ----- #
    def am_sound(term_subset, sign_pos, delta, n_pts=400000, hi=700.0):
        """Sample a,b for each enveloped term inside the child; return worst
        AM violation LMTD-(a+b)/2 (>0 == UNSOUND cut)."""
        rng = np.random.default_rng(0)
        worst = -np.inf
        for (w, ac, bc, an, bn) in term_subset:
            bb = 10.0 + rng.random(n_pts) * (hi - 10.0)
            if sign_pos:  # a >= b + delta
                aa = bb + delta + rng.random(n_pts) * (hi - (bb + delta)).clip(0)
            else:         # a <= b - delta  (a in [10, b-delta])
                aa = 10.0 + rng.random(n_pts) * (bb - delta - 10.0).clip(0)
            ok = np.isfinite(aa) & (aa >= 10.0) & (aa <= hi)
            aa, bb = aa[ok], bb[ok]
            lm = _lmtd(aa, bb)
            am = 0.5 * (aa + bb)
            v = np.nanmax(lm - am)
            worst = max(worst, float(v))
        return worst

    print("\n=== soundness of AM (w<=(a+b)/2) on pole-excluded regions ===")
    for delta in (1e-3, 1e-2, 0.1, 1.0, 5.0):
        vp = am_sound(terms, True, delta)
        vn = am_sound(terms, False, delta)
        print(f"  delta={delta:<6g}  child a>=b+d worstAMviol={vp:+.4g}   "
              f"child a<=b-d worstAMviol={vn:+.4g}")

    # pick the smallest delta that is sound on the positive child
    delta_sound = None
    for delta in (1e-3, 1e-2, 0.1, 1.0, 5.0, 20.0):
        if am_sound(terms, True, delta) <= 1e-7:
            delta_sound = delta
            break
    print(f"\nchosen sound delta (a>=b+delta child) = {delta_sound}")

    # -- Experiment A: literal single branch on term 0's pole line --------- #
    j0, a0, b0, an0, bn0 = terms[0]
    print(f"\n=== Experiment A: branch ONCE on term0 pole ({an0}={_EPS}+{bn0}) ===")
    d = delta_sound if delta_sound else 1.0
    # child_R: a0>=b0+d, AM only on term0 (only its pole is excluded here)
    stR, bR = solve([branch_pos(a0, b0, d), am_row(j0, a0, b0)])
    stL, bL = solve([branch_neg(a0, b0, d), am_row(j0, a0, b0)])
    print(f"  child_R (a0>=b0+{d}) + AM(term0): status={stR} bound={bR}")
    print(f"  child_L (a0<=b0-{d}) + AM(term0): status={stL} bound={bL}")

    # -- Experiment B: pole-excluded ORTHANT (all 8 terms) ----------------- #
    print("\n=== Experiment B: pole-excluded orthant a_k>=b_k+delta, AM on ALL 8 ===")
    for d in (delta_sound or 1.0,):
        vpos = am_sound(terms, True, d)
        extra = [branch_pos(ac, bc, d) for (_, ac, bc, _, _) in terms]
        extra += [am_row(w, ac, bc) for (w, ac, bc, _, _) in terms]
        st, bd = solve(extra)
        impr = None if bd is None else 100 * (bd - _BASELINE) / _BASELINE
        print(f"  delta={d}: AM-soundness worst={vpos:+.3g}  status={st}  bound={bd}  "
              f"improvement vs baseline={impr:.2f}%" if bd else
              f"  delta={d}: status={st} bound={bd}")

    # branch only (no AM) to isolate the branch's own effect
    st, bd = solve([branch_pos(ac, bc, delta_sound or 1.0) for (_, ac, bc, _, _) in terms])
    print(f"  [control] orthant branch only, NO AM: status={st} bound={bd}")

    # -- WHY the AM cut is inert: inspect the LP optimum ------------------- #
    print("\n=== mechanism: LMTD aux vs AM at the LP optimum (baseline) ===")
    _, _, x = solve([], want_x=True)
    for (w, ac, bc, an, bn) in terms:
        am = 0.5 * (x[ac] + x[bc])
        print(f"  {an},{bn}: w={x[w]:+.4g}  a={x[ac]:.4g} b={x[bc]:.4g}  "
              f"AM=(a+b)/2={am:.4g}  AM-cut slack={am - x[w]:.4g}")
    print("  => temps park at the [10,inf) floor (a=b=10), LMTD aux w=0, so the AM\n"
          "     over-estimator w<=(a+b)/2=10 is fully slack: it cannot move the bound.")


if __name__ == "__main__":
    main()
