#!/usr/bin/env python
"""Issue #282 Stage 0 — throwaway ensemble root-cutting probe.

Out-of-solver root cutting loop on the convex NLP-BB instances rsyn0805m and
syn40m (MAXIMIZE). Builds a root LP = big-M linear rows + OA tangents of the
convex nonlinear rows, then iterates 30 rounds of

    solve LP  ->  separate at the LP vertex (discopt's trusted MIR/c-MIR/cover)
              ->  add OA tangents for violated nonlinear rows  ->  re-solve

tracking the LP objective (an UPPER bound, MAXIMIZE) each round.

Two arms:
  A = discopt separators as-is (separate_cmir + single-row MIR structure + cover)
  B = A plus a prototype Marchand-Wolsey variable-upper-bound (VUB) substitution
      (detect x <= U*y, substitute x -> U*y - s before the MIR delta-scan).

Kill criterion (binding): arm B survives only if it closes >= 10% of the
discopt->SCIP root spread on at least one instance at 30 rounds.

THROWAWAY: no shipped solver code; probe + results JSON + verdict only.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
from scipy.optimize import linprog

from discopt.modeling.core import from_nl, VarType
from discopt._jax.gdp_reformulate import reformulate_gdp
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.cmir_cuts import separate_cmir, _cmir_row
from discopt._jax.cover_cuts import separate_cover_cuts

SNAP = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/"

# (opt, discopt root excess %, SCIP root excess %) from the #282 attribution.
PANEL = {
    "rsyn0805m": dict(opt=1296.1206030, discopt_root=62.873, scip_root=16.07),
    "syn40m": dict(opt=67.71325586, discopt_root=2608.35, scip_root=3.49),
}

ROUNDS = 30
OA_TOL = 1e-6            # nonlinear-row violation tolerance for OA refinement
FEAS_TOL = 1e-7


# ---------------------------------------------------------------------------
# Model extraction
# ---------------------------------------------------------------------------
class RootModel:
    def __init__(self, name):
        self.name = name
        m = reformulate_gdp(from_nl(SNAP + name + ".nl"), method="big-m")
        self.model = m
        lb, ub = flat_variable_bounds(m)
        self.n = len(lb)
        self.lb = lb.astype(float)
        self.ub = ub.astype(float)
        is_bin = np.zeros(self.n, bool)
        is_int = np.zeros(self.n, bool)
        k = 0
        for v in m._variables:
            for _ in range(v.size):
                if v.var_type == VarType.BINARY:
                    is_bin[k] = True
                    is_int[k] = True
                elif v.var_type == VarType.INTEGER:
                    is_int[k] = True
                k += 1
        self.is_bin = is_bin
        self.is_int = is_int
        self.ev = NLPEvaluator(m)
        self.senses = [
            c.sense if isinstance(c.sense, str) else c.sense.value for c in m._constraints
        ]
        # Objective (MAXIMIZE): evaluator negates for max, so grad of its
        # internal (minimization) objective is -c.  We MAXIMIZE c.x.
        assert m._objective.sense.name == "MAXIMIZE"
        x_probe = np.where(np.isfinite(lb), lb, 0.0)
        self.c = -self.ev.evaluate_gradient(x_probe)  # maximize c.x

        # Linear vs nonlinear rows: Jacobian constant across two random points.
        rng = np.random.default_rng(0)
        lo = np.where(np.isfinite(lb), lb, 0.0)
        hi = np.where(np.isfinite(ub), ub, lo + 5.0)
        xa = lo + rng.random(self.n) * (hi - lo)
        xb = lo + rng.random(self.n) * (hi - lo)
        Ja = self.ev.evaluate_jacobian(xa)
        Jb = self.ev.evaluate_jacobian(xb)
        self.lin = np.all(np.isclose(Ja, Jb, atol=1e-9), axis=1)
        self.m_rows = Ja.shape[0]

        # Constant offset for linear rows: g_i(x)=J_i.x + c_i.
        g0 = self.ev.evaluate_constraints(x_probe)
        self.Jlin = Ja  # constant jac rows for linear constraints
        self.clin = g0 - Ja @ x_probe  # offset per row (valid for linear rows)
        self.nl_rows = [i for i in range(self.m_rows) if not self.lin[i]]

        # Assemble the STATIC linear system in <= form:  A_le x <= b_le.
        # normalized row body: g_i(x) sense 0.
        A_le, b_le, A_eq, b_eq = [], [], [], []
        for i in range(self.m_rows):
            if not self.lin[i]:
                continue
            a = Ja[i]
            ci = self.clin[i]  # a.x + ci  (=g_i)
            s = self.senses[i]
            if s == "<=":            # a.x + ci <= 0 -> a.x <= -ci
                A_le.append(a.copy()); b_le.append(-ci)
            elif s == ">=":          # a.x + ci >= 0 -> -a.x <= ci
                A_le.append(-a.copy()); b_le.append(ci)
            else:                    # ==  ->  a.x = -ci
                A_eq.append(a.copy()); b_eq.append(-ci)
        self.A_le = np.array(A_le) if A_le else np.zeros((0, self.n))
        self.b_le = np.array(b_le) if b_le else np.zeros(0)
        self.A_eq = np.array(A_eq) if A_eq else np.zeros((0, self.n))
        self.b_eq = np.array(b_eq) if b_eq else np.zeros(0)

        # Finite bounds for the separators (FBBT over the <= and == linear rows).
        self.ub_sep = self._fbbt_upper_bounds()

        # VUB detection:  x_j <= U * y_k  (continuous j, binary k).
        self.vub = self._detect_vub()

    # ---- FBBT to finitize continuous upper bounds for the separators --------
    def _fbbt_upper_bounds(self):
        lb = self.lb.copy()
        ub = self.ub.copy()
        # combine <= rows and both directions of == rows
        rows_a = list(self.A_le) + list(self.A_eq) + list(-self.A_eq)
        rows_b = list(self.b_le) + list(self.b_eq) + list(-self.b_eq)
        for _ in range(20):
            changed = False
            for a, b in zip(rows_a, rows_b):
                nz = np.where(np.abs(a) > 1e-12)[0]
                for j in nz:
                    aj = a[j]
                    # min activity of the rest
                    rest = 0.0
                    ok = True
                    for k in nz:
                        if k == j:
                            continue
                        ak = a[k]
                        if ak > 0:
                            v = lb[k]
                        else:
                            v = ub[k]
                        if not np.isfinite(v):
                            ok = False
                            break
                        rest += ak * v
                    if not ok:
                        continue
                    bound = (b - rest) / aj
                    if aj > 0:  # x_j <= bound
                        if bound < ub[j] - 1e-9:
                            ub[j] = bound
                            changed = True
                    else:       # x_j >= bound
                        if bound > lb[j] + 1e-9:
                            lb[j] = bound
                            changed = True
            if not changed:
                break
        self.lb_sep = np.maximum(lb, self.lb)
        return ub

    def _detect_vub(self):
        """Map continuous j -> (binary k, U) for rows x_j <= U*y_k (rhs 0)."""
        vub = {}
        for a, b in zip(self.A_le, self.b_le):
            nz = np.where(np.abs(a) > 1e-9)[0]
            if len(nz) != 2:
                continue
            j, k = nz
            for (p, q) in [(j, k), (k, j)]:
                if (not self.is_int[p]) and self.is_bin[q] and a[p] > 0 and a[q] < 0 and abs(b) < 1e-7:
                    U = -a[q] / a[p]
                    # keep the tightest U
                    if p not in vub or U < vub[p][1]:
                        vub[p] = (q, U)
                    break
        return vub

    # ---- OA tangent for a convex nonlinear <= row at point x ---------------
    def oa_tangent(self, row_idx, x):
        g = self.ev.evaluate_constraints(x)[row_idx]
        J = self.ev.evaluate_jacobian(x)[row_idx]
        s = self.senses[row_idx]
        # convex g, sense <=  ->  g(x)+J.(z-x) <= 0  ->  J.z <= J.x - g(x)
        if s == "<=":
            return J.copy(), float(J @ x - g)
        elif s == ">=":  # -g convex? assume feasible region convex; tangent of g>=0
            return (-J).copy(), float(-(J @ x) + g)
        else:
            return None

    def nonlinear_violation(self, x):
        """Max violation of nonlinear rows at x, and per-row values."""
        g = self.ev.evaluate_constraints(x)
        viols = {}
        for i in self.nl_rows:
            s = self.senses[i]
            if s == "<=":
                v = g[i]  # want <= 0
            elif s == ">=":
                v = -g[i]
            else:
                v = abs(g[i])
            viols[i] = v
        return viols


# ---------------------------------------------------------------------------
# LP solve
# ---------------------------------------------------------------------------
def solve_lp(rm, extra_A, extra_b):
    """maximize c.x s.t. A_le x<=b_le, extra cuts, A_eq x=b_eq, bounds.
    Returns (upper_bound, x, duals) or (None,None,None) if infeasible/unbounded.
    ``duals`` are the >=0 magnitudes of the A_ub-row marginals, aligned with the
    combined [A_le ; extra cuts] row order (for dual-weighted MIR aggregation)."""
    A_ub = np.vstack([rm.A_le] + ([np.array(extra_A)] if extra_A else []))
    b_ub = np.concatenate([rm.b_le] + ([np.array(extra_b)] if extra_b else []))
    bounds = []
    for j in range(rm.n):
        lo = rm.lb[j] if np.isfinite(rm.lb[j]) else None
        hi = rm.ub[j] if np.isfinite(rm.ub[j]) else None
        bounds.append((lo, hi))
    res = linprog(
        c=-rm.c,  # minimize -c.x
        A_ub=A_ub, b_ub=b_ub,
        A_eq=rm.A_eq if rm.A_eq.shape[0] else None,
        b_eq=rm.b_eq if rm.A_eq.shape[0] else None,
        bounds=bounds, method="highs",
    )
    if not res.success:
        return None, None, None
    duals = None
    try:
        duals = np.abs(np.asarray(res.ineqlin.marginals, float).ravel())
    except Exception:
        duals = None
    return -res.fun, res.x, duals  # upper bound on max


# ---------------------------------------------------------------------------
# VUB-substituted c-MIR (Arm B prototype, Marchand-Wolsey)
# ---------------------------------------------------------------------------
def cmir_row_vub(a, b, xstar, lb, ub, is_int, vub):
    """c-MIR on aggregated row a.x<=b with VUB substitution x_j=U*y_k - s_j."""
    n = len(a)
    a2 = np.asarray(a, float).copy()
    ext_a, ext_x, ext_lb, ext_ub, ext_int, subs = [], [], [], [], [], []
    for j in range(n):
        if abs(a2[j]) > 1e-12 and (j in vub) and (not is_int[j]):
            k, U = vub[j]
            if not np.isfinite(U) or U <= 0:
                continue
            a2[k] += a2[j] * U
            coef = a2[j]
            a2[j] = 0.0
            sk = U * xstar[k] - xstar[j]           # slack value at LP point
            ext_a.append(-coef)
            ext_x.append(max(sk, 0.0))
            ext_lb.append(0.0)
            ext_ub.append(U)
            ext_int.append(False)
            subs.append((n + len(ext_a) - 1, j, k, U))
    if not subs:
        return None
    A_full = np.concatenate([a2, np.array(ext_a)])
    x_full = np.concatenate([xstar, np.array(ext_x)])
    lb_full = np.concatenate([lb, np.array(ext_lb)])
    ub_full = np.concatenate([ub, np.array(ext_ub)])
    int_full = np.concatenate([is_int, np.array(ext_int, bool)])
    res = _cmir_row(A_full, float(b), x_full, lb_full, ub_full, int_full)
    if res is None:
        return None
    cx, crhs, viol = res
    cx = np.asarray(cx, float)
    out = cx[:n].copy()
    for (si, j, k, U) in subs:
        coef = cx[si]
        # coef * s_j = coef*(U*y_k - x_j)
        out[k] += coef * U
        out[j] += -coef
    return out, crhs, viol


def separate_cmir_vub(A, b, xstar, lb, ub, is_int, vub, max_cuts=16, duals=None):
    """Mirror separate_cmir's aggregation set but apply VUB-substituted c-MIR."""
    A = np.asarray(A, float)
    b = np.asarray(b, float).ravel()
    m = A.shape[0]
    resid = b - A @ xstar
    binding = [r for r in range(m) if abs(resid[r]) < 1e-6]
    rows = [(A[r], float(b[r])) for r in range(m)]
    if duals is not None and len(binding) > 1:
        w = np.asarray(duals, float).ravel()
        if w.shape[0] == m:
            agg_a = np.zeros(A.shape[1]); agg_b = 0.0
            for r in binding:
                wr = abs(float(w[r])); agg_a += wr * A[r]; agg_b += wr * b[r]
            rows.append((agg_a, agg_b))
    for ii in range(min(len(binding), 8)):
        for jj in range(ii + 1, min(len(binding), 8)):
            r1, r2 = binding[ii], binding[jj]
            rows.append((A[r1] + A[r2], float(b[r1] + b[r2])))
    found, seen = [], set()
    for a_row, b_row in rows:
        res = cmir_row_vub(a_row, b_row, xstar, lb, ub, is_int, vub)
        if res is None:
            continue
        cx, crhs, viol = res
        key = tuple(np.round(cx, 5))
        if key in seen:
            continue
        seen.add(key)
        found.append((viol, cx, crhs))
    found.sort(key=lambda t: -t[0])
    return [(cx, crhs) for _v, cx, crhs in found[:max_cuts]]


# ---------------------------------------------------------------------------
# The root cutting loop
# ---------------------------------------------------------------------------
def run_arm(rm, arm, opt):
    """arm in {'A','B'}.  Returns list of per-round upper-bound excess %."""
    cuts_A, cuts_b = [], []   # separation cuts + OA tangents (all <=)

    def add_oa(x):
        added = 0
        viols = rm.nonlinear_violation(x)
        for i, v in viols.items():
            if v > OA_TOL:
                tang = rm.oa_tangent(i, x)
                if tang is None:
                    continue
                a, bb = tang
                cuts_A.append(a); cuts_b.append(bb)
                added += 1
        return added

    # --- OA warm start: converge the nonlinear part = continuous relaxation ---
    ub_val, x, duals = solve_lp(rm, cuts_A, cuts_b)
    for _ in range(60):
        if x is None:
            break
        if add_oa(x) == 0:
            break
        ub_val, x, duals = solve_lp(rm, cuts_A, cuts_b)
    B0 = ub_val
    traj = []

    ub_sep = np.minimum(rm.ub, rm.ub_sep)
    lb_sep = rm.lb_sep
    finite = np.isfinite(ub_sep) & np.isfinite(lb_sep)
    added_per_round = []
    duals_seen = duals is not None

    for rnd in range(ROUNDS):
        if x is None:
            traj.append(traj[-1] if traj else B0)
            continue
        # full <= system at the vertex for separation
        A_all = np.vstack([rm.A_le] + ([np.array(cuts_A)] if cuts_A else []))
        b_all = np.concatenate([rm.b_le] + ([np.array(cuts_b)] if cuts_b else []))

        # bounds for separators: finite box (FBBT), binaries [0,1]
        lb_s = lb_sep.copy(); ub_s = ub_sep.copy()
        # clamp any remaining inf to a large finite cap so separators run
        cap = 1e5
        ub_s = np.where(np.isfinite(ub_s), ub_s, cap)
        lb_s = np.where(np.isfinite(lb_s), lb_s, 0.0)

        new_cuts = []
        # (1) c-MIR / aggregation (single + dual-weighted + pairwise binding rows)
        mir = separate_cmir(A_all, b_all, x, lb_s, ub_s, rm.is_int, max_cuts=24, duals=duals)
        new_cuts += mir
        # (2) knapsack cover on binary rows
        cov = separate_cover_cuts(A_all, b_all, x, rm.is_bin, max_cuts=32)
        for C, rhs in cov:
            a = np.zeros(rm.n)
            for j in C:
                a[j] = 1.0
            new_cuts.append((a, float(rhs)))
        # (3) Arm B: VUB-substituted c-MIR
        if arm == "B":
            mirv = separate_cmir_vub(A_all, b_all, x, lb_s, ub_s, rm.is_int, rm.vub,
                                     max_cuts=24, duals=duals)
            new_cuts += mirv

        # filter: keep cuts violated at x (numerical guard)
        added = 0
        for a, rhs in new_cuts:
            a = np.asarray(a, float)
            if a @ x - rhs > 1e-6:
                cuts_A.append(a); cuts_b.append(rhs); added += 1

        added_per_round.append(added)
        # re-solve, then OA refine, re-solve
        ub_val, x, duals = solve_lp(rm, cuts_A, cuts_b)
        if x is not None:
            if duals is not None:
                duals_seen = True
            if add_oa(x) > 0:
                ub_val, x, duals = solve_lp(rm, cuts_A, cuts_b)
        traj.append(ub_val if ub_val is not None else (traj[-1] if traj else B0))

    return B0, traj, added_per_round, duals_seen


def excess_pct(bound, opt):
    return (bound - opt) / abs(opt) * 100.0


def main():
    results = {}
    for name, info in PANEL.items():
        opt = info["opt"]
        spread = info["discopt_root"] - info["scip_root"]  # pts of excess %
        print(f"\n===== {name}  opt={opt}  discopt_root=+{info['discopt_root']}%  "
              f"scip_root=+{info['scip_root']}%  spread={spread:.2f} pts =====")
        rm = RootModel(name)
        print(f"  n={rm.n} bin={int(rm.is_bin.sum())} int={int(rm.is_int.sum())} "
              f"nl_rows={len(rm.nl_rows)} le={rm.A_le.shape[0]} eq={rm.A_eq.shape[0]} "
              f"vub={len(rm.vub)}")
        inst = {"opt": opt, "discopt_root_pct": info["discopt_root"],
                "scip_root_pct": info["scip_root"], "spread_pts": spread,
                "n_vub": len(rm.vub), "arms": {}}
        for arm in ["A", "B"]:
            t0 = time.time()
            B0, traj, added_per_round, duals_seen = run_arm(rm, arm, opt)
            dt = time.time() - t0
            B0_ex = excess_pct(B0, opt)
            traj_ex = [excess_pct(b, opt) for b in traj]
            final_ex = traj_ex[-1]
            # % of discopt->scip spread closed = (B0_excess - final_excess)/spread
            closed = (B0_ex - final_ex) / spread * 100.0
            # SOUNDNESS: valid relaxation of a MAXIMIZE gives bound >= opt, i.e.
            # excess >= 0 at every round. A negative excess means a cut removed the
            # true optimum (invalid). Flag it.
            min_ex = min([B0_ex] + traj_ex)
            sound = min_ex >= -1e-4
            inst["arms"][arm] = {
                "B0_excess_pct": B0_ex,
                "traj_excess_pct": traj_ex,
                "final_excess_pct": final_ex,
                "spread_closed_pct": closed,
                "min_excess_pct": min_ex,
                "sound_bound_ge_opt": bool(sound),
                "cuts_added_per_round": added_per_round,
                "duals_used": bool(duals_seen),
                "rounds_with_cuts": int(sum(1 for a in added_per_round if a > 0)),
                "seconds": dt,
            }
            if not sound:
                print(f"  !! SOUNDNESS VIOLATION arm {arm}: min excess {min_ex:.4f}% < 0")
            print(f"  arm {arm}: B0=+{B0_ex:.3f}%  ->  round30=+{final_ex:.3f}%   "
                  f"closed {closed:.2f}% of spread   ({dt:.1f}s)  "
                  f"duals={duals_seen} rounds_with_cuts="
                  f"{sum(1 for a in added_per_round if a > 0)} added={added_per_round[:8]}")
            # print a few rounds
            show = [0, 4, 9, 19, 29]
            print("     rounds:", ", ".join(
                f"r{r+1}=+{traj_ex[r]:.2f}%" for r in show if r < len(traj_ex)))
        results[name] = inst

    # verdict
    print("\n===== VERDICT =====")
    survivor = False
    for name, inst in results.items():
        cB = inst["arms"]["B"]["spread_closed_pct"]
        cA = inst["arms"]["A"]["spread_closed_pct"]
        print(f"  {name}: arm A closed {cA:.2f}%, arm B closed {cB:.2f}% of spread "
              f"(kill bar = 10%)")
        if cB >= 10.0:
            survivor = True
    print(f"\n  Arm B survivor (>=10% on >=1 instance): {survivor}")
    results["_verdict"] = {
        "survivor": survivor,
        "kill_bar_pct_of_spread": 10.0,
        "recommendation": "GO" if survivor else "FRONTIER",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    # results/issue282 relative to the repo root (this file lives in
    # discopt_benchmarks/scripts/).
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.normpath(os.path.join(here, "..", "results", "issue282"))
    os.makedirs(outdir, exist_ok=True)
    outpath = f"{outdir}/stage0_ensemble_probe_{stamp}.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  wrote {outpath}")
    return results


if __name__ == "__main__":
    main()
