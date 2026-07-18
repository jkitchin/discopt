#!/usr/bin/env python
"""Issue #282 Stage 2 entry experiment — VUB substitution + sustained aggregation.

Base = Stage-1 coefficient tightening ON (DISCOPT_COEF_TIGHTEN=1). We recompute
the post-Stage-1 root excess B0 as the baseline, then measure the INCREMENTAL
cut-side root gain of:
  (a) VUB/VLB substitution in the aggregation delta-scan
  (b) a sustained aggregation loop (deeper many-row MW aggregations + cut pool)
      that does not saturate at round 3.

Gain reported as % of the REMAINING post-Stage-1 spread (B0_excess - scip_root).

Kill: VUB + sustained aggregation must close >=10% of the remaining spread on at
least one of the four instances, else Stage 2 is falsified.

VERDICT (2026-07-18): FALSIFIED. arm B closes 3.84 / 2.31 / 4.24 / 8.68 % of the
remaining post-Stage-1 spread on rsyn0805m / rsyn0810m / rsyn0815m / syn40m — all
below the 10% bar. VUB is negligible on the tighter root and the sustained loop
still saturates in <=3 rounds. See docs/dev/issue-282-stage2-verdict.md. No cut
code shipped; skip to Stage 3 (NLP-BB root LP-OA ensemble).

THROWAWAY probe — no shipped solver code.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["DISCOPT_COEF_TIGHTEN"] = "1"  # Stage 1 ON — the base for Stage 2

import numpy as np
from discopt._jax.cmir_cuts import _cmir_row, separate_cmir
from discopt._jax.cover_cuts import separate_cover_cuts
from discopt._jax.gdp_reformulate import reformulate_gdp
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.modeling.core import VarType, from_nl
from discopt.solvers._root_presolve import tighten_bigm_coefficients
from scipy.optimize import linprog

SNAP = "/Users/jkitchin/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/"

# (opt, SCIP default root excess %) — discopt_root recomputed below post-Stage-1.
PANEL = {
    "rsyn0805m": dict(opt=1296.1206030, scip_root=16.07),
    "rsyn0810m": dict(opt=1721.4477110, scip_root=9.5),
    "rsyn0815m": dict(opt=1269.9256490, scip_root=17.9),
    "syn40m": dict(opt=67.71325586, scip_root=3.49),
}

ROUNDS = 30
OA_TOL = 1e-6
FEAS_TOL = 1e-7


class RootModel:
    def __init__(self, name, coef_tighten=True):
        self.name = name
        m = reformulate_gdp(from_nl(SNAP + name + ".nl"), method="big-m")
        self.model = m
        self.n_tightened = 0
        if coef_tighten:
            self.n_tightened = tighten_bigm_coefficients(m)
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
        assert m._objective.sense.name == "MAXIMIZE"
        x_probe = np.where(np.isfinite(lb), lb, 0.0)
        self.c = -self.ev.evaluate_gradient(x_probe)

        rng = np.random.default_rng(0)
        lo = np.where(np.isfinite(lb), lb, 0.0)
        hi = np.where(np.isfinite(ub), ub, lo + 5.0)
        xa = lo + rng.random(self.n) * (hi - lo)
        xb = lo + rng.random(self.n) * (hi - lo)
        Ja = self.ev.evaluate_jacobian(xa)
        Jb = self.ev.evaluate_jacobian(xb)
        self.lin = np.all(np.isclose(Ja, Jb, atol=1e-9), axis=1)
        self.m_rows = Ja.shape[0]

        g0 = self.ev.evaluate_constraints(x_probe)
        self.Jlin = Ja
        self.clin = g0 - Ja @ x_probe
        self.nl_rows = [i for i in range(self.m_rows) if not self.lin[i]]

        A_le, b_le, A_eq, b_eq = [], [], [], []
        for i in range(self.m_rows):
            if not self.lin[i]:
                continue
            a = Ja[i]
            ci = self.clin[i]
            s = self.senses[i]
            if s == "<=":
                A_le.append(a.copy()); b_le.append(-ci)
            elif s == ">=":
                A_le.append(-a.copy()); b_le.append(ci)
            else:
                A_eq.append(a.copy()); b_eq.append(-ci)
        self.A_le = np.array(A_le) if A_le else np.zeros((0, self.n))
        self.b_le = np.array(b_le) if b_le else np.zeros(0)
        self.A_eq = np.array(A_eq) if A_eq else np.zeros((0, self.n))
        self.b_eq = np.array(b_eq) if b_eq else np.zeros(0)

        self.ub_sep = self._fbbt_upper_bounds()
        self.vub = self._detect_vub()

    def _fbbt_upper_bounds(self):
        lb = self.lb.copy()
        ub = self.ub.copy()
        rows_a = list(self.A_le) + list(self.A_eq) + list(-self.A_eq)
        rows_b = list(self.b_le) + list(self.b_eq) + list(-self.b_eq)
        for _ in range(20):
            changed = False
            for a, b in zip(rows_a, rows_b):
                nz = np.where(np.abs(a) > 1e-12)[0]
                for j in nz:
                    aj = a[j]
                    rest = 0.0
                    ok = True
                    for k in nz:
                        if k == j:
                            continue
                        ak = a[k]
                        v = lb[k] if ak > 0 else ub[k]
                        if not np.isfinite(v):
                            ok = False
                            break
                        rest += ak * v
                    if not ok:
                        continue
                    bound = (b - rest) / aj
                    if aj > 0:
                        if bound < ub[j] - 1e-9:
                            ub[j] = bound
                            changed = True
                    else:
                        if bound > lb[j] + 1e-9:
                            lb[j] = bound
                            changed = True
            if not changed:
                break
        self.lb_sep = np.maximum(lb, self.lb)
        return ub

    def _detect_vub(self):
        """Map continuous j -> (binary k, U) for rows x_j <= U*y_k (rhs 0).
        Also detect VLB: x_j >= L*y_k."""
        vub = {}
        for a, b in zip(self.A_le, self.b_le):
            nz = np.where(np.abs(a) > 1e-9)[0]
            if len(nz) != 2:
                continue
            j, k = nz
            for (p, q) in [(j, k), (k, j)]:
                if (
                    (not self.is_int[p]) and self.is_bin[q]
                    and a[p] > 0 and a[q] < 0 and abs(b) < 1e-7
                ):
                    U = -a[q] / a[p]
                    if p not in vub or vub[p][1] > U:
                        vub[p] = (q, U)
                    break
        return vub

    def oa_tangent(self, row_idx, x):
        g = self.ev.evaluate_constraints(x)[row_idx]
        J = self.ev.evaluate_jacobian(x)[row_idx]
        s = self.senses[row_idx]
        if s == "<=":
            return J.copy(), float(J @ x - g)
        elif s == ">=":
            return (-J).copy(), float(-(J @ x) + g)
        else:
            return None

    def nonlinear_violation(self, x):
        g = self.ev.evaluate_constraints(x)
        viols = {}
        for i in self.nl_rows:
            s = self.senses[i]
            if s == "<=":
                v = g[i]
            elif s == ">=":
                v = -g[i]
            else:
                v = abs(g[i])
            viols[i] = v
        return viols


def solve_lp(rm, extra_A, extra_b):
    A_ub = np.vstack([rm.A_le] + ([np.array(extra_A)] if extra_A else []))
    b_ub = np.concatenate([rm.b_le] + ([np.array(extra_b)] if extra_b else []))
    bounds = []
    for j in range(rm.n):
        lo = rm.lb[j] if np.isfinite(rm.lb[j]) else None
        hi = rm.ub[j] if np.isfinite(rm.ub[j]) else None
        bounds.append((lo, hi))
    res = linprog(
        c=-rm.c, A_ub=A_ub, b_ub=b_ub,
        A_eq=rm.A_eq if rm.A_eq.shape[0] else None,
        b_eq=rm.b_eq if rm.A_eq.shape[0] else None,
        bounds=bounds, method="highs",
    )
    if not res.success:
        return None, None, None
    try:
        duals = np.abs(np.asarray(res.ineqlin.marginals, float).ravel())
    except Exception:
        duals = None
    return -res.fun, res.x, duals


# ---- VUB-substituted c-MIR (Marchand-Wolsey) -----------------------------
def cmir_row_vub(a, b, xstar, lb, ub, is_int, vub):
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
            sk = U * xstar[k] - xstar[j]
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
        out[k] += coef * U
        out[j] += -coef
    return out, crhs, viol


def _greedy_agg_rows(A, b, xstar, is_int, binding, max_rows=4, max_aggs=40):
    """Sustained many-row MW aggregation: greedily chain rows that cancel a
    continuous / fractional column, growing aggregates of up to max_rows rows.
    Returns a list of (a_agg, b_agg) implied <= rows (nonneg combos)."""
    A = np.asarray(A, float)
    b = np.asarray(b, float).ravel()
    n = A.shape[1]
    out = []
    # candidate cancel columns: fractional at xstar (the interesting faces)
    frac = [j for j in range(n) if abs(xstar[j] - round(xstar[j])) > 1e-6]
    if not frac:
        return out
    cand = binding[:12] if binding else list(range(min(A.shape[0], 12)))
    count = 0
    for start in cand:
        agg_a = A[start].copy()
        agg_b = float(b[start])
        used = {start}
        for _depth in range(max_rows - 1):
            # pick a cancel target col that agg currently has nonzero on & is frac
            targets = [j for j in frac if abs(agg_a[j]) > 1e-9]
            if not targets:
                break
            t = max(targets, key=lambda j: abs(agg_a[j]))
            # find a row (not used) with opposite sign on t
            best_r = None
            for r in cand:
                if r in used or abs(A[r, t]) < 1e-9:
                    continue
                if np.sign(A[r, t]) == np.sign(agg_a[t]):
                    continue
                best_r = r
                break
            if best_r is None:
                break
            lam = abs(agg_a[t])
            lam2 = abs(A[best_r, t])
            new_a = lam2 * agg_a + lam * A[best_r]
            new_a[t] = 0.0
            new_b = lam2 * agg_b + lam * b[best_r]
            mx = np.max(np.abs(new_a))
            if mx < 1e-9 or mx > 1e10 or not np.isfinite(new_b):
                break
            agg_a, agg_b = new_a, new_b
            used.add(best_r)
            out.append((agg_a.copy(), float(agg_b)))
            count += 1
            if count >= max_aggs:
                return out
    return out


def separate_stage2(A, b, xstar, lb, ub, is_int, vub, duals=None, max_cuts=48):
    """VUB-substituted c-MIR over: single rows, dual-weighted, pairwise binding,
    AND deep greedy many-row MW aggregations (the sustained loop)."""
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
    # deep many-row aggregations (the sustained-loop lever)
    rows += _greedy_agg_rows(A, b, xstar, is_int, binding)

    found, seen = [], set()
    for a_row, b_row in rows:
        # try both plain c-MIR and VUB-substituted c-MIR on the aggregate
        for res in (
            cmir_row_vub(a_row, b_row, xstar, lb, ub, is_int, vub),
            _cmir_row(np.asarray(a_row, float), float(b_row), xstar, lb, ub, is_int),
        ):
            if res is None:
                continue
            cx, crhs, viol = res
            cx = np.asarray(cx, float)
            key = tuple(np.round(cx, 5))
            if key in seen:
                continue
            seen.add(key)
            found.append((viol, cx, crhs))
    found.sort(key=lambda t: -t[0])
    return [(cx, crhs) for _v, cx, crhs in found[:max_cuts]]


def run_arm(rm, arm, opt):
    cuts_A, cuts_b = [], []

    def add_oa(x):
        added = 0
        for i, v in rm.nonlinear_violation(x).items():
            if v > OA_TOL:
                tang = rm.oa_tangent(i, x)
                if tang is None:
                    continue
                a, bb = tang
                cuts_A.append(a); cuts_b.append(bb); added += 1
        return added

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
    added_per_round = []
    pool = []  # persistent cut pool (a_row, rhs)

    for _rnd in range(ROUNDS):
        if x is None:
            traj.append(traj[-1] if traj else B0)
            added_per_round.append(0)
            continue
        A_all = np.vstack([rm.A_le] + ([np.array(cuts_A)] if cuts_A else []))
        b_all = np.concatenate([rm.b_le] + ([np.array(cuts_b)] if cuts_b else []))

        lb_s = np.where(np.isfinite(lb_sep), lb_sep, 0.0)
        ub_s = np.where(np.isfinite(ub_sep), ub_sep, 1e5)

        new_cuts = []
        if arm == "A":
            new_cuts += separate_cmir(A_all, b_all, x, lb_s, ub_s, rm.is_int,
                                      max_cuts=24, duals=duals)
            cov = separate_cover_cuts(A_all, b_all, x, rm.is_bin, max_cuts=32)
            for C, rhs in cov:
                a = np.zeros(rm.n)
                for j in C:
                    a[j] = 1.0
                new_cuts.append((a, float(rhs)))
        else:  # arm B: VUB + sustained aggregation + cut pool
            new_cuts += separate_stage2(A_all, b_all, x, lb_s, ub_s, rm.is_int,
                                        rm.vub, duals=duals, max_cuts=48)
            cov = separate_cover_cuts(A_all, b_all, x, rm.is_bin, max_cuts=32)
            for C, rhs in cov:
                a = np.zeros(rm.n)
                for j in C:
                    a[j] = 1.0
                new_cuts.append((a, float(rhs)))
            # cut pool: re-check previously found (not yet added) cuts for violation
            for a, rhs in pool:
                if a @ x - rhs > 1e-6:
                    new_cuts.append((a, rhs))

        added = 0
        for a, rhs in new_cuts:
            a = np.asarray(a, float)
            if a @ x - rhs > 1e-6:
                cuts_A.append(a); cuts_b.append(rhs); added += 1
                if arm == "B":
                    pool.append((a, rhs))
        added_per_round.append(added)

        ub_val, x, duals = solve_lp(rm, cuts_A, cuts_b)
        if x is not None and add_oa(x) > 0:
            ub_val, x, duals = solve_lp(rm, cuts_A, cuts_b)
        traj.append(ub_val if ub_val is not None else (traj[-1] if traj else B0))

    return B0, traj, added_per_round


def excess_pct(bound, opt):
    return (bound - opt) / abs(opt) * 100.0


def main():
    results = {}
    for name, info in PANEL.items():
        opt = info["opt"]
        scip = info["scip_root"]
        print(f"\n===== {name}  opt={opt}  scip_root=+{scip}% =====")
        rm = RootModel(name, coef_tighten=True)
        print(f"  n={rm.n} bin={int(rm.is_bin.sum())} int={int(rm.is_int.sum())} "
              f"nl_rows={len(rm.nl_rows)} le={rm.A_le.shape[0]} eq={rm.A_eq.shape[0]} "
              f"vub={len(rm.vub)} coef_tightened_rows={rm.n_tightened}")
        inst = {"opt": opt, "scip_root_pct": scip,
                "n_vub": len(rm.vub), "coef_tightened_rows": int(rm.n_tightened),
                "arms": {}}
        for arm in ["A", "B"]:
            t0 = time.time()
            B0, traj, added_per_round = run_arm(rm, arm, opt)
            dt = time.time() - t0
            B0_ex = excess_pct(B0, opt)  # post-Stage-1 baseline (same for both arms)
            traj_ex = [excess_pct(b, opt) for b in traj]
            final_ex = traj_ex[-1]
            # remaining post-Stage-1 spread = B0_excess - scip_root
            remaining = B0_ex - scip
            closed = (B0_ex - final_ex) / remaining * 100.0 if remaining > 1e-9 else 0.0
            min_ex = min([B0_ex] + traj_ex)
            sound = min_ex >= -1e-4
            inst["arms"][arm] = {
                "B0_excess_pct": B0_ex,
                "remaining_spread_pts": remaining,
                "final_excess_pct": final_ex,
                "remaining_closed_pct": closed,
                "min_excess_pct": min_ex,
                "sound_bound_ge_opt": bool(sound),
                "cuts_added_per_round": added_per_round,
                "rounds_with_cuts": int(sum(1 for a in added_per_round if a > 0)),
                "seconds": dt,
            }
            if not sound:
                print(f"  !! SOUNDNESS VIOLATION arm {arm}: min excess {min_ex:.4f}% < 0")
            print(f"  arm {arm}: B0=+{B0_ex:.3f}% (post-S1)  remaining_spread={remaining:.2f}pts"
                  f"  -> r30=+{final_ex:.3f}%   closed {closed:.2f}% of REMAINING   ({dt:.1f}s)"
                  f"  rounds_with_cuts={sum(1 for a in added_per_round if a > 0)}"
                  f"  added={added_per_round[:8]}")
        results[name] = inst

    print("\n===== VERDICT (Stage 2, base = Stage-1 coef-tighten ON) =====")
    survivor = False
    for name, inst in results.items():
        cB = inst["arms"]["B"]["remaining_closed_pct"]
        cA = inst["arms"]["A"]["remaining_closed_pct"]
        print(f"  {name}: arm A closed {cA:.2f}%, arm B closed {cB:.2f}% of REMAINING "
              f"post-S1 spread (kill bar=10%)")
        if cB >= 10.0:
            survivor = True
    print(f"\n  Stage 2 survivor (arm B >=10% of remaining on >=1 instance): {survivor}")
    results["_verdict"] = {
        "survivor": survivor, "kill_bar_pct_of_remaining_spread": 10.0,
        "base": "Stage-1 DISCOPT_COEF_TIGHTEN=1",
        "recommendation": "GO" if survivor else "FRONTIER->Stage3",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.normpath(os.path.join(here, "..", "results", "issue282"))
    os.makedirs(outdir, exist_ok=True)
    outpath = f"{outdir}/stage2_probe_{stamp}.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  wrote {outpath}")
    return results


if __name__ == "__main__":
    main()
