#!/usr/bin/env python
"""Issue #781 entry experiment — cut MANAGEMENT (pooled scored selection) + GMI.

The #282 Stage-3 attribution: discopt's root separation saturates after <=3
rounds because every violated cut is added (the LP degenerates), while SCIP
sustains 21-28 productive rounds by applying only a scored subset (efficacy x
orthogonality over a persistent pool) — and discopt lacks the GMI family
entirely. This probe tests exactly those two mechanisms, out-of-solver, on the
convex NLP-BB panel.

IMPORTANT BASELINE CORRECTION: Stages 2/3 measured against the #770 coefficient
tightening, whose write-back RELAXED consumer-read rows (#772) — their B0s and
"remaining spread" numbers were computed on a corrupted base. This probe uses
the corrected #780 tightening, so all baselines are re-measured here.

Arms (all: root LP = linear big-M rows + OA tangents to convergence per round):
  BASE    existing separators (c-MIR + knapsack cover), add every violated cut
          (reproduces the saturation behaviour; fresh sound-base B0)
  SEL     same candidates + SCIP-style management: persistent pool, efficacy x
          orthogonality scoring, apply only top-K per round
  GMI     BASE + Gomory mixed-integer cuts from the HiGHS basis/tableau
  SELGMI  GMI candidates under SEL management

Kill criterion (#781): no arm closes >=25% of the remaining post-Stage-1
discopt->SCIP root spread on ANY of the 4 instances at 30 rounds -> the
cut-side frontier stands, measured on the sound base; the in-solver build does
not proceed. Diagnostic (not a gate): SEL arms should sustain productive
rounds >>3 if the management hypothesis is right.

Soundness: every reported bound must stay >= opt (MAXIMIZE panel); a bound
below opt is a probe bug and voids the run.

THROWAWAY probe — no shipped solver code.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["DISCOPT_COEF_TIGHTEN"] = "1"  # corrected Stage 1 (#780) — the base

import highspy
import numpy as np
from discopt._jax.cmir_cuts import separate_cmir
from discopt._jax.cover_cuts import separate_cover_cuts
from discopt._jax.gdp_reformulate import reformulate_gdp
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.modeling.core import VarType, from_nl

SNAP = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/")

# (opt, SCIP default root excess %) — same references as Stage 0/2 (#282).
PANEL = {
    "rsyn0805m": dict(opt=1296.1206030, scip_root=16.07),
    "rsyn0810m": dict(opt=1721.4477110, scip_root=9.5),
    "rsyn0815m": dict(opt=1269.9256490, scip_root=17.9),
    "syn40m": dict(opt=67.71325586, scip_root=3.49),
}

ROUNDS = 30
OA_TOL = 1e-6
CUT_VIOL_TOL = 1e-6
SEL_TOP_K = 8
SEL_PARALLEL_MAX = 0.90  # cosine cap vs cuts already selected this round
GMI_F0_MIN = 0.01
GMI_DYNAMISM_MAX = 1e6
KILL_BAR_PCT = 25.0


class RootModel:
    """Linearised root of a convex NLP-BB instance (Stage-2 probe lineage)."""

    def __init__(self, name: str):
        self.name = name
        m = reformulate_gdp(from_nl(SNAP + name + ".nl"), method="big-m")
        self.model = m
        from discopt.solvers._root_presolve import tighten_bigm_coefficients

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
        self.c = -self.ev.evaluate_gradient(x_probe)  # maximize c'x

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
                A_le.append(a.copy())
                b_le.append(-ci)
            elif s == ">=":
                A_le.append(-a.copy())
                b_le.append(ci)
            else:
                A_eq.append(a.copy())
                b_eq.append(-ci)
        self.A_le = np.array(A_le) if A_le else np.zeros((0, self.n))
        self.b_le = np.array(b_le) if b_le else np.zeros(0)
        self.A_eq = np.array(A_eq) if A_eq else np.zeros((0, self.n))
        self.b_eq = np.array(b_eq) if b_eq else np.zeros(0)
        self._fbbt_bounds()

    def _fbbt_bounds(self):
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
                    for kk in nz:
                        if kk == j:
                            continue
                        ak = a[kk]
                        v = lb[kk] if ak > 0 else ub[kk]
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
        self.ub_sep = ub

    def oa_tangent(self, row_idx, x):
        g = self.ev.evaluate_constraints(x)[row_idx]
        J = self.ev.evaluate_jacobian(x)[row_idx]
        s = self.senses[row_idx]
        if s == "<=":
            return J.copy(), float(J @ x - g)
        elif s == ">=":
            return (-J).copy(), float(-(J @ x) + g)
        return None

    def nonlinear_violation(self, x):
        g = self.ev.evaluate_constraints(x)
        out = {}
        for i in self.nl_rows:
            s = self.senses[i]
            v = g[i] if s == "<=" else (-g[i] if s == ">=" else abs(g[i]))
            out[i] = v
        return out


# ── HiGHS LP with basis access ───────────────────────────────────────────────


def solve_lp_highs(rm: RootModel, cuts_A, cuts_b):
    """Maximize c'x over linear rows + cuts. Returns (obj, x, duals, highs, nrow_le)."""
    inf = highspy.kHighsInf
    A_ub = np.vstack([rm.A_le] + ([np.array(cuts_A)] if cuts_A else []))
    b_ub = np.concatenate([rm.b_le] + ([np.array(cuts_b)] if cuts_b else []))
    n_le = A_ub.shape[0]
    n_eq = rm.A_eq.shape[0]
    lp = highspy.HighsLp()
    lp.num_col_ = rm.n
    lp.num_row_ = n_le + n_eq
    lp.sense_ = highspy.ObjSense.kMinimize
    lp.col_cost_ = -rm.c
    lp.col_lower_ = np.where(np.isfinite(rm.lb), rm.lb, -inf)
    lp.col_upper_ = np.where(np.isfinite(rm.ub), rm.ub, inf)
    lp.row_lower_ = np.concatenate([np.full(n_le, -inf), rm.b_eq])
    lp.row_upper_ = np.concatenate([b_ub, rm.b_eq])
    A_all = np.vstack([A_ub, rm.A_eq]) if n_eq else A_ub
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    starts, idx, vals = [0], [], []
    for r in range(A_all.shape[0]):
        nz = np.where(np.abs(A_all[r]) > 1e-13)[0]
        idx.extend(nz.tolist())
        vals.extend(A_all[r, nz].tolist())
        starts.append(len(idx))
    lp.a_matrix_.start_ = np.array(starts, np.int32)
    lp.a_matrix_.index_ = np.array(idx, np.int32)
    lp.a_matrix_.value_ = np.array(vals, float)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.passModel(lp)
    h.run()
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return None, None, None, None, n_le
    sol = h.getSolution()
    x = np.array(sol.col_value, float)
    duals = np.abs(np.array(sol.row_dual, float))[:n_le]
    return float(rm.c @ x), x, duals, h, n_le


# ── GMI separator from the HiGHS basis ───────────────────────────────────────


def separate_gmi(rm: RootModel, h, x, A_all, b_all, n_eq, max_cuts=24):
    """Gomory mixed-integer cuts for fractional basic integer structurals.

    Tableau row for basic integer x_B(i): x_B(i) + sum_N abar_j x_j + sum_R
    binv_r s_r = bbar_i, slacks s_r >= 0 for <= rows (s_r = 0 rows are the eq
    rows: their 'slack' is fixed, coefficient handled as continuous with value
    0 — safe). Nonbasic at upper bound are complemented. GMI in the shifted
    space, then substituted back to x. Every returned cut is (alpha, beta)
    meaning alpha.x <= beta, violated at x.
    """
    st, basic = h.getBasicVariables()
    if st != highspy.HighsStatus.kOk:
        return []
    basic = np.asarray(basic, np.int64)
    bas = h.getBasis()
    col_st = np.array([int(s) for s in bas.col_status])
    row_st = np.array([int(s) for s in bas.row_status])
    k_lower = int(highspy.HighsBasisStatus.kLower)
    k_upper = int(highspy.HighsBasisStatus.kUpper)
    k_basic = int(highspy.HighsBasisStatus.kBasic)
    n = rm.n

    # HiGHS convention (verified numerically): with row-activity variables
    # z_r = a_r.x in [L_r, U_r], the tableau identity for basic x_B(i) is
    #   x_B(i) = -sum_j red_ij x_j + sum_r binv_ir z_r   (nonbasic j, r)
    # Deviation variables (all >= 0): structural at lower p = x_j - l_j
    # (t = red_ij), at upper p = u_j - x_j (t = -red_ij); a binding <= row has
    # z_r at its UPPER bound b_r, deviation q_r = b_r - a_r.x (t = binv_ir).
    # Equality rows contribute constants only (absorbed by d0 = x[bv]).

    cuts = []
    order = []
    for i, bv in enumerate(basic):
        if bv >= 0 and rm.is_int[bv]:
            xv = x[bv]
            f = xv - np.floor(xv)
            if GMI_F0_MIN < f < 1.0 - GMI_F0_MIN:
                order.append((min(f, 1 - f), i, bv))
    order.sort(reverse=True)

    for _score, i, bv in order[: max_cuts * 2]:
        st1, red = h.getReducedRow(i)  # B^-1 A over structural cols
        st2, binv = h.getBasisInverseRow(i)  # B^-1 over rows (slack cols)
        if st1 != highspy.HighsStatus.kOk or st2 != highspy.HighsStatus.kOk:
            continue
        red = np.asarray(red, float)
        binv = np.asarray(binv, float)

        # shifted-space data: d0 and coefficients for nonbasic entries
        d0 = float(x[bv] + 0.0)
        # d0 in shifted space equals value of basic var = x[bv]; f0:
        f0 = d0 - np.floor(d0)
        if not (GMI_F0_MIN < f0 < 1.0 - GMI_F0_MIN):
            continue
        one_mf = 1.0 - f0

        alpha = np.zeros(n)
        beta = 0.0
        ok = True
        maxc, minc = 0.0, np.inf

        # structural nonbasics
        for j in range(n):
            if j == bv or abs(red[j]) < 1e-12:
                continue
            stj = col_st[j]
            if stj == k_basic:
                continue
            if stj == k_lower:
                t = red[j]  # x_j = l_j + xhat
                shift = rm.lb[j]
                sign = 1.0
            elif stj == k_upper:
                t = -red[j]  # x_j = u_j - xhat
                shift = rm.ub[j]
                sign = -1.0
            else:
                # free/other nonbasic with a real tableau coefficient: the
                # deviation is two-sided, the GMI derivation does not apply.
                if abs(red[j]) > 1e-9:
                    ok = False
                    break
                continue
            if not np.isfinite(shift):
                ok = False
                break
            if rm.is_int[j]:
                fj = t - np.floor(t)
                g = fj if fj <= f0 else f0 * (1.0 - fj) / one_mf
            else:
                g = t if t > 0 else -t * f0 / one_mf
            if g < 1e-13:
                continue
            # gamma * xhat contributes: xhat = sign*(x_j - shift)
            alpha[j] += -g * sign  # move to alpha.x <= beta form later
            beta += -g * sign * shift
            maxc = max(maxc, abs(g))
            minc = min(minc, abs(g))
        if not ok:
            continue

        # slack nonbasics: only <= rows binding at their UPPER activity bound
        # (deviation q_r = b_r - a_r.x >= 0, tableau coefficient binv_ir).
        # A <= row with L = -inf cannot be nonbasic at lower; eq rows (r >=
        # m_le) contribute constants only and are excluded by the range.
        m_le = A_all.shape[0]
        for r in range(m_le):
            if abs(binv[r]) < 1e-12:
                continue
            if row_st[r] != k_upper:
                continue
            t = binv[r]
            g = t if t > 0 else -t * f0 / one_mf
            if g < 1e-13:
                continue
            # s = b_r - a_r.x  => gamma*s = gamma*b_r - gamma*a_r.x
            alpha += g * A_all[r]
            beta += g * b_all[r]
            maxc = max(maxc, abs(g))
            minc = min(minc, abs(g))

        # cut: sum gamma xhat >= f0  ->  (-sum gamma xhat) <= -f0
        # alpha.x - beta accumulated as the NEGATIVE of sum gamma xhat pieces:
        # for structural: -g*sign*x_j + g*sign*shift ; slack: +g*a_r.x - g*b_r.
        # So requirement sum >= f0 is: -(alpha.x - beta) >= f0
        #   ->  alpha.x <= beta - f0
        rhs = beta - f0
        nrm = np.linalg.norm(alpha)
        if nrm < 1e-10 or not np.isfinite(rhs):
            continue
        if maxc > 0 and minc < np.inf and maxc / max(minc, 1e-300) > GMI_DYNAMISM_MAX:
            continue
        viol = float(alpha @ x - rhs)
        if viol < CUT_VIOL_TOL:
            continue
        cuts.append((alpha, float(rhs)))
        if len(cuts) >= max_cuts:
            break
    return cuts


# ── cut management (the SEL mechanism) ───────────────────────────────────────


class CutPool:
    def __init__(self):
        self.pool = []  # (a, rhs)
        self.seen = set()

    def offer(self, a, rhs):
        key = tuple(np.round(np.asarray(a, float), 5)) + (round(float(rhs), 5),)
        if key in self.seen:
            return False
        self.seen.add(key)
        self.pool.append((np.asarray(a, float), float(rhs)))
        if len(self.pool) > 4000:
            self.pool = self.pool[-4000:]
        return True

    def violated(self, x, tol=CUT_VIOL_TOL):
        return [(a, r) for a, r in self.pool if a @ x - r > tol]


def select_cuts(candidates, x, top_k=SEL_TOP_K, par_max=SEL_PARALLEL_MAX):
    """Efficacy x orthogonality greedy selection (SCIP-style hybrid, simplified)."""
    scored = []
    for a, rhs in candidates:
        nrm = np.linalg.norm(a)
        if nrm < 1e-12:
            continue
        eff = (a @ x - rhs) / nrm
        if eff > 1e-9:
            scored.append((eff, a, rhs, nrm))
    scored.sort(key=lambda t: -t[0])
    chosen = []
    for eff, a, rhs, nrm in scored:
        if len(chosen) >= top_k:
            break
        ortho_ok = True
        for _e, ca, _r, cn in chosen:
            if abs(a @ ca) / (nrm * cn) > par_max:
                ortho_ok = False
                break
        if ortho_ok:
            chosen.append((eff, a, rhs, nrm))
    return [(a, rhs) for _e, a, rhs, _n in chosen]


# ── arm runner ───────────────────────────────────────────────────────────────


def run_arm(rm: RootModel, arm: str):
    cuts_A, cuts_b = [], []
    pool = CutPool()
    managed = arm in ("SEL", "SELGMI")
    with_gmi = arm in ("GMI", "SELGMI")

    def add_oa(x):
        added = 0
        for i, v in rm.nonlinear_violation(x).items():
            if v > OA_TOL:
                tang = rm.oa_tangent(i, x)
                if tang is None:
                    continue
                a, bb = tang
                cuts_A.append(a)
                cuts_b.append(bb)
                added += 1
        return added

    def oa_converge():
        nonlocal_state = solve_lp_highs(rm, cuts_A, cuts_b)
        obj, x, duals, h, n_le = nonlocal_state
        for _ in range(60):
            if x is None:
                break
            if add_oa(x) == 0:
                break
            obj, x, duals, h, n_le = solve_lp_highs(rm, cuts_A, cuts_b)
        return obj, x, duals, h, n_le

    obj, x, duals, h, n_le = oa_converge()
    B0 = obj
    traj, added_per_round = [], []

    lb_s = np.where(np.isfinite(rm.lb_sep), rm.lb_sep, 0.0)
    ub_s = np.where(np.isfinite(np.minimum(rm.ub, rm.ub_sep)), np.minimum(rm.ub, rm.ub_sep), 1e5)

    for _rnd in range(ROUNDS):
        if x is None:
            traj.append(traj[-1] if traj else B0)
            added_per_round.append(0)
            continue
        A_all = np.vstack([rm.A_le] + ([np.array(cuts_A)] if cuts_A else []))
        b_all = np.concatenate([rm.b_le] + ([np.array(cuts_b)] if cuts_b else []))

        cands = []
        cands += separate_cmir(A_all, b_all, x, lb_s, ub_s, rm.is_int, max_cuts=24, duals=duals)
        cov = separate_cover_cuts(A_all, b_all, x, rm.is_bin, max_cuts=32)
        for C, rhs in cov:
            a = np.zeros(rm.n)
            for j in C:
                a[j] = 1.0
            cands.append((a, float(rhs)))
        if with_gmi and h is not None:
            cands += separate_gmi(rm, h, x, A_all, b_all, rm.A_eq.shape[0], max_cuts=24)

        # keep only violated candidates
        cands = [(np.asarray(a, float), float(r)) for a, r in cands]
        cands = [(a, r) for a, r in cands if a @ x - r > CUT_VIOL_TOL]

        if managed:
            for a, r in cands:
                pool.offer(a, r)
            candidates = pool.violated(x)
            chosen = select_cuts(candidates, x)
        else:
            chosen = cands  # add everything (BASE behaviour)

        added = 0
        for a, r in chosen:
            cuts_A.append(a)
            cuts_b.append(r)
            added += 1
        added_per_round.append(added)

        obj, x, duals, h, n_le = oa_converge()
        traj.append(obj if obj is not None else (traj[-1] if traj else B0))

    return B0, traj, added_per_round


def excess_pct(bound, opt):
    return (bound - opt) / abs(opt) * 100.0


def main():
    results = {}
    for name, info in PANEL.items():
        opt = info["opt"]
        scip = info["scip_root"]
        print(f"\n===== {name}  opt={opt}  scip_root=+{scip}% =====", flush=True)
        rm = RootModel(name)
        print(
            f"  n={rm.n} bin={int(rm.is_bin.sum())} nl_rows={len(rm.nl_rows)} "
            f"le={rm.A_le.shape[0]} eq={rm.A_eq.shape[0]} "
            f"coef_tightened_rows={rm.n_tightened} (sound #780 base)",
            flush=True,
        )
        inst = {
            "opt": opt,
            "scip_root_pct": scip,
            "coef_tightened_rows": int(rm.n_tightened),
            "arms": {},
        }
        for arm in ["BASE", "SEL", "GMI", "SELGMI"]:
            t0 = time.time()
            B0, traj, added = run_arm(rm, arm)
            dt = time.time() - t0
            B0_ex = excess_pct(B0, opt)
            traj_ex = [excess_pct(b, opt) for b in traj]
            final_ex = traj_ex[-1] if traj_ex else B0_ex
            remaining = B0_ex - scip
            closed = (B0_ex - final_ex) / remaining * 100.0 if remaining > 1e-9 else 0.0
            min_ex = min([B0_ex] + traj_ex)
            sound = min_ex >= -1e-4
            prod_rounds = int(sum(1 for a in added if a > 0))
            inst["arms"][arm] = {
                "B0_excess_pct": B0_ex,
                "remaining_spread_pts": remaining,
                "final_excess_pct": final_ex,
                "remaining_closed_pct": closed,
                "min_excess_pct": min_ex,
                "sound_bound_ge_opt": bool(sound),
                "cuts_added_per_round": added,
                "rounds_with_cuts": prod_rounds,
                "seconds": dt,
            }
            flag = "" if sound else "  !! SOUNDNESS VIOLATION (probe bug — run void)"
            print(
                f"  {arm:7s}: B0=+{B0_ex:.3f}%  -> r{ROUNDS}=+{final_ex:.3f}%  "
                f"closed {closed:6.2f}% of remaining ({remaining:.1f}pts)  "
                f"productive_rounds={prod_rounds}  added={added[:10]}  ({dt:.0f}s){flag}",
                flush=True,
            )
        results[name] = inst

    print("\n===== VERDICT (#781 entry experiment, sound #780 base) =====")
    best = {}
    survivor = False
    for name, inst in results.items():
        line = f"  {name}:"
        for arm in ["BASE", "SEL", "GMI", "SELGMI"]:
            c = inst["arms"][arm]["remaining_closed_pct"]
            line += f"  {arm} {c:.2f}%"
            if c >= KILL_BAR_PCT:
                survivor = True
            best[name] = max(best.get(name, -1e9), c)
        print(line + f"   (kill bar {KILL_BAR_PCT}%)")
    print(f"\n  survivor (any arm >= {KILL_BAR_PCT}% of remaining on any instance): {survivor}")
    results["_verdict"] = {
        "survivor": survivor,
        "kill_bar_pct_of_remaining_spread": KILL_BAR_PCT,
        "base": "corrected DISCOPT_COEF_TIGHTEN=1 (#780)",
        "recommendation": "GO (build in-solver)" if survivor else "FRONTIER (record and stop)",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.normpath(os.path.join(here, "..", "results", "issue781"))
    os.makedirs(outdir, exist_ok=True)
    outpath = f"{outdir}/cutmgmt_probe_{stamp}.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  wrote {outpath}")
    return results


if __name__ == "__main__":
    main()
