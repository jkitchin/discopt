"""NLP-BB root cutting-plane stage (issue #781, ``DISCOPT_NLPBB_ROOT_CUTS``).

Generates globally valid, integrality-based cutting planes at the root of the
NLP-BB path and returns them for insertion as model constraints, together with
the final root-LP dual bound. The mechanism is the one the #781 entry
experiment validated (probe: ``discopt_benchmarks/scripts/issue781_cutmgmt_probe.py``):

  * root LP = the model's linear rows + outer-approximation tangents of the
    convex nonlinear rows, iterated to OA convergence each round;
  * separators: Gomory mixed-integer cuts from the HiGHS basis/tableau (the
    load-bearing family — closes 75–93% of the remaining root spread on the
    convex synthesis panel), plus the existing c-MIR and knapsack-cover
    separators;
  * SCIP-style cut management: persistent pool, efficacy × orthogonality
    scoring, apply only the top-K per round (compounds on GMI: same-or-better
    bound with ~4× fewer cuts applied).

Soundness contract (CLAUDE.md §1):
  * Only runs when the model is CONVEX-certified (OA tangents of convex ≤ rows
    / concave ≥ rows are valid outer approximations) and the objective is
    verified linear (the LP objective must represent the true objective).
  * Every cut is integrality-valid: satisfied by every point of the model's
    feasible set with integral integer variables. Adding them as constraints
    removes no integer-feasible point; node NLP relaxations only tighten.
  * The returned LP bound is the optimum of an outer approximation of the
    integer-feasible set, hence a valid dual bound for the MINLP.
  * The GMI derivation was validated by exact per-integer-corner LP
    enumeration on seeded random MILPs (0 unsound cuts; regression test
    ``python/tests/test_nlpbb_root_cuts_781.py``), and each cut's rhs carries a
    small relative safety relaxation against tableau roundoff.
  * Everything is failure-safe: any error returns "no cuts" and the solve
    proceeds exactly as with the flag off.

Default OFF (bound-changing, CLAUDE.md §5); graduation requires the Regime-2
panel (cert-clean + net-positive).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

ROUNDS = 30
OA_TOL = 1e-6
OA_MAX_ITERS = 60
CUT_VIOL_TOL = 1e-6
SEL_TOP_K = 8
SEL_PARALLEL_MAX = 0.90
GMI_F0_MIN = 0.01
GMI_DYNAMISM_MAX = 1e6
GMI_RHS_SAFETY_REL = 1e-9  # relative rhs relaxation absorbing tableau roundoff
MAX_CUTS_PER_FAMILY = 24
POOL_MAX = 4000


def nlpbb_root_cuts_enabled() -> bool:
    """Whether the ``DISCOPT_NLPBB_ROOT_CUTS`` opt-in flag is set (default OFF)."""
    val = os.environ.get("DISCOPT_NLPBB_ROOT_CUTS", "0").strip().lower()
    return val not in ("", "0", "false", "off", "no")


@dataclass
class RootCutResult:
    """Applied cuts (``alpha·x <= rhs`` each) and the final root-LP dual bound."""

    cuts: list = field(default_factory=list)  # [(alpha: np.ndarray, rhs: float)]
    lp_bound: float | None = None  # in the model's objective sense
    rounds_run: int = 0
    productive_rounds: int = 0


# ── linearised root view ─────────────────────────────────────────────────────


class _RootLP:
    """Linear rows + OA machinery over the (FBBT-tightened) root box."""

    def __init__(self, model, evaluator, lb, ub, is_int, is_bin, sense_max: bool):
        self.n = len(lb)
        self.lb = np.asarray(lb, float)
        self.ub = np.asarray(ub, float)
        self.is_int = is_int
        self.is_bin = is_bin
        self.ev = evaluator
        self.sense_max = sense_max
        self.senses = [
            c.sense if isinstance(c.sense, str) else c.sense.value for c in model._constraints
        ]

        # Objective must be LINEAR for the LP objective to represent it.
        rng = np.random.default_rng(0)
        lo = np.where(np.isfinite(self.lb), self.lb, 0.0)
        hi = np.where(np.isfinite(self.ub), self.ub, lo + 5.0)
        xa = lo + rng.random(self.n) * (hi - lo)
        xb = lo + rng.random(self.n) * (hi - lo)
        ga = np.asarray(evaluator.evaluate_gradient(xa), float)
        gb = np.asarray(evaluator.evaluate_gradient(xb), float)
        if not np.allclose(ga, gb, atol=1e-9):
            raise ValueError("nonlinear objective — root-cut LP bound would be invalid")
        # NLPEvaluator compiles a minimize-internal objective: it NEGATES the
        # declared objective for MAXIMIZE models (``NLPEvaluator._negate``).
        # Recover the DECLARED objective coefficients so the LP bound is in the
        # model's sense (max → LP max c'x is a valid upper bound).
        negate = bool(getattr(evaluator, "_negate", sense_max))
        self.c = -ga if negate else ga

        ja = evaluator.evaluate_jacobian(xa)
        jb = evaluator.evaluate_jacobian(xb)
        lin_rows = np.all(np.isclose(ja, jb, atol=1e-9), axis=1)
        g0 = np.asarray(evaluator.evaluate_constraints(xa), float)
        const = g0 - ja @ xa
        self.nl_rows = [i for i in range(ja.shape[0]) if not lin_rows[i]]

        a_le, b_le, a_eq, b_eq = [], [], [], []
        for i in range(ja.shape[0]):
            if not lin_rows[i]:
                continue
            a = np.asarray(ja[i], float)
            ci = float(const[i])
            s = self.senses[i]
            if s == "<=":
                a_le.append(a)
                b_le.append(-ci)
            elif s == ">=":
                a_le.append(-a)
                b_le.append(ci)
            else:
                a_eq.append(a)
                b_eq.append(-ci)
        self.A_le = np.array(a_le) if a_le else np.zeros((0, self.n))
        self.b_le = np.array(b_le) if b_le else np.zeros(0)
        self.A_eq = np.array(a_eq) if a_eq else np.zeros((0, self.n))
        self.b_eq = np.array(b_eq) if b_eq else np.zeros(0)
        self._fbbt_separation_bounds()

    def _fbbt_separation_bounds(self):
        """Interval FBBT over the linear rows for the separators' activity bounds."""
        lb = self.lb.copy()
        ub = self.ub.copy()
        rows_a = list(self.A_le) + list(self.A_eq) + list(-self.A_eq)
        rows_b = list(self.b_le) + list(self.b_eq) + list(-self.b_eq)
        for _ in range(20):
            changed = False
            for a, b in zip(rows_a, rows_b, strict=True):
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
                    elif bound > lb[j] + 1e-9:
                        lb[j] = bound
                        changed = True
            if not changed:
                break
        self.lb_sep = np.maximum(lb, self.lb)
        self.ub_sep = np.minimum(ub, self.ub)

    def oa_tangent(self, row_idx, x):
        g = self.ev.evaluate_constraints(x)[row_idx]
        jrow = self.ev.evaluate_jacobian(x)[row_idx]
        s = self.senses[row_idx]
        if s == "<=":
            return np.asarray(jrow, float).copy(), float(jrow @ x - g)
        if s == ">=":
            return -np.asarray(jrow, float), float(-(jrow @ x) + g)
        return None  # nonlinear equality: model would not be convex-certified

    def nonlinear_violations(self, x):
        g = self.ev.evaluate_constraints(x)
        out = {}
        for i in self.nl_rows:
            s = self.senses[i]
            out[i] = g[i] if s == "<=" else (-g[i] if s == ">=" else abs(g[i]))
        return out


def _solve_lp(root: _RootLP, cuts_a, cuts_b):
    """Solve the root LP with HiGHS. Returns (obj_in_model_sense, x, duals, highs)."""
    import highspy

    inf = highspy.kHighsInf
    a_ub = np.vstack([root.A_le] + ([np.array(cuts_a)] if cuts_a else []))
    b_ub = np.concatenate([root.b_le] + ([np.array(cuts_b)] if cuts_b else []))
    n_le = a_ub.shape[0]
    n_eq = root.A_eq.shape[0]
    lp = highspy.HighsLp()
    lp.num_col_ = root.n
    lp.num_row_ = n_le + n_eq
    lp.sense_ = highspy.ObjSense.kMinimize
    lp.col_cost_ = -root.c if root.sense_max else root.c
    lp.col_lower_ = np.where(np.isfinite(root.lb), root.lb, -inf)
    lp.col_upper_ = np.where(np.isfinite(root.ub), root.ub, inf)
    lp.row_lower_ = np.concatenate([np.full(n_le, -inf), root.b_eq])
    lp.row_upper_ = np.concatenate([b_ub, root.b_eq])
    a_all = np.vstack([a_ub, root.A_eq]) if n_eq else a_ub
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    starts, idx, vals = [0], [], []
    for r in range(a_all.shape[0]):
        nz = np.where(np.abs(a_all[r]) > 1e-13)[0]
        idx.extend(nz.tolist())
        vals.extend(a_all[r, nz].tolist())
        starts.append(len(idx))
    lp.a_matrix_.start_ = np.array(starts, np.int32)
    lp.a_matrix_.index_ = np.array(idx, np.int32)
    lp.a_matrix_.value_ = np.array(vals, float)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.passModel(lp)
    h.run()
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return None, None, None, None
    sol = h.getSolution()
    x = np.array(sol.col_value, float)
    duals = np.abs(np.array(sol.row_dual, float))[:n_le]
    return float(root.c @ x), x, duals, h


# ── GMI separator (validated: exact enumeration, 0 unsound) ──────────────────


def separate_gmi(root: _RootLP, h, x, a_all, b_all, max_cuts=MAX_CUTS_PER_FAMILY):
    """Gomory mixed-integer cuts for fractional basic integer structurals.

    HiGHS convention (verified numerically): with row-activity variables
    ``z_r = a_r·x`` the tableau identity for basic ``x_B(i)`` is
    ``x_B(i) = −Σ red_ij·x_j + Σ binv_ir·z_r`` over nonbasic j, r. Deviation
    variables (all ≥ 0): structural at lower ``p = x_j − l_j`` (t = red_ij),
    at upper ``p = u_j − x_j`` (t = −red_ij); a binding ≤ row has z at its
    UPPER bound, deviation ``q_r = b_r − a_r·x`` (t = binv_ir). Equality rows
    contribute constants only (absorbed by d0 = x[bv]). GMI over
    ``x_B + Σ t_k v_k = d0`` then substituted back to x-space.
    """
    import highspy

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
    n = root.n

    order = []
    for i, bv in enumerate(basic):
        if bv >= 0 and root.is_int[bv]:
            f = x[bv] - np.floor(x[bv])
            if GMI_F0_MIN < f < 1.0 - GMI_F0_MIN:
                order.append((min(f, 1 - f), i, int(bv)))
    order.sort(reverse=True)

    cuts = []
    for _score, i, bv in order[: max_cuts * 2]:
        st1, red = h.getReducedRow(i)
        st2, binv = h.getBasisInverseRow(i)
        if st1 != highspy.HighsStatus.kOk or st2 != highspy.HighsStatus.kOk:
            continue
        red = np.asarray(red, float)
        binv = np.asarray(binv, float)

        d0 = float(x[bv])
        f0 = d0 - np.floor(d0)
        if not (GMI_F0_MIN < f0 < 1.0 - GMI_F0_MIN):
            continue
        one_mf = 1.0 - f0

        alpha = np.zeros(n)
        beta = 0.0
        ok = True
        maxc, minc = 0.0, np.inf

        for j in range(n):
            if j == bv or abs(red[j]) < 1e-12:
                continue
            stj = col_st[j]
            if stj == k_basic:
                continue
            if stj == k_lower:
                t = red[j]
                shift = root.lb[j]
                sign = 1.0
            elif stj == k_upper:
                t = -red[j]
                shift = root.ub[j]
                sign = -1.0
            else:
                if abs(red[j]) > 1e-9:  # free nonbasic: derivation inapplicable
                    ok = False
                    break
                continue
            if not np.isfinite(shift):
                ok = False
                break
            if root.is_int[j]:
                fj = t - np.floor(t)
                g = fj if fj <= f0 else f0 * (1.0 - fj) / one_mf
            else:
                g = t if t > 0 else -t * f0 / one_mf
            if g < 1e-13:
                continue
            alpha[j] += -g * sign
            beta += -g * sign * shift
            maxc = max(maxc, abs(g))
            minc = min(minc, abs(g))
        if not ok:
            continue

        m_le = a_all.shape[0]
        for r in range(m_le):
            if abs(binv[r]) < 1e-12 or row_st[r] != k_upper:
                continue
            t = binv[r]
            g = t if t > 0 else -t * f0 / one_mf
            if g < 1e-13:
                continue
            alpha += g * a_all[r]
            beta += g * b_all[r]
            maxc = max(maxc, abs(g))
            minc = min(minc, abs(g))

        rhs = beta - f0
        nrm = float(np.linalg.norm(alpha))
        if nrm < 1e-10 or not np.isfinite(rhs):
            continue
        if maxc > 0 and minc < np.inf and maxc / max(minc, 1e-300) > GMI_DYNAMISM_MAX:
            continue
        rhs += GMI_RHS_SAFETY_REL * (1.0 + abs(rhs))  # roundoff safety relaxation
        if float(alpha @ x - rhs) < CUT_VIOL_TOL:
            continue
        cuts.append((alpha, float(rhs)))
        if len(cuts) >= max_cuts:
            break
    return cuts


# ── cut pool + hybrid selection ──────────────────────────────────────────────


class _CutPool:
    def __init__(self):
        self.pool = []
        self.seen = set()

    def offer(self, a, rhs):
        key = tuple(np.round(np.asarray(a, float), 5)) + (round(float(rhs), 5),)
        if key in self.seen:
            return
        self.seen.add(key)
        self.pool.append((np.asarray(a, float), float(rhs)))
        if len(self.pool) > POOL_MAX:
            self.pool = self.pool[-POOL_MAX:]

    def violated(self, x, tol=CUT_VIOL_TOL):
        return [(a, r) for a, r in self.pool if a @ x - r > tol]


def _select_cuts(candidates, x, top_k=SEL_TOP_K, par_max=SEL_PARALLEL_MAX):
    """Efficacy × orthogonality greedy selection (simplified SCIP hybrid rule)."""
    scored = []
    for a, rhs in candidates:
        nrm = float(np.linalg.norm(a))
        if nrm < 1e-12:
            continue
        eff = float(a @ x - rhs) / nrm
        if eff > 1e-9:
            scored.append((eff, a, rhs, nrm))
    scored.sort(key=lambda t: -t[0])
    chosen = []
    for eff, a, rhs, nrm in scored:
        if len(chosen) >= top_k:
            break
        if all(abs(a @ ca) / (nrm * cn) <= par_max for _e, ca, _r, cn in chosen):
            chosen.append((eff, a, rhs, nrm))
    return [(a, rhs) for _e, a, rhs, _n in chosen]


# ── driver ───────────────────────────────────────────────────────────────────


def generate_root_cuts(
    model, evaluator, lb, ub, is_int, is_bin, time_budget_s: float = 10.0
) -> RootCutResult:
    """Run the root cutting loop; return applied cuts + the final LP bound.

    Caller contract: the model is CONVEX-certified and routed to NLP-BB;
    ``lb``/``ub`` are the FBBT-tightened root bounds; ``is_int``/``is_bin``
    are flat masks. Raises on structural inapplicability (nonlinear objective,
    missing highspy) — the caller wraps and degrades to no-op.

    The returned ``cuts`` are only those BINDING at the final LP optimum (they
    are the ones carrying the bound); the rest served their purpose inside the
    loop. This keeps the constraint set added to the tree small — the full
    applied set (measured: ~170 dense rows on rsyn0805m) collapses node NLP
    throughput. ``time_budget_s`` bounds the stage's wall time.
    """
    import time as _time

    from discopt._jax.cmir_cuts import separate_cmir
    from discopt._jax.cover_cuts import separate_cover_cuts
    from discopt.modeling.core import ObjectiveSense

    sense_max = model._objective.sense == ObjectiveSense.MAXIMIZE
    root = _RootLP(model, evaluator, lb, ub, is_int, is_bin, sense_max)
    if not np.any(is_int):
        return RootCutResult()

    cuts_a: list = []
    cuts_b: list = []
    pool = _CutPool()

    def add_oa(x):
        added = 0
        for i, v in root.nonlinear_violations(x).items():
            if v > OA_TOL:
                tang = root.oa_tangent(i, x)
                if tang is None:
                    continue
                a, bb = tang
                cuts_a.append(a)
                cuts_b.append(bb)
                added += 1
        return added

    def oa_converge():
        obj, x, duals, h = _solve_lp(root, cuts_a, cuts_b)
        for _ in range(OA_MAX_ITERS):
            if x is None or add_oa(x) == 0:
                break
            obj, x, duals, h = _solve_lp(root, cuts_a, cuts_b)
        return obj, x, duals, h

    obj, x, duals, h = oa_converge()
    if x is None:
        return RootCutResult()
    b0 = obj  # OA-only root LP bound (pre-cut baseline for the quality gate)

    applied: list = []
    productive = 0
    rounds = 0
    t0 = _time.perf_counter()
    lb_s = np.where(np.isfinite(root.lb_sep), root.lb_sep, 0.0)
    ub_s = np.where(np.isfinite(root.ub_sep), root.ub_sep, 1e5)

    for _rnd in range(ROUNDS):
        if _time.perf_counter() - t0 > time_budget_s:
            break
        rounds += 1
        a_all = np.vstack([root.A_le] + ([np.array(cuts_a)] if cuts_a else []))
        b_all = np.concatenate([root.b_le] + ([np.array(cuts_b)] if cuts_b else []))

        cands = []
        try:
            cands += separate_cmir(
                a_all, b_all, x, lb_s, ub_s, is_int, max_cuts=MAX_CUTS_PER_FAMILY, duals=duals
            )
        except Exception as exc:  # pragma: no cover - separator robustness
            logger.debug("root-cuts: c-MIR separation failed: %s", exc)
        try:
            for cover, rhs in separate_cover_cuts(a_all, b_all, x, is_bin, max_cuts=32):
                a = np.zeros(root.n)
                for j in cover:
                    a[j] = 1.0
                cands.append((a, float(rhs)))
        except Exception as exc:  # pragma: no cover - separator robustness
            logger.debug("root-cuts: cover separation failed: %s", exc)
        if h is not None:
            cands += separate_gmi(root, h, x, a_all, b_all)

        for a, r in cands:
            a = np.asarray(a, float)
            if a @ x - float(r) > CUT_VIOL_TOL:
                pool.offer(a, float(r))
        chosen = _select_cuts(pool.violated(x), x)
        if chosen:
            productive += 1
        for a, r in chosen:
            cuts_a.append(a)
            cuts_b.append(r)
            applied.append((a, r))

        new_obj, x, duals, h = oa_converge()
        if x is None:
            break
        obj = new_obj
        if not chosen:
            break

    # Quality gate: keep the stage's output only when the cutting loop actually
    # MOVED the root LP bound past the OA-only baseline. On instances where the
    # OA outer approximation is structurally weak (measured: clay0303hfsg's
    # root LP bound is 0.0 vs opt 26669 — trivial), the cuts cannot help the
    # bound and their per-node row cost is pure regression; skipping them makes
    # the flag a no-op there (trivially sound).
    if obj is None or b0 is None:
        return RootCutResult(rounds_run=rounds, productive_rounds=productive)
    gain = (b0 - obj) if root.sense_max else (obj - b0)
    if gain <= 1e-6 * max(1.0, abs(b0)):
        return RootCutResult(rounds_run=rounds, productive_rounds=productive)

    # Keep only the cuts binding at the final LP optimum: they carry the final
    # bound; slack cuts only bloat every node NLP. (Validity is unaffected —
    # this merely drops rows.)
    if x is not None and applied:
        applied = [(a, r) for a, r in applied if abs(float(a @ x) - r) <= 1e-6 * (1.0 + abs(r))]

    return RootCutResult(
        cuts=applied,
        lp_bound=obj,
        rounds_run=rounds,
        productive_rounds=productive,
    )
