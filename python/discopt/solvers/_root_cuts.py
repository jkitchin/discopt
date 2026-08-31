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

Default **ON** since 2026-08-20. The bound-changing Regime-2 panel (CLAUDE.md
§5) passed both bars on 153 in-repo corpus instances -- cert-clean (0 bounds
above a reference optimum, 0 certification regressions, proven-optimal 69 both
arms) and net-positive (dual bound tighter on 22 / looser on 9, primal shortfall
better on 6 / worse on 1, total nodes -13.3%, wall +0.5%). The record is
``docs/dev/performance-plan.md`` §21; ``DISCOPT_NLPBB_ROOT_CUTS=0`` opts out and
takes the untouched legacy path.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Round cap. This is a runaway backstop, NOT the working terminator: the stage is
# bounded by its wall budget and by STALL_ROUNDS below. It used to be 30, which
# was the *binding* limit on exactly the instances the stage helps — measured on
# rsyn0840m the loop stopped on the cap with the bound improving in 30 of 30
# rounds and 43% of its wall budget still unspent (2778.18 -> 861.77 and still
# falling). Raising it only lets a *productive* loop keep going; an unproductive
# one now exits after STALL_ROUNDS.
ROUNDS = 200
# Consecutive rounds allowed to miss the quality gate's per-unit tolerance before
# the loop gives up. Measured separation on the probe set is total: the two
# instances the stage regressed (clay0205hfsg, clay0303hfsg) improved in 0 of 16
# and 0 of 19 rounds, with the root LP bound pinned at 0.0 from the first round;
# the two it helped improved in 18 of 20 and 30 of 30. There is no instance in
# between, so 2 is well clear of both. 2 rather than 1 because a round's cuts can
# need the NEXT re-convergence to pay off.
STALL_ROUNDS = 2
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
    """Whether the root cutting-plane stage runs (``DISCOPT_NLPBB_ROOT_CUTS``).

    **Default ON** since the 2026-08-20 graduation panel (issues #1061/#1062;
    ``docs/dev/performance-plan.md`` §21). Set the variable to ``0``/``off``/
    ``false``/``no`` to opt out and take the legacy no-root-cut path.

    Empty is deliberately NOT an off-value: it is what an unset variable expands
    to, and a graduated default-ON path must not be switched off by an accident
    of shell quoting (#993). Same shape as ``_gdp_config_primal_enabled``.
    """
    val = os.environ.get("DISCOPT_NLPBB_ROOT_CUTS", "").strip().lower()
    return val not in ("0", "false", "off", "no")


#: Whether an unset ``DISCOPT_ROOT_CUT_DEADLINE`` enforces the stage budget on
#: the individual LPs.
#:
#: **Default ON since the 2026-08-31 graduation panel (#1141; performance-plan
#: §25.9).** Set the variable to ``0``/``off``/``false``/``no`` to opt out and
#: take the legacy unbounded-LP path, which stays tested.
#:
#: Enforcing the budget changes which cuts the stage separates on an instance
#: whose OA prologue outruns it, hence that instance's root bound -- a
#: bound-changing knob (CLAUDE.md §5 regime 2), so it shipped default-off until
#: a panel cleared both bars. What the panel found:
#:
#: * *cert-clean*: 331 corpus checks (119 instances, interleaved, incumbents
#:   feasibility-verified) plus 40 stage-replay checks, 0 surviving violations.
#:   The single flagged row, ``clay0303hfsg``, is a time-limit boundary race --
#:   5 interleaved reps put ON ahead, ``optimal`` 4/5 against OFF's 3/5.
#: * *net-positive*: measured as CONTRACT ENFORCEMENT, which is what this flag
#:   is for. Worst-case overrun past the stage's own budget falls +0.297 s ->
#:   +0.076 s and runs overrunning by >20% fall 2/35 -> 0/35, while the root
#:   bound at the budgets ``solver.py`` actually hands the stage (2-10 s) is
#:   IDENTICAL on 3 of 4 instances and within 9.4e-6 on the fourth. Corpus wall
#:   is neutral (-0.1%), no certificate lost.
#:
#: Stated plainly, because the distinction matters: this is NOT the broad
#: corpus speed-up bar 2 normally asks for, and no such evidence exists here.
#: The class that would show it -- an instance whose prologue outruns the budget
#: at 2-10 s, measured on rsyn0830m in #1066 at 81.3 s of a 150 s solve -- is
#: not in the vendored corpus (0/119 deadline bites), so the corpus can neither
#: confirm nor falsify a speed-up. What is measured is that OFF leaves
#: ``generate_root_cuts``' docstring promise ("``time_budget_s`` bounds the
#: stage's wall time") false on a default-ON stage, and that closing it costs
#: nothing measurable. Deleting the flag -- #1141's only other permitted outcome
#: -- would delete the deadline mechanism and reopen #1066.
_ROOT_CUT_DEADLINE_DEFAULT = True


def _deadline_enabled() -> bool:
    """Is the #1066 per-LP stage deadline switched on (``DISCOPT_ROOT_CUT_DEADLINE``)?

    Default ON since #1141's graduation panel; ``0``/``false``/``off``/``no``
    opts out. With it off, ``generate_root_cuts`` behaves exactly as before: the
    initial OA convergence and every LP inside it run unbounded, and
    ``time_budget_s`` is consulted only between cut rounds. That legacy path
    stays tested.

    Empty is deliberately NOT an off-value: it is what an unset variable expands
    to, and a graduated default-ON path must not be switched off by an accident
    of shell quoting (#993) -- same shape as ``nlpbb_root_cuts_enabled``.

    Read per call, not cached at import, so a test can flip it without reloading
    the module and an A/B panel can drive both arms from one build.
    """
    raw = os.environ.get("DISCOPT_ROOT_CUT_DEADLINE")
    if raw is None or not raw.strip():
        return _ROOT_CUT_DEADLINE_DEFAULT
    return raw.strip().lower() not in ("0", "false", "off", "no")


def flat_column_terms(model) -> list:
    """One modeling term per *flat* column, in the model's own column order.

    ``generate_root_cuts`` returns each cut as a dense coefficient vector over
    flat columns. Writing that vector back as a ``Constraint`` needs an
    expression per column, and a block variable of size ``n`` occupies ``n``
    consecutive columns in C order -- the same flattening the evaluator and the
    FBBT bound vectors use (``np.concatenate([np.asarray(x[v.name]).ravel()
    for v in model._variables])``).

    Indexing ``model._variables`` directly by flat column is only correct when
    every block is scalar. The call site used to *gate on* that
    (``all(size == 1)``), which silently disabled the whole root-cut stage for
    any model built with vector variables -- a modeling style, not a problem
    class, so the restriction violated CLAUDE.md SS2. Unravelling here removes
    it: ``.nl`` and ``.gms`` imports (all-scalar blocks) are unchanged, and
    array-API models now get the same cuts.
    """
    cols: list = []
    for v in model._variables:
        size = int(getattr(v, "size", 1))
        if size == 1:
            cols.append(v)
            continue
        shape = getattr(v, "shape", None)
        if shape is None:
            shape = (size,)
        shape = tuple(int(d) for d in shape)
        if int(np.prod(shape)) != size:
            raise ValueError(
                f"variable {getattr(v, 'name', '?')!r}: shape {shape} does not "
                f"match size {size}; cannot map flat columns"
            )
        for k in range(size):
            idx = tuple(int(t) for t in np.unravel_index(k, shape))
            cols.append(v[idx[0] if len(idx) == 1 else idx])
    return cols


@dataclass
class RootCutResult:
    """Applied cuts (``alpha·x <= rhs`` each) and the final root-LP dual bound."""

    cuts: list = field(default_factory=list)  # [(alpha: np.ndarray, rhs: float)]
    lp_bound: float | None = None  # in the model's objective sense
    rounds_run: int = 0
    productive_rounds: int = 0
    #: Root LP bound after each round's OA re-convergence, in the model's sense.
    #: ``productive_rounds`` counts rounds that CHOSE cuts, which is not the same
    #: thing and is useless as a progress signal: measured on clay0205hfsg the
    #: loop chose cuts in 16 of 16 rounds and the end-of-loop quality gate then
    #: discarded every one. The trace is what says whether the bound MOVED.
    bound_trace: list = field(default_factory=list)
    #: Rounds whose bound gain cleared the quality gate's own per-unit tolerance.
    improving_rounds: int = 0
    #: Why the loop stopped: "budget" | "rounds" | "no_lp" | "no_cuts" | "stall".
    stop_reason: str = ""


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


def _solve_lp(root: _RootLP, cuts_a, cuts_b, time_limit: float | None = None):
    """Solve the root LP with HiGHS. Returns (obj_in_model_sense, x, duals, highs).

    ``time_limit`` (seconds) is handed to HiGHS so a single LP cannot outrun the
    stage budget. Measured on ``rsyn0830m`` (#1066): one unbounded call here
    burned 81.3 s of a 150 s solve against a 10 s stage budget. A run that stops
    on the limit returns a non-optimal status and is reported as a declined LP
    (all-``None``), exactly as any other non-optimal status already was.
    """
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
    if time_limit is not None:
        # Floor at a strictly positive value: HiGHS treats <= 0 as "no limit",
        # which would restore the very overrun this bounds.
        h.setOptionValue("time_limit", max(1e-3, float(time_limit)))
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

    # ``h``'s rows are [<= rows..., == rows...] and the row loop below pairs
    # ``binv[r]`` / ``row_st[r]`` with ``a_all[r]`` POSITIONALLY. A caller whose
    # row system is wider than the LP this basis was factorized from would
    # multiply an equality row's basis entry by a cut row and emit an INVALID
    # cut. It is an invariant of the caller, not a condition to tolerate, so
    # refuse loudly (CLAUDE.md SS3) rather than separate from a stale basis.
    n_le_basis = int(h.getNumRow()) - int(root.A_eq.shape[0])
    if int(a_all.shape[0]) != n_le_basis:
        raise ValueError(
            f"separate_gmi: row system has {int(a_all.shape[0])} '<=' rows but the "
            f"basis was factorized from {n_le_basis}; refusing to separate from a "
            "mismatched basis"
        )

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

    from discopt._relax.cmir_cuts import separate_cmir
    from discopt._relax.cover_cuts import separate_cover_cuts
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

    # The STAGE clock starts HERE, before the first OA convergence -- not after
    # it. #1066: the round loop's clock used to start below, so the whole
    # prologue ran outside the budget the docstring promises, and on rsyn0830m
    # that prologue took 81.3 s against a 10 s budget. ``_remaining`` is what
    # every LP below is bounded by. With the flag off, ``t_stage`` is read and
    # never used: the round loop keeps its own late clock (see ``t0``), so the
    # legacy path is unchanged to the instruction.
    t_stage = _time.perf_counter()

    def _remaining() -> float:
        return time_budget_s - (_time.perf_counter() - t_stage)

    def oa_converge():
        """Converge the OA loop; return the last LP that closed to OPTIMALITY.

        Two #1141 defects, both confined to the deadline arm, are fixed here.

        1. A declined LP used to discard the whole convergence -- including an
           EARLIER LP in the same call that had closed optimally. Under the
           deadline a decline is usually just a HiGHS time-limit stop, and
           ``x is None`` propagates to ``generate_root_cuts``' empty-result
           return, so ONE truncated LP at the end of a ~240-LP stage threw away
           every cut the stage had accumulated. That is the mechanism behind the
           ``tls2`` certification regression (performance-plan §25.9).
        2. The budget break below fires AFTER ``add_oa`` has appended tangent
           rows, so the returned basis was factorized from fewer ``<=`` rows
           than ``cuts_a`` now holds. ``separate_gmi`` pairs ``binv[r]``/
           ``row_st[r]`` with ``a_all[r]`` POSITIONALLY, so those extra rows
           read equality-row basis entries as if they were cut rows -- an
           invalid cut (or an ``IndexError``, depending on how many).

        Both are fixed by returning a state that is CONSISTENT: the last LP that
        reached optimality, together with exactly the rows it was solved from.
        That LP is an LP over a SUBSET of the OA rows, hence a relaxation, so its
        optimum is a valid root bound and any cut separated from it is valid.
        The rows added after it are dropped with it -- they are OA tangents we
        do not get to use, and dropping valid rows only ever weakens.

        The legacy arm is untouched, deliberately: with the flag off a decline is
        a structural or numerical LP failure rather than a budget stop, restoring
        an earlier solve would change the default path's cuts, and no basis
        mismatch can arise there (a declined LP returns ``h = None``, which the
        round loop already refuses to separate from).
        """
        deadline = _deadline_enabled()
        obj, x, duals, h = _solve_lp(root, cuts_a, cuts_b, _remaining() if deadline else None)
        # (bound, point, duals, basis, number of cut rows that basis was built from)
        last = (obj, x, duals, h, len(cuts_a)) if x is not None else None
        for _ in range(OA_MAX_ITERS):
            if x is None or add_oa(x) == 0:
                break
            if deadline and _remaining() <= 0.0:
                # Out of budget mid-convergence. The OA is incomplete, so the LP
                # is over FEWER constraints than the true feasible set -- a
                # relaxation of a relaxation. Its bound and every cut separated
                # from it stay valid; only their strength is given up.
                break
            obj, x, duals, h = _solve_lp(root, cuts_a, cuts_b, _remaining() if deadline else None)
            if x is not None:
                last = (obj, x, duals, h, len(cuts_a))
        if deadline and last is not None:
            # No-op on the converged path (``add_oa`` added nothing, so
            # ``n_keep == len(cuts_a)``); a rollback on the two truncated paths.
            obj, x, duals, h, n_keep = last
            del cuts_a[n_keep:]
            del cuts_b[n_keep:]
        return obj, x, duals, h

    obj, x, duals, h = oa_converge()
    if x is None:
        # Includes "the budget expired before the first LP closed": the stage
        # then contributes no cuts, which is always sound -- it is the same
        # outcome as the stage being switched off.
        return RootCutResult()
    b0 = obj  # OA-only root LP bound (pre-cut baseline for the quality gate)

    applied: list = []
    productive = 0
    rounds = 0
    trace: list = [b0]
    improving = 0
    stalled = 0
    stop_reason = "rounds"
    # Per-round progress is judged against the SAME tolerance the end-of-loop
    # quality gate uses, so "not improving" here means "on track to be
    # discarded there" rather than a second, unrelated notion of progress.
    gate_tol = 1e-6 * max(1.0, abs(b0))
    # Flag ON: the round loop is bounded by what is LEFT of the stage budget.
    # Flag OFF: it restarts the clock here, exactly as it always did.
    t0 = t_stage if _deadline_enabled() else _time.perf_counter()
    lb_s = np.where(np.isfinite(root.lb_sep), root.lb_sep, 0.0)
    ub_s = np.where(np.isfinite(root.ub_sep), root.ub_sep, 1e5)

    for _rnd in range(ROUNDS):
        if _time.perf_counter() - t0 > time_budget_s:
            stop_reason = "budget"
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

        new_obj, new_x, new_duals, new_h = oa_converge()
        if new_x is None:
            stop_reason = "no_lp"
            if not _deadline_enabled():
                x = None
            # Deadline arm: the PREVIOUS round's optimum is the last point this
            # stage actually proved anything at, so it -- not ``None`` -- is what
            # the binding filter below judges cuts against. Cuts chosen this
            # round were never solved against; they are violated at ``x``, so the
            # filter drops them, which is exactly the right outcome. Leaving
            # ``x`` at ``None`` instead SKIPS the filter and returns the whole
            # applied set, contradicting this function's own contract and
            # loading every node with rows no LP ever priced.
            break
        x, duals, h = new_x, new_duals, new_h
        prev_obj = obj
        obj = new_obj
        trace.append(obj)
        round_gain = (prev_obj - obj) if root.sense_max else (obj - prev_obj)
        if round_gain > gate_tol:
            improving += 1
            stalled = 0
        else:
            stalled += 1
        if not chosen:
            stop_reason = "no_cuts"
            break
        # Stall exit. Cuts keep being FOUND on these instances — clay0205hfsg
        # chose cuts in 16 of 16 rounds — they just never move the bound, so
        # `chosen` cannot detect it and the loop ran to budget exhaustion before
        # the end-of-loop gate discarded every cut. That cost a quarter of the
        # whole solve's time limit for nothing, which is the regression, not the
        # cuts' arithmetic (a valid cut cannot loosen a valid bound).
        if stalled >= STALL_ROUNDS:
            stop_reason = "stall"
            break

    # Quality gate: keep the stage's output only when the cutting loop actually
    # MOVED the root LP bound past the OA-only baseline. On instances where the
    # OA outer approximation is structurally weak (measured: clay0303hfsg's
    # root LP bound is 0.0 vs opt 26669 — trivial), the cuts cannot help the
    # bound and their per-node row cost is pure regression; skipping them makes
    # the flag a no-op there (trivially sound).
    if obj is None or b0 is None:
        return RootCutResult(
            rounds_run=rounds,
            productive_rounds=productive,
            bound_trace=trace,
            improving_rounds=improving,
            stop_reason=stop_reason,
        )
    gain = (b0 - obj) if root.sense_max else (obj - b0)
    if gain <= gate_tol:
        return RootCutResult(
            rounds_run=rounds,
            productive_rounds=productive,
            bound_trace=trace,
            improving_rounds=improving,
            stop_reason=stop_reason,
        )

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
        bound_trace=trace,
        improving_rounds=improving,
        stop_reason=stop_reason,
    )
