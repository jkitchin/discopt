"""
MILP Relaxation Builder for AMP (Adaptive Multivariate Partitioning).

Builds a linear programming relaxation of the original MINLP by:
  1. Replacing bilinear terms x_i*x_j with auxiliary variables w_ij and
     adding standard McCormick envelope constraints.
  2. Replacing monomial terms x_i^n with auxiliary variables s_i and adding
     piecewise tangent-cut underestimators plus partition-activated secant
     overestimators when the variable is discretized.
  3. Linearizing the original objective and constraints.

The LP relaxation gives a valid lower bound:
  LP_opt ≤ global NLP_opt

As the partition becomes finer (more intervals in disc_state), more tangent and
local secant cuts are added for monomials, tightening the lower bound.

Theory: Nagarajan et al., JOGO 2018, Section 4 (piecewise McCormick relaxation).
"""

from __future__ import annotations

import itertools
import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import scipy.sparse as sp

from discopt._relax._numeric import is_effectively_finite as _is_effectively_finite
from discopt._relax.discretization import DiscretizationState
from discopt._relax.model_utils import flat_variable_bounds
from discopt._relax.term_classifier import (
    NonlinearTerms,
    _compute_var_offset,
    _get_flat_index,
    distribute_products,
)
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Expression,
    FunctionCall,
    IndexExpression,
    Model,
    SumExpression,
    SumOverExpression,
    UnaryOp,
    Variable,
    VarType,
)
from discopt.solver_tuning import current as _tuning

logger = logging.getLogger(__name__)

# Dedupe identical warnings emitted across repeated relaxation builds (AMP iterates).
_warned_messages: set[str] = set()
_MAX_INTEGER_COS_ENUM = 10000
_MAX_FINITE_EXP_ARG = float(np.log(np.finfo(np.float64).max))
_MAX_TRIG_PIECEWISE_SPAN = 2.0 * math.pi
# Conditioning guard shared by the fractional-power envelopes (issue #158) and the
# lifted reciprocal/sqrt envelopes (issue #154). A power ``x**p`` with ``p<0`` (or
# ``0<p<1``) near a small lower bound, the convex ``1/g`` slope ``-1/g**2``, and the
# concave ``sqrt(g)`` slope ``1/(2*sqrt(g))`` all blow up (reaching ~1e9+) as the
# interval's lower end approaches zero. An LP row carrying such a coefficient against
# an RHS of order 1 is numerically unreliable: HiGHS returns a polytope that EXCLUDES
# feasible points (so OBBT shrinks a variable past its true feasible range and the
# per-node relaxer reports a feasible node "infeasible" — the nvs08 false optimum),
# or stalls at ``iteration_limit`` with a partial objective that is NOT a valid dual
# bound. Refuse to emit/lift any envelope cut whose slope exceeds this limit;
# dropping a cut only ENLARGES the relaxation, so abstention is always sound.
_LIFT_MAX_ENVELOPE_SLOPE = 1e6

# Largest argument magnitude a *cross-term* sqrt/reciprocal lift (issue #154,
# increment 2) may carry. Lifting ``sqrt(g)`` of a cross-term polynomial (e.g.
# nvs05/nvs22 ``sqrt(x4**2 + 2*x4*x5*x7 + x5**2)``) folds the product factors
# into McCormick aux columns whose bounds — and the resulting envelope row
# coefficients — scale with ``|g|``. When that magnitude reaches ~1e9 the LP is
# ill-conditioned: the fast simplex backend returns a wrong "optimal" bound that
# can exceed the true optimum (an unsound dual bound), while HiGHS still solves
# it correctly. Fast-simplex is demonstrated reliable up to ~1e7 on these
# relaxations, so cross-term lifts abstain above this limit. Abstention only
# drops a cut (the constraint is simply left un-lifted), which ENLARGES the
# relaxation and is therefore always sound.
_LIFT_MAX_CROSS_TERM_ARG_MAGNITUDE = 1e7

# Multi-argument atoms that are *jointly convex* over their domain and therefore
# admit a rigorous supporting-hyperplane (gradient) underestimator via the
# composite-multivariate lift. Maps atom name → required argument count. The
# convexity/domain licence is enforced by ``classify_expr`` in the collector, so
# this table only names the candidate atoms (a structural pre-filter). Currently
# just the GAMS relative-entropy intrinsic ``centropy(x, y) = x·log(x/y)`` (Boyd
# & Vandenberghe §3.2.6), which powers the MINLPLib ``ex6_2_*`` entropy family.
_JOINTLY_CONVEX_MULTIVAR_ATOMS: dict[str, int] = {"centropy": 2}


_MAX_TRIG_PIECEWISE_INTERVALS = 32
_MAX_TRIG_IMPORTED_BREAKPOINTS = _MAX_TRIG_PIECEWISE_INTERVALS + 1
_MAX_TRIG_PIECEWISE_WIDTH = math.pi / 6.0
_MAX_OBJECTIVE_LIFT_POWER = 6
_MAX_FINITE_DOMAIN_TRIG_TABLE_VALUES = 256

# A relaxation row coefficient / RHS, or a variable bound, at or above this
# magnitude is "numerically catastrophic": it comes from a McCormick/secant
# envelope over a variable with an enormous (or effectively infinite) range, and
# leaves the LP so ill-conditioned (dynamic range ~1e13+) that the backend
# (HiGHS) returns kSolveError and *no* bound at all. The cap sits above any
# legitimate modeling coefficient (big-M ~1e9, gear4's 1e6 linking term) and
# below the 1e11..1e37 entries such envelopes produce, so well-scaled models are
# never affected. Used only by ``sanitize_relaxation_for_conditioning``, which in
# turn only feeds the last-resort root-relaxation fallback bound — never the main
# solve — so dropping a borderline-large row can at most weaken that fallback,
# never the primary result. The recovered bound is empirically stable for caps
# from 1e11 down to 1e6 on the affected instances.
_RELAX_NUMERIC_CAP = 1e10

# Equilibrate the lifted relaxation before an external LP/MILP solve when its
# coefficient dynamic range exceeds this (matches the Rust simplex's own scaling
# trigger). The lifted McCormick rows of a product over a wide variable box mix
# tiny constants (~1e-9) with large bound-derived coefficients (~1e7), giving a
# >1e15 spread on ex1252's boundary sub-boxes — HiGHS stalls on it (a 452x96 LP
# hits its time limit) while the pure-Rust simplex, which equilibrates, solves it
# in ~0.03s. Geometric-mean row/column scaling brings the spread down so the
# external (HiGHS) path converges instead of timing out (issue #184).
_RELAX_EQUILIBRATE_TRIGGER = 1e6

# Coefficient-spread above which an ``infeasible`` verdict from the (numerically
# fragile) Rust simplex is *distrusted* and re-verified with exact equilibration.
# The Rust simplex's internal equilibration is insufficient for the lifted
# relaxation's worst conditioning — RLT cuts on a wide box yield degree-3 monomial
# coefficients with a ~1e5 spread, on which the simplex returns a *false*
# infeasible even though the LP (and HiGHS / the Python-equilibrated simplex) is
# feasible. A false-infeasible at a B&B node would prune the region containing the
# optimum, so re-solving is a soundness guard, not just speed. Set well below the
# (HiGHS-tuned) 1e6 trigger above because the simplex fails at lower spreads.
_RELAX_FALSE_INFEAS_TRIGGER = 1e3

# Smallest positive *normal* double (2.2250738585072014e-308). Coefficients below
# it are subnormal: they carry fewer significand bits than a normal double, so
# they are underflow artifacts rather than modelled quantities. The one that
# motivated this floor is ``±5e-324``, produced by outward-rounding an exact zero
# in the interval-arithmetic layer (#957) — see ``_coefficient_spread_exceeds``.
_SUBNORMAL_FLOOR = float(np.finfo(np.float64).tiny)


def _coefficient_spread_exceeds(data: np.ndarray, trigger: float) -> bool:
    """Is the nonzero-coefficient dynamic range of ``data`` above ``trigger``?

    Shared by the two conditioning guards below, both of which ask "is this LP
    badly enough scaled to be worth an exact equilibration". Two properties the
    naive ``nz.max() / nz.min() > trigger`` does not have (#957):

    * **It cannot overflow.** The cross-multiplied form is the identical boolean
      once the zeros are filtered (``min > 0``), but on the very spreads this
      guard exists for — ~1e26 over ~1e-300 — the division itself overflows to
      ``inf``, so the test answers ``True`` unconditionally while emitting a
      ``RuntimeWarning`` at the moment it makes its decision (#732 Stage 1-B
      already applied this form at the false-infeasible seam).
    * **A single subnormal cannot dominate it.** ``inf`` aside, one ``5e-324``
      entry drags ``nz.min()`` 300 orders of magnitude below every modelled
      coefficient and the answer becomes ``True`` regardless of how benign the
      matrix actually is. Filtering below :data:`_SUBNORMAL_FLOOR` measures the
      conditioning of the *model*, not of an underflow artifact.

    Both directions are sound whichever way this answers — equilibration is an
    exact, feasible-set-preserving rescale — so this only decides whether the
    work is worth doing, never what the LP means.
    """
    if data is None or data.size == 0:
        return False
    nz = np.abs(data[data != 0.0])
    nz = nz[nz >= _SUBNORMAL_FLOOR]
    if nz.size == 0 or not np.isfinite(nz).all():
        return False
    return bool(nz.max() > trigger * nz.min())


def equilibrate_relaxation_lp(
    c: np.ndarray,
    A_ub: Optional[Union[np.ndarray, sp.spmatrix]],
    b_ub: Optional[np.ndarray],
    bounds: list[tuple[float, float]],
    integrality: Optional[np.ndarray],
    *,
    iters: int = 20,
) -> tuple[
    np.ndarray,
    Optional[sp.spmatrix],
    Optional[np.ndarray],
    list[tuple[float, float]],
    np.ndarray,
]:
    """Geometric-mean (Ruiz) equilibration of ``min c·x s.t. A_ub x <= b_ub``.

    Alternating row/column infinity-norm sweeps drive every row and column to unit
    scale; the factors are snapped to powers of two so the transform is exact in
    floating point. **Integer columns are never scaled** (that would corrupt
    integrality), so the conditioning of the integer part is left to the solver.

    Soundness: this is an exact diagonal rescaling ``x = D·x'`` of the LP. Row
    scaling ``s_r`` rewrites ``A_r x <= b_r`` as ``(s_r A_r) x <= s_r b_r`` — same
    feasible set. Column scaling ``d_j`` substitutes ``x_j = d_j x'_j`` into the
    objective and rows consistently. The optimal objective value is therefore
    **unchanged**, so the LP/MILP bound from the scaled solve is the true bound;
    only the returned point needs ``x = col_scale · x'`` to map back. Returns
    ``(c', A', b', bounds', col_scale)``.
    """
    if A_ub is None or b_ub is None:
        return np.asarray(c, dtype=np.float64), A_ub, b_ub, bounds, np.ones(len(c))

    A = sp.csr_matrix(A_ub).astype(np.float64).copy()
    m, n = A.shape
    col_scale = np.ones(n)
    row_scale = np.ones(m)
    # Columns we are allowed to scale: continuous only (integer cols stay at 1).
    scalable = np.ones(n, dtype=bool) if integrality is None else (np.asarray(integrality) == 0)

    for _ in range(iters):
        absA = A.copy()
        absA.data = np.abs(absA.data)
        rmax = np.asarray(absA.max(axis=1).todense()).ravel()
        rmax[rmax == 0.0] = 1.0
        dr = 2.0 ** np.round(np.log2(1.0 / np.sqrt(rmax)))
        A = sp.diags(dr) @ A
        row_scale *= dr

        absA = A.copy()
        absA.data = np.abs(absA.data)
        cmax = np.asarray(absA.max(axis=0).todense()).ravel()
        cmax[cmax == 0.0] = 1.0
        dc = 2.0 ** np.round(np.log2(1.0 / np.sqrt(cmax)))
        dc[~scalable] = 1.0
        A = A @ sp.diags(dc)
        col_scale *= dc

    b2 = np.asarray(b_ub, dtype=np.float64) * row_scale
    c2 = np.asarray(c, dtype=np.float64) * col_scale
    bounds2 = [
        (
            lo / d if np.isfinite(lo) else lo,
            hi / d if np.isfinite(hi) else hi,
        )
        for (lo, hi), d in zip(bounds, col_scale)
    ]
    return c2, sp.csr_matrix(A), b2, bounds2, col_scale


# ---------------------------------------------------------------------------
# Result and model wrappers
# ---------------------------------------------------------------------------


@dataclass
class MilpRelaxationResult:
    """Result of solving a MILP relaxation.

    ``objective`` is the relaxation MILP's incumbent. ``bound`` is the rigorous
    dual lower bound on the relaxation optimum (hence on the original problem);
    it is the value AMP/OA-style callers must use as the global lower bound. It
    is ``None`` when no valid dual bound is available (or the relaxation
    objective is not itself a valid bound on the original).
    """

    status: str  # "optimal", "infeasible", "error", "time_limit"
    objective: Optional[float] = None
    bound: Optional[float] = None
    x: Optional[np.ndarray] = None
    # Pure-Rust certificate side-channel (issue #356), populated only on the
    # warm-started-simplex pure-LP path. ``safe_bound`` is a Neumaier–Shcherbina
    # safe lower bound from the simplex's own row duals — ``<=`` the true optimum
    # at any conditioning, so a caller can fathom on it without an independent
    # (HiGHS) cross-check. ``farkas_certified`` is ``True`` when an ``infeasible``
    # verdict was independently proven by a verified Farkas dual ray. Both default
    # to "unavailable" so the generic / MILP-B&B paths are unaffected.
    safe_bound: Optional[float] = None
    farkas_certified: bool = False
    # Node-LP marginals (cert:T2.4a / Phase 2), populated only when a solve is
    # requested with ``want_marginals=True`` on the direct warm-simplex pure-LP
    # path (``_solve_lp_warm``). ``row_dual`` is the row-dual vector ``y`` of the
    # standard-form ``A_ub x <= b_ub`` system; ``reduced_costs`` is ``d = c - Aᵀy``
    # over the FULL column set of THIS solve (the caller slices the first ``n_orig``
    # structural columns for DBBT). Both are on the ORIGINAL (un-equilibrated)
    # objective/constraint scale — the direct path never rescales — so they need no
    # unscaling. A pure read-only side-channel: nothing in the default path consumes
    # them, so populating them is bound-neutral. ``None`` on every other path
    # (equilibrated/generic/MILP-B&B), where DBBT simply no-ops (still sound).
    row_dual: Optional[np.ndarray] = None
    reduced_costs: Optional[np.ndarray] = None


def _lp_warm_deadline_enabled() -> bool:
    """Honour the caller's ``time_limit`` on the warm pure-LP node path.

    Tracking issue: **#928**. (This helper's panel script and companion test are
    named ``issue917_*`` / ``test_917_*`` for historical reasons only — #917 is a
    different, closed issue about the #844 LP-spatial reserve. Do not follow that
    number.)

    **The defect it fixes.** ``MilpRelaxationModel.solve`` takes a ``time_limit``,
    but its default ``backend="simplex"`` pure-LP fast path dropped it on the floor:
    ``_solve_lp_warm`` / ``_solve_lp_warm_equilibrated`` / ``solve_lp_warm_std`` took
    no deadline, and ``lp_bindings.rs`` hardcoded ``SimplexOptions { deadline: None }``
    — while the MILP route (``solve_milp_csc_py(time_limit_s=…)``) wired it up and the
    dual/primal pivot loops already poll it every 256 pivots. ~13 call sites in
    ``_relax/mccormick_lp.py`` plus ``lp_spatial_bb.py`` and ``integer_ratio.py`` compute
    a per-LP budget and pass it here, so the drop was general.

    Measured on nvs24 (``scratchpad/nvs24_arm.py``, ``nvs24_profile_evidence.txt``):
    ``solve(time_limit=0.202)`` -> ``_solve_lp_warm`` -> **47.03 s**, one
    ``DualPivotLoop`` 59 494 degenerate dual pivots deep with Bland never activated,
    turning a 3.9 s solve budget into 53 s (13.5x).

    **Bound-changing, so panel-gated; default OFF.** Opt in with
    ``DISCOPT_LP_WARM_DEADLINE=1``. Cutting an LP short changes the bound it returns,
    and on nvs24 the wall win is decisive:

    ======  ==============  ==============  ==============  ==============
    budget  wall OFF        wall ON         bound OFF       bound ON
    ======  ==============  ==============  ==============  ==============
    6 s     55.5 (9.25x)    28.9 (4.81x)    -56272.47       -56272.47
    30 s    67.8 (2.26x)    37.3 (1.24x)    -56272.47       -56272.47
    ======  ==============  ==============  ==============  ==============

    **Panel** (``issue917_lp_warm_deadline_panel.py``; 66 in-repo instances at a 15 s
    budget, OFF/ON interleaved per instance in isolated subprocesses;
    ``discopt_benchmarks/results/issue917_lp_warm_deadline_panel.json``):

    * *cert-clean*: **YES** — ``cert_regressions=0  lost_incumbents=0  lost_bound=0
      unsound=0  false_primals=0``. Bounds 3 tighter / 2 looser (contvar 171244.81 ->
      98924.53 and flay03m 46.51 -> 41.95 are the looser two; both sound).
    * *net-positive*: **not demonstrated** — total overrun 82.1 s -> 70.9 s (-14%),
      which sits inside the metric's own noise: three runs of the OFF arm alone gave
      82.1 / 79.2 / 175.6 s. Cells over budget went 15 -> 18.

    So it stays OFF: cert-clean is necessary, not sufficient (CLAUDE.md §5 bar 2, the
    ``DISCOPT_CUT_INHERIT`` rule).

    **The settling experiment has now been run, and it does not graduate the flag.**
    The paragraph here used to say what would settle it: *a load-gated, multi-rep panel
    over the instances where the deadline actually binds* — the corpus average being
    diluted by the ~50 instances that finish far inside the budget and are bit-identical
    either way. That panel is ``issue917_lp_warm_deadline_panel.py --instances`` over the
    17 non-closing in-repo instances, 3 reps at a 20 s budget, load-gated per CLAUDE.md §9
    (``scratchpad/issue917_bind.log``):

    ====  ============  ===========  ==========  =========
    rep   overrun OFF   overrun ON   cells OFF   cells ON
    ====  ============  ===========  ==========  =========
    1     30.9 s        27.7 s       7           5
    2     21.3 s        27.9 s       5           6
    3     29.6 s        35.3 s       7           5
    ====  ============  ===========  ==========  =========

    All three reps ``CERT_CLEAN=True`` (0 unsound / 0 lost_bound / 0 cert_regressions /
    0 lost_incumbents), so bar 1 holds on the binding subset too. Bar 2 does not: the
    paired ON-OFF deltas are -3.2 / +6.6 / +5.7 s — they *flip sign*, mean +3.0 s with
    sd 5.3 s, and the OFF arm alone spans 21.3-30.9 s. Concentrating the panel on the
    subset where the deadline binds did not resolve the effect out of the noise; if
    anything it leans the wrong way. **The flag stays OFF, and this is now a measured
    negative rather than an open question.**

    Two by-products of that run are worth keeping. First, ``nvs05`` and
    ``clay0303hfsg`` reported ``tighter_bound``/``looser_bound`` in *opposite
    directions between reps at identical flag state* — their bounds are
    timing-nondeterministic, so a single-rep bound delta on either is not evidence of
    anything. Second, on ``hda`` at ``time_limit=10`` (5 reps per arm,
    ``scratchpad/hda10.py``) the flag buys wall consistency at the price of the bound:

    ==============  ==================  ================================
    WARM_DEADLINE   wall                dual bound
    ==============  ==================  ================================
    0 (default)     12.03 s ± 1.23      -64473 in 3/5 reps, -141697 in 2
    1               11.14 s ± 0.04      -141697 in 5/5 reps
    ==============  ==================  ================================

    The OFF arm is bimodal because the #138 fallback's separated-relaxation phase is
    what overruns: when its grant is already spent by the time a first bound is in
    hand, ``_fb_stop`` returns the weak bound at ~10.7 s; when the grant still has room
    the phase starts and then ignores the budget passed to it — *this* drop — for a
    full ~4 s, reaching the full bound at ~12.9 s. Turning the flag on clamps that
    phase, which is why the wall tightens to ±0.04 s and why the bound is the weak one
    every time. Trading a rigorous bound for punctuality is the wrong trade under
    CLAUDE.md §1, which is the second, independent reason the flag is not the fix here.

    **What it took to become cert-clean** — the first two cuts were NOT, and both
    failures are worth keeping (CLAUDE.md §11):

    1. *The recovery cascade.* Without a ``_timed_out`` guard a deadline exit fell
       into the equilibrated retry and then the ~170x-slower cold ``solve_milp`` — the
       recovery meant for ill-conditioning. That *doubled* the corpus overrun the
       change exists to remove (175.6 s -> 310.3 s), all from one instance:
       heatexch_gen3 25.7 s -> **254.5 s** (1.7x -> 17.0x). With the guard, 30.7 s.
    2. *Honouring the deadline cost a BOUND.* bchoco08 went 1.0 -> ``None`` and
       contvar 171244.81 -> ``None`` — trading a budget overrun for a lost
       certificate, the wrong trade (CLAUDE.md §1). Three links had to change:
       :meth:`_stash_deadline_bound` banks the floor, ``mccormick_lp`` adopts it from
       a non-optimal node, and — the actual blocker — the Rust primal simplex now
       exports its ``y = B⁻ᵀc_B`` dual candidate on ``IterLimit`` instead of an empty
       vector, so there is a dual to build the floor from at all. Before that the
       Python plumbing was banking nothing: measured, ``solve_lp_warm_csc_py`` with
       ``time_limit_s=0.0`` returned ``dual=[]``.
    3. *And the panel could not see (2).* It compared bound quality only where BOTH
       arms were finite, so ``1.0 -> None`` scored as clean and it reported
       CERT_CLEAN=True twice. ``lost_bound``/``gained_bound`` exist because of that.

    **Update 2026-08-09: the deadline exit is now bound-PRESERVING, and the flag
    STILL does not graduate — but for a new, measured reason.** The #928 fix
    (``SimplexOptions::bank_deadline_duals``; set only by ``solve_lp_warm_csc_py``
    when a ``time_limit`` is passed, i.e. exactly this flag's path) makes a
    deadline exit return the dual loop's current dual-feasible ``y = B⁻ᵀc_B`` — the
    monotone best-so-far dual objective — instead of a cold-fallback initial-basis
    dual that evaluated to the trivial ``g(y=0)`` box floor. Measured on the hda
    separated-relaxation node LP: banked floor -141697 (= exactly ``g(0)``) at
    15/40/75% deadline fractions alike before; -118439/-118783 after (gap to the
    -64473 optimum cut ~30%). Whole-solve hda at ``time_limit=10``: the ON arm's
    bound went -141697 -> -123462 in 3/3 reps at 11.2-11.8 s wall. Flag-OFF
    bound-neutrality: 13 certifying instances byte-identical vs the merge-base
    (``scratchpad/issue928_neutrality.py``, marker-gated both directions).

    Seven §5 panel runs on that build, ALL cert-clean (0 cert_regressions /
    lost_incumbents / lost_bound / unsound in every run — the bound-losing trade
    recorded above is gone; contvar is now often TIGHTER than OFF, tspn12/tls2
    gained incumbents):

    * full corpus, 15 s, 1 run: total overrun 73.8 -> 31.6 s (-57%).
    * binding subset (19 non-closing instances), 15 s, 3 reps: ON-OFF deltas
      -2.5 / 0.0 / -8.2 s.
    * binding subset, 20 s, 3 reps: deltas **+325.4 / +68.5 / +12.7 s** — the ON
      arm is WORSE at this budget in 3/3 reps, including sporadic severe modes
      (contvar 500.6 s, bchoco08 80.9 s; the OFF arm has its own pre-existing
      class, heatexch_gen3 200.5 s in the same rep set).

    **Update 2026-08-11: the compounding this flag hit at the round level is
    fixed.** The coupled panel (performance-plan §14b) failed ``CERT_CLEAN`` on one
    item — contvar's bound going ``183632.766 -> None`` with this flag AND
    ``DISCOPT_NODE_ROUND_BUDGET`` on — and attributed it to the *interaction*, not
    to either flag alone. The mechanism is now measured directly: hand
    ``solve_at_node`` a round grant that is already spent and **16 of 114 cells of
    the binding subset return no bound at all** where the same box under an
    unclamped round certifies one, in BOTH arms of this flag
    (``scratchpad/issue928_round_cut_short_entry.py``) — the deadline-truncated
    build leaves a 0-row relaxation that solves to LP optimality and that every
    certification route then declines. In every one of those cells the relaxation
    still carried a valid finite box-interval objective floor, equal to the
    unclamped control bound on 4 of the 5 instances
    (``scratchpad/issue928_floor_inventory.py``). ``_solve_at_node_impl`` now
    reports that floor — the round-level analogue of what this flag's own
    ``_stash_deadline_bound`` does for an LP — and takes the tighter of it and any
    banked dual; the #966 round-admission check no longer declines the ROOT round,
    which holds no parent bound to fall back on.

    **The re-panel on that build passes the net-positive bar and still does not
    graduate** (19 binding instances, 20 s, 3 reps, three arms;
    ``discopt_benchmarks/results/issue928_round_floor_binding20.json``,
    performance-plan §14d). ON-OFF overrun is **-94.4 / -313.5 / -16.7 s** — negative
    in 3/3 where §14b measured +325.4/+68.5/+12.7, so the sign flip is gone — with
    ``lost_bound`` empty in 3/3 (contvar's ``-> None`` does not recur), one bound
    GAINED (clay0303hfsg ``None -> -1.23e-05``), 7-8 tighter against 1-2 tiny looser
    per rep, hda ``-2.07e13 -> -124296.9``, and zero soundness violations over 63
    counted bound-vs-ceiling comparisons. ``CERT_CLEAN`` is False in 2 of 3 reps on
    exactly one item — tspn12's incumbent — and that item is the ROUND budget's, not
    this flag's: ``seam vs base`` loses it too while ``cand vs seam`` loses no
    incumbent in any rep. Bar 1 has no slack, so all three flags stay OFF pending
    that #966 residual.

    Net-positive therefore FAILED at the time of writing: the sign flips with
    budget, which is not
    "measurably helpful broadly" (the ``DISCOPT_CUT_INHERIT`` rule). The residual
    is measurably NOT this seam any more: per-LP deadline violations are zero
    across every probe, and on contvar the ON arm spends 2.8 s in 30 relaxation
    solves against a 27-33 s wall (OFF: 6.3 s in 20 solves, ~22 s wall) — the
    budget-honoring LPs let the enclosing separation loop fit ~10 more rounds
    whose non-LP cost (~1.1 s relaxation build each) is not clamped by the
    round's grant, and downstream phases with coarse global-deadline compliance
    do the rest (``scratchpad/issue928_{contvar_probe,blowup_hunt}.py``,
    ``discopt_benchmarks/results/issue928_*.json``). Fixing THAT is caller-side
    budget accounting — a different seam with its own issue; this flag stays OFF
    until it lands and a re-run panel passes both bars.

    **Update 2026-08-12: GRADUATED to default-ON.** Every panel above shares one
    defect: no arm ever carried this flag *alone*. ``f2565241``'s arms are ``base``
    (all OFF) / ``seam`` (#966's two ON, this one OFF) / ``cand`` (all three), so
    ``cand - seam`` isolates this flag only under the counterfactual that #966's two
    are ON — and #966's own panel kept them OFF. The §14d bar-1 failure was already
    attributed to the ROUND budget rather than to here (``seam vs base`` loses tspn12's
    incumbent too, ``cand vs seam`` loses none), which is a hypothesis the three-arm
    design cannot itself test. The missing arm was run: ``base`` vs ``warm``
    (``DISCOPT_LP_WARM_DEADLINE=1``, the other two explicitly ``0``), 19 binding
    instances, 20 s, arms interleaved per instance in isolated subprocesses
    (``issue928_rate_score.py``, ``results/issue928_warmalone20_rep{1,2,3}.json``).

    *Bar 1 — PASS.* 0 unsound / 0 incumbent-verification failures / 0 certification
    regressions / 0 lost incumbents / 0 lost bounds, in 3/3 reps, over **96 executed
    oracle comparisons** against ``minlplib.solu`` (the earlier container runs reached
    1 of 19 instances via the narrow ``_optima`` fallback, so their soundness line
    rested on almost nothing). The §14d tspn12 incumbent loss does NOT recur without
    the round budget, which confirms that attribution.

    *Bar 2 — PASS.* Wall: overrun delta -2.6 / -1.0 / -0.3 s, negative in 3/3 (total
    overrun 6.5/4.8/4.0 s -> 3.9/3.8/3.7 s); the sign no longer flips with budget.
    Nodes: 4647 -> 4611, neutral. Bound: the two cells that move are both stable
    GAINS — hda ``-2.07e13 -> -64473.44`` in 3/3 (identical to 8 digits; the base
    arm's value is not a usable bound at all) and heatexch_gen1 ``46750/44196 ->
    49000``, where the ON arm is also the reproducible one. The three cells scored
    "looser" are the known timing-nondeterministic class: tspn10 by 6e-8 relative,
    and tspn08/tspn12 flip direction between reps *in both arms* (the base arm alone
    reports tspn12 as 202.181 in two reps and 202.418 in the third) — the
    nvs05/clay0303hfsg signature recorded above, not a flag effect.

    Scored by per-cell RATE over reps, not one-shot: the bar-1 failures that kept this
    flag OFF in earlier panels (nvs05, tls2, syn05hfsg, casctanks) are cells *neither*
    arm holds reliably, and one-shot scoring cannot separate a flag effect from a race.
    Deviation from the pre-registration, recorded per §11: it fixed 5 reps and 3 were
    run (the run was cut short); the regression margin is 2 of 3, proportionally at or
    above the pre-registered 3 of 5. The ``=0`` opt-out and the legacy no-deadline path
    are unchanged.
    """
    import os as _os

    return _os.environ.get("DISCOPT_LP_WARM_DEADLINE", "1") not in (
        "0",
        "",
        "false",
        "False",
        "off",
    )


class MilpRelaxationModel:
    """Wrapper around a MILP that exposes a .solve() method.

    Stores the LP data and delegates solving to solve_milp (HiGHS).
    """

    def __init__(
        self,
        c: np.ndarray,
        A_ub: Optional[Union[np.ndarray, sp.spmatrix]],
        b_ub: Optional[np.ndarray],
        bounds: list[tuple[float, float]],
        obj_offset: float = 0.0,
        integrality: Optional[np.ndarray] = None,
        objective_bound_valid: bool = True,
    ):
        self._c = c
        self._A_ub = A_ub
        self._b_ub = b_ub
        self._bounds = bounds
        self._obj_offset = obj_offset
        self._integrality = integrality
        self._objective_bound_valid = objective_bound_valid
        # Rigorous box-interval objective floor (#640 Bucket 2, nvs22); set by
        # ``build_uniform_relaxation``. ``None`` unless a finite floor was computed.
        self._objective_floor: Optional[float] = None
        # Issue #694 anytime-build provenance, set by ``build_uniform_relaxation``.
        # ``_build_truncated`` is True when the constraint-row loop stopped early on
        # a ``build_deadline`` (the relaxation is still a valid, weaker outer
        # approximation). Default: a full, un-truncated build.
        self._build_truncated: bool = False
        self._build_constraints_done: Optional[int] = None
        self._build_constraints_total: Optional[int] = None
        # Warm-start state for the pure-LP simplex fast path (cutting-plane loop):
        # the previous solve's optimal basis and the (structural-cols, rows) it was
        # produced at, so the next ``.solve()`` on the SAME columns with rows only
        # appended can dual-simplex re-optimize from it. See ``_solve_lp_warm``.
        self._warm_basis: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._warm_struct_n: Optional[int] = None
        self._warm_rows: Optional[int] = None

    def solve(
        self,
        time_limit: Optional[float] = None,
        gap_tolerance: float = 1e-4,
        backend: str = "auto",
        *,
        want_marginals: bool = False,
    ) -> MilpRelaxationResult:
        from discopt.solvers import SolveStatus
        from discopt.solvers.lp_backend import get_milp_solver

        # #517 last-resort dual floor (flag-gated, default OFF): reset per solve.
        # A numerically-failed node LP may still yield a sound Neumaier–Shcherbina
        # safe bound from the in-house simplex's own dual; the warm/equilibrated
        # paths stash it here and it is attached below only if nothing else produced
        # a bound (the hda-class no-bound nodes).
        self._pending_numerical_bound: Optional[float] = None
        # Neumaier-Shcherbina floor recovered from an LP that YIELDED on the shared
        # deadline, kept separately from the #517 numerical-failure stash above so
        # this path cannot change #517's behaviour. Consumed only by
        # ``_time_limit_result``, i.e. only when the budget is genuinely spent.
        self._pending_deadline_bound: Optional[float] = None

        # ONE absolute budget for this call, shared by every attempt below
        # (``_lp_warm_deadline_enabled``). ``time_limit`` is a duration and this
        # method may try up to three solves — warm, equilibrated, then the
        # generic/cold path — so handing each a fresh copy of the caller's duration
        # would silently triple it; spending a single deadline keeps the sum inside
        # what the caller asked for.
        #
        # With the flag OFF this reproduces the historical behaviour exactly: the
        # warm attempts get no limit at all and the cold path gets a fresh copy of
        # the caller's duration.
        _budget_t0 = time.perf_counter()
        _warm_deadline = _lp_warm_deadline_enabled()

        def _remaining() -> Optional[float]:
            if time_limit is None:
                return None
            if not _warm_deadline:
                # Pre-flag behaviour: each attempt gets the full duration afresh.
                return float(time_limit)
            return max(0.0, float(time_limit) - (time.perf_counter() - _budget_t0))

        def _warm_budget() -> Optional[float]:
            return _remaining() if _warm_deadline else None

        def _timed_out() -> bool:
            """The shared budget is spent (only reachable with the flag on)."""
            if not _warm_deadline or time_limit is None:
                return False
            left = _remaining()
            # ``_remaining`` is Optional only because ``time_limit`` may be None, which
            # the guard above has already excluded; bind it so the comparison is typed.
            return left is not None and left <= 0.0

        # Warm-startable pure-LP fast path: the spatial cut-separation loop
        # re-solves the SAME structural columns with only rows (cuts) appended, so
        # the previous optimal basis is an ideal dual-simplex warm start. Engage
        # only for the Rust simplex backend on a pure LP (no integrality). A
        # bad/mismatched basis is ignored inside Rust (cold fallback) and the dual
        # simplex converges to the same LP optimum, so the bound is unchanged --
        # warm-start only changes speed. Disable with ``DISCOPT_LP_WARMSTART=0``.
        if (
            backend == "simplex"
            and self._integrality is None
            and self._A_ub is not None
            and _tuning().lp_warmstart
        ):
            warm = self._solve_lp_warm(want_marginals=want_marginals, time_limit=_warm_budget())
            # A warm-start ``infeasible`` on an ill-conditioned LP can be a
            # numerical false-negative; fall through to the equilibrated re-verify
            # below rather than trust it (a false-infeasible would unsoundly prune
            # a B&B node). Any other warm verdict is the true LP optimum.
            if warm is not None and warm.status != "infeasible":
                return warm
            # The bare warm simplex returned ``None`` (iter-limit/numerical) or a
            # possibly-false ``infeasible`` on a badly-scaled LP. Retry with the
            # SAME fast warm simplex on the geometric-mean-equilibrated LP — an
            # exact, feasible-set-preserving rescale that yields the identical
            # optimum (verified equal to the old cold ``solve_milp`` path) at warm
            # speed. This replaces the 170x-slower cold MILP-B&B fallthrough that
            # used to handle these ill-conditioned relaxation solves (nvs21).
            # A deadline exit is NOT a numerical failure and must not trigger the
            # recovery cascade below. The equilibrated retry and the cold
            # ``solve_milp`` fallthrough exist for ill-conditioning — the cold one is
            # ~170x slower (see ``_solve_lp_warm_equilibrated``) — and re-running the
            # same LP on a budget that is already spent can only overshoot further.
            # The warm attempt is handed the WHOLE remaining budget, so its yielding
            # means the budget is gone; stop here and report the limit.
            #
            # Measured: without this guard the first cut of this change doubled the
            # corpus overrun it was meant to remove (175.6 s -> 310.3 s over 66
            # instances at a 15 s budget), and every second of that came from ONE
            # instance — heatexch_gen3 at 25.7 s -> 254.5 s (1.7x -> 17.0x), each
            # timed-out node LP cascading into the cold path. Every other instance
            # improved or was neutral.
            if _timed_out():
                return self._time_limit_result()
            equil = self._solve_lp_warm_equilibrated(time_limit=_warm_budget())
            if equil is not None and equil.status in ("optimal", "infeasible", "unbounded"):
                return equil
            if _timed_out():
                return self._time_limit_result()

        # Same rule at the last seam: the generic/cold path is the most expensive of
        # the three, so entering it on a spent budget is the worst possible use of an
        # overrun. (Unreachable unless the warm path declined for a NON-deadline
        # reason and the budget expired meanwhile.)
        if _timed_out():
            return self._time_limit_result()

        # backend="auto": HiGHS if present, else POUNCE. backend="simplex" routes
        # to the warm-started-simplex B&B (falls back to auto if unavailable).
        solve_milp = get_milp_solver(backend=backend)

        # Equilibrate the LP for the external (HiGHS/POUNCE) backends when it is
        # badly scaled. The pure-Rust simplex already equilibrates internally, so
        # skip the (redundant) Python pre-scaling there; HiGHS does not cope with
        # the lifted relaxation's >1e15 coefficient spread and stalls without it
        # (issue #184). The transform is exact, so the returned bound/objective is
        # unchanged — only the solution point is mapped back through ``col_scale``.
        c_s, A_s, b_s, bounds_s = self._c, self._A_ub, self._b_ub, self._bounds
        col_scale = None
        if backend != "simplex" and self._A_ub is not None:
            data = sp.csr_matrix(self._A_ub).data
            if _coefficient_spread_exceeds(data, _RELAX_EQUILIBRATE_TRIGGER):
                c_s, A_s, b_s, bounds_s, col_scale = equilibrate_relaxation_lp(
                    self._c, self._A_ub, self._b_ub, self._bounds, self._integrality
                )

        result = solve_milp(
            c=c_s,
            A_ub=A_s,
            b_ub=b_s,
            bounds=bounds_s,
            integrality=self._integrality,
            # What is LEFT of the caller's budget, not a fresh copy of it: the warm
            # and equilibrated attempts above may already have spent some of it.
            # (Flag OFF: a fresh copy, as before.)
            time_limit=_remaining(),
            gap_tolerance=gap_tolerance,
        )

        # Re-verify a possible *false* infeasible. The Rust simplex can report an
        # ill-conditioned relaxation LP infeasible when it is actually feasible
        # (RLT cuts on a wide box produce a ~1e5 coefficient spread on which the
        # simplex's internal scaling is insufficient). Accepting that verdict at a
        # B&B node would prune a region that may contain the optimum — unsound. The
        # raw solve was NOT Python-equilibrated above for ``backend="simplex"``, so
        # re-solve once with exact geometric-mean equilibration before accepting
        # ``infeasible``. Equilibration preserves the feasible set exactly, so a
        # genuinely infeasible LP stays infeasible; only a numerical false-negative
        # flips to optimal.
        if result.status == SolveStatus.INFEASIBLE and col_scale is None and self._A_ub is not None:
            # ``_coefficient_spread_exceeds`` carries the overflow-free
            # cross-multiplied form this seam has used since #732 Stage 1-B, plus
            # the subnormal floor from #957.
            _data = sp.csr_matrix(self._A_ub).data
            if _coefficient_spread_exceeds(_data, _RELAX_FALSE_INFEAS_TRIGGER):
                c_s, A_s, b_s, bounds_s, col_scale = equilibrate_relaxation_lp(
                    self._c, self._A_ub, self._b_ub, self._bounds, self._integrality
                )
                result = solve_milp(
                    c=c_s,
                    A_ub=A_s,
                    b_ub=b_s,
                    bounds=bounds_s,
                    integrality=self._integrality,
                    time_limit=time_limit,
                    gap_tolerance=gap_tolerance,
                )

        # Map the scaled solution point back to original variables (x = D·x').
        if col_scale is not None and result.x is not None:
            result.x = np.asarray(result.x, dtype=np.float64) * col_scale

        # Map SolveStatus enum to string
        status_map = {
            SolveStatus.OPTIMAL: "optimal",
            SolveStatus.INFEASIBLE: "infeasible",
            SolveStatus.UNBOUNDED: "unbounded",
            SolveStatus.TIME_LIMIT: "time_limit",
            SolveStatus.ITERATION_LIMIT: "iteration_limit",
            SolveStatus.ERROR: "error",
        }
        status_str = status_map.get(result.status, str(result.status))

        # A non-finite objective/bound is never a valid dual bound: report no
        # bound (None) rather than propagate a NaN/inf that would silently pass
        # every ``bound <= incumbent`` comparison (NaN compares false) and corrupt
        # fathoming. Seen on a raw, un-presolved lifted relaxation whose subnormal
        # aux bounds / +inf originals let the LP optimum's objective evaluate to
        # NaN while the solver still reports ``optimal`` (#649). Refuse loudly-by-
        # omission here; the caller then treats the node as unbounded-below (no
        # prune), which is sound.
        obj = None
        if (
            result.objective is not None
            and self._objective_bound_valid
            and math.isfinite(float(result.objective))
        ):
            obj = float(result.objective) + self._obj_offset

        # The sound lower bound on the original problem is the MILP's dual bound
        # (not its incumbent), and only when this relaxation's objective is a
        # valid bound on the original (``_objective_bound_valid``).
        bound = None
        if (
            result.bound is not None
            and self._objective_bound_valid
            and math.isfinite(float(result.bound))
        ):
            bound = float(result.bound) + self._obj_offset

        # #517 last-resort dual floor: if the whole in-house chain produced no
        # bound but a numerically-failed node LP yielded a sound NS safe bound
        # (from the engine's own dual), use it (flag-gated). Never overrides a real
        # bound — it only fills a ``None`` — and never fathoms on its own.
        if (
            bound is None
            and _tuning().node_numerical_dual_bound
            and self._pending_numerical_bound is not None
        ):
            bound = self._pending_numerical_bound

        # #362 (same flag): surface the stashed NS bound as ``safe_bound`` on an
        # ``optimal`` generic-path solve. The generic MILP path computes no
        # certificate of its own, so before this a node whose warm/equilibrated
        # simplex attempts broke down numerically — but whose duals already yielded
        # a rigorous Neumaier–Shcherbina bound on this very LP — reached
        # ``_certify`` with ``safe_bound=None`` and was DECLINED by the
        # conditioning guard, leaving the node's failure sentinel to non-rigorously
        # fathom it (the nvs05 taint at the certification edge: LP optimum 5.47073,
        # discarded NS bound 5.46581, taint floor 2.4e-4 below the incumbent). The
        # NS bound is valid for ANY multiplier vector by weak duality, so a
        # drifted-basis dual only loosens it — it can never exceed the true LP
        # optimum — and a finite value is itself a proof the LP is bounded, so this
        # can never fabricate a bound on a genuinely unbounded relaxation
        # (himmel16 class). Attached only on ``optimal`` (elsewhere the ``bound``
        # fill above already carries it; ``_certify`` only reads ``safe_bound``
        # from optimal results).
        safe_bound = None
        if (
            status_str == "optimal"
            and _tuning().node_numerical_dual_bound
            and self._pending_numerical_bound is not None
        ):
            safe_bound = self._pending_numerical_bound

        return MilpRelaxationResult(
            status=status_str, objective=obj, bound=bound, x=result.x, safe_bound=safe_bound
        )

    def _stash_deadline_bound(self, cert) -> None:
        """Record the NS safe bound of an LP that yielded, for the deadline path.

        Deliberately NOT gated on ``node_numerical_dual_bound`` (#517). That flag
        guards using a *numerically broken* solve's dual as a last-resort floor; this
        stash is consumed only by :meth:`_time_limit_result`, i.e. only when the LP
        was cut short by a deadline the caller set. The soundness argument is the
        stronger one and is unconditional: ``g(y)`` is a valid lower bound for ANY
        multiplier vector by weak duality, so stopping the simplex early can only
        make the recovered floor looser — never lift it above the true optimum.

        Without this, honouring the deadline COSTS a bound outright: measured at a
        15 s budget, bchoco08 went from ``bound=1.0`` to ``bound=None`` and contvar
        from 171244.81 to ``None`` — trading a budget overrun for a lost certificate,
        which is the wrong trade (CLAUDE.md §1).
        """
        if cert is None or getattr(cert, "safe_bound", None) is None:
            return
        if not self._objective_bound_valid:
            return
        sb = float(cert.safe_bound) + self._obj_offset
        if not math.isfinite(sb):
            return
        prev = self._pending_deadline_bound
        self._pending_deadline_bound = sb if prev is None else max(prev, sb)

    def _time_limit_result(self) -> "MilpRelaxationResult":
        """The result of a solve whose shared budget ran out inside the warm path.

        Carries no incumbent and no LP optimum — the LP never finished, so claiming
        either would be a fabricated bound. It DOES surface the #517 stashed
        Neumaier-Shcherbina floor when one was recovered (flag-gated, same rule as
        the generic path below): that bound comes from the engine's own dual and is
        valid for any multiplier vector by weak duality, so a yielded LP cannot make
        it unsound — a shorter solve only loosens it.
        """
        bound = self._pending_deadline_bound
        if bound is None and (
            _tuning().node_numerical_dual_bound and self._pending_numerical_bound is not None
        ):
            bound = self._pending_numerical_bound
        return MilpRelaxationResult(
            status="time_limit", objective=None, bound=bound, x=None, safe_bound=bound
        )

    def _solve_lp_warm(
        self, *, want_marginals: bool = False, time_limit: Optional[float] = None
    ) -> Optional["MilpRelaxationResult"]:
        """Pure-LP warm-started re-solve via the Rust dual simplex.

        Reuses the cached optimal basis from the previous ``.solve()`` when the
        structural column set is unchanged and rows have only grown (the
        cutting-plane case), extending it for the appended slacks. Returns the
        mapped :class:`MilpRelaxationResult`, or ``None`` to defer to the generic
        path (binding unavailable, or an ``iter_limit``/``numerical`` exit). The
        returned objective/bound is the true LP optimum — warm-start is a pure
        speed optimization, never a correctness one.

        When ``want_marginals`` is set and the solve is optimal, the returned
        result additionally carries ``row_dual`` (``y``) and ``reduced_costs``
        (``d = c - Aᵀy`` over the full column set), both on the original scale (this
        path never rescales). Pure side-channel — never affects the bound/status.
        """
        from discopt.solvers import SolveStatus

        try:
            from discopt.solvers.milp_simplex import solve_lp_warm_std
        except Exception:  # pragma: no cover - binding absent
            return None

        n_struct = np.asarray(self._c, dtype=np.float64).ravel().shape[0]
        m_now = 0 if self._A_ub is None else sp.csr_matrix(self._A_ub).shape[0]
        in_basis = None
        if (
            self._warm_basis is not None
            and self._warm_struct_n == n_struct
            and self._warm_rows is not None
            and self._warm_rows <= m_now
        ):
            in_basis = self._warm_basis

        try:
            result, out_basis, cert = solve_lp_warm_std(
                self._c,
                self._A_ub,
                self._b_ub,
                self._bounds,
                in_basis=in_basis,
                return_cert=True,
                time_limit=time_limit,
            )
        except Exception:  # pragma: no cover - defensive; fall back to generic path
            return None
        if result is None:
            # iter_limit / numerical: let the generic path (with its HiGHS option)
            # handle it; drop the stale basis so the next round cold-starts.
            self._stash_numerical_bound(cert)  # #517 last-resort floor (flag-gated)
            self._stash_deadline_bound(cert)
            self._warm_basis = None
            return None
        if out_basis is not None:
            self._warm_basis = out_basis
            self._warm_struct_n = n_struct
            self._warm_rows = m_now
        else:
            self._warm_basis = None

        status_map = {
            SolveStatus.OPTIMAL: "optimal",
            SolveStatus.INFEASIBLE: "infeasible",
            SolveStatus.UNBOUNDED: "unbounded",
            SolveStatus.TIME_LIMIT: "time_limit",
            SolveStatus.ITERATION_LIMIT: "iteration_limit",
            SolveStatus.ERROR: "error",
        }
        status_str = status_map.get(result.status, str(result.status))
        obj = None
        if result.objective is not None and self._objective_bound_valid:
            obj = float(result.objective) + self._obj_offset
        bound = None
        if result.bound is not None and self._objective_bound_valid:
            bound = float(result.bound) + self._obj_offset
        safe_bound = None
        if cert.safe_bound is not None and self._objective_bound_valid:
            safe_bound = float(cert.safe_bound) + self._obj_offset
        row_dual = None
        reduced_costs = None
        if (
            want_marginals
            and status_str == "optimal"
            and self._objective_bound_valid
            and getattr(cert, "dual", None) is not None
        ):
            # d = c - Aᵀy over the FULL column set of this (original-scale) LP. The
            # objective is stored scaled by ``_obj_sign``/offset; reduced costs are a
            # property of the constraint geometry + objective *direction*, and DBBT
            # only reads their SIGN and magnitude relative to the same-scale bound
            # gap, so compute them on the stored ``self._c`` (the solver's own
            # objective row) to stay self-consistent with ``safe_bound``. A shape
            # mismatch (cuts changed the column count vs the cached ``_c``) yields
            # ``None`` — sound, DBBT no-ops. Never raises into the solve path.
            try:
                y = np.asarray(cert.dual, dtype=np.float64).ravel()
                c_vec = np.asarray(self._c, dtype=np.float64).ravel()
                A_csr = sp.csr_matrix(self._A_ub)
                if A_csr.shape[0] == y.shape[0] and A_csr.shape[1] == c_vec.shape[0]:
                    d = c_vec - np.asarray(A_csr.T @ y, dtype=np.float64).ravel()
                    if np.all(np.isfinite(d)):
                        row_dual = y
                        reduced_costs = d
            except Exception:  # pragma: no cover - defensive; marginals are optional
                row_dual = None
                reduced_costs = None
        return MilpRelaxationResult(
            status=status_str,
            objective=obj,
            bound=bound,
            x=result.x,
            safe_bound=safe_bound,
            farkas_certified=bool(cert.farkas_certified),
            row_dual=row_dual,
            reduced_costs=reduced_costs,
        )

    def _solve_lp_warm_equilibrated(
        self, *, time_limit: Optional[float] = None
    ) -> Optional["MilpRelaxationResult"]:
        """Warm-simplex re-solve on the *equilibrated* LP.

        The bare warm simplex (:meth:`_solve_lp_warm`) returns ``None`` /
        false-``infeasible`` on a badly-scaled relaxation (the lifted McCormick
        envelope of a high-degree term spans many orders of magnitude — nvs21's
        ``x1**4`` reaches ~1e9). The legacy fallback then cold-solved the same LP
        through the MILP-B&B entry (``solve_milp``) — same Rust engine, no extra
        robustness, ~170x slower. Geometric-mean (Ruiz) equilibration is an exact,
        feasible-set-preserving rescaling, so solving the equilibrated LP with the
        same fast warm simplex yields the *identical* optimum (verified equal to
        the old cold path on nvs21) at warm speed. The objective value is invariant
        under the rescaling; only the returned point maps back via ``col_scale``.
        Returns the result, or ``None`` to defer to the generic path.
        """
        from discopt.solvers import SolveStatus

        if self._A_ub is None:
            return None
        try:
            from discopt.solvers.milp_simplex import solve_lp_warm_std
        except Exception:  # pragma: no cover - binding absent
            return None
        try:
            c_s, A_s, b_s, bounds_s, col_scale = equilibrate_relaxation_lp(
                self._c, self._A_ub, self._b_ub, self._bounds, None
            )
            result, _, cert = solve_lp_warm_std(
                c_s,
                sp.csr_matrix(A_s),
                b_s,
                bounds_s,
                in_basis=None,
                return_cert=True,
                time_limit=time_limit,
            )
        except Exception:  # pragma: no cover - defensive
            return None
        if result is None:
            self._stash_numerical_bound(cert)  # #517 last-resort floor (flag-gated)
            self._stash_deadline_bound(cert)
            return None
        status_map = {
            SolveStatus.OPTIMAL: "optimal",
            SolveStatus.INFEASIBLE: "infeasible",
            SolveStatus.UNBOUNDED: "unbounded",
            SolveStatus.TIME_LIMIT: "time_limit",
            SolveStatus.ITERATION_LIMIT: "iteration_limit",
            SolveStatus.ERROR: "error",
        }
        status_str = status_map.get(result.status, str(result.status))
        obj = None
        if result.objective is not None and self._objective_bound_valid:
            obj = float(result.objective) + self._obj_offset
        bound = None
        if result.bound is not None and self._objective_bound_valid:
            bound = float(result.bound) + self._obj_offset
        # Equilibration is objective-invariant, so the safe bound computed on the
        # rescaled LP is a valid safe bound on the original objective (issue #356).
        safe_bound = None
        if cert.safe_bound is not None and self._objective_bound_valid:
            safe_bound = float(cert.safe_bound) + self._obj_offset
        # Map the scaled solution point back to the original variables (x = D x').
        x_mapped = None
        if result.x is not None:
            x_mapped = np.asarray(result.x, dtype=np.float64) * np.asarray(
                col_scale, dtype=np.float64
            )
        return MilpRelaxationResult(
            status=status_str,
            objective=obj,
            bound=bound,
            x=x_mapped,
            safe_bound=safe_bound,
            farkas_certified=bool(cert.farkas_certified),
        )

    def _stash_numerical_bound(self, cert) -> None:
        """Record a Neumaier–Shcherbina safe bound recovered from a numerically-
        failed node LP as this solve's last-resort dual floor (#517, flag-gated).

        The NS bound comes from the in-house simplex's *own* dual candidate on a
        phase-2 breakdown and is valid for **any** multiplier vector, so it can
        never exceed the true optimum — a drifted basis only loosens it. Attached
        in :meth:`solve` only when nothing else produced a bound, so it never
        overrides (or tightens away from) a real bound. Tracks the tightest (max)
        floor seen across the warm/equilibrated attempts of one solve.
        """
        if not _tuning().node_numerical_dual_bound:
            return
        if cert is None or cert.safe_bound is None or not self._objective_bound_valid:
            return
        sb = float(cert.safe_bound) + self._obj_offset
        if not math.isfinite(sb):
            return
        prev = getattr(self, "_pending_numerical_bound", None)
        self._pending_numerical_bound = sb if prev is None else max(prev, sb)


def sanitize_relaxation_for_conditioning(
    model: "MilpRelaxationModel",
) -> "MilpRelaxationModel":
    """Return a copy of *model* with numerically catastrophic content removed, so
    the LP backend can produce a (sound, possibly weaker) bound instead of failing.

    Two transforms, both of which only *relax* the feasible set — the LP optimum
    therefore remains a valid lower bound for a minimization (weaker, never
    higher than the true optimum):

    1. Drop any constraint row whose coefficient or RHS is non-finite or has
       magnitude >= ``_RELAX_NUMERIC_CAP``. Removing a constraint enlarges the
       feasible set.
    2. Clamp any variable bound of magnitude >= ``_RELAX_NUMERIC_CAP`` to +/-inf.
       Widening a variable's box enlarges the feasible set. (A clamped objective
       variable can make the LP unbounded -> bound becomes -inf/None, still sound.)

    Both are no-ops on well-scaled models (no row or bound reaches the cap), so
    this is safe to apply unconditionally before a fallback root-bound solve.
    """
    cap = _RELAX_NUMERIC_CAP

    A = model._A_ub
    b = model._b_ub
    if A is not None and b is not None and A.shape[0] > 0:
        A = sp.csr_matrix(A)  # accepts dense or sparse; typed CSR for .data/.indptr
        b = np.asarray(b, dtype=np.float64)
        row_of_nz = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
        bad_nz = ~np.isfinite(A.data) | (np.abs(A.data) >= cap)
        keep = np.isfinite(b) & (np.abs(b) < cap)
        if bad_nz.any():
            keep[row_of_nz[bad_nz]] = False
        if not keep.all():
            logger.debug(
                "relaxation conditioning: dropped %d catastrophic constraint row(s)",
                int((~keep).sum()),
            )
            A = A[keep]
            b = b[keep]
        if A.shape[0] == 0:
            A = None
            b = None
    else:
        A = None
        b = None

    # Directional widening (#732 Stage 1): a crossing lower bound always drops to
    # -inf and a crossing upper bound always rises to +inf. The old sign-based
    # mapping pinned a large-positive lower bound to +inf (a [+inf, +inf) box —
    # not a widening), which is how the docstring's "widening" contract was
    # silently violated; see the solve_at_node clamp for the measured failure.
    bounds = [
        (
            lo if abs(lo) < cap else -np.inf,
            hi if abs(hi) < cap else np.inf,
        )
        for (lo, hi) in model._bounds
    ]

    return MilpRelaxationModel(
        c=model._c,
        A_ub=A,
        b_ub=b,
        bounds=bounds,
        obj_offset=model._obj_offset,
        integrality=model._integrality,
        objective_bound_valid=model._objective_bound_valid,
    )


# ---------------------------------------------------------------------------
# Helpers: variable bounds
# ---------------------------------------------------------------------------


def _constant_value(expr: Expression) -> Optional[float]:
    if not isinstance(expr, Constant):
        return None
    values = np.asarray(expr.value, dtype=np.float64).ravel()
    if values.size != 1:
        return None
    return float(values[0])


def _eval_constant_expr(expr: Expression) -> Optional[float]:
    """Evaluate a *variable-free* subexpression to a scalar, else return None.

    Unlike :func:`_constant_value` (which only recognizes a literal ``Constant``
    node), this folds composite constant subexpressions such as ``neg(2.5)``
    (a unary negation of a literal), ``(-3) * (-3)``, and other arithmetic over
    constants. It is deliberately conservative: it returns ``None`` the moment a
    variable, index, function call, or any unhandled node is encountered, so it
    can never mis-fold an expression that actually depends on a decision
    variable. Folding such factors is exact (value-preserving), so using it in
    the product linearizer only ever tightens the relaxation while staying
    sound.
    """
    direct = _constant_value(expr)
    if direct is not None:
        return direct
    if isinstance(expr, UnaryOp):
        val = _eval_constant_expr(expr.operand)
        if val is None:
            return None
        if expr.op == "neg":
            return -val
        if expr.op == "abs":
            return abs(val)
        return None
    if isinstance(expr, BinaryOp):
        left = _eval_constant_expr(expr.left)
        if left is None:
            return None
        right = _eval_constant_expr(expr.right)
        if right is None:
            return None
        if expr.op == "+":
            return left + right
        if expr.op == "-":
            return left - right
        if expr.op == "*":
            return left * right
        if expr.op == "/":
            if right == 0.0:
                return None
            return left / right
        if expr.op == "**":
            try:
                result = left**right
            except (ValueError, OverflowError, ZeroDivisionError):
                return None
            # A negative base to a fractional power yields a complex result;
            # that is not a real constant we can fold, so bail conservatively.
            if isinstance(result, complex):
                return None
            return float(result)
        return None
    return None


def _affine_var_base(expr: Expression, model: Model) -> Optional[tuple[float, int]]:
    """Return ``(coeff, flat_idx)`` if ``expr`` is a single scaled variable.

    Matches ``coeff * x`` in any of: a bare scalar variable, a constant-scaled
    variable (``c*x`` / ``x*c``), a variable divided by a constant (``x/c``), or
    a negation thereof.  Returns ``None`` for anything with additive structure or
    more than one variable.  Used to recognize an affine single-variable power
    base ``(c*x)**n`` so it can be lifted in well-conditioned scaled ``r = c*x``
    space rather than as a raw ``x**n`` monomial.
    """
    flat = _get_flat_index(expr, model)
    if flat is not None:
        return 1.0, flat
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        inner = _affine_var_base(expr.operand, model)
        if inner is not None:
            return -inner[0], inner[1]
        return None
    if isinstance(expr, BinaryOp):
        if expr.op == "*":
            lc = _constant_value(expr.left)
            if lc is not None:
                inner = _affine_var_base(expr.right, model)
                return (lc * inner[0], inner[1]) if inner is not None else None
            rc = _constant_value(expr.right)
            if rc is not None:
                inner = _affine_var_base(expr.left, model)
                return (rc * inner[0], inner[1]) if inner is not None else None
            return None
        if expr.op == "/":
            rc = _constant_value(expr.right)
            if rc is not None and rc != 0.0:
                inner = _affine_var_base(expr.left, model)
                return (inner[0] / rc, inner[1]) if inner is not None else None
            return None
    return None


# Flat-variable monomial: a sorted tuple of original variable indices, repeated
# by power (e.g. ``x1**2 * x0`` → ``(0, 1, 1)``).  An affine-square residual is
# represented as ``(const, [(coeff, monomial), ...])``.
_Monomial = tuple[int, ...]
_AffineSquare = tuple[float, list[tuple[float, _Monomial]]]


def _product_to_monomial(expr: Expression, model: Model) -> tuple[float, _Monomial] | None:
    """Fold a pure product of original variables / integer powers / constants.

    Returns ``(scalar, monomial)`` where ``monomial`` is the sorted tuple of
    original flat variable indices (repeated by power), or ``None`` if the
    product contains a non-polynomial leaf (a function call, a non-constant
    division, a fractional power, …).
    """
    scalar = [1.0]
    idxs: list[int] = []

    def visit(e: Expression) -> bool:
        if isinstance(e, BinaryOp) and e.op == "*":
            return visit(e.left) and visit(e.right)
        if isinstance(e, UnaryOp) and e.op == "neg":
            scalar[0] *= -1.0
            return visit(e.operand)
        if isinstance(e, Constant):
            scalar[0] *= float(e.value)
            return True
        flat = _get_flat_index(e, model)
        if flat is not None:
            idxs.append(flat)
            return True
        if isinstance(e, BinaryOp) and e.op == "**" and isinstance(e.right, Constant):
            base_flat = _get_flat_index(e.left, model)
            if base_flat is not None:
                exp_val = float(e.right.value)
                n = int(exp_val)
                if exp_val == n and n >= 1:
                    idxs.extend([base_flat] * n)
                    return True
        return False

    if not visit(expr):
        return None
    return scalar[0], tuple(sorted(idxs))


def _expr_to_polynomial(expr: Expression, model: Model) -> _AffineSquare | None:
    """Walk a *distributed* expression into ``(const, [(coeff, monomial), ...])``.

    Returns ``None`` if any leaf is not a polynomial in the original variables
    (so the caller falls back to the existing relaxation paths instead of
    misclassifying a transcendental residual).
    """
    const = [0.0]
    terms: list[tuple[float, _Monomial]] = []

    def visit(e: Expression, scale: float) -> bool:
        if isinstance(e, Constant):
            const[0] += scale * float(e.value)
            return True
        flat = _get_flat_index(e, model)
        if flat is not None:
            terms.append((scale, (flat,)))
            return True
        if isinstance(e, UnaryOp):
            if e.op == "neg":
                return visit(e.operand, -scale)
            return False
        if isinstance(e, SumExpression):
            return visit(e.operand, scale)
        if isinstance(e, SumOverExpression):
            # ``dm.sum(f, over=...)`` / ``dm.sum(term for ...)`` builds an n-ary
            # ``SumOverExpression`` rather than the binary ``+`` chain Python's
            # builtin ``sum`` produces. Without this branch the walk returned
            # ``False`` on a body that is byte-identically polynomial when written
            # with builtin ``sum`` — so ``extract_quadratic`` abstained on every
            # indexed-summation model and no such model could be certified convex
            # (issue #936, Defect 1). A sum is polynomial iff every term is, and
            # the scale distributes over the terms, so this is the same recursion
            # the ``+`` branch below performs, just n-ary.
            for term in e.terms:
                if not visit(term, scale):
                    return False
            return True
        if isinstance(e, BinaryOp):
            if e.op == "+":
                return visit(e.left, scale) and visit(e.right, scale)
            if e.op == "-":
                return visit(e.left, scale) and visit(e.right, -scale)
            if e.op == "/":
                if isinstance(e.right, Constant):
                    denom = float(e.right.value)
                    if denom == 0.0:
                        return False
                    return visit(e.left, scale / denom)
                return False
            if e.op == "*":
                decomp = _product_to_monomial(e, model)
                if decomp is None:
                    return False
                coeff, monomial = decomp
                terms.append((scale * coeff, monomial))
                return True
            if e.op == "**":
                decomp = _product_to_monomial(e, model)
                if decomp is None:
                    return False
                coeff, monomial = decomp
                terms.append((scale * coeff, monomial))
                return True
        return False

    if not visit(expr, 1.0):
        return None
    return const[0], terms


def _collect_affine_powers(
    model: Model, already_lifted: set[int]
) -> list[tuple[Expression, float, int, int]]:
    """Find ``(c*x)**n`` nodes (integer ``n >= 3``, ``c*x`` a scaled variable).

    Returns ``(node, scale, var_idx, power)`` for each distinct node whose base
    is a *non-bare* scaled single variable — a bare ``x**n`` is left to the
    standard monomial machinery (scaling it would not improve the aux-column
    conditioning).  Nodes already claimed by an issue-#155 affine-square lift
    (``already_lifted``) are skipped.
    """
    found: list[tuple[Expression, float, int, int]] = []
    seen: set[int] = set()

    def visit(e: Expression) -> None:
        if id(e) in already_lifted:
            return
        if isinstance(e, BinaryOp) and e.op == "**" and isinstance(e.right, Constant):
            exp_f = float(e.right.value)
            n = int(exp_f)
            # Only a genuinely scaled base benefits: a bare variable base lifts
            # identically in raw space, so leave it to the monomial path.
            if exp_f == n and n >= 3 and _get_flat_index(e.left, model) is None:
                base = _affine_var_base(e.left, model)
                if base is not None and id(e) not in seen:
                    scale, var_idx = base
                    seen.add(id(e))
                    found.append((e, float(scale), int(var_idx), n))
                    return  # the whole node is lifted; do not descend
        if isinstance(e, BinaryOp):
            visit(e.left)
            visit(e.right)
        elif isinstance(e, UnaryOp):
            visit(e.operand)
        elif isinstance(e, FunctionCall):
            for arg in e.args:
                visit(arg)
        elif isinstance(e, IndexExpression):
            if not isinstance(e.base, Variable):
                visit(e.base)
        elif isinstance(e, SumExpression):
            visit(e.operand)
        elif isinstance(e, SumOverExpression):
            for term in e.terms:
                visit(term)

    if model._objective is not None:
        visit(model._objective.expression)
    for constraint in model._constraints:
        visit(constraint.body)
    return found


def _normalize_convhull_formulation(formulation: str) -> str:
    """Normalize accepted bilinear convex-hull mode names."""
    aliases = {
        "disaggregated": "disaggregated",
        "piecewise": "disaggregated",
        "sos2": "sos2",
        "facet": "facet",
        "lambda": "sos2",
    }
    try:
        return aliases[formulation]
    except KeyError as err:
        raise ValueError(
            f"Unsupported convhull_formulation: {formulation!r}. "
            "Choose from 'disaggregated', 'sos2', 'facet', or 'lambda'."
        ) from err


# ---------------------------------------------------------------------------
# Helpers: expression decomposition
# ---------------------------------------------------------------------------


def _decompose_product(
    expr: Expression,
    model: Model,
    fractional_power_var_map: Optional[dict[tuple[int, float], int]] = None,
    univariate_var_map: Optional[dict[object, int]] = None,
    monomial_var_map: Optional[dict[tuple[int, int], int]] = None,
    composite_var_map: Optional[dict[int, int]] = None,
    composite_coeff_map: Optional[dict[int, float]] = None,
    pinned_value: Optional[Callable[[int], Optional[float]]] = None,
) -> tuple[float, list[int]] | None:
    """Decompose a product expression into (scalar, [flat_or_aux_idx, ...]).

    Returns None if expr contains non-constant, non-variable leaves.
    Constants are accumulated into the scalar; variable references and
    registered lifted sub-expressions are appended to the index list (using
    their MILP column indices).

    When ``monomial_var_map`` is supplied, a *mixed repeated-factor* product
    such as ``x*x*y`` is collapsed: the repeated original-variable group ``x*x``
    is replaced by its monomial aux column (``x**2``), leaving ``[col(x**2), y]``
    — a lifted bilinear pair. This lets the standard McCormick pipeline relax
    ``x**2 * y`` (one monomial envelope + one bilinear envelope) instead of
    rejecting it as an unsupported repeated-factor term.

    When ``pinned_value`` is supplied (a ``flat_idx -> exact value or None``
    lookup over the node's bounds), a power factor ``x**p`` whose base is pinned
    (lb==ub) is folded into the scalar as the exact constant ``x**p`` rather than
    requiring a fractional-power aux column — which the builder skips on a
    degenerate domain — so a branch/OBBT-pinned ``y * x**p`` term still decomposes
    (to ``[col(y)]`` scaled by ``x**p``) instead of dropping from the relaxation.
    """
    scalar: list[float] = [1.0]
    var_indices: list[int] = []

    def visit(e: Expression) -> bool:
        if isinstance(e, BinaryOp) and e.op == "*":
            return visit(e.left) and visit(e.right)
        if isinstance(e, UnaryOp) and e.op == "neg":
            # A negated factor ``neg(g)`` is ``-1 * g``: peel the sign into the
            # scalar and decompose ``g``. Without this, ``neg(x) * x`` — the
            # internal form a maximize→minimize flip produces for ``-x**2``, and
            # the shape the parser builds for any ``-a*b`` — is an undecomposable
            # product, so the whole term drops from the relaxation and the dual
            # bound freezes. For a pure-integer maximize-of-a-convex objective the
            # spatial B&B then certifies a stationary incumbent (x=0 for x**2) as
            # the optimum: a false-optimal (e.g. ``max x**2`` over integer [-3,3]
            # returned 0 instead of 9). Peeling is exact and sign-only, so it only
            # ever lets the existing envelopes fire.
            scalar[0] *= -1.0
            return visit(e.operand)
        if isinstance(e, Constant):
            scalar[0] *= float(e.value)
            return True
        flat = _get_flat_index(e, model)
        if flat is not None:
            var_indices.append(flat)
            return True
        if univariate_var_map:
            aux_col = univariate_var_map.get(id(e))
            if aux_col is not None:
                var_indices.append(aux_col)
                return True
        if composite_var_map:
            aux_col = composite_var_map.get(id(e))
            if aux_col is not None:
                # A composite node carrying a non-unit substitution coefficient
                # (e.g. a ratio-of-products aux scaled by the numerator constant,
                # issue #185) cannot be represented as a plain product factor here;
                # abstain so the linearizer's coefficient-aware path handles it.
                if composite_coeff_map and composite_coeff_map.get(id(e), 1.0) != 1.0:
                    return False
                var_indices.append(aux_col)
                return True
        # Recognize var^p (fractional p) when an aux column was allocated, or
        # fold it to an exact constant when the base is pinned (lb==ub) at this
        # node. The pinned fold matters because the fractional-power aux column
        # is *skipped* for a degenerate [lb==ub] domain (the builder's bounds
        # guard), so without it a branched/OBBT-pinned base turns ``y * x^p``
        # into an undecomposable product and the whole term — or the objective —
        # drops from the relaxation, sinking the node's dual bound. Folding the
        # pinned power is variable-free and exact, so it only ever tightens.
        if isinstance(e, BinaryOp) and e.op == "**" and isinstance(e.right, Constant):
            base_flat = _get_flat_index(e.left, model)
            if base_flat is not None:
                exp_val = float(e.right.value)
                key = (base_flat, exp_val)
                if fractional_power_var_map and key in fractional_power_var_map:
                    var_indices.append(fractional_power_var_map[key])
                    return True
                # Integer power x**n (n >= 2) → monomial aux column. Without this
                # an integer-power factor inside a product (e.g. ``x**0.5 * y**2``
                # in ex1226's e1) makes the whole product undecomposable and the
                # constraint drops from the relaxation, freezing the dual bound.
                # The monomial aux carries a rigorous power envelope and the
                # bilinear envelope between the two lifted columns is registered by
                # the uniform engine's product relaxation, so resolving it here only
                # ever shrinks the relaxed set toward the true one (sound).
                if monomial_var_map and exp_val == int(exp_val) and int(exp_val) >= 2:
                    mono_key = (base_flat, int(exp_val))
                    if mono_key in monomial_var_map:
                        var_indices.append(monomial_var_map[mono_key])
                        return True
                if pinned_value is not None:
                    pv = pinned_value(base_flat)
                    # x^p is real only for x >= 0 (fractional p) or any integer p.
                    if pv is not None and (pv >= 0.0 or exp_val == int(exp_val)):
                        try:
                            scalar[0] *= float(pv) ** exp_val
                        except (ValueError, OverflowError):
                            return False
                        return True
        # Fold a *composite* variable-free factor (e.g. ``neg(1e6)`` from a
        # parsed ``-1e6*i1*i2``, or ``(-3)*(-3)``) into the scalar. A bare
        # ``Constant`` is handled above; this catches negations/arithmetic over
        # constants that would otherwise look like an undecomposable factor and
        # cause the whole product (and its constraint) to be dropped from the
        # relaxation. Exact and variable-free, so it only ever tightens.
        cval = _eval_constant_expr(e)
        if cval is not None:
            scalar[0] *= cval
            return True
        return False

    if not visit(expr):
        return None

    # Collapse mixed repeated-factor groups (x*x*y) into monomial aux columns
    # so the product reduces to distinct lifted factors the McCormick pipeline
    # can relax. Pure monomials (x*x with a single unique base) are left intact
    # for the dedicated monomial branch in the linearizer.
    if monomial_var_map:
        n_orig = sum(v.size for v in model._variables)
        counts: dict[int, int] = {}
        for i in var_indices:
            if i < n_orig:
                counts[i] = counts.get(i, 0) + 1
        repeated = {i for i, c in counts.items() if c >= 2}
        if repeated and len(set(var_indices)) >= 2:
            collapsed: list[int] = []
            seen: set[int] = set()
            ok = True
            for i in var_indices:
                if i in repeated:
                    if i in seen:
                        continue
                    col = monomial_var_map.get((i, counts[i]))
                    if col is None:
                        ok = False
                        break
                    collapsed.append(col)
                    seen.add(i)
                else:
                    collapsed.append(i)
            if ok:
                var_indices = collapsed

    return scalar[0], var_indices


# Univariate functions whose superposition cuts are supported: smooth on any
# box the lifted aux already validated, so the Chebyshev kernel encloses them
# rigorously. ``abs`` (non-smooth) and ``tan`` (poles) are deliberately omitted.
_SUPERPOSITION_FUNCS = {
    "exp",
    "log",
    "log2",
    "log10",
    "sqrt",
    "reciprocal",
    "sin",
    "cos",
}


def _linear_constraint_forms(model: Model, n_vars: int) -> list[tuple[np.ndarray, float]]:
    """Return each *linear* model constraint as ``(coeff, const)`` meaning the
    valid inequality ``coeff . x + const <= 0``.

    Constraints are stored as ``body <= 0`` or ``body == 0`` (``>=`` is
    normalized to ``<=`` by the expression operators), so a ``<=`` constraint
    contributes one form and an ``==`` constraint contributes both ``g <= 0`` and
    ``-g <= 0``. Nonlinear constraint bodies are skipped (they have no affine
    form). These are the factors level-1 RLT multiplies by variable bound
    factors to generate valid product cuts.

    Callers that only need to know *whether* any such factor exists must use
    :func:`_any_linear_constraint_form` instead — materializing the dense arrays to
    take ``bool(...)`` of the list is ``O(n_constraints * n_vars)`` in both time and
    memory (issue #875).
    """
    forms: list[tuple[np.ndarray, float]] = []
    for constraint in model._constraints:
        sense = constraint.sense
        if sense not in ("<=", "=="):
            continue  # no form is emitted below; skip the linearization entirely
        terms = _linear_form_terms(constraint.body, model, n_vars)
        if terms is None:
            continue
        coeff = np.zeros(n_vars, dtype=np.float64)
        for j, c in terms[0].items():
            coeff[j] = c
        const = terms[1]
        if sense == "<=":
            forms.append((coeff, const))
        else:
            forms.append((coeff, const))
            forms.append((-coeff, -const))
    return forms


def _linear_form_terms(
    body: Expression, model: Model, n_vars: int
) -> Optional[tuple[dict[int, float], float]]:
    """Sparse ``(terms, const)`` for a linear constraint body, or ``None``.

    ``None`` means "not an affine factor": either the body is nonlinear (the
    linearizer refuses) or a coefficient/constant is non-finite. Shared by
    :func:`_linear_constraint_forms` and :func:`_any_linear_constraint_form` so both
    apply the same acceptance test.
    """
    try:
        terms, const = _linearize_affine_expr_sparse(body, model, n_vars)
    except ValueError:
        return None  # nonlinear body — not an affine factor
    if not np.isfinite(const):
        return None
    for j, c in terms.items():
        # The dense form raised IndexError (uncaught) on an out-of-range index and
        # dropped a non-finite coefficient; keep the drop, and treat an unindexable
        # coefficient the same way rather than writing outside the array.
        if not (0 <= j < n_vars) or not np.isfinite(c):
            return None
    return terms, float(const)


def _any_linear_constraint_form(model: Model, n_vars: int) -> bool:
    """True when the model has at least one linear constraint factor for RLT.

    Short-circuits on the first hit, and never allocates an ``n_vars`` array.
    ``bool(_linear_constraint_forms(model, n_vars))`` answers the same question but
    linearizes every row into a dense vector first and *keeps them all* — on
    ``watercontamination0202`` (106,711 vars / 107,209 rows) that is ~91 GB of
    coefficient arrays to compute one boolean, and it sits in the
    ``MccormickLPRelaxer`` constructor on the pre-B&B root path (issue #875).
    """
    for constraint in model._constraints:
        if constraint.sense not in ("<=", "=="):
            continue
        if _linear_form_terms(constraint.body, model, n_vars) is not None:
            return True
    return False


# A quadratic constraint factor for level-1 RLT (issue #15, Phase 2): the body
# ``g(x) = const + sum_i lin_i x_i + sum_{(k,l)} quad_{kl} x_k x_l`` with the sense
# of the parent constraint. ``quad`` keys are sorted index pairs ``(k, l)``,
# ``k <= l`` (``(k, k)`` is the square ``x_k**2``).
_QuadForm = tuple[dict[tuple[int, int], float], dict[int, float], float, str]


def _quadratic_constraint_forms(model: Model, n_vars: int) -> list[_QuadForm]:
    """Return each *genuinely quadratic* (degree-exactly-2 polynomial) model
    constraint as ``(quad, lin, const, sense)``.

    These are the nonlinear factors that Phase-2 level-1 RLT multiplies by
    variable bound factors. Purely linear bodies are skipped (the affine path in
    :func:`_linear_constraint_forms` already handles them); cubic-or-higher and
    non-polynomial (transcendental, fractional-power) bodies are skipped because
    their RLT products are out of scope for the degree-3 lifting implemented
    here. The parent ``sense`` (``"<="`` or ``"=="``) is carried so an equality
    parent can emit a two-sided equality product row.
    """
    forms: list[_QuadForm] = []
    for constraint in model._constraints:
        if constraint.sense not in ("<=", "=="):
            continue
        poly = _expr_to_polynomial(distribute_products(constraint.body), model)
        if poly is None:
            continue
        const, terms = poly
        quad: dict[tuple[int, int], float] = {}
        lin: dict[int, float] = {}
        const_acc = float(const)
        max_degree = 0
        ok = True
        for coeff, monomial in terms:
            degree = len(monomial)
            max_degree = max(max_degree, degree)
            if degree == 0:
                const_acc += coeff
            elif degree == 1:
                idx = monomial[0]
                if idx >= n_vars:
                    ok = False
                    break
                lin[idx] = lin.get(idx, 0.0) + coeff
            elif degree == 2:
                ka, kb = monomial  # already sorted by _product_to_monomial
                if ka >= n_vars or kb >= n_vars:
                    ok = False
                    break
                quad[(ka, kb)] = quad.get((ka, kb), 0.0) + coeff
            else:
                ok = False  # degree >= 3 body: out of scope for quadratic-factor RLT
                break
        if not ok or max_degree != 2:
            continue
        if not (np.isfinite(const_acc) and all(np.isfinite(v) for v in lin.values())):
            continue
        if not all(np.isfinite(v) for v in quad.values()):
            continue
        forms.append((quad, lin, const_acc, constraint.sense))
    return forms


def _linearize_affine_expr_sparse(
    expr: Expression, model: Model, n_vars: int
) -> tuple[dict[int, float], float]:
    """Sparse core of :func:`_linearize_affine_expr`: ``({flat_index: coeff}, const)``.

    Same recognition rules and same ``ValueError`` refusals as the dense wrapper —
    only the accumulator differs. Callers that just need the *support* of an affine
    body (how many variables it touches, and which) should use this: the dense
    wrapper costs ``O(n_vars)`` per call to allocate and zero its array, so scanning
    every constraint through it is ``O(n_constraints * n_vars)`` no matter how sparse
    the bodies are.

    Measured (issue #875, ``watercontamination0202``: 106,711 vars / 107,209 rows):
    ``_fix_single_var_equalities`` — a scan whose bodies have a SINGLE leaf each —
    spent ~460 s of a 30 s solve budget entirely in that dense allocation plus the
    caller's Python walk over all 106,711 entries. The synthetic scaling probe is
    exactly linear in ``n_vars`` at a fixed constraint count (0.068 s at n=2,000 →
    3.650 s at n=128,000 for 400 rows), which is the signature of the dense array
    rather than of the bodies.

    Keys are the flat original-variable indices; a key is emitted only for a
    variable the body actually references, so the dict size is ``O(body leaves)``.
    Indices are NOT range-checked here (the dense wrapper's array indexing is what
    enforces ``n_vars``); a sparse caller that indexes arrays with them must bound
    them itself.
    """
    coeff: dict[int, float] = {}
    const_acc: list[float] = [0.0]

    def visit(e: Expression, scale: float) -> None:
        if isinstance(e, Constant):
            const_acc[0] += scale * float(e.value)
            return

        if isinstance(e, Variable):
            offset = _compute_var_offset(e, model)
            if e.size == 1:
                coeff[offset] = coeff.get(offset, 0.0) + scale
                return
            raise ValueError(f"Cannot use array variable as scalar affine argument: {e}")

        if isinstance(e, IndexExpression):
            flat = _get_flat_index(e, model)
            if flat is None:
                raise ValueError(f"Cannot linearize IndexExpression: {e}")
            coeff[flat] = coeff.get(flat, 0.0) + scale
            return

        if isinstance(e, UnaryOp) and e.op == "neg":
            visit(e.operand, -scale)
            return

        if isinstance(e, BinaryOp):
            if e.op == "+":
                visit(e.left, scale)
                visit(e.right, scale)
                return
            if e.op == "-":
                visit(e.left, scale)
                visit(e.right, -scale)
                return
            if e.op == "*":
                if isinstance(e.left, Constant):
                    visit(e.right, scale * float(e.left.value))
                    return
                if isinstance(e.right, Constant):
                    visit(e.left, scale * float(e.right.value))
                    return
                raise ValueError(f"Non-affine product in univariate argument: {e}")
            if e.op == "/":
                if isinstance(e.right, Constant):
                    visit(e.left, scale / float(e.right.value))
                    return
                raise ValueError(f"Non-affine division in univariate argument: {e}")
            if e.op == "**":
                if isinstance(e.right, Constant):
                    exp = float(e.right.value)
                    if exp == 1.0:
                        visit(e.left, scale)
                        return
                    if exp == 0.0:
                        const_acc[0] += scale
                        return
                raise ValueError(f"Non-affine power in univariate argument: {e}")

        if isinstance(e, SumExpression):
            op = e.operand
            if isinstance(op, Variable):
                offset = _compute_var_offset(op, model)
                for k in range(op.size):
                    coeff[offset + k] = coeff.get(offset + k, 0.0) + scale
                return
            visit(op, scale)
            return

        if isinstance(e, SumOverExpression):
            for term in e.terms:
                visit(term, scale)
            return

        raise ValueError(f"Unsupported affine argument node {type(e).__name__}: {e}")

    visit(expr, 1.0)
    return coeff, const_acc[0]


def _linearize_affine_expr(expr: Expression, model: Model, n_vars: int) -> tuple[np.ndarray, float]:
    """Linearize an affine expression over original variables.

    Raises ValueError when the expression contains nonlinear structure: only
    affine arguments are soundly supported here, since any nonlinear structure
    is relaxed by the uniform engine's atom envelopes rather than linearized.

    Dense view of :func:`_linearize_affine_expr_sparse` — identical coefficients,
    identical constant, identical refusals, and (as before) an ``IndexError`` from
    the array store when a body references a flat index at or beyond ``n_vars``.
    Prefer the sparse core in any per-constraint scan: this allocates and zeroes an
    ``n_vars`` array per call (issue #875).
    """
    terms, const = _linearize_affine_expr_sparse(expr, model, n_vars)
    coeff = np.zeros(n_vars, dtype=np.float64)
    for j, c in terms.items():
        coeff[j] = c
    return coeff, const


def _match_scaled_constant_division(
    expr: Expression,
    scale: float,
) -> Optional[tuple[float, Expression]]:
    """Return (scaled numerator, denominator) for scale * (c / denominator)."""
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return _match_scaled_constant_division(expr.operand, -scale)

    if isinstance(expr, BinaryOp) and expr.op == "*":
        left_const = _constant_value(expr.left)
        if left_const is not None:
            return _match_scaled_constant_division(expr.right, scale * left_const)
        right_const = _constant_value(expr.right)
        if right_const is not None:
            return _match_scaled_constant_division(expr.left, scale * right_const)
        return None

    if not isinstance(expr, BinaryOp) or expr.op != "/":
        return None
    numerator = _constant_value(expr.left)
    if numerator is None or abs(numerator) <= 1e-12:
        return None
    return scale * numerator, expr.right


@dataclass
class CompositeMultivarRelaxation:
    """Outer relaxation for a *multivariate* convex/concave nonlinear node.

    The multivariate counterpart of the composite univariate convex/concave
    relaxation: a node ``g(x)`` that
    depends on **more than one** original variable but whose global curvature is
    certified by the DCP classifier — e.g. the Euclidean distance
    ``sqrt((x0-x2)**2 + (x1-x3)**2) = ||A x + b||`` of a TSP-with-neighbourhoods
    objective (MINLPLib ``tspn*``). The aux column ``d`` replaces ``g(x)`` so a
    product such as ``g(x) * x10`` decomposes through the standard McCormick
    bilinear envelope (``d`` registered in ``composite_var_map``).

    Soundness:

    * **CONVEX** ``g`` — each gradient cut ``d ≥ g(x_k) + ∇g(x_k)·(x − x_k)`` is
      a supporting hyperplane, valid *everywhere* for a convex function (no
      finiteness of bounds required for the cut itself), and the column upper
      bound ``d ≤ U`` uses a sound interval over-enclosure of ``g`` on the box.
      Together the tangent cuts (below) and the constant cap (above) form a
      rigorous outer band.
    * **CONCAVE** ``g`` — the roles swap: gradient cuts over-estimate
      (``d ≤ …``) and the constant column lower bound ``d ≥ L`` bounds below.

    Each line is sparse: ``((col, coeff), …), intercept`` meaning
    ``d (≥|≤) Σ coeff·x_col + intercept``.
    """

    expr_id: int
    aux_col: int
    curvature: str
    lower_lines: tuple[tuple[tuple[tuple[int, float], ...], float], ...]
    upper_lines: tuple[tuple[tuple[tuple[int, float], ...], float], ...]
    # Dependent original-variable columns and the compiled value/gradient of the
    # lifted node, so a separator can add the exact supporting hyperplane at the LP
    # point each round (issue #358 Phase 2). ``None`` disables LP-point separation.
    idxs: tuple[int, ...] = ()
    value_fn: Optional[Callable] = None
    grad_fn: Optional[Callable] = None


_COMPOSITE_CURV_TOL = 1e-9
_COMPOSITE_MAX_SUBDIV = 256
# Max sub-boxes a multivariate box-convexity certificate may enumerate across all
# refinement levels of its partition (keeps the interval-Hessian sweep bounded for
# high-dimensional nodes; pinned axes are excluded from the product).
_MULTIVAR_MAX_SUBBOXES = 64


def _build_convexity_box(model: Model, flat_lb: np.ndarray, flat_ub: np.ndarray) -> dict:
    """Build the ``{Variable: Interval}`` box the convexity certificate expects."""
    from discopt._relax.convexity.interval import Interval

    box: dict = {}
    offset = 0
    for v in model._variables:
        size = v.size
        shape = v.shape if v.shape else (1,)
        lo = np.asarray(flat_lb[offset : offset + size], dtype=np.float64).reshape(shape)
        hi = np.asarray(flat_ub[offset : offset + size], dtype=np.float64).reshape(shape)
        box[v] = Interval(lo, hi)
        offset += size
    return box


def _extract_positive_product(
    expr: Expression, model: Model, n_orig: int, flat_lb: np.ndarray, flat_ub: np.ndarray
) -> Optional[tuple[float, dict[int, float]]]:
    """Return ``(coef, {orig_var_idx: exponent})`` if ``expr`` is a product/power of
    strictly-positive original variables, else ``None``.

    Strict positivity (lb > 0 on the node box) is required for every factor — the
    log lift is undefined otherwise — and is the H-LOG precondition (no epsilon
    shift). Only *original* variables (index < ``n_orig``) are accepted as factors.
    """
    factors: dict[int, float] = {}
    coef = [1.0]

    def visit(e: Expression, power: float) -> bool:
        if isinstance(e, Constant):
            v = float(e.value)
            if v <= 0.0:
                return False  # a non-positive constant factor breaks the log lift
            coef[0] *= v**power
            return True
        idx = _get_flat_index(e, model)
        if idx is not None:
            if idx >= n_orig:
                return False
            lo = float(flat_lb[idx])
            if not (lo > 1e-9) or not np.isfinite(lo):
                return False  # strict-positivity precondition (no epsilon shift)
            factors[idx] = factors.get(idx, 0.0) + power
            return True
        if isinstance(e, BinaryOp):
            if e.op == "*":
                return visit(e.left, power) and visit(e.right, power)
            if e.op == "/":
                return visit(e.left, power) and visit(e.right, -power)
            if e.op == "**" and isinstance(e.right, Constant):
                return visit(e.left, power * float(e.right.value))
            if e.op in ("+", "-"):
                # Additive-identity passthrough: canonical reconstruct wraps a
                # monomial factor as ``0 + m`` / ``m + 0`` / ``m - 0``. Strip a zero
                # constant operand and recurse into the real factor.
                if isinstance(e.left, Constant) and abs(float(e.left.value)) <= 1e-12:
                    if e.op == "+":
                        return visit(e.right, power)
                elif isinstance(e.right, Constant) and abs(float(e.right.value)) <= 1e-12:
                    return visit(e.left, power)
        # Fallback: a factor that is an affine multiple of a SINGLE positive
        # variable, ``c·x`` with ``c>0`` and zero constant (e.g. ``(30000·x)^-0.48``
        # in a signomial). Then ``(c·x)^power = c^power · x^power`` is still a
        # positive monomial factor. A nonzero constant or >1 variable breaks the
        # monomial form (reject); a non-positive coefficient breaks the log lift.
        try:
            # Sparse: this runs per factor of every monomial body, and the dense
            # form allocated an ``n_orig`` array plus a Python walk over it just to
            # count the nonzeros (#875).
            aff_terms, aff_const = _linearize_affine_expr_sparse(e, model, n_orig)
        except ValueError:
            return False
        if abs(float(aff_const)) > 1e-12:
            return False
        nz = [(j, c) for j, c in aff_terms.items() if abs(c) > 0.0]
        if len(nz) != 1:
            return False
        j, c = nz[0]
        if j < 0 or j >= n_orig or c <= 0.0:
            return False
        lo = float(flat_lb[j])
        if not (lo > 1e-9) or not np.isfinite(lo):
            return False
        coef[0] *= c**power
        factors[j] = factors.get(j, 0.0) + power
        return True

    if not visit(expr, 1.0):
        return None
    if not factors:
        return None
    return coef[0], factors


def _multivar_box_curvature(
    expr: Expression,
    model: Model,
    idxs: list[int],
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    box: dict,
) -> Optional[str]:
    """Sound box-restricted ``"convex"``/``"concave"`` certificate for a node.

    The multivariate curvature certificate. ``expr`` is C²,
    so it is convex on the (convex) box iff ``∇²expr ⪰ 0`` at *every* point of the
    box, and concave iff ``∇²expr ⪯ 0`` everywhere. We enclose the Hessian with
    interval AD on each sub-box of an axis-aligned partition and apply a per-row
    interval-Gershgorin eigenvalue bound (identical math to
    :func:`alphabb.rigorous_alpha`): for the dependent submatrix ``H``,

        λ_min(H) ≥ min_i ( H[i,i].lo − Σ_{j≠i} max(|H[i,j].lo|, |H[i,j].hi|) )
        λ_max(H) ≤ max_i ( H[i,i].hi + Σ_{j≠i} max(|H[i,j].lo|, |H[i,j].hi|) ).

    Because PSD-ness is a *pointwise* condition and the sub-boxes cover the box,
    certifying the sign on every sub-box certifies it on the whole box — so a
    function that is only *locally* convex (nonconvex elsewhere, e.g. a ``sqrt`` of
    an indefinite polynomial) is still soundly certified over the region the
    relaxation actually uses. Refines the partition until conclusive or the
    sub-box budget is hit; abstains (returns ``None``) otherwise. This certificate
    is general: it depends on no algebraic shape, only on the interval Hessian, so
    it covers every twice-differentiable multivariate node, not one problem class.

    Only certifies when the dependent axes are finitely bounded (the interval
    Hessian needs a finite box). Off-diagonal couplings to non-dependent variables
    are exactly zero (``expr`` does not depend on them), so restricting Gershgorin
    to the dependent submatrix loses nothing.
    """

    from discopt._relax.convexity.interval import Interval
    from discopt._relax.convexity.interval_ad import interval_hessian

    dep = [int(j) for j in idxs]
    d = len(dep)
    if d == 0:
        return None
    los = np.array([float(flat_lb[j]) for j in dep], dtype=np.float64)
    his = np.array([float(flat_ub[j]) for j in dep], dtype=np.float64)
    if not (np.all(np.isfinite(los)) and np.all(np.isfinite(his))) or np.any(his < los):
        return None
    widths = his - los

    # Locate each dependent flat index within its (possibly vector) Variable.
    var_at: dict[int, tuple[Variable, int]] = {}
    offset = 0
    for v in model._variables:
        for c in range(v.size):
            var_at[offset + c] = (v, c)
        offset += v.size
    if any(j not in var_at for j in dep):
        return None
    loc = {j: var_at[j] for j in dep}
    affected_vars = {loc[j][0] for j in dep}
    saved = {v: box[v] for v in affected_vars}

    tol = _COMPOSITE_CURV_TOL
    non_pinned = [i for i in range(d) if widths[i] > tol]
    ix = np.ix_(dep, dep)

    def _verdict_at(k: int) -> Optional[str]:
        edges = [
            np.linspace(los[i], his[i], k + 1) if widths[i] > tol else np.array([los[i], his[i]])
            for i in range(d)
        ]
        all_convex = True
        all_concave = True
        for combo in itertools.product(*[range(len(e) - 1) for e in edges]):
            lo_arr = {v: np.array(saved[v].lo, dtype=np.float64).reshape(-1) for v in affected_vars}
            hi_arr = {v: np.array(saved[v].hi, dtype=np.float64).reshape(-1) for v in affected_vars}
            for i, j in enumerate(dep):
                v, c = loc[j]
                lo_arr[v][c] = float(edges[i][combo[i]])
                hi_arr[v][c] = float(edges[i][combo[i] + 1])
            for v in affected_vars:
                box[v] = Interval(
                    lo_arr[v].reshape(saved[v].lo.shape), hi_arr[v].reshape(saved[v].hi.shape)
                )
            try:
                ad = interval_hessian(expr, model, box=box)
            except Exception:
                return None
            h_lo = np.asarray(ad.hess.lo, dtype=np.float64)[ix]
            h_hi = np.asarray(ad.hess.hi, dtype=np.float64)[ix]
            if not (np.all(np.isfinite(h_lo)) and np.all(np.isfinite(h_hi))):
                return None
            abs_max = np.maximum(np.abs(h_lo), np.abs(h_hi))
            row_radius = abs_max.sum(axis=1) - np.abs(np.diag(abs_max))
            gersh_lo = np.diag(h_lo) - row_radius  # ≤ λ_min(H)
            gersh_hi = np.diag(h_hi) + row_radius  # ≥ λ_max(H)
            if np.any(gersh_lo < -tol):
                all_convex = False
            if np.any(gersh_hi > tol):
                all_concave = False
            if not all_convex and not all_concave:
                break
        if all_convex:
            return "convex"
        if all_concave:
            return "concave"
        return ""  # inconclusive at this refinement → keep refining

    try:
        if not non_pinned:
            return _verdict_at(1) or None
        k = 1
        while True:
            verdict = _verdict_at(k)
            if verdict is None:
                return None  # non-finite Hessian / AD failure → abstain
            if verdict:
                return verdict
            next_k = k * 2
            if next_k ** len(non_pinned) > _MULTIVAR_MAX_SUBBOXES:
                return None
            k = next_k
    finally:
        for v in affected_vars:
            box[v] = saved[v]


_EMPTY_VARMAP_KEYS: tuple[str, ...] = (
    "bilinear",
    "trilinear",
    "trilinear_stages",
    "multilinear",
    "multilinear_stages",
    "monomial",
    "monomial_pw",
    "univariate",
    "univariate_signatures",
    "univariate_relaxations",
    "composite_relaxations",
    "composite_multivar_relaxations",
    "univariate_piecewise_relaxations",
    "univariate_square",
    "univariate_square_relaxations",
    "univariate_square_piecewise_relaxations",
    "finite_domain_trig_square_tables",
    "fractional_power",
    "bilinear_pw",
    "bilinear_lambda",
    "generation_guardrails",
    "ratio",
)


def _empty_varmap(n_orig: int, convhull_mode: str) -> dict:
    """A drop-in ``varmap`` for the engine path: originals mapped, families empty."""
    vm: dict = {k: {} for k in _EMPTY_VARMAP_KEYS}
    vm["original"] = {k: k for k in range(n_orig)}
    vm["minmax_objective_lift"] = None
    vm["convhull_formulation"] = convhull_mode
    vm["convhull_ebd"] = False
    vm["convhull_ebd_encoding"] = "gray"
    vm["generation_guardrails"] = []
    return vm


def _uniform_relaxation_delegate(
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
    n_orig: int,
    convhull_mode: str,
    rlt_level1: bool = False,
    skip_separable_floor: bool = False,
    skip_convex_lift: bool = False,
    disc_state: object = None,
    build_deadline: Optional[float] = None,
) -> tuple["MilpRelaxationModel", dict]:
    """Build the default relaxation through the uniform factorable engine (#632).

    ``build_uniform_relaxation`` (``uniform_relax.py``) relaxes every canonical
    atom class soundly by the auxiliary-variable method and returns a
    :class:`MilpRelaxationModel` with the SAME output contract as the historical
    federated builder — original variables in columns ``0..n_orig-1``, aux columns
    appended after. Soundness is by construction (every emitted row is a valid
    outer inequality at the lifted point); tightness parity with the deleted
    product-side separators (RLT/PSD/finite-domain trig) is the deferred polish
    pass. Original-variable integrality is preserved (aux columns continuous), so
    the integer-aware node solve and every legacy caller keep their contract.
    """
    from discopt._relax.uniform_relax import build_uniform_relaxation

    # Quadratic constraint-factor RLT (#640 Bucket 2) fires only when the caller
    # engaged level-1 RLT AND ``DISCOPT_RLT_QUAD`` is on (default on). Off => the
    # base build is byte-identical to before.
    rlt_quad = bool(rlt_level1 and _tuning().rlt_quad)
    rel = build_uniform_relaxation(
        model,
        box=(flat_lb, flat_ub),
        rlt_quad=rlt_quad,
        skip_separable_floor=skip_separable_floor,
        skip_convex_lift=skip_convex_lift,
        disc_state=disc_state,
        build_deadline=build_deadline,
    )
    milp = rel.model
    n_total = int(np.size(milp._c))
    flags = np.zeros(n_total, dtype=np.int32)
    off = 0
    for v in model._variables:
        if v.var_type in (VarType.BINARY, VarType.INTEGER):
            flags[off : off + v.size] = 1
        off += int(v.size)
    # OR in ENGINE-created integer aux columns (e.g. the finite-domain trig-square
    # selector binaries, #640 Bucket 1). The exact selector table is only exact when
    # its ``λ`` are integer, so the node MILP must see them as such; the pure-LP root
    # (continuous ``λ``) keeps the sound convex-hull relaxation.
    aux_int = np.asarray(rel.integrality, dtype=np.int32)
    if aux_int.size == n_total:
        flags = np.maximum(flags, aux_int)
    milp._integrality = flags if int(flags.sum()) else None
    # Populate the structural varmap families the proven legacy separators (PSD /
    # RLT / edge-concave / univariate-square / multilinear) consume, from the
    # engine's own decomposition (uniform_relax registered each lifted product /
    # power of ORIGINAL variables to its aux column). This restores product-side
    # tightness parity: the separators now fire on the engine's relaxation exactly
    # as they did on the deleted federation, driven by the uniform factorable
    # decomposition. Every registered aux equals the named product/power and is
    # tied to its originals by the emitted McCormick / secant-tangent rows, so
    # every separated cut is a valid inequality at the lifted feasible point
    # (soundness by construction; see uniform_relax.UniformRelaxation).
    vm = _empty_varmap(n_orig, convhull_mode)
    vm["bilinear"] = dict(rel.bilinear_map)
    vm["monomial"] = dict(rel.monomial_map)
    vm["trilinear"] = dict(rel.trilinear_map)
    vm["multilinear"] = dict(rel.multilinear_map)
    vm["univariate_square"] = dict(rel.univariate_square_map)
    # Affine squares ``(c·x_j+d)**2`` (#640 Bucket 3): ``(var, aux) -> (coeff, const)``
    # for the incremental McCormick patch's closed-form envelope regeneration.
    vm["affine_square"] = dict(rel.affine_square_map)
    # Ratio-of-products lifts (issue #309): the quotient aux column each exact
    # ``(Π x_i)/(Π y_j)`` of bare originals was lifted to, for the integer-ratio
    # partition bound.
    vm["ratio"] = dict(rel.ratio_map)
    # Composite convex/concave lifts (issue #632 P2): each certified-convex/-concave
    # multivariate node the engine lifted to a single aux is registered here so the
    # existing ``MccormickLPRelaxer._separate_convex`` outer-approximation (Kelley)
    # loop adds its exact supporting tangent at the LP point each round, recovering
    # the composite-convex tightness class generally. Each spec carries a jax value
    # / gradient over the ORIGINAL affine-free variables and its aux column; the
    # tangent of a certified convex (resp. concave) function is a global under-
    # (over-) estimator, so the cut never removes a feasible point (sound by
    # construction; the loop is a sound no-op on any failure).
    vm["composite_multivar_relaxations"] = list(rel.composite_multivar_specs)
    # Piecewise univariate/monomial/bilinear refinement (#640 S8) is now emitted
    # DIRECTLY as relaxation rows by ``build_uniform_relaxation`` when a ``disc_state``
    # partition is supplied (the AMP path), not surfaced through this legacy census
    # list — so the family stays the honest empty list here (the tightening rows are
    # already in the relaxation, keyed to the atoms' aux columns).
    vm["univariate_piecewise_relaxations"] = []
    # Finite-domain trig-square selector tables (#640 Bucket 1): exact one-hot
    # encodings of sin/cos(int-affine)^2 the engine emitted, surfaced for callers
    # that census them (the rows are already in the relaxation).
    vm["finite_domain_trig_square_tables"] = list(rel.finite_domain_trig_square_tables)
    return milp, vm


# ── #671 hda-certification lever: float64-intractable-row filter ─────────────
# Per-row thresholds validated by the entry experiment on hda's exported root LP
# (docs/dev/hda-certification-rowfilter-entry-2026-07-18.md): a row whose
# nonzero coefficients span more than RATIO orders, or contain a coefficient
# outside [ABS_LO, ABS_HI], cannot have its satisfaction resolved in float64 at
# the LP feasibility tolerance (hda: 130 such rows made every float64 engine
# false-fail while contributing ZERO root tightness — dropping them let the
# in-house simplex solve clean at tau=0 with the NS-certified tight bound).
_ROW_FILTER_RATIO = 1e6
_ROW_FILTER_ABS_HI = 1e8
_ROW_FILTER_ABS_LO = 1e-8


def _filter_unresolvable_rows(milp: "MilpRelaxationModel") -> int:
    """Drop float64-intractable rows from ``milp`` in place; return the count.

    SOUND BY CONSTRUCTION: removing relaxation rows yields a superset feasible
    region — a valid (weaker) outer approximation. The dual bound can only
    loosen, never falsify ("weaken but never falsify"). Tightness impact is
    instance-dependent and gated by the §5 corpus differential panel; on the
    hda class the dropped rows carry no tightness and un-poison the LP.

    Preserves the container kind (sparse stays CSR, dense stays ndarray) so
    downstream row-append paths keep working; empty rows are never dropped (an
    empty infeasible row ``0 <= b < 0`` is a rigorous infeasibility proof).
    """
    a_ub = milp._A_ub
    if a_ub is None or milp._b_ub is None:
        return 0
    was_sparse = sp.issparse(a_ub)
    a_csr = sp.csr_matrix(a_ub)
    m = a_csr.shape[0]
    if m == 0:
        return 0
    absd = np.abs(a_csr.data)
    keep = np.ones(m, dtype=bool)
    indptr = a_csr.indptr
    for i in range(m):
        s, e = indptr[i], indptr[i + 1]
        if s == e:
            continue  # empty row: keep (may encode a rigorous infeasibility)
        seg = absd[s:e]
        hi = seg.max()
        lo = seg.min()
        # Absolute checks first, then the ratio as a multiply: `hi / lo` overflows
        # on a denormal `lo` (same verdict, noisy RuntimeWarning); after the
        # absolute checks pass, `lo >= ABS_LO` so `lo * RATIO` cannot overflow.
        if hi > _ROW_FILTER_ABS_HI or lo < _ROW_FILTER_ABS_LO or hi > lo * _ROW_FILTER_RATIO:
            keep[i] = False
    dropped = int(m - keep.sum())
    if dropped == 0:
        return 0
    filtered = a_csr[keep]
    milp._A_ub = filtered if was_sparse else filtered.toarray()
    milp._b_ub = np.asarray(milp._b_ub)[keep]
    logger.debug("relax_row_filter: dropped %d/%d float64-intractable rows (#671)", dropped, m)
    return dropped


def build_milp_relaxation(
    model: Model,
    terms: NonlinearTerms,
    disc_state: DiscretizationState,
    incumbent: Optional[np.ndarray] = None,
    oa_cuts: Optional[list] = None,
    convhull_formulation: str = "disaggregated",
    convhull_ebd: bool = False,
    convhull_ebd_encoding: str = "gray",
    bound_override: Optional[tuple[np.ndarray, np.ndarray]] = None,
    superposition: bool = False,
    rlt_level1: bool = False,
    skip_separable_floor: bool = False,
    skip_convex_lift: bool = False,
    build_deadline: Optional[float] = None,
) -> tuple["MilpRelaxationModel", dict]:
    """Build a MILP relaxation with piecewise McCormick for bilinear/monomial terms.

    .. note::
        **#632 cutover — this function now delegates the entire default build to
        the uniform factorable engine** (:func:`uniform_relax.build_uniform_relaxation`),
        which relaxes every canonical atom class soundly via the AVM. As a result
        the following parameters are currently **IGNORED** on the default path and
        kept only for signature compatibility: ``terms``, ``incumbent``, ``oa_cuts``
        (OA/Kelley tangents are added lazily by the separators at ``solve_at_node``,
        not pre-seeded here), ``convhull_ebd``/``convhull_ebd_encoding``,
        ``superposition``. ``disc_state`` is now **consumed** again (#640 S8): its
        partition breakpoints drive sound piecewise-McCormick refinement of every
        bilinear/monomial/univariate atom depending on a partitioned variable, so
        the AMP adaptive-partition loop tightens the node bound as it refines. Only
        ``model``, ``convhull_formulation`` (validated), ``bound_override``,
        ``disc_state`` and ``rlt_level1`` affect the result. The engine remains a
        valid outer relaxation by construction, verified ``incorrect_count = 0`` on
        the global50 panel. The docstring below describes the superseded federated
        build and is retained for historical context.

    For each bilinear term x_i*x_j: adds standard McCormick envelope constraints
    (4 linear inequalities).  These give the convex hull of the bilinear set on the
    bounding box and are independent of the partition (piecewise refinement via binary
    variables is left for future enhancement).

    For each monomial x_i^n (currently n=2 handled precisely):
    - Piecewise tangent underestimators (one per partition interval midpoint) — gets
      tighter as disc_state gains more intervals.
    - Global secant overestimator — bounds s from above.

    The LP objective and constraints are obtained by substituting auxiliary vars for
    all nonlinear terms.

    Parameters
    ----------
    model : Model
    terms : NonlinearTerms
        Output of classify_nonlinear_terms(model).
    disc_state : DiscretizationState
        Current partition; provides intervals for tangent cut placement.
    incumbent : np.ndarray, optional
        Current best NLP solution (flat).  Used to add OA tangent cuts for
        general nonlinear terms; currently unused (reserved for future use).
    convhull_formulation : str, default "disaggregated"
        Piecewise bilinear formulation. ``"disaggregated"`` keeps the existing
        xbar/wbar construction; ``"sos2"`` and ``"facet"`` use a λ-based
        convex-hull reformulation similar to Alpine.jl.
    convhull_ebd : bool, default False
        Replace SOS2 interval binaries with a logarithmic embedded encoding.
        Only supported with ``convhull_formulation="sos2"`` or ``"lambda"``.
    convhull_ebd_encoding : str, default "gray"
        Embedded encoding scheme. ``"gray"`` is the Alpine-style default and
        the only option that remains SOS2-compatible for arbitrary partition
        counts. ``"binary"`` is only valid for two partitions.

    Returns
    -------
    (MilpRelaxationModel, varmap)
        MilpRelaxationModel has a .solve() method returning MilpRelaxationResult.
        varmap maps auxiliary variable keys to MILP column indices.
    """
    if bound_override is None:
        flat_lb, flat_ub = flat_variable_bounds(model)
    else:
        flat_lb = np.asarray(bound_override[0], dtype=np.float64)
        flat_ub = np.asarray(bound_override[1], dtype=np.float64)
    n_orig = len(flat_lb)
    convhull_mode = _normalize_convhull_formulation(convhull_formulation)
    if convhull_ebd and convhull_mode != "sos2":
        raise ValueError(
            "convhull_ebd is only supported with convhull_formulation='sos2' or its 'lambda' alias."
        )
    # ── #632 cutover: the uniform factorable engine is the DEFAULT relaxation ──
    # Route the build through build_uniform_relaxation (uniform_relax.py), which
    # relaxes every canonical atom class soundly via the AVM and returns a
    # MilpRelaxationModel with the same column contract. This supersedes the
    # federated collectors/separators below (being deleted stage-by-stage). The
    # engine is a valid outer relaxation by construction; product-side tightness
    # parity is the deferred polish pass.
    # #671 row filter is FAILURE-TRIGGERED at the solve layer
    # (mccormick_lp._solve_at_node_impl), NOT applied here at build time: an
    # always-on build-time filter drops rows that carry genuine tightness on
    # already-solving instances (in-repo panel: 10/66 regressions incl. nvs09
    # losing its `optimal` certificate). Only when a node LP actually breaks down
    # numerically do we drop the float64-intractable rows and re-solve — sound by
    # superset, and byte-identical on every already-solving node.
    return _uniform_relaxation_delegate(
        model,
        flat_lb,
        flat_ub,
        n_orig,
        convhull_mode,
        rlt_level1=rlt_level1,
        skip_separable_floor=skip_separable_floor,
        skip_convex_lift=skip_convex_lift,
        disc_state=disc_state,
        build_deadline=build_deadline,
    )


# --------------------------------------------------------------------------- #
# Separable objective lower bound (issue #640 Bucket 1 — federation-parity)
#
# Recovered from the #632 federation cutover: a sound constant lower bound on a
# *separable* (minimize-equivalent) objective, derived term by term. Every term
# yields a valid lower bound over the box, and for ANY additive decomposition
# ``f = sum_k g_k`` we have ``min f >= sum_k min g_k`` (each ``g_k`` is >= its own
# box-minimum pointwise), so the sum is a valid global lower bound — soundness
# does NOT require the supports to be disjoint. Recognized shapes:
#   * constant terms;
#   * ``x*exp(x)`` (global inf ``-1/e``; the loss-sign case needs a finite box);
#   * ``cos(integer-affine)`` — exact enumeration over the small integer domain;
#   * single-variable polynomials — vertex/critical-point minimization;
#   * reciprocals ``k/D(x)`` with a provably strictly-positive denominator;
#   * even powers ``c*(E(x))**n`` of a multivariate base (``>= 0`` for ``c>=0``);
#   * affine terms (box-vertex minimization).
# Any unrecognized/unbounded-below term makes the whole bound abstain (``None``),
# so a fabricated bound is never returned (see the guard regressions in
# test_amp.py — e.g. ``-x*exp(x)`` on a free box stays unbounded). The engine
# (uniform_relax.build_uniform_relaxation) consumes this via a sound
# ``obj_lin >= sep_lb`` cut; see its call site for the validity argument.
# --------------------------------------------------------------------------- #
def _finite_bound_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if not _is_effectively_finite(value):
        return None
    return value


def _expand_integer_powers_for_relaxation(expr: Expression, model: Model) -> Expression:
    """Expand small integer powers of affine expressions for existing monomial lifts."""

    def visit(node: Expression) -> Expression:
        if isinstance(node, BinaryOp):
            left = visit(node.left)
            right = visit(node.right)
            if node.op == "**":
                exp = _constant_value(right)
                if exp is not None:
                    n = int(exp)
                    if exp == n and 2 <= n <= _MAX_OBJECTIVE_LIFT_POWER:
                        base = left
                        product = base
                        for _ in range(n - 1):
                            product = BinaryOp("*", product, base)
                        return distribute_products(product)
            # ``left``/``right`` are already fully distributed by the recursive
            # ``visit``. Only a ``*`` combines them in a way that can create a new
            # product-of-sums needing (re-)distribution; ``+``/``-``/``/`` of
            # distributed children are already in distributed form. Re-running
            # ``distribute_products`` on those merely re-walks the whole subtree —
            # an O(N^2) blow-up on a large sum objective (qap's 21 424-term
            # objective spent ~30 s here, #654). Reconstruct them directly, and
            # preserve node identity when nothing changed so id()-keyed lift maps
            # still match. This is bound-neutral: the emitted expression is
            # identical, only the redundant re-walk is removed.
            if node.op == "*":
                return distribute_products(BinaryOp("*", left, right))
            if left is node.left and right is node.right:
                return node
            return BinaryOp(node.op, left, right)
        if isinstance(node, UnaryOp):
            return UnaryOp(node.op, visit(node.operand))
        if isinstance(node, SumExpression):
            return SumExpression(visit(node.operand), axis=node.axis)
        if isinstance(node, SumOverExpression):
            return SumOverExpression([visit(term) for term in node.terms])
        # Preserve FunctionCall object identity so existing univariate lift maps
        # keyed by id(expr) remain usable during branch linearization.
        return node

    return distribute_products(visit(expr))


def _expression_lower_bound_for_lift(
    expr: Expression,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[float]:
    expanded = _expand_integer_powers_for_relaxation(expr, model)
    lower = _separable_objective_lower_bound(expanded, model, flat_lb, flat_ub)
    return _finite_bound_or_none(lower)


def _expression_upper_bound_for_lift(
    expr: Expression,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[float]:
    lower_of_negated = _expression_lower_bound_for_lift(
        UnaryOp("neg", expr),
        model,
        flat_lb,
        flat_ub,
    )
    if lower_of_negated is None:
        return None
    return -lower_of_negated


def _sorted_unique_points(points: list[float]) -> list[float]:
    """Return sorted points with near-duplicates removed."""
    unique: list[float] = []
    for point in sorted(float(p) for p in points):
        if not unique or abs(point - unique[-1]) > 1e-12:
            unique.append(point)
    return unique


def _flatten_additive_terms(
    expr: Expression, scale: float, out: list[tuple[float, Expression]]
) -> None:
    if isinstance(expr, BinaryOp) and expr.op == "+":
        _flatten_additive_terms(expr.left, scale, out)
        _flatten_additive_terms(expr.right, scale, out)
        return
    if isinstance(expr, BinaryOp) and expr.op == "-":
        _flatten_additive_terms(expr.left, scale, out)
        _flatten_additive_terms(expr.right, -scale, out)
        return
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        _flatten_additive_terms(expr.operand, -scale, out)
        return
    if isinstance(expr, SumOverExpression):
        for term in expr.terms:
            _flatten_additive_terms(term, scale, out)
        return
    out.append((scale, expr))


def _flatten_product_factors(expr: Expression, out: list[Expression]) -> None:
    if isinstance(expr, BinaryOp) and expr.op == "*":
        _flatten_product_factors(expr.left, out)
        _flatten_product_factors(expr.right, out)
        return
    out.append(expr)


def _monomial_power_term(expr: Expression, model: Model) -> Optional[tuple[int, int]]:
    flat = _get_flat_index(expr, model)
    if flat is not None:
        return flat, 1
    if isinstance(expr, BinaryOp) and expr.op == "**" and isinstance(expr.right, Constant):
        base = _get_flat_index(expr.left, model)
        if base is None:
            return None
        exp_val = float(expr.right.value)
        n_int = int(exp_val)
        if exp_val == n_int and n_int >= 1:
            return base, n_int
    return None


def _match_scaled_monomial(expr: Expression, model: Model) -> Optional[tuple[float, int, int]]:
    factors: list[Expression] = []
    _flatten_product_factors(expr, factors)
    scalar = 1.0
    var_idx: Optional[int] = None
    power_total = 0
    for factor in factors:
        const = _constant_value(factor)
        if const is not None:
            scalar *= const
            continue
        power_term = _monomial_power_term(factor, model)
        if power_term is None:
            return None
        factor_var, factor_power = power_term
        if var_idx is None:
            var_idx = factor_var
        elif var_idx != factor_var:
            return None
        power_total += factor_power
    if var_idx is None or power_total < 1:
        return None
    return scalar, var_idx, power_total


def _match_x_exp_product(expr: Expression, model: Model) -> Optional[tuple[float, int]]:
    factors: list[Expression] = []
    _flatten_product_factors(expr, factors)
    scalar = 1.0
    var_idx: Optional[int] = None
    exp_arg_idx: Optional[int] = None
    for factor in factors:
        const = _constant_value(factor)
        if const is not None:
            scalar *= const
            continue
        flat = _get_flat_index(factor, model)
        if flat is not None:
            if var_idx is not None:
                return None
            var_idx = flat
            continue
        if isinstance(factor, FunctionCall) and factor.func_name == "exp" and len(factor.args) == 1:
            arg_idx = _get_flat_index(factor.args[0], model)
            if arg_idx is None or exp_arg_idx is not None:
                return None
            exp_arg_idx = arg_idx
            continue
        return None
    if var_idx is None or exp_arg_idx is None or var_idx != exp_arg_idx:
        return None
    return scalar, var_idx


def _safe_x_exp_value(x: float) -> Optional[float]:
    if not np.isfinite(x) or x > _MAX_FINITE_EXP_ARG:
        return None
    if x < -745.0:
        return 0.0
    return float(x * np.exp(x))


def _x_exp_upper_bound(var_idx: int, flat_lb: np.ndarray, flat_ub: np.ndarray) -> Optional[float]:
    lb = float(flat_lb[var_idx])
    ub = float(flat_ub[var_idx])
    if not (_is_effectively_finite(lb) and _is_effectively_finite(ub)):
        return None
    values = [_safe_x_exp_value(lb), _safe_x_exp_value(ub)]
    finite_values = [value for value in values if value is not None and np.isfinite(value)]
    if len(finite_values) != len(values):
        return None
    return max(finite_values)


def _is_cos_call(expr: Expression) -> bool:
    return isinstance(expr, FunctionCall) and expr.func_name == "cos" and len(expr.args) == 1


def _flat_variable_types(model: Model) -> list[VarType]:
    types: list[VarType] = []
    for var in model._variables:
        types.extend([var.var_type] * var.size)
    return types


def _integer_domain_values(
    var_idx: int,
    flat_types: list[VarType],
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[range]:
    var_type = flat_types[var_idx]
    if var_type not in (VarType.BINARY, VarType.INTEGER):
        return None
    lb_i = float(flat_lb[var_idx])
    ub_i = float(flat_ub[var_idx])
    if not (_is_effectively_finite(lb_i) and _is_effectively_finite(ub_i)):
        return None
    lo = int(np.ceil(lb_i - 1e-9))
    hi = int(np.floor(ub_i + 1e-9))
    if var_type == VarType.BINARY:
        lo = max(lo, 0)
        hi = min(hi, 1)
    if lo > hi:
        return None
    return range(lo, hi + 1)


def _integer_affine_cos_lower_bound(
    expr: Expression,
    scale: float,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[float]:
    """Return exact lower bound for scale*cos(integer-affine expr) on a small box."""
    if not isinstance(expr, FunctionCall) or expr.func_name != "cos" or len(expr.args) != 1:
        return None
    n_vars = len(flat_lb)
    try:
        terms, const = _linearize_affine_expr_sparse(expr.args[0], model, n_vars)
    except ValueError:
        return None

    flat_types = _flat_variable_types(model)
    entries: list[tuple[float, range]] = []
    n_values = 1
    # Ascending index order, matching the dense ``enumerate(coeff)`` walk this
    # replaces — the enumeration below is over a cartesian product, so the entry
    # order is not observable in the result, but keeping it makes the two forms
    # trivially comparable. Sparse: the dense walk was O(n_vars) per call (#875).
    for var_idx in sorted(terms):
        c = terms[var_idx]
        if abs(c) <= 1e-12:
            continue
        if not (0 <= var_idx < n_vars):
            return None  # unindexable support: the dense form raised here
        values = _integer_domain_values(var_idx, flat_types, flat_lb, flat_ub)
        if values is None:
            return None
        n_values *= len(values)
        if n_values > _MAX_INTEGER_COS_ENUM:
            return None
        entries.append((c, values))

    if not entries:
        value = scale * float(np.cos(const))
        return value if np.isfinite(value) else None

    best = np.inf
    for assignment in itertools.product(*(values for _c, values in entries)):
        arg = float(const)
        for (c, _values), value in zip(entries, assignment):
            arg += c * float(value)
        best = min(best, scale * float(np.cos(arg)))
    return float(best) if np.isfinite(best) else None


def _scaled_affine_lower_bound(
    expr: Expression,
    scale: float,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[float]:
    n_vars = len(flat_lb)
    try:
        terms, const = _linearize_affine_expr_sparse(expr, model, n_vars)
    except ValueError:
        return None
    want_lower = scale >= 0.0
    bound = float(const)
    # Sparse walk over the body's support; the dense form zipped all ``n_vars``
    # coefficients against the box on every call (#875). Ascending index order keeps
    # the floating-point accumulation identical to the dense walk's.
    for var_idx in sorted(terms):
        c = terms[var_idx]
        if abs(c) <= 1e-12:
            continue
        if not (0 <= var_idx < n_vars):
            return None  # unindexable support: the dense form raised here
        lb_i = flat_lb[var_idx]
        ub_i = flat_ub[var_idx]
        chosen = float(lb_i) if (c >= 0.0) == want_lower else float(ub_i)
        if not _is_effectively_finite(chosen):
            return None
        bound += c * chosen
    return scale * bound


def _evaluate_polynomial(coeffs: dict[int, float], x: float) -> Optional[float]:
    max_power = max(coeffs)
    value = 0.0
    for power in range(max_power, -1, -1):
        value = value * x + float(coeffs.get(power, 0.0))
        if not np.isfinite(value):
            return None
    return float(value)


def _polynomial_lower_bound(
    coeffs: dict[int, float],
    lb: float,
    ub: float,
) -> Optional[float]:
    clean = {power: coeff for power, coeff in coeffs.items() if abs(coeff) > 1e-12}
    if not clean:
        return 0.0
    max_power = max(clean)
    if max_power == 0:
        return float(clean[0])

    leading = float(clean[max_power])
    lo_unbounded = not _is_effectively_finite(lb)
    hi_unbounded = not _is_effectively_finite(ub)
    if hi_unbounded and leading < 0.0:
        return None
    if lo_unbounded:
        if max_power % 2 == 0 and leading < 0.0:
            return None
        if max_power % 2 == 1 and leading > 0.0:
            return None

    candidates: list[float] = []
    if not lo_unbounded:
        candidates.append(float(lb))
    if not hi_unbounded:
        candidates.append(float(ub))

    deriv_coeffs = [power * clean.get(power, 0.0) for power in range(max_power, 0, -1)]
    roots = np.roots(deriv_coeffs) if deriv_coeffs else np.array([])
    for root in roots:
        if abs(float(np.imag(root))) > 1e-9:
            continue
        x = float(np.real(root))
        if (lo_unbounded or x >= lb - 1e-9) and (hi_unbounded or x <= ub + 1e-9):
            candidates.append(x)

    values: list[float] = []
    for x in _sorted_unique_points(candidates):
        value = _evaluate_polynomial(clean, x)
        if value is not None and np.isfinite(value):
            values.append(value)
    if not values:
        return None
    return min(values)


def _reciprocal_term_lower_bound(
    scaled_numerator: float,
    denominator: Expression,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[float]:
    """Rigorous constant lower bound for ``scaled_numerator / denominator(x)``.

    Encloses ``denominator`` over the box via the same separable-polynomial
    machinery used for objective lifts (``_expression_lower_bound_for_lift`` /
    ``_expression_upper_bound_for_lift``), which recovers ``D_lo``/``D_hi`` from a
    distributed quadratic by per-variable vertex minimization — crucially, this
    works on the live solve's already-distributed denominator (e.g. ``0.1 +
    x0**2 - 8*x0 + 16 + ...``) where a naive interval evaluation of ``x*x`` on an
    unbounded box would collapse to ``[-inf, inf]``. When the enclosure is
    strictly positive (``D_lo > 0``), ``1/D`` is decreasing in ``D`` so the term
    ``k/D`` is minimized at ``D_hi`` when ``k > 0`` and at ``D_lo`` when
    ``k < 0``. Returns ``None`` (caller abstains) if the denominator cannot be
    proven strictly positive or the bound is not finite.
    """
    if abs(scaled_numerator) <= 1e-12:
        return 0.0
    denom_lo = _expression_lower_bound_for_lift(denominator, model, flat_lb, flat_ub)
    denom_hi = _expression_upper_bound_for_lift(denominator, model, flat_lb, flat_ub)
    # A strictly-positive, finite lower end is required for a sound reciprocal;
    # _expression_lower_bound_for_lift returns None when it cannot prove one.
    if denom_lo is None or not np.isfinite(denom_lo) or denom_lo <= 1e-12:
        return None
    if scaled_numerator > 0.0:
        # k/D minimized at the largest D; an unbounded/unknown D_hi drives the
        # positive term toward 0 from above, which is still a valid lower bound.
        bound = (
            0.0
            if (denom_hi is None or not np.isfinite(denom_hi))
            else (scaled_numerator / denom_hi)
        )
    else:
        bound = scaled_numerator / denom_lo
    if not np.isfinite(bound):
        return None
    return float(bound)


def _match_scaled_even_power(
    term: Expression, scale: float
) -> Optional[tuple[float, Expression, int]]:
    """Match ``scale * c * (base)**n`` with ``n`` a positive even integer.

    Returns ``(coeff, base, n)`` where ``coeff`` folds ``scale`` and every
    constant factor of the product. Exactly one even-power factor is allowed;
    any other non-constant factor (or a second power) disqualifies the match.
    """
    factors: list[Expression] = []
    _flatten_product_factors(term, factors)
    coeff = scale
    base: Optional[Expression] = None
    power = 0
    for factor in factors:
        const = _constant_value(factor)
        if const is not None:
            coeff *= const
            continue
        if (
            isinstance(factor, BinaryOp)
            and factor.op == "**"
            and isinstance(factor.right, Constant)
        ):
            exp_val = float(factor.right.value)
            n_int = int(round(exp_val))
            if abs(exp_val - n_int) < 1e-12 and n_int >= 2 and n_int % 2 == 0:
                if base is not None:
                    return None
                base = factor.left
                power = n_int
                continue
        return None
    if base is None:
        return None
    return coeff, base, power


def _count_distinct_scalar_refs(expr: Expression, model: Model) -> int:
    """Count distinct scalar variable columns referenced by ``expr``.

    Used to gate the even-power lower bound to genuinely multivariate bases:
    a single-variable square is handled more tightly by the distribute /
    polynomial path (which combines it with any linear term in the same
    variable), so only multivariate bases — which that path cannot bound at
    all — are routed through the sum-of-squares relaxation.
    """
    seen: set = set()

    def visit(e: Expression) -> None:
        idx = _get_flat_index(e, model)
        if idx is not None:
            seen.add(idx)
            return
        if isinstance(e, Variable):
            seen.add(("var", id(e)))
            return
        if isinstance(e, IndexExpression):
            visit(e.base)
            return
        if isinstance(e, BinaryOp):
            visit(e.left)
            visit(e.right)
        elif isinstance(e, UnaryOp):
            visit(e.operand)
        elif isinstance(e, FunctionCall):
            for a in e.args:
                visit(a)
        elif isinstance(e, SumExpression):
            visit(e.operand)
        elif isinstance(e, SumOverExpression):
            for t in e.terms:
                visit(t)

    visit(expr)
    return len(seen)


def _even_power_term_lower_bound(
    coeff: float,
    base: Expression,
    n: int,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[float]:
    """Rigorous constant lower bound for ``coeff * base(x)**n`` with ``n`` even.

    ``base**n >= 0`` always, so for ``coeff >= 0`` the term is nonnegative — a
    valid lower bound of ``0`` even when ``base`` cannot be enclosed. When the
    box yields a finite enclosure ``base in [bl, bh]`` the bound tightens to the
    vertex minimum of ``base**n`` (``0`` if the interval straddles zero, else
    ``min(|bl|, |bh|)**n``). For ``coeff < 0`` the term is maximized in
    magnitude at the larger ``|base|`` endpoint, so a finite enclosure is
    required; ``None`` (caller abstains) when it is unavailable.
    """
    bl = _expression_lower_bound_for_lift(base, model, flat_lb, flat_ub)
    bh = _expression_upper_bound_for_lift(base, model, flat_lb, flat_ub)
    if coeff >= 0.0:
        if bl is None or bh is None or not (np.isfinite(bl) and np.isfinite(bh)):
            return 0.0
        if bl <= 0.0 <= bh:
            pow_min = 0.0
        else:
            pow_min = min(abs(bl), abs(bh)) ** n
        return float(coeff * pow_min)
    if bl is None or bh is None or not (np.isfinite(bl) and np.isfinite(bh)):
        return None
    pow_max = max(abs(bl), abs(bh)) ** n
    bound = coeff * pow_max
    if not np.isfinite(bound):
        return None
    return float(bound)


def _separable_objective_lower_bound(
    expr: Expression,
    model: Model,
    flat_lb: np.ndarray,
    flat_ub: np.ndarray,
) -> Optional[float]:
    """Compute a conservative constant lower bound for simple separable objectives.

    ``expr`` is flattened additively and matched term-by-term.  Reciprocal terms
    ``k / D(x)`` are matched on the term's ORIGINAL (un-distributed) structure so
    the denominator's square/power shape survives for a sound interval enclosure;
    every other term is distributed individually before the polynomial / affine
    matchers run (the union over terms equals distributing the whole expression,
    so non-reciprocal behavior is unchanged).
    """
    terms: list[tuple[float, Expression]] = []
    _flatten_additive_terms(expr, 1.0, terms)

    total = 0.0
    polynomial_terms: dict[int, dict[int, float]] = {}

    def _accumulate_simple_term(scale: float, term: Expression) -> bool:
        """Fold one already-distributed, non-reciprocal term into the running bound.

        Returns ``False`` (caller abstains entirely) when the term is not one of
        the recognized separable shapes.
        """
        nonlocal total
        if abs(scale) <= 1e-12:
            return True
        const = _constant_value(term)
        if const is not None:
            total += scale * const
            return True

        x_exp = _match_x_exp_product(term, model)
        if x_exp is not None:
            scalar, var_idx = x_exp
            term_scale = scale * scalar
            if abs(term_scale) <= 1e-12:
                return True
            if term_scale > 0.0:
                total += term_scale * (-1.0 / np.e)
                return True
            upper = _x_exp_upper_bound(var_idx, flat_lb, flat_ub)
            if upper is None:
                return False
            total += term_scale * upper
            return True

        if _is_cos_call(term):
            integer_lb = _integer_affine_cos_lower_bound(term, scale, model, flat_lb, flat_ub)
            total += integer_lb if integer_lb is not None else -abs(scale)
            return True

        monomial = _match_scaled_monomial(term, model)
        if monomial is not None:
            scalar, var_idx, power = monomial
            polynomial_terms.setdefault(var_idx, {})
            polynomial_terms[var_idx][power] = (
                polynomial_terms[var_idx].get(power, 0.0) + scale * scalar
            )
            return True

        affine_bound = _scaled_affine_lower_bound(term, scale, model, flat_lb, flat_ub)
        if affine_bound is None:
            return False
        total += affine_bound
        return True

    for scale, term in terms:
        if abs(scale) <= 1e-12:
            continue

        # Reciprocal term ``k / D(x)`` with a strictly-positive denominator (e.g.
        # ex8_1_6's ``-1/(0.1 + (x0-4)**2 + (x1-4)**2)``). The MILP linearizer
        # cannot relax a non-constant division, so without this the whole
        # objective is dropped and AMP can never certify. A rigorous interval
        # enclosure ``D in [D_lo, D_hi]`` with ``D_lo > 0`` yields a valid
        # constant lower bound for the term: ``k/D_hi`` when ``k > 0`` else
        # ``k/D_lo`` (``1/D`` is decreasing in ``D``). The bound tightens as B&B
        # branching shrinks the box, eventually enabling certification. Matched
        # on the un-distributed term so ``D``'s ``(x-a)**2`` shape survives for a
        # tight interval enclosure (distribution would expand it to a polynomial
        # whose naive interval enclosure is uselessly loose on a wide box).
        recip = _match_scaled_constant_division(term, scale)
        if recip is not None:
            recip_bound = _reciprocal_term_lower_bound(recip[0], recip[1], model, flat_lb, flat_ub)
            if recip_bound is None:
                return None
            total += recip_bound
            continue

        # Even-power term ``c * (E(x))**n`` (n even) with a *multivariate* base,
        # e.g. Rosenbrock's ``100 * (x1 - x0**2)**2``. Distributing it yields a
        # bilinear/multivariate polynomial whose cross terms no single-variable
        # matcher accepts, so the whole objective would be dropped. But a square
        # is nonnegative regardless of its argument's structure: for ``c >= 0``
        # the term is ``>= 0`` (tightened to the vertex minimum of ``E**n`` when
        # the box encloses ``E``). Recognizing it on the un-distributed term lets
        # AMP certify sum-of-squares objectives at the root. Single-variable
        # bases are left to the polynomial path, which combines them with any
        # linear term in the same variable for a strictly tighter bound.
        even_pow = _match_scaled_even_power(term, scale)
        if even_pow is not None and _count_distinct_scalar_refs(even_pow[1], model) >= 2:
            coeff, base, power = even_pow
            ep_bound = _even_power_term_lower_bound(coeff, base, power, model, flat_lb, flat_ub)
            if ep_bound is None:
                return None
            total += ep_bound
            continue

        # Distribute this single term and fold each resulting sub-term through the
        # simple-shape matchers (polynomial path needs the expanded form).
        sub_terms: list[tuple[float, Expression]] = []
        _flatten_additive_terms(distribute_products(term), scale, sub_terms)
        for sub_scale, sub_term in sub_terms:
            if not _accumulate_simple_term(sub_scale, sub_term):
                return None

    for var_idx, coeffs in polynomial_terms.items():
        lower = _polynomial_lower_bound(coeffs, float(flat_lb[var_idx]), float(flat_ub[var_idx]))
        if lower is None:
            return None
        total += lower

    if not np.isfinite(total):
        return None
    return float(total)
