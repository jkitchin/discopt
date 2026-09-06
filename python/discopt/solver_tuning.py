"""Typed, per-call solver tuning — the Pythonic replacement for ``DISCOPT_*`` flags.

The spatial-B&B relaxation has a number of advanced tuning levers (RLT families,
McCormick separation toggles, node-bound mode, …) that were historically read
straight from ``DISCOPT_*`` environment variables at scattered points inside the
relaxer. That made them global process state: not per-``Model``, not per-solve,
not thread-safe, invisible to ``help(model.solve)``, and unvalidated.

:class:`SolverTuning` collects them into one validated, typed object. Each field
still *defaults* to its ``DISCOPT_*`` env var (read at instantiation, not at
import — so it is never frozen and an explicit field always wins), so existing
env-var workflows keep working as deprecated defaults while the object is the
supported, discoverable surface::

    from discopt import SolverTuning
    model.solve(tuning=SolverTuning(rlt_quad=False, node_bound_mode="nlp"))

Internally ``solve_model`` resolves the object once and publishes it on a
:class:`~contextvars.ContextVar`; the relaxer read sites call :func:`current`
instead of touching ``os.environ``. Outside a solve, :func:`current` falls back
to a fresh env-resolved instance, so direct relaxer use (e.g. in tests) is
unaffected.
"""

from __future__ import annotations

import math
import os
from contextvars import ContextVar
from dataclasses import dataclass, field, fields
from typing import Optional


def _env_flag(name: str, *, default: bool) -> bool:
    """``DISCOPT_<name>`` as a boolean (``"0"`` is false, anything else true)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw != "0"


def _env_cut_inherit(name: str) -> Optional[bool]:
    """``DISCOPT_CUT_INHERIT`` as a tri-state:

    * unset ⇒ ``False`` — **force-off is the shipped default** (opt-in flag).
      CUT-INHERIT-GRAD validated the structure gate as broadly beneficial where
      it fires and byte-identical where it does not, BUT surfaced a flag-path
      false-optimal on the pure-integer / MINLP cold-path class (nvs22 certifies
      33.55 vs the oracle 6.058; the nvs06-class reroute C-42 only partially
      fixed). Per CLAUDE.md §1 a false certificate blocks any default-ON flip, so
      the gated behaviour stays OPT-IN until that soundness bug is fixed.
    * ``"0"`` ⇒ ``False`` (explicit force-off, identical to unset today);
    * ``"gated"`` / ``"auto"`` ⇒ ``None`` (structure-gated opt-in: inherit iff a
      non-empty root pool is separated — the pool-fires predicate);
    * anything else (e.g. ``"1"``) ⇒ ``True`` (force-on).
    """
    raw = os.environ.get(name)
    if raw is None:
        return False
    low = raw.strip().lower()
    if low in ("gated", "auto"):
        return None
    return raw != "0"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


#: Default deterministic operation caps for the root integer local search
#: (#912). Both are sized from the measured per-operation costs and the measured
#: consumption of the legacy wall-clock gate over the in-repo MINLPLib corpus;
#: see ``docs/dev/work-budget-calibration-2026-08-01.md`` for the tables.
#:
#: Measured over the 22 ILS-firing in-repo instances, legacy arm at
#: ``time_limit=60`` (so the full 5 s legacy budget): a search that converged on
#: its own used at most **796 evaluations and 217 sub-NLP solves**, while the
#: slowest instance the clock actually cut managed only **13.3 solves/s**
#: (syn05hfsg).
#:
#: Those two facts do not both fit. "Never cut a search the clock let finish"
#: wants ≥ 217 solves; "never exceed the legacy 5 s envelope" wants ≤ 67 on the
#: slowest instance. The gap is not a modelling failure — it *is* the bug: the
#: old gate handed ex1224 217 solves and syn05hfsg 67 in the same five seconds,
#: purely because their sub-NLPs differ 5x in cost. Any deterministic budget
#: must land somewhere between, and this one lands in the middle: 128 solves is
#: ~59 % of the largest natural extent and ~2x the slowest instance's legacy
#: allowance (≈9.6 s there, against a 5 s legacy ceiling).
#:
#: The evaluation cap is not contested — 20 000 is 25x the largest natural
#: consumption, so it only ever stops a genuinely runaway descent.
#: ``docs/dev/work-budget-calibration-2026-08-01.md`` records the A/B that
#: validated the choice.
_ILS_EVAL_BUDGET_DEFAULT = 20_000
_ILS_SOLVE_BUDGET_DEFAULT = 128


@dataclass(frozen=True)
class SolverTuning:
    """Advanced relaxation / branch-and-bound tuning for :meth:`Model.solve`.

    Every field defaults to its legacy ``DISCOPT_*`` environment variable
    (resolved when the instance is created), so a bare ``SolverTuning()`` exactly
    reproduces the env-driven behavior, and any explicitly-set field overrides
    it. All fields are validated on construction.
    """

    # --- #517/#362 NS dual safe bound on numerically-failed node LPs ----------
    node_numerical_dual_bound: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_NODE_NUMERICAL_DUAL_BOUND", default=True)
    )
    """Attach a Neumaier–Shcherbina safe lower bound from the in-house simplex's
    *own* dual candidate when the node LP solve breaks down numerically
    (``DISCOPT_NODE_NUMERICAL_DUAL_BOUND``, default ON since the #362 graduation
    — ``DISCOPT_NODE_NUMERICAL_DUAL_BOUND=0`` restores the legacy no-rescue
    behavior). Fires only where the certified in-house chain (warm →
    equilibrated) produced no bound: the hda-class flowsheets whose
    ill-conditioned LPs defeat phase-2 (#517, see
    ``docs/dev/hda-no-bound-simplex-robustness-2026-07-16.md``) and the nvs05
    certification-edge decline class, where the stashed NS bound is also
    surfaced as ``safe_bound`` on an ``optimal`` generic-path solve so
    ``_certify`` can certify the node instead of declining it into a
    non-rigorous sentinel fathom (#362, see
    ``docs/dev/nvs05-decline-taint-2026-07-16.md``). The NS bound is valid for
    ANY dual vector, so a drifted-basis dual only loosens it, never lifts it
    above the optimum; never fathoms on its own; a finite NS value proves the
    LP is bounded, so it can never fabricate a bound on a genuinely unbounded
    relaxation. No external solver (the removed #517 HiGHS rescue is NOT
    resurrected). Graduation evidence (2026-07-16, in-container): 65-instance
    panel ON-vs-OFF — 0 proofs lost, no bound loosened (one beuster wall-jitter
    artifact, byte-identical in isolation); differential gate GREEN
    (at-least-as-tight per box, feasible-point 0 cuts, worst 1.8e-11);
    graduation_gate cert-neutrality eligible=YES; nvs05 gains its first full
    rigorous certificate (``optimal``, bound 5.47057)."""

    # --- #1064 structured-convex engine for node-bound recovery -----------------
    structured_node_recovery: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_STRUCTURED_NODE_RECOVERY", default=True)
    )
    """Solve ``_pounce_recover_node_bound``'s re-solve with POUNCE's *structured
    convex* engine (``pounce.solve_qp``) instead of the generic callback TNLP
    path (``qp_pounce.solve_qp`` → ``pounce.Problem(problem_obj=_QPCallbacks)``)
    (``DISCOPT_STRUCTURED_NODE_RECOVERY``, default **ON** since the §5 panel
    below; ``=0`` keeps the callback path, which is retained intact).

    This is the same migration ``_solve_node_lp_pounce`` already carries, and
    for the same measured reason: the callback path hides the linear structure,
    so POUNCE's presolve cannot engage and its IPM runs ~100 iterations where
    the structured path presolves/scales and converges in ~20. Node-bound
    recovery was left behind when the MILP node path was moved over, which is
    why a *node* QP costs ~8 ms while recovering the *same* problem costs
    8-33 s.

    Measured (#1064 entry experiment 4, same fixings, both arms at each node,
    4 comparisons/instance): squfl015-060 21.8x, squfl025-040 19.4x,
    squfl020-150 5.9x. Verdicts agreed 4/4 on the first two, and objectives
    where both returned ``optimal`` agreed to 2.8e-7 / 4.8e-7 relative. On
    squfl020-150 the callback arm returned *no answer at all* (limit, ~32 s) on
    all four, where the structured arm settled every one in ~5.5 s (3
    Phase-1-certified infeasible, 1 optimal) — so the disagreement is the legacy
    path failing to answer, not a soundness split.

    Soundness is unchanged: an ``optimal`` result from the structured convex IPM
    is KKT-valid (the property ``_solve_node_lp_pounce`` already relies on) and
    ``primal_infeasible`` is Phase-1-certified; anything else stays ``None`` so
    the caller keeps the node open and never prunes on an unsettled solve.

    Graduation evidence (2026-08-19, §5 regime 2, 69 instances x 60 s, in-repo
    corpus plus the three squfl instances the corpus lacks). This flag does NOT
    graduate on its own: alone it is cert-clean (unsound=0, cert_regressions=0)
    but **not** net-positive — 0 incumbents gained, bound tighter on 1 and
    looser on 5 — which is the ``DISCOPT_CUT_INHERIT`` case exactly. It
    graduates as half of a *pair*, because it is what lets
    ``round_fix_resolve``'s re-solve answer at all on the largest instance:
    with this flag OFF, all 3 rungs of the rounding ladder return ``None``
    after 14-18 s each on squfl020-150; with it ON, rung 1 certifies infeasible
    in 5.5 s and rung 2 returns optimal in 0.13 s. The paired arm
    (``ROUND_FIX_RESOLVE`` + this) scores unsound=0, cert_regressions=0,
    3 incumbents gained / 0 lost, bound tighter 3 / looser 2, nodes fewer 9 /
    more 3, total wall 1196.7 s vs 1210.6 s OFF — cert-clean AND net-positive,
    and better than ``ROUND_FIX_RESOLVE`` alone on every axis, so the pair is
    the shipping configuration.""" ""

    # --- #1064 round-fix-resolve primal heuristic (first incumbent) -------------
    round_fix_resolve: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_ROUND_FIX_RESOLVE", default=True)
    )
    """Round a *fractional* MIQP node relaxation point to the nearest integers,
    fix them, and re-solve for the continuous completion, so a search that never
    lands near-integral can still obtain a first incumbent
    (``DISCOPT_ROUND_FIX_RESOLVE``, default **ON** since the §5 panel recorded
    under ``structured_node_recovery`` above; ``=0`` restores the old
    snap-only behaviour).

    The existing purification (``_pounce_snap_incumbent``) only accepts points
    already integral to within ``_SNAP_TOL`` = 1e-4. On squfl020-150 and
    squfl025-040 (#1064) that gate never opens: both run a full 120 s budget with
    **zero** snap re-solves and finish with no incumbent, hence no primal bound
    and nothing to prune against. Rounding is the missing step, not a faster
    engine.

    Spent only while ``tree.incumbent() is None`` and capped at 64 attempts per
    solve, so a family where every rounding is infeasible cannot consume the
    search. A rounded point can be genuinely infeasible — measured on
    squfl025-040, 7 forced fixings split 2 optimal / 3 Phase-1 infeasible / 2
    unsettled — so a non-optimal verdict is retried once with the most fractional
    coordinate rounded the other way.

    This can only ever produce an *upper* bound: it never prunes, never tightens
    a node bound, and never decertifies, and the completed point is run through
    the same ``_node_point_feasible`` gate as a snapped one before injection. A
    wrong guess costs time, not correctness."""

    # --- #671 LP iterative refinement (RHS-regularized dual + rigorous NS) ------
    lp_iterative_refinement: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_LP_ITERATIVE_REFINEMENT", default=False)
    )
    """When a node LP breaks down numerically (the hda-class ill-conditioned
    McCormick relaxations whose near-singular bases the float64 simplex cannot
    certify), recover a *tight* dual bound by re-solving a few RHS-regularized
    neighbours ``[A|I] z = b + tau`` in the **in-house** simplex and keeping the
    tightest Neumaier–Shcherbina safe bound the recovered duals imply
    (``DISCOPT_LP_ITERATIVE_REFINEMENT``, default OFF — issue #671). A small
    ``tau`` perturbs the RHS just enough for the simplex to certify the
    well-conditioned neighbour and hand back a good multiplier; the NS bound is
    then evaluated against the **original** ``b`` (never ``b+tau``), so it is a
    valid lower bound for *any* recovered dual — clamping/regularization can only
    tighten it, never lift it above the optimum. The reported bound is the **max**
    over the sweep and candidate A's drifted-dual bound (#662), so it is never
    looser than the candidate-A floor it supersedes and never unsound. Fires only
    on the numerical-failure path (root / failure-triggered), never the hot
    per-node engine, and uses **no external solver** (the #517 HiGHS rescue is not
    resurrected). On hda this moves the root dual bound from candidate A's
    −1.80e10 to ≈ −6.47e4 (the true root McCormick value). See
    ``docs/dev/issue-671-gsw-iterative-refinement-2026-07-18.md`` and the
    `crates/discopt-core/src/lp/simplex/refine.rs` kernel."""

    # --- #671 float64-intractable-row filter (hda certification path) ----------
    relax_row_filter: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_RELAX_ROW_FILTER", default=True)
    )
    """**Failure-triggered**, **default ON** (#671 graduated 2026-07-18; opt out
    with ``DISCOPT_RELAX_ROW_FILTER=0``). When a node LP breaks down without a
    certified
    verdict (`numerical`, or a spurious `infeasible` with no Farkas proof — the
    hda-class ill-conditioned relaxations), drop the rows whose coefficients
    float64 cannot resolve at the feasibility tolerance (nonzero coefficient
    magnitudes spanning > 1e6 orders, or outside ``[1e-8, 1e8]``) and re-solve once
    (``DISCOPT_RELAX_ROW_FILTER``, default ON — issue #671). Fires in
    ``mccormick_lp._solve_at_node_impl`` after the primary solve; **not** at build
    time. **Sound by construction**: removing relaxation rows yields a superset
    feasible region — a valid (weaker) outer approximation; the bound can only
    loosen, never falsify.

    Failure-triggered (not always-on) *because* the in-repo differential panel
    showed an always-on build-time filter drops rows carrying genuine tightness on
    already-solving instances — 10/66 regressions, including **nvs09 losing its
    `optimal` certificate** (see the panel in
    ``discopt_benchmarks/results/issue671/rowfilter_diff_panel.py``). Firing only
    on a failed solve makes the flag **byte-identical on every already-solving
    node** (the un-filtered solve is `optimal`/Farkas-`infeasible` there), while
    still recovering hda: its root LP false-fails, the filter drops the 130
    float64-intractable rows (raw spread 2.837e26, κ≈1e14; measured to carry ZERO
    root tightness), and the in-house simplex then solves cleanly with the tight
    NS-certified bound — no τ-sweep, no factorization hardening, no external
    solver.

    **Graduated default-ON** (§5, 2026-07-18): the graduation panel over the
    66-instance in-repo corpus + hda is cert-clean — every already-solving
    instance is byte-identical (the filter is inert there: ``rows_dropped == 0``,
    proven directly rather than via the noisy, non-deterministic ``node_count``
    proxy) — and net-positive (hda ``−1.80e10 → −64473.44``, sound ≤ opt). Sound
    SOFT losses (looser *partial* bounds when the filter fires on both-arms-
    timeout ill-conditioned instances, e.g. bchoco07/08) are acceptable per the
    §5 net-positive rule; a scoped interval-fallback max-combine on filtered nodes
    is the recorded follow-up tightening. Opt out with
    ``DISCOPT_RELAX_ROW_FILTER=0`` (the legacy no-filter path is intact). See
    ``docs/dev/hda-certification-rowfilter-entry-2026-07-18.md`` and
    ``docs/dev/issue-671-resolution-plan-2026-07-18.md``."""

    # --- #309 sharp NS safe-bound margin ---------------------------------------
    ns_sharp_margin: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_NS_SHARP_MARGIN", default=True)
    )
    """Replace the flat ``1e-9``-relative Neumaier–Shcherbina evaluation margin
    with a rigorous forward-error bound computed from the actual data (Higham
    dot-product gammas + interval corners on sign-uncertain reduced costs)
    (``DISCOPT_NS_SHARP_MARGIN``, default ON; ``=0`` restores the flat margin).
    Graduated default-ON 2026-07-16 on the owner's direction after the
    66-instance differential panel passed (incorrect_count=0, no certification
    regression; docs/dev/integer-ratio-partition-2026-07-16.md §5b). On
    magnitude ~1e5 decompositions (gear4 piece LPs) the flat margin costs
    2.9e-4 of every certified LP bound; the sharp margin costs ~1e-6. The sharp
    path also *abstains* when a sign-uncertain reduced cost sits next to an
    unbounded box side (the legacy path silently contributes 0 there — a latent
    soundness gap the sharp path closes), so it can return ``None`` where the
    legacy path returned a value — never the reverse."""

    # --- RLT (reformulation-linearization) families ---------------------------
    rlt: bool = field(default_factory=lambda: _env_flag("DISCOPT_RLT", default=False))
    """Legacy whole-relaxation RLT toggle (``DISCOPT_RLT``). The ``rlt=`` argument
    to :meth:`Model.solve` is the primary control; this OR-s in alongside it."""

    rlt_quad: bool = field(default_factory=lambda: _env_flag("DISCOPT_RLT_QUAD", default=True))
    """Quadratic RLT row generation (``DISCOPT_RLT_QUAD``, default on)."""

    rlt_sparse_auto: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_RLT_SPARSE_AUTO", default=False)
    )
    """Structure-aware widening of the RLT auto-engage gate for **sparse-bilinear**
    models (``DISCOPT_RLT_SPARSE_AUTO``, default **off**; issue #727).

    The default RLT auto policy gates build-time level-1 RLT and the per-node RLT
    cut family on a raw *variable count* (``_AUTO_RLT_LEVEL1_MAX_VARS`` /
    ``_AUTO_CUTS_MAX_VARS`` in ``solver.py``). That is a poor cost proxy: RLT's cost
    is driven by the number of lifted product columns / rows, not the variable count.
    A pooling / bilinear-flow network has a *sparse* bilinear structure — the number
    of product terms grows ~linearly with the variable count — so its RLT relaxation
    stays small and solvable well past the raw-count cap, while a *dense* QCQP grows
    its products quadratically and is correctly excluded.

    When on, the auto gate additionally admits a model whose product-term count is
    within ``rlt_sparse_max_terms`` AND whose variable count is within
    ``rlt_sparse_max_vars`` — the sparse-bilinear envelope. RLT is always sound (a
    constraint×bound-factor product is non-negative at every feasible point), so this
    only ever trades relaxation size for bound tightness, never correctness.
    Bound-changing → default-off pending the corpus-wide differential graduation panel
    (see ``docs/dev/performance-plan.md``)."""

    rlt_sparse_max_vars: int = field(
        default_factory=lambda: _env_int("DISCOPT_RLT_SPARSE_MAX_VARS", 200)
    )
    """Variable-count ceiling for the sparse-bilinear RLT widening
    (``DISCOPT_RLT_SPARSE_MAX_VARS``, default 200). Bounds the per-node re-solve cost
    of the enlarged relaxation when a model does not close at the root. Only consulted
    when ``rlt_sparse_auto`` is on."""

    rlt_sparse_max_terms: int = field(
        default_factory=lambda: _env_int("DISCOPT_RLT_SPARSE_MAX_TERMS", 300)
    )
    """Product-term (lifted-column) budget for the sparse-bilinear RLT widening
    (``DISCOPT_RLT_SPARSE_MAX_TERMS``, default 300). Counts bilinear + trilinear +
    multilinear product terms; caps the RLT relaxation size directly, so a dense QCQP
    (products ~ n^2) is excluded while a sparse pooling network (products ~ n) is
    admitted. Only consulted when ``rlt_sparse_auto`` is on."""

    rlt_sparse_root_probe: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_RLT_SPARSE_ROOT_PROBE", default=True)
    )
    """Root-**productivity** gate on the sparse-bilinear RLT widening
    (``DISCOPT_RLT_SPARSE_ROOT_PROBE``, default **on** when ``rlt_sparse_auto`` is on;
    issue #727).

    Structure alone is *necessary but not sufficient*: a sparse-bilinear model is
    only worth the enlarged per-node relaxation if RLT actually **tightens the root
    bound**. RLT helps precisely when it closes / near-closes the root (a pooling
    network — RLT is paid once and the tree collapses); it is net-*negative* when the
    root stays open (a heat-exchanger network — the heavier node LP just starves
    branching and the incumbent search, with no bound gain). Measured root gain
    ``|b_rlt − b_noRLT| / (|b_noRLT| + 1)``: pooling ≈ 0.68, heatexch ≈ 1e-11 — ten
    orders of magnitude apart.

    When on, the widening additionally requires a bounded root probe to show a
    relative root-bound improvement ≥ ``rlt_sparse_min_root_gain``. The probe solves
    the root McCormick LP with and without RLT once at setup (cheap: the size ceiling
    keeps it small); on any probe failure the widening is *declined* (never regress the
    default). Set to off to recover the structure-only gate (for A/B)."""

    rlt_sparse_min_root_gain: float = field(
        default_factory=lambda: _env_float("DISCOPT_RLT_SPARSE_MIN_ROOT_GAIN", 1e-2)
    )
    """Minimum relative root-bound improvement for the productivity gate to engage the
    sparse RLT widening (``DISCOPT_RLT_SPARSE_MIN_ROOT_GAIN``, default 1e-2 = 1%). The
    RLT-inert vs RLT-productive populations are ~10 orders of magnitude apart
    (heatexch 1e-11 vs pooling 0.68), so the exact value is not sensitive; 1% requires
    a real, non-noise tightening. Only consulted when ``rlt_sparse_root_probe`` is on."""

    rlt_quad_max: int = field(default_factory=lambda: _env_int("DISCOPT_RLT_QUAD_MAX", 256))
    """Column cap for quadratic RLT (``DISCOPT_RLT_QUAD_MAX``, default 256)."""

    rlt_lineq: bool = field(default_factory=lambda: _env_flag("DISCOPT_RLT_LINEQ", default=False))
    """Build-time RLT products of *linear equality* constraints with the original
    variables (``DISCOPT_RLT_LINEQ``, default **off** pending the graduation panel).

    For a linear equality ``a'x + c == 0`` and a variable ``x_j``, the product
    ``(a'x + c) * x_j == 0`` linearizes over the lifted product columns to
    ``sum_i a_i X_ij + c x_j == 0`` — the Sherali–Adams level-1 constraint-factor
    row. It holds with equality at every feasible point, so it never cuts one.

    Unlike the bound-factor RLT families (``rlt``/``rlt_quad``), this row contains
    **no box data**: it is identical at every node, which is why it can be emitted
    once as a fixed row and carried into the native spatial kernel unchanged.

    Motivation (measured, QPLIB continuous nonconvex QPs, root LP bound vs the
    published optimum): the McCormick-only root bound is hopeless on this class,
    and *all* of the recoverable gap on the instances that respond comes from this
    equality family — the bound-factor rows added nothing on top of it:

    ==============  ===========  ================  ==========  ===========
    instance        McCormick    +linear-eq RLT    optimum     gap closed
    ==============  ===========  ================  ==========  ===========
    QPLIB_1157        -14.8046          -11.7716    -10.9482        78.6 %
    QPLIB_1493        -85.6492          -65.5915    -43.1604        47.2 %
    QPLIB_1143       -140.3576         -114.7060    -57.2467        30.9 %
    QPLIB_1423        -25.6777          -22.6093    -14.9675        28.6 %
    QPLIB_1507        -18.1057          -15.8070     -8.3014        23.4 %
    ==============  ===========  ================  ==========  ===========

    No bound crossed its published optimum on any instance in the probe."""

    rlt_lineq_max: int = field(default_factory=lambda: _env_int("DISCOPT_RLT_LINEQ_MAX", 4096))
    """New-column cap for linear-equality RLT (``DISCOPT_RLT_LINEQ_MAX``, default
    4096). Products already registered by the base decomposition are free; only
    columns this pass has to lift itself count against the cap."""

    multilinear_rlt_max: int = field(
        default_factory=lambda: _env_int("DISCOPT_MULTILINEAR_RLT_MAX", 4)
    )
    """Max arity for multilinear RLT lifting (``DISCOPT_MULTILINEAR_RLT_MAX``)."""

    multilinear_separate: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_MULTILINEAR_SEPARATE", default=True)
    )
    """Separate multilinear McCormick cuts (``DISCOPT_MULTILINEAR_SEPARATE``)."""

    trilinear_nested: bool = field(
        default_factory=lambda: os.environ.get("DISCOPT_TRILINEAR") == "nested"
    )
    """Force the legacy nested-bilinear trilinear path
    (``DISCOPT_TRILINEAR=nested``; equivalent to the default unless another
    trilinear selector is explicitly set)."""

    trilinear_meyer: bool = field(
        default_factory=lambda: os.environ.get("DISCOPT_TRILINEAR") == "meyer"
    )
    """Use the Meyer-Floudas/Rikun trilinear convex-hull envelope
    (``DISCOPT_TRILINEAR=meyer``, default off)."""

    trilinear_exact: bool = field(
        default_factory=lambda: os.environ.get("DISCOPT_TRILINEAR") == "exact"
    )
    """Use the best-of-three nested trilinear envelope
    (``DISCOPT_TRILINEAR=exact``, default off)."""

    trilinear_rlt: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_TRILINEAR_RLT", default=True)
    )
    """Trilinear RLT rows (``DISCOPT_TRILINEAR_RLT``, default on)."""

    integer_multilinear_reform: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_INTEGER_MULTILINEAR_REFORM", default=False)
    )
    """Flow-aware exact linearization of *integer-multilinear* products — products
    of >=3 variable factors where every factor but at most one is integer- or
    binary-valued (declared or implied), e.g. ``(c + k*x_cont)*x_i*x_j*x_ind`` with
    ``x_i,x_j`` integer flow factors and ``x_ind`` a 0/1 indicator
    (``DISCOPT_INTEGER_MULTILINEAR_REFORM``, default **off**; issue #707).

    Each integer factor is binary-expanded (``x = lo + sum 2^k e_k``) and the
    resulting product of binaries is lifted to its **exact** hull — an n-ary AND
    (``z <= e_i``, ``z >= sum e_i - (n-1)``, ``z`` binary) for the pure-integer
    monomials, plus one big-M product (``v = e*x_cont``) for the single continuous
    factor. The rewrite is a value-preserving algebraic identity; only the
    *relaxation* changes (the loose term-wise trilinear McCormick envelope over the
    continuous box is replaced by the per-integer-level exact envelope), so it is
    sound and can only tighten the dual bound. Unlike the pure-bilinear integer
    reform (which is adopted only when it yields a pure MILP), this pass is retained
    on the spatial branch-and-bound path when residual *continuous* nonlinearity
    remains — the tightening of the integer-multilinear terms is a strict gain there
    (ex1252: lifts the SOS1-selector-branch dual bound off its 5134 floor).

    Bound-changing (CLAUDE.md §5): default-off behind this flag until a corpus-wide
    differential panel graduates it."""

    # --- McCormick separation toggles -----------------------------------------
    square_separate: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_SQUARE_SEPARATE", default=True)
    )
    """Separate tightened square (``x**2``) cuts (``DISCOPT_SQUARE_SEPARATE``)."""

    edge_concave: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_EDGE_CONCAVE", default=True)
    )
    """Edge-concave aggregation cuts (``DISCOPT_EDGE_CONCAVE``, default on)."""

    # --- cost-aware PSD moment-cut gate (THRU-2a; G1.3 graduated default-ON) ----
    psd_cost_gate: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_PSD_COST_GATE", default=True)
    )
    """Adaptive cost-aware gate on the per-node PSD (moment) cut separation loop
    (``DISCOPT_PSD_COST_GATE``, default **ON** since G1.3; ``DISCOPT_PSD_COST_GATE=0``
    is the escape hatch). PSD separation dominates the QCQP root wall (~60% on
    nvs17/19/24 per THRU-1) while the certified bound is set by McCormick+RLT and
    reached by branching when PSD is absent, so unbudgeted PSD *starves the tree
    search*. When on, this bounds the wall each node's PSD loop may spend to
    :attr:`psd_cost_gate_budget` × that node's own base LP-solve wall, and abandons
    the loop early once a round's relative LP-bound improvement falls below
    :attr:`psd_cost_gate_tau` (diminishing returns). It gates ONLY the PSD
    (moment-cut) loop — the univariate-square separator was measured to over-reach
    onto non-QCQP instances (tspn05 optimal→feasible), so it is left untouched.
    Keys purely on observed per-node cost/bound-delta — never on instance
    name/shape (§0.2). SOUND by construction: dropping valid cuts can only loosen
    the relaxation, never cut a feasible point or cross the optimum.

    Graduated to default-ON (G1.3-redo, post-C-38) on gate evidence: the isolated
    held-out arm (N=40, seed 0, tl 25 s) verdicts eligible — 0 soundness
    violations, cert-neutral (bound-changing regime), regression 0 % — plus the
    bound-changing verification (differential dual bound a valid underestimator
    that never crosses =opt= on nvs17/ex5_3_3; feasible-point sample recovers the
    identical incumbent ON vs OFF). See
    ``docs/dev/flag-graduation-redo-2026-07-07.md``."""

    psd_cost_gate_budget: float = field(
        default_factory=lambda: _env_float("DISCOPT_PSD_COST_GATE_BUDGET", 1.0)
    )
    """PSD wall budget per node as a multiple of that node's base LP-solve wall
    (``DISCOPT_PSD_COST_GATE_BUDGET``, default 1.0). The PSD loop stops once its
    cumulative wall this node exceeds ``budget × base_solve_wall``. Only consulted
    when :attr:`psd_cost_gate` is on."""

    psd_cost_gate_tau: float = field(
        default_factory=lambda: _env_float("DISCOPT_PSD_COST_GATE_TAU", 1e-4)
    )
    """Relative diminishing-returns threshold for the PSD loop
    (``DISCOPT_PSD_COST_GATE_TAU``, default 1e-4). A round whose LP-bound
    improvement ``Δ ≤ tau × (1 + |lb_before|)`` abandons the remaining PSD rounds
    at that node. Only consulted when :attr:`psd_cost_gate` is on."""

    # NOTE (#581): the cost-aware univariate-square gate (``DISCOPT_SQUARE_COST_GATE``)
    # was DEPRECATED and removed. It was a default-OFF, bound-changing flag that
    # graduated-gated net-negative (PR #685: benefit 17% / regression 22% on the
    # held-out N=20 arm) — sound but not helpful, the DISCOPT_CUT_INHERIT outcome —
    # so per this issue's protocol it is removed rather than left in default-OFF
    # limbo. The per-node square-separation loop now runs unconditionally (its
    # pre-THRU-3 behaviour), which is byte-identical to the shipped default.

    # --- root-cut-pool inheritance (THRU-4, structure-gated default) -----------
    cut_inherit: Optional[bool] = field(
        default_factory=lambda: _env_cut_inherit("DISCOPT_CUT_INHERIT")
    )
    """Root-cut-pool inheritance for the per-node square/PSD separation loops —
    **tri-state, opt-in** (``DISCOPT_CUT_INHERIT``: unset / ``0`` ⇒ force-off = the
    shipped default; ``gated``/``auto`` ⇒ structure-gated opt-in; ``1`` ⇒
    force-on. Programmatically ``cut_inherit=None`` selects the structure gate.)

    THRU-3 measured that the two per-node point separators — the univariate-square
    tangent loop and the PSD (moment) loop — are the dominant per-node cost on the
    cut-firing quadratic class (nvs24: 73% + 12% of the solve wall), each
    re-deriving cuts via up to 8 full MILP re-solves at EVERY node. When active,
    the root separates the full cut chain ONCE (unchanged root behaviour), the
    accepted rows are stored in the root cut pool, and every node *inherits* the
    pool instead of re-running the square/PSD separation loops.

    **Structure gate (CUT-INHERIT-GRAD).** The activating predicate is *whether a
    non-empty root cut pool is separated at the root* — a cheap, general,
    root-time signal that keys on measured structure, never on instance
    name/shape (CLAUDE.md §2). When the model carries the square/PSD-liftable
    structure the pool populates and inheritance engages (measured broadly
    beneficial: nvs17/19/23/24 1.6–5.4×, kall_circles 1.8–2.6×, knp3-12 ~4–9×,
    dispatch 3.3×; nvs19 gains its certificate); when it does not, the pool is
    empty, nothing is inherited or skipped, and the solve is **byte-identical to
    the force-off path** (node_count + objective unchanged).

    **Why still opt-in (default force-off).** The CUT-INHERIT-GRAD entry experiment
    falsified THRU-4-graduate's "the 2–5× is specific to the dense integer-QP
    class, broad flip is throughput-neutral" (that 1.004× was a TL=30s /
    parallel-contention artifact; under clean serial measurement every pool-firing
    instance benefits, so the honest gate is *pool-fires ⇒ ON*). BUT the same
    validation surfaced a **flag-path false-optimal on the pure-integer / MINLP
    cold-path class**: nvs22 certifies 33.55 against the oracle optimum 6.0582 —
    an nvs06-class incumbent-search reroute that C-42 (#553) only partially fixed
    (its pool-drop-retry does not trigger when the pool solve *succeeds* but the
    pre-tree pump is rerouted). Per CLAUDE.md §1 a false certificate blocks any
    default-ON flip, so the flag stays OPT-IN until that bug is fixed. See
    ``docs/dev/cut-inherit-grad-2026-07-08.md``.

    SOUND (where it fires cleanly): the inherited square tangents
    (``s ≥ 2·x0·x − x0²``) and PSD eigencuts
    (``vᵀMv ≥ 0``) are valid at every feasible lifted point independent of the
    node box, and every other captured family is valid over the ROOT box, hence
    over every descendant sub-box; skipping per-node re-separation only *loosens*
    the node relaxation (never cuts a feasible point). See
    ``docs/dev/thru4-cut-inheritance-2026-07-07.md`` for the per-family validity
    classification and measurements."""

    # --- branch-and-bound / bound levers --------------------------------------
    # NOTE (#581): ``DISCOPT_ALPHABB_WITH_LP`` (force the alpha-BB bound alongside
    # the LP relaxation) and ``DISCOPT_LIFTED_FBBT`` (FBBT on lifted columns) were
    # DEPRECATED and removed. Both were default-OFF, bound-changing flags that
    # graduated-gated net-negative/flat (PR #685: alphabb_with_lp benefit 16% /
    # regression 16%, redundant when the LP relaxer supplies every node bound;
    # lifted_fbbt benefit 22% / regression 28%) — sound but not helpful. Removing
    # each default-OFF gated branch is byte-identical to the shipped default path.

    sparse_large_lp: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_SPARSE_LARGE_LP", default=False)
    )
    """Solve the per-node McCormick LP even when its lift exceeds the
    ``_MAX_RELAX_DENSE_CELLS`` dense-cell guard (``DISCOPT_SPARSE_LARGE_LP``, default
    off). The whole per-node path is now sparse — relaxation build (CSR), incremental
    patch (T11), simplex (CSC), exact-LP oracle (T8) — so the guard's "would force a
    multi-GB dense allocation" premise is obsolete: a huge lift (e.g. qap's 85756-row
    McCormick relaxation) solves in ~0.1 s at <1 GB. On this flag the guard becomes
    nonzero-based (``_MAX_INCREMENTAL_NNZ``) instead of dense-cell-based, so a large
    *sparse* lift earns its rigorous McCormick LP bound instead of being declined
    (no per-node relaxation at all when ``n_vars > 50``, where alpha-BB is
    ineligible). Sound: the LP bound is a valid lower bound and the B&B keeps the
    parent bound as a floor, so enabling it never loosens a node — only adds a bound.
    Default off pending a benchmark instance that measurably benefits (qap's
    indefinite-QP McCormick bound is ~0; see docs/dev/sparse-milp-plan.md T7/T12)."""

    root_lp_probe_tight: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_ROOT_LP_PROBE_TIGHT", default=True)
    )
    """Probe the spatial-path McCormick LP relaxer over the FBBT/OBBT-**tightened**
    root box rather than the raw declared model bounds when deciding whether to keep
    it for the whole search (``DISCOPT_ROOT_LP_PROBE_TIGHT``, **GRADUATED default-ON**
    #282 Workstream A; ``=0`` restores the legacy raw-box probe).

    The keep/discard "probe" solves the McCormick LP once at the root. Run over the
    *raw declared* bounds, a model whose continuous variables are declared ``[0, inf]``
    (the process-synthesis ``*hfsg`` family, issue #282; also ``casctanks``) makes that
    LP unbounded/None, so the rigorous relaxer is wrongly discarded (``_mc_mode =
    "none"``) and the whole spatial search falls back to a far looser
    alphaBB/interval/NLP root bound — even though the SAME relaxer yields a valid, much
    tighter bound on the tightened box the solver has already computed (measured:
    syn30hfsg root excess +955%→+571%, syn40hfsg +3041%→+2350%, syn15m02hfsg
    +124.7%→+118.1%). With the tightened box the relaxer is kept and every node gets
    the LP bound.

    SOUND by construction (CLAUDE.md §5): the probe only decides *whether* to keep the
    (rigorous outer-approximation) relaxer; every node still solves its own sub-box and
    the relaxer bound joins via ``max``, so it can only *tighten* a node bound — never
    cut a feasible point or cross the optimum. Bound-neutral (byte-identical) on every
    model whose ``_mc_mode`` the probe does not change (the convex nlp_bb half, and all
    bounded-box spatial models).

    GRADUATED default-ON on the CLAUDE.md §5 Regime-2 panel
    (``discopt_benchmarks/scripts/issue282_root_lp_probe_graduation_panel.py``; verdict
    JSON under ``discopt_benchmarks/results/issue282/``): flag ON vs OFF over the
    vendored 66-instance corpus + the 7-instance #282 panel. cert-clean — 0 flag-induced
    soundness regressions across 73 instances (no dual bound past ``=opt=``, no
    ``gap_certified True→False``, certified optima identical, incumbents differentially
    feasibility-verified: the flag introduces no infeasibility); net-positive — the
    affected spatial-McCormick set (``syn15m02hfsg``/``syn30hfsg``/``syn40hfsg`` +
    ``casctanks``, the out-of-family generality probe) tightens the root dual bound,
    everything else bound-neutral. A/B root values re-confirmed load-independent
    (``results/issue282/root_lp_probe_ab_reconfirm_*.json``)."""

    root_probe_seeds_fallback: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_ROOT_PROBE_SEEDS_FALLBACK", default=False)
    )
    """Let the root LP probe's already-proved bound count as "a bound in hand" for the
    #138 fallback's own checkpoint rules (``DISCOPT_ROOT_PROBE_SEEDS_FALLBACK``,
    default off; §5 bound-changing). Issue #930.

    ``_root_relaxation_lower_bound`` accumulates candidates in ``_have`` and consults
    it in two places that already exist: rule 1 of ``_fb_stop`` (never decline an
    optional tightening phase while NO valid bound is in hand) and the ``_sep_budget``
    clamp (once a bound has landed, the separated solve must fit in what is genuinely
    left of the grant). Both rules ask "do we already have a bound?" — and on the
    spatial path the answer is often *yes* before the fallback starts, because the
    root LP probe in ``solve_model`` proved one. ``_have`` just never learned it.

    This flag seeds ``_have`` with the probe bound when — and only when — it passes
    ``_admissible_probe_bound``'s exact box-equality gate, i.e. it is the identical
    quantity over the identical box. Measured on ``hda`` at a 10 s limit: the fallback
    re-ran ``solve_at_node`` over the same root box for 2.72 s and returned
    ``-64473.442402437024``, the same value to all 17 digits the probe had proved
    3 s earlier — pure duplicate work, and the whole of the fallback's 4.06 s against
    a 1.69 s grant.

    NOT bound-neutral, which is why it is flagged rather than unconditional. Seeding
    ``_have`` lets rule 2 decline the separated phase once the grant is spent, and the
    probe bound is not always as tight as that phase would prove: ``solve_at_node``
    relabels *unbounded*, *time_limit* and *numerical* outcomes as
    ``status="optimal"`` carrying a weak Neumaier-Shcherbina floor
    (``mccormick_lp.py`` lines 1624/1642/1654), so a starved probe can report a valid
    but much looser bound than a full separated solve would. ``status`` therefore
    cannot be used as evidence of convergence, and the trade — punctuality for
    possible bound quality — is exactly the one rule 2 already makes; this flag only
    widens what rule 2 counts as "in hand". Never unsound: every seeded value is a
    valid lower bound over the root box, and it reaches the caller through the same
    ``max``.

    Independent of #930's other half (re-admitting the probe bound as a ``max``
    candidate), which is unconditional because it can only *tighten* the reported
    bound and costs no wall time.

    **Panel verdict: stays OFF.** Sec.5 bar 1 PASSES, bar 2 does not clear the
    ``DISCOPT_CUT_INHERIT`` precedent (sound but not broadly helpful). Three-arm
    differential over the 17 non-closing in-repo instances at ``time_limit=8``,
    2 reps, arms interleaved per instance and marker-asserted on both trees
    (``discopt_benchmarks/scripts/issue930_root_probe_bound_panel.py``, raw
    results in ``discopt_benchmarks/results/issue930/``):

    * bar 1 cert-clean — over 22 flag ON/OFF bound pairs: 0 lost, 0 looser,
      0 tighter, 0 ``gap_certified True->False``, 0 bound past an ``=opt=``
      oracle or across its own incumbent (6 invariant + 11 oracle checks). The
      flag did not move a single dual bound. Five closing instances were
      node-identical and objective-identical across all three arms, confirming
      both halves are inert where neither code path is reached.
    * bar 2 net-positive — NOT met, and not merely unproven: paired wall
      ``on - off`` is **+0.048 s per run** (sd 0.369, n=34, total +1.62 s). The
      flag is fractionally *worse*, not better. The duplicate solve it removes
      is real (2.72 s on ``hda`` at a 10 s limit, above), but on this corpus the
      saving does not survive into wall time.

    RETRACTION (CLAUDE.md Sec.11). An earlier run of this same panel reported
    ``on - off`` = -0.230 s/run (sd 0.734, total -7.82 s), of which 2.89 s was
    attributed to ``contvar`` and 0.77 s to ``tls2``. That measurement is
    withdrawn on two counts. It was taken against a baseline on the #75
    JAX-removal branch rather than ``main``, so it did not measure the change
    being shipped; and the ``contvar`` component was an artifact — that
    instance's *baseline* is bimodal (11.23 s vs 8.53 s across two reps of the
    main-based panel), so the "saving" was baseline spread read as a flag
    effect. The verdict it supported (stays OFF) is unchanged and now rests on
    a stronger footing: the flag shows no benefit at all, rather than a
    benefit concentrated in 2 of 17 instances.

    What would settle it: a load-gated panel at several time limits over a corpus
    where the fallback duplicates the probe more often than it does here. Until
    then the duplicate solve is a known, measured, opt-in cost rather than a
    silent default change."""

    rlt1_root_bound: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_RLT1_ROOT_BOUND", default=False)
    )
    """Add an RLT level-1 lower bound at the root for constrained **binary** QPs
    (``DISCOPT_RLT1_ROOT_BOUND``, default off; §5 bound-changing).

    Term-wise McCormick envelopes on an indefinite ``x'Qx`` are trivially loose —
    every ``X_ij`` drops to its independent lower face, so the root LP bound is ~0
    and fathoms nothing (the qap phenomenon, issue #661). RLT-1 multiplies each
    linear **equality** ``a·x = beta`` of the model by each variable ``x_p`` to add
    the valid identities ``sum_k a_k X_{p,k} = beta x_p`` (binary diagonal
    ``X_pp = x_p``). These constraint-factor products couple the lifted variables
    across a whole constraint and tighten a constrained binary QP toward its
    Shor/SDP bound. Purely LP (no SDP solver); solved with the exact vertex simplex,
    and the surfaced value is the **Neumaier-Shcherbina safe dual bound** from that
    solve — rigorous at *any* conditioning (``<=`` the true LP min for any ``y>=0``
    by weak duality), not the raw vertex objective, which on the wide-coefficient
    RLT LP can drift above the true minimum (issue #145). It joins
    ``_root_relaxation_lower_bound``'s candidates via ``max`` — it can only *raise*
    the bound, never loosen it.

    Sound by construction: each added row is a product of valid model constraints,
    so the RLT LP minimum is a valid lower bound (``<=`` the true optimum) and never
    cuts a feasible point. Measured: qap root 0 -> ~352891 (vs true optimum 388214;
    HiGHS-ipm gauge) and small synthetic Koopmans-Beckmann QAPs 0 -> optimum via the
    exact oracle. **Default off** because the *rigorous* solve is affordable only up
    to small/medium ``n``: the exact vertex simplex is fast there (n<=6 QAP in <3 s)
    but explodes on qap's highly degenerate all-pairs RLT-1 LP (114k rows), and the
    POUNCE IPM — the only in-house alternative now that HiGHS is removed — does not
    converge on these LPs (measured: ~25 iters in 90 s on a 2778x666 RLT LP). It
    graduates once a fast *sparse* rigorous LP oracle exists at that scale (see
    docs/dev/sparse-milp-plan.md §RLT1)."""

    rlt1_max_pairs: int = field(default_factory=lambda: _env_int("DISCOPT_RLT1_MAX_PAIRS", 60_000))
    """Size guard for :attr:`rlt1_root_bound`: skip (sound no-op) when the all-pairs
    lift ``n(n-1)/2`` exceeds this (``DISCOPT_RLT1_MAX_PAIRS``, default 60000 —
    admits qap's 25200 pairs, blocks a runaway build on a much larger model)."""

    rlt1_lagrangian: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_RLT1_LAGRANGIAN", default=False)
    )
    """Compute the RLT-1 root bound by the **Lagrangian dual** of the coupling rows
    instead of the monolithic LP (``DISCOPT_RLT1_LAGRANGIAN``, default off; §5).

    Same rigorous RLT-1 bound as :attr:`rlt1_root_bound`, but reached without ever
    forming the degenerate all-pairs RLT-1 LP: the RLT product identities ``C z = 0``
    are dualized and ``g(mu) = min_{z in P_McC}(c + C^T mu)^T z`` is maximized by
    adaptive-target-level subgradient ascent, each step a cheap sparse McCormick
    solve made rigorous by the Neumaier-Shcherbina safe bound. ``g(mu) <= RLT-1 opt
    <= true opt`` for *every* ``mu`` (weak duality), so each iterate is a valid lower
    bound; it joins ``_root_relaxation_lower_bound`` via ``max``. This is the route
    that beats the exact simplex's ~10-20x-per-n wall at qap scale (the inner
    McCormick LP stays ~0.1 s while the monolithic solve is >25 min). Measured on
    synthetic QAPs: reaches 100 % of the monolithic RLT-1 bound, target-free, sound.
    **Default off** pending the qap-scale entry experiment on the real instance with
    the sparse inner oracle (see docs/dev/rlt-lagrangian-plan.md §3)."""

    rlt1_lagrangian_max_iter: int = field(
        default_factory=lambda: _env_int("DISCOPT_RLT1_LAGRANGIAN_MAX_ITER", 300)
    )
    """Subgradient iteration budget for :attr:`rlt1_lagrangian`
    (``DISCOPT_RLT1_LAGRANGIAN_MAX_ITER``, default 300). More iterations tighten the
    bound toward the RLT-1 optimum; each iterate is already a valid lower bound, so
    an early stop is sound (just looser)."""

    shor_sdp_root_bound: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_SHOR_SDP_ROOT_BOUND", default=False)
    )
    """Add a **strong-Shor SDP** lower bound at the root for constrained all-binary
    QPs (``DISCOPT_SHOR_SDP_ROOT_BOUND``, default off; root-only, §5 bound-changing).

    The global moment-matrix PSD constraint ``M = [[1, x'],[x, X]] >= 0`` that no
    local (<=6-var) moment cut can enforce, *plus* the lifted-equality RLT rows,
    the McCormick box on ``X``, and the gangster rows (the *plain* Shor SDP is
    falsified on this class — unbounded on qap; see
    ``docs/dev/issue-661-qap-sdp-entry-experiment-2026-07-17.md``). Solved with the
    first-order conic solver SCS (optional dependency ``discopt[sdp]``; missing
    solver -> sound no-op), and the surfaced value is the **safe dual bound**
    recomputed from the returned multipliers (``shor_sdp_safe_dual_bound`` — the
    SDP analogue of the Neumaier-Shcherbina safe LP bound: valid for *any*
    multipliers by weak duality plus an eigenvalue-shift on the dual slack matrix,
    so solver convergence affects only tightness, never soundness), never the
    solver's approximate objective. Joins ``_root_relaxation_lower_bound``'s
    candidates via ``max`` — it can only raise the bound. Measured (entry
    experiment): qap root ~0 (McCormick) -> 377098 = 97.1 % of the optimum 388214
    (RLT-1 LP gauge 352891, published dual 149106) at ~86 s; exact on brute-forced
    synthetic Koopmans-Beckmann QAPs. **Default off**: an ~86 s root cost is a
    deliberate opt-in, and graduation needs the §5 corpus-wide differential panel."""

    shor_sdp_max_dim: int = field(default_factory=lambda: _env_int("DISCOPT_SHOR_SDP_MAX_DIM", 400))
    """Size guard for :attr:`shor_sdp_root_bound`: skip (sound no-op) when the
    moment-matrix dimension ``n + 1`` exceeds this (``DISCOPT_SHOR_SDP_MAX_DIM``,
    default 400 — admits qap's 226, blocks a runaway SDP on a much larger model)."""

    shor_sdp_time_limit: float = field(
        default_factory=lambda: _env_float("DISCOPT_SHOR_SDP_TIME_LIMIT", 120.0)
    )
    """Wall-clock budget in seconds for the SCS solve behind
    :attr:`shor_sdp_root_bound` (``DISCOPT_SHOR_SDP_TIME_LIMIT``, default 120 —
    covers qap's ~86 s root solve). An early stop is sound: the safe dual bound is
    valid at any iterate, just looser."""

    node_bound_mode: str = field(
        default_factory=lambda: os.environ.get("DISCOPT_NODE_BOUND_MODE", "lp")
    )
    """Per-node dual bound: ``"lp"`` (default, lifted-McCormick LP) or ``"milp"``
    (legacy nested integer MILP node solve) — ``DISCOPT_NODE_BOUND_MODE``."""

    relax_space: str = field(
        default_factory=lambda: os.environ.get("DISCOPT_RELAX_SPACE", "lifted")
    )
    """Per-node relaxation *space* for the McCormick dual bound
    (``DISCOPT_RELAX_SPACE``, MAiNGO-parity plan §2 P2.3). Values:

    - ``"lifted"`` (**default**, byte-identical to pre-P2.3): today's lifted
      McCormick LP with auxiliary columns (``MccormickLPRelaxer.solve_at_node``).
    - ``"auto"``: currently an alias for ``"lifted"`` — no structural policy has
      graduated yet (P2.4). Preserves today's behavior exactly.
    - ``"reduced"``: MAiNGO-style **reduced-space** McCormick — a Kelley
      cutting-plane bound over the *original* variables only (no lifted columns),
      computed by ``reduced_mccormick_lp_bound``. The evaluator is built once per
      solve; if the model is outside the sound MCBox scope
      (``UnsupportedRelaxation`` at build time) the whole solve falls back to the
      lifted path (logged once, never an error). Per-node, an ``"unsupported"`` /
      ``"unbounded"`` status yields no reduced bound for that node (lifted-only);
      ``"infeasible"`` fathoms the node; ``"optimal"`` is a **valid** node dual
      lower bound and is combined soundly (max with any lifted bound).
    - ``"hybrid"``: reserved for P2.5 (Najman-style MC↔AVM per-term lift); raises
      ``NotImplementedError`` until then rather than silently degrading.

    CORRECTNESS-CRITICAL: the reduced-space bound certifies the node dual bound.
    A ``"reduced"`` bound is only ever used where its status is ``"optimal"``
    (valid LB) or ``"infeasible"`` (empty relaxed set → fathom); it can only
    *raise* a node bound up to (never above) the true box optimum, never cut a
    feasible point."""

    node_nlp_stride: int = field(default_factory=lambda: _env_int("DISCOPT_NODE_NLP_STRIDE", 4))
    """Solve the node NLP every k-th node (``DISCOPT_NODE_NLP_STRIDE``, default 4)."""

    phase2_dbbt: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_PHASE2_DBBT", default=False)
    )
    """Per-node cheap reduced-cost DBBT + cutoff-FBBT (Phase 2, issue #764).

    ``DISCOPT_PHASE2_DBBT`` (**default OFF**, bound-changing / Regime-2). After each
    spatial node LP solve, run ``reduce_node`` — free duality-based bound tightening
    from the node LP's reduced costs (``d_j > 0 ⇒ x_j ≤ l_j + gap/d_j``,
    ``gap = cutoff − safe_bound ≥ 0``, integer endpoints rounded inward) plus
    cutoff-FBBT — and feed the tightened box to the children via ``set_node_bounds``.
    This is BARON's signature move: cheap reduction from information the node LP
    already produced (no extra LP solves), the intended replacement for the
    exhaustive ~2n-LP per-node OBBT on the loose-root / big-tree class.

    Distinct from the retired ``DISCOPT_NODE_REDUCE`` (removed #581): that flag ran
    the same reduction but the reduced costs it needed were produced ONLY on the
    incremental/warm LP path, so on the non-composite-lift class (``tanksize`` etc.,
    which takes the cold path) it silently no-op'd — the #685 net-negative verdict.
    #764 step 1 added cold-path marginals, so DBBT now actually fires there; this
    flag re-introduces the sound consumer on top of that prerequisite. Ships OFF
    until the graduation panel (cert-clean + net-positive), and until the step-3
    OBBT-coupling entry experiment shows total-wall-to-close improves without
    regressing the #685 set."""

    adaptive_nlp: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_ADAPTIVE_NLP", default=True)
    )
    """Adaptive back-off for the *strided in-tree node NLP*
    (``DISCOPT_ADAPTIVE_NLP``, **default ON** since G2 — flag-graduation
    convention: ``=0`` restores today's fixed ``node_nlp_stride``).

    TX1 (``docs/dev/tenx-plan.md`` §3). The strided node-NLP is a **pure primal
    heuristic** — it fires ONLY where the McCormick LP relaxer supplies the node
    dual bound and the model is nonconvex (``_gate_node_nlp`` in ``solve_model``);
    there its objective is never a bound (the LP is), so throttling it can only
    change *incumbent arrival*, never the certificate. TX0 measured this bucket as
    idle waste on integer-heavy nonconvex models (nvs09: 14.3 s skippable, identical
    proof/bound). Fixed stride 4 keeps re-solving it long after the incumbent has
    stopped improving.

    When ON, the *effective* stride starts at ``node_nlp_stride`` and doubles
    (capped) after each batch whose strided node-NLP fired but did **not** improve
    the incumbent, resetting to the base stride the moment it does. Convex nodes and
    the no-LP-relaxer path (where the NLP objective IS the bound) are never touched
    — the gate that admits this back-off is exactly the existing heuristic-only
    envelope. Sound (heuristic-policy regime, CLAUDE.md §5): every injected point is
    still sub-NLP/constraint-verified and ``inject_incumbent`` enforces strict
    improvement, so the dual bound and gap certification are byte-identical to the
    fixed-stride run; only *which* nodes get a primal probe changes."""

    continuous_multistart: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_CONTINUOUS_MULTISTART", default=True)
    )
    """Stratified continuous multistart at the root for pure-continuous
    nonconvex models (``DISCOPT_CONTINUOUS_MULTISTART``, default ON; issue #188).

    The primal-heuristic suite is integer-centric: on a model with no integer
    variables, pump/ILS/diving/RINS/RENS all no-op, the root multistart NLP is
    skipped on the McCormick-LP spatial path, and the strided node NLP
    warm-starts from the parent point — zero basin diversification end to end.
    Measured on the kall_congruentcircles_c51 class (#188): the default path
    parks at the 1.5371 two-row local packing forever, while 32 stratified
    starts (~2.8 s, ~90 ms/solve) reach the 1.0730 global basin on every seed
    tried; the 4 deterministic anchors and the LP-vertex-seeded solves never do.

    When ON, ``solve_model`` runs ``primal_heuristics.continuous_multistart``
    once at the end of the root iteration on the spatial McCormick-LP path for
    nonconvex models with no integer variables: ``min(64, max(32, 2·n))``
    stratified starts, deadline-gated between starts and per-solve capped, seed
    fixed for determinism. Sound (heuristic-policy regime, CLAUDE.md §5): a
    primal finder only — every point is constraint-re-verified and
    ``inject_incumbent`` enforces strict improvement; the dual bound and
    certificate math are untouched. Set ``DISCOPT_CONTINUOUS_MULTISTART=0`` to
    restore the prior behavior."""

    ils_solve_cap: int = field(default_factory=lambda: _env_int("DISCOPT_ILS_SOLVE_CAP", 2))
    """Sub-NLP solve cap for the ``integer_local_search`` objective-descent
    (``DISCOPT_ILS_SOLVE_CAP``, **default 2 = ON** since ILS-DEFAULT, #530-followup).

    ``integer_local_search._objective_improve`` runs a first-improvement coordinate
    descent over ``int_idx × {±1,±2}``, each move a full continuous-repair sub-NLP,
    re-sweeping until its wall deadline. VOLUME-1 (#530) measured its incumbent hit
    rate at **0 %** on every ILS-firing panel instance (the incumbent is already
    found by the root multistart's first start) — the descent issues *hundreds* of
    no-op sub-NLPs. This caps a single descent to ``ils_solve_cap × max(1, n_int)``
    sub-NLP solves (a full first-improvement sweep or two — where any real gain
    lands), keyed on the integer dimension, never an instance name (§0.2).

    Default 2, broad-validated on a held-out integer MINLPLib sample
    (``docs/dev/ils-default-validation-2026-07-06.md``): 0 lost incumbents, 0
    soundness violations, geomean speedup on the ILS-firing subset. Set
    ``DISCOPT_ILS_SOLVE_CAP=0`` (or ``ils_solve_cap=0``) to restore the old
    UNCAPPED behavior — the debugging escape hatch, not a dead flag. Sound: capping
    this descent only ever *weakens* the incumbent it might find (every point is
    sub-NLP-verified and re-verified by ``inject_incumbent``); it never touches the
    dual bound or the certificate (heuristic-policy regime, CLAUDE.md §5)."""

    ils_eval_budget: int = field(
        default_factory=lambda: _env_int("DISCOPT_ILS_EVAL_BUDGET", _ILS_EVAL_BUDGET_DEFAULT)
    )
    """Evaluation cap for the root ``integer_local_search``
    (``DISCOPT_ILS_EVAL_BUDGET``, **default ON**; issue #912).

    The root integer local search used to bound its own extent with a wall clock
    (``time_budget=min(5.0, 0.15 * time_limit)``). Its descent routinely never
    converges, so *how far it gets* — and therefore the incumbent it hands the
    tree — was a function of machine speed: #912 measured ``gear2`` closing in 3
    nodes at a 5 s budget and 91 nodes at 3 s, with the default sitting exactly
    on that cliff, and reproduced the flip by scaling the process clock. A
    search whose result depends on how fast the box is cannot sit on the
    certificate path, and it silently invalidates every "node counts unchanged"
    verdict this repo relies on (CLAUDE.md §5, bound-neutral regime).

    The search now counts *operations* instead (:mod:`discopt._work_budget`) and
    stops at the same point on every machine. This field caps the cheap kind:
    constraint/objective evaluations, 0.7-2.7 us each as measured. The solve
    deadline is still passed down as a backstop so ``time_limit`` is honoured;
    it decides *when to stop*, never *how much work to do*.

    Set this **and** ``ils_solve_budget`` to 0 to restore the legacy wall-clock
    gate — the debugging escape hatch, not a dead flag. Sound either way: ILS is
    a pure incumbent finder (every point is sub-NLP-verified and re-verified by
    ``inject_incumbent``), so changing its extent can only change *which*
    feasible point it finds, never the dual bound or the certificate."""

    ils_solve_budget: int = field(
        default_factory=lambda: _env_int("DISCOPT_ILS_SOLVE_BUDGET", _ILS_SOLVE_BUDGET_DEFAULT)
    )
    """Sub-NLP-solve cap for the root ``integer_local_search``
    (``DISCOPT_ILS_SOLVE_BUDGET``, **default ON**; issue #912).

    The companion to :attr:`ils_eval_budget`, capping the expensive kind:
    continuous-repair NLP solves, 1.9-104 ms each as measured. Whichever cap is
    reached first ends the search.

    They are separate on purpose, and the separation is a measurement, not a
    preference. Converting both to one currency at their geomean cost ratio
    (12 364) was tried first and produced a real regression: ``nvs09``'s
    evaluation-dominated search was starved (5 -> 29 nodes) at a budget that
    already gave the solve-dominated ``syn05hfsg`` three times its legacy wall
    time. The ratio varies 27x across the corpus, so one number cannot price
    both. Full data in ``docs/dev/work-budget-calibration-2026-08-01.md``.

    Distinct from :attr:`ils_solve_cap`, which limits sub-NLPs *per objective
    descent* keyed on the integer dimension; this one bounds the whole call."""

    gdp_sumover: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_GDP_SUMOVER", default=True)
    )
    """Teach the GDP/relaxation expression walkers the indexed-summation node
    ``SumOverExpression`` (``dm.sum(f(i) for i in S)``)
    (``DISCOPT_GDP_SUMOVER``, default **ON** -- graduated by the §5 panel below;
    opt out with ``DISCOPT_GDP_SUMOVER=0``, which restores the pre-#1154 walkers
    byte for byte. #1154).

    ``SumOverExpression`` holds its already-expanded term list in ``.terms``. Six
    walkers in :mod:`discopt._relax.gdp_reformulate` had no case for it, so a
    disjunct body built with ``dm.sum`` was under-reported as having no variables
    (``_collect_variables``), non-linear (``_is_linear``), unbounded
    (``_bound_expression``) and un-evaluable at the origin (``_body_at_zero``).
    On ``main`` that costs three loud refusals: ``auto``/``big-m`` cannot bound
    the body, ``hull`` cannot form ``g(0)``.

    With the flag ON the six walkers treat ``Σ[t1, …, tn]`` **exactly** as the
    left-folded chain ``t1 + … + tn`` — the desugaring the modeling layer could
    equally have produced — so nothing about the reformulation's mathematics is
    new; only the node type is newly recognised. The invariant is machine-checked
    per walker in ``python/tests/test_1154_gdp_sumover_hull.py``.

    **Graduated default-ON** (§5, 2026-09-04) on a panel that meets both bars;
    full tables in ``docs/dev/issue-1154-gdp-sumover-panel-2026-09-04.md``.

    *Cert-clean.* The flag is **structurally inert** on the ``.nl`` corpus -- the
    node is created only by the Python modeling API's ``dm.sum`` and the ``.nl``
    reader never emits one, measured at 0 occurrences in 33 376 DAG nodes over all
    66 vendored instances -- and the A/B differential is byte-identical on 63/66,
    the other 3 reproducing their whole difference *within a single arm* (a
    role-1 ``time_limit`` truncation artifact, #1116). Scored against the oracle:
    52 instances, 0 bound violations, 0 primal violations. On the class where the
    mechanism fires (108 generated GDP models x 3 routes x 2 arms = 648 solves,
    every incumbent feasibility-verified in numpy against the original
    disjunction): **0 invalid bounds**.

    *Net-positive.* Three loud refusals on the issue's repro become three
    certified optima at the true ``-30.0``. ``auto`` and ``big-m`` are
    bit-identical between a ``Σ[...]`` body and the equivalent folded chain on all
    108 cases; on ``hull`` with a nonlinear body the ``Σ`` form certifies
    **46/54** against the chain's **29/54**, sd **0.00** over 3 interleaved reps
    on a quiet machine. 731 GDP/OA/Benders/GBD/MPEC tests give identical results
    in both arms.

    Note the ordering that PR #1150 got wrong: widening ``_is_linear`` **alone**
    made hull emit the disjunct body globally with its selector coefficient
    collapsed to zero (``all_vars`` was empty, so no disaggregated variables were
    created) and return a dual bound of −3.0 on a model whose true minimum is
    −30.0. The walkers must move together; the independent-walker cross-check in
    ``_reformulate_disjunction_hull`` now refuses loudly rather than emit that row."""

    disjunctive_config_bound: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_DISJUNCTIVE_CONFIG_BOUND", default=False)
    )
    """Root disjunctive configuration bound for the gated-configuration class
    (``DISCOPT_DISJUNCTIVE_CONFIG_BOUND``, default **OFF**; #732 Stage 2).

    When the integer-multilinear reform (#707) applies, enumerate the reform's
    configuration-indicator patterns at the root, bound each configuration box by
    FBBT -> OBBT -> LP with best-first unit-peeling on the configuration count
    variables, and floor every optimal node bound at the min over configuration
    leaves — a valid bound by partition (anytime-valid: unprocessed leaves
    inherit the caller's existing root bound). Measured (ex1252, plan doc Stage
    2): the standalone root pass certifies 37945 at a 48-leaf budget (tree at
    400 nodes: 16304) and 63080 at 120 leaves once the Stage-4 spatial bisection
    engages (through the ~48k plateau — the #721 acceptance bar); end-to-end
    the reported global dual goes 0.0 -> 42725 (240 s solve) and 0.0 -> 74915
    (600 s solve, deadline-governed leaf budget). Default-OFF pending the
    CLAUDE.md §5 corpus differential panel."""

    obj_branch_priority: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_OBJ_BRANCH_PRIORITY", default=True)
    )
    """Prioritize branching on objective-defining variables
    (``DISCOPT_OBJ_BRANCH_PRIORITY``, default ON).

    Graduated per T2.6 with 3 consecutive green held-out verdicts (composed
    with the density LU route): BR-3 #602 (verdict 1), FLAG-GRAD #612
    (verdict 2), and the P0 SPATIAL-CERT re-run
    (``docs/dev/p0-spatial-cert-2026-07-10.md``, verdict 3 — incorrect 0,
    oracle-cross 0, cert-loss 0; both-certified nodes 1092 -> 1054). Set
    ``DISCOPT_OBJ_BRANCH_PRIORITY=0`` to restore the old default."""

    sos1_selector_branch: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_SOS1_SELECTOR_BRANCH", default=False)
    )
    """Spatially branch continuous SOS1 selectors before drilling aux-binaries
    (``DISCOPT_SOS1_SELECTOR_BRANCH``, default OFF; issue #196).

    A continuous one-of-N selector ``s`` (member of a selection row ``Σ s_i = 1``,
    upper-coupled to a 0/1 indicator by ``s ≤ y``, and in a nonlinear product term)
    that stays spread across a multi-line box keeps the McCormick bound of the
    gated products pinned near 0. When on, :func:`_sos1_selector_vars` detects such
    selectors and the Rust tree branches one spatially (box-midpoint) with
    precedence, concentrating the selection so a single product is forced positive
    (ex1252: an ambiguous box's bound 12658 → ~67–83k once a selector is pinned).

    Branch-ORDER metadata only (never a bound/feasibility input), so it cannot
    change a bound's validity — the midpoint split is a sound cover and its
    width-halving keeps the search complete. Default OFF pending a corpus
    differential panel (CLAUDE.md §5, ``incorrect_count = 0`` + net-positive)
    before any graduation."""

    lp_warmstart: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_LP_WARMSTART", default=True)
    )
    """Warm-start the node LP from the parent basis (``DISCOPT_LP_WARMSTART``)."""

    lp_cold_dual_start: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_LP_COLD_DUAL_START", default=False)
    )
    """Start a COLD pure-LP solve from the sign-matched dual-feasible slack basis
    even with no deadline (``DISCOPT_LP_COLD_DUAL_START``).

    ``solve_lp_warm_std`` already builds that basis (``_dual_start_slack_basis``),
    but #928 engaged it only when the caller passed a finite ``time_limit``,
    because #928's *motivation* was an anytime bankable floor rather than speed.
    Since ``lp_warm_deadline`` is itself default-OFF, no deadline reaches the LP on
    the default path, so in practice **every** cold node LP takes the Rust cold
    PRIMAL loop — the loop whose own comment reads "a dense, degenerate
    lifted-McCormick LP can otherwise grind toward ``max_iter`` and run
    uninterruptibly for minutes" (``lp/simplex/primal.rs``).

    It does. Measured on the QPLIB relaxation LPs, one variable changed (the start
    basis) and **no deadline on either arm**, so both run to optimality and the LP
    optimum is a control (``scratchpad/qplib_run/coldstart_ab.py``):

    ==========  =====  ==================  =========================
    instance    RLT    dual-slack start    cold primal (pre-flag)
    ==========  =====  ==================  =========================
    QPLIB_1157  off    0.27 s              0.18 s
    QPLIB_1157  on     **6.17 s** optimal  **>150 s, iter-cap fail**
    QPLIB_1493  off    0.34 s              0.30 s
    QPLIB_1493  on     **2.91 s** optimal  **11.79 s, iter-cap fail**
    QPLIB_1507  off    0.19 s              0.38 s
    QPLIB_1507  on     0.56 s              0.88 s
    QPLIB_1143  off    0.88 s              1.63 s
    QPLIB_1143  on     5.17 s              14.59 s
    ==========  =====  ==================  =========================

    The dual start wins 6 of 8; both losses are under 0.1 s absolute. On 2 of 8 the
    cold primal does not merely run long, it **fails** — ``max_iter`` exhausted,
    ``None`` returned, which then cascades into the equilibrated retry and the
    ~170x-slower cold ``solve_milp``. Wherever both arms returned, the objectives
    agree to <=3e-12 (and to HiGHS likewise), so this is bound-neutral on those.

    Why the degeneracy: an equality row reaches the LP layer as two opposing
    ``<=`` rows, both tight at every feasible point. Relaxations rich in linear
    equalities (constraint-factor RLT above all — see ``rlt_lineq``) are therefore
    massively primal-degenerate, and the cold primal stalls where a dual simplex
    started dual-feasible does not. That is a property of the row structure, not of
    any instance.

    Default **off** pending the graduation panel. It is not provably bound-neutral:
    a different optimal basis can be returned on a degenerate LP, which can move
    downstream branching, and the iter-cap cases change status ``None`` -> optimal
    (a strict gain, but a change). Ineligible LPs — an open bound on the side the
    objective sign selects — keep the primal path exactly as before.
    """

    # --- determinism (#1116) --------------------------------------------------
    deterministic: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_DETERMINISTIC", default=False)
    )
    """Make the search a function of the *model* rather than of machine speed by
    rendering every **role-2** wall budget inert. ``DISCOPT_DETERMINISTIC``,
    default **OFF**; also reachable as ``Model.solve(deterministic=True)``.

    #912 draws the line this flag acts on. A clock that answers *"when do we
    stop?"* — the user's ``time_limit`` — is correct by definition (role 1). A
    clock that answers *"how much work do we do?"* — a sub-budget carved as a
    fraction of ``time_limit``, or a fixed number of seconds handed to a stage —
    makes the ANSWER a function of how fast the machine happens to be that
    minute. ``_work_budget.py`` calls that "a correctness-of-process bug, not a
    performance detail".

    #1116 measured the consequence. ``kriging_peaks-full200`` at ``max_nodes=1``,
    same process, same binary, no user time pressure, returned root dual bounds of
    −25371.8 / −28852.0 / −28072.6 — a 14 % swing — with the incumbent
    bit-identical. Neutralizing the wall budgets made the same solve reproduce
    exactly (``-1044.819…`` twice at ``max_nodes=1``, ``-754.478794470719`` twice
    at ``max_nodes=300``), and *tighter* than every wall-bounded run.

    When this is on, role-2 budgets become ``None``/``math.inf`` at the 27 sites
    that carve them (``solver._role2_budget`` / ``_role2_deadline`` /
    ``_role2_horizon``, plus the integer-ratio dive in ``_relax/mccormick_lp.py``)
    and each stage is bounded by the deterministic caps it already carries (round
    counts, iteration caps, the node budget). Expect a run to take longer:
    nothing truncates a stage early any more.

    A 28th site joined them with #1187, in a shape the first three helpers could
    not express: a slice handed to a *nested* :func:`~discopt.solver.solve_model`
    as its ``time_limit``, where neither ``None`` nor ``math.inf`` is a legal
    value. ``solver._role2_slice`` returns the caller's own ``time_limit`` there —
    elapsed-independent, and still finite, so the nested solve keeps a role-1
    bound. The gate was the NLP-BB root RENS budget, and it was measured returning
    three incumbents 25 % apart (55092.52 / 46785.55 / 41573.26) on
    ``clay0303hfsg`` at an *identical* 27 nodes with the dual bound agreeing to 12
    significant figures — this flag on, in one process, on one binary. As in
    #1116, routing it made the instance reproduce **and** find a better incumbent
    (26669.11) than any wall-truncated repetition.

    **What the flag does not cover, and why.** Two role-1 mechanisms are left
    alone on purpose, because neutralizing them would let preprocessing overrun
    the user's ``time_limit`` without bound — trading a reproducibility bug for a
    broken role-1 promise, which CLAUDE.md §1 does not permit:

    * the phase-entry gates (``_deadline_exhausted()`` / ``_remaining_budget() >
      x``), which decide whether an optional preprocessing phase *starts* at all;
    * the two POUNCE funnels' ``max_wall_time = min(30.0, caller_limit)`` stall
      backstop.

    So the guarantee is: **a solve reproduces when the role-1 budget never binds**
    — i.e. it terminates on work (``max_nodes``/gap) with real slack against
    ``time_limit``. A run cut short by ``time_limit``, or run on a machine slow
    enough that a phase-entry gate flips, is not reproducible and does not claim
    to be.

    That is not a footnote for a panel author, it is the panel's admission rule.
    **This flag cannot equalise work on a run that terminates on the wall clock,
    because the terminating condition IS the wall clock.** Measured on ``beuster``
    at ``time_limit=120`` with this flag on: two builds differing only in Python
    marshaling cost issued 3858 OBBT probe LPs against 942 — 4.1x the work — for
    the same 3 nodes and the same bound, both ending ``status=time_limit``. A gate
    that compares such rows is measuring the budget; #1180's sweep did, on 13 of
    66 rows, and manufactured a reproducible "0.516x regression" that re-measured
    as a 5x-more-nodes, 30 %-tighter-bound improvement. The benchmark harness now
    refuses those rows and reports them as unmeasured rather than passing them
    (``discopt_benchmarks/utils/cert_neutrality.wall_limited_rows``).

    The residual was measured not to bind on the reproduction instance: the 30 s
    POUNCE cap was live and real during the arm that reproduced bit-exactly, and
    the solve finished in ~7 min against the default 3600 s limit.
    """

    # --- branch-and-reduce (cert:T2.3 / T2.4) ---------------------------------
    root_fixpoint: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_ROOT_FIXPOINT", default=True)
    )
    """Run the cutoff-aware root branch-and-reduce fixpoint (cert:T2.3) at the end
    of iteration 0: iterate {FBBT-with-cutoff, OBBT/DBBT-with-cutoff} to a fixpoint
    on the root box, refreshing the root cut pool + incremental engine base from the
    tightened box. ``DISCOPT_ROOT_FIXPOINT``, default **ON** (GRADUATED per #581
    under the one-successful-graduation-gate-run policy — CLAUDE.md §5). The
    graduation gate (`graduation_gate.py --flags root_fixpoint`, held-out N=20
    seed 0 + 41-instance cert panel) returned cert-clean (incorrect_count 0, 0
    cert violations, objective/optimal-status enforced) and net-positive
    (benefit 29% — crudeoil_pooling_ct1, ex5_3_3, powerflow0014r, qap, eg_all_s —
    with 0% regression). Set ``DISCOPT_ROOT_FIXPOINT=0`` to restore the old
    default OFF."""

    anytime_root_build: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_ANYTIME_ROOT_BUILD", default=False)
    )
    """Make the root-relaxation *fallback* build anytime/incremental so its dual
    bound accrues and the build can honor the grant (``DISCOPT_ANYTIME_ROOT_BUILD``,
    default **off**; §5 bound-changing; issue #694).

    #654 left a measured floor: on a class of large sparse network-design/QAP/graph
    -partition MINLPs (sonet\\*, qap, eg_all_s, super3t) the fallback's dual bound is
    produced by a single **uninterruptible** McCormick-LP *build* (sonet23v4: 16.8 s,
    not bounded by the solve's ``time_limit``), so ``solve(time_limit=2)`` still took
    24.5 s — truncate the build and you lose the bound (baron-gap-plan.md §8.1),
    don't and you blow the budget. This flag dissolves that fork: when on,
    ``_root_relaxation_lower_bound`` passes a ``build_deadline`` (its own grant) to
    the base ``build_milp_relaxation`` and the separated ``solve_at_node`` build, so
    the constraint-row loop **stops adding rows once the grant is spent** and the
    partial relaxation is solved for a valid (weaker) bound.

    Sound by construction: a relaxation with FEWER constraint rows is still a valid
    outer approximation, so its LP minimum is a valid lower bound — dropping rows can
    only *weaken*, never falsify (the "weaken but never falsify" property of
    baron-gap-plan.md §8, and the #694 entry experiment: a finite bound exists by
    8–45 % of build on every tested structure, because the objective is fully
    linearized before the constraint loop). It does NOT re-add the Rust LP native
    deadline (§8.2, TX2b): this truncates the Python relaxation *build*, never the LP
    *solve*. **Default off** pending the §5 corpus-wide differential panel (flag ON
    vs OFF; ``incorrect_count = 0``, no bound above its reference optimum, no
    certification regression, incumbents feasibility-verified) AND net-positive on
    the #654 class — with the must-not-regress bounds casctanks 5.698, super3t −1.0,
    sonet23v4 −53974.375 kept sound. Entry evidence:
    ``docs/dev/issue694-anytime-build-entry-2026-07-17.md``. NOTE: with the flag on
    the fallback bound becomes timing-dependent (an anytime algorithm), so it is not
    bit-reproducible run-to-run; the default (off) path is unaffected and stays
    deterministic."""

    root_build_deadline: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_ROOT_BUILD_DEADLINE", default=True)
    )
    """Deadline the **base** root-relaxation ``build_milp_relaxation``
    (``DISCOPT_ROOT_BUILD_DEADLINE``, default **ON** — GRADUATED per §5, set ``=0`` to
    opt out; §5 bound-changing; issues #832/#814).

    The #694 ``anytime_root_build`` flag truncates the *separated* build but its
    companion base build was deliberately left WHOLE — so on large ill-conditioned
    instances the base build alone overruns the grant several-fold. Measured
    (``gastrans582_mild11``, budget 3.0s): the base DCP build takes ~10.5 s and the
    fallback returns ``None`` after ~15.5 s (5.2x overrun) — 93 % of the time is the
    Python DCP relaxation build, **not** the Rust LP (2.77 ms factorize); see #832.

    When on, ``_root_relaxation_lower_bound`` passes a ``build_deadline`` (its own
    grant) to the base build, so its constraint-row loop stops adding rows once the
    grant is spent. Sound: the objective is fully linearized before the constraint
    loop, so a prefix of rows is a valid **weaker** outer relaxation (dropping rows
    only enlarges the feasible set -> the LP min stays a valid lower bound); if a
    dropped constraint un-bounds an objective cost column the existing
    ``_objective_bound_valid`` gate returns ``None`` (weaker, never falsified). On
    ``gastrans582`` the truncated base build is still ``obj_bound_valid=True`` and
    yields a valid weaker bound in ~budget instead of ``None`` after 5x the grant.

    Bound-**changing** (a truncated base build can weaken a bound or drop it to
    ``None``). GRADUATED default-ON per §5 (one passing graduation-gate run suffices):
    the ``graduation_gate.py --flags root_build_deadline`` panel (held-out N=40 seed 0
    + cert panel) returned **eligible** — soundness ok (0 violations, no bound above
    its reference optimum), cert-neutral (certified objective + optimal-status
    enforced), net-positive (benefit 23% / regression 8.6%, and 2 of the 3
    "regressions" are node-count-only labels on large wall wins: sonet24v5 33.6→14.5s,
    sonet25v6 38.2→19.5s). Set ``DISCOPT_ROOT_BUILD_DEADLINE=0`` to restore the legacy
    whole-base-build path. Like #694, with the deadline active the fallback bound
    becomes timing-dependent (an anytime algorithm), so it is not bit-reproducible
    run-to-run; the ``=0`` opt-out path stays deterministic."""

    root_setup_build_deadline: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_ROOT_SETUP_BUILD_DEADLINE", default=True)
    )
    """Bound the **pre-B&B root-setup** relaxation builds by the solve's own
    deadline, minus the root-fallback reserve
    (``DISCOPT_ROOT_SETUP_BUILD_DEADLINE``, default **ON** — GRADUATED per §5, set
    ``=0`` to opt out; §5 bound-changing; issue #1152).

    #832 gave the root-relaxation *fallback*'s builds a deadline; the root-setup
    phases that run BEFORE it — root OBBT's per-round envelope rebuild, the root LP
    probe, the root cut pool — still build unbudgeted. Each of those phases polls a
    deadline between its LP solves but not inside the build, so a build that starts
    just before the deadline runs to completion past it. Measured on ``casctanks`` at
    ``time_limit=5`` (in-repo corpus): root OBBT starts its round build at t=4.39 s
    with 0.61 s left and spends 1.85 s in it, so the solve returns at 6.4 s (1.29x)
    and — the second half of the defect — the #654 short-circuit then finds
    ``_remaining_budget() == 0`` and skips the root-relaxation fallback entirely, so
    the solve reports ``bound=None`` where a slice of a second would have proved one.

    That is issue #1152's contradiction in miniature: "honour the deadline" and "keep
    the dual bound" read as opposing contracts only because the long root operations
    are all-or-nothing. When on, ``solve_model`` threads one absolute root-setup
    deadline (``time_limit`` minus the ``_ROOT_FALLBACK_RESERVE_S`` slice the
    fallback needs) into those builds as the ``build_deadline`` the #694/#832 anytime
    mechanism already implements: the constraint-row loop stops at the deadline and
    the partial relaxation is still a valid outer approximation. It also declines to
    *start* the native-kernel spec build (two whole relaxation builds whose row sets
    must correspond, so it is not truncatable) once that deadline is spent.

    Sound by construction, in each direction the truncation reaches:

    * **OBBT** tightens ``x_i`` to the optimum of ``min x_i`` over the relaxation
      polytope. Dropping constraint rows only ENLARGES that polytope, so the LP
      optimum moves outward and the tightening is weaker — never invalid. The #208
      cascade's carried aux-column bounds are keyed by column index and a truncated
      build has a different lifted layout, so they are neither applied to nor
      captured from a truncated build.
    * **The root LP probe** decides whether to keep the relaxer and banks a bound
      that ``_root_relaxation_lower_bound`` re-gates on exact box equality; a
      truncated build can only make that bound weaker (fewer rows), and #928's
      cut-short objective floor keeps it finite.
    * **The root cut pool** only adds cuts; a truncated build separates fewer of
      them, which loosens per-node bounds and never invalidates one.

    Bound-**changing** (a truncated setup build can weaken the root box, the probe
    bound or the pool), hence flag-gated. Off is byte-identical: no build deadline is
    computed, no entry gate is consulted, and ``_setup_remaining_budget`` returns the
    plain remainder.

    GRADUATED default-ON per §5 (one passing graduation-gate run suffices) on the
    in-repo corpus differential — 66 instances x ``time_limit`` {5 s, 20 s}, both arms
    interleaved per instance, 132 pairs / 559 comparisons
    (``scratchpad/i1152/panel.py`` + ``panel_report.py``):

    * *cert-clean*: **0** soundness violations — no dual bound above its reference
      optimum, none crossing its own incumbent — **0** certification regressions and
      **0** certified objectives changed.
    * *net-positive*: 2 bounds RECOVERED from ``None`` (``casctanks`` -> 1.2584,
      ``bchoco08`` -> 1.0) and 3 tightened (``4stufen`` 18770 -> 19055, ``beuster``
      5942 -> 6352, ``nvs09`` -50.59 -> -48.99, all still below the -43.13 oracle),
      against **1** marginally looser (``tanksize`` at 20 s, 1.253535 -> 1.253040, a
      4e-4 relative move, both below the 1.26864 certified optimum) and **0** lost.
      Punctuality is neutral-to-better: mean ``wall/time_limit`` 0.447 -> 0.440 with
      the two large wins on the class the issue names (``casctanks`` 1.23x -> 1.02x,
      ``beuster`` 1.13x -> 0.98x) and no pair newly over 1.25x.

    One measured counter-example, recorded rather than averaged away: ``hda`` at an
    8 s budget (not one of the panel's two) goes the other way — 9.68 s (sd 0.25) ->
    10.26 s (sd 0.10) over 3 interleaved reps per arm, 1.21x -> 1.28x, with a
    bit-identical bound. Root setup finishing 2.4 s earlier lets the search start two
    more nodes and the last one straddles the deadline: the pre-existing per-node
    overrun (#966), given more chances to fire, not a setup overrun.

    Scope of that panel, stated rather than glossed: it is the in-repo corpus, which
    does contain ``casctanks`` (one of the four overrun instances the issue lists,
    reproducing both symptoms) but not the two the issue names. Those were run
    separately on the owner's machine during the PR review and are the arm that
    closes it — ``sonet23v4`` at ``tl=2`` and ``watercontamination0202`` at
    ``tl=30``/``tl=60``, the instances behind #1152's own two tests: **3 pass** with
    the flag on its default and **3 xfail** under
    ``DISCOPT_ROOT_SETUP_BUILD_DEADLINE=0``, back to back at load 5.4. The OFF arm is
    what attributes the pass to this flag rather than to the machine, and doubles as
    an end-to-end check that the §5 opt-out is live. See
    ``docs/dev/1152-time-limit-root-setup-contract-2026-09-04.md``."""

    node_round_budget: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_NODE_ROUND_BUDGET", default=False)
    )
    """Make a per-node separated-relaxation ROUND honor its grant end-to-end
    (``DISCOPT_NODE_ROUND_BUDGET``, default **off**; §5 bound-changing; issue #966).

    #928 made the warm pure-LP path honor its per-solve ``time_limit``, and the
    residual budget overrun measurably moved OUT of the LP layer: a round's grant
    clamps only the LP solves, while the round's non-LP cost — the cold
    ``build_uniform_relaxation`` (~1.1–3.3 s on contvar) plus separation — is spent
    AFTER the admission check, unclamped. Measured (contvar @ 20 s, both arms,
    ``scratchpad/issue966_phase_probe.py``): a node round granted 2.0 s runs
    5.3–5.8 s, ~3.3 s of it the build; the budget-honoring LPs return sooner, the
    loop fits MORE such rounds, and the ON arm's wall goes UP (issue #966's
    20 s-budget sign flip: ON−OFF +325.4/+68.5/+12.7 s over 3 reps).

    When on, the spatial B&B node loops:

    * pass the round's grant to ``solve_at_node`` as an absolute
      ``round_deadline``, which (a) truncates the cold build's constraint-row
      loop at the grant (the #694/#832 anytime-build mechanism — a prefix of
      rows is a valid weaker outer relaxation) and (b) anchors the node's
      internal solve/separation deadline at the grant instead of restarting the
      clock after the build; and
    * run a round whose grant cannot cover the relaxer's measured cold-build cost
      (an EMA over this solve's builds) in **yield mode** instead of a full
      round: no per-node separation chain, and the cold build truncated at the
      grant, so the round banks the weaker-but-valid bound and the LP vertex it
      can afford rather than nothing — **except in the ROOT batch, which always
      runs a full round** (#928 rule 1). The root has no parent bound, so a
      weakened round there leaves the whole tree with no bound source; that is
      the ``bound=None`` collapse the first coupled graduation panel measured on
      contvar (7 → 287 nodes, nothing ever certified). The same rule already
      governs the root-relaxation fallback's ``_fb_stop``: a phase is optional
      tightening only once some valid bound is in hand.

    Yield mode replaced an outright *decline* (the first cut of this flag, and
    the measured cause of the coupled panel's bound ledger: casctanks
    2.9098 → −56.5001, contvar's bound lost outright). A skipped round banks no
    bound AND no LP point, and the point is what the spatial brancher and the
    primal heuristics run on: forcing every round to skip on nvs05 @ 20 s cost
    the bound (3.514 → 0.684), the incumbent (8.73 → 523.69) and the search
    itself (29 → 1 nodes, ending 16 s inside its own budget), while yielding the
    same rounds kept the incumbent, recovered half the lost bound (1.353) and
    branched 45 nodes (``scratchpad/issue966_yield_vs_decline.py``).

    Sound by construction: build truncation and skipped separation only drop
    rows/cuts (weaker, never falsified — the ``_objective_bound_valid`` gate
    catches an un-bounded cost column), deadline-cut LP solves already bank the
    Neumaier-Shcherbina floor, and a round that IS cut short no longer returns
    nothing: it reports the rigorous box-interval objective floor of the
    (possibly truncated) relaxation it built — measured, a spent round grant lost
    the bound outright on 16 of 114 cells of the binding subset before that
    (#928, see ``_solve_at_node_impl``). Should even that floor be unavailable, a
    yielded round leaves its node OPEN at ``-inf`` (floored at the proved parent
    bound) rather than fathomed-without-proof on the failure sentinel — see
    ``solver._yield_keeps_node_open``. Default off pending the §5 corpus-wide
    differential panel (cert-clean AND net-positive); graduation is coupled to
    the #928 ``DISCOPT_LP_WARM_DEADLINE`` panel this flag exists to unblock."""

    hessian_compile_gate: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_HESS_COMPILE_GATE", default=False)
    )
    """Refuse to START a nonconvex node NLP whose first-time sparse-Hessian XLA
    compile cannot fit the solve's remaining budget
    (``DISCOPT_HESS_COMPILE_GATE``, default **off**; §5 bound-changing; issue #966).

    #966's rare severe overruns (200–500 s past a 20 s budget, BOTH
    ``DISCOPT_LP_WARM_DEADLINE`` arms) were caught in flight with
    ``faulthandler.dump_traceback_later``: the phase is the **uninterruptible
    first-time XLA compile of the colored-HVP Lagrangian-Hessian kernel**
    (``jit_batch_hvp``, ``sparse_hessian.py``), fired from a node NLP —
    heatexch_gen3 @ 20 s: entered near the deadline, XLA's own alarm reported the
    compile at 124 s, run wall 162.5 s. The F4 root-heuristic budget gate
    (``_root_heur_nlp_entry_ok``) exists for exactly this risk but the per-node
    NLP entries (root multistart on the no-relaxer class, strided node NLP,
    batch POUNCE) bypass it. An eager (``jax.disable_jit``) fallback was
    falsified as the fix (§4 entry experiment,
    ``scratchpad/issue966_eager_hessian_entry.py``): eager evaluation costs
    8–10 s PER CALL steady-state on this class — worse than useless inside a
    20 s budget. A compile cannot be interrupted, so entry refusal is the whole
    mechanism (the F4 philosophy).

    When on, the spatial node loops' per-node NLP entries (the root multistart
    on the no-relaxer class — the caught severe mode — and the strided node
    NLP; ``_hess_compile_refuses`` in ``solver.py``) plus the JAX-callback
    batch-POUNCE path decline to start a solve when the model is NONCONVEX and
    the evaluator's (conservative) first-compile estimate exceeds the remaining
    budget. Convex solves are never gated — on the convex path the node NLP is
    the bound producer (rule 1: never skip the sole bound source) — which is
    why the check lives at the loop sites, where ``_model_is_convex`` is
    authoritative (the multistart chain does not thread convexity down). The
    native-AD POUNCE path never compiles XLA and is never gated. Bound-changing
    because a refused nonconvex node also skips the
    alphaBB/interval bounds nested behind the NLP on the serial path: the node
    stays OPEN at its inherited parent bound (weaker, never false) and the
    incumbent that NLP might have found is forgone. Residual exposure, recorded
    honestly: the compile cost is measured-unpredictable (1–186 s, R² ≈ 0 vs
    size — see ``estimate_hessian_compile_s``), so an entry admitted with
    remaining budget above the risk floor can still overrun; this flag kills the
    observed late-entry severe modes, not the early-entry gamble the F4 floor
    deliberately accepts. Default off pending the §5 differential panel
    (graduation coupled to the #928/#966 panel)."""

    singular_tangent: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_SINGULAR_TANGENT", default=False)
    )
    """Recover the univariate-envelope tangent facet dropped at a **vertical
    tangent** (``DISCOPT_SINGULAR_TANGENT``, default **off**; §5 bound-changing;
    issue #1111).

    ``uniform_relax._emit_1d`` places tangents at ``lo``, the midpoint and ``hi``.
    Where ``f`` is finite at an endpoint but ``f'`` diverges there, ``_tangent_row``
    returned without emitting and the facet was silently lost, leaving the envelope
    one-sided on that side — reached by ``sqrt`` at ``t=0`` and ``asin``/``acos`` at
    ``t=±1``. When on, the facet is re-anchored at an interior ladder point (see
    ``_interior_tangent_point``); the path only ever ADDS a row where none was
    emitted, so the flag-ON polytope is a subset of the flag-OFF polytope and the
    node LP bound can only improve or stay equal.

    **The §5 panel ran. Gate 1 (cert-clean) PASSES; gate 2 (net-positive) FAILS:
    the EAGER anchor is measured HARMFUL.** On the only instance that terminates
    inside a deterministic node budget and moves at all, ``tspn08`` goes 135 → 191
    nodes (**+41.5 %**) to buy a bound gain in the 11th digit
    (290.56592504129753 → 290.56599569540646). ``mathopt5_6`` is flat at 5 → 5 with
    a bit-identical bound, and ``kriging_peaks-full010`` is flat to 14 digits. The
    subset property still holds — the ON bound *is* tighter at every node — but a
    different LP vertex changes the branching choice, so a tighter relaxation can
    still grow the tree. Soundness held throughout: no bound above its ``.solu``
    reference, no certification regression, incumbents identical.

    The root win also does not survive branching: ``kriging_peaks`` gains 13–40× at
    the root (``full200`` −74356.93 → −5596.46) yet after a few hundred nodes the
    arms agree to 14 digits. B&B was already recovering that bound cheaply.

    **DOUBLE RETRACTION (§11) — a retraction that was itself wrong.** This docstring,
    the CHANGELOG and PR #1113 previously *withdrew* the ``tspn08`` 135 → 191 result
    and recorded the flag as "sound, and neutral", on the strength of a panel showing
    135 → 135 with bit-identical bounds. **That withdrawal is itself withdrawn, and
    the +41 % is reinstated as measured above.** The panel behind it never fired.

    The instrument defect, measured not inferred: :func:`solve` is wrapped by a
    decorator (``solver.py:6274-6280``) that begins ``_set_tuning(kwargs.pop(
    "tuning", None))``, and :func:`set_current` with ``None`` publishes a **fresh,
    env-resolved** ``SolverTuning()``. A probe that installs a tuning with
    ``set_current(...)`` and then calls ``m.solve()`` without ``tuning=`` therefore
    has that context **silently discarded at the solve boundary**. Counting
    ``_interior_tangent_point`` calls inside a full solve of
    ``kriging_peaks-full010``:

    ==========================================  ======  =====
    how the flag was set                         calls  nodes
    ==========================================  ======  =====
    off (control)                                    0    311
    ``set_current(...)`` around ``m.solve()``        0    311
    ``m.solve(tuning=...)``                       2671    313
    ``DISCOPT_SINGULAR_TANGENT=1``                2671    313
    ==========================================  ======  =====

    ``tuning=`` and the environment variable agree bit-for-bit; ``set_current`` was
    inert. So "both arms bit-identical" was not a neutrality result — it is the
    signature of a probe that never fired, and it explains exactly why the arms
    agreed to the last digit. That panel carried no drops counter, so nothing caught
    it (§6). Every root-relaxation measurement quoted below is unaffected: those
    drive ``build_uniform_relaxation`` directly, where ``set_current`` does reach
    (instrumented 10/10).

    **That delivery gap is now fixed (#1117)** — the solve boundary calls
    :func:`enter_scope`, which inherits an ambient context instead of overwriting it,
    and the deep-recursion worker thread carries the caller's contextvars context
    across. The table above is kept as the record of what the broken instrument
    produced, not as current behavior; ``set_current(...)`` around a plain
    ``m.solve()`` now fires the flag exactly like ``tuning=``.

    The reinstated number was re-measured on a corrected instrument — ``tuning=``
    delivery, the ON arm asserted to have fired, a ``max_nodes`` budget with **no**
    ``time_limit`` (a wall limit changes the kernel path and makes the run
    non-reproducible; ``max_nodes`` alone is bit-reproducible over 3 repetitions per
    arm), both arms in one process on one binary — and reproduces at 135 → 191 on
    every run since.

    Still retracted, and for reasons unrelated to the above: the first panel's
    time-limited rows (``kriging_peaks-full100`` −1.144, ``tspn12`` −0.520, …), taken
    at a 60 s wall limit under load 37–87 on 14 cores, fail the §9 load gate
    outright; and that panel's printed tally ``better 10 worse 2 unchanged 1``, whose
    classifier scored ``nodes_on < nodes_off`` as "better" — correct for a
    terminating run, backwards for a time-limited one, where fewer nodes means the
    arm did *less* work in the same budget.

    #1111's own motivating hypothesis is **falsified**: on the adiabatic LVC MECP
    model the facet is recovered but root bound and node count are unchanged (1 node
    either way).

    **Two further defects in #1111's framing, both measured.** (a) The issue expects
    the inverse-trig rows to matter most; across all 1610 MINLPLib ``.nl`` instances
    ``sqrt`` appears in 63 and ``asin``/``acos``/``acosh`` in **zero**, so on this
    corpus the feature is a sqrt feature. (b) The first panel was ~36 % diluted — 5 of
    its 14 instances (``kriging_peaks-red*``) contain no ``sqrt`` at all, so the flag
    could not act on them and those rows were pure noise. That dilution is why its
    numbers are retracted above; the panel quoted here is its successor, drawn by
    screening every corpus instance for an actually-dropped facet.

    **The mechanism, not the constant, is what is wrong.** A fixed-offset
    reformulation (``delta = width/8``) was built and measured against this ladder and
    is **strictly worse** — about half the root-bound gain on every instance where the
    bound moves at all — and the anchor sweep behind
    :attr:`singular_tangent_kappa` shows the best static offset is problem-dependent
    by orders of magnitude, with both ends of the range degenerate. A static geometric
    anchor cannot be right for every box. What the measurements point at instead is
    placing the tangent *where the LP binds* rather than at a geometric offset — which
    is what ``MccormickLPRelaxer._separate_convex``'s Kelley loop already does for
    composite convex/concave lifts (``concave: d <= g(x0) + grad g(x0)·(x - x0)`` at
    the LP point ``x0``). The soundness argument carries over unchanged — every tangent
    of a concave ``f`` is a global overestimator — and lazy separation also stops the
    solver paying for an eager extra row at every node whether or not that node's LP
    is near the singular endpoint, which is precisely the cost the ``tspn08``
    regression above is charging.

    **That successor was built and measured — #1115, and it is the default
    placement.** See :attr:`singular_tangent_lazy`. Eager anchoring is retained only
    as the A/B control for that measurement.

    **Final disposition (#1115).** The lazy form was *not* falsified, so the #581
    removal precedent does not apply: it is sound, it never loses the bound on the
    corpus, and it buys a real bound gain at an identical node count on
    ``kriging_peaks``. It fails gate 2 only because it costs +25.6 %/+54.2 % on the
    instances it does not help. This flag therefore stays default-OFF **and stays in
    the tree**; what is missing is a trigger keyed on whether the facet binds often
    enough to pay for its rows, not a better mechanism. Flag-OFF is byte-identical to
    the pre-#1111 relaxation."""

    singular_tangent_lazy: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_SINGULAR_TANGENT_LAZY", default=True)
    )
    """Place the recovered vertical-tangent facet at the LP point (#1115).

    Consulted only when :attr:`singular_tangent` is on; it selects *where* the
    recovered tangent goes, never *whether* the facet is recovered.

    ``True`` (the default) defers the facet to
    ``MccormickLPRelaxer._separate_singular_tangent``, which adds the supporting
    tangent at the current LP point and only when that point violates it.
    ``False`` restores #1111's eager behaviour: one tangent per node at a fixed
    geometric ladder anchor near the endpoint, whether or not it binds.

    Where the LP vertex sits *on* the singularity — the case the facet exists for,
    and where no finite-slope tangent is available — the touch point falls back to
    #1111's conditioning-capped ladder anchor while the violation test still decides
    whether the row goes in. Without that fallback the separator degrades to a no-op
    precisely where it is needed: on ``min 2x - sqrt(x)`` over ``[0,4]`` the LP vertex
    lands exactly at ``t=0``.

    **Measured.** The eager form is **harmful** — ``tspn08`` 135 → 191 nodes
    (+41.5 %) for a bound gain in the 11th digit — which is what motivated #1115, and
    it is retained only as the A/B control. Lazy placement removes that regression
    (``tspn08`` back to 135 nodes) and, on isolated atoms, does what it was designed
    to do: ``min 2x - sqrt(x)`` over ``[0,4]`` goes −0.7071 → **−0.12503** against a
    true optimum of −0.125, where the eager anchor reaches only −0.6557.

    **On the corpus lazy never loses the bound, and that is not enough.** Three-arm
    panel (off / eager / lazy), 300-node budget with no wall limit, each arm's firing
    asserted, 12 scorable instances: ``eager vs off: BETTER 1, WORSE 2, flat 9``;
    ``lazy vs off: BETTER 3, flat 9, worse 0``. Soundness checked on every arm of
    every run against ``minlplib.solu`` — 0 violations, including per-repetition
    asserts in the reproducibility and timing panels. The lazy gains are
    ``kriging_peaks-full050`` (−142.657 → −139.918) and ``full100`` (−348.054 →
    −342.588), ~1.6–1.9 % of gap, plus a marginal ``tspn15``; each occurs at an
    *identical node count and identical incumbent*, i.e. a better bound for the same
    tree. Eager's one BETTER cell (``tspn15``, +0.002) against two real losses leaves
    it net-negative.

    **The timing panel decides it, and it decides against graduation.** Interleaved
    off/lazy, 3 repetitions, per-arm and pooled standard deviations, load gate
    recorded per instance (2.5–5.2 on 14 cores), both arms verified to explore
    identical trees::

        eq6_1                 off  10.82+-0.12s   lazy  13.59+-0.09s   +25.6%   0 gain
        maxmin                off  33.52+-0.13s   lazy  51.68+-0.04s   +54.2%   0 gain
        kriging_peaks-full050 off  42.21+-0.21s   lazy  44.09+-0.12s    +4.5%  +1.9%
        kriging_peaks-full100 off 101.21+-0.21s   lazy 105.22+-0.22s    +4.0%  +1.6%

    Every delta is 12–200 pooled sd; none is noise. The shape disqualifies it:
    **lazy is cheap where it helps and expensive where it does not** — the two
    instances charged +25.6 % and +54.2 % are exactly the two that gain nothing,
    because they draw the most rows. Default-ON would pay the ``maxmin`` bill
    corpus-wide to collect the ``full050`` gain occasionally. Gate 1 (cert-clean)
    passes; **gate 2 (net-positive) fails**; the flag stays default-OFF. Same
    disposition as ``cut_inherit``, reached by a different route — that one was
    neutral-or-harmful, this one is helpful on a narrow class and unaffordable off it.

    Corrections to earlier readings recorded here (§11):

    * ``kriging_peaks-full200`` was reported as a third gain (−746.566 → −734.495,
      +1.6 %) and is **withdrawn**. Over 3 reps on a quiet machine it is
      nondeterministic in *both* arms: off gives nodes {301, 303} with two distinct
      bounds and zero separated rows; lazy gives {301, 303} with rows {32451, 32472,
      32583}. Two single runs of a nondeterministic instance are not a comparison.
      Pre-existing at that size, unrelated to this flag, filed separately.
    * An intermediate claim that lazy *caused* that nondeterminism is also withdrawn:
      it rested on two reps in which off agreed, and the third falsified it.
    * **The quoted bounds are load-dependent to ~3 significant figures.** Quiet,
      ``full100`` gives −350.768 → −345.634 over 12 498 rows; under the original
      panel's contention, −348.054 → −342.588 over 13 214. Separation is wall-bounded
      (every ``_separate_*`` breaks on a shared ``_deadline``; the node LP re-solve
      takes ``time_limit=_remaining()``), so row counts and the bounds they produce
      shift with machine load. The direction reproduces; the digits do not.

    Remaining gap:

    * **A coverage hole eager does not have.** The separation chain is gated on
      ``if separate:`` (``mccormick_lp.py``), forced off on yield rounds and pool-free
      re-solves. On all three ``elec*`` instances the lazy arm registered thousands of
      specs (19 236 on ``elec100``) and the separator ran **zero** times, while eager
      fired at build time throughout. Lazy is not a coverage superset of eager: it
      trades "emit a geometric guess wherever a relaxation is built" for "emit an
      exact tangent only where a separation round runs".

    Where the gains come from: both sit in ``kriging_peaks``, the family with the
    largest root gain (13–40×). On ``full010``, 12 separator invocations move the node
    LP objective 7 times but only by ~1e-4 against a bound of −5.69 — live, and far too
    small to change a branching decision. So the #1111 finding "the root win does not
    survive branching" is scoped to the **eager** anchor: where the dropped facet
    dominates the root relaxation, lazy placement does retain part of it.

    **Why this stays default-OFF rather than becoming conditional (#1119, closed
    falsified 2026-08-23).** The successor question was whether a hit-rate gate —
    keep the rows that bind in the LP optimal basis, drop the ones that do not —
    could remove the ``eq6_1``/``maxmin`` overhead while retaining the
    ``kriging_peaks`` gain. It cannot, and the measurement runs the wrong way:
    over 200 nodes the two instances that PAY bind at 1.0000 and 0.9580, the two
    that GAIN bind at 0.9173 and 0.8509, and ``eq6_1`` has **zero** non-binding
    rows — so the gate drops 0 of its 22 874 rows and saves none of the +25.6 %,
    while discarding 14.9 % of the rows on the instance the feature exists for.
    Binding is near-tautological here: a row is emitted because it is violated at
    the current LP point, so the re-solve that follows lands on it. Full record
    (including the screen showing only 8 of 96 candidate instances emit any row at
    all) in ``docs/dev/performance-plan.md`` §17; the instrument is
    ``MccormickLPRelaxer.singular_tangent_stats``.
    """

    singular_tangent_kappa: float = field(
        default_factory=lambda: _env_float("DISCOPT_SINGULAR_TANGENT_KAPPA", 100.0)
    )
    """Cap on ``|f'(t0)|`` for :attr:`singular_tangent`, as a multiple of the box's
    own slope scale (``DISCOPT_SINGULAR_TANGENT_KAPPA``, default 100).

    A measurement knob for the §5 panel, not a user-facing tuning parameter. Only
    consulted when :attr:`singular_tangent` is on.

    **The cap is right; the rationale first written here was not, and is retracted
    (§11).** The original justification was that the outward-rounding guard
    (``outward_rounding.envelope_1d_slack``) grows linearly with ``|slope|``, so an
    uncapped near-vertical tangent buys tightness with rounding slack. The linearity
    is real (``outward_rounding.py`` returns ``outward_slack(d*tmag + ...)``) but the
    magnitude is not: at ``kappa=100`` the added slack is ~1e-14 against a cut depth
    of ~0.35 — six orders of magnitude too small to be the reason for anything.

    The cap's actual justification is a measured degeneracy at the uncapped end.
    Sweeping the anchor offset ``delta`` over eight orders on the three
    ``kriging_peaks-full`` instances that move the root bound at all, the root LP
    bound tightens as the anchor approaches the singular endpoint, peaks near
    ``delta ~ 1e-5``, and then **collapses back to the flag-OFF value** as ``delta``
    shrinks further (``full020``: OFF −1113.72; gain +552 at 1/8, +1087 at 1.2e-4,
    +1098 at 1e-5, +1097 at 1e-7, +1039 at 1e-9, **+0** at 1e-12 — same shape on
    ``full010`` and ``full050``). Past the peak the tangent is numerically degenerate
    and the emitted row stops constraining anything. Without a cap the ladder takes
    its smallest rung, ``delta = 0.5*8**-20 ~ 5.5e-19``, which is inside the collapse:
    the feature would emit a row on every singular atom and buy nothing. That is what
    the cap is for, and it reproduces on an isolated atom too — ``min 3x - sqrt(x)``
    over ``[0,4]``, whose LP optimum sits at the singular end, gains **+4.9e-09** at
    ``kappa=1e12`` against **+0.083** at ``kappa=100``.

    **``kappa=100`` is not measured-optimal, and the optimum is problem-dependent.**
    On that same isolated atom the best rung is near ``kappa ~ 10`` (gain +0.619, 7x
    the default's +0.083), i.e. an anchor *further* from the endpoint; on
    ``kriging_peaks`` the peak is at a *smaller* ``delta`` than the default selects.
    The two disagree by orders of magnitude, so no single constant is right for both
    — which is the deeper reason a static geometric rule is the wrong mechanism here
    (see :attr:`singular_tangent`). Treat 100 as a serviceable midpoint that is on
    the stable side of the collapse, not as a tuned value. (The isolated-atom figure
    is a synthetic proxy and is quoted only for the *shape* of the curve; per the
    #727 RLT lesson the ``kriging_peaks`` numbers are the ones drawn from the real
    corpus.)

    This also **falsifies a mean-envelope-slack argument** considered as a
    replacement for the ladder (a fixed ``delta = width/8``, chosen because it
    minimises mean envelope slack over the box, measured 3.31x better than the
    near-corner anchor on that metric). Mean slack over the box does not predict
    bound tightness: on the same instances the near-corner anchor gives ~**2x** the
    root-bound gain of ``width/8`` (``full010`` +2975 vs +1513, ``full200`` +68769
    vs +34967). Only slack *where the relaxation binds* matters, and on this family
    the LP optimum sits at the singular endpoint. The fixed-offset reformulation was
    measured, found strictly worse, and discarded."""

    # NOTE (#581): ``DISCOPT_NODE_REDUCE`` (per-node cheap reduction: cutoff-FBBT +
    # free DBBT from node-LP reduced costs + integer RC-fixing, feeding the
    # tightened box to the children) was DEPRECATED and removed. It was a
    # default-OFF, bound-changing flag that graduated-gated net-negative (PR #685:
    # benefit 24% / regression 18% on the held-out N=20 arm — regressed ex5_3_3,
    # spring, qapw) — sound but not helpful, so it is removed rather than left in
    # default-OFF limbo. Removing the default-OFF gated branch (and its
    # ``discopt._relax.node_reduce`` module) is byte-identical to the shipped
    # default path (which never entered it).

    def __post_init__(self) -> None:
        if self.rlt_quad_max < 1:
            raise ValueError(f"rlt_quad_max must be >= 1, got {self.rlt_quad_max}")
        if self.rlt_lineq_max < 0:
            raise ValueError(f"rlt_lineq_max must be >= 0, got {self.rlt_lineq_max}")
        if self.rlt_sparse_max_vars < 1:
            raise ValueError(f"rlt_sparse_max_vars must be >= 1, got {self.rlt_sparse_max_vars}")
        if self.rlt_sparse_max_terms < 1:
            raise ValueError(f"rlt_sparse_max_terms must be >= 1, got {self.rlt_sparse_max_terms}")
        if self.rlt_sparse_min_root_gain < 0:
            raise ValueError(
                f"rlt_sparse_min_root_gain must be >= 0, got {self.rlt_sparse_min_root_gain}"
            )
        if self.multilinear_rlt_max < 1:
            raise ValueError(f"multilinear_rlt_max must be >= 1, got {self.multilinear_rlt_max}")
        if self.node_nlp_stride < 1:
            raise ValueError(f"node_nlp_stride must be >= 1, got {self.node_nlp_stride}")
        if self.ils_solve_cap < 0:
            raise ValueError(f"ils_solve_cap must be >= 0 (0 = uncapped), got {self.ils_solve_cap}")
        if self.node_bound_mode not in ("lp", "milp"):
            raise ValueError(
                f"node_bound_mode must be 'lp' or 'milp', got {self.node_bound_mode!r}"
            )
        if self.relax_space not in ("auto", "lifted", "reduced", "hybrid"):
            raise ValueError(
                "relax_space must be 'auto', 'lifted', 'reduced', or 'hybrid', "
                f"got {self.relax_space!r}"
            )
        if self.psd_cost_gate_budget <= 0:
            raise ValueError(f"psd_cost_gate_budget must be > 0, got {self.psd_cost_gate_budget}")
        if self.psd_cost_gate_tau < 0:
            raise ValueError(f"psd_cost_gate_tau must be >= 0, got {self.psd_cost_gate_tau}")
        if not (self.singular_tangent_kappa > 0.0 and math.isfinite(self.singular_tangent_kappa)):
            raise ValueError(
                f"singular_tangent_kappa must be finite and > 0, got {self.singular_tangent_kappa}"
            )

    def replace(self, **changes) -> SolverTuning:
        """Return a copy with ``changes`` applied (validated)."""
        valid = {f.name for f in fields(self)}
        bad = set(changes) - valid
        if bad:
            raise TypeError(f"unknown SolverTuning field(s): {sorted(bad)}")
        return SolverTuning(**{**{f.name: getattr(self, f.name) for f in fields(self)}, **changes})


# Published for the duration of a solve_model() call; relaxer read sites consult
# current() instead of os.environ. Default None -> current() reads env fresh.
_current: ContextVar[SolverTuning | None] = ContextVar("discopt_solver_tuning", default=None)


def current() -> SolverTuning:
    """The active :class:`SolverTuning` (a fresh env-resolved one outside a solve)."""
    active = _current.get()
    return active if active is not None else SolverTuning()


def set_current(tuning: SolverTuning | None):
    """Publish ``tuning`` (or a fresh env-resolved one) as active; returns the token."""
    return _current.set(tuning if tuning is not None else SolverTuning())


def enter_scope(tuning: SolverTuning | None):
    """Publish ``tuning`` for a solve scope, **inheriting** the active context when
    ``tuning`` is ``None``; returns the token to hand to :func:`reset_current`.

    This is the difference between "no override was requested" and "override with
    env defaults". :func:`set_current` cannot express the former — ``None`` there
    means *a fresh env-resolved instance* — so the solve boundary used to overwrite
    whatever a caller had installed with :func:`set_current`, discarding it in
    silence (issue #1117). The precedence here is explicit ``tuning=`` kwarg >
    ambient context > environment defaults, which is what a caller reading

    .. code-block:: python

        token = solver_tuning.set_current(tuning)
        try:
            m.solve()
        finally:
            solver_tuning.reset_current(token)

    already expects. Nested solves inherit the outer scope for the same reason.
    """
    active = _current.get()
    if tuning is not None:
        return _current.set(tuning)
    return _current.set(active if active is not None else SolverTuning())


def reset_current(token) -> None:
    _current.reset(token)
