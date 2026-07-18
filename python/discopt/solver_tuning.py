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

    rlt_quad_max: int = field(default_factory=lambda: _env_int("DISCOPT_RLT_QUAD_MAX", 256))
    """Column cap for quadratic RLT (``DISCOPT_RLT_QUAD_MAX``, default 256)."""

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

    # NOTE (#581): ``DISCOPT_NODE_REDUCE`` (per-node cheap reduction: cutoff-FBBT +
    # free DBBT from node-LP reduced costs + integer RC-fixing, feeding the
    # tightened box to the children) was DEPRECATED and removed. It was a
    # default-OFF, bound-changing flag that graduated-gated net-negative (PR #685:
    # benefit 24% / regression 18% on the held-out N=20 arm — regressed ex5_3_3,
    # spring, qapw) — sound but not helpful, so it is removed rather than left in
    # default-OFF limbo. Removing the default-OFF gated branch (and its
    # ``discopt._jax.node_reduce`` module) is byte-identical to the shipped
    # default path (which never entered it).

    def __post_init__(self) -> None:
        if self.rlt_quad_max < 1:
            raise ValueError(f"rlt_quad_max must be >= 1, got {self.rlt_quad_max}")
        if self.rlt_sparse_max_vars < 1:
            raise ValueError(f"rlt_sparse_max_vars must be >= 1, got {self.rlt_sparse_max_vars}")
        if self.rlt_sparse_max_terms < 1:
            raise ValueError(f"rlt_sparse_max_terms must be >= 1, got {self.rlt_sparse_max_terms}")
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


def reset_current(token) -> None:
    _current.reset(token)
