"""Explicit state objects for ``solve_model``.

**Why this module exists.** ``solve_model`` is 7,600 lines and four of the five
modules Card 4b named (``setup``/``reformulate``/``root``/``spatial_loop``) are
*inline statement blocks* of it rather than functions. The census in
``discopt_benchmarks/scripts/solve_model_locals_census.py`` measured what actually
couples them: of 851 names bound in the function's own scope, **153 are bound in
one of those regions and read in another** — 85 across the ``root``→``loop``
boundary alone. Those 153 are the implicit closure a carve would have to turn into
a parameter list. This module turns them, group by group, into named typed objects
so that the later carve is mechanical rather than a signature-design problem.

**Nothing here changes solver behavior.** Each class is a container for locals that
already existed with exactly these lifetimes; threading them is Regime N
(``node_count`` and certified ``objective`` exactly unchanged).

**Naming rule.** A field's name is the original local's name with leading
underscores stripped, minus the holder's own subject prefix where the holder
already carries it. A holder that covers two subjects keeps the distinguishing
prefix on its fields:

============================  ==================================
local in ``solve_model``      field
============================  ==================================
``rust_time``                 ``_timers.rust_time``
``_lazy_probe_spent``         ``_lazy.probe_spent``
``_pn_obbt_spent``            ``_pn_obbt.spent``
``_subnlp_calls``             ``_heur.subnlp_calls``  (``_heur`` also carries LNS)
``_lns_lb_calls``             ``_heur.lns_lb_calls``
============================  ==================================

The authoritative mapping is the ``MIGRATED`` table in
``python/tests/test_solver_state.py``; the tests enforce it in **both**
directions, so a field with no table entry and a table entry with no field both
fail. That is deliberately stricter than a naming convention, because a
convention is exactly the thing a later edit stops honouring silently.

**Why several small classes and not one.** The census's bind-region → read-region
matrix is not a single blob: the crossing names fall into cohesive clusters
(timing accounting, primal-heuristic budgets, lazy-constraint arming, per-node OBBT
budget, the dual-bound certificate flags) that each cross ``root``→``loop`` and
``root``→``results`` together. A carved ``spatial_loop(...)`` taking six named
state objects is reviewable; one taking an 85-field god-object is not.

All classes use ``slots=True``: a mistyped attribute must raise rather than
silently create a new field, because several of these carry soundness state (the
certificate flags in particular) where a silent typo would be a wrong answer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "LazyStallSeparationState",
    "PerNodeOBBTBudget",
    "PhaseTimers",
    "PrimalHeuristicState",
    "RootConfig",
]


@dataclass(frozen=True, slots=True)
class RootConfig:
    """Solve-wide configuration that is **decided once and never mutated**.

    .. rubric:: Why the admission rule is the interesting part

    This is the only holder in this module that is ``frozen=True``, and freezing is
    a promise. A 2026-07-30 design review caught the previous revision of the plan
    about to break it: the locals census's ``CONFIG`` label meant "the *name* is
    never rebound", **not** "the *value* is immutable", and eleven names carrying
    that label are mutated in place — including the live B&B ``tree``, the
    per-node deadline dict ``opts``, and ``kwargs``. Because ``frozen`` blocks only
    field *rebinding*, a holder that admitted those would have offered a guarantee
    it does not make, and the mutation would still have worked, silently. The
    review's verdict was that such a holder is *worse than leaving them as locals*.

    So membership here is not a judgement call. A name is admitted only if
    ``discopt_benchmarks/scripts/solve_model_config_mutability_audit.py`` returns
    ``IMMUTABLE_TYPE`` or ``CLEAN`` for it, over four channels: syntactic mutation
    inside ``solve_model``, methods resolved through the receiver's class or a Rust
    ``&mut self`` signature, transitive mutation through callees, and — the one
    that settles most of this class outright — a proof from the type that no
    mutation is possible at all. Every field below is a ``str``/``int``/``bool``
    or an ``Optional`` of one: CPython gives these no in-place mutation, so the
    freeze is a *true* guarantee rather than a decorative one.

    ``reports/solve_model_config_mutability_audit.md`` is the per-name table, and
    ``python/tests/test_solver_state.py`` re-derives it: a field whose audit
    verdict is not a clear fails the suite rather than being caught by review.

    .. rubric:: Resolved forms only

    Three of ``solve_model``'s parameters have a *derived* twin computed later
    (``root_cut_max``/``_root_cut_max``, ``root_cut_rounds``/``_root_cut_rounds``,
    ``solver``/``_solver``). Carrying both would create a shadow pair that drifts
    the first time one side is updated and the other is not, so none of the three
    raw parameters is admitted; their resolved forms belong to a later holder built
    after the root region computes them. A test enforces the exclusion.

    .. rubric:: What deliberately stays a loose local

    The mutable handles — ``tree``, ``evaluator``, ``opts``, ``kwargs``,
    ``_adaptive_nlp_state``, ``_heuristic_governor``, ``_reduce_timers`` — stay
    plain locals. They are the four highest-load names in the group, and putting
    them behind a type named *config* is exactly the mislabel this class exists to
    avoid. They are threaded onto the *mutable* holders in this module, or not at
    all.

    .. rubric:: Cost

    Threading a local onto a ``slots=True`` dataclass replaces a ``LOAD_FAST`` with
    a ``LOAD_ATTR``, and per the plan's ledger row 15b the solver reads a clock at
    78 Python sites to decide how much work to do — so a large enough slowdown
    could move a node count with no logic change. Every field admitted so far is
    read only in the ``setup``/``reformulate``/``root``/``results`` regions, never
    inside the innermost node loop, which keeps that variable out of the first
    gate run rather than assuming it away. The names that *are* read in the loop
    are a later tranche, and they are where 15b has to be watched.
    """

    #: GDP reformulation strategy (``"bigm"``, ``"hull"``, …).
    gdp_method: str
    #: How McCormick envelope bounds are sourced.
    mccormick_bounds: str
    #: Skip the root convexity detection pass.
    skip_convex_check: bool
    #: Node-selection strategy for the Rust tree manager.
    strategy: str
    #: Worker threads for batched node evaluation.
    threads: int
    #: RLT engagement: ``False``, ``True``, or a named mode.
    rlt: bool | str
    #: Whether RLT cut separation is enabled.
    rlt_cuts: bool
    #: Force/forbid the NLP branch-and-bound path; ``None`` means auto.
    nlp_bb: bool | None
    #: Piecewise partition count for relaxations.
    partitions: int
    #: PSD cut separation.
    psd_cuts: bool
    #: Cut family selection string.
    cuts: str
    #: Use learned relaxation coefficients where available.
    use_learned_relaxations: bool
    #: Eigenvalue-based root bounding.
    eigenvalue_root_bound: bool
    #: Lagrangian dual bound at the root.
    lagrangian_bound: bool
    #: How often the Lagrangian bound is recomputed.
    lagrangian_frequency: int
    #: Polynomial presolve rewrites.
    presolve_polynomial: bool
    #: Reverse-AD presolve pass.
    presolve_reverse_ad: bool
    #: Sub-NLP backend name.
    subnlp_backend: str
    #: Whether the sub-NLP primal heuristic layer runs at all.
    subnlp_enabled: bool


@dataclass(slots=True)
class PhaseTimers:
    """Wall-clock split of a solve across the Rust / JAX / Python layers.

    ``rust_time`` and ``jax_time`` accumulate over the whole solve and are read
    back in the ``results`` region (and by the early-return result builders in
    ``root``) to compute ``python_time = wall_time - rust_time - jax_time``.
    ``t_rust_start`` / ``t_jax_start`` are the open marks of the currently-timed
    section; they are written and read within a region but the pair straddles the
    ``root``→``loop`` boundary, which is why they belong here rather than staying
    loose.

    Accumulators are plain floats rather than a context manager because the
    existing call sites open and close the marks across ``if``/``try`` boundaries
    that a ``with`` block cannot span without moving code — and moving code is
    explicitly out of scope for this change.
    """

    #: Seconds spent inside Rust (presolve, tree management, the LP simplex).
    rust_time: float = 0.0
    #: Seconds spent inside JAX (relaxation compilation, batched node evaluation).
    jax_time: float = 0.0
    #: ``perf_counter`` mark opening the currently-timed Rust section.
    t_rust_start: float = 0.0
    #: ``perf_counter`` mark opening the currently-timed JAX section.
    t_jax_start: float = 0.0


@dataclass(slots=True)
class PrimalHeuristicState:
    """Call/success accounting for the sub-NLP and LNS primal heuristics.

    These are *budgets and statistics*, not search state: the sub-NLP counters gate
    further calls against ``subnlp_max_calls`` and are reported on the
    ``SolveResult``; the LNS counters escalate the local-branching radius ``k``
    across calls and throttle the one-hot swap improver after consecutive misses.

    They are grouped because they share a lifetime — all are initialised in the
    ``root`` region's heuristic-state block, mutated only inside the spatial loop,
    and (for the sub-NLP three) read once more in ``results``. That is exactly the
    ``root``→``loop``→``results`` triangle the census flagged, so a carve has to
    pass them all across both boundaries or none.
    """

    #: Resolved sub-NLP backend callable, or ``None`` when the layer is disabled.
    subnlp_backend_fn: Callable[..., Any] | None = None
    #: Sub-NLP solves attempted (gated against ``subnlp_max_calls``).
    subnlp_calls: int = 0
    #: Sub-NLP solves that returned a feasible point.
    subnlp_feasible: int = 0
    #: Sub-NLP solves whose feasible point improved the incumbent.
    subnlp_incumbent_updates: int = 0
    #: Local-branching LNS invocations; indexes the ``k`` escalation schedule.
    lns_lb_calls: int = 0
    #: Node-diving LNS invocations (throttled separately from local branching).
    lns_dive_calls: int = 0
    #: Consecutive one-hot swap searches that failed to improve; the improver
    #: self-disables past a small threshold.
    lns_swap_misses: int = 0


@dataclass(slots=True)
class LazyStallSeparationState:
    """Stall-driven re-separation state for the lazy-constraint cut pool.

    Under active pool inheritance the driver watches the global lower bound: when
    it stops improving for ``_LAZY_RESEP_STALL_WINDOW`` solves the layer moves from
    ``idle`` to ``probing`` (re-separating cuts that inheritance would otherwise
    have suppressed), and once ``_LAZY_RESEP_PROBE_BUDGET`` is spent it goes
    ``muted``. That is a three-state machine plus two counters and a reference
    bound, and it was six loose locals: a reader had to reconstruct the machine
    from assignments scattered across the loop.

    ``resep_fires`` is the one field with a lifetime past the loop — it is
    reported as ``pool/stall_reseparations`` when the result is built.
    """

    #: Global lower bound at the last genuine improvement, or ``None`` before the
    #: first finite bound is seen.
    glb_ref: float | None = None
    #: Whether an in-tree bound improvement has armed stall detection at all.
    armed: bool = False
    #: Node-solves since the last bound improvement, against the stall window.
    stagnant_solves: int = 0
    #: Node-solves spent probing, against the probe budget.
    probe_spent: int = 0
    #: ``"idle"`` | ``"probing"`` | ``"muted"`` — the machine's current state.
    mode: str = "idle"
    #: Node-solves during which probing actually fired; reported on the result.
    resep_fires: int = 0


@dataclass(slots=True)
class PerNodeOBBTBudget:
    """Engagement gate and effort budget for per-node OBBT (Lever A).

    ``enabled``/``budget_total``/``topk`` are decided once in the ``root`` region
    and never rebound; ``spent`` accumulates inside the loop and is reported in
    ``results`` as the ``obbt`` reduction timer. The constant three are grouped
    with the accumulator deliberately: the loop reads all four at the single gate
    that decides whether a node gets OBBT
    (``enabled and spent < budget_total``), and a carve that passed only the
    mutable one would split that gate across two argument sources — exactly the
    kind of split that makes a boundary hard to review.
    """

    #: Whether per-node OBBT is engaged for this solve at all.
    enabled: bool = False
    #: Wall-clock budget for the whole solve's per-node OBBT.
    budget_total: float = 0.0
    #: Candidate cap per pass, or ``None`` for "no cap".
    topk: int | None = None
    #: Budget already consumed; compared against ``budget_total`` at the gate.
    spent: float = 0.0
