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

**Naming rule, deliberately mechanical.** A field's name is the original local's
name with leading underscores stripped, so every migration site is a one-token
edit that ``grep`` can audit:

===========================  ==================================
local in ``solve_model``     field
===========================  ==================================
``rust_time``                ``_timers.rust_time``
``t_rust_start``             ``_timers.t_rust_start``
``_subnlp_calls``            ``_heur.subnlp_calls``
``_lns_lb_calls``            ``_heur.lns_lb_calls``
===========================  ==================================

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
    "PhaseTimers",
    "PrimalHeuristicState",
]


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
