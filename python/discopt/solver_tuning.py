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


def _env_flag(name: str, *, default: bool) -> bool:
    """``DISCOPT_<name>`` as a boolean (``"0"`` is false, anything else true)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
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

    # --- RLT (reformulation-linearization) families ---------------------------
    rlt: bool = field(default_factory=lambda: _env_flag("DISCOPT_RLT", default=False))
    """Legacy whole-relaxation RLT toggle (``DISCOPT_RLT``). The ``rlt=`` argument
    to :meth:`Model.solve` is the primary control; this OR-s in alongside it."""

    rlt_quad: bool = field(default_factory=lambda: _env_flag("DISCOPT_RLT_QUAD", default=True))
    """Quadratic RLT row generation (``DISCOPT_RLT_QUAD``, default on)."""

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
    """Use nested bilinear instead of the tight trilinear envelope
    (``DISCOPT_TRILINEAR=nested``, default off)."""

    trilinear_exact: bool = field(
        default_factory=lambda: os.environ.get("DISCOPT_TRILINEAR") == "exact"
    )
    """Use the best-of-three nested trilinear envelope instead of Meyer-Floudas
    (``DISCOPT_TRILINEAR=exact``, default off)."""

    trilinear_rlt: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_TRILINEAR_RLT", default=True)
    )
    """Trilinear RLT rows (``DISCOPT_TRILINEAR_RLT``, default on)."""

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

    # --- branch-and-bound / bound levers --------------------------------------
    alphabb_with_lp: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_ALPHABB_WITH_LP", default=False)
    )
    """Force the alpha-BB bound alongside the LP relaxation
    (``DISCOPT_ALPHABB_WITH_LP``, default off)."""

    lifted_fbbt: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_LIFTED_FBBT", default=False)
    )
    """Feasibility-based bound tightening on lifted columns
    (``DISCOPT_LIFTED_FBBT``, default off)."""

    node_bound_mode: str = field(
        default_factory=lambda: os.environ.get("DISCOPT_NODE_BOUND_MODE", "lp")
    )
    """Per-node dual bound: ``"lp"`` (default, lifted-McCormick LP) or ``"milp"``
    (legacy nested integer MILP node solve) — ``DISCOPT_NODE_BOUND_MODE``."""

    node_nlp_stride: int = field(default_factory=lambda: _env_int("DISCOPT_NODE_NLP_STRIDE", 4))
    """Solve the node NLP every k-th node (``DISCOPT_NODE_NLP_STRIDE``, default 4)."""

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
        default_factory=lambda: _env_flag("DISCOPT_OBJ_BRANCH_PRIORITY", default=False)
    )
    """Prioritize branching on objective-defining variables
    (``DISCOPT_OBJ_BRANCH_PRIORITY``, default off)."""

    lp_warmstart: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_LP_WARMSTART", default=True)
    )
    """Warm-start the node LP from the parent basis (``DISCOPT_LP_WARMSTART``)."""

    # --- branch-and-reduce (cert:T2.3 / T2.4, default OFF until T2.6) ----------
    root_fixpoint: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_ROOT_FIXPOINT", default=False)
    )
    """Run the cutoff-aware root branch-and-reduce fixpoint (cert:T2.3) at the end
    of iteration 0: iterate {FBBT-with-cutoff, OBBT/DBBT-with-cutoff} to a fixpoint
    on the root box, refreshing the root cut pool + incremental engine base from the
    tightened box. ``DISCOPT_ROOT_FIXPOINT``, default OFF (bound-changing; unlocked
    by the R1 GO, flipped default-on only after nightly-green per T2.6)."""

    node_reduce: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_NODE_REDUCE", default=False)
    )
    """Run the per-node cheap reduction (cert:T2.4b ``reduce_node``) after each
    spatial node LP solve: cutoff-FBBT + free DBBT from the node LP reduced costs
    (z = safe_bound, the C-15 rule) + integer RC-fixing, feeding the tightened box
    to the child boxes. ``DISCOPT_NODE_REDUCE``, default OFF (bound-changing)."""

    def __post_init__(self) -> None:
        if self.rlt_quad_max < 1:
            raise ValueError(f"rlt_quad_max must be >= 1, got {self.rlt_quad_max}")
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
