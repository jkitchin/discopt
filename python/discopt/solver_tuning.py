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

    obj_branch_priority: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_OBJ_BRANCH_PRIORITY", default=False)
    )
    """Prioritize branching on objective-defining variables
    (``DISCOPT_OBJ_BRANCH_PRIORITY``, default off)."""

    lp_warmstart: bool = field(
        default_factory=lambda: _env_flag("DISCOPT_LP_WARMSTART", default=True)
    )
    """Warm-start the node LP from the parent basis (``DISCOPT_LP_WARMSTART``)."""

    def __post_init__(self) -> None:
        if self.rlt_quad_max < 1:
            raise ValueError(f"rlt_quad_max must be >= 1, got {self.rlt_quad_max}")
        if self.multilinear_rlt_max < 1:
            raise ValueError(f"multilinear_rlt_max must be >= 1, got {self.multilinear_rlt_max}")
        if self.node_nlp_stride < 1:
            raise ValueError(f"node_nlp_stride must be >= 1, got {self.node_nlp_stride}")
        if self.node_bound_mode not in ("lp", "milp"):
            raise ValueError(
                f"node_bound_mode must be 'lp' or 'milp', got {self.node_bound_mode!r}"
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


def reset_current(token) -> None:
    _current.reset(token)
