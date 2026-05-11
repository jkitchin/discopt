"""Python presolve pass protocol (A3 of the roadmap).

The Rust orchestrator emits a structured ``PresolveDelta`` per pass; we
mirror that contract on the Python side so that Python passes
(JAX-driven convexity, NN bound tightening, reverse-AD propagation)
participate in the same fixed-point loop with the same deltas.

A Python pass implements :class:`PresolvePass`:

- ``name``: stable string identifier, e.g. ``"convex_reform"``.
- ``run(model_repr) -> dict``: take the Rust ``PyModelRepr`` (already
  reflecting any tightening Rust passes did this sweep), do the work
  (mutating ``model_repr`` via its ``tighten_var_bounds`` setter where
  applicable), and return a delta dict.

The delta dict has the same shape Rust passes emit through PyO3 — so
downstream consumers (post-solve recovery, logging) handle Python and
Rust deltas uniformly. :func:`make_python_delta` constructs a
canonical empty delta the pass can fill in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PresolvePass(Protocol):
    """Structural protocol for a Python-side presolve pass."""

    name: str

    def run(self, model_repr: Any) -> dict:
        """Run one invocation of this pass.

        Args:
            model_repr: a ``PyModelRepr`` (Rust IR exposed via PyO3).
                The pass may mutate variable bounds via
                ``tighten_var_bounds`` but should not rewrite the model
                topology in v0 (rewrites would invalidate Rust-side
                bounds caches).

        Returns:
            A delta dict matching the Rust delta shape — see
            :func:`make_python_delta` for the canonical fields.
        """
        ...


def make_python_delta(pass_name: str, pass_iter: int = 0) -> dict:
    """Construct an empty delta dict for a Python pass.

    Mirrors the field shape produced by the Rust ``PresolveDelta``
    serialization in ``crates/discopt-python/src/expr_bindings.rs``.
    """
    return {
        "pass_name": pass_name,
        "pass_iter": pass_iter,
        "bounds_tightened": 0,
        "aux_vars_introduced": 0,
        "aux_constraints_introduced": 0,
        "constraints_removed": [],
        "constraints_rewritten": [],
        "vars_fixed": [],
        "vars_aggregated": [],
        "work_units": 0,
        "wall_time_ms": 0.0,
    }


def delta_made_progress(delta: dict) -> bool:
    """Whether a delta represents observable progress.

    Mirrors ``PresolveDelta::made_progress`` on the Rust side. Used by
    the orchestrator wrapper to detect a fixed point across Rust and
    Python passes.
    """
    return (
        int(delta.get("bounds_tightened", 0)) > 0
        or int(delta.get("aux_vars_introduced", 0)) > 0
        or int(delta.get("aux_constraints_introduced", 0)) > 0
        or len(delta.get("constraints_removed", []) or []) > 0
        or len(delta.get("constraints_rewritten", []) or []) > 0
        or len(delta.get("vars_fixed", []) or []) > 0
        or len(delta.get("vars_aggregated", []) or []) > 0
    )


@dataclass
class PresolveDelta:
    """Dataclass mirror of a Rust ``PresolveDelta``.

    Convenience for Python passes that prefer attribute access over
    dict keys; the orchestrator accepts either form via
    :func:`as_dict`.
    """

    pass_name: str
    pass_iter: int = 0
    bounds_tightened: int = 0
    aux_vars_introduced: int = 0
    aux_constraints_introduced: int = 0
    constraints_removed: list[int] = field(default_factory=list)
    constraints_rewritten: list[int] = field(default_factory=list)
    vars_fixed: list[tuple[int, float]] = field(default_factory=list)
    vars_aggregated: list[dict] = field(default_factory=list)
    work_units: int = 0
    wall_time_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "pass_name": self.pass_name,
            "pass_iter": self.pass_iter,
            "bounds_tightened": self.bounds_tightened,
            "aux_vars_introduced": self.aux_vars_introduced,
            "aux_constraints_introduced": self.aux_constraints_introduced,
            "constraints_removed": list(self.constraints_removed),
            "constraints_rewritten": list(self.constraints_rewritten),
            "vars_fixed": list(self.vars_fixed),
            "vars_aggregated": list(self.vars_aggregated),
            "work_units": self.work_units,
            "wall_time_ms": self.wall_time_ms,
        }


@dataclass
class PresolveResult:
    """Aggregate result from :func:`run_orchestrated_presolve`."""

    model_repr: Any
    deltas: list[dict]
    iterations: int
    terminated_by: str

    def __iter__(self):
        # Allow ``model, stats = run_orchestrated_presolve(...)``.
        yield self.model_repr
        yield {
            "iterations": self.iterations,
            "terminated_by": self.terminated_by,
            "deltas": self.deltas,
        }
