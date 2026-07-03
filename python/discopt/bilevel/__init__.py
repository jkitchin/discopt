"""Bilevel optimization for discopt.

A bilevel (Stackelberg) program optimizes a leader's objective subject to a
follower solving its own optimization problem. For a lower level that is convex
in the follower's variables, the follower is replaced by its KKT conditions,
collapsing the two levels into a single-level MPEC that discopt already solves
(:mod:`discopt.mpec`).

See ``docs/dev/bilevel-module-plan.md`` for the full design. Phase 0 ships the
:mod:`~discopt.bilevel.symbolic_diff` engine — the symbolic differentiator that
lets the KKT stationarity condition ``∇_y L == 0`` be emitted as ordinary model
constraints, keeping the reformulation inside the certified global path.
"""

from __future__ import annotations

from discopt.bilevel.problem import BilevelProblem
from discopt.bilevel.symbolic_diff import diff, grad

__all__ = ["BilevelProblem", "diff", "grad"]
