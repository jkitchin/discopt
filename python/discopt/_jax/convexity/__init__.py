"""Convexity detection for expression DAGs.

Classifies each (sub)expression by its curvature (CONVEX / CONCAVE /
AFFINE / UNKNOWN) using sound composition rules from disciplined convex
programming. Soundness invariant: a CONVEX or CONCAVE verdict is a
mathematical proof — heuristic / sampling-based methods that could
produce false positives are not used here.

Public API is intentionally small:
    classify_expr(expr, model=None) -> Curvature
    classify_constraint(constraint, model=None) -> bool
    classify_model(model) -> (is_convex, per_constraint_mask)
    Curvature  (enum)

References
----------
Grant, Boyd, Ye (2006), "Disciplined Convex Programming," in
  Global Optimization: From Theory to Implementation.
Ceccon, Siirola, Misener (2020), "SUSPECT: MINLP special structure
  detector for Pyomo," TOP.
"""

from __future__ import annotations

from .lattice import Curvature
from .rules import classify_constraint, classify_expr, classify_model

__all__ = [
    "Curvature",
    "classify_constraint",
    "classify_expr",
    "classify_model",
]
