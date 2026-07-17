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

from .certificate import certify_convex, refresh_convex_mask
from .g_convex_cut import (
    GConvexCut,
    g_concave_overestimator_cut,
    g_convex_supporting_cut,
)
from .g_convexity import (
    GConvexCertificate,
    certify_g_convex,
    is_g_convex_pointwise,
    least_convexifying_rho,
)
from .g_products_ratios import (
    ProductRatioGClass,
    classify_product_ratio,
    is_product_or_ratio,
)
from .g_prop9 import transformation_adds_value
from .g_transform import (
    AffineOverestimator,
    ExpTransform,
    GTransform,
    least_convexifying_transform,
)
from .lattice import Curvature
from .log_lattice import (
    LogCurvature,
    classify_log_curvature,
    log_combine_product,
    log_combine_sum,
    log_negate,
    log_scale_pow,
)
from .rules import (
    OACutConvexity,
    classify_constraint,
    classify_expr,
    classify_model,
    classify_oa_cut_convexity,
)

__all__ = [
    "AffineOverestimator",
    "Curvature",
    "ExpTransform",
    "GConvexCertificate",
    "GConvexCut",
    "GTransform",
    "LogCurvature",
    "OACutConvexity",
    "ProductRatioGClass",
    "certify_convex",
    "certify_g_convex",
    "classify_constraint",
    "classify_expr",
    "classify_log_curvature",
    "classify_model",
    "classify_oa_cut_convexity",
    "classify_product_ratio",
    "g_concave_overestimator_cut",
    "g_convex_supporting_cut",
    "is_g_convex_pointwise",
    "is_product_or_ratio",
    "least_convexifying_rho",
    "least_convexifying_transform",
    "log_combine_product",
    "log_combine_sum",
    "log_negate",
    "log_scale_pow",
    "refresh_convex_mask",
    "transformation_adds_value",
]
