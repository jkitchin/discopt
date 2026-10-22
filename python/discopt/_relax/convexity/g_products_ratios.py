"""G-convexity of products & ratios of convex/concave functions (item 5).

KMS 2012 §5 shows that a broad class of *composite intermediates* — products
and ratios of convex and/or concave functions — is G-convex or G-concave with
explicit transforming functions, even though such products/ratios are
generally neither convex nor concave. This module scopes the general
augmented-Hessian detector (:mod:`g_convexity`) to that structural class,
giving a named §5 entry point plus the factor curvatures for context.

Design note. The paper derives *explicit* transforming functions per §5
sub-case; discopt already carries dedicated envelopes for the common shapes
(bilinear, reciprocal, linear-fractional, monomial joint hulls). Rather than duplicate
those or hand-transcribe a fragile per-case ``G`` table (a soundness risk),
the general detector *is* the sound §5 recognizer: its constant-``ρ`` witness
``G=exp(ρ·)`` convexifies exactly the G-convex products/ratios, verified
rigorously per box. This module identifies the §5 structure and delegates the
verdict, so callers get a scoped, sound classification without a bespoke
transform catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from discopt.modeling.core import BinaryOp, Expression, Model

from .g_convexity import GConvexCertificate, certify_g_convex
from .lattice import Curvature
from .rules import classify_expr


def is_product_or_ratio(expr: Expression) -> Optional[str]:
    """Return ``"product"``/``"ratio"`` if ``expr`` is a top-level ``*``/``/``.

    Returns ``None`` for any other node. This is the structural gate that
    scopes the §5 recognizer — the KMS §5 results are about products and
    ratios of factor functions, so only these node shapes are in scope.
    """
    if isinstance(expr, BinaryOp):
        if expr.op == "*":
            return "product"
        if expr.op == "/":
            return "ratio"
    return None


@dataclass(frozen=True)
class ProductRatioGClass:
    """Structural + G-convexity classification of a §5 product/ratio.

    * ``kind`` — ``"product"`` or ``"ratio"``.
    * ``left_curvature`` / ``right_curvature`` — the recursive-walker
      curvature of each factor (context for *why* the composite is a §5
      form; e.g. convex/concave).
    * ``certificate`` — the sound G-convexity verdict from the detector, or
      ``None`` if it abstains on the box.
    """

    kind: str
    left_curvature: Curvature
    right_curvature: Curvature
    certificate: Optional[GConvexCertificate]

    @property
    def is_g_convex(self) -> bool:
        return self.certificate is not None and self.certificate.kind == "g_convex"

    @property
    def is_g_concave(self) -> bool:
        return self.certificate is not None and self.certificate.kind == "g_concave"


def classify_product_ratio(
    expr: Expression,
    model: Model,
    box: Optional[dict] = None,
) -> Optional[ProductRatioGClass]:
    """Classify a §5 product/ratio intermediate, or ``None`` if not one.

    Returns ``None`` when ``expr`` is not a top-level product or ratio. When
    it is, returns a :class:`ProductRatioGClass` carrying the factor
    curvatures and the sound G-convexity certificate (which may itself be
    ``None`` if the detector abstains on the box — a conservative outcome,
    never a false claim).
    """
    kind = is_product_or_ratio(expr)
    if kind is None:
        return None
    assert isinstance(expr, BinaryOp)  # narrowed by is_product_or_ratio
    left_curv = classify_expr(expr.left, model)
    right_curv = classify_expr(expr.right, model)
    cert = certify_g_convex(expr, model, box=box)
    return ProductRatioGClass(
        kind=kind,
        left_curvature=left_curv,
        right_curvature=right_curv,
        certificate=cert,
    )


__all__ = [
    "ProductRatioGClass",
    "classify_product_ratio",
    "is_product_or_ratio",
]
