"""Transformation-based supporting cut for a G-convex intermediate (item 3).

Completes the smallest end-to-end slice of issue #181: detect (item 1) →
transform (item 2) → **valid linear cut** for the LP node relaxation.

Given a factorable intermediate ``φ`` certified G-convex on a box (a fresh
auxiliary variable ``t`` bounding it, ``φ(x) ≤ t``), this produces a valid
linear inequality in ``(x, t)`` that tightens the relaxation of that
intermediate set.

Construction (KMS 2012 §3)
--------------------------
With the constant-``ρ`` transform ``G(t)=exp(ρt)`` (item 2), the composite
``h(x)=exp(ρφ(x))`` is convex on the box. The transformation relaxation of
``{φ(x) ≤ t}`` is ``{h(x) ≤ Ḡ(t)}`` where ``Ḡ = conc_I G`` is the concave
(here affine/secant) overestimator of ``G`` over the range ``I=[φ_lo,φ_hi]``
of ``φ`` on the box. Because ``h`` is convex, any supporting hyperplane at a
point ``x₀`` underestimates it, giving the valid cut

    ``h(x₀) + ∇h(x₀)ᵀ(x − x₀)  ≤  Ḡ(t)``            (linear in ``x`` and ``t``)

with ``∇h(x₀)=ρ·h(x₀)·∇φ(x₀)``. Validity: for any feasible ``(x,t)``
(``φ(x) ≤ t``, ``x`` in box), the LHS ``≤ h(x) = exp(ρφ(x)) ≤ exp(ρt) =
G(t) ≤ Ḡ(t)``. For ``ρ=0`` it reduces exactly to the ordinary convex
tangent underestimator ``φ(x₀)+∇φ(x₀)ᵀ(x−x₀) ≤ t``.

Soundness posture
-----------------
This is a **default-off primitive**: it is *not* auto-injected into
``mccormick_lp.py`` or node bounding. A ``safety`` margin relaxes the cut's
RHS to absorb floating-point error in the tangent, and the cut is
numerically soundness-tested. Before any solver wiring (a bound-changing
change under the CLAUDE.md §5 graduation gate) the tangent must be made
interval-rigorous; that follow-up is called out in the issue. Until then the
routine emits cuts for study and validation only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from discopt.modeling.core import Expression, Model

from .g_convexity import GConvexCertificate, certify_g_convex
from .g_transform import ExpTransform
from .interval import Interval
from .interval_ad import interval_hessian


@dataclass(frozen=True)
class GConvexCut:
    """A valid linear cut ``x_coeffs·x + t_coeff·t ≤ rhs`` in ``(x, t)``.

    * ``x_coeffs`` — length-``n`` coefficients over the model's flat variable
      vector (``n = Σ var.size``).
    * ``t_coeff`` — coefficient on the auxiliary/epigraph variable ``t`` that
      bounds ``φ`` (``φ(x) ≤ t``).
    * ``rhs`` — right-hand side (already includes any safety relaxation).
    * ``rho`` — the transform witness (0 ⇒ ordinary convex tangent cut).
    * ``x0`` — the linearization point used.

    A point ``(x, t)`` with ``φ(x) ≤ t`` and ``x`` in the certifying box
    satisfies ``x_coeffs·x + t_coeff·t ≤ rhs``.
    """

    x_coeffs: np.ndarray
    t_coeff: float
    rhs: float
    rho: float
    x0: np.ndarray

    def violation(self, x: np.ndarray, t: float) -> float:
        """``x_coeffs·x + t_coeff·t − rhs`` — ``≤ 0`` means satisfied."""
        return float(
            np.dot(self.x_coeffs, np.asarray(x, dtype=float)) + self.t_coeff * t - self.rhs
        )


def _point_box(model: Model, x0: np.ndarray) -> dict:
    box: dict = {}
    off = 0
    for v in model._variables:
        sz = v.size
        shape = v.shape if v.shape else (1,)
        seg = np.asarray(x0[off : off + sz], dtype=np.float64).reshape(shape)
        box[v] = Interval(seg, seg)
        off += sz
    return box


def _flat_box_bounds(model: Model, box: Optional[dict]) -> tuple[np.ndarray, np.ndarray]:
    lbs, ubs = [], []
    for v in model._variables:
        if box is not None and v in box:
            lo = np.asarray(box[v].lo, dtype=float).ravel()
            hi = np.asarray(box[v].hi, dtype=float).ravel()
        else:
            lo = np.asarray(v.lb, dtype=float).ravel()
            hi = np.asarray(v.ub, dtype=float).ravel()
        lbs.append(lo)
        ubs.append(hi)
    return np.concatenate(lbs), np.concatenate(ubs)


def g_convex_supporting_cut(
    expr: Expression,
    model: Model,
    box: Optional[dict] = None,
    *,
    x0: Optional[np.ndarray] = None,
    cert: Optional[GConvexCertificate] = None,
    safety: float = 1e-9,
) -> Optional[GConvexCut]:
    """Build a valid transformation cut for the intermediate ``φ(x) ≤ t``.

    Args:
        expr: The scalar intermediate ``φ``.
        model: Model defining the variable layout.
        box: Optional tightened ``{Variable: Interval}`` box; defaults to the
            declared variable bounds.
        x0: Linearization point (flat vector). Defaults to the box midpoint.
        cert: A precomputed :class:`GConvexCertificate`; if omitted the
            detector is run on the box. The cut is emitted only for a
            ``g_convex`` verdict (a ``g_concave`` intermediate would need the
            negated construction / an overestimator, not built here).
        safety: Nonnegative RHS relaxation absorbing tangent round-off, as an
            absolute + relative (to the RHS magnitude) margin.

    Returns:
        A :class:`GConvexCut`, or ``None`` when ``φ`` is not certified
        ``g_convex`` on the box or the enclosure is unusable.
    """
    if cert is None:
        cert = certify_g_convex(expr, model, box=box)
    if cert is None or cert.kind != "g_convex":
        return None
    rho = float(cert.rho)

    lb, ub = _flat_box_bounds(model, box)
    n = lb.shape[0]
    if x0 is None:
        x0 = 0.5 * (lb + ub)
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    if x0.shape[0] != n:
        raise ValueError(f"x0 length {x0.shape[0]} != n_vars {n}")
    x0 = np.clip(x0, lb, ub)

    # Value + gradient of φ at x0 (degenerate point box → tight enclosure;
    # midpoint is the float evaluation).
    pt = _point_box(model, x0)
    try:
        ad0 = interval_hessian(expr, model, box=pt)
    except ValueError:
        return None
    phi0 = float(ad0.value.mid)
    grad0 = np.asarray(ad0.grad.mid, dtype=np.float64).ravel()
    if not (np.isfinite(phi0) and np.all(np.isfinite(grad0))):
        return None

    # Range I = [φ_lo, φ_hi] of φ over the whole box (sound enclosure).
    try:
        ad_box = interval_hessian(expr, model, box=box)
    except ValueError:
        return None
    t_lo = float(ad_box.value.lo)
    t_hi = float(ad_box.value.hi)
    if not (np.isfinite(t_lo) and np.isfinite(t_hi)) or t_hi < t_lo:
        return None

    transform = ExpTransform(rho=rho)
    sec = transform.concave_overestimator(t_lo, t_hi)  # Ḡ(t) = a + b·t

    # h(x0), ∇h(x0) = ρ·h·∇φ ; for ρ=0, h=φ and ∇h=∇φ (identity transform).
    if rho == 0.0:
        h0 = phi0
        grad_h0 = grad0
    else:
        h0 = float(np.exp(rho * phi0))
        grad_h0 = rho * h0 * grad0

    # Cut: grad_h0·x − b·t ≤ a − h0 + grad_h0·x0.
    x_coeffs = grad_h0
    t_coeff = -sec.b
    rhs = sec.a - h0 + float(np.dot(grad_h0, x0))
    if not np.all(np.isfinite(x_coeffs)) or not np.isfinite(rhs):
        return None
    # Relax RHS outward by a safety margin (absolute + relative) so tangent
    # round-off cannot make the cut shave a feasible point.
    rhs = rhs + safety * (1.0 + abs(rhs))

    return GConvexCut(
        x_coeffs=np.asarray(x_coeffs, dtype=np.float64),
        t_coeff=float(t_coeff),
        rhs=float(rhs),
        rho=rho,
        x0=x0,
    )


__all__ = ["GConvexCut", "g_convex_supporting_cut"]
