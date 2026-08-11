"""Least convexifying transformation ``G*`` for constant-``ρ`` G-convexity.

Companion to :mod:`g_convexity` (issue #181, item 2). The detector there
certifies a factorable intermediate ``φ`` G-convex on a box by exhibiting a
constant ``ρ ≥ 0`` with ``∇²φ + ρ∇φ∇φᵀ ⪰ 0``. This module turns that
witness into the concrete monotone transformation that convexifies ``φ`` —
the object the transformation relaxation (item 3) linearizes.

The transform for a constant ``ρ``
----------------------------------
The least convexifying transform ``G*`` is the increasing univariate ``G``
with ``G''/G' = ρ`` (the augmented-Hessian relation ``ρ = G''/G'``; KMS
2012 §2), unique up to an increasing affine map. Solving the ODE:

    ``ρ > 0``:  ``G(t) = exp(ρ t)``      (strictly convex, strictly increasing)
    ``ρ = 0``:  ``G(t) = t``             (affine — ``φ`` already convex)

Correctness of the ``exp`` family is direct: with ``h = exp(ρφ)``,

    ``∇²h = ρ·exp(ρφ)·(∇²φ + ρ∇φ∇φᵀ) ⪰ 0``   (since ``ρ ≥ 0``, ``exp > 0``),

so ``exp(ρφ)`` is **convex** on exactly the box where the detector certified
the augmented Hessian PSD. That convex composite is what item 3 supports with
a hyperplane.

Composition calculus (KMS 2012 §2/3)
------------------------------------
Two rules the paper proves for ``G*`` propagation are exposed:

* **Affine inner maps preserve ``G*``** — ``φ(Ax+b)`` has the same ``G*`` as
  ``φ`` (the box/point moves, ``ρ`` does not). :func:`compose_affine_inner`.
* **Increasing outer ``f``** — if ``φ`` is ``G*``-convex then ``f(φ)`` is
  least-convexified by ``G* ∘ f⁻¹`` for increasing ``f``.
  :func:`compose_increasing_outer` returns that (generally non-``exp``)
  transform as a callable ``GTransform``.

This module is analytical/representational only — it evaluates and validates
transforms; it does not touch node bounding. It stays sound by construction:
every routine either returns a mathematically exact transform or refuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .g_convexity import GConvexCertificate


@dataclass(frozen=True)
class AffineOverestimator:
    """The affine (secant) concave overestimator ``Ḡ(t) = a + b·t`` of a
    convex ``G`` over an interval ``[t_lo, t_hi]``.

    The concave overestimator of a *convex* function on an interval is its
    chord (Boyd & Vandenberghe §3.1.3); being affine it is both the tightest
    concave overestimator ``conc_I G`` and trivially linear — exactly the
    form the transformation relaxation needs so the relaxed set stays convex
    (``convex(x) ≤ affine(t)``).
    """

    a: float  # intercept
    b: float  # slope
    t_lo: float
    t_hi: float

    def __call__(self, t: float) -> float:
        return self.a + self.b * t

    def inverse(self, s: float) -> float:
        """``Ḡ⁻¹(s)`` — defined when the chord is non-degenerate (``b≠0``)."""
        if self.b == 0.0:
            raise ZeroDivisionError("degenerate (constant) overestimator has no inverse")
        return (s - self.a) / self.b


class GTransform:
    """A monotone convexifying transform ``G`` with the operations item 3 needs.

    Concrete callable form used both by the analytic ``exp`` family
    (:class:`ExpTransform`) and by composed transforms
    (:func:`compose_increasing_outer`). ``apply``/``deriv``/``inverse`` are
    the transform and its calculus; ``concave_overestimator`` builds the
    secant ``conc_I G`` an outer relaxation replaces ``G(t)`` with.
    """

    def apply(self, t: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def deriv(self, t: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def inverse(self, s: float) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def is_increasing(self) -> bool:  # pragma: no cover - interface
        return True

    @property
    def is_convex(self) -> bool:  # pragma: no cover - interface
        return True

    def concave_overestimator(self, t_lo: float, t_hi: float) -> AffineOverestimator:
        """Secant (chord) concave overestimator of ``G`` over ``[t_lo, t_hi]``.

        For a convex increasing ``G`` the chord lies above ``G`` on the whole
        interval, so ``a + b·t ≥ G(t)`` there — a valid concave (affine)
        overestimator. Degenerate ``t_lo == t_hi`` yields the point value
        with zero slope.
        """
        if not (np.isfinite(t_lo) and np.isfinite(t_hi)) or t_hi < t_lo:
            raise ValueError(f"invalid interval [{t_lo}, {t_hi}]")
        g_lo = self.apply(t_lo)
        if t_hi == t_lo:
            return AffineOverestimator(a=g_lo, b=0.0, t_lo=t_lo, t_hi=t_hi)
        g_hi = self.apply(t_hi)
        b = (g_hi - g_lo) / (t_hi - t_lo)
        a = g_lo - b * t_lo
        return AffineOverestimator(a=a, b=b, t_lo=t_lo, t_hi=t_hi)


@dataclass(frozen=True)
class ExpTransform(GTransform):
    """The constant-``ρ`` least convexifying transform ``G(t)=exp(ρt)``.

    ``rho == 0`` degenerates to the identity ``G(t)=t`` (``φ`` already
    convex — no transformation needed). ``rho > 0`` is strictly convex and
    strictly increasing, so it convexifies a merely-G-convex ``φ``.
    """

    rho: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.rho) or self.rho < 0.0:
            raise ValueError(f"rho must be finite and >= 0, got {self.rho}")

    def apply(self, t: float) -> float:
        if self.rho == 0.0:
            return float(t)
        return float(np.exp(self.rho * t))

    def deriv(self, t: float) -> float:
        if self.rho == 0.0:
            return 1.0
        return float(self.rho * np.exp(self.rho * t))

    def inverse(self, s: float) -> float:
        if self.rho == 0.0:
            return float(s)
        if s <= 0.0:
            raise ValueError("exp transform inverse requires s > 0")
        return float(np.log(s) / self.rho)

    @property
    def is_increasing(self) -> bool:
        return True

    @property
    def is_convex(self) -> bool:
        return True


def least_convexifying_transform(cert: GConvexCertificate) -> ExpTransform:
    """The ``ExpTransform`` witnessing a :class:`GConvexCertificate`.

    For a ``g_convex`` verdict this convexifies ``φ`` directly. For a
    ``g_concave`` verdict the same ``ρ`` convexifies ``-φ`` — callers relax
    the negated intermediate — so the returned transform's ``ρ`` is the
    witness in both cases.
    """
    return ExpTransform(rho=float(cert.rho))


def compose_affine_inner(transform: ExpTransform) -> ExpTransform:
    """``G*`` of ``φ(Ax+b)`` given ``G*`` of ``φ`` — unchanged (KMS 2012).

    Affine inner maps do not change the least convexifying transform: the
    argument box moves but ``ρ = G''/G'`` is a property of the outer shape of
    ``φ``, not the coordinates. Provided for explicitness in the composition
    calculus; it is the identity on the transform.
    """
    return transform


def compose_increasing_outer(
    transform: GTransform,
    f_inverse: Callable[[float], float],
    f_inverse_deriv: Optional[Callable[[float], float]] = None,
) -> GTransform:
    """``G* ∘ f⁻¹`` — the least convexifier of ``f(φ)`` for increasing ``f``.

    If ``φ`` is ``transform``-convex and ``f`` is increasing, then ``f(φ)``
    is least-convexified by ``G* ∘ f⁻¹`` (KMS 2012 §2). The result is a
    general :class:`GTransform` (no longer an ``exp`` in general), with
    ``apply(s) = G*(f⁻¹(s))`` and, when ``f_inverse_deriv`` is supplied,
    ``deriv(s) = G*'(f⁻¹(s))·(f⁻¹)'(s)`` by the chain rule. ``inverse`` is
    ``f(G*⁻¹(·))`` — available only if ``f`` (the forward map) is recovered,
    so here we expose it through ``transform.inverse`` composed with the
    caller-supplied forward map lazily; callers that only linearize
    ``apply``/``deriv`` (item 3) never need it.
    """
    base = transform

    class _Composed(GTransform):
        def apply(self, s: float) -> float:
            return base.apply(f_inverse(s))

        def deriv(self, s: float) -> float:
            if f_inverse_deriv is None:
                raise NotImplementedError("deriv requires f_inverse_deriv")
            return base.deriv(f_inverse(s)) * f_inverse_deriv(s)

        def inverse(self, s: float) -> float:  # pragma: no cover - unused in item 3
            raise NotImplementedError("composed inverse not represented")

        @property
        def is_increasing(self) -> bool:
            return base.is_increasing

        @property
        def is_convex(self) -> bool:
            # G*∘f⁻¹ need not be convex for a general increasing f; the paper
            # only guarantees it is the *least convexifying* transform of the
            # composite, not that it is itself convex. Report unknown-safe.
            return False

    return _Composed()


__all__ = [
    "AffineOverestimator",
    "ExpTransform",
    "GTransform",
    "compose_affine_inner",
    "compose_increasing_outer",
    "least_convexifying_transform",
]
