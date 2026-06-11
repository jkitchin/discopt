"""Ellipsoidal arithmetic for rigorous range enclosure (M7 of issue #81).

Axis-aligned interval arithmetic loses the *affine correlations* between
intermediate quantities: after ``a = x + y`` and ``b = x - y`` it forgets that
``a + b = 2x``, so ``a + b`` re-inflates to the sum of the two intervals.
Ellipsoidal arithmetic keeps those correlations by enclosing a vector of
quantities in an ellipsoid

    E(c, Q) = { c + Q^{1/2} u : ||u||_2 <= 1 }
            = { x : (x - c)^T Q^{-1} (x - c) <= 1 }   (Q positive definite),

whose shape matrix ``Q`` carries the off-diagonal correlation terms an interval
box cannot represent.

The implementation follows the **affine-form** (noise-symbol) view of
ellipsoidal calculus used by Houska, Villanueva and Chachuat: every scalar
quantity is tracked as

    v = center + gens . xi + [-rem, +rem],     ||xi||_2 <= 1,

where ``xi`` is a *shared* vector of input noise symbols constrained to the unit
2-ball (so affine combinations across quantities reuse the same ``xi`` and stay
correlated), ``gens`` are the linear generator coefficients, and ``rem`` is an
independent interval remainder radius accumulating the rigorously-bounded
nonlinear defects. The enclosure of a single scalar is therefore the interval

    [ center - (||gens||_2 + rem),  center + (||gens||_2 + rem) ],

and the support of a linear functional ``s`` over a vector of forms sharing the
same ``xi`` is

    s . c  +/-  ( sqrt(s^T A A^T s)  +  |s| . rem ) ,

i.e. the ellipsoidal support ``s^T c + sqrt(s^T Q s)`` (with ``Q = A A^T``) plus
the box contribution of the remainders. The ``sqrt`` (2-norm) of the generator
contribution is what beats interval arithmetic's 1-norm whenever two or more
noise symbols combine with cancellation.

* **Affine operators** (``+``, ``-``, scalar ``*``/``+``) act in closed form on
  ``(center, gens, rem)`` and are *exact* on the generator part — no correlation
  is lost.
* **Bilinear products** and **unary nonlinear** operators (``exp``, ``log``,
  ``sqrt``, ``sin``, ...) linearise around the centre and fold the rigorous
  second-order-and-higher defect into ``rem``. The unary defect is bounded with
  the Chebyshev-model kernel (:mod:`chebyshev_model`, M2 of #51), reusing the
  same Taylor-remainder machinery as the polynomial relaxations.

All interval/remainder/support results are *outward rounded* by a small relative
factor so floating-point error can only loosen an enclosure, never invalidate
it. This preserves discopt's rigorous-bound invariant (a computed lower bound
never exceeds the true global minimum).

References:
- Villanueva, Rajyaguru, Houska, Chachuat, *Comput. Aided Chem. Eng.* 37,
  767-772, 2015.
- Villanueva, *Set-theoretic methods for analysis and control of dynamic
  systems*, PhD thesis, Imperial College London, 2016.
- Houska, Villanueva, Chachuat, *A validated integration algorithm for
  nonlinear ODEs using Taylor models and ellipsoidal calculus*, CDC 2013.

See issue #81 (M7) and #51 for the convexification roadmap context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from discopt._jax import chebyshev_model as cm

# Outward-rounding factor: every enclosure radius / support value is inflated by
# this relative amount (plus an absolute floor) so rounding error only loosens.
_ROUND_SAFETY = 1e-12
_ROUND_FLOOR = 1e-15

# Eigenvalue floor used when regularising a shape matrix back to PSD.
_PSD_FLOOR = 0.0


def _outward(radius: float) -> float:
    """Inflate a non-negative radius outward so rounding error cannot shave it."""
    r = float(radius)
    return r + _ROUND_SAFETY * abs(r) + _ROUND_FLOOR


# ---------------------------------------------------------------------------
# Ellipsoid (c, Q) — the geometric object and its closed-form calculus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Ellipsoid:
    """An ellipsoid ``E(c, Q) = {c + Q^{1/2} u : ||u|| <= 1}`` in ``R^n``.

    ``Q`` is symmetric positive *semi*-definite (a degenerate ellipsoid — a
    lower-dimensional slab or segment — is allowed and represented by a rank
    deficient ``Q``).
    """

    center: np.ndarray
    shape: np.ndarray

    def __post_init__(self) -> None:
        c = np.asarray(self.center, dtype=np.float64).reshape(-1)
        q = np.asarray(self.shape, dtype=np.float64)
        if q.shape != (c.size, c.size):
            raise ValueError(f"shape must be {(c.size, c.size)}, got {q.shape}")
        object.__setattr__(self, "center", c)
        object.__setattr__(self, "shape", 0.5 * (q + q.T))  # symmetrise

    @property
    def dim(self) -> int:
        return self.center.size

    def support(self, s: np.ndarray) -> tuple[float, float]:
        """Range ``[min, max]`` of the linear functional ``s . x`` over ``E``.

        Exactly ``s^T c +/- sqrt(s^T Q s)`` (outward rounded). This is the
        closed-form support function that drops into the polyhedral cut path.
        """
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        mid = float(s @ self.center)
        quad = float(s @ self.shape @ s)
        rad = _outward(np.sqrt(max(quad, 0.0)))
        return (mid - rad, mid + rad)

    def coordinate_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Tightest axis-aligned box ``[lb, ub]`` containing ``E``."""
        rad = np.sqrt(np.maximum(np.diag(self.shape), 0.0))
        rad = np.array([_outward(r) for r in rad])
        return self.center - rad, self.center + rad

    def affine_image(self, a: np.ndarray, b: np.ndarray | None = None) -> "Ellipsoid":
        """The exact image ``A . E + b = E(A c + b, A Q A^T)``."""
        a = np.asarray(a, dtype=np.float64)
        if a.ndim != 2:
            raise ValueError("A must be a 2-D matrix")
        c = a @ self.center
        if b is not None:
            c = c + np.asarray(b, dtype=np.float64).reshape(-1)
        q = a @ self.shape @ a.T
        return Ellipsoid(c, q)

    def minkowski_sum(self, other: "Ellipsoid") -> "Ellipsoid":
        """Trace-minimising ellipsoidal outer approximation of ``E1 ⊕ E2``.

        The Minkowski sum of two ellipsoids is not an ellipsoid; the family

            E(c1 + c2, (1 + 1/k) Q1 + (1 + k) Q2),   k > 0,

        outer-bounds it for every ``k``, and ``k = sqrt(tr Q2 / tr Q1)``
        minimises the trace (sum of squared semi-axes) of the result.
        """
        if other.dim != self.dim:
            raise ValueError("dimension mismatch in Minkowski sum")
        t1 = float(np.trace(self.shape))
        t2 = float(np.trace(other.shape))
        if t1 <= 0.0:
            return Ellipsoid(self.center + other.center, other.shape)
        if t2 <= 0.0:
            return Ellipsoid(self.center + other.center, self.shape)
        k = np.sqrt(t2 / t1)
        q = (1.0 + 1.0 / k) * self.shape + (1.0 + k) * other.shape
        return Ellipsoid(self.center + other.center, q)

    def contains(self, x: np.ndarray, tol: float = 1e-9) -> bool:
        """True if point ``x`` lies in ``E`` (within ``tol``)."""
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        d = x - self.center
        # Use a pseudo-inverse so degenerate ellipsoids are handled: a point off
        # the supporting subspace is reported outside.
        q_pinv = np.linalg.pinv(self.shape, rcond=1e-12)
        proj = self.shape @ q_pinv @ d
        if np.linalg.norm(proj - d) > tol * (1.0 + np.linalg.norm(d)):
            return False
        return float(d @ q_pinv @ d) <= 1.0 + tol

    def regularize(self, floor: float = _PSD_FLOOR) -> "Ellipsoid":
        """Project the shape matrix back to PSD (clamp eigenvalues at ``floor``)."""
        w, v = np.linalg.eigh(self.shape)
        w = np.maximum(w, floor)
        return Ellipsoid(self.center, (v * w) @ v.T)


def bounding_ellipsoid_of_box(lb: np.ndarray, ub: np.ndarray) -> Ellipsoid:
    """Smallest-volume ellipsoid enclosing the axis-aligned box ``[lb, ub]``.

    For a box of half-widths ``r`` in ``R^n`` the Loewner-John enclosing
    ellipsoid is ``E(midpoint, n * diag(r^2))`` (its boundary passes through the
    box corners). Note the dimension-``n`` inflation: ellipsoidal arithmetic
    only pays off when the inputs carry genuine affine correlation, not for an
    axis-aligned box of independent ranges.
    """
    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)
    c = 0.5 * (lb + ub)
    r = 0.5 * (ub - lb)
    n = c.size
    return Ellipsoid(c, float(n) * np.diag(r * r))


# ---------------------------------------------------------------------------
# EllipsoidalForm — scalar affine-form arithmetic over a shared input ball
# ---------------------------------------------------------------------------


@dataclass
class EllipsoidalForm:
    """A scalar quantity ``center + gens . xi + [-rem, rem]``, ``||xi||_2 <= 1``.

    ``gens`` are coefficients on a *shared* input noise vector, so two forms
    built from the same inputs stay correlated under affine combination. ``rem``
    is an independent, outward-rounded interval radius collecting nonlinear
    defects.
    """

    center: float
    gens: np.ndarray
    rem: float = 0.0

    def __post_init__(self) -> None:
        self.center = float(self.center)
        self.gens = np.asarray(self.gens, dtype=np.float64).reshape(-1)
        self.rem = float(self.rem)

    # -- enclosures ---------------------------------------------------------
    def ellipsoidal_radius(self) -> float:
        """Half-width of the rigorous interval enclosure (2-norm of generators)."""
        return _outward(float(np.linalg.norm(self.gens)) + self.rem)

    def interval_radius(self) -> float:
        """Half-width an interval/affine-arithmetic enclosure would give (1-norm)."""
        return _outward(float(np.sum(np.abs(self.gens))) + self.rem)

    def bounds(self) -> tuple[float, float]:
        """Rigorous ``[lo, hi]`` enclosure of the quantity."""
        r = self.ellipsoidal_radius()
        return (self.center - r, self.center + r)

    def interval_bounds(self) -> tuple[float, float]:
        """The looser interval-arithmetic enclosure (for comparison/tests)."""
        r = self.interval_radius()
        return (self.center - r, self.center + r)

    # -- affine operators (exact on the generator part) ---------------------
    def __add__(self, other: "EllipsoidalForm | float") -> "EllipsoidalForm":
        if isinstance(other, EllipsoidalForm):
            return EllipsoidalForm(
                self.center + other.center, self.gens + other.gens, self.rem + other.rem
            )
        return EllipsoidalForm(self.center + float(other), self.gens, self.rem)

    __radd__ = __add__

    def __neg__(self) -> "EllipsoidalForm":
        return EllipsoidalForm(-self.center, -self.gens, self.rem)

    def __sub__(self, other: "EllipsoidalForm | float") -> "EllipsoidalForm":
        if isinstance(other, EllipsoidalForm):
            return EllipsoidalForm(
                self.center - other.center, self.gens - other.gens, self.rem + other.rem
            )
        return EllipsoidalForm(self.center - float(other), self.gens, self.rem)

    def __rsub__(self, other: float) -> "EllipsoidalForm":
        return EllipsoidalForm(float(other) - self.center, -self.gens, self.rem)

    def scaled(self, a: float) -> "EllipsoidalForm":
        a = float(a)
        return EllipsoidalForm(a * self.center, a * self.gens, abs(a) * self.rem)

    # -- nonlinear operators ------------------------------------------------
    def __mul__(self, other: "EllipsoidalForm | float") -> "EllipsoidalForm":
        if not isinstance(other, EllipsoidalForm):
            return self.scaled(other)
        cu, au, ru = self.center, self.gens, self.rem
        cv, av, rv = other.center, other.gens, other.rem
        nau = float(np.linalg.norm(au))
        nav = float(np.linalg.norm(av))
        center = cu * cv
        gens = cu * av + cv * au
        # All second-order-and-cross terms folded into the remainder, bounded by
        # Cauchy-Schwarz over the unit 2-ball: |(au.xi)(av.xi)| <= ||au|| ||av||.
        rem = nau * nav + abs(cu) * rv + abs(cv) * ru + nau * rv + nav * ru + ru * rv
        return EllipsoidalForm(center, gens, _outward(rem))

    __rmul__ = __mul__

    def apply_unary(self, f: Callable, degree: int = 8) -> "EllipsoidalForm":
        """Return ``f(self)`` as an ellipsoidal form with rigorous remainder.

        Linearises ``f`` around the centre; the rigorous bound on the
        linearisation defect over the form's interval enclosure comes from the
        Chebyshev-model kernel and is folded into ``rem``.
        """
        cu = self.center
        radius = self.ellipsoidal_radius()
        if radius < 1e-15:
            return EllipsoidalForm(float(f(jnp.asarray(cu))), np.zeros_like(self.gens), 0.0)
        box = (cu - radius, cu + radius)
        fc = float(f(jnp.asarray(cu, dtype=jnp.float64)))
        fpc = float(jax.grad(lambda t: f(t))(jnp.asarray(cu, dtype=jnp.float64)))

        x = cm.from_variable(box, degree)
        fm = cm.compose_unary(f, x)
        lin = cm.scalar_add(fc, cm.scalar_mul(fpc, cm.scalar_add(-cu, x)))
        defect_lo, defect_hi = cm.sub(fm, lin).bounds()

        center = fc + 0.5 * (defect_lo + defect_hi)
        gens = fpc * self.gens
        rem = abs(fpc) * self.rem + 0.5 * (defect_hi - defect_lo)
        return EllipsoidalForm(center, gens, _outward(rem))


# ---------------------------------------------------------------------------
# Seeding forms from an input ellipsoid / box, and re-assembling enclosures
# ---------------------------------------------------------------------------


def forms_from_ellipsoid(ellipsoid: Ellipsoid) -> list[EllipsoidalForm]:
    """One :class:`EllipsoidalForm` per coordinate of an input ``E(c, Q)``.

    Factoring ``Q = G G^T`` gives ``x_i = c_i + (G xi)_i`` with ``||xi|| <= 1``,
    so the generator row of coordinate ``i`` is ``G[i]`` and the forms share the
    common noise vector ``xi`` — exactly the correlation an input ellipsoid
    encodes.
    """
    c = ellipsoid.center
    g = _psd_sqrt(ellipsoid.shape)
    return [EllipsoidalForm(float(c[i]), g[i].copy(), 0.0) for i in range(ellipsoid.dim)]


def forms_from_box(lb: np.ndarray, ub: np.ndarray) -> list[EllipsoidalForm]:
    """Independent-coordinate input forms for an axis-aligned box ``[lb, ub]``.

    Each coordinate gets its own noise symbol (diagonal generator matrix), so a
    single coordinate's enclosure is exact and there is no spurious correlation.
    """
    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)
    n = lb.size
    c = 0.5 * (lb + ub)
    r = 0.5 * (ub - lb)
    forms = []
    for i in range(n):
        row = np.zeros(n)
        row[i] = r[i]
        forms.append(EllipsoidalForm(float(c[i]), row, 0.0))
    return forms


def joint_ellipsoid(forms: list[EllipsoidalForm]) -> Ellipsoid:
    """Assemble the joint ellipsoidal enclosure ``E(c, A A^T) ⊕ box(rem)``.

    The generator rows of the forms (padded to a common width) form ``A``; the
    independent remainders contribute an axis-aligned box that is Minkowski-
    summed in as its Loewner-John enclosing ellipsoid.
    """
    if not forms:
        raise ValueError("need at least one form")
    width = max(f.gens.size for f in forms)
    a = np.zeros((len(forms), width))
    c = np.zeros(len(forms))
    rem = np.zeros(len(forms))
    for i, f in enumerate(forms):
        a[i, : f.gens.size] = f.gens
        c[i] = f.center
        rem[i] = f.rem
    ell = Ellipsoid(c, a @ a.T)
    if np.any(rem > 0.0):
        ell = ell.minkowski_sum(bounding_ellipsoid_of_box(-rem, rem))
    return ell


def _psd_sqrt(q: np.ndarray) -> np.ndarray:
    """A real symmetric square root ``G`` with ``G G^T = Q`` for PSD ``Q``."""
    w, v = np.linalg.eigh(0.5 * (q + q.T))
    w = np.maximum(w, 0.0)
    return np.asarray((v * np.sqrt(w)) @ v.T, dtype=np.float64)


__all__ = [
    "Ellipsoid",
    "EllipsoidalForm",
    "bounding_ellipsoid_of_box",
    "forms_from_box",
    "forms_from_ellipsoid",
    "joint_ellipsoid",
]
