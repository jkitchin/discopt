"""Sound interval arithmetic with outward rounding.

This module implements the interval arithmetic primitives used by the
box-local convexity certificate. The design goal is **soundness**: an
:class:`Interval` ``[lo, hi]`` returned by any operation must enclose
every possible result of the true computation on any point in its
arguments' intervals, *including* all floating-point roundoff errors.

Soundness is achieved by outward rounding after every arithmetic
operation:

* Lower endpoints are pushed toward ``-inf`` with ``np.nextafter``.
* Upper endpoints are pushed toward ``+inf`` with ``np.nextafter``.

This loses at most one ULP per operation and composes safely — the
resulting enclosure is always conservative. The implementation stays
in plain numpy (no JAX): interval arithmetic does not flow through a
differentiable tracer, and the per-constraint work is small enough
that the JAX overhead is not warranted.

References
----------
Moore, R. E. (1966). *Interval Analysis*. Prentice-Hall.
Neumaier, A. (1990). *Interval Methods for Systems of Equations*.
Cambridge University Press.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

Number = Union[int, float, np.ndarray]


# ──────────────────────────────────────────────────────────────────────
# Outward rounding helpers
# ──────────────────────────────────────────────────────────────────────


def _round_down(x: Number) -> np.ndarray:
    """Push ``x`` one ULP toward ``-inf`` (sound lower bound).

    Accepts a scalar or ndarray; returns the matching shape. Zero and
    finite values are nudged toward -inf; infinities are left alone
    (``nextafter(-inf, -inf) = -inf``).
    """
    return np.nextafter(x, np.float64(-np.inf))


def _round_up(x: Number) -> np.ndarray:
    """Push ``x`` one ULP toward ``+inf`` (sound upper bound)."""
    return np.nextafter(x, np.float64(np.inf))


# ──────────────────────────────────────────────────────────────────────
# Interval
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Interval:
    """Closed real interval ``[lo, hi]`` with numpy-array endpoints.

    Endpoints broadcast against each other and support element-wise
    operations. A degenerate interval ``lo == hi`` is allowed and
    represents a (floating-point) point value. Scalars are accepted
    and normalised to 0-d arrays in ``__post_init__``.
    """

    # ``__post_init__`` normalises both endpoints to contiguous
    # ``float64`` ndarrays. The runtime invariant is that both fields
    # are 0-d or higher ndarrays after construction; the annotation
    # is kept generic so callers can pass Python scalars or 0-d numpy
    # scalars without tripping strict type-checking.
    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self) -> None:
        # Frozen dataclass: use object.__setattr__ to normalize inputs.
        lo = np.asarray(self.lo, dtype=np.float64)
        hi = np.asarray(self.hi, dtype=np.float64)
        # Broadcast to a common shape once so downstream ops can assume
        # shape agreement without re-broadcasting every time.
        lo, hi = np.broadcast_arrays(lo, hi)
        if np.any(lo > hi):
            raise ValueError(f"Interval lo > hi: lo={lo}, hi={hi}")
        object.__setattr__(self, "lo", np.ascontiguousarray(lo))
        object.__setattr__(self, "hi", np.ascontiguousarray(hi))

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def point(cls, x: Number) -> "Interval":
        """Degenerate interval ``[x, x]`` (exact floating-point point)."""
        arr = np.asarray(x, dtype=np.float64)
        return cls(arr, arr)

    @classmethod
    def from_bounds(cls, lb: Number, ub: Number) -> "Interval":
        """Construct an interval from separate lower and upper bounds."""
        return cls(np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64))

    # ------------------------------------------------------------------
    # Basic queries
    # ------------------------------------------------------------------

    @property
    def width(self) -> np.ndarray:
        """Element-wise width ``hi - lo``."""
        return _round_up(self.hi - self.lo)

    @property
    def mid(self) -> np.ndarray:
        """Element-wise midpoint (no rounding guarantee — diagnostic only)."""
        return 0.5 * (self.lo + self.hi)

    def contains_zero(self) -> np.ndarray:
        """Element-wise test for ``0 ∈ [lo, hi]``."""
        return (self.lo <= 0) & (self.hi >= 0)

    # ------------------------------------------------------------------
    # Arithmetic — always with outward rounding
    # ------------------------------------------------------------------

    def __neg__(self) -> "Interval":
        return Interval(-self.hi, -self.lo)

    def __add__(self, other: Union["Interval", Number]) -> "Interval":
        other = _as_interval(other)
        return Interval(_round_down(self.lo + other.lo), _round_up(self.hi + other.hi))

    __radd__ = __add__

    def __sub__(self, other: Union["Interval", Number]) -> "Interval":
        other = _as_interval(other)
        return Interval(_round_down(self.lo - other.hi), _round_up(self.hi - other.lo))

    def __rsub__(self, other: Union["Interval", Number]) -> "Interval":
        return _as_interval(other) - self

    def __mul__(self, other: Union["Interval", Number]) -> "Interval":
        other = _as_interval(other)
        # The four corner products enclose the result on any sign
        # combination; pick element-wise min/max to handle all cases.
        a = self.lo * other.lo
        b = self.lo * other.hi
        c = self.hi * other.lo
        d = self.hi * other.hi
        lo = np.minimum(np.minimum(a, b), np.minimum(c, d))
        hi = np.maximum(np.maximum(a, b), np.maximum(c, d))
        return Interval(_round_down(lo), _round_up(hi))

    __rmul__ = __mul__

    def __truediv__(self, other: Union["Interval", Number]) -> "Interval":
        other = _as_interval(other)
        # Division is only well-defined when 0 is not in the
        # denominator. When it is, return an unbounded enclosure so
        # soundness is preserved — downstream certificate code
        # treats unbounded Hessian entries as UNKNOWN.
        contains = other.contains_zero()
        if np.any(contains):
            inf = np.float64(np.inf)
            return Interval(np.full_like(self.lo, -inf), np.full_like(self.hi, inf))
        inv_lo = 1.0 / other.hi
        inv_hi = 1.0 / other.lo
        return self * Interval(_round_down(inv_lo), _round_up(inv_hi))

    def __rtruediv__(self, other: Union["Interval", Number]) -> "Interval":
        return _as_interval(other) / self

    def __pow__(self, n: int) -> "Interval":
        """Integer power. Sound for ``n >= 0``."""
        if not isinstance(n, (int, np.integer)):
            raise TypeError("Interval power supports integer exponents only")
        if n < 0:
            # x^(-k) = 1 / x^k
            return Interval.point(1.0) / (self**-n)
        if n == 0:
            shape = self.lo.shape
            return Interval(np.ones(shape), np.ones(shape))
        if n == 1:
            return self
        if n == 2:
            # Special-case squaring — tighter than the generic product
            # because the corner products from __mul__ would not use
            # the fact that both factors are the same interval.
            zero_in = self.contains_zero()
            lo_sq = self.lo * self.lo
            hi_sq = self.hi * self.hi
            sq_max = np.maximum(lo_sq, hi_sq)
            sq_min = np.minimum(lo_sq, hi_sq)
            lo = np.where(zero_in, 0.0, sq_min)
            hi = sq_max
            return Interval(_round_down(lo), _round_up(hi))
        # General integer power via repeated squaring; soundness
        # preserved because every op outward-rounds.
        result = self
        for _ in range(n - 1):
            result = result * self
        return result


# ──────────────────────────────────────────────────────────────────────
# Unary elementary functions
# ──────────────────────────────────────────────────────────────────────


def _monotonic_nondec(x: Interval, f) -> Interval:
    """Image of ``x`` under a nondecreasing function ``f``, outward-rounded."""
    # Wide boxes can intentionally overflow to unbounded enclosures. The
    # certificate code treats non-finite endpoints as an abstention signal, so
    # keep the interval sound without emitting benchmark-visible warnings.
    with np.errstate(over="ignore", invalid="ignore"):
        lo = f(x.lo)
        hi = f(x.hi)
    return Interval(_round_down(lo), _round_up(hi))


def _monotonic_nonincr(x: Interval, f) -> Interval:
    """Image of ``x`` under a nonincreasing function ``f``, outward-rounded."""
    with np.errstate(over="ignore", invalid="ignore"):
        lo = f(x.hi)
        hi = f(x.lo)
    return Interval(_round_down(lo), _round_up(hi))


def exp(x: Interval) -> Interval:
    """Sound enclosure of ``exp([lo, hi])``."""
    return _monotonic_nondec(x, np.exp)


def log(x: Interval) -> Interval:
    """Sound enclosure of ``log([lo, hi])``.

    Domain: requires ``lo > 0``. Returns an unbounded lower endpoint if
    the argument touches zero, preserving soundness (downstream checks
    should then refuse to certify).
    """
    lo = x.lo
    if np.any(lo <= 0):
        # Replace nonpositive parts with -inf to keep the enclosure
        # sound without poisoning well-defined entries.
        with np.errstate(divide="ignore", invalid="ignore"):
            safe_lo = np.where(lo > 0, np.log(np.where(lo > 0, lo, 1.0)), -np.inf)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            safe_lo = np.log(lo)
    with np.errstate(divide="ignore", invalid="ignore"):
        hi = np.log(x.hi)
    return Interval(_round_down(safe_lo), _round_up(hi))


def sqrt(x: Interval) -> Interval:
    """Sound enclosure of ``sqrt([lo, hi])`` on the nonneg domain."""
    lo = x.lo
    with np.errstate(invalid="ignore"):
        safe_lo = np.where(lo >= 0, np.sqrt(np.where(lo >= 0, lo, 0.0)), -np.inf)
        hi = np.sqrt(x.hi)
    return Interval(_round_down(safe_lo), _round_up(hi))


def absolute(x: Interval) -> Interval:
    """Sound enclosure of ``|[lo, hi]|``."""
    zero_in = x.contains_zero()
    abs_lo = np.abs(x.lo)
    abs_hi = np.abs(x.hi)
    lo = np.where(zero_in, 0.0, np.minimum(abs_lo, abs_hi))
    hi = np.maximum(abs_lo, abs_hi)
    return Interval(_round_down(lo), _round_up(hi))


def tanh(x: Interval) -> Interval:
    """``tanh`` is nondecreasing on all of R."""
    return _monotonic_nondec(x, np.tanh)


def sinh(x: Interval) -> Interval:
    """``sinh`` is nondecreasing on all of R."""
    return _monotonic_nondec(x, np.sinh)


def cosh(x: Interval) -> Interval:
    """``cosh`` has a minimum at 0 and is otherwise monotone.

    Returns an enclosure whose lower endpoint is ``1`` when ``0 ∈ x``
    and the pointwise minimum of ``cosh(lo), cosh(hi)`` otherwise.
    """
    zero_in = x.contains_zero()
    ch_lo = np.cosh(x.lo)
    ch_hi = np.cosh(x.hi)
    lo = np.where(zero_in, 1.0, np.minimum(ch_lo, ch_hi))
    hi = np.maximum(ch_lo, ch_hi)
    return Interval(_round_down(lo), _round_up(hi))


def sin(x: Interval) -> Interval:
    """Sound enclosure of ``sin([lo, hi])``.

    Reduces the interval modulo ``2π`` and reasons about the location
    of ``±π/2`` critical points within it. Falls back to the safe
    enclosure ``[-1, 1]`` when the interval is wider than ``2π``.
    """
    return _sinusoid(x, offset=0.0)


def cos(x: Interval) -> Interval:
    """Sound enclosure of ``cos([lo, hi]) = sin([lo + π/2, hi + π/2])``."""
    return _sinusoid(x, offset=np.pi / 2)


def _sinusoid(x: Interval, *, offset: float) -> Interval:
    """Enclosure of ``sin(x + offset)``; used by ``sin`` and ``cos``."""
    two_pi = 2.0 * np.pi
    lo = np.asarray(x.lo + offset, dtype=np.float64)
    hi = np.asarray(x.hi + offset, dtype=np.float64)
    width = hi - lo

    # Wide intervals contain a full period → safe enclosure [-1, 1].
    wide = width >= two_pi

    # Reduce lo modulo 2π into [0, 2π); shift hi by the same amount.
    k = np.floor(lo / two_pi)
    lo_r = lo - k * two_pi
    hi_r = hi - k * two_pi

    sin_lo = np.sin(lo_r)
    sin_hi = np.sin(hi_r)

    # Critical points π/2 + 2πk (maxima = +1) and 3π/2 + 2πk (minima = -1).
    # After reduction, the next maximum after lo_r is at:
    max_pt_1 = np.pi / 2
    max_pt_2 = np.pi / 2 + two_pi
    min_pt_1 = 3 * np.pi / 2
    min_pt_2 = 3 * np.pi / 2 + two_pi

    hits_max = ((lo_r <= max_pt_1) & (max_pt_1 <= hi_r)) | ((lo_r <= max_pt_2) & (max_pt_2 <= hi_r))
    hits_min = ((lo_r <= min_pt_1) & (min_pt_1 <= hi_r)) | ((lo_r <= min_pt_2) & (min_pt_2 <= hi_r))

    lo_candidate = np.minimum(sin_lo, sin_hi)
    hi_candidate = np.maximum(sin_lo, sin_hi)
    lo_out = np.where(hits_min, -1.0, lo_candidate)
    hi_out = np.where(hits_max, 1.0, hi_candidate)

    # Apply the wide-interval fallback everywhere it holds.
    lo_out = np.where(wide, -1.0, lo_out)
    hi_out = np.where(wide, 1.0, hi_out)

    return Interval(_round_down(lo_out), _round_up(hi_out))


def tan(x: Interval) -> Interval:
    """Sound enclosure of ``tan([lo, hi])``.

    ``tan`` has vertical asymptotes at ``π/2 + kπ`` so any interval
    straddling one of these produces an unbounded enclosure. Intervals
    within a single continuous branch are handled by the monotonic
    rule.
    """
    half_pi = np.pi / 2
    pi = np.pi
    lo = np.asarray(x.lo, dtype=np.float64)
    hi = np.asarray(x.hi, dtype=np.float64)

    # Normalize so the branch check is easy: find k such that
    # (lo, hi) ⊂ (−π/2 + kπ, π/2 + kπ).
    k_lo = np.floor((lo + half_pi) / pi)
    k_hi = np.floor((hi + half_pi) / pi)
    same_branch = k_lo == k_hi

    inf = np.float64(np.inf)
    tan_lo = np.tan(lo)
    tan_hi = np.tan(hi)
    out_lo = np.where(same_branch, tan_lo, -inf)
    out_hi = np.where(same_branch, tan_hi, inf)
    return Interval(_round_down(out_lo), _round_up(out_hi))


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _as_interval(x) -> Interval:
    """Coerce a scalar / array / Interval to an :class:`Interval`."""
    if isinstance(x, Interval):
        return x
    arr = np.asarray(x, dtype=np.float64)
    return Interval(arr, arr)


__all__ = [
    "Interval",
    "exp",
    "log",
    "sqrt",
    "absolute",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
]
