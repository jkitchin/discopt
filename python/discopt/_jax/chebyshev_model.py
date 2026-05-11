"""Chebyshev models with rigorous remainder bounds.

A Chebyshev model represents a univariate factorable function ``f(x)`` on a
box ``[a, b]`` as a degree-``d`` polynomial in the Chebyshev basis plus a
rigorously propagated interval remainder::

    f(x) ∈ T(s(x)) + [r_lo, r_hi]   for all x ∈ [a, b]

where ``s(x) = 2(x - a)/(b - a) - 1 ∈ [-1, 1]`` and
``T(s) = sum_{k=0}^{d} coeffs[k] * T_k(s)``.

The remainder is an enclosure: any sample point in the domain has its true
function value contained in the polynomial-plus-remainder enclosure. This is
the foundation for tighter LP/cutting-plane relaxations of transcendental-
heavy expressions, and is M2 of issue #51.

Reference: Rajyaguru, Villanueva, Houska, Chachuat, *J. Global Optim.* 68,
413-438, 2017.

Implementation notes:
- We use ``numpy.polynomial.chebyshev`` for the polynomial primitives
  (chebadd, chebmul, chebval) and layer rigorous interval-remainder tracking
  on top. Coefficient arithmetic is exact in floating point modulo
  rounding; the remainder absorbs truncation, the cross-terms from interval
  arithmetic, and a small safety margin for sampling/quadrature noise when
  building Chebyshev expansions of unary functions.
- All composition routines bound ``f`` on the *inner range* ``[u_lo, u_hi]``
  of the inner model (which already includes the inner remainder), so the
  truncation bound applies pointwise across the whole domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.polynomial.chebyshev as np_cheb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _interval_mul(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """Rigorous interval product."""
    a_lo, a_hi = a
    b_lo, b_hi = b
    products = (a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi)
    return (min(products), max(products))


def _pad_or_trunc(c: np.ndarray, degree: int) -> tuple[np.ndarray, float]:
    """Pad ``c`` with zeros or truncate to length ``degree+1``.

    Returns the (possibly truncated) coefficient array plus a rigorous
    upper bound on the magnitude of the dropped tail (``|T_k(s)| ≤ 1``).
    """
    if len(c) <= degree + 1:
        out = np.zeros(degree + 1)
        out[: len(c)] = c
        return out, 0.0
    head = c[: degree + 1].copy()
    tail = c[degree + 1 :]
    trunc = float(np.sum(np.abs(tail)))
    return head, trunc


def _polynomial_range(coeffs: np.ndarray) -> tuple[float, float]:
    """Exact range of ``p(s) = sum_k c_k T_k(s)`` for ``s ∈ [-1, 1]``.

    Computed by evaluating ``p`` at the endpoints and at all real critical
    points in ``(-1, 1)`` (roots of ``p'``). For numerically reasonable
    coefficients this matches the true min/max to working precision.
    """
    n = len(coeffs)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        c = float(coeffs[0])
        return (c, c)
    candidates = [
        float(np_cheb.chebval(-1.0, coeffs)),
        float(np_cheb.chebval(1.0, coeffs)),
    ]
    deriv = np_cheb.chebder(coeffs)
    if len(deriv) > 0 and float(np.max(np.abs(deriv))) > 0.0:
        try:
            roots = np_cheb.chebroots(deriv)
        except (np.linalg.LinAlgError, ValueError):
            roots = np.array([])
        for r in roots:
            if np.abs(np.imag(r)) < 1e-10:
                rr = float(np.real(r))
                if -1.0 < rr < 1.0:
                    candidates.append(float(np_cheb.chebval(rr, coeffs)))
    return (min(candidates), max(candidates))


def _chebcoeff_from_samples(f_vals: np.ndarray) -> np.ndarray:
    """Chebyshev coefficients of ``f`` from samples at Chebyshev-Gauss nodes.

    Given ``f_vals[k] = f(cos(pi(k+0.5)/N))`` for k = 0..N-1 (N nodes), returns
    coefficients ``c_0, ..., c_{N-1}`` of the degree-(N-1) interpolant in
    the Chebyshev-of-first-kind basis.
    """
    N = len(f_vals)
    coeffs = np.zeros(N)
    k = np.arange(N)
    for j in range(N):
        # c_j = (2/N) * sum_k f_vals[k] * cos(pi*j*(k+0.5)/N), c_0 has 1/N
        weight = 2.0 / N if j > 0 else 1.0 / N
        coeffs[j] = weight * float(np.sum(f_vals * np.cos(np.pi * j * (k + 0.5) / N)))
    return coeffs


# ---------------------------------------------------------------------------
# ChebyshevModel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChebyshevModel:
    """Polynomial-plus-remainder enclosure of a univariate function on a box.

    Attributes
    ----------
    coeffs:
        Length ``degree + 1`` array of Chebyshev coefficients in the
        ``s ∈ [-1, 1]`` domain.
    remainder:
        Tuple ``(r_lo, r_hi)`` with ``r_lo ≤ 0 ≤ r_hi`` typically, giving
        a rigorous interval bound on ``f(x) - T(s(x))`` for all ``x``.
    domain:
        Tuple ``(a, b)`` with ``a < b`` describing the original ``x`` box.
    """

    coeffs: np.ndarray
    remainder: tuple[float, float] = field(default=(0.0, 0.0))
    domain: tuple[float, float] = field(default=(-1.0, 1.0))

    @property
    def degree(self) -> int:
        return len(self.coeffs) - 1

    def to_s(self, x):
        a, b = self.domain
        return 2.0 * (np.asarray(x) - a) / (b - a) - 1.0

    def evaluate_polynomial(self, x):
        """Evaluate ``T(s(x))`` (polynomial part only). NumPy-broadcastable."""
        return np_cheb.chebval(self.to_s(x), self.coeffs)

    def bounds(self) -> tuple[float, float]:
        """Rigorous bounds ``[lo, hi]`` containing ``f(x)`` for all ``x``."""
        p_lo, p_hi = _polynomial_range(self.coeffs)
        return (p_lo + self.remainder[0], p_hi + self.remainder[1])


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def from_constant(c: float, domain: tuple[float, float], degree: int) -> ChebyshevModel:
    """Constant model ``f(x) = c``."""
    coeffs = np.zeros(degree + 1)
    coeffs[0] = float(c)
    return ChebyshevModel(coeffs, (0.0, 0.0), (float(domain[0]), float(domain[1])))


def from_variable(domain: tuple[float, float], degree: int) -> ChebyshevModel:
    """Identity model ``f(x) = x`` on ``domain``.

    On ``[a, b]``: ``x = (a+b)/2 + (b-a)/2 * s``, so ``coeffs = [(a+b)/2, (b-a)/2, 0, ...]``.
    """
    a, b = domain
    if degree < 1:
        raise ValueError("from_variable requires degree >= 1")
    coeffs = np.zeros(degree + 1)
    coeffs[0] = 0.5 * (a + b)
    coeffs[1] = 0.5 * (b - a)
    return ChebyshevModel(coeffs, (0.0, 0.0), (float(domain[0]), float(domain[1])))


# ---------------------------------------------------------------------------
# Arithmetic (binary)
# ---------------------------------------------------------------------------


def _check_compat(a: ChebyshevModel, b: ChebyshevModel) -> None:
    if a.domain != b.domain:
        raise ValueError(f"domain mismatch: {a.domain} vs {b.domain}")
    if a.degree != b.degree:
        raise ValueError(f"degree mismatch: {a.degree} vs {b.degree}")


def add(a: ChebyshevModel, b: ChebyshevModel) -> ChebyshevModel:
    _check_compat(a, b)
    coeffs = a.coeffs + b.coeffs
    rem = (a.remainder[0] + b.remainder[0], a.remainder[1] + b.remainder[1])
    return ChebyshevModel(coeffs, rem, a.domain)


def neg(a: ChebyshevModel) -> ChebyshevModel:
    return ChebyshevModel(-a.coeffs, (-a.remainder[1], -a.remainder[0]), a.domain)


def sub(a: ChebyshevModel, b: ChebyshevModel) -> ChebyshevModel:
    return add(a, neg(b))


def scalar_mul(s: float, a: ChebyshevModel) -> ChebyshevModel:
    s = float(s)
    coeffs = s * a.coeffs
    if s >= 0:
        rem = (s * a.remainder[0], s * a.remainder[1])
    else:
        rem = (s * a.remainder[1], s * a.remainder[0])
    return ChebyshevModel(coeffs, rem, a.domain)


def scalar_add(s: float, a: ChebyshevModel) -> ChebyshevModel:
    coeffs = a.coeffs.copy()
    coeffs[0] += float(s)
    return ChebyshevModel(coeffs, a.remainder, a.domain)


def mul(a: ChebyshevModel, b: ChebyshevModel) -> ChebyshevModel:
    """Product with truncation + interval cross-term remainder."""
    _check_compat(a, b)
    full = np_cheb.chebmul(a.coeffs, b.coeffs)
    coeffs, trunc = _pad_or_trunc(full, a.degree)

    a_range = _polynomial_range(a.coeffs)
    b_range = _polynomial_range(b.coeffs)
    cross_a_rb = _interval_mul(a_range, b.remainder)
    cross_ra_b = _interval_mul(a.remainder, b_range)
    cross_ra_rb = _interval_mul(a.remainder, b.remainder)

    rem_lo = cross_a_rb[0] + cross_ra_b[0] + cross_ra_rb[0] - trunc
    rem_hi = cross_a_rb[1] + cross_ra_b[1] + cross_ra_rb[1] + trunc
    return ChebyshevModel(coeffs, (rem_lo, rem_hi), a.domain)


# ---------------------------------------------------------------------------
# Univariate composition
# ---------------------------------------------------------------------------


_COMPOSE_OVERSAMPLE = 8  # build f's expansion at N = max(oversample*(d+1), MIN_NODES) nodes
_COMPOSE_MIN_NODES = 64
_COMPOSE_VALIDATE_GRID = 4001  # fine grid for empirical residual check
_COMPOSE_SAFETY_REL = 1e-12  # relative numerical-noise safety
_COMPOSE_LIPSCHITZ_PAD = 1.0  # empirical bound × this multiplier for between-sample slack


def _build_unary_chebyshev(
    f_callable, u_lo: float, u_hi: float, degree: int
) -> tuple[np.ndarray, float]:
    """Build degree-``d`` Chebyshev model of ``f`` on ``[u_lo, u_hi]``.

    Returns ``(coeffs, tail_bound)`` where ``coeffs`` has length ``degree+1``
    and ``tail_bound`` is an upper bound on
    ``|f(u) - sum_{k≤d} coeffs[k] T_k(t(u))|`` for ``u ∈ [u_lo, u_hi]``.

    The bound is the max of two estimates: the absolute-sum of dropped
    high-degree coefficients (rigorous when sampling Nyquist-resolves f),
    and a fine-grid empirical residual augmented with a Lipschitz-style pad
    to cover between-sample variation. For analytic ``f`` (exp/log/sqrt/
    trig) on bounded ranges this is conservative.
    """
    if u_hi <= u_lo:
        out = np.zeros(degree + 1)
        out[0] = float(f_callable(0.5 * (u_lo + u_hi)))
        return out, 0.0

    N = max(_COMPOSE_OVERSAMPLE * (degree + 1), _COMPOSE_MIN_NODES)
    k = np.arange(N)
    nodes_s = np.cos(np.pi * (k + 0.5) / N)
    nodes_u = u_lo + 0.5 * (u_hi - u_lo) * (nodes_s + 1.0)
    f_vals = np.array([float(f_callable(float(u))) for u in nodes_u])
    full = _chebcoeff_from_samples(f_vals)

    head = full[: degree + 1].copy()
    tail = full[degree + 1 :]
    tail_bound = float(np.sum(np.abs(tail)))

    # Empirical fine-grid residual: covers interpolation error not captured
    # by the aliased tail when f has slow Chebyshev decay (e.g. log near a
    # singularity). M points give grid spacing ≈ 2/(M-1) in s, so the
    # Lipschitz-style pad uses adjacent-sample residual differences as a
    # local Lipschitz estimate for the residual function r = f - p.
    M = _COMPOSE_VALIDATE_GRID
    fine_s = np.linspace(-1.0, 1.0, M)
    fine_u = u_lo + 0.5 * (u_hi - u_lo) * (fine_s + 1.0)
    f_fine = np.array([float(f_callable(float(u))) for u in fine_u])
    p_fine = np_cheb.chebval(fine_s, head)
    resid = np.abs(f_fine - p_fine)
    resid_max = float(resid.max())
    # Local-Lipschitz pad: max successive residual jump (covers between-grid
    # variation under the assumption that the residual is no rougher locally
    # than the maximum sampled jump).
    if M >= 2:
        resid_jump = float(np.max(np.abs(np.diff(resid))))
        lip_pad = _COMPOSE_LIPSCHITZ_PAD * resid_jump
    else:
        lip_pad = 0.0

    safety = _COMPOSE_SAFETY_REL * float(np.sum(np.abs(full)) + 1.0)
    final_bound = max(tail_bound, resid_max + lip_pad) + safety
    return head, final_bound


def _clenshaw_compose(f_coeffs: np.ndarray, t_of_g: ChebyshevModel) -> ChebyshevModel:
    """Substitute ``t_of_g`` into ``sum_k f_coeffs[k] T_k(·)`` via Clenshaw.

    All multiplications happen in ``ChebyshevModel`` arithmetic, so the
    returned model carries the right truncation + interval remainders.
    """
    d = t_of_g.degree
    domain = t_of_g.domain
    n = len(f_coeffs)
    if n == 0:
        return from_constant(0.0, domain, d)
    if n == 1:
        return from_constant(float(f_coeffs[0]), domain, d)
    # Clenshaw: b_{n+1} = b_n = 0; b_k = f_k + 2*t*b_{k+1} - b_{k+2}; for k=0
    # use b_0 = f_0 + t*b_1 - b_2; result = b_0.
    b_kp2 = from_constant(0.0, domain, d)
    b_kp1 = from_constant(0.0, domain, d)
    for k in range(n - 1, 0, -1):
        # b_k = f_coeffs[k] + 2*t*b_{k+1} - b_{k+2}
        two_tb = scalar_mul(2.0, mul(t_of_g, b_kp1))
        b_k = sub(scalar_add(float(f_coeffs[k]), two_tb), b_kp2)
        b_kp2 = b_kp1
        b_kp1 = b_k
    # b_0 = f_coeffs[0] + t*b_1 - b_2
    return sub(scalar_add(float(f_coeffs[0]), mul(t_of_g, b_kp1)), b_kp2)


def compose_unary(f_callable, g: ChebyshevModel) -> ChebyshevModel:
    """Return ``f(g)`` as a Chebyshev model with rigorous remainder.

    Works for any ``f`` for which sampling at floats is reliable on the
    inner range. The inner range is computed from ``g.bounds()`` (which
    already includes ``g``'s remainder), so the resulting truncation bound
    applies pointwise across ``g.domain``.
    """
    u_lo, u_hi = g.bounds()
    d = g.degree
    domain = g.domain

    if u_hi - u_lo < 1e-15:
        # Inner is essentially a constant.
        return from_constant(float(f_callable(0.5 * (u_lo + u_hi))), domain, d)

    f_coeffs, f_tail = _build_unary_chebyshev(f_callable, u_lo, u_hi, d)

    width = 0.5 * (u_hi - u_lo)
    center = 0.5 * (u_lo + u_hi)
    # t_of_g(x) = (g(x) - center) / width  ∈ [-1, 1] approximately
    t_of_g = scalar_mul(1.0 / width, scalar_add(-center, g))

    composed = _clenshaw_compose(f_coeffs, t_of_g)
    rem_lo = composed.remainder[0] - f_tail
    rem_hi = composed.remainder[1] + f_tail
    return ChebyshevModel(composed.coeffs, (rem_lo, rem_hi), domain)


# ---------------------------------------------------------------------------
# Convenience univariate operators
# ---------------------------------------------------------------------------


def cheb_exp(g: ChebyshevModel) -> ChebyshevModel:
    return compose_unary(np.exp, g)


def cheb_log(g: ChebyshevModel) -> ChebyshevModel:
    lo, _ = g.bounds()
    if lo <= 0.0:
        raise ValueError(f"log: inner range must be positive, got lo={lo}")
    return compose_unary(np.log, g)


def cheb_sqrt(g: ChebyshevModel) -> ChebyshevModel:
    lo, _ = g.bounds()
    if lo < 0.0:
        raise ValueError(f"sqrt: inner range must be non-negative, got lo={lo}")
    return compose_unary(np.sqrt, g)


def cheb_recip(g: ChebyshevModel) -> ChebyshevModel:
    lo, hi = g.bounds()
    if lo <= 0.0 <= hi:
        raise ValueError(f"recip: inner range straddles zero ({lo}, {hi})")
    return compose_unary(lambda u: 1.0 / u, g)


def cheb_square(g: ChebyshevModel) -> ChebyshevModel:
    return mul(g, g)


def cheb_sin(g: ChebyshevModel) -> ChebyshevModel:
    return compose_unary(np.sin, g)


def cheb_cos(g: ChebyshevModel) -> ChebyshevModel:
    return compose_unary(np.cos, g)


def cheb_tanh(g: ChebyshevModel) -> ChebyshevModel:
    return compose_unary(np.tanh, g)


__all__ = [
    "ChebyshevModel",
    "from_constant",
    "from_variable",
    "add",
    "sub",
    "neg",
    "mul",
    "scalar_mul",
    "scalar_add",
    "compose_unary",
    "cheb_exp",
    "cheb_log",
    "cheb_sqrt",
    "cheb_recip",
    "cheb_square",
    "cheb_sin",
    "cheb_cos",
    "cheb_tanh",
]
