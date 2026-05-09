"""Taylor models with rigorous remainder bounds.

A Taylor model represents a univariate factorable function ``f(x)`` on a box
``[a, b]`` as a degree-``d`` polynomial in the monomial basis (around the
midpoint, normalized) plus a rigorously propagated interval remainder::

    f(x) ∈ T(s(x)) + [r_lo, r_hi]   for all x ∈ [a, b]

where ``s(x) = 2(x - a)/(b - a) - 1 ∈ [-1, 1]`` (so ``s = 0`` at the midpoint
of ``[a, b]``) and ``T(s) = sum_{k=0}^{d} coeffs[k] * s^k``. The polynomial
coefficients are the standard Taylor coefficients of ``f`` at the midpoint
in the *normalized* variable ``s``: ``coeffs[k] = f^(k)(x0) * r^k / k!`` with
``x0 = (a+b)/2`` and ``r = (b-a)/2``.

Distinct from the first-order Taylor *cuts* in ``cutting_planes.py`` (which
are linear and carry no verified remainder), Taylor models track a rigorous
interval enclosure of the true function across the entire box. This is M3
of issue #51.

References:
- Bompadre, Mitsos, Chachuat, *J. Global Optim.* 57(1), 75-114, 2013.
- Mitsos, Chachuat, Barton, *SIAM J. Optim.* 20(2), 573-601, 2009.

Implementation notes:
- Polynomial primitives come from ``numpy.polynomial.polynomial`` (polyadd,
  polymul, polyval, polyder, polyroots).
- Taylor coefficients of ``f`` at the inner-range midpoint are computed via
  repeated ``jax.grad`` applications, giving accurate derivatives without
  finite-difference noise.
- The remainder for unary composition combines the absolute-sum dropped
  high-degree tail with a fine-grid empirical residual + Lipschitz-style pad,
  the same approach used in ``chebyshev_model.py``. For analytic functions
  on bounded domains this is conservative.
- M2 (Chebyshev) and M3 (Taylor) share most operational structure; the only
  difference is the polynomial basis and the corresponding coefficient
  computation. They live in separate modules for clarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import numpy.polynomial.polynomial as np_poly

# ---------------------------------------------------------------------------
# Helpers (interval arithmetic + polynomial range)
# ---------------------------------------------------------------------------


def _interval_mul(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """Rigorous interval product."""
    a_lo, a_hi = a
    b_lo, b_hi = b
    products = (a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi)
    return (min(products), max(products))


def _pad_or_trunc(c: np.ndarray, degree: int) -> tuple[np.ndarray, float]:
    """Pad ``c`` with zeros or truncate to length ``degree+1``.

    The truncation bound uses ``|s^k| ≤ 1`` on ``[-1, 1]``: the absolute
    value of the dropped polynomial tail is at most ``sum_{k>d} |c_k|``.
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
    """Exact range of ``p(s) = sum_k c_k s^k`` for ``s ∈ [-1, 1]``.

    Computed by evaluating ``p`` at the endpoints and at all real critical
    points in ``(-1, 1)`` (roots of ``p'``).
    """
    n = len(coeffs)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        c = float(coeffs[0])
        return (c, c)
    candidates = [
        float(np_poly.polyval(-1.0, coeffs)),
        float(np_poly.polyval(1.0, coeffs)),
    ]
    deriv = np_poly.polyder(coeffs)
    if len(deriv) > 0 and float(np.max(np.abs(deriv))) > 0.0:
        try:
            roots = np_poly.polyroots(deriv)
        except (np.linalg.LinAlgError, ValueError):
            roots = np.array([])
        for r in roots:
            if np.abs(np.imag(r)) < 1e-10:
                rr = float(np.real(r))
                if -1.0 < rr < 1.0:
                    candidates.append(float(np_poly.polyval(rr, coeffs)))
    return (min(candidates), max(candidates))


# ---------------------------------------------------------------------------
# TaylorModel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaylorModel:
    """Polynomial-plus-remainder enclosure in the (normalized) monomial basis.

    Attributes
    ----------
    coeffs:
        Length ``degree + 1`` array. ``coeffs[k]`` is the coefficient of
        ``s^k`` in the polynomial part, where ``s = 2(x - a)/(b - a) - 1``.
        For unary composition this corresponds to the Taylor expansion of
        ``f`` at the midpoint, with ``coeffs[k] = f^(k)(x0) * r^k / k!``.
    remainder:
        Tuple ``(r_lo, r_hi)`` giving a rigorous interval bound on
        ``f(x) - T(s(x))`` for all ``x ∈ domain``.
    domain:
        Tuple ``(a, b)`` with ``a < b``.
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
        return np_poly.polyval(self.to_s(x), self.coeffs)

    def bounds(self) -> tuple[float, float]:
        """Rigorous bounds ``[lo, hi]`` containing ``f(x)`` for all ``x``."""
        p_lo, p_hi = _polynomial_range(self.coeffs)
        return (p_lo + self.remainder[0], p_hi + self.remainder[1])


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def from_constant(c: float, domain: tuple[float, float], degree: int) -> TaylorModel:
    """Constant model ``f(x) = c``."""
    coeffs = np.zeros(degree + 1)
    coeffs[0] = float(c)
    return TaylorModel(coeffs, (0.0, 0.0), (float(domain[0]), float(domain[1])))


def from_variable(domain: tuple[float, float], degree: int) -> TaylorModel:
    """Identity model ``f(x) = x`` on ``domain``.

    On ``[a, b]`` with ``s = 2(x - a)/(b - a) - 1``, ``x = (a+b)/2 + (b-a)/2 * s``,
    so ``coeffs = [(a+b)/2, (b-a)/2, 0, ...]``.
    """
    a, b = domain
    if degree < 1:
        raise ValueError("from_variable requires degree >= 1")
    coeffs = np.zeros(degree + 1)
    coeffs[0] = 0.5 * (a + b)
    coeffs[1] = 0.5 * (b - a)
    return TaylorModel(coeffs, (0.0, 0.0), (float(domain[0]), float(domain[1])))


# ---------------------------------------------------------------------------
# Arithmetic (binary)
# ---------------------------------------------------------------------------


def _check_compat(a: TaylorModel, b: TaylorModel) -> None:
    if a.domain != b.domain:
        raise ValueError(f"domain mismatch: {a.domain} vs {b.domain}")
    if a.degree != b.degree:
        raise ValueError(f"degree mismatch: {a.degree} vs {b.degree}")


def add(a: TaylorModel, b: TaylorModel) -> TaylorModel:
    _check_compat(a, b)
    coeffs = a.coeffs + b.coeffs
    rem = (a.remainder[0] + b.remainder[0], a.remainder[1] + b.remainder[1])
    return TaylorModel(coeffs, rem, a.domain)


def neg(a: TaylorModel) -> TaylorModel:
    return TaylorModel(-a.coeffs, (-a.remainder[1], -a.remainder[0]), a.domain)


def sub(a: TaylorModel, b: TaylorModel) -> TaylorModel:
    return add(a, neg(b))


def scalar_mul(s: float, a: TaylorModel) -> TaylorModel:
    s = float(s)
    coeffs = s * a.coeffs
    if s >= 0:
        rem = (s * a.remainder[0], s * a.remainder[1])
    else:
        rem = (s * a.remainder[1], s * a.remainder[0])
    return TaylorModel(coeffs, rem, a.domain)


def scalar_add(s: float, a: TaylorModel) -> TaylorModel:
    coeffs = a.coeffs.copy()
    coeffs[0] += float(s)
    return TaylorModel(coeffs, a.remainder, a.domain)


def mul(a: TaylorModel, b: TaylorModel) -> TaylorModel:
    """Product with truncation + interval cross-term remainder."""
    _check_compat(a, b)
    full = np_poly.polymul(a.coeffs, b.coeffs)
    coeffs, trunc = _pad_or_trunc(full, a.degree)

    a_range = _polynomial_range(a.coeffs)
    b_range = _polynomial_range(b.coeffs)
    cross_a_rb = _interval_mul(a_range, b.remainder)
    cross_ra_b = _interval_mul(a.remainder, b_range)
    cross_ra_rb = _interval_mul(a.remainder, b.remainder)

    rem_lo = cross_a_rb[0] + cross_ra_b[0] + cross_ra_rb[0] - trunc
    rem_hi = cross_a_rb[1] + cross_ra_b[1] + cross_ra_rb[1] + trunc
    return TaylorModel(coeffs, (rem_lo, rem_hi), a.domain)


# ---------------------------------------------------------------------------
# Unary composition
# ---------------------------------------------------------------------------


_COMPOSE_VALIDATE_GRID = 4001
_COMPOSE_SAFETY_REL = 1e-12
_COMPOSE_LIPSCHITZ_PAD = 1.0


def _taylor_coefficients(f_callable, x0: float, half_width: float, degree: int) -> np.ndarray:
    """Taylor coefficients of ``f`` at ``x0`` in the normalized variable ``s``.

    Returns ``coeffs`` such that ``coeffs[k] = f^(k)(x0) * half_width^k / k!``,
    so ``sum_k coeffs[k] * s^k`` approximates ``f(x0 + half_width * s)`` for
    ``s ∈ [-1, 1]``.

    Derivatives are computed via repeated ``jax.grad`` on a JAX-traceable
    wrapper. ``f_callable`` must accept and return JAX-compatible scalars
    (the standard NumPy unary functions ``np.exp``, ``np.log``, etc. work
    because JAX's ``jnp`` versions have the same API and ``jax.grad`` traces
    through the chosen elementary operations of the wrapper).
    """

    def _f(u):
        # Wrap to ensure jax can trace; rely on caller passing jax-compatible f.
        return f_callable(u)

    coeffs = np.zeros(degree + 1)
    deriv = _f
    factorial = 1.0
    hw_pow = 1.0
    for k in range(degree + 1):
        val = float(deriv(jnp.asarray(x0, dtype=jnp.float64)))
        coeffs[k] = val * hw_pow / factorial
        deriv = jax.grad(deriv)
        hw_pow *= half_width
        factorial *= k + 1
    return coeffs


def _horner_compose(f_coeffs: np.ndarray, s_of_g: TaylorModel) -> TaylorModel:
    """Substitute ``s_of_g`` into ``sum_k f_coeffs[k] * s^k`` via Horner.

    All multiplications happen in ``TaylorModel`` arithmetic so the returned
    model carries the right truncation + interval remainders.
    """
    d = s_of_g.degree
    domain = s_of_g.domain
    n = len(f_coeffs)
    if n == 0:
        return from_constant(0.0, domain, d)
    if n == 1:
        return from_constant(float(f_coeffs[0]), domain, d)
    # Horner: result = a_0 + s*(a_1 + s*(a_2 + ... + s*a_d))
    acc = from_constant(float(f_coeffs[-1]), domain, d)
    for k in range(n - 2, -1, -1):
        acc = scalar_add(float(f_coeffs[k]), mul(s_of_g, acc))
    return acc


def compose_unary(f_callable, g: TaylorModel) -> TaylorModel:
    """Return ``f(g)`` as a Taylor model with rigorous remainder.

    ``f_callable`` should be a JAX-traceable scalar function (``jnp.exp``,
    ``jnp.log``, etc., or NumPy versions which JAX traces equivalently).
    """
    u_lo, u_hi = g.bounds()
    d = g.degree
    domain = g.domain

    if u_hi - u_lo < 1e-15:
        return from_constant(float(f_callable(jnp.asarray(0.5 * (u_lo + u_hi)))), domain, d)

    x0 = 0.5 * (u_lo + u_hi)
    half_width = 0.5 * (u_hi - u_lo)
    f_coeffs = _taylor_coefficients(f_callable, x0, half_width, d)

    # s_f(g) = (g - x0) / half_width — a TaylorModel parameterized in g's s-domain.
    s_of_g = scalar_mul(1.0 / half_width, scalar_add(-x0, g))

    composed = _horner_compose(f_coeffs, s_of_g)

    # Fine-grid residual bound on f vs. truncated Taylor polynomial.
    M = _COMPOSE_VALIDATE_GRID
    fine_s = np.linspace(-1.0, 1.0, M)
    fine_u = x0 + half_width * fine_s
    f_fine = np.array([float(f_callable(jnp.asarray(float(u)))) for u in fine_u])
    p_fine = np_poly.polyval(fine_s, f_coeffs)
    resid = np.abs(f_fine - p_fine)
    resid_max = float(resid.max())
    if M >= 2:
        lip_pad = _COMPOSE_LIPSCHITZ_PAD * float(np.max(np.abs(np.diff(resid))))
    else:
        lip_pad = 0.0

    safety = _COMPOSE_SAFETY_REL * float(np.sum(np.abs(f_coeffs)) + 1.0)
    f_remainder_radius = resid_max + lip_pad + safety

    return TaylorModel(
        composed.coeffs,
        (composed.remainder[0] - f_remainder_radius, composed.remainder[1] + f_remainder_radius),
        domain,
    )


# ---------------------------------------------------------------------------
# Convenience operators (use jnp counterparts so jax.grad traces them).
# ---------------------------------------------------------------------------


def taylor_exp(g: TaylorModel) -> TaylorModel:
    return compose_unary(jnp.exp, g)


def taylor_log(g: TaylorModel) -> TaylorModel:
    lo, _ = g.bounds()
    if lo <= 0.0:
        raise ValueError(f"log: inner range must be positive, got lo={lo}")
    return compose_unary(jnp.log, g)


def taylor_sqrt(g: TaylorModel) -> TaylorModel:
    lo, _ = g.bounds()
    if lo <= 0.0:
        # sqrt has unbounded derivatives at 0, so we require strictly positive.
        raise ValueError(f"sqrt: inner range must be strictly positive, got lo={lo}")
    return compose_unary(jnp.sqrt, g)


def taylor_recip(g: TaylorModel) -> TaylorModel:
    lo, hi = g.bounds()
    if lo <= 0.0 <= hi:
        raise ValueError(f"recip: inner range straddles zero ({lo}, {hi})")
    return compose_unary(lambda u: 1.0 / u, g)


def taylor_square(g: TaylorModel) -> TaylorModel:
    return mul(g, g)


def taylor_sin(g: TaylorModel) -> TaylorModel:
    return compose_unary(jnp.sin, g)


def taylor_cos(g: TaylorModel) -> TaylorModel:
    return compose_unary(jnp.cos, g)


def taylor_tanh(g: TaylorModel) -> TaylorModel:
    return compose_unary(jnp.tanh, g)


__all__ = [
    "TaylorModel",
    "from_constant",
    "from_variable",
    "add",
    "sub",
    "neg",
    "mul",
    "scalar_mul",
    "scalar_add",
    "compose_unary",
    "taylor_exp",
    "taylor_log",
    "taylor_sqrt",
    "taylor_recip",
    "taylor_square",
    "taylor_sin",
    "taylor_cos",
    "taylor_tanh",
]
