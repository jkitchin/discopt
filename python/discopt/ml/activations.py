"""Activation functions for embedded networks: numpy evaluation, interval
images, and symbolic modeling expressions, from one table (#2135).

Every activation is described here once and consumed by
:meth:`NetworkDefinition.forward`, :func:`discopt.ml.bounds.propagate_bounds`
and the ``full_space`` / ``reduced_space`` / ``relu_bigm`` / trainable
formulations, so a sharpness parameter cannot apply in one place and be
silently dropped in another.

Sharpness parameter ``beta``
----------------------------
The ReLU-like smooth activations take a sharpness (inverse temperature)
``beta > 0``, all through the same scaling

.. math:: f_\\beta(z) = g(\\beta z) / \\beta ,

so that ``f_beta -> relu`` (softplus, squareplus) or ``f_beta -> relu`` from
below (Swish, GELU) as ``beta -> inf``, and ``beta = 1`` is the textbook form:

============  =========================================  ===========================
activation    ``f_beta(z)``                              ``g(u)``
============  =========================================  ===========================
softplus      ``log(1 + exp(beta z)) / beta``            ``log(1 + e^u)``
squareplus    ``(z + sqrt(z^2 + 4 / beta^2)) / 2``       ``(u + sqrt(u^2 + 4)) / 2``
swish / SiLU  ``z * sigmoid(beta z)``                    ``u * sigmoid(u)``
GELU          ``z * Phi(beta z)``                        ``u * Phi(u)``
============  =========================================  ===========================

Softplus matches ``torch.nn.Softplus(beta=...)``; squareplus is the CHKS
smoothing ``(z + sqrt(z^2 + b)) / 2`` with ``b = 4 / beta^2`` (``b = 4`` at
``beta = 1``, the squareplus default); Swish matches the usual ``x *
sigmoid(beta x)``.  LINEAR, RELU, SIGMOID and TANH take no parameter and
refuse ``beta != 1`` rather than ignore it.

Monotonicity
------------
softplus, squareplus, sigmoid, tanh, relu and linear are nondecreasing, so
the interval image of ``[lb, ub]`` is ``[f(lb), f(ub)]``.  Swish and GELU are
**not** monotone: ``g`` decreases on ``(-inf, u*]`` and increases on
``[u*, inf)`` with a global minimum ``g* < 0``, so an interval containing the
minimizer ``z* = u*/beta`` has lower end ``g*/beta``, not ``min(f(lb),
f(ub))``.  :func:`activation_interval` implements that case.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.special import erf as _np_erf
from scipy.special import expit as _expit

import discopt.modeling as dm
from discopt.ml.network import Activation

#: Activations that accept a sharpness parameter ``beta``.
PARAMETERIZED_ACTIVATIONS = frozenset(
    {Activation.SOFTPLUS, Activation.SQUAREPLUS, Activation.SWISH, Activation.GELU}
)

#: Non-monotone activations (unimodal: decreasing then increasing).
NONMONOTONE_ACTIVATIONS = frozenset({Activation.SWISH, Activation.GELU})

_SQRT1_2 = 1.0 / math.sqrt(2.0)


def check_beta(activation: Activation, beta: float) -> float:
    """Validate ``beta`` for ``activation`` and return it as a float.

    Raises ``ValueError`` for a non-finite or non-positive ``beta``, and for
    ``beta != 1`` on an activation that has no sharpness parameter (refusing it
    rather than silently embedding a different function).
    """
    try:
        b = float(beta)
    except (TypeError, ValueError):
        raise ValueError(f"beta must be a positive finite number, got {beta!r}") from None
    if not math.isfinite(b) or b <= 0.0:
        raise ValueError(f"beta must be a positive finite number, got {beta!r}")
    if b != 1.0 and activation not in PARAMETERIZED_ACTIVATIONS:
        valid = ", ".join(sorted(a.value for a in PARAMETERIZED_ACTIVATIONS))
        raise ValueError(
            f"activation {activation.value!r} takes no beta parameter (got beta={b!r}); "
            f"beta applies only to: {valid}."
        )
    return b


# ── numpy evaluation of the unit-sharpness g(u) ─────────────────────────────


def _g_softplus(u):
    return np.logaddexp(u, 0.0)


def _g_squareplus(u):
    # (u + sqrt(u^2 + 4)) / 2, written for u -> -inf without cancellation:
    # for u < 0 it equals 2 / (sqrt(u^2 + 4) - u).
    u = np.asarray(u, dtype=np.float64)
    # Both branches are evaluated; the discarded one may divide by zero.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        r = np.hypot(u, 2.0)
        out = np.where(u >= 0, 0.5 * (u + r), 2.0 / (r - u))
    return out


def _g_swish(u):
    u = np.asarray(u, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        out = u * _expit(u)
    # -inf * 0 is nan; the limit is 0.
    return np.where(np.isneginf(u), 0.0, out)


def _g_gelu(u):
    u = np.asarray(u, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        out = u * 0.5 * (1.0 + _np_erf(u * _SQRT1_2))
    return np.where(np.isneginf(u), 0.0, out)


_G: dict[Activation, Callable[[np.ndarray], np.ndarray]] = {
    Activation.SOFTPLUS: _g_softplus,
    Activation.SQUAREPLUS: _g_squareplus,
    Activation.SWISH: _g_swish,
    Activation.GELU: _g_gelu,
}


def _unimodal_minimum(g, dg, d2g, u0: float) -> tuple[float, float]:
    """Newton on ``g'(u) = 0`` from ``u0``; returns ``(u*, g(u*))``."""
    u = u0
    for _ in range(100):
        step = dg(u) / d2g(u)
        u -= step
        if abs(step) < 1e-16:
            break
    return u, float(g(np.float64(u)))


def _swish_derivs():
    s = lambda u: 1.0 / (1.0 + math.exp(-u))  # noqa: E731
    dg = lambda u: s(u) * (1.0 + u * (1.0 - s(u)))  # noqa: E731
    d2g = lambda u: s(u) * (1.0 - s(u)) * (2.0 + u * (1.0 - 2.0 * s(u)))  # noqa: E731
    return dg, d2g


def _gelu_derivs():
    pdf = lambda u: math.exp(-0.5 * u * u) / math.sqrt(2.0 * math.pi)  # noqa: E731
    cdf = lambda u: 0.5 * (1.0 + math.erf(u * _SQRT1_2))  # noqa: E731
    dg = lambda u: cdf(u) + u * pdf(u)  # noqa: E731
    d2g = lambda u: pdf(u) * (2.0 - u * u)  # noqa: E731
    return dg, d2g


# (u*, g*) for the unit-sharpness non-monotone activations, solved to machine
# precision at import: Swish u* = -1.27846..., g* = -0.27846...; GELU
# u* = -0.75179..., g* = -0.16997....  The lower bound used for an interval
# containing u* is pushed down by a relative 1e-12 (`_MIN_SAFETY`) so rounding
# in g(u*) cannot put the bound above the true minimum.
_SWISH_DG, _SWISH_D2G = _swish_derivs()
_GELU_DG, _GELU_D2G = _gelu_derivs()
_UNIMODAL_MIN = {
    Activation.SWISH: _unimodal_minimum(_g_swish, _SWISH_DG, _SWISH_D2G, -1.28),
    Activation.GELU: _unimodal_minimum(_g_gelu, _GELU_DG, _GELU_D2G, -0.75),
}
_MIN_SAFETY = 1e-12


def activation_minimizer(activation: Activation, beta: float = 1.0) -> tuple[float, float]:
    """``(z*, f(z*))`` of a non-monotone activation at sharpness ``beta``."""
    if activation not in NONMONOTONE_ACTIVATIONS:
        raise ValueError(f"activation {activation.value!r} is monotone; it has no interior minimum")
    b = check_beta(activation, beta)
    u_star, g_star = _UNIMODAL_MIN[activation]
    return u_star / b, g_star / b


def evaluate(activation: Activation, z, beta: float = 1.0) -> np.ndarray:
    """Evaluate ``activation`` (sharpness ``beta``) elementwise with numpy."""
    b = check_beta(activation, beta)
    za: np.ndarray = np.asarray(z, dtype=np.float64)
    out: np.ndarray
    if activation == Activation.LINEAR:
        out = za
    elif activation == Activation.RELU:
        out = np.maximum(za, 0.0)
    elif activation == Activation.SIGMOID:
        out = _expit(za)
    elif activation == Activation.TANH:
        out = np.tanh(za)
    elif activation in _G:
        if b == 1.0:
            out = _G[activation](za)
        else:
            with np.errstate(over="ignore"):
                out = _G[activation](b * za) / b
    else:
        raise ValueError(f"Unknown activation: {activation}")
    return out


def activation_interval(
    activation: Activation, pre_lb, pre_ub, beta: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Exact interval image ``f([pre_lb, pre_ub])``, elementwise.

    Monotone activations map endpoints to endpoints.  For Swish and GELU the
    lower end is the (slightly lowered) global minimum whenever the interval
    contains the minimizer, and the upper end is the larger endpoint value.
    """
    b = check_beta(activation, beta)
    lb = np.asarray(pre_lb, dtype=np.float64)
    ub = np.asarray(pre_ub, dtype=np.float64)
    f_lb = evaluate(activation, lb, b)
    f_ub = evaluate(activation, ub, b)
    if activation not in NONMONOTONE_ACTIVATIONS:
        return f_lb.copy(), f_ub.copy()
    z_star, f_star = activation_minimizer(activation, b)
    f_star_safe = f_star - _MIN_SAFETY * abs(f_star)
    lo = np.minimum(f_lb, f_ub)
    hi = np.maximum(f_lb, f_ub)
    contains = (lb <= z_star) & (z_star <= ub)
    lo = np.where(contains, f_star_safe, lo)
    return lo, hi


# ── symbolic expressions ────────────────────────────────────────────────────


def symbolic(activation: Activation, z, beta: float = 1.0):
    """Modeling expression for ``activation`` (sharpness ``beta``) applied to ``z``.

    Elementwise over array expressions.  RELU is *not* handled here: it has a
    formulation-specific encoding (big-M or ``dm.maximum``).  At ``beta = 1``
    the expression is the same one the formulations emitted before ``beta``
    existed (``dm.softplus(z)``, not ``dm.softplus(1.0 * z) / 1.0``), so
    unparameterized networks build an identical model.
    """
    b = check_beta(activation, beta)
    if activation == Activation.LINEAR:
        return z
    if activation == Activation.SIGMOID:
        return dm.sigmoid(z)
    if activation == Activation.TANH:
        return dm.tanh(z)
    u = z if b == 1.0 else b * z
    if activation == Activation.SOFTPLUS:
        return dm.softplus(u) if b == 1.0 else dm.softplus(u) / b
    if activation == Activation.SQUAREPLUS:
        return 0.5 * (z + dm.sqrt(z**2 + 4.0 / (b * b)))
    if activation == Activation.SWISH:
        return z * dm.sigmoid(u)
    if activation == Activation.GELU:
        return 0.5 * z * (1.0 + dm.erf(_SQRT1_2 * u))
    raise ValueError(f"No symbolic form for activation {activation}")
