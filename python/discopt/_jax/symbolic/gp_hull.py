"""Geometric-programming (GP) log-lift joint relaxation for monomial terms.

This module implements a certified convex/concave relaxation for a multivariate
monomial term

    t = c * prod_j x_j^{a_j},    c > 0,  x_j > 0 in [x_j^L, x_j^U],  a_j real,

that is provably tighter than applying compositional McCormick relaxations to the
product. It addresses issues #189 and #181.

Log-lift
--------
Introduce the lifted variables ``u_j = log(x_j)`` with bounds
``[log x_j^L, log x_j^U]``. Then

    t = exp(log c + sum_j a_j u_j) = exp(s),    s = log c + a . u,

so ``s`` is *affine* in ``u``. Because ``exp`` is convex and increasing and ``s``
is affine, ``t = exp(s)`` is convex in ``u``.

Since ``s`` is affine, over the ``u``-box it attains its extremes at a box vertex:

    sL = log c + sum_j (a_j * u_j^L if a_j >= 0 else a_j * u_j^U)
    sU = log c + sum_j (a_j * u_j^U if a_j >= 0 else a_j * u_j^L)

Relaxation (in ``u``-space)
---------------------------
    cv(u) = exp(s)
    cc(u) = exp(sL) + (exp(sU) - exp(sL)) / (sU - sL) * (s - sL)

with ``cc(u) = exp(s)`` in the degenerate case ``sU == sL``.

Proof of validity
-----------------
``exp`` is convex on ``R``. Convexity gives two certified bounds on ``[sL, sU]``:

* Below its chord: for every ``s in [sL, sU]`` the chord (secant) of ``exp``
  through ``(sL, exp sL)`` and ``(sU, exp sU)`` lies on or above ``exp(s)``.
  Hence ``cc``, which is exactly that secant evaluated at ``s = log c + a . u``,
  is an affine-in-``s`` (therefore concave-in-``u``) overestimator with
  ``cc(u) >= exp(s) = t``.
* Above its tangents: ``exp`` lies on or above each of its tangent lines, and
  more directly ``exp(s)`` is itself convex in ``u`` (composition of the convex
  increasing ``exp`` with the affine ``s``). Thus ``cv(u) = exp(s) = t`` is the
  tight convex underestimator, with ``cv(u) = t`` everywhere.

Therefore ``cv(u) <= t <= cc(u)`` for all ``u`` in the box, ``cv`` is convex,
``cc`` is concave, and ``cv`` coincides with ``t`` exactly.

Reference
---------
Boyd et al. (2007), "A tutorial on geometric programming."
"""

from __future__ import annotations

import jax.numpy as jnp


def monomial_log_envelope(u, log_c, exps, u_lb, u_ub):
    """Convex/concave envelope of a monomial under the log-lift ``u = log(x)``.

    Computes the GP relaxation of ``t = c * prod_j x_j^{a_j} = exp(log c + a . u)``
    at a single point ``u`` of shape ``(n,)``, returning ``(cv, cc)`` where ``cv``
    is the (tight) convex underestimator and ``cc`` the concave overestimator.

    With ``s = log c + a . u`` affine in ``u`` and ``exp`` convex-increasing,
    ``t = exp(s)`` is convex in ``u``. The secant of ``exp`` over the affine range
    ``[sL, sU]`` of ``s`` is a certified concave overestimator (``exp`` lies below
    its chord), while ``exp(s)`` itself is the tight convex underestimator (``exp``
    lies above its tangents). See Boyd et al. (2007),
    "A tutorial on geometric programming," and the module docstring for the proof.

    Parameters
    ----------
    u : array-like, shape (n,)
        Lifted point ``u_j = log(x_j)``.
    log_c : scalar array-like
        ``log(c)`` for the positive monomial coefficient ``c``.
    exps : array-like, shape (n,)
        Monomial exponents ``a_j`` (real).
    u_lb, u_ub : array-like, shape (n,)
        Lower/upper bounds on ``u``: ``u_j^L = log x_j^L``, ``u_j^U = log x_j^U``.

    Returns
    -------
    cv, cc : jax.Array
        Scalar convex underestimator and concave overestimator at ``u``. Pure JAX,
        jit-compatible.
    """
    u = jnp.asarray(u)
    log_c = jnp.asarray(log_c)
    exps = jnp.asarray(exps)
    u_lb = jnp.asarray(u_lb)
    u_ub = jnp.asarray(u_ub)

    # s is affine in u.
    s = log_c + jnp.sum(exps * u)

    # Extremes of the affine s over the u-box (vertex of the box).
    pos = exps >= 0
    s_lb = log_c + jnp.sum(exps * jnp.where(pos, u_lb, u_ub))
    s_ub = log_c + jnp.sum(exps * jnp.where(pos, u_ub, u_lb))

    exp_sl = jnp.exp(s_lb)
    exp_su = jnp.exp(s_ub)

    # cv = exp(s) = t is convex in u and tight.
    cv = jnp.exp(s)

    # cc = secant of exp over [sL, sU] at s; degenerate sU == sL -> exp(s).
    denom = s_ub - s_lb
    slope = jnp.where(denom == 0, 0.0, (exp_su - exp_sl) / jnp.where(denom == 0, 1.0, denom))
    cc = jnp.where(denom == 0, jnp.exp(s), exp_sl + slope * (s - s_lb))

    return cv, cc
