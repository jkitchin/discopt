"""Difference-of-convex (DC) relaxation for a SIGNED signomial in the GP log domain.

This module implements a certified convex/concave relaxation for a multivariate
*signed* signomial

    s(x) = sum_k sigma_k * c_k * prod_j x_j^{a_kj},

with sign ``sigma_k in {+1, -1}``, positive coefficient ``c_k > 0`` and positive
variables ``x_j > 0``. It addresses issue #114 (signomial mixed-sign global
solver).

Log-lift
--------
Introduce the lifted variables ``u_j = log(x_j)``. Each monomial

    m_k(u) = c_k * exp(a_k . u) = exp(log c_k + a_k . u)

is the exponential of an affine function of ``u`` and is therefore *convex* in
``u`` (``exp`` is convex and increasing; ``log c_k + a_k . u`` is affine). Split
the signomial into its positive and negative posynomial parts:

    Pplus(u)  = sum_{k: sigma_k = +1} m_k(u)     # convex (sum of convex)
    Pminus(u) = sum_{k: sigma_k = -1} m_k(u)     # convex (sum of convex)

so that

    s = Pplus - Pminus

is a *difference of convex* (DC) function of ``u``.

Affine overestimator SEC[P]
---------------------------
Let ``P`` be convex on the ``u``-box ``B = [u_lb, u_ub]``. We use the simplest
valid affine overestimator of ``P`` on the box: the *constant* corner-maximum

    SEC[P](u) = max_{v in corners(B)} P(v),

where ``corners(B) = {u_lb, u_ub}^n`` is the set of ``2^n`` box vertices. A
constant is affine, and by convexity it dominates ``P`` everywhere on ``B`` (see
proof below).

Relaxation (in ``u``-space)
---------------------------
    cv(u) = Pplus(u)  - SEC[Pminus]        # convex,  cv <= s
    cc(u) = SEC[Pplus] - Pminus(u)         # concave, cc >= s

Proof of validity
-----------------
*Corner-max is a valid affine overestimator.* A convex function on a polytope
attains its maximum at an extreme point (vertex); for the box ``B`` the extreme
points are exactly its ``2^n`` corners. Hence for all ``u in B``

    P(u) <= max_{v in corners(B)} P(v) = SEC[P](u),

and ``SEC[P]`` is a constant, hence affine.

*Soundness of cv.* Since ``SEC[Pminus] >= Pminus`` on ``B``,

    cv(u) = Pplus(u) - SEC[Pminus] <= Pplus(u) - Pminus(u) = s(u).

``cv`` is convex (a convex function ``Pplus`` minus a constant).

*Soundness of cc.* Since ``SEC[Pplus] >= Pplus`` on ``B``,

    cc(u) = SEC[Pplus] - Pminus(u) >= Pplus(u) - Pminus(u) = s(u).

``cc`` is concave (a constant minus a convex function ``Pminus``).

Thus ``cv(u) <= s(u) <= cc(u)`` for every ``u in B``, ``cv`` is convex and ``cc``
is concave, which is exactly the required certified DC relaxation.

References
----------
* Lundell, A. & Westerlund, T. Signomial global optimization (SGO):
  convexification of signomial terms by the difference-of-convex / power
  transformations underlying the alpha-SGO framework.
* Khajavirad, A., Michalek, J. J. & Sahinidis, N. V. (2012). Relaxations of
  factorable functions with convex-transformable intermediates.
"""

from __future__ import annotations

import itertools

import jax.numpy as jnp


def _posynomial_parts(u, terms):
    """Evaluate the positive and negative posynomial parts at ``u``.

    Parameters
    ----------
    u : array of shape (n,)
        Lifted point ``u_j = log x_j``.
    terms : list of (sigma, log_c, exps)
        ``sigma in {+1.0, -1.0}``, ``log_c = log(c_k)``, ``exps`` shape ``(n,)``.

    Returns
    -------
    (Pplus, Pminus) : scalars
        ``Pplus`` is the sum of monomials with ``sigma = +1`` and ``Pminus`` is
        the sum of monomials with ``sigma = -1`` (both stored with a +sign).
    """
    u = jnp.asarray(u)
    p_plus = jnp.asarray(0.0)
    p_minus = jnp.asarray(0.0)
    for sigma, log_c, exps in terms:
        exps = jnp.asarray(exps)
        m_k = jnp.exp(jnp.asarray(log_c) + jnp.dot(exps, u))
        p_plus = jnp.where(jnp.asarray(sigma) > 0.0, p_plus + m_k, p_plus)
        p_minus = jnp.where(jnp.asarray(sigma) > 0.0, p_minus, p_minus + m_k)
    return p_plus, p_minus


def _corner_maxima(terms, u_lb, u_ub):
    """Constant corner-maxima ``SEC[Pplus]`` and ``SEC[Pminus]`` over the box.

    A convex function on a box attains its maximum at a vertex, so enumerating
    the ``2^n`` corners of ``[u_lb, u_ub]`` and taking the maximum of each
    posynomial part yields a valid (constant, hence affine) overestimator.
    """
    u_lb = jnp.asarray(u_lb)
    u_ub = jnp.asarray(u_ub)
    n = u_lb.shape[0]
    sec_plus = -jnp.inf
    sec_minus = -jnp.inf
    for mask in itertools.product((0, 1), repeat=n):
        corner = jnp.where(jnp.asarray(mask, dtype=bool), u_ub, u_lb)
        p_plus, p_minus = _posynomial_parts(corner, terms)
        sec_plus = jnp.maximum(sec_plus, p_plus)
        sec_minus = jnp.maximum(sec_minus, p_minus)
    return sec_plus, sec_minus


def signed_signomial_dc_envelope(u, terms, u_lb, u_ub):
    """Certified DC convex/concave envelope of a signed signomial in log domain.

    Computes the difference-of-convex relaxation described in the module
    docstring at the lifted point ``u`` over the box ``[u_lb, u_ub]``:

        cv(u) = Pplus(u)  - max_corner Pminus     # convex,  cv <= s
        cc(u) = max_corner Pplus - Pminus(u)      # concave, cc >= s

    Parameters
    ----------
    u : array of shape (n,)
        Lifted point ``u_j = log x_j``.
    terms : list of (sigma, log_c, exps)
        Signomial terms; ``sigma in {+1.0, -1.0}``, ``log_c = log(c_k)`` (float),
        ``exps`` an array ``a_k`` of shape ``(n,)``.
    u_lb, u_ub : arrays of shape (n,)
        Lower/upper bounds of the ``u``-box.

    Returns
    -------
    (cv, cc) : scalars
        Convex underestimator ``cv`` and concave overestimator ``cc`` of
        ``s(u) = Pplus(u) - Pminus(u)`` at ``u``, satisfying
        ``cv(u) <= s(u) <= cc(u)`` on the box.
    """
    p_plus, p_minus = _posynomial_parts(u, terms)
    sec_plus, sec_minus = _corner_maxima(terms, u_lb, u_ub)
    cv = p_plus - sec_minus
    cc = sec_plus - p_minus
    return cv, cc
