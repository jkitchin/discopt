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


def _secant_overestimators(u, terms, u_lb, u_ub):
    """Tighter *sloped-affine* overestimators ``SEC[Pplus]``, ``SEC[Pminus]``.

    This is the KMS 2012 §4 term-wise transforming-function overestimator
    (issue #181, item 4), strictly tighter than the constant corner-maximum.
    Each monomial in the log domain is ``m_k(u) = exp(xi_k)`` with the affine
    argument ``xi_k(u) = log_c_k + a_k . u`` ranging over
    ``[xi_lo_k, xi_hi_k]`` on the box. The chord (secant) of the convex
    ``exp`` over that interval,

        chord_k(u) = exp(xi_lo_k)
                     + (exp(xi_hi_k) - exp(xi_lo_k)) / (xi_hi_k - xi_lo_k)
                       * (xi_k(u) - xi_lo_k),

    dominates ``exp(xi_k)`` on ``[xi_lo_k, xi_hi_k]`` (chord above a convex
    function) and is **affine in ``u``**. Summing the chords of a posynomial
    part gives a valid affine overestimator of it. Because the chord touches
    ``exp`` at both endpoints and lies strictly below the constant endpoint
    maximum in the interior, this is a genuinely tighter (never looser)
    overestimator than :func:`_corner_maxima`.

    Returns ``(sec_plus, sec_minus)`` evaluated at ``u`` (sloped-affine values).
    """
    u = jnp.asarray(u)
    u_lb = jnp.asarray(u_lb)
    u_ub = jnp.asarray(u_ub)
    sec_plus = jnp.asarray(0.0)
    sec_minus = jnp.asarray(0.0)
    for sigma, log_c, exps in terms:
        exps = jnp.asarray(exps)
        log_c = jnp.asarray(log_c)
        # Range of the affine argument xi_k = log_c + a_k . u over the box:
        # min/max of a linear form are attained at the sign-matched corners.
        lin_lo = jnp.dot(jnp.where(exps >= 0.0, exps, 0.0), u_lb) + jnp.dot(
            jnp.where(exps < 0.0, exps, 0.0), u_ub
        )
        lin_hi = jnp.dot(jnp.where(exps >= 0.0, exps, 0.0), u_ub) + jnp.dot(
            jnp.where(exps < 0.0, exps, 0.0), u_lb
        )
        xi_lo = log_c + lin_lo
        xi_hi = log_c + lin_hi
        xi = log_c + jnp.dot(exps, u)
        e_lo = jnp.exp(xi_lo)
        e_hi = jnp.exp(xi_hi)
        width = xi_hi - xi_lo
        # Degenerate width (monomial constant over the box): chord is the
        # point value with zero slope. Guard the division so it stays finite.
        safe_width = jnp.where(width > 0.0, width, 1.0)
        slope = jnp.where(width > 0.0, (e_hi - e_lo) / safe_width, 0.0)
        chord_k = e_lo + slope * (xi - xi_lo)
        sec_plus = jnp.where(jnp.asarray(sigma) > 0.0, sec_plus + chord_k, sec_plus)
        sec_minus = jnp.where(jnp.asarray(sigma) > 0.0, sec_minus, sec_minus + chord_k)
    return sec_plus, sec_minus


def signed_signomial_dc_envelope(u, terms, u_lb, u_ub, *, overestimator="corner"):
    """Certified DC convex/concave envelope of a signed signomial in log domain.

    Computes the difference-of-convex relaxation described in the module
    docstring at the lifted point ``u`` over the box ``[u_lb, u_ub]``:

        cv(u) = Pplus(u)  - SEC[Pminus]     # convex,  cv <= s
        cc(u) = SEC[Pplus] - Pminus(u)      # concave, cc >= s

    Parameters
    ----------
    u : array of shape (n,)
        Lifted point ``u_j = log x_j``.
    terms : list of (sigma, log_c, exps)
        Signomial terms; ``sigma in {+1.0, -1.0}``, ``log_c = log(c_k)`` (float),
        ``exps`` an array ``a_k`` of shape ``(n,)``.
    u_lb, u_ub : arrays of shape (n,)
        Lower/upper bounds of the ``u``-box.
    overestimator : {"corner", "secant"}, optional
        Which affine overestimator ``SEC[P]`` to use. ``"corner"`` (default)
        is the constant corner-maximum :func:`_corner_maxima` — unchanged
        legacy behavior, bound-neutral for existing callers. ``"secant"`` is
        the tighter KMS 2012 §4 per-monomial sloped-affine chord
        :func:`_secant_overestimators` (issue #181, item 4). The secant
        option only ever *tightens* (``cv`` up, ``cc`` down); it is a
        bound-changing option and therefore opt-in, default-off, pending the
        CLAUDE.md §5 graduation panel before any default flip.

    Returns
    -------
    (cv, cc) : scalars
        Convex underestimator ``cv`` and concave overestimator ``cc`` of
        ``s(u) = Pplus(u) - Pminus(u)`` at ``u``, satisfying
        ``cv(u) <= s(u) <= cc(u)`` on the box.
    """
    p_plus, p_minus = _posynomial_parts(u, terms)
    if overestimator == "secant":
        sec_plus, sec_minus = _secant_overestimators(u, terms, u_lb, u_ub)
    elif overestimator == "corner":
        sec_plus, sec_minus = _corner_maxima(terms, u_lb, u_ub)
    else:
        raise ValueError(f"overestimator must be 'corner' or 'secant', got {overestimator!r}")
    cv = p_plus - sec_minus
    cc = sec_plus - p_minus
    return cv, cc
