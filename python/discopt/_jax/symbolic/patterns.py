"""Catalog of certified relaxation / structured-cut patterns.

Each pattern is a recognizable structural template with a sound relaxation or cut
and an analytic correctness proof (see ``design/relaxation-patterns.md`` for the
full catalog with fields and citations). The generators here are pure JAX /
numeric; the :data:`PATTERNS` registry records the metadata so a recognizer can
enumerate them.

Soundness convention: a *relaxation* of ``f`` over a box returns ``(cv, cc)`` with
``cv <= f <= cc``, ``cv`` convex and ``cc`` concave; a *cut* returns a valid
equality/inequality implied by the constraints. Every generator below is
certified by sampling in ``test_patterns.py`` in addition to the proof in its
docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp

from discopt._jax.symbolic import runtime

# ---------------------------------------------------------------------------
# P3. Bilinear x*y — McCormick / Al-Khayyal-Falk convex hull
# ---------------------------------------------------------------------------


def bilinear_envelope(x, y, x_lb, x_ub, y_lb, y_ub):
    """Convex/concave hull of ``x*y`` over the box. Returns ``(cv, cc)``.

    Proof of validity (bound-factor products, each ``>= 0``):

    * ``(x - x_lb)(y - y_lb) >= 0  =>  xy >= x_lb*y + x*y_lb - x_lb*y_lb``
    * ``(x - x_ub)(y - y_ub) >= 0  =>  xy >= x_ub*y + x*y_ub - x_ub*y_ub``
    * ``(x_ub - x)(y - y_lb) >= 0  =>  xy <= x_ub*y + x*y_lb - x_ub*y_lb``
    * ``(x - x_lb)(y_ub - y) >= 0  =>  xy <= x_lb*y + x*y_ub - x_lb*y_ub``

    ``cv`` is a max of affine functions (convex), ``cc`` a min of affine functions
    (concave). These four facets are the convex hull of ``{(x,y,xy)}`` over the
    box (Al-Khayyal & Falk, 1983).
    """
    cv = jnp.maximum(x_lb * y + x * y_lb - x_lb * y_lb, x_ub * y + x * y_ub - x_ub * y_ub)
    cc = jnp.minimum(x_ub * y + x * y_lb - x_ub * y_lb, x_lb * y + x * y_ub - x_lb * y_ub)
    return cv, cc


# ---------------------------------------------------------------------------
# P4. Reciprocal 1/y and linear-fractional x/y  (y > 0)
# ---------------------------------------------------------------------------


def reciprocal_envelope(y, y_lb, y_ub):
    """Relaxation of ``1/y`` on ``[y_lb, y_ub]`` with ``y > 0``. Returns ``(cv, cc)``.

    Proof: ``(1/y)'' = 2/y^3 > 0`` for ``y > 0`` so ``1/y`` is convex; the convex
    underestimator is the function itself and the concave overestimator is the
    secant (a convex function lies below its chord).
    """
    f = lambda t: 1.0 / t  # noqa: E731
    return f(y), runtime.secant(f, y, y_lb, y_ub)


def linear_fractional_lifted(x, recip, x_lb, x_ub, y_lb, y_ub):
    """Relaxation of ``x/y`` via the lift ``recip = 1/y``. Returns ``(cv, cc)``.

    ``recip`` is the (separately relaxed) reciprocal variable; the value is the
    bilinear hull (P3) of ``x*recip`` over ``recip in [1/y_ub, 1/y_lb]``.

    Proof: ``x/y = x * (1/y)``. Relaxing ``1/y`` by :func:`reciprocal_envelope`
    and lifting it to ``recip``, the bilinear hull of ``x*recip`` over the
    reciprocal's range is valid for every admissible ``recip`` (P3), hence valid
    for ``recip = 1/y``. The reciprocal's range follows from ``1/y`` being
    monotone decreasing: ``1/y in [1/y_ub, 1/y_lb]``.
    """
    return bilinear_envelope(x, recip, x_lb, x_ub, 1.0 / y_ub, 1.0 / y_lb)


# ---------------------------------------------------------------------------
# P6. RLT cut for sum-to-constant bilinears
# ---------------------------------------------------------------------------


def rlt_sum_constant_cut(products, x_i, total):
    """RLT equality cut from a conservation constraint ``sum_j x_j = total``.

    Args:
        products: sequence of the bilinear values ``w_{ij} = x_i * x_j`` for all
            ``j`` in the conservation set (including ``j = i``).
        x_i: the multiplier variable value.
        total: the conserved constant ``C``.

    Returns:
        ``(lhs, rhs)`` of the valid cut ``sum_j w_{ij} == total * x_i``.

    Proof: multiply the valid equality ``sum_j x_j - total = 0`` by ``x_i >= 0``:
    ``sum_j x_i x_j = total * x_i``, i.e. ``sum_j w_{ij} = total * x_i``. This
    RLT equality is implied by the constraints and is generally not implied by the
    per-term McCormick hulls, so it tightens the relaxation (Sherali & Adams,
    1990).
    """
    lhs = jnp.sum(jnp.asarray(products), axis=0)
    return lhs, total * x_i


# ---------------------------------------------------------------------------
# P7. Posynomial / signomial monomial  c * prod_j x_j^{a_j}  (x_j > 0)
# ---------------------------------------------------------------------------


def posynomial_logconvex(log_coeff, exps, logs):
    """Log-domain convex value of a monomial ``c * prod_j x_j^{a_j}``.

    With ``u_j = log x_j`` and ``log_coeff = log c``, the monomial equals
    ``exp(log_coeff + sum_j a_j u_j)``, which is **convex in ``u``**.

    Args:
        log_coeff: ``log c`` (``c > 0``).
        exps: exponents ``a_j``.
        logs: ``u_j = log x_j``.

    Returns:
        The monomial value ``exp(log_coeff + a . u)`` (= ``c * prod x_j^{a_j}``).

    Proof: ``exp`` is convex and increasing and ``log_coeff + a . u`` is affine in
    ``u``; the composition of a convex increasing function with an affine map is
    convex (Boyd & Vandenberghe, §3.2.4). A posynomial (a sum of such monomials
    with positive coefficients) is therefore convex in ``u`` — the basis of
    geometric-programming convexification (Boyd et al., 2007).
    """
    return jnp.exp(log_coeff + jnp.dot(jnp.asarray(exps), jnp.asarray(logs)))


# ---------------------------------------------------------------------------
# Pattern registry (metadata for enumeration / documentation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pattern:
    """Metadata for a recognizable relaxation/cut pattern."""

    name: str
    fields: tuple
    template: str
    citation: str
    status: str  # "done" | "engine" | "roadmap"
    generator: Optional[Callable] = None


PATTERNS: dict[str, Pattern] = {
    "convex_concave_atom": Pattern(
        "convex_concave_atom",
        ("general NLP/MINLP",),
        "g(x) with g'' of one sign on the box",
        "McCormick (1976)",
        "engine",
    ),
    "single_inflection_atom": Pattern(
        "single_inflection_atom",
        ("gas friction f|f|", "kinetics", "ML activations", "odd powers"),
        "g(x) with one inflection (concave-convex or convex-concave)",
        "Tawarmalani & Sahinidis (2002)",
        "done",
        runtime.single_inflection_envelope,
    ),
    "bilinear": Pattern(
        "bilinear",
        ("pooling/blending", "robust optimization", "portfolio", "process design"),
        "x * y over a box",
        "Al-Khayyal & Falk (1983)",
        "done",
        bilinear_envelope,
    ),
    "linear_fractional": Pattern(
        "linear_fractional",
        ("efficiency/DEA", "fractional programming", "blending ratios"),
        "x / y with y > 0",
        "Tawarmalani & Sahinidis (2001)",
        "done",
        linear_fractional_lifted,
    ),
    "square_diff_network": Pattern(
        "square_diff_network",
        ("gas networks", "water networks", "district heating"),
        "chain of f^2 = C(p_in^2 - p_out^2) + ratio p_out = r p_in",
        "Sherali & Adams (1990)",
        "done",
        None,  # see symbolic.cut_recognizer
    ),
    "rlt_sum_constant": Pattern(
        "rlt_sum_constant",
        ("pooling/blending", "refinery", "supply chain", "transportation"),
        "conservation sum_j x_j = C with bilinears x_i x_j",
        "Sherali & Adams (1990)",
        "done",
        rlt_sum_constant_cut,
    ),
    "posynomial": Pattern(
        "posynomial",
        ("geometric programming", "circuit design", "structural design", "chem design"),
        "c * prod_j x_j^{a_j} with x_j > 0",
        "Boyd et al. (2007)",
        "done",
        posynomial_logconvex,
    ),
    "ac_power_qc": Pattern(
        "ac_power_qc",
        ("AC optimal power flow", "state estimation"),
        "V_i V_j cos(theta_ij) / sin(theta_ij), |theta| < pi/2",
        "Coffrin, Hijazi & Van Hentenryck (2016)",
        "roadmap",
    ),
    "log_mean_lmtd": Pattern(
        "log_mean_lmtd",
        ("heat-exchanger networks", "process integration"),
        "(dT1 - dT2) / ln(dT1/dT2)",
        "Mistry & Misener (2016)",
        "roadmap",
    ),
}


def available(status: Optional[str] = None) -> list[str]:
    """Return pattern names, optionally filtered by status."""
    return sorted(n for n, p in PATTERNS.items() if status is None or p.status == status)
