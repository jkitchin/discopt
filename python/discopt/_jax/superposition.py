"""Superposition relaxations for bilinear-of-nonlinear terms (M8 of issue #81).

The univariate polyhedral outer-approximation wrapper (:mod:`polyhedral_oa`,
M11) cannot demonstrate *strict* gap closure from combining relaxation
hierarchies: for a single-variable atom ``y = f(x)`` the convex hull is exactly
achievable, so whichever arithmetic gets closer to it dominates the other in
every direction. A union of cuts is then never strictly tighter than the best
single component. This is a structural obstruction, not a numerical accident.

Strict superposition gains require a **genuinely multivariate** term where no
single arithmetic reaches the convex hull and the components are *complementary*
— each tight in a region where the other is loose. The canonical case is the
**bilinear-of-nonlinear** product

    w = f(x) * y ,      x in [xL, xU],  y in [yL, yU]

handled by discopt's compositional McCormick path as ``u = f(x)`` (a lifted
convex/concave atom) followed by the bilinear envelope of ``u * y``. That
compositional relaxation is *exact on the box boundary* but loose in the
interior. A polynomial (Taylor/Chebyshev) model of the same product is tight
near the box centre and loose toward the boundary. The two are complementary;
their intersection is strictly tighter than either alone.

This module produces **rigorous linear cuts** that realise that intersection in
a single LP. Every cut is a supporting hyperplane of the lifted graph
``w = f(x) y`` built at a *reference point* ``(xr, yr)``:

    tangent plane   T(x, y) = f(xr) yr + f'(xr) yr (x - xr) + f(xr) (y - yr)
    concavity defect  delta  = sup / inf over the box of  T(x, y) - f(x) y
    valid cut        w >= T(x, y) - delta_under     (delta_under = max(0, sup))
                     w <= T(x, y) - delta_over      (delta_over  = min(0, inf))

Because ``T - f(x) y`` is linear in ``y`` for fixed ``x``, its box extrema occur
at ``y in {yL, yU}``; the inner univariate range over ``x`` is computed with the
rigorous Chebyshev-model kernel (:mod:`chebyshev_model`), so the defect — and
hence the cut — is provably valid on the whole box, not merely at samples.

References placed at the four box **corners** give a boundary-tight,
McCormick-like relaxation (the corner under/over defects vanish, so each corner
cut touches the graph there). Adding **interior** references yields cuts the
corner family does not have, closing the interior gap. The combined family is
the "superposition" relaxation: sound by construction (each cut is individually
valid) and never looser than the corner family (it is a superset of its cuts).

This module's cuts are designed to be *added to* discopt's existing
compositional-McCormick LP relaxation (which lifts ``u = f(x)`` then takes the
``u * y`` bilinear envelope). The strict tightening over that genuine McCormick
baseline is exercised by the solver-integration tests and the regression
instance; the standalone tests here verify soundness and that interior
references strictly tighten the corner-only family.

See issue #81 (M8) and #51 for the convexification roadmap context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import numpy as np

from discopt._jax import chebyshev_model as cm
from discopt._jax.cutting_planes import LinearCut

# Local column convention for the returned cuts: (x, y, w).
_X_COL, _Y_COL, _W_COL = 0, 1, 2

# Outward-rounding factor applied to defects so floating-point error in the
# Chebyshev range never makes a cut shave the true surface.
_DEFECT_SAFETY = 1e-9


@dataclass(frozen=True)
class BilinearNonlinearTerm:
    """A term ``w = f(x) * y`` over a box, with its Chebyshev model of ``f``.

    Attributes:
        f: jax-traceable univariate callable ``f(x)``.
        x_box: ``(xL, xU)`` with ``xL < xU``.
        y_box: ``(yL, yU)`` with ``yL < yU``.
        degree: Chebyshev degree used to enclose ``f`` on ``x_box``.
    """

    f: Callable[[float], float]
    x_box: tuple[float, float]
    y_box: tuple[float, float]
    degree: int = 10


def mccormick_references(
    x_box: tuple[float, float], y_box: tuple[float, float]
) -> list[tuple[float, float]]:
    """The four box corners — reference points that reproduce McCormick.

    A cut built at a corner has zero concavity defect, so the corner family is
    exactly the standard McCormick bilinear envelope of ``f(x) * y`` (with the
    convex/concave envelope of ``f`` baked into the tangent slope).
    """
    xL, xU = x_box
    yL, yU = y_box
    return [(xL, yL), (xL, yU), (xU, yL), (xU, yU)]


def interior_references(
    x_box: tuple[float, float],
    y_box: tuple[float, float],
    nx: int = 3,
    ny: int = 3,
) -> list[tuple[float, float]]:
    """An ``nx x ny`` grid of strictly-interior reference points.

    These add cuts McCormick lacks, tightening the interior of the box where the
    compositional bilinear envelope is loose.
    """
    if nx < 1 or ny < 1:
        return []
    xL, xU = x_box
    yL, yU = y_box
    xs = (xL + (xU - xL) * (i + 1) / (nx + 1) for i in range(nx))
    ys = [yL + (yU - yL) * (j + 1) / (ny + 1) for j in range(ny)]
    return [(float(x), float(y)) for x in xs for y in ys]


def superposition_references(
    x_box: tuple[float, float],
    y_box: tuple[float, float],
    nx: int = 3,
    ny: int = 3,
) -> list[tuple[float, float]]:
    """Corners (McCormick) plus an interior grid — the superposition family."""
    return mccormick_references(x_box, y_box) + interior_references(x_box, y_box, nx, ny)


def _defects(
    term: BilinearNonlinearTerm,
    fm: cm.ChebyshevModel,
    xc: cm.ChebyshevModel,
    ax: float,
    ay: float,
    c: float,
) -> tuple[float, float]:
    """Rigorous ``(delta_under, delta_over)`` for the plane ``ax*x + ay*y + c``.

    ``delta_under = max(0, sup_box (plane - f(x) y))`` and
    ``delta_over  = min(0, inf_box (plane - f(x) y))``. The map is linear in
    ``y`` (fixed ``x``), so box extrema are at ``y in {yL, yU}``; the inner range
    over ``x`` comes from the Chebyshev kernel and is rigorous.
    """
    yL, yU = term.y_box
    sup_d = -np.inf
    inf_d = np.inf
    for yv in (yL, yU):
        # h(x) = ax*x + (ay*yv + c) - yv*f(x)
        h = cm.scalar_add(ay * yv + c, cm.sub(cm.scalar_mul(ax, xc), cm.scalar_mul(yv, fm)))
        lo, hi = h.bounds()
        sup_d = max(sup_d, hi)
        inf_d = min(inf_d, lo)
    delta_under = max(0.0, sup_d)
    delta_over = min(0.0, inf_d)
    return delta_under, delta_over


def bilinear_nonlinear_cuts(
    term: BilinearNonlinearTerm,
    references: list[tuple[float, float]],
) -> list[LinearCut]:
    """Rigorous linear under/over cuts for ``w = f(x) * y`` over ``(x, y, w)``.

    Each reference ``(xr, yr)`` contributes one underestimator and one
    overestimator :class:`LinearCut` (length-3 ``coeffs`` over ``(x, y, w)``),
    each a globally valid supporting hyperplane on the box. Corner references
    reproduce McCormick; interior references tighten the interior.
    """
    f = term.f
    grad_f = jax.grad(lambda t: f(t))
    xc = cm.from_variable(term.x_box, term.degree)
    fm = cm.compose_unary(f, xc)

    cuts: list[LinearCut] = []
    for xr, yr in references:
        fr = float(f(xr))
        fpr = float(grad_f(xr))
        # Tangent plane T(x,y) = ax*x + ay*y + c.
        ax = fpr * yr
        ay = fr
        c = -fpr * yr * xr
        delta_under, delta_over = _defects(term, fm, xc, ax, ay, c)
        # Outward rounding so rounding error can only loosen, never invalidate.
        delta_under += _DEFECT_SAFETY * (1.0 + abs(delta_under))
        delta_over -= _DEFECT_SAFETY * (1.0 + abs(delta_over))
        coeffs = np.array([-ax, -ay, 1.0], dtype=np.float64)
        # w >= ax*x + ay*y + (c - delta_under)
        cuts.append(LinearCut(coeffs=coeffs.copy(), rhs=float(c - delta_under), sense=">="))
        # w <= ax*x + ay*y + (c - delta_over)
        cuts.append(LinearCut(coeffs=coeffs.copy(), rhs=float(c - delta_over), sense="<="))
    return cuts


def remap_cut(cut: LinearCut, n_total: int, x_col: int, y_col: int, w_col: int) -> LinearCut:
    """Re-express a local ``(x, y, w)`` cut over a full ``n_total``-column LP row."""
    row = np.zeros(n_total, dtype=np.float64)
    row[x_col] = cut.coeffs[_X_COL]
    row[y_col] = cut.coeffs[_Y_COL]
    row[w_col] = cut.coeffs[_W_COL]
    return LinearCut(coeffs=row, rhs=cut.rhs, sense=cut.sense)


__all__ = [
    "BilinearNonlinearTerm",
    "bilinear_nonlinear_cuts",
    "interior_references",
    "mccormick_references",
    "remap_cut",
    "superposition_references",
]
