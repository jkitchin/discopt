"""Perspective strengthening of lifted univariate squares (#1064).

A convex MIQP whose continuous variables are *switched off* by binaries carries
structure the plain convex relaxation throws away. On the MINLPLib ``squfl``
family (separable quadratic uncapacitated facility location) every continuous
``x`` is tied to a binary ``y`` by a variable-upper-bound row ``x - u*y <= 0``,
so ``y = 0`` forces ``x = 0``. Such an ``x`` is *semicontinuous*.

For a semicontinuous ``x`` the epigraph ``s >= x**2`` can be replaced by the
**perspective** ``s >= x**2 / y``, which is the convex hull of
``{(x, s, y) : y in {0,1}, x = 0 if y = 0, s >= x**2}``. Its linearizations are
the Frangioni-Gentile perspective cuts: for any ``z``,

    s >= 2*z*x - z**2 * y                                                  (P)

Validity is immediate on the two integral values of ``y``:

* ``y = 1``: (P) is ``s >= 2*z*x - z**2``, the ordinary tangent to ``x**2`` at
  ``z`` -- a global underestimator of a convex function.
* ``y = 0``: semicontinuity gives ``x = 0`` and hence ``s = x**2 = 0``, and (P)
  reduces to ``s >= 0``. True.

and (P) dominates the plain tangent ``s >= 2*z*x - z**2`` everywhere in
``y <= 1``, strictly wherever ``y < 1`` and ``z != 0``. Taking ``z = x_bar /
y_bar`` at the LP point makes (P) the supporting hyperplane of ``x**2/y`` there,
whose violation is ``x_bar**2 / y_bar - s_bar`` against the plain tangent's
``x_bar**2 - s_bar``.

This is what SCIP does automatically (Bestuzheva, Gleixner and Vigerske, "A
computational study of perspective cuts") and what MINLPLib's hand-written
``squfl*persp`` variants encode by hand; see ``docs/references.bib``.

Nothing here is keyed to a problem name or shape (CLAUDE.md §2): the detector
reads the relaxation's own rows.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import scipy.sparse as sp

__all__ = ["semicontinuous_indicators", "perspective_reference"]

#: A binary's LP value below which ``z = x_bar / y_bar`` is not formed. The row
#: ``x <= u*y`` bounds ``z`` by ``u``, so ``z`` cannot actually blow up -- but a
#: ratio of two values both at the edge of LP tolerance carries no information,
#: and the plain tangent is used instead.
_MIN_INDICATOR_VALUE = 1e-6


def _binary_columns(
    bounds: list[tuple[float, float]], integrality: Optional[np.ndarray]
) -> set[int]:
    """Columns that are integer with a node box inside ``[0, 1]``.

    ``integrality`` covers the ORIGINAL columns only; lifted aux columns sit past
    its end and are therefore never taken for binaries, which is correct -- an
    aux column is a function of the originals, not a switch.
    """
    if integrality is None:
        return set()
    flags = np.asarray(integrality).ravel()
    out: set[int] = set()
    for j, (lo, hi) in enumerate(bounds):
        if j >= flags.size or not int(flags[j]):
            continue
        if lo > -1e-9 and hi < 1.0 + 1e-9:
            out.add(j)
    return out


def semicontinuous_indicators(
    A_ub: Optional[Union[np.ndarray, "sp.spmatrix"]],
    b_ub: Optional[np.ndarray],
    bounds: list[tuple[float, float]],
    integrality: Optional[np.ndarray],
) -> dict[int, int]:
    """Map each semicontinuous column to a binary that switches it off.

    A two-term row ``a*x + b*y <= c`` with ``a > 0``, ``b < 0`` and ``c <= 0``
    gives ``x <= (c - b*y)/a``, so at ``y = 0`` it forces ``x <= c/a <= 0``;
    combined with a node lower bound ``x >= 0`` that is ``x = 0``. Every row of a
    valid relaxation holds at every feasible point of the original model, so the
    implication holds there too -- which is what a cut derived from it needs.

    Reading it off the relaxation's rows rather than the model's constraint
    objects means separated rows are eligible sources as well. That is sound for
    the same reason, and it costs one pass over the nonzeros.

    Returns ``{x_col: y_col}``. When several binaries switch the same ``x``, the
    first one found is kept -- each yields a valid cut, and one is enough to
    dominate the plain tangent.
    """
    if A_ub is None or b_ub is None:
        return {}
    binaries = _binary_columns(bounds, integrality)
    if not binaries:
        return {}
    A = sp.csr_matrix(A_ub)
    b = np.asarray(b_ub, dtype=np.float64).ravel()
    found: dict[int, int] = {}
    for r in range(A.shape[0]):
        lo, hi = int(A.indptr[r]), int(A.indptr[r + 1])
        if hi - lo != 2:
            continue
        if not np.isfinite(b[r]) or float(b[r]) > 1e-9:
            continue
        cols = A.indices[lo:hi]
        vals = A.data[lo:hi]
        for k in (0, 1):
            y_col, x_col = int(cols[k]), int(cols[1 - k])
            y_coef, x_coef = float(vals[k]), float(vals[1 - k])
            if y_col not in binaries or x_col in binaries:
                continue
            if x_coef <= 1e-12 or y_coef >= -1e-12:
                continue
            if x_col >= len(bounds) or bounds[x_col][0] < -1e-9:
                continue
            found.setdefault(x_col, y_col)
    return found


def perspective_reference(x0: float, y0: float) -> Optional[float]:
    """The ``z`` whose perspective cut supports ``x**2/y`` at ``(x0, y0)``.

    ``None`` when the indicator is too close to zero for the ratio to mean
    anything; the caller then falls back to the plain tangent, which is still
    sound (dropping a cut only loosens).
    """
    if not (np.isfinite(x0) and np.isfinite(y0)):
        return None
    if y0 < _MIN_INDICATOR_VALUE:
        return None
    # ``y`` never exceeds 1 in the original model, and a value above it in the LP
    # would make the cut weaker than the plain tangent rather than unsound; clamp
    # so the reference is the strongest valid one.
    return float(x0) / min(float(y0), 1.0)
