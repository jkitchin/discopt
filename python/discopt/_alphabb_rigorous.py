"""Rigorous per-variable alphaBB parameters, with no JAX import.

Split out of ``_relax/alphabb`` because it never needed JAX: it works entirely from
``convexity.interval_ad``'s sound interval Hessian and numpy. While it lived in
that module, ``solver.py``'s per-node alphaBB bound imported it and thereby
dragged the whole JAX stack onto an otherwise JAX-free solve (#75) -- measured on
ex14_1_9, oaer and tspn08, the last three corpus instances still loading JAX
after Stages 2 and 3.

The rest of ``_relax/alphabb`` (the sampled ``estimate_alpha``, the
under/overestimators, and ``compile_alphabb_relaxation``) genuinely uses JAX and
stays there.
"""

from __future__ import annotations


def rigorous_alpha(expr, model, box=None):
    """Guaranteed per-variable alphaBB parameters for ``expr`` over its box.

    Uses a sound interval enclosure of the Hessian and a per-row interval
    Gershgorin bound. For row ``i``,

        lambda_min >= H[i,i].lo - sum_{j != i} max(|H[i,j].lo|, |H[i,j].hi|)

    is a valid lower bound on the smallest eigenvalue contribution, giving

        alpha_i = max(0, -0.5 * gershgorin_lo_i).

    Variables absent from ``expr`` (or appearing linearly) have a
    zero Hessian row, hence ``alpha_i = 0`` — the perturbation is applied
    only to the nonlinear variables, keeping the relaxation as tight as the
    diagonal-dominance bound allows.

    The row bounds come from ``convexity.eigenvalue.gershgorin_row_lower_bounds``
    (#1397). This function used to evaluate that formula itself with a plain
    round-to-nearest ``np.sum`` and no outward rounding, which made it *not*
    rigorous despite its name: rounding the off-diagonal radius down raises
    ``gershgorin_lo`` above its true value and leaves ``alpha`` below
    ``-lambda_min/2``, so the alphaBB body is nonconvex and its box minimum —
    used as this node's lower bound — can exceed the true minimum. That is a
    false dual bound, not a loose one. Measured by calling *this* function and
    grading its output against the same Gershgorin formula in exact rational
    arithmetic over the same float entries (every binary64 is a rational, so the
    oracle has no error of its own): **513 of 1120 rows** came back with
    ``alpha`` provably below ``-lambda_min/2``, worst shortfall 5.76e-4 — and at
    *every* scale from ``||A||_F = 1e0`` to 1e12, not only large ones, because the
    old code carried no margin at all. An absolute margin could not have fixed it
    either, the error being O(u*||A||); the correct, outward-rounded computation
    already existed one module away. Regression:
    ``python/tests/test_1397_alphabb_alpha_dominates_nonconvexity.py``.

    Args:
        expr: Scalar :class:`~discopt.modeling.core.Expression`.
        model: Model defining the flat variable layout.
        box: Optional ``{Variable: Interval}`` overriding declared bounds.

    Returns:
        ``np.ndarray`` of shape ``(n,)``. Entries are ``+inf`` wherever the
        interval Hessian abstained (unbounded), signalling that no useful
        alphaBB relaxation exists for this box.
    """
    import numpy as np

    from discopt._relax.convexity.eigenvalue import gershgorin_row_lower_bounds
    from discopt._relax.convexity.interval_ad import interval_hessian

    iad = interval_hessian(expr, model, box)
    # Rigorous, outward-rounded, and per-row: a row whose entries are unbounded
    # comes back as ``-inf``, which becomes ``alpha_i = +inf`` below and signals
    # that no useful alphaBB relaxation exists for that variable on this box.
    gershgorin_lo = gershgorin_row_lower_bounds(iad.hess)
    # ``0.5 *`` is exact in binary floating point, so halving the (already
    # outward-rounded) bound introduces no error of its own.
    return np.asarray(np.maximum(0.0, -0.5 * gershgorin_lo), dtype=np.float64)
