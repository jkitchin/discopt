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

    from discopt._relax.convexity.interval_ad import interval_hessian

    iad = interval_hessian(expr, model, box)
    h_lo = np.asarray(iad.hess.lo, dtype=float)
    h_hi = np.asarray(iad.hess.hi, dtype=float)
    abs_max = np.maximum(np.abs(h_lo), np.abs(h_hi))
    with np.errstate(invalid="ignore"):
        # Per-row off-diagonal radius = sum of |.| over the row minus the diagonal.
        # ``inf - inf`` on an abstaining (unbounded) row yields NaN, mapped to +inf
        # below; suppress the benign RuntimeWarning for the whole computation.
        row_radius = abs_max.sum(axis=1) - np.abs(np.diag(abs_max))
        gershgorin_lo = np.diag(h_lo) - row_radius
        alpha = np.maximum(0.0, -0.5 * gershgorin_lo)
    # NaN arises from inf - inf at abstaining nodes; treat as unbounded.
    alpha = np.where(np.isnan(alpha), np.inf, alpha)
    return alpha
