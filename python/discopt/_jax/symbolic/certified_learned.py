"""Certify guaranteed-convex ML networks as sound relaxations (Phase 8 prototype).

discopt already ships ICNN-based learned relaxations
(:mod:`discopt._jax.learned_relaxations`): a pair of Input Convex Neural Networks
produces a convex ``cv`` and a concave ``cc`` *by construction*. The missing
ingredient for a **valid relaxation** is the *bound* — convexity does not imply
``cv(x) <= f(x)``.

The existing wrapper closes that gap by clamping against the true value
(``cv = min(cv_pred, f)``). That is pointwise-bounding but **not convex**:
``min`` of a convex network and a non-convex ``f`` re-introduces non-convexity,
so the clamped output is not a sound *convex* under-estimator for a
lower-bounding subproblem.

This module provides the structure-preserving alternative: a **constant certified
margin**. For a given box the margin is constant in ``x``, so shifting the convex
prediction down by it,

    cv_cert(x) = cv_pred(x) - delta_lo ,   cc_cert(x) = cc_pred(x) + delta_hi ,

keeps ``cv_cert`` convex and ``cc_cert`` concave while restoring soundness. The
margins are estimated by the same randomized-box certification used for the
symbolic engine (:func:`discopt._jax.symbolic.verify_envelope`).

.. note::
   Sampling-based margins are certified *over the sampled boxes* with a safety
   factor — a bug-catching gate, not a proof. A rigorous margin uses the
   outward-rounded interval arithmetic in :mod:`discopt._jax.convexity.interval`
   plus a Lipschitz bound; that is the planned refinement.

This module imports SymPy-free; it only depends on JAX and (for the adapter)
the optional ``equinox`` learned-relaxation module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp

from discopt._jax.symbolic.verification import (
    VerificationReport,
    verify_envelope,
)


def raw_learned_relax_fn(learned) -> Callable:
    """Adapt a :class:`LearnedRelaxation` to the *unclamped* ``(x, lb, ub)`` form.

    Reproduces the network's convex/concave predictions **without** the runtime
    ``min``/``max`` soundness clamp, so the raw (guaranteed-convex but possibly
    unsound) output can be studied and certified. Univariate ops only.

    Args:
        learned: A ``discopt._jax.learned_relaxations.LearnedRelaxation``.

    Returns:
        A closure ``(x, lb, ub) -> (cv_pred, cc_pred)``.
    """
    if learned.input_dim != 1:
        raise NotImplementedError("raw adapter currently supports univariate ops only")

    def fn(x, lb, ub):
        width = jnp.maximum(ub - lb, 1e-15)
        x_norm = (x - lb) / width
        features = jnp.stack([x_norm, width])
        cv_pred = learned.cv_net(features)
        cc_pred = -learned.cc_net(-features)  # concave via negated ICNN
        return cv_pred, cc_pred

    return fn


@dataclass(frozen=True)
class CertifiedRelaxation:
    """A constant-margin-certified relaxation and its before/after reports.

    Attributes:
        relax_fn: The certified closure ``(x, lb, ub) -> (cv, cc)``.
        lower_margin: Constant ``delta_lo`` subtracted from ``cv_pred``.
        upper_margin: Constant ``delta_hi`` added to ``cc_pred``.
        raw_report: Certification of the raw (unshifted) relaxation.
        certified_report: Certification of the shifted relaxation.
    """

    relax_fn: Callable
    lower_margin: float
    upper_margin: float
    raw_report: VerificationReport
    certified_report: VerificationReport


def certify_relaxation(
    raw_fn: Callable,
    f_numeric: Callable,
    *,
    domain: tuple[float, float],
    safety_factor: float = 1.5,
    n_boxes: int = 400,
    n_points: int = 64,
    seed: int = 0,
) -> CertifiedRelaxation:
    """Wrap a convex/concave relaxation with a constant certified margin.

    Estimates the worst-case containment violations of ``raw_fn`` over randomized
    boxes, then shifts the convex/concave predictions by those violations (times
    ``safety_factor``). Because the shift is constant in ``x`` for any box, the
    curvature of ``cv``/``cc`` is preserved — unlike pointwise clamping.

    Args:
        raw_fn: Unclamped closure ``(x, lb, ub) -> (cv_pred, cc_pred)`` with
            ``cv_pred`` convex and ``cc_pred`` concave (e.g. from
            :func:`raw_learned_relax_fn`).
        f_numeric: True function ``f(x)`` (JAX-callable).
        domain: Outer domain to sample boxes from.
        safety_factor: Multiplier on the estimated margins (headroom for unsampled
            points). Must be ``>= 1``.
        n_boxes, n_points, seed: Forwarded to certification sampling.

    Returns:
        A :class:`CertifiedRelaxation`.
    """
    raw_report = verify_envelope(
        raw_fn, f_numeric, domain=domain, n_boxes=n_boxes, n_points=n_points, seed=seed
    )
    delta_lo = max(0.0, raw_report.max_lower_violation) * safety_factor
    delta_hi = max(0.0, raw_report.max_upper_violation) * safety_factor

    def certified(x, lb, ub):
        cv_pred, cc_pred = raw_fn(x, lb, ub)
        return cv_pred - delta_lo, cc_pred + delta_hi

    certified_report = verify_envelope(
        certified,
        f_numeric,
        domain=domain,
        n_boxes=n_boxes,
        n_points=n_points,
        seed=seed + 1,  # fresh boxes to test generalization of the margin
    )
    return CertifiedRelaxation(
        relax_fn=certified,
        lower_margin=delta_lo,
        upper_margin=delta_hi,
        raw_report=raw_report,
        certified_report=certified_report,
    )
