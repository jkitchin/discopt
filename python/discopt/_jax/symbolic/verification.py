"""Numerical certification of derived envelopes.

Soundness is the non-negotiable property of any relaxation: for every ``x`` in
the box ``[lb, ub]`` a valid envelope must satisfy ``cv(x) <= f(x) <= cc(x)``,
with ``cv`` convex and ``cc`` concave. :func:`verify_envelope` checks these
properties by dense sampling over randomized boxes (the same theorem-style
discipline used by ``python/tests/relaxation_harness.py``) and reports the
worst-case violations and the relaxation tightness (mean / max gap).

This is a *certification* harness, not a proof: it is meant to catch derivation
bugs before an atom is registered. A future phase will add SymPy-backed
symbolic proofs and outward-rounded interval evaluation
(:mod:`discopt._jax.convexity.interval`) for fully rigorous guarantees; the
sampling check is the gate that the existing test suite already trusts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class VerificationReport:
    """Result of certifying an envelope over a family of random boxes.

    Attributes:
        sound: ``True`` if no containment violation exceeded ``tol``.
        max_lower_violation: Worst ``cv(x) - f(x)`` (>0 means cv overshot f).
        max_upper_violation: Worst ``f(x) - cc(x)`` (>0 means cc undershot f).
        max_convexity_violation: Worst non-convexity of ``cv`` (>0 is bad).
        max_concavity_violation: Worst non-concavity of ``cc`` (>0 is bad).
        mean_gap: Mean ``cc - cv`` over all samples (tightness; smaller better).
        max_gap: Max ``cc - cv`` over all samples.
        n_boxes: Number of random boxes tested.
        n_points: Points sampled per box.
    """

    sound: bool
    max_lower_violation: float
    max_upper_violation: float
    max_convexity_violation: float
    max_concavity_violation: float
    mean_gap: float
    max_gap: float
    n_boxes: int
    n_points: int


def verify_envelope(
    relax_fn: Callable,
    f_numeric: Callable,
    *,
    domain: tuple[float, float],
    n_boxes: int = 200,
    n_points: int = 64,
    tol: float = 1e-7,
    min_width: float = 1e-3,
    seed: int = 0,
) -> VerificationReport:
    """Certify ``relax_fn`` against the true function ``f_numeric``.

    Random sub-boxes ``[lb, ub]`` are drawn from ``domain``; on each, ``n_points``
    interior samples check containment, and the curvature of ``cv``/``cc`` is
    checked via the midpoint (Jensen) inequality on random chords.

    Args:
        relax_fn: Closure ``(x, lb, ub) -> (cv, cc)`` (JAX).
        f_numeric: The true function ``f(x)`` (JAX-callable).
        domain: Outer domain ``(lo, hi)`` to sample boxes from.
        n_boxes: Number of random boxes.
        n_points: Samples per box.
        tol: Containment tolerance (absolute slack for float round-off).
        min_width: Minimum box width to avoid degenerate intervals.
        seed: RNG seed for reproducibility.

    Returns:
        A :class:`VerificationReport`.
    """
    lo, hi = domain
    rng = np.random.default_rng(seed)

    # Build random boxes within the domain.
    p = rng.uniform(lo, hi, size=(n_boxes, 2))
    lbs = np.minimum(p[:, 0], p[:, 1])
    ubs = np.maximum(p[:, 0], p[:, 1])
    widen = ubs - lbs < min_width
    ubs[widen] = np.minimum(lbs[widen] + min_width, hi)

    # Sample interior points per box (fractions in [0,1]).
    fracs = rng.uniform(0.0, 1.0, size=(n_boxes, n_points))

    relax_v = jax.vmap(jax.vmap(relax_fn, in_axes=(0, None, None)), in_axes=(0, 0, 0))
    f_v = jax.vmap(f_numeric)

    lb_j = jnp.asarray(lbs)
    ub_j = jnp.asarray(ubs)
    xs = lb_j[:, None] + fracs * (ub_j - lb_j)[:, None]  # (n_boxes, n_points)

    cv, cc = relax_v(xs, lb_j, ub_j)
    fx = f_v(xs.reshape(-1)).reshape(xs.shape)

    lower_viol = float(jnp.max(cv - fx))  # cv should be <= f
    upper_viol = float(jnp.max(fx - cc))  # cc should be >= f
    gaps = cc - cv
    mean_gap = float(jnp.mean(gaps))
    max_gap = float(jnp.max(gaps))

    # Curvature: cv convex  =>  cv(mid) <= 0.5(cv(x1)+cv(x2)) for random chords.
    # cc concave =>  cc(mid) >= 0.5(cc(x1)+cc(x2)).
    f1 = rng.uniform(0.0, 1.0, size=(n_boxes, n_points))
    f2 = rng.uniform(0.0, 1.0, size=(n_boxes, n_points))
    x1 = lb_j[:, None] + jnp.asarray(f1) * (ub_j - lb_j)[:, None]
    x2 = lb_j[:, None] + jnp.asarray(f2) * (ub_j - lb_j)[:, None]
    xm = 0.5 * (x1 + x2)
    cv1, cc1 = relax_v(x1, lb_j, ub_j)
    cv2, cc2 = relax_v(x2, lb_j, ub_j)
    cvm, ccm = relax_v(xm, lb_j, ub_j)
    convexity_viol = float(jnp.max(cvm - 0.5 * (cv1 + cv2)))  # should be <= 0
    concavity_viol = float(jnp.max(0.5 * (cc1 + cc2) - ccm))  # should be <= 0

    sound = (lower_viol <= tol) and (upper_viol <= tol)

    return VerificationReport(
        sound=sound,
        max_lower_violation=lower_viol,
        max_upper_violation=upper_viol,
        max_convexity_violation=convexity_viol,
        max_concavity_violation=concavity_viol,
        mean_gap=mean_gap,
        max_gap=max_gap,
        n_boxes=n_boxes,
        n_points=n_points,
    )
