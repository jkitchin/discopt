"""Quality indicators for Pareto front approximations.

Implements:

* :func:`hypervolume` -- exact 2-D and 3-D Lebesgue-measure dominated volume,
  Monte-Carlo estimate for ``k >= 4``. Pareto-compliant.
* :func:`igd` -- inverted generational distance against a reference front.
* :func:`spread` -- coefficient of variation of consecutive point distances
  in objective space (lower is more uniform).
* :func:`epsilon_indicator` -- additive epsilon indicator I_eps(A, B).

All indicators operate on :class:`~discopt.mo.pareto.ParetoFront` objects and
use the senses stored on the front to convert to the all-minimize convention
internally (output values are sense-invariant).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from discopt.mo.pareto import ParetoFront, filter_nondominated


def _to_min_form(front: ParetoFront) -> np.ndarray:
    """Return an ``(n, k)`` objective array in all-minimize convention."""
    objs = front.objectives()
    signs = front._senses_array()
    return np.asarray(signs * objs, dtype=np.float64)


def _flip_reference(reference: np.ndarray, senses: np.ndarray) -> np.ndarray:
    """Transform a reference point in original senses to the min convention."""
    return np.asarray(senses * reference, dtype=np.float64)


def _hv_2d(points: np.ndarray, reference: np.ndarray) -> float:
    """Exact 2-D hypervolume for minimization.

    Only strictly nondominated points are counted; any row of *points* that
    does not dominate the reference contributes zero.
    """
    mask = np.all(points < reference[None, :], axis=1)
    pts = points[mask]
    if pts.size == 0:
        return 0.0
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    kept: list[np.ndarray] = []
    best_y = np.inf
    for row in pts:
        if row[1] < best_y:
            kept.append(row)
            best_y = row[1]
    if not kept:
        return 0.0
    pts = np.vstack(kept)
    hv = 0.0
    for i, p in enumerate(pts):
        x_next = pts[i + 1, 0] if i + 1 < len(pts) else reference[0]
        hv += (x_next - p[0]) * (reference[1] - p[1])
    return float(hv)


def _hv_hso(points: np.ndarray, reference: np.ndarray) -> float:
    """Exact hypervolume via Hypervolume by Slicing Objectives (HSO).

    Works for any ``k >= 2`` by recursion on the last coordinate. Complexity
    ``O(n^{k-1} log n)``; fine for small ``k`` and ``n`` but quickly blows up
    past ``k = 4``.
    """
    k = points.shape[1]
    if k == 2:
        return _hv_2d(points, reference)

    # Filter points that dominate the reference.
    mask = np.all(points < reference[None, :], axis=1)
    pts = points[mask]
    if pts.size == 0:
        return 0.0

    # Sort ascending on the last coordinate.
    order = np.argsort(pts[:, -1])
    pts = pts[order]

    hv = 0.0
    prev_z = pts[0, -1]
    active: list[np.ndarray] = []
    # Sweep slices along the last axis.
    for i, p in enumerate(pts):
        # Close out the previous slice using the previously-active front.
        dz = p[-1] - prev_z
        if dz > 0 and active:
            slice_pts = np.vstack([a[:-1] for a in active])
            hv += dz * _hv_hso(slice_pts, reference[:-1])
        # Add the new point to the active set and prune dominated entries.
        active.append(p)
        active_arr = np.vstack(active)
        nd = filter_nondominated(active_arr[:, :-1])
        active = [active_arr[j] for j in range(len(active)) if nd[j]]
        prev_z = p[-1]

    # Final slice from last point up to the reference.
    dz = reference[-1] - prev_z
    if dz > 0 and active:
        slice_pts = np.vstack([a[:-1] for a in active])
        hv += dz * _hv_hso(slice_pts, reference[:-1])
    return float(hv)


def _hv_monte_carlo(
    points: np.ndarray,
    reference: np.ndarray,
    *,
    n_samples: int = 100_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Monte-Carlo hypervolume estimate for minimization.

    Samples uniformly in the bounding box ``[lb, reference]`` where
    ``lb = points.min(axis=0)``, counts the fraction dominated by at least
    one point of the front, and multiplies by the box volume.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    lb = points.min(axis=0)
    span = reference - lb
    if np.any(span <= 0):
        return 0.0
    box_vol = float(np.prod(span))
    samples = lb + rng.random((n_samples, points.shape[1])) * span
    # A sample is dominated if some point is componentwise <= sample and
    # strictly less in at least one component.
    dom = np.zeros(n_samples, dtype=bool)
    for p in points:
        dom |= np.all(samples >= p[None, :], axis=1)
    return box_vol * float(dom.mean())


def hypervolume(
    front: ParetoFront,
    reference: Optional[np.ndarray] = None,
    *,
    method: str = "auto",
    n_samples: int = 100_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Hypervolume dominated by *front* with respect to *reference*.

    Parameters
    ----------
    front : ParetoFront
        Front whose dominated hypervolume is computed.
    reference : numpy.ndarray, optional
        Reference point in the original senses of the front. If ``None``,
        uses the nadir (if available) or the per-objective worst value of
        the front, inflated by 1 % of the range, so every returned
        hypervolume is positive.
    method : {"auto", "exact", "mc"}, default "auto"
        ``"auto"`` uses exact HSO for ``k <= 3`` and Monte-Carlo otherwise.
    n_samples : int
        Monte-Carlo sample count when the MC branch is used.
    rng : numpy.random.Generator, optional
        Random generator used by the Monte-Carlo branch.

    Returns
    -------
    float
        Dominated hypervolume. Returns ``0.0`` if the front is empty or no
        point dominates the reference.
    """
    if front.n == 0:
        return 0.0
    senses = front._senses_array()
    pts_min = _to_min_form(front)

    if reference is None:
        if front.nadir is not None:
            ref_orig = front.nadir.copy()
        else:
            ref_orig = np.array(
                [
                    front.objectives()[:, j].max()
                    if senses[j] > 0
                    else front.objectives()[:, j].min()
                    for j in range(front.k)
                ],
                dtype=np.float64,
            )
        # Inflate by 1% of range in the unfavorable direction.
        obj_min = front.objectives().min(axis=0)
        obj_max = front.objectives().max(axis=0)
        margin = 0.01 * np.maximum(obj_max - obj_min, 1.0)
        ref_orig = ref_orig + senses * margin
    else:
        ref_orig = np.asarray(reference, dtype=np.float64)

    ref_min = _flip_reference(ref_orig, senses)

    if method == "exact" or (method == "auto" and front.k <= 3):
        return _hv_hso(pts_min, ref_min)
    return _hv_monte_carlo(pts_min, ref_min, n_samples=n_samples, rng=rng)


def igd(front: ParetoFront, reference_front: ParetoFront) -> float:
    """Inverted generational distance from *reference_front* to *front*.

    Both fronts must share the same senses order; objectives are compared
    in min-form, so the result is sense-invariant. Smaller is better.
    """
    if front.senses != reference_front.senses:
        raise ValueError("front and reference_front must have the same senses")
    if front.n == 0 or reference_front.n == 0:
        return float("inf")
    a = _to_min_form(front)
    r = _to_min_form(reference_front)
    # For each r_j, find nearest a_i by Euclidean distance.
    d = np.sqrt(((r[:, None, :] - a[None, :, :]) ** 2).sum(axis=-1))
    nearest = d.min(axis=1)
    return float(nearest.mean())


def spread(front: ParetoFront) -> float:
    """Coefficient-of-variation spread metric.

    For ``k = 2``, sort points by the first objective and compute the
    coefficient of variation (std / mean) of consecutive Euclidean
    distances. For higher ``k``, uses the coefficient of variation of
    nearest-neighbor distances. Lower values indicate more uniform spacing.
    Returns ``0.0`` if the front has fewer than two points.
    """
    if front.n < 2:
        return 0.0
    pts = _to_min_form(front)
    if front.k == 2:
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    else:
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=-1))
        np.fill_diagonal(d, np.inf)
        diffs = d.min(axis=1)
    mu = float(diffs.mean())
    if mu == 0.0:
        return 0.0
    return float(diffs.std() / mu)


def epsilon_indicator(front_a: ParetoFront, front_b: ParetoFront) -> float:
    """Additive epsilon indicator ``I_eps(A, B)``.

    The smallest ``eps`` such that every point in B is weakly dominated by
    some point in A shifted by ``eps`` (in min-form). Smaller is better;
    ``I_eps(A, B) <= 0`` means A already weakly dominates B. Sense-invariant.
    """
    if front_a.senses != front_b.senses:
        raise ValueError("fronts must have the same senses")
    if front_a.n == 0 or front_b.n == 0:
        return float("inf")
    a = _to_min_form(front_a)
    b = _to_min_form(front_b)
    # For each b in B, min over a in A of max over i of (a_i - b_i).
    diffs = a[:, None, :] - b[None, :, :]  # shape (|A|, |B|, k)
    per_pair = diffs.max(axis=-1)  # shape (|A|, |B|)
    per_b = per_pair.min(axis=0)  # shape (|B|,)
    return float(per_b.max())
