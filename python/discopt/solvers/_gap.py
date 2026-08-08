"""Shared optimality-gap accounting for the decomposition solvers (issue #945).

``solvers.oa`` and ``solvers.gdpopt_loa`` each carried a near-copy of the same
six-line ``_compute_gap``, and the copies disagreed. This module is the single
definition both now call, so a fix to one is a fix to both.

Three things it gets right that the copies did not:

**1. The absolute criterion is discopt's absolute tolerance, not 1e-9.**
Both copies closed the gap only at ``ub - lb <= 1e-9``. That floor is tighter
than discopt's own ``1e-6`` absolute tolerance
(``validation.feasibility.ABS_TOL``, ``solvers.amp.solve(abs_tol=1e-6)``) and
tighter than the interior-point method can deliver, so on a problem whose optimum
sits at ~0 it was only ever beaten by an incumbent that was *not* feasible — see
(3). The dual test here (``abs_gap <= abs_tol`` OR ``rel_gap <= rel_tol``) is the
one ``amp.py`` has always used.

**2. A near-zero objective no longer inflates a rounding gap to 100%.**
``gdpopt_loa`` floored the relative denominator at ``1e-10``, so on ``min x``
with optimum 0 an honest incumbent of ``2.46e-9`` against a dual bound of ``0``
produced ``2.46e-9 / 2.46e-9 = 1.0`` — a 100% relative gap on a numerically exact
solve, which downgraded ``optimal`` to ``feasible``. The absolute criterion in
(1) now fires first and reports the gap closed. The floors themselves are left as
each solver had them (see ``denom_floor``): they set the *reporting* scale for an
unconverged run, and changing OA's is a separate, panel-worthy change.

**3. A dual bound ABOVE the incumbent is no longer clamped to "gap 0".**
Both copies computed ``abs_gap = max(0.0, ub - lb)``. When the incumbent sits
below the dual bound — the certificate invariant ``bound <= incumbent`` for a
minimization, violated — that ``max`` turned a *negative* gap into ``0.0`` and
the solver reported ``optimal``. That is not a rounding nicety: before #945 the
MindtPy constraint-qualification fixture (true optimum exactly 3.0) certified
``optimal`` at objective ``2.9999000025`` — super-optimal by ``1e-4`` — with a
reported dual bound of ``2.99995``, i.e. ``5e-5`` ABOVE its own incumbent, and a
reported gap of ``0.0``. Rounding-scale inversions are still absorbed (an
IPM-converged pair genuinely lands either side of the optimum by ~1 ulp), on the
same tolerance ``amp._amp_abs_gap_with_bound_tolerance`` uses; anything larger is
now reported as *not closed* rather than as a certificate.
"""

from __future__ import annotations

# discopt's absolute optimality tolerance. Same value as
# ``validation.feasibility.ABS_TOL`` and ``amp.solve``'s ``abs_tol`` default.
GAP_ABS_TOL = 1e-6

# The ±1e20 sentinel the solvers use for "no bound yet" (never ``inf``).
BOUND_INF = 1e19


def bound_inversion_tolerance(lb: float, ub: float, abs_tol: float = GAP_ABS_TOL) -> float:
    """How far ``lb`` may exceed ``ub`` before the ordering is *materially* wrong.

    Mirrors :func:`discopt.solvers.amp._amp_abs_gap_with_bound_tolerance` so the
    three solvers agree on what counts as rounding. Scale-aware, because at
    ``|obj| ~ 1e6`` an inversion of ``1e-9`` is a single ulp.
    """
    scale = max(1.0, abs(lb), abs(ub))
    return max(10.0 * abs_tol, 1e-8 * scale)


def optimality_gap(
    lb: float,
    ub: float,
    *,
    abs_tol: float = GAP_ABS_TOL,
    denom_floor: float = 1.0,
) -> float:
    """Relative optimality gap between dual bound ``lb`` and incumbent ``ub``.

    Returns exactly ``0.0`` when the gap is closed and ``1.0`` when nothing is
    known — including when the bound ordering is materially invalid, which is a
    broken certificate and must never read as "converged".

    Args:
        lb: Dual (lower) bound, in minimization sense.
        ub: Incumbent (upper) bound, in minimization sense.
        abs_tol: Absolute gap at or below which the gap counts as closed.
        denom_floor: Floor on the relative denominator, i.e. the objective scale
            below which the *reported* gap becomes the absolute gap. ``oa`` uses
            ``1.0`` (report the absolute gap for sub-unit objectives); the
            stricter ``gdpopt_loa`` uses ``1e-10`` (report the gap relative to the
            objective however small it is). Both are only reached once the
            absolute criterion above has declined to close the gap.
    """
    if ub >= BOUND_INF or lb <= -BOUND_INF:
        return 1.0

    raw_gap = ub - lb
    if raw_gap < -bound_inversion_tolerance(lb, ub, abs_tol):
        # The dual bound is above the incumbent by more than rounding. One of the
        # two is wrong; clamping to 0.0 here is what let a super-optimal incumbent
        # be certified (#945). Report "nothing proved" and let the caller's
        # certification logic decline to say `optimal`.
        return 1.0

    abs_gap = max(0.0, raw_gap)
    if abs_gap <= abs_tol:
        return 0.0
    return abs_gap / max(abs(ub), abs(lb), denom_floor)
