"""Numeric helper predicates shared by the relaxation layer.

Two different questions live here, and #1397 is the record of them being confused.
:data:`EFFECTIVE_INF` answers "does this *bound* carry any scale information?" --
the default box is +-9.999e19, so a magnitude at or above 1e19 is the absence of a
bound, not a large one. :func:`roundoff_slack` answers "how much of this *computed*
quantity could be the rounding error of the arithmetic that produced it?" -- where a
1e24 is a genuine 1e24 and must be counted. Do not use one predicate for the other's
question; see :func:`roundoff_slack`.
"""

from __future__ import annotations

import math

import numpy as np

EFFECTIVE_INF = 1e19

#: Unit round-off of IEEE double (``2**-52``).
FLOAT_EPS = float(np.finfo(np.float64).eps)


def is_effectively_finite(value: float) -> bool:
    """Return True when a bound is finite in the solver sense."""
    return bool(np.isfinite(value) and abs(float(value)) < EFFECTIVE_INF)


#: #1397: operation-count slack in the round-off bound below. A tightened bound is
#: the result of a short chain of multiplies, divides and a sum, so its error is a
#: few ulps of the largest operand; 8 covers the longest chain any caller here
#: builds. This is a *count*, not a tolerance: raising it cannot admit a
#: constraint violation, only defer an ambiguous one to the node NLP.
ROUNDOFF_OPS = 8.0


def roundoff_slack(*terms: float) -> float:
    """#1397: the floating-point round-off bound on a quantity built from *terms*.

    An absolute feasibility tolerance answers "is this violation inside the
    solver's feasibility tolerance?" — a scale-free question about the *model*, and the
    constant is right for it. It does not answer the second question every one of
    its call sites also asks: "is this crossover real, or is it the rounding error
    of the arithmetic that produced it?" That one is **not** scale-free. Summing or
    differencing floats of magnitude ``M`` carries an error of order ``n·u·M``, and
    at ``M = 1e10`` a single ulp is ``1.9e-6`` — already past the whole 1e-6
    tolerance. So on a large box an ordinary rounding crossover was read as a
    *proved* empty interval and the node was pruned, discarding a box that may hold
    the optimum.

    Measured before this helper existed (``scripts/audit_1397_empty_interval_roundoff.py``,
    graded against an ulp-exact oracle rather than against the solver): **11 of 36**
    crossovers of 1, 2 and 8 ulps were pruned, at every magnitude from 1e10 up —
    worst case a crossover of 0.70·u·M at ``M = 1e14``.

    The bound is **added** to the tolerance, never maxed with it: the two cover
    different errors and the guard must cover both. Maxing would discard whichever
    is smaller, which is the #1392 defect. With all-zero terms this returns
    ``0.0``, so an O(1) problem keeps exactly today's behaviour.

    Not a new invention: ``nonlinear_bound_tightening.SeparableQuadraticUpperBoundRule`` already
    builds this same ``(n + 4)·eps·Σ|terms|`` bound by hand as its ``float_err``,
    added after a ±9.999e19 box certified a feasible LP infeasible. This is that
    computation, named, justified once and shared.

    A non-finite term contributes ``0.0``. An ``inf`` slack would widen every guard
    without limit -- swallowing the genuine ``+inf > ub`` emptiness along with the
    artifact, and making a caller's outward widening unbounded -- so where a term
    carries no magnitude at all the conservative answer is today's behaviour.

    Large *finite* terms do count, sentinel-magnitude ones included. The house
    :func:`is_effectively_finite` predicate treats
    ``|v| >= 1e19`` as "no scale information", which is right for a *bound* and
    wrong here: this helper is also handed computed row quantities, where ``b*b``
    for ``|b| = 1e12`` is a genuine 1e24 and filtering it returns a slack of zero.
    Including them costs nothing, because every use of this bound moves a yardstick
    *outward* -- a wider empty-interval threshold prunes less, and a widened ``rhs``
    or square range only enlarges a box. An over-estimated slack therefore
    costs tightening strength; an under-estimated one costs soundness, which §1 does
    not trade.
    """
    return ROUNDOFF_OPS * FLOAT_EPS * float(sum(abs(t) for t in terms if math.isfinite(t)))


def roundoff_slack_arr(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Elementwise :func:`roundoff_slack` over a whole box.

    Same bound, same constant, evaluated per column so a box with one 1e14 variable
    and forty O(1) ones gets the wide slack only on the column that needs it.
    Non-finite magnitudes contribute ``0.0``, for the reason given in
    :func:`roundoff_slack`.
    """

    def _mag(a: np.ndarray) -> np.ndarray:
        absa = np.abs(np.asarray(a, dtype=np.float64))
        return np.where(np.isfinite(absa), absa, 0.0)

    out: np.ndarray = ROUNDOFF_OPS * FLOAT_EPS * (_mag(lo) + _mag(hi))
    return out
