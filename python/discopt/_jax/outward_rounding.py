"""Outward rounding for relaxation envelope rows and auxiliary bounds (issue #956).

The invariant every envelope generator must hold is

    for every ``x`` in the box, the point ``(x, f(x))`` satisfies every emitted
    row and lies inside the auxiliary column's own bounds.

In exact arithmetic the closed forms hold it by construction. In f64 they do not:
an envelope right-hand side is a *cancelling* combination of box-endpoint
quantities (``slope*t0 - f(t0)``) while the auxiliary bound is an independent
rounding of the same quantity, so at the box corner the two disagree by ~1 ulp of
the row's magnitude. Once that magnitude is large the disagreement is an absolute
residual no LP feasibility tolerance absorbs, the node LP has no feasible point at
all, and the simplex reports it cannot decide.

Measured on the real generators before the fix (``probe956_python_generators.py``),
sweeping boxes over six orders of magnitude and evaluating the exactly-feasible
point ``(x, f(x))``:

    monomial p=3    row 1.3e+5     affine square   row 6.0e+0
    bilinear        row 3.1e-2     monomial p=2    row 2.3e-2

So every envelope rhs is relaxed **outward** by an ulp-scaled guard computed from
the magnitudes of the terms that formed it, and every auxiliary bound is widened
outward by the matching guard. A flat epsilon cannot do this job — the magnitudes
span 1e-12 to 1e18.

Direction matters for soundness: the guard only ever *loosens* (a larger rhs
admits more points, a wider aux box admits more points), so it can never cut a
feasible point and never lift a bound above the true optimum. Its cost is
~3.6e-15 relative, below every tolerance the solver reasons with (abs 1e-6,
rel 1e-4, integrality 1e-5).

WHY THE FORMULAS TAKE THE ARGUMENTS THEY DO — this is load-bearing. Three engines
must emit *bit-identical* envelope rows:

* the cold build (``uniform_relax._emit_1d`` / ``_emit_mccormick``),
* the incremental patch (``incremental_mccormick._monomial_rows`` and friends),
* the native Rust node kernel (``bnb/mccormick_patch.rs``).

``IncrementalMcCormickLP._validate`` enforces the first two against each other by
comparing ``_rowset``, which rounds each rhs to 6 decimals ABSOLUTELY — so on a
large-magnitude row a guard applied to one engine and not the other, or computed
from different intermediates, drops the incremental fast path to ``ok=False``.
Hence the guard is a pure function of quantities all three engines already hold
(the box endpoints in ``t``-space, the function values there, the row's own
derivative and rhs) and is computed here, once, in a fixed operation order —
never re-derived at a call site.

The guard is **default ON** (graduated on the §5 panel, 2026-08-09);
``DISCOPT_ENVELOPE_OUTWARD_ROUND=0`` opts out and is bit-for-bit the pre-#956
arithmetic. See :func:`outward_rounding_enabled` for the measurement.
"""

from __future__ import annotations

import math
import os
import sys

__all__ = [
    "bounded_mag",
    "box_finite",
    "outward_slack",
    "widen",
    "envelope_1d_slack",
    "envelope_product_slack",
    "outward_rounding_enabled",
]

#: The LP layer's infinity sentinel (``lp/simplex/primal.rs::INF``). A bound at or
#: beyond it means "unbounded", not "this large" — it must never scale a guard
#: (``1e20 * _ULP_GUARD`` would be a 3.6e5 relaxation, not a rounding repair).
_INF_SENTINEL = 1e20

#: Guard size, in ulps of the row's term magnitude. Sized from measurement: the
#: worst violation of an exactly-feasible point across a ten-order-of-magnitude
#: sweep of all the generators was ~1.9 ulp; 16 ulp is that with an ~8x margin.
_ULP_GUARD = 16.0 * sys.float_info.epsilon

_ENABLED: bool | None = None


def outward_rounding_enabled() -> bool:
    """Whether the guard is applied. **Default ON**; ``=0`` opts out. Read once.

    Mirrors ``mccormick_patch.rs::outward_rounding_enabled`` exactly — the three
    engines must agree row for row, so they must agree on this too.

    It shipped default-OFF first, on a measurement that read as "cert-clean but
    harmful". Re-measured after the #956 T3′ fix, neither half survives: the
    regression signal sits inside this harness's measured noise floor (two builds
    running *identical* code produce win/loss splits up to 5/7), and the
    undecided-fraction evidence is no longer measurable at all (``n_undecided`` is
    0 across every corpus instance that reaches the native kernel, in both arms).
    What remains is a demonstrated soundness defect — the unguarded generators cut
    ``(x, f(x))`` out of their own envelope — repaired at no measurable cost, which
    CLAUDE.md §1 settles in favour of correctness. See
    ``docs/dev/issue-956-followthrough-plan.md`` §4 (T6/T7).

    With the guard off, ``outward_slack`` is identically 0.0 and ``widen`` is the
    identity, so the whole relaxation is bit-for-bit the pre-#956 arithmetic.
    """
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("DISCOPT_ENVELOPE_OUTWARD_ROUND", "1").strip() not in (
            "0",
            "false",
            "False",
        )
    return _ENABLED


def bounded_mag(v: float) -> float:
    """Magnitude of a quantity for guard sizing; non-finite contributes **zero**.

    Note this does NOT clamp at the infinity sentinel. A *derived* value may
    legitimately exceed it — ``x**3`` over a box reaching ``9e6`` encloses
    ``7.3e20``, whose ulp is ``1.3e5`` — and zeroing it left exactly that value
    unguarded, which is the defect #956 is about. The sentinel means "this BOX is
    unbounded", so it is tested on the box endpoints only, by :func:`box_finite`.
    """
    v = float(v)
    return abs(v) if math.isfinite(v) else 0.0


def box_finite(v: float) -> bool:
    """Whether a BOX endpoint is a real bound rather than the infinity sentinel.

    On an unbounded box a *relative* guard has no meaning and scaling by ``1e20``
    (or by an ``f`` image of it) would swamp the relaxation, so the guard there
    degrades to its absolute floor — exactly the pre-#956 behaviour, never a
    regression.
    """
    return math.isfinite(v) and abs(v) < _INF_SENTINEL


def outward_slack(mag_sum: float) -> float:
    """The outward slack for a row rhs / bound whose terms total ``mag_sum``.

    Always ``>= 0``, so adding it to a ``<=`` rhs (or to the high side of a bound,
    negated on the low side) can only ever relax. Floored at ``_ULP_GUARD``
    absolute so a row whose magnitudes are all sub-unit still gets its last bit
    repaired.
    """
    if not outward_rounding_enabled():
        return 0.0
    return _ULP_GUARD * max(mag_sum, 1.0)


def widen(lo: float, hi: float) -> tuple[float, float]:
    """Widen ``[lo, hi]`` outward, each end sized by ITS OWN magnitude.

    Sizing both ends by the interval's larger end would make the guard on the
    small end wildly coarse: the aux enclosure of ``x**5`` over ``[-18.3, -0.64]``
    is ``[-2.06e6, -0.106]``, where a guard scaled by ``2.06e6`` moves the upper
    end by ``7.3e-9`` — a ``7e-8`` RELATIVE widening of a quantity that is only
    wrong by an ulp of its own size. Per-end sizing is both tighter and the
    correct model of where the rounding actually happened.
    """
    return lo - outward_slack(bounded_mag(lo)), hi + outward_slack(bounded_mag(hi))


def envelope_1d_slack(
    dfdt: float,
    t_lo: float,
    t_hi: float,
    f_lo: float,
    f_hi: float,
    cst: float,
    rhs: float,
) -> float:
    """Outward slack for a 1-D envelope row of ``w = f(t)``, ``t = form + cst``.

    ``dfdt`` is the row's slope in ``t`` (a tangent's ``f'(t0)`` or the secant
    slope); ``[t_lo, t_hi]`` is the box in ``t``-space and ``f_lo``/``f_hi`` are
    ``f`` at its endpoints. ``f`` is monotone or convex over the box for every
    atom relaxed this way, so ``max(|f_lo|, |f_hi|)`` bounds ``|w|`` — including
    at the midpoint tangent.

    Computed in ``t``-space (not the base variable's) because that is where the
    cancellation happens and where all three engines share intermediates.
    """
    if not (box_finite(t_lo) and box_finite(t_hi)):
        return outward_slack(0.0)  # unbounded box: absolute floor only
    d = bounded_mag(dfdt)
    tmag = max(bounded_mag(t_lo), bounded_mag(t_hi))
    fmag = max(bounded_mag(f_lo), bounded_mag(f_hi))
    return outward_slack(d * tmag + d * bounded_mag(cst) + fmag + bounded_mag(rhs))


def envelope_product_slack(
    coef_a: float,
    coef_b: float,
    a_lo: float,
    a_hi: float,
    b_lo: float,
    b_hi: float,
    a_const: float,
    b_const: float,
    rhs: float,
) -> float:
    """Outward slack for a McCormick row of ``w = A*B`` over the forms' enclosures.

    ``[a_lo, a_hi]`` / ``[b_lo, b_hi]`` are the interval enclosures of the two
    affine forms over the box (constants included), ``coef_a``/``coef_b`` the row's
    multipliers on them. The forms' constants are folded into ``rhs`` by the
    emitters, so their contribution is added back here explicitly.
    """
    if not (box_finite(a_lo) and box_finite(a_hi) and box_finite(b_lo) and box_finite(b_hi)):
        return outward_slack(0.0)  # unbounded form enclosure: absolute floor only
    ca, cb = bounded_mag(coef_a), bounded_mag(coef_b)
    amag = max(bounded_mag(a_lo), bounded_mag(a_hi))
    bmag = max(bounded_mag(b_lo), bounded_mag(b_hi))
    wmag = max(
        bounded_mag(a_lo * b_lo),
        bounded_mag(a_lo * b_hi),
        bounded_mag(a_hi * b_lo),
        bounded_mag(a_hi * b_hi),
    )
    return outward_slack(
        ca * amag
        + ca * bounded_mag(a_const)
        + cb * bmag
        + cb * bounded_mag(b_const)
        + wmag
        + bounded_mag(rhs)
    )
