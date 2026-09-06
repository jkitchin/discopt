"""``interval_eval`` must REDUCE a ``SumExpression``, and round outward doing it.

Found via the #1148 source-residual probe and reviewed in
`#1158 <https://github.com/jkitchin/discopt/pull/1158>`_. ``_eval`` returned
``_eval(expr.operand, ...)`` for a ``SumExpression`` — the operand's *elementwise*
enclosure, unreduced — while ``dag_compiler`` lowers the same node to
``jnp.sum(operand, axis=expr.axis)``. Two consequences, and the second is the
serious one:

* the #1148 residual read ``0.0`` for a row ``sum(x) <= 10`` at ``x = (6, 6)``,
  where the true violation is ``2.0``;
* ``evaluate_interval`` is **on the solve path** — nonlinear bound tightening,
  the uniform and OA relaxations, the g-convex injection — so an enclosure that
  does not contain the value is an unsound FBBT tightening, the class that cuts
  the optimum out of the box.

This is a Python-API exposure: the ``.nl`` reader emits no ``SumExpression`` at
all (measured, 0 of the 66 files in ``python/tests/data/minlplib_nl/``), so no
``.nl`` panel covers it and the differential evidence has to live here.
"""

from __future__ import annotations

import discopt.modeling.core as dm
import numpy as np
import pytest
from discopt._relax.convexity.interval import Interval
from discopt._relax.convexity.interval_eval import evaluate_interval

pytestmark = pytest.mark.smoke

#: Executed comparison count (CLAUDE.md §6) — see ``teardown_module``.
COMPARISONS = 0


def _checked(condition: bool, message: str) -> None:
    global COMPARISONS
    COMPARISONS += 1
    assert condition, message


# ─────────────────────────── the enclosure itself ───────────────────────────


def test_sum_enclosure_contains_the_value():
    m = dm.Model("enc")
    x = m.continuous("x", shape=3, lb=0, ub=10)
    m.minimize(x[0])
    enc = evaluate_interval(dm.sum(x), m)
    _checked(isinstance(enc, Interval), "returns an Interval")
    _checked(
        float(enc.hi) >= 30.0 and float(enc.lo) <= 0.0,
        f"sum over [0,10]^3 must enclose [0, 30], got [{float(enc.lo)!r}, {float(enc.hi)!r}]",
    )
    _checked(
        np.asarray(enc.lo).shape == (),
        f"a full reduction is scalar, got shape {np.asarray(enc.lo).shape}",
    )


def test_axis_reduction_keeps_the_remaining_axis():
    m = dm.Model("axis")
    a = m.continuous("a", shape=(2, 3), lb=0, ub=1)
    m.minimize(a[0, 0])
    enc = evaluate_interval(dm.sum(a, axis=1), m)
    _checked(
        np.asarray(enc.hi).shape == (2,),
        f"sum(axis=1) over a (2,3) operand has shape (2,), got {np.asarray(enc.hi).shape}",
    )
    _checked(float(np.max(enc.hi)) >= 3.0, f"and encloses 3, got {enc.hi!r}")


def test_the_enclosure_contains_the_exact_sum_under_cancellation():
    """The reduction must be widened by the ACCUMULATION error, not by one ULP.

    A single ``nextafter`` per endpoint is the right convention for a binary
    operation and is what ``_eval_matmul`` uses, but a reduction over n terms
    accumulates far more than one ULP. Measured against an exact
    :class:`fractions.Fraction` reference before the fix: the one-ULP form
    returned an enclosure that did **not** contain the true sum on 2289 of 3000
    random trials (n in [4, 600], heavy cancellation at ~1e8), worst shortfall
    1.5e-6 (#1158 review 3).

    The reference is ``Fraction``, deliberately **not** ``np.longdouble``: that is
    float64 on arm64, so the comparison degrades to ``fl == fl`` and the probe
    measures nothing — the CLAUDE.md §6 failure, which caught the reviewer's own
    first two attempts at this.
    """
    from fractions import Fraction  # noqa: PLC0415

    rng = np.random.default_rng(20260905)
    misses = 0
    trials = 200
    for trial in range(trials):
        n = int(rng.integers(4, 601))
        vals = rng.normal(0.0, 1e8, size=n)
        half = n // 2
        vals[:half] = -vals[half : 2 * half][:half]  # force cancellation

        m = dm.Model(f"cancel{trial}")
        x = m.continuous("x", shape=n, lb=0, ub=1)
        m.minimize(x[0])
        enc = evaluate_interval(dm.sum(x), m, {x: Interval(vals.copy(), vals.copy())})
        exact = sum((Fraction(float(v)) for v in vals), Fraction(0))
        if not (Fraction(float(enc.lo)) <= exact <= Fraction(float(enc.hi))):
            misses += 1
    _checked(
        misses == 0,
        f"{misses}/{trials} enclosures did not contain the exact sum — the widening "
        "does not bound the accumulation error",
    )


def test_slow_axis_reduction_is_widened_for_a_sequential_accumulation():
    """The widening must be valid for the reduction order NumPy actually used.

    NumPy's pairwise summation applies only along the *fast* (unit-stride) axis;
    a reduction across a strided axis is a plain sequential accumulation, whose
    forward error bound is ``(n-1)*u*sum|x_i|`` (Higham, *ASNA* 2nd ed. §4.2),
    not the pairwise ``(log2(n)+1)*u*sum|x_i|``. Applying the pairwise factor to
    a strided reduction understates the error by ~n/log2(n) and yields an
    enclosure that does **not** contain the value.

    Counterexample from the #1158 review, reproduced before the fix: summing a
    ``(10002, 2)`` C-contiguous array down ``axis=0`` — the slow axis — with
    ``1e16`` first, ``-1e16`` last and ones between gives an exact column sum of
    ``10000``, and the pairwise-factored enclosure returned
    ``[-67.89, 67.89]``, missing it by three orders of magnitude. The transposed
    control reduces along the fast axis, stays pairwise, and was correct all
    along; keeping it here is what shows the fix did not simply widen everything.
    """
    n = 10002
    a = np.ones((n, 2), dtype=np.float64, order="C")
    a[0, :] = 1e16
    a[-1, :] = -1e16
    exact = 10000.0

    m = dm.Model("slowaxis")
    v = m.continuous("v", shape=(n, 2), lb=-1e17, ub=1e17)
    enc = evaluate_interval(dm.sum(v, axis=0), m, {v: Interval(a.copy(), a.copy())})
    lo = np.asarray(enc.lo).ravel()
    hi = np.asarray(enc.hi).ravel()
    _checked(
        bool(np.all(lo <= exact) and np.all(exact <= hi)),
        f"slow-axis enclosure [{lo[0]!r}, {hi[0]!r}] does not contain the exact "
        f"column sum {exact!r} — the widening assumed a pairwise reduction that "
        "NumPy does not perform across a strided axis",
    )

    at = np.ascontiguousarray(a.T)
    m2 = dm.Model("fastaxis")
    v2 = m2.continuous("v2", shape=at.shape, lb=-1e17, ub=1e17)
    enc2 = evaluate_interval(dm.sum(v2, axis=1), m2, {v2: Interval(at.copy(), at.copy())})
    lo2 = np.asarray(enc2.lo).ravel()
    hi2 = np.asarray(enc2.hi).ravel()
    _checked(
        bool(np.all(lo2 <= exact) and np.all(exact <= hi2)),
        f"fast-axis control [{lo2[0]!r}, {hi2[0]!r}] does not contain {exact!r}",
    )
    # The control must stay TIGHT: a pairwise reduction keeps the log2(n) factor,
    # so its half-width is smaller than the sequential one by orders of magnitude.
    _checked(
        (hi2[0] - lo2[0]) < (hi[0] - lo[0]) / 100.0,
        f"the fast-axis control widened to {(hi2[0] - lo2[0])!r}, within 100x of the "
        f"slow-axis {(hi[0] - lo[0])!r} — the fix widened the pairwise case too",
    )


def test_slow_axis_enclosure_holds_over_random_strided_reductions():
    """The sequential bound must hold across sizes, not just on one counterexample.

    Differential against an exact :class:`fractions.Fraction` reference, on the
    **slow** axis of a C-contiguous 2-D array so numpy accumulates sequentially,
    with heavy cancellation so the error actually shows. This is also what backs
    the magnitude-sum argument in ``_accumulation_factor``: ``sum |x_i|`` is a
    float64 reduction of the same length, and if the factor's headroom did not
    absorb its own accumulation error these enclosures would start missing.
    """
    from fractions import Fraction  # noqa: PLC0415

    rng = np.random.default_rng(20260906)
    misses = 0
    trials = 40
    for trial in range(trials):
        n = int(rng.integers(500, 8001))
        col = rng.normal(0.0, 1e9, size=n)
        half = n // 2
        col[:half] = -col[half : 2 * half][:half]
        a = np.ascontiguousarray(np.stack([col, col[::-1]], axis=1))  # (n, 2), C-order

        m = dm.Model(f"strided{trial}")
        v = m.continuous("v", shape=a.shape, lb=-1e10, ub=1e10)
        m.minimize(v[0, 0])
        enc = evaluate_interval(dm.sum(v, axis=0), m, {v: Interval(a.copy(), a.copy())})
        lo = np.asarray(enc.lo).ravel()
        hi = np.asarray(enc.hi).ravel()
        for j in range(a.shape[1]):
            exact = sum((Fraction(float(x)) for x in a[:, j]), Fraction(0))
            if not (Fraction(float(lo[j])) <= exact <= Fraction(float(hi[j]))):
                misses += 1
    _checked(
        misses == 0,
        f"{misses}/{2 * trials} slow-axis enclosures did not contain the exact sum",
    )


def test_the_sequential_bound_never_narrows_an_enclosure():
    """The fix must only ever widen: a soundness fix that tightens is bound-changing.

    ``n - 1`` and ``log2(n) + 2`` cross at ``n = 5``, so a bare sequential factor
    would make strided enclosures *narrower* than the code it replaces for a
    handful of terms. Both are sound, but tightening a bound anywhere needs its
    own differential evidence (CLAUDE.md §5) and buys nothing at ``n <= 5``, so
    the sequential factor is the max of the two. Checked across the crossover and
    well past it.
    """
    from discopt._relax.convexity.interval_eval import _accumulation_factor  # noqa: PLC0415

    for n in (2, 3, 4, 5, 6, 7, 16, 100, 10002, 10**6):
        seq = _accumulation_factor(n, pairwise=False)
        pw = _accumulation_factor(n, pairwise=True)
        _checked(
            seq >= pw,
            f"n={n}: the sequential factor {seq!r} is below the pairwise {pw!r} — "
            "this fix would narrow a strided enclosure",
        )
    _checked(
        _accumulation_factor(1, pairwise=False) == 0.0
        and _accumulation_factor(0, pairwise=True) == 0.0,
        "the identity reduction is exact and gets no widening at all",
    )


def test_widening_does_not_break_a_degenerate_point_evaluation():
    """The error bound must stay far below the residual probe's degeneracy guard.

    The two requirements pull in opposite directions: wide enough to be sound,
    tight enough that ``mpec_report.evaluate_at_point`` still accepts a point-box
    evaluation as degenerate. Checked at sizes where the ``log2(n)`` term is
    largest.
    """
    from discopt.mpec_report import evaluate_at_point, point_from_flat  # noqa: PLC0415

    for n in (10, 1000, 20000):
        m = dm.Model(f"degenerate{n}")
        x = m.continuous("x", shape=n, lb=0, ub=1e6)
        m.minimize(x[0])
        vals = np.full(n, 1234.5678)
        got = float(evaluate_at_point(m, dm.sum(x), point_from_flat(m, vals))[0])
        _checked(
            abs(got - n * 1234.5678) <= 1e-6 * max(1.0, abs(got)),
            f"n={n}: point evaluation returned {got!r}, expected {n * 1234.5678!r}",
        )


def test_the_reduction_rounds_outward():
    """The module's stated invariant, and what ``_eval_matmul`` already did.

    Summing n terms accumulates ~n ULP per endpoint, so an un-rounded sum can be
    strictly narrower than the true image — unsound in the direction that breaks
    a certificate.
    """
    m = dm.Model("round")
    x = m.continuous("x", shape=3, lb=0.1, ub=0.1)
    m.minimize(x[0])
    enc = evaluate_interval(dm.sum(x), m)
    exact = 0.1 + 0.1 + 0.1
    _checked(
        float(enc.lo) < exact < float(enc.hi),
        f"the enclosure must straddle {exact!r}, got [{float(enc.lo)!r}, {float(enc.hi)!r}]",
    )


def test_nested_sums_stay_sound():
    m = dm.Model("nested")
    x = m.continuous("x", shape=4, lb=1.0, ub=2.0)
    m.minimize(x[0])
    enc = evaluate_interval(dm.sum(x * x), m)
    _checked(
        float(enc.lo) <= 4.0 and float(enc.hi) >= 16.0,
        f"sum(x*x) over [1,2]^4 encloses [4, 16], got [{float(enc.lo)!r}, {float(enc.hi)!r}]",
    )


# ──────────────── the differential panel: bounds on real solves ────────────────
#
# The point of the enclosure is the bound it produces. For a MINIMIZE, a dual
# bound must never exceed the true optimum: one that does has fathomed the
# optimum away, which is the CLAUDE.md §1 failure with no slack.

_SOUND_CASES = [
    # (name, builder, true optimum)
    (
        "sum_linear",
        lambda m: (
            m.subject_to(dm.sum(m._v) <= 7),
            m.minimize(-dm.sum(m._v)),
        ),
        (3,),
        (0, 10),
        -7.0,
    ),
    (
        "sum_squares",
        lambda m: (
            m.subject_to(dm.sum(m._v * m._v) <= 12),
            m.minimize(-dm.sum(m._v)),
        ),
        (3,),
        (0, 3),
        -6.0,
    ),
    (
        "sum_exp",
        lambda m: (
            m.subject_to(dm.sum(m._v) >= 3),
            m.minimize(dm.sum(dm.exp(m._v))),
        ),
        (3,),
        (0, 2),
        3.0 * float(np.exp(1.0)),
    ),
]


@pytest.mark.parametrize(("name", "build", "shape", "box", "optimum"), _SOUND_CASES)
def test_dual_bound_never_exceeds_the_optimum(name, build, shape, box, optimum):
    m = dm.Model(name)
    m._v = m.continuous("v", shape=shape, lb=box[0], ub=box[1])
    build(m)
    res = m.solve(time_limit=60, gap_tolerance=1e-6)
    tol = 1e-4 + 1e-4 * abs(optimum)
    _checked(
        res.objective is not None and abs(res.objective - optimum) <= tol,
        f"{name}: objective {res.objective!r} should be {optimum!r}",
    )
    _checked(
        res.bound is None or res.bound <= optimum + tol,
        f"{name}: dual bound {res.bound!r} is ABOVE the true optimum {optimum!r} — "
        "it has fathomed the optimum away",
    )


def test_axis_reduced_constraint_bound_is_sound():
    """An axis-reduced constraint is several rows, not one (jkitchin/discopt#1160).

    ``dm.sum(a, axis=1) <= 2`` caps each of the two rows, so the total is at most
    4 and the optimum of ``min -sum(a)`` is -4.  Extracting it as the single
    collapsed row ``sum(a) <= 2`` certifies -2 and fathoms the optimum away.
    This was an xfail(strict=True) here until #1160 landed.
    """
    m = dm.Model("axis_bound")
    a = m.continuous("a", shape=(2, 3), lb=0, ub=1)
    m.subject_to(dm.sum(a, axis=1) <= 2)  # per-row cap => total <= 4
    m.minimize(-dm.sum(a))
    res = m.solve(time_limit=60, gap_tolerance=1e-6)
    _checked(
        res.bound is not None and res.bound <= -4.0 + 1e-4,
        f"dual bound {res.bound!r} is above the true optimum -4.0",
    )


def teardown_module(module):  # noqa: ARG001 - pytest hook signature
    """CLAUDE.md §6: an enclosure suite that compared nothing is not a pass."""
    print(f"\ninterval sum-reduction probe: {COMPARISONS} comparisons executed")
    assert COMPARISONS > 0, "the interval enclosure probes executed ZERO comparisons"
