"""Soundness tests for the interval arithmetic primitives.

Every operation must produce an enclosure that contains the pointwise
result at every sample inside the input. This is the defining
contract — a violation propagates into the box-local convexity
certificate as an unsound verdict, so it must be caught here.

References
----------
Moore (1966), *Interval Analysis*.
Neumaier (1990), *Interval Methods for Systems of Equations*.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from discopt._jax.convexity import interval as iv
from discopt._jax.convexity.interval import Interval

pytestmark = pytest.mark.unit

SAMPLES = 64


def _sample_in(x: Interval, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low=x.lo, high=x.hi, size=(n,) + x.lo.shape)


def _assert_encloses(result: Interval, values: np.ndarray) -> None:
    """All ``values`` must lie inside ``result``."""
    assert np.all(result.lo <= values + 1e-12), (
        f"result.lo {result.lo} > sample {values[values < result.lo - 1e-12]}"
    )
    assert np.all(values <= result.hi + 1e-12), (
        f"sample {values[values > result.hi + 1e-12]} > result.hi {result.hi}"
    )


class TestConstruction:
    def test_point_interval(self):
        p = Interval.point(3.5)
        assert p.lo == 3.5 and p.hi == 3.5

    def test_from_bounds(self):
        p = Interval.from_bounds(-1.0, 2.0)
        assert p.lo == -1.0 and p.hi == 2.0

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            Interval.from_bounds(1.0, 0.0)

    def test_contains_zero(self):
        assert bool(Interval.from_bounds(-1, 1).contains_zero())
        assert not bool(Interval.from_bounds(0.1, 1).contains_zero())
        assert bool(Interval.from_bounds(0, 1).contains_zero())


class TestArithmetic:
    """Four-operation soundness: enclosure contains every pointwise result."""

    @pytest.mark.parametrize(
        "seed,op,lo1,hi1,lo2,hi2",
        [
            (0, "+", -2.0, 3.0, 1.0, 4.0),
            (1, "-", -2.0, 3.0, -5.0, 5.0),
            (2, "*", -2.0, 3.0, -4.0, 4.0),
            (3, "*", 0.1, 2.0, 0.1, 2.0),
            (4, "/", -2.0, 3.0, 0.1, 2.0),
            (5, "/", 1.0, 5.0, -5.0, -0.1),
        ],
    )
    def test_binary_op_encloses(self, seed, op, lo1, hi1, lo2, hi2):
        a = Interval.from_bounds(lo1, hi1)
        b = Interval.from_bounds(lo2, hi2)
        ops = {
            "+": (a + b, lambda x, y: x + y),
            "-": (a - b, lambda x, y: x - y),
            "*": (a * b, lambda x, y: x * y),
            "/": (a / b, lambda x, y: x / y),
        }
        out, truth = ops[op]
        xs = _sample_in(a, SAMPLES, seed)
        ys = _sample_in(b, SAMPLES, seed + 1000)
        _assert_encloses(out, truth(xs, ys))

    def test_division_through_zero_unbounded(self):
        """``1 / [−1, 1]`` must yield an unbounded enclosure."""
        a = Interval.point(1.0)
        b = Interval.from_bounds(-1.0, 1.0)
        out = a / b
        assert (
            not np.isfinite(out.lo)
            or not np.isfinite(out.hi)
            or (out.lo == -np.inf and out.hi == np.inf)
        )
        assert np.isinf(out.lo) and np.isinf(out.hi)

    def test_negation(self):
        a = Interval.from_bounds(-2.0, 3.0)
        neg = -a
        assert neg.lo == -3.0 and neg.hi == 2.0


class TestC36MulZeroTimesInfinity:
    """C-36: ``Interval.__mul__`` must not leak NaN on ``0 · ±∞``.

    A finite zero endpoint times an infinite endpoint is ``0 * ±∞ = NaN`` in
    IEEE-754. Before the fix, ``np.minimum``/``np.maximum`` propagated that NaN
    and the interval collapsed to ``[NaN, NaN]`` — every downstream comparison
    against NaN is False, so the convexity certificate's ``np.isfinite`` gate
    abstains (sound but weak: lost tightening). The interval convention makes
    ``0 · [anything] = 0``, so the fix maps NaN corners to ``0`` while leaving
    genuine ±∞ corners intact. These are the Python siblings of the Rust C-22
    tests (PR #460).
    """

    @pytest.mark.smoke
    def test_zero_point_times_entire_is_zero(self):
        # The exact repro: [0,0] · [-inf, inf]. Every corner is 0 * (±inf) = NaN.
        # Sound, informative enclosure is [0, 0] (up to one ULP of outward
        # rounding), NOT [NaN, NaN].
        a = Interval.point(0.0)
        b = Interval.from_bounds(-np.inf, np.inf)
        r = a * b
        assert not np.isnan(r.lo) and not np.isnan(r.hi), "must not be NaN"
        assert r.lo <= 0.0 <= r.hi, "must enclose 0"
        # Tightest representable enclosure of the point 0: within one ULP.
        assert abs(float(r.lo)) <= 1e-300 and abs(float(r.hi)) <= 1e-300

    @pytest.mark.smoke
    def test_zero_touching_times_entire_is_unbounded(self):
        # [0,5] · [-inf, inf]: the 0-corners are NaN→0, but the genuine ±inf
        # corners (5 * ±inf) still dominate → [-inf, +inf]. The fix must NOT
        # clamp those infinities.
        a = Interval.from_bounds(0.0, 5.0)
        b = Interval.from_bounds(-np.inf, np.inf)
        r = a * b
        assert not np.isnan(r.lo) and not np.isnan(r.hi)
        assert r.lo == -np.inf and r.hi == np.inf

    @pytest.mark.smoke
    def test_no_runtime_warning_on_zero_times_inf(self):
        # The `RuntimeWarning: invalid value encountered in multiply` that the
        # unguarded corner products raised must be gone.
        a = Interval.point(0.0)
        b = Interval.from_bounds(-np.inf, np.inf)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            a * b  # no RuntimeWarning

    @pytest.mark.smoke
    def test_mul_never_nan_and_encloses_true_product(self):
        # Grid over ±∞ endpoints and zero-width / zero-touching factors. For
        # every valid pair the product must (a) never be NaN, (b) satisfy
        # lo <= hi, and (c) rigorously enclose the true product of every pair of
        # representative points drawn from the two intervals. A NaN endpoint
        # would fail containment silently, so both are checked explicitly.
        ninf, pinf = -np.inf, np.inf
        bounds = [ninf, -3.0, -1.0, 0.0, 1.0, 2.5, pinf]
        for alo in bounds:
            for ahi in bounds:
                if alo > ahi:
                    continue
                for blo in bounds:
                    for bhi in bounds:
                        if blo > bhi:
                            continue
                        a = Interval.from_bounds(alo, ahi)
                        b = Interval.from_bounds(blo, bhi)
                        r = a * b
                        assert not np.isnan(r.lo) and not np.isnan(r.hi), (
                            f"NaN endpoint for [{alo},{ahi}]*[{blo},{bhi}]"
                        )
                        assert r.lo <= r.hi, f"lo>hi for [{alo},{ahi}]*[{blo},{bhi}]"
                        # True product set at the (finite) representative points.
                        for x in (alo, ahi):
                            for y in (blo, bhi):
                                if not (np.isfinite(x) and np.isfinite(y)):
                                    continue
                                p = x * y
                                assert r.lo <= p + 1e-9, (
                                    f"{p} < lo {r.lo} for [{alo},{ahi}]*[{blo},{bhi}]"
                                )
                                assert p - 1e-9 <= r.hi, (
                                    f"{p} > hi {r.hi} for [{alo},{ahi}]*[{blo},{bhi}]"
                                )


class TestPower:
    def test_square_nonneg_is_nonneg(self):
        a = Interval.from_bounds(1.0, 3.0)
        sq = a**2
        assert sq.lo >= 1.0 - 1e-12 and sq.hi <= 9.0 + 1e-12

    def test_square_containing_zero(self):
        """``[-2, 3]^2 = [0, 9]`` — minimum is attained at 0, not the endpoints."""
        a = Interval.from_bounds(-2.0, 3.0)
        sq = a**2
        assert sq.lo <= 0.0
        assert sq.hi >= 9.0

    def test_cube_encloses_samples(self):
        a = Interval.from_bounds(-2.0, 3.0)
        c = a**3
        xs = _sample_in(a, SAMPLES, seed=0)
        _assert_encloses(c, xs**3)

    def test_negative_integer_power_strict_positive(self):
        a = Interval.from_bounds(0.5, 2.0)
        inv = a**-2
        xs = _sample_in(a, SAMPLES, seed=0)
        _assert_encloses(inv, xs**-2)

    def test_pow_zero_is_one(self):
        a = Interval.from_bounds(-2.0, 3.0)
        assert np.all((a**0).lo == 1.0) and np.all((a**0).hi == 1.0)


class TestElementaryFunctions:
    """Each elementary function must enclose pointwise evaluation."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_exp_encloses(self, seed):
        a = Interval.from_bounds(-3.0, 3.0)
        out = iv.exp(a)
        xs = _sample_in(a, SAMPLES, seed=seed)
        _assert_encloses(out, np.exp(xs))

    def test_exp_wide_interval_overflow_is_quiet(self):
        """Overflow to an unbounded enclosure is an expected abstention path."""
        a = Interval.from_bounds(-1000.0, 1000.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = iv.exp(a)

        assert np.isfinite(out.lo)
        assert np.isposinf(out.hi)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_log_encloses(self, seed):
        a = Interval.from_bounds(0.1, 10.0)
        out = iv.log(a)
        xs = _sample_in(a, SAMPLES, seed=seed)
        _assert_encloses(out, np.log(xs))

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_sqrt_encloses(self, seed):
        a = Interval.from_bounds(0.0, 5.0)
        out = iv.sqrt(a)
        xs = _sample_in(a, SAMPLES, seed=seed)
        _assert_encloses(out, np.sqrt(xs))

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_abs_encloses(self, seed):
        a = Interval.from_bounds(-3.0, 2.0)
        out = iv.absolute(a)
        xs = _sample_in(a, SAMPLES, seed=seed)
        _assert_encloses(out, np.abs(xs))

    def test_abs_zero_is_zero_lower(self):
        """``|[-1, 1]|`` must include 0 as a lower endpoint."""
        a = Interval.from_bounds(-1.0, 1.0)
        out = iv.absolute(a)
        assert out.lo <= 0.0

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_tanh_encloses(self, seed):
        a = Interval.from_bounds(-3.0, 3.0)
        out = iv.tanh(a)
        xs = _sample_in(a, SAMPLES, seed=seed)
        _assert_encloses(out, np.tanh(xs))

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_cosh_encloses(self, seed):
        a = Interval.from_bounds(-2.0, 3.0)
        out = iv.cosh(a)
        xs = _sample_in(a, SAMPLES, seed=seed)
        _assert_encloses(out, np.cosh(xs))

    def test_cosh_minimum_at_zero(self):
        a = Interval.from_bounds(-1.0, 2.0)
        out = iv.cosh(a)
        assert out.lo <= 1.0 + 1e-12

    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize("lo, hi", [(0.1, 3.0), (0.05, 0.2), (1.0, 5.0), (0.3, 0.5)])
    def test_entropy_encloses(self, lo, hi, seed):
        a = Interval.from_bounds(lo, hi)
        out = iv.entropy(a)
        xs = _sample_in(a, SAMPLES, seed=seed)
        _assert_encloses(out, xs * np.log(xs))

    def test_entropy_minimum_captured_when_straddled(self):
        # 1/e ≈ 0.368 is inside, so the lower endpoint must reach -1/e.
        out = iv.entropy(Interval.from_bounds(0.1, 3.0))
        assert out.lo <= -1.0 / np.e + 1e-12

    def test_entropy_out_of_domain_abstains(self):
        # A box touching nonpositive x is out of domain -> collapse outward.
        out = iv.entropy(Interval.from_bounds(-1.0, 2.0))
        assert np.isneginf(out.lo) and np.isposinf(out.hi)


class TestTrigEnclosures:
    """Periodic-trig soundness on small intervals and on wide intervals."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_sin_small_interval(self, seed):
        a = Interval.from_bounds(0.1, 1.0)
        out = iv.sin(a)
        xs = _sample_in(a, SAMPLES, seed=seed)
        _assert_encloses(out, np.sin(xs))

    def test_sin_spans_critical_point(self):
        """An interval straddling π/2 must have upper endpoint ≥ 1."""
        a = Interval.from_bounds(1.0, 2.0)
        out = iv.sin(a)
        assert out.hi >= 1.0 - 1e-12

    def test_sin_wide_interval_safe_fallback(self):
        a = Interval.from_bounds(-10.0, 10.0)
        out = iv.sin(a)
        # [-1, 1] is the tightest sound enclosure.
        assert out.lo <= -1.0 + 1e-12
        assert out.hi >= 1.0 - 1e-12

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_cos_small_interval(self, seed):
        a = Interval.from_bounds(0.1, 1.0)
        out = iv.cos(a)
        xs = _sample_in(a, SAMPLES, seed=seed)
        _assert_encloses(out, np.cos(xs))

    def test_tan_same_branch(self):
        """Within a single branch (−π/2, π/2) tan is monotone."""
        a = Interval.from_bounds(-1.0, 1.0)
        out = iv.tan(a)
        xs = _sample_in(a, SAMPLES, seed=0)
        _assert_encloses(out, np.tan(xs))

    def test_tan_crosses_asymptote_unbounded(self):
        """Straddling π/2 → unbounded enclosure."""
        a = Interval.from_bounds(1.0, 2.0)
        out = iv.tan(a)
        assert np.isinf(out.lo) or np.isinf(out.hi)


class TestCompositeExpressions:
    """Full composition soundness on realistic derivative expressions."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_derivative_of_exp_squared(self, seed):
        """d/dx exp(x^2) = 2x exp(x^2)."""
        x = Interval.from_bounds(-2.0, 2.0)
        out = 2.0 * x * iv.exp(x**2)
        xs = _sample_in(x, SAMPLES, seed=seed)
        _assert_encloses(out, 2.0 * xs * np.exp(xs**2))

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_rational_composition(self, seed):
        """(1 + x^2) / (1 + x^4)."""
        x = Interval.from_bounds(-3.0, 3.0)
        num = Interval.point(1.0) + x**2
        den = Interval.point(1.0) + x**4
        out = num / den
        xs = _sample_in(x, SAMPLES, seed=seed)
        _assert_encloses(out, (1.0 + xs**2) / (1.0 + xs**4))

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_log_of_sum_of_squares(self, seed):
        """log(x^2 + y^2 + 1) — arises in many convex MLE objectives."""
        x = Interval.from_bounds(-2.0, 2.0)
        y = Interval.from_bounds(-2.0, 2.0)
        z = iv.log(x**2 + y**2 + Interval.point(1.0))
        xs = _sample_in(x, SAMPLES, seed=seed)
        ys = _sample_in(y, SAMPLES, seed=seed + 100)
        _assert_encloses(z, np.log(xs**2 + ys**2 + 1.0))


class TestOutwardRounding:
    """Verify that endpoints are actually rounded outward."""

    def test_add_rounding_direction(self):
        """Adding two identical small values shouldn't collapse the interval."""
        a = Interval.from_bounds(0.1, 0.1)
        b = Interval.from_bounds(0.2, 0.2)
        out = a + b
        # The true sum 0.3 is not exactly representable; the interval
        # must enclose it. lo ≤ 0.3 ≤ hi strictly.
        assert out.lo <= 0.3 <= out.hi

    def test_exp_rounding_direction(self):
        """exp(1) endpoint should enclose the true value."""
        a = Interval.point(1.0)
        out = iv.exp(a)
        # The true value e is not a float; the interval must enclose it.
        true_e = np.float64(np.e)
        assert out.lo <= true_e <= out.hi
