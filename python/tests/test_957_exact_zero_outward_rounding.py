"""#957 — outward rounding must not turn an exact zero into a subnormal.

``np.nextafter(0.0, ±inf)`` is ``±5e-324``, so the interval layer's outward
rounding used to manufacture a subnormal endpoint out of *every* exact zero.
That value is a sound enclosure, but it is ~300 orders of magnitude below any
modelled coefficient, and downstream code that reasons about magnitudes is not
prepared for it: ``milp_relaxation``'s coefficient-spread guard divides by the
smallest nonzero entry and overflows to ``inf``, so it fires unconditionally.

Two halves to the fix, and this file pins both — plus, importantly, the
*soundness* half that must NOT change: the nudge at zero is doing real work
wherever the producing operation can underflow (``exp``, a product of two tiny
factors), and dropping it there would produce an enclosure that excludes the
true value.
"""

import warnings

import discopt._jax.convexity.interval as iv
import numpy as np
import pytest
from discopt._jax.convexity.eigenvalue import (
    gershgorin_lambda_max,
    gershgorin_lambda_min,
    psd_2x2_sufficient,
)
from discopt._jax.convexity.interval import (
    Interval,
    _round_down,
    _round_down_exact0,
    _round_up,
    _round_up_exact0,
)
from discopt._jax.milp_relaxation import (
    _RELAX_EQUILIBRATE_TRIGGER,
    _SUBNORMAL_FLOOR,
    _coefficient_spread_exceeds,
)

TINY = float(np.finfo(np.float64).tiny)


def _endpoints(interval):
    return (
        float(np.asarray(interval.lo).ravel()[0]),
        float(np.asarray(interval.hi).ravel()[0]),
    )


def _box(lo, hi):
    return Interval(np.array([float(lo)]), np.array([float(hi)]))


def _is_subnormal(x):
    return x != 0.0 and abs(x) < TINY


# ----------------------------------------------------------------------
# The helpers themselves
# ----------------------------------------------------------------------


def test_unconditional_helpers_still_nudge_zero():
    """The always-sound helpers keep their old behaviour — the fix is opt-in."""
    assert float(_round_down(np.float64(0.0))) == -5e-324
    assert float(_round_up(np.float64(0.0))) == 5e-324


def test_exact0_helpers_preserve_zero_and_nudge_everything_else():
    assert float(_round_down_exact0(np.float64(0.0))) == 0.0
    assert float(_round_up_exact0(np.float64(0.0))) == 0.0
    # -0.0 normalises to +0.0 rather than becoming a subnormal.
    assert float(_round_down_exact0(np.float64(-0.0))) == 0.0
    # Nonzero values are still widened by exactly one ULP, in the right direction.
    assert float(_round_down_exact0(np.float64(1.0))) == np.nextafter(1.0, -np.inf)
    assert float(_round_up_exact0(np.float64(1.0))) == np.nextafter(1.0, np.inf)
    assert float(_round_down_exact0(np.float64(-np.inf))) == -np.inf
    assert float(_round_up_exact0(np.float64(np.inf))) == np.inf


# ----------------------------------------------------------------------
# Exact zeros stay exactly zero
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, build",
    [
        # A zero factor: ``0 * x`` is exact in IEEE-754. This is the common
        # shape — a zero gradient or Hessian entry meeting anything.
        ("point-zero product", lambda: Interval.point(0.0) * _box(-3.0, 7.0)),
        ("product with zero endpoint", lambda: _box(0.0, 2.0) * _box(0.0, 5.0)),
        # A float sum/difference is zero only when the exact result is zero.
        ("self difference", lambda: _box(2.0, 2.0) - _box(2.0, 2.0)),
        ("cancelling sum", lambda: _box(-4.0, -4.0) + _box(4.0, 4.0)),
        ("width of a degenerate box", lambda: Interval.point(3.0)),
        # A square is never negative; nor is |x|; nor is sqrt.
        ("square straddling zero", lambda: _box(-1.0, 1.0) ** 2),
        ("abs straddling zero", lambda: iv.absolute(_box(-1.0, 1.0))),
        ("sqrt at zero", lambda: iv.sqrt(_box(0.0, 4.0))),
        ("log at one", lambda: iv.log(_box(1.0, 1.0))),
        # 1/x is zero only at x = ±inf, where the exact reciprocal is zero.
        ("reciprocal of an unbounded box", lambda: _box(1.0, 1.0) / _box(1.0, np.inf)),
    ],
)
def test_no_subnormal_endpoints_from_exact_operations(label, build):
    result = build()
    lo, hi = _endpoints(result)
    assert not _is_subnormal(lo), f"{label}: lower endpoint is subnormal ({lo!r})"
    assert not _is_subnormal(hi), f"{label}: upper endpoint is subnormal ({hi!r})"
    assert not _is_subnormal(float(np.asarray(result.width).ravel()[0])), f"{label}: width"


def test_specific_exact_zero_endpoints():
    """The endpoints that used to be ``±5e-324`` are now exactly ``0.0``."""
    assert _endpoints(Interval.point(0.0) * _box(-3.0, 7.0)) == (0.0, 0.0)
    assert _endpoints(_box(2.0, 2.0) - _box(2.0, 2.0)) == (0.0, 0.0)
    assert _endpoints(iv.log(_box(1.0, 1.0))) == (0.0, 0.0)
    # A square and an absolute value now have a lower endpoint of exactly zero
    # instead of a *negative* subnormal.
    assert _endpoints(_box(-1.0, 1.0) ** 2)[0] == 0.0
    assert _endpoints(iv.absolute(_box(-1.0, 1.0)))[0] == 0.0
    assert _endpoints(iv.sqrt(_box(0.0, 4.0)))[0] == 0.0
    assert float(np.asarray(Interval.point(3.0).width).ravel()[0]) == 0.0


def test_ieee_addition_cannot_produce_a_false_zero():
    """The premise ``__add__``/``__sub__``/``width`` rest on, made executable.

    Every double is an integer multiple of ``2**-1074``, so the exact sum of
    two doubles is too; if it is nonzero its magnitude is at least ``2**-1074``,
    which round-to-nearest cannot map to zero. Hence ``fl(a + b) == 0`` implies
    ``a + b == 0`` exactly, and there is no underflow at zero for the outward
    nudge to cover. Fuzzed over a wide exponent range including subnormals.
    """
    rng = np.random.default_rng(957)
    n = 100_000
    # Random pairs never cancel, so aim straight at the only regime where a
    # false zero could appear: ``b`` within a few ULPs of ``-a``, sampled across
    # the whole exponent range including the subnormals.
    exponents = rng.integers(-1074, 1024, size=n)
    mantissas = rng.random(n) * rng.choice([-1.0, 1.0], size=n)
    with np.errstate(over="ignore", under="ignore"):
        a = np.ldexp(mantissas, exponents)
    b = -a.copy()
    shifts = rng.integers(-3, 4, size=n)
    for _ in range(3):  # walk |shift| ULPs outward, vectorised
        moving = shifts != 0
        direction = np.where(shifts > 0, np.inf, -np.inf)
        b = np.where(moving, np.nextafter(b, direction), b)
        shifts -= np.sign(shifts).astype(shifts.dtype)
    total = a + b

    zero_sum = np.isfinite(total) & (total == 0.0)
    nonzero_sum = np.isfinite(total) & (total != 0.0)
    assert zero_sum.sum() > 0, "fuzz produced no zero sums — the probe did not fire"
    assert nonzero_sum.sum() > 0, "fuzz produced only zero sums — the probe is trivial"

    # Compare in a wider type, so the check does not use the arithmetic under
    # test to judge itself: every ``fl(a + b) == 0`` must be an exact zero.
    exact = np.array(a[zero_sum], dtype=np.longdouble) + np.array(b[zero_sum], dtype=np.longdouble)
    assert np.all(exact == 0.0), "found fl(a+b) == 0 with a nonzero exact sum"
    print(f"checked {zero_sum.sum()} zero sums and {nonzero_sum.sum()} nonzero sums")


# ----------------------------------------------------------------------
# Soundness: where the nudge is load-bearing it must survive
# ----------------------------------------------------------------------


def test_exp_underflow_still_encloses_the_true_image():
    """``fl(exp(-800)) == 0.0`` while the exact value is ~3.6e-348.

    Dropping the nudge for ``exp`` would give an upper endpoint *below* the
    true image — an unsound enclosure. This is the counter-example that makes
    the ``_exact0`` variants opt-in rather than the default.
    """
    assert np.exp(-800.0) == 0.0  # the premise: numpy really does flush to zero
    lo, hi = _endpoints(iv.exp(_box(-900.0, -800.0)))
    assert hi > 0.0, "exp upper endpoint must stay above the true 3.6e-348"
    assert lo <= 0.0


@pytest.mark.parametrize(
    "label, build, true_lo, true_hi",
    [
        # 1e-200 * 1e-200 = 1e-400, which underflows to 0.0 in float64.
        ("positive product", lambda: _box(1e-200, 2e-200) * _box(1e-200, 2e-200), 1e-400, 4e-400),
        (
            "sign-crossing product",
            lambda: _box(-2e-200, -1e-200) * _box(1e-200, 2e-200),
            -4e-400,
            -1e-400,
        ),
        ("underflowing square", lambda: _box(-1e-200, 1e-200) ** 2, 0.0, 1e-400),
    ],
)
def test_underflowing_products_keep_the_outward_nudge(label, build, true_lo, true_hi):
    """A product that flushes to zero is NOT an exact zero — it must widen."""
    lo, hi = _endpoints(build())
    # ``true_lo``/``true_hi`` are below the subnormal range and cannot be
    # represented; the point is only that the enclosure straddles them.
    assert lo < 0.0 or true_lo >= 0.0, f"{label}: lower endpoint {lo!r} excludes {true_lo}"
    assert hi > 0.0, f"{label}: upper endpoint {hi!r} excludes {true_hi}"
    assert lo <= 0.0 <= hi


def test_a_zero_factor_and_an_underflow_in_the_same_array_stay_sound():
    """Vectorised: one lane underflows, so the whole array keeps the nudge."""
    lhs = Interval(np.array([0.0, 1e-200]), np.array([0.0, 2e-200]))
    rhs = Interval(np.array([3.0, 1e-200]), np.array([7.0, 2e-200]))
    prod = lhs * rhs
    hi = np.asarray(prod.hi).ravel()
    lo = np.asarray(prod.lo).ravel()
    # Lane 1 genuinely underflowed (true value ~1e-400 > 0), so its upper
    # endpoint must be strictly positive. Falling back conservatively for the
    # whole array is allowed; being unsound in lane 1 is not.
    assert hi[1] > 0.0
    assert lo[1] <= 0.0


# ----------------------------------------------------------------------
# Gershgorin: the same artifact, one step further downstream
# ----------------------------------------------------------------------


def _hessian(lo, hi=None):
    lo = np.asarray(lo, dtype=float)
    return Interval(lo, lo if hi is None else np.asarray(hi, dtype=float))


def test_gershgorin_bounds_of_a_zero_hessian_are_exactly_zero():
    """A linear function's Hessian is exactly 0; both bounds must say so.

    The unconditional nudge gave ``λ_min = -5e-324``, which fails a boundary
    ``λ_min >= 0`` convexity test on a function that is trivially convex.
    """
    H = _hessian(np.zeros((3, 3)))
    assert gershgorin_lambda_min(H) == 0.0
    assert gershgorin_lambda_max(H) == 0.0


def test_gershgorin_lower_bound_of_a_psd_boundary_hessian():
    """``diag(0, 2)`` with no off-diagonal: ``λ_min`` is exactly 0, not -5e-324."""
    H = _hessian(np.array([[0.0, 0.0], [0.0, 2.0]]))
    assert gershgorin_lambda_min(H) == 0.0


def test_gershgorin_still_rounds_outward_on_nonzero_bounds():
    """The soundness widening is untouched wherever the bound is nonzero."""
    H = _hessian(np.array([[1.0, 0.0], [0.0, 1.0]]))
    lam_min = gershgorin_lambda_min(H)
    lam_max = gershgorin_lambda_max(H)
    assert lam_min <= 1.0 and lam_max >= 1.0
    # An off-diagonal genuinely shrinks the lower bound.
    H2 = _hessian(np.array([[1.0, 0.5], [0.5, 1.0]]))
    assert gershgorin_lambda_min(H2) <= 0.5


def test_psd_2x2_zero_matrix_is_accepted():
    """The all-zero corner the hand-rolled zero check was written for."""
    assert psd_2x2_sufficient(_hessian(np.zeros((2, 2)))) is True
    assert psd_2x2_sufficient(_hessian(np.array([[0.0, 0.0], [0.0, 3.0]]))) is True


def test_psd_2x2_rejects_an_underflowed_off_diagonal():
    """``off**2`` underflowing to zero must NOT be treated as an exact zero.

    ``lo = diag(0, 1)`` with ``off = 1e-200`` has exact determinant
    ``0 * 1 - 1e-400 = -1e-400 < 0``, so the matrix is not PSD. Reading the
    flushed ``fl(off**2) == 0.0`` as an exact zero would have accepted it —
    an unsound certificate, and the inverse of the #957 artifact.
    """
    H = _hessian(np.array([[0.0, 1e-200], [1e-200, 1.0]]))
    assert float(np.asarray(H.lo)[0, 1]) ** 2 == 0.0, "premise: off**2 flushes to zero"
    assert psd_2x2_sufficient(H) is False


def test_psd_2x2_still_rejects_an_indefinite_matrix():
    assert psd_2x2_sufficient(_hessian(np.array([[1.0, 2.0], [2.0, 1.0]]))) is False
    assert psd_2x2_sufficient(_hessian(np.array([[-1.0, 0.0], [0.0, 1.0]]))) is False


# ----------------------------------------------------------------------
# The downstream consequence: the coefficient-spread guard
# ----------------------------------------------------------------------


def test_spread_guard_does_not_overflow_on_a_subnormal():
    """The old ``nz.max() / nz.min()`` overflowed to ``inf`` and warned."""
    data = np.array([5e-324, 1.0, 3.0])
    nz = np.abs(data[data != 0.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        naive = nz.max() / nz.min()
    assert not np.isfinite(naive), "premise: the naive ratio overflows"
    assert any("overflow" in str(w.message) for w in caught), "premise: it warns"

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any RuntimeWarning fails the test
        assert _coefficient_spread_exceeds(data, _RELAX_EQUILIBRATE_TRIGGER) is False


def test_spread_guard_measures_the_model_not_the_artifact():
    benign = np.array([1.0, 2.0, 3.0])
    assert _coefficient_spread_exceeds(benign, _RELAX_EQUILIBRATE_TRIGGER) is False
    # The same benign matrix plus one subnormal must still read as benign.
    with_artifact = np.array([1.0, 2.0, 3.0, -5e-324])
    assert _coefficient_spread_exceeds(with_artifact, _RELAX_EQUILIBRATE_TRIGGER) is False


def test_spread_guard_still_fires_on_genuine_ill_conditioning():
    ill = np.array([1e-9, 1e7])  # the lifted-McCormick spread the guard exists for
    assert _coefficient_spread_exceeds(ill, _RELAX_EQUILIBRATE_TRIGGER) is True
    # Right at the boundary the strict inequality is what decides.
    assert _coefficient_spread_exceeds(np.array([1.0, 1e6]), 1e6) is False
    assert _coefficient_spread_exceeds(np.array([1.0, 1.1e6]), 1e6) is True


def test_spread_guard_edge_cases():
    assert _coefficient_spread_exceeds(np.array([]), _RELAX_EQUILIBRATE_TRIGGER) is False
    assert _coefficient_spread_exceeds(np.zeros(4), _RELAX_EQUILIBRATE_TRIGGER) is False
    # Non-finite data is not something equilibration can help with.
    assert _coefficient_spread_exceeds(np.array([1.0, np.inf]), 1e6) is False
    assert _coefficient_spread_exceeds(np.array([1.0, np.nan]), 1e6) is False
    # A matrix of *only* subnormals has no measurable modelled spread.
    assert _coefficient_spread_exceeds(np.array([5e-324, 1e-320]), 1e6) is False
    assert _SUBNORMAL_FLOOR == TINY
