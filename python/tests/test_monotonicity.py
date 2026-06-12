"""Unit tests for per-expression monotonicity detection.

Exercises :func:`discopt._jax.monotonicity.classify_monotonicity` directly (the
SUSPECT head-to-head lives in ``test_monotonicity_suspect_parity.py``). The
contract under test:

* ``NONDECREASING`` / ``NONINCREASING`` are *proofs* obtained from an interval
  enclosure of the gradient over the variable box.
* ``CONSTANT`` means the expression does not vary with any decision variable.
* ``UNKNOWN`` is a sound abstention -- mixed gradient signs, an unsupported
  atom, a non-finite enclosure, or a domain issue.
"""

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.convexity.interval import Interval
from discopt._jax.monotonicity import Monotonicity, classify_monotonicity
from discopt.modeling.core import Constant, FunctionCall, Model, _wrap

ND = Monotonicity.NONDECREASING
NI = Monotonicity.NONINCREASING
C = Monotonicity.CONSTANT
U = Monotonicity.UNKNOWN


def _model_with(name="x", lb=0.0, ub=1.0):
    m = Model("t")
    x = m.continuous(name, lb=lb, ub=ub)
    return m, x


class TestLeaves:
    def test_constant_is_constant(self):
        assert classify_monotonicity(Constant(3.0)) == C

    def test_bare_variable_is_nondecreasing(self):
        _m, x = _model_with(lb=-2.0, ub=5.0)
        assert classify_monotonicity(x) == ND


class TestLinear:
    def test_positive_coefficient_nondecreasing(self):
        _m, x = _model_with(lb=-5.0, ub=5.0)
        assert classify_monotonicity(3.0 * x + 1.0) == ND

    def test_negative_coefficient_nonincreasing(self):
        _m, x = _model_with(lb=-5.0, ub=5.0)
        assert classify_monotonicity(-2.0 * x + 7.0) == NI

    def test_mixed_signs_two_vars_is_unknown(self):
        m = Model("t")
        x = m.continuous("x", lb=-5.0, ub=5.0)
        y = m.continuous("y", lb=-5.0, ub=5.0)
        # +x - y : nondecreasing in x, nonincreasing in y -> not jointly either
        assert classify_monotonicity(x - y) == U

    def test_both_positive_two_vars_nondecreasing(self):
        m = Model("t")
        x = m.continuous("x", lb=-5.0, ub=5.0)
        y = m.continuous("y", lb=-5.0, ub=5.0)
        assert classify_monotonicity(x + 2.0 * y) == ND

    def test_self_cancellation_is_not_falsely_directional(self):
        """``x - x`` has zero slope, but interval subtraction cannot prove the
        exact cancellation (it yields a sub-ULP enclosure around 0). The sound
        result is therefore UNKNOWN -- never a (wrong) proven direction."""
        _m, x = _model_with(lb=-5.0, ub=5.0)
        assert classify_monotonicity(x - x) in (C, U)


class TestMonotoneAtoms:
    def test_exp_nondecreasing(self):
        _m, x = _model_with(lb=-3.0, ub=3.0)
        assert classify_monotonicity(dm.exp(x)) == ND

    def test_log_nondecreasing(self):
        _m, x = _model_with(lb=0.05, ub=10.0)
        assert classify_monotonicity(dm.log(x)) == ND

    def test_reciprocal_nonincreasing_on_positive(self):
        _m, x = _model_with(lb=0.05, ub=10.0)
        assert classify_monotonicity(1.0 / x) == NI

    def test_sqrt_nondecreasing_strictly_positive(self):
        _m, x = _model_with(lb=0.1, ub=10.0)
        assert classify_monotonicity(dm.sqrt(x)) == ND

    def test_tanh_nondecreasing(self):
        _m, x = _model_with(lb=-3.0, ub=3.0)
        assert classify_monotonicity(dm.tanh(x)) == ND


class TestPowers:
    def test_even_power_straddling_zero_unknown(self):
        _m, x = _model_with(lb=-3.0, ub=3.0)
        assert classify_monotonicity(x**2) == U

    def test_even_power_positive_branch_nondecreasing(self):
        _m, x = _model_with(lb=0.5, ub=3.0)
        assert classify_monotonicity(x**2) == ND

    def test_negative_even_power_nonincreasing_on_positive(self):
        _m, x = _model_with(lb=0.1, ub=5.0)
        # x^-2 is decreasing on x > 0
        assert classify_monotonicity(x**-2) == NI


class TestInverseTrig:
    def test_asin_nondecreasing(self):
        _m, x = _model_with(lb=0.1, ub=0.9)
        expr = FunctionCall("asin", _wrap(x))
        assert classify_monotonicity(expr) == ND

    def test_acos_nonincreasing(self):
        _m, x = _model_with(lb=0.1, ub=0.9)
        expr = FunctionCall("acos", _wrap(x))
        assert classify_monotonicity(expr) == NI

    def test_atan_nondecreasing(self):
        _m, x = _model_with(lb=-3.0, ub=3.0)
        expr = FunctionCall("atan", _wrap(x))
        assert classify_monotonicity(expr) == ND


class TestSoundAbstentions:
    def test_abs_straddling_zero_unknown(self):
        _m, x = _model_with(lb=-5.0, ub=5.0)
        assert classify_monotonicity(abs(x)) == U

    def test_sine_wide_box_unknown(self):
        _m, x = _model_with(lb=-3.0, ub=3.0)
        assert classify_monotonicity(dm.sin(x)) == U

    def test_unsupported_atom_unknown(self):
        _m, x = _model_with(lb=0.1, ub=0.9)
        assert classify_monotonicity(FunctionCall("erf", _wrap(x))) == U

    def test_bilinear_unknown(self):
        m = Model("t")
        x = m.continuous("x", lb=-5.0, ub=5.0)
        y = m.continuous("y", lb=-5.0, ub=5.0)
        assert classify_monotonicity(x * y) == U


class TestBoxOverride:
    def test_box_override_changes_verdict(self):
        """Restricting abs to a positive branch via ``box`` proves the direction."""
        _m, x = _model_with(lb=-5.0, ub=5.0)
        # On the declared box [-5, 5], |x| is UNKNOWN ...
        assert classify_monotonicity(abs(x)) == U
        # ... but restricted to a strictly positive box it is nondecreasing.
        box = {x: Interval(np.float64(1.0), np.float64(5.0))}
        assert classify_monotonicity(abs(x), box=box) == ND


class TestSoundnessAgainstSampling:
    """A proven direction must match a dense numeric sampling of the body.

    Each case pairs the symbolic builder (fed to ``classify_monotonicity``) with
    the matching closed form (sampled with numpy), so the proof is checked
    against the real function, not just a label.
    """

    @pytest.mark.parametrize(
        "build, closed_form, lb, ub, expected",
        [
            (lambda x: dm.exp(x), np.exp, -2.0, 2.0, ND),
            (lambda x: 1.0 / x, lambda t: 1.0 / t, 0.1, 4.0, NI),
            (lambda x: dm.log(x), np.log, 0.2, 9.0, ND),
            (lambda x: -3.0 * x, lambda t: -3.0 * t, -4.0, 4.0, NI),
            (lambda x: dm.sqrt(x), np.sqrt, 0.1, 9.0, ND),
        ],
    )
    def test_direction_matches_samples(self, build, closed_form, lb, ub, expected):
        _m, x = _model_with(lb=lb, ub=ub)
        assert classify_monotonicity(build(x)) == expected

        ys = closed_form(np.linspace(lb, ub, 101))
        if expected is ND:
            assert np.all(np.diff(ys) >= -1e-9)
        else:
            assert np.all(np.diff(ys) <= 1e-9)
