"""Tests for Optimality-Based Bound Tightening (OBBT)."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
from discopt._jax.obbt import (
    AuxTighteningReport,
    ObbtResult,
    _extract_linear_constraints,
    measure_discarded_aux_tightening,
    obbt_tighten_root,
    run_obbt,
)
from discopt.modeling.core import Model
from discopt.solvers import LPResult, SolveStatus

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _flat_size(model: Model) -> int:
    return sum(v.size for v in model._variables)


# ─────────────────────────────────────────────────────────────
# Test 1: Linear constraint extraction
# ─────────────────────────────────────────────────────────────


class TestLinearExtraction:
    def test_simple_inequality(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        y = m.continuous("y", lb=0, ub=100)
        m.minimize(x)
        m.subject_to(x + y <= 10)

        A_ub, b_ub, A_eq, b_eq, n_vars = _extract_linear_constraints(m)
        assert n_vars == 2
        assert A_ub is not None
        assert b_ub is not None
        assert A_ub.shape == (1, 2)
        assert np.isclose(A_ub[0, 0], 1.0)
        assert np.isclose(A_ub[0, 1], 1.0)
        assert np.isclose(b_ub[0], 10.0)

    def test_scaled_inequality(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        y = m.continuous("y", lb=0, ub=100)
        m.minimize(x)
        m.subject_to(2 * x + 3 * y <= 12)

        A_ub, b_ub, _, _, _ = _extract_linear_constraints(m)
        assert A_ub is not None
        assert np.isclose(A_ub[0, 0], 2.0)
        assert np.isclose(A_ub[0, 1], 3.0)
        assert np.isclose(b_ub[0], 12.0)

    def test_equality_constraint(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        y = m.continuous("y", lb=0, ub=100)
        m.minimize(x)
        m.subject_to(x + y == 5)

        _, _, A_eq, b_eq, _ = _extract_linear_constraints(m)
        assert A_eq is not None
        assert b_eq is not None
        assert A_eq.shape == (1, 2)
        assert np.isclose(b_eq[0], 5.0)

    def test_ge_constraint_converted(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        m.minimize(x)
        m.subject_to(x >= 5)

        A_ub, b_ub, _, _, _ = _extract_linear_constraints(m)
        assert A_ub is not None
        # x >= 5 becomes -x <= -5
        assert np.isclose(A_ub[0, 0], -1.0)
        assert np.isclose(b_ub[0], -5.0)

    def test_nonlinear_skipped(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        y = m.continuous("y", lb=0, ub=100)
        m.minimize(x)
        m.subject_to(x * y <= 10)  # Non-linear, should be skipped

        A_ub, b_ub, A_eq, b_eq, _ = _extract_linear_constraints(m)
        assert A_ub is None
        assert A_eq is None

    def test_mixed_linear_nonlinear(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        y = m.continuous("y", lb=0, ub=100)
        m.minimize(x)
        m.subject_to(x + y <= 10)  # Linear
        m.subject_to(x * y <= 50)  # Non-linear, skipped

        A_ub, b_ub, _, _, _ = _extract_linear_constraints(m)
        assert A_ub is not None
        assert A_ub.shape == (1, 2)  # Only the linear constraint

    def test_no_constraints(self):
        m = Model("test")
        m.continuous("x", lb=0, ub=100)
        m.minimize(m._variables[0])

        A_ub, b_ub, A_eq, b_eq, _ = _extract_linear_constraints(m)
        assert A_ub is None
        assert A_eq is None

    def test_with_constant_offset(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        m.minimize(x)
        m.subject_to(x + 5 <= 15)  # Should give A=[1], b=[10]

        A_ub, b_ub, _, _, _ = _extract_linear_constraints(m)
        assert A_ub is not None
        assert np.isclose(A_ub[0, 0], 1.0)
        assert np.isclose(b_ub[0], 10.0)


# ─────────────────────────────────────────────────────────────
# Test 2: OBBT basic functionality
# ─────────────────────────────────────────────────────────────


class TestObbtBasic:
    def test_simple_bound_tightening(self):
        """x + y <= 10, x,y >= 0 with initial bounds [0,100].
        OBBT should tighten to x <= 10, y <= 10."""
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        y = m.continuous("y", lb=0, ub=100)
        m.minimize(x)
        m.subject_to(x + y <= 10)

        result = run_obbt(m)
        assert isinstance(result, ObbtResult)
        assert result.n_lp_solves > 0
        assert result.n_tightened > 0
        assert np.isclose(result.tightened_ub[0], 10.0, atol=1e-6)
        assert np.isclose(result.tightened_ub[1], 10.0, atol=1e-6)
        assert np.isclose(result.tightened_lb[0], 0.0, atol=1e-6)
        assert np.isclose(result.tightened_lb[1], 0.0, atol=1e-6)

    def test_two_constraints(self):
        """x + 2y <= 10, 3x + y <= 12, x,y >= 0."""
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        y = m.continuous("y", lb=0, ub=100)
        m.minimize(x)
        m.subject_to(x + 2 * y <= 10)
        m.subject_to(3 * x + y <= 12)

        result = run_obbt(m)
        # Optimal x_max: max x s.t. x + 2y <= 10, 3x + y <= 12, x,y >= 0
        # At y = 0: x <= min(10, 4) = 4
        assert result.tightened_ub[0] <= 4.0 + 1e-6
        # Optimal y_max: max y s.t. x + 2y <= 10, 3x + y <= 12, x,y >= 0
        # At x = 0: y <= min(5, 12) = 5
        assert result.tightened_ub[1] <= 5.0 + 1e-6

    def test_equality_constraint_tightening(self):
        """x + y = 5, x,y >= 0, x,y <= 100."""
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        y = m.continuous("y", lb=0, ub=100)
        m.minimize(x)
        m.subject_to(x + y == 5)

        result = run_obbt(m)
        assert np.isclose(result.tightened_ub[0], 5.0, atol=1e-6)
        assert np.isclose(result.tightened_ub[1], 5.0, atol=1e-6)

    def test_no_tightening_when_already_tight(self):
        """If bounds are already tight, OBBT should not change them."""
        m = Model("test")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x)
        m.subject_to(x + y <= 20)  # Doesn't help tighten [0,10]

        result = run_obbt(m)
        assert np.isclose(result.tightened_lb[0], 0.0, atol=1e-6)
        assert np.isclose(result.tightened_ub[0], 10.0, atol=1e-6)

    def test_lower_bound_tightening(self):
        """x >= 5 should tighten lb to 5."""
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        m.minimize(x)
        m.subject_to(x >= 5)

        result = run_obbt(m)
        assert np.isclose(result.tightened_lb[0], 5.0, atol=1e-6)

    def test_total_time_limit_stops_before_all_variables(self, monkeypatch):
        """The total OBBT deadline should cap the full candidate loop."""
        import discopt._jax.obbt as obbt_mod

        m = Model("deadline")
        x = m.continuous("x", lb=0, ub=100, shape=(3,))
        m.minimize(x[0])
        m.subject_to(x[0] + x[1] + x[2] <= 10)

        clock = {"now": 100.0}
        calls = []

        monkeypatch.setattr(obbt_mod.time, "perf_counter", lambda: clock["now"])

        def fake_solve_lp(*, c, time_limit=None, **kwargs):
            del c, kwargs
            calls.append(time_limit)
            clock["now"] += 0.11
            return LPResult(status=SolveStatus.OPTIMAL, objective=0.0, wall_time=0.11)

        # OBBT resolves its LP oracle through the exact-oracle seams (it must use
        # an exact simplex backend for sound tightening — see #145); since C-15 it
        # prefers ``get_exact_dual_lp_solver`` (vertex duals feed the NS-safe
        # clamp), falling back to ``get_exact_lp_solver``. Patch both.
        monkeypatch.setattr(obbt_mod, "get_exact_dual_lp_solver", lambda: fake_solve_lp)
        monkeypatch.setattr(obbt_mod, "get_exact_lp_solver", lambda: fake_solve_lp)

        result = run_obbt(m, time_limit_per_lp=1.0, total_time_limit=0.2)

        assert result.n_lp_solves == 2
        assert len(calls) == 2
        assert np.isclose(calls[0], 0.2)
        assert 0.0 < calls[1] < 0.1


# ─────────────────────────────────────────────────────────────
# Test 3: OBBT with custom initial bounds
# ─────────────────────────────────────────────────────────────


class TestObbtCustomBounds:
    def test_custom_initial_bounds(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        y = m.continuous("y", lb=0, ub=100)
        m.minimize(x)
        m.subject_to(x + y <= 10)

        # Provide tighter initial bounds
        lb = np.array([2.0, 0.0])
        ub = np.array([50.0, 50.0])

        result = run_obbt(m, lb=lb, ub=ub)
        assert result.tightened_lb[0] >= 2.0 - 1e-6
        assert result.tightened_ub[0] <= 10.0 + 1e-6


# ─────────────────────────────────────────────────────────────
# Test 4: OBBT result statistics
# ─────────────────────────────────────────────────────────────


class TestObbtStatistics:
    def test_lp_count(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        y = m.continuous("y", lb=0, ub=100)
        m.minimize(x)
        m.subject_to(x + y <= 10)

        result = run_obbt(m)
        # 2 variables * 2 LPs each = 4 LP solves
        assert result.n_lp_solves == 4

    def test_wall_time_positive(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        y = m.continuous("y", lb=0, ub=100)
        m.minimize(x)
        m.subject_to(x + y <= 10)

        result = run_obbt(m)
        assert result.total_lp_time >= 0.0

    def test_no_constraints_no_solves(self):
        m = Model("test")
        m.continuous("x", lb=0, ub=100)
        m.minimize(m._variables[0])

        result = run_obbt(m)
        assert result.n_lp_solves == 0
        assert result.n_tightened == 0


# ─────────────────────────────────────────────────────────────
# Test 5: OBBT with multiple variable types
# ─────────────────────────────────────────────────────────────


class TestObbtMultipleVarTypes:
    def test_with_binary_and_continuous(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        y = m.binary("y")
        m.minimize(x)
        m.subject_to(x <= 50 * y)  # x <= 50*y

        result = run_obbt(m)
        # With y in [0,1], x can be at most 50
        assert result.tightened_ub[0] <= 50.0 + 1e-6

    def test_three_variable_system(self):
        """x + y + z <= 15, 2x + z <= 10, x,y,z >= 0."""
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        y = m.continuous("y", lb=0, ub=100)
        z = m.continuous("z", lb=0, ub=100)
        m.minimize(x)
        m.subject_to(x + y + z <= 15)
        m.subject_to(2 * x + z <= 10)

        result = run_obbt(m)
        # x_max: max x s.t. x+y+z <= 15, 2x+z <= 10
        # At y=0, z=0: x <= min(15, 5) = 5
        assert result.tightened_ub[0] <= 5.0 + 1e-6
        # y_max: max y at x=0, z=0: y <= 15
        assert result.tightened_ub[1] <= 15.0 + 1e-6
        # z_max: max z at x=0, y=0: z <= min(15, 10) = 10
        assert result.tightened_ub[2] <= 10.0 + 1e-6


# ─────────────────────────────────────────────────────────────
# Test 6: OBBT soundness - tightened bounds are valid
# ─────────────────────────────────────────────────────────────


class TestObbtSoundness:
    def test_tightened_bounds_contain_feasible_region(self):
        """Verify that feasible points remain inside tightened bounds."""
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        y = m.continuous("y", lb=0, ub=100)
        m.minimize(x)
        m.subject_to(x + y <= 10)
        m.subject_to(x >= 2)

        result = run_obbt(m)

        # Check various feasible points
        feasible_points = [
            (2.0, 0.0),
            (5.0, 5.0),
            (3.0, 7.0),
            (10.0, 0.0),
            (2.0, 8.0),
        ]
        for xv, yv in feasible_points:
            if xv + yv <= 10.0 + 1e-8 and xv >= 2.0 - 1e-8:
                assert xv >= result.tightened_lb[0] - 1e-6
                assert xv <= result.tightened_ub[0] + 1e-6
                assert yv >= result.tightened_lb[1] - 1e-6
                assert yv <= result.tightened_ub[1] + 1e-6

    def test_bounds_monotone_tightening(self):
        """OBBT should only tighten, never loosen bounds."""
        m = Model("test")
        x = m.continuous("x", lb=0, ub=50)
        y = m.continuous("y", lb=0, ub=50)
        m.minimize(x)
        m.subject_to(x + y <= 10)

        result = run_obbt(m)
        assert result.tightened_lb[0] >= 0.0 - 1e-8
        assert result.tightened_ub[0] <= 50.0 + 1e-8
        assert result.tightened_lb[1] >= 0.0 - 1e-8
        assert result.tightened_ub[1] <= 50.0 + 1e-8


# ─────────────────────────────────────────────────────────────
# Test 7: OBBT with warm-starting
# ─────────────────────────────────────────────────────────────


class TestObbtWarmStart:
    def test_warm_start_used(self):
        """Verify OBBT uses warm-starting (should be faster on 2nd run)."""
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        y = m.continuous("y", lb=0, ub=100)
        z = m.continuous("z", lb=0, ub=100)
        m.minimize(x)
        m.subject_to(x + y + z <= 15)
        m.subject_to(2 * x + z <= 10)
        m.subject_to(y + 2 * z <= 12)

        result = run_obbt(m)
        # Just verify it completes and produces valid results
        assert result.n_lp_solves == 6  # 3 vars * 2 LPs
        assert result.n_tightened > 0


# ─────────────────────────────────────────────────────────────
# Test 8: Edge cases
# ─────────────────────────────────────────────────────────────


class TestObbtEdgeCases:
    def test_single_variable(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        m.minimize(x)
        m.subject_to(x <= 10)

        result = run_obbt(m)
        assert np.isclose(result.tightened_ub[0], 10.0, atol=1e-6)

    def test_fixed_variable_skipped(self):
        """Variables with lb == ub should be skipped."""
        m = Model("test")
        x = m.continuous("x", lb=5, ub=5)
        y = m.continuous("y", lb=0, ub=100)
        m.minimize(y)
        m.subject_to(x + y <= 10)

        result = run_obbt(m)
        # x is fixed, only y should be tightened
        assert result.n_lp_solves == 2  # Only y: min and max
        assert np.isclose(result.tightened_ub[1], 5.0, atol=1e-6)

    def test_subtraction_constraint(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        y = m.continuous("y", lb=0, ub=100)
        m.minimize(x)
        m.subject_to(x - y <= 5)

        result = run_obbt(m)
        # x - y <= 5 with x,y >= 0
        # x_max at y = 100: x <= 105 (but model ub is 100)
        # So no tightening on x_ub
        # y has no upper bound constraint -> y_ub stays at 100
        assert result.tightened_ub[0] <= 100.0 + 1e-6

    def test_division_by_constant_constraint(self):
        m = Model("test")
        x = m.continuous("x", lb=0, ub=100)
        m.minimize(x)
        m.subject_to(x / 2 <= 5)

        result = run_obbt(m)
        assert np.isclose(result.tightened_ub[0], 10.0, atol=1e-6)


# ─────────────────────────────────────────────────────────────
# Test 9: Incumbent cutoff (Phase C)
# ─────────────────────────────────────────────────────────────


class TestObbtIncumbentCutoff:
    def test_cutoff_tightens_bounds(self):
        """Incumbent cutoff should tighten bounds beyond standard OBBT."""
        m = Model("cutoff")
        x = m.continuous("x", lb=0, ub=100)
        y = m.continuous("y", lb=0, ub=100)
        m.minimize(x + y)
        m.subject_to(x + y >= 5)

        # Without cutoff: x in [0, 100], y in [0, 100]
        r1 = run_obbt(m)
        # With cutoff z*=20: x+y <= 20
        r2 = run_obbt(m, incumbent_cutoff=20.0)

        # With cutoff, ub should be tightened
        assert r2.tightened_ub[0] <= 20.0 + 1e-6
        assert r2.tightened_ub[1] <= 20.0 + 1e-6
        # More tightened than without cutoff
        assert r2.n_tightened >= r1.n_tightened

    def test_cutoff_preserves_soundness(self):
        """Cutoff-tightened bounds must contain all points with obj <= z*."""
        m = Model("sound")
        x = m.continuous("x", lb=0, ub=50)
        y = m.continuous("y", lb=0, ub=50)
        m.minimize(2 * x + 3 * y)
        m.subject_to(x + y <= 30)
        m.subject_to(x >= 5)

        cutoff = 40.0  # 2x + 3y <= 40
        result = run_obbt(m, incumbent_cutoff=cutoff)

        # All feasible points with obj <= 40 must be inside bounds
        # x=5, y=10 -> obj=40 (boundary)
        assert 5.0 >= result.tightened_lb[0] - 1e-6
        assert 10.0 <= result.tightened_ub[1] + 1e-6

    def test_cutoff_no_effect_when_loose(self):
        """A very loose cutoff should not affect bounds."""
        m = Model("loose")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x)
        m.subject_to(x <= 5)

        r_no = run_obbt(m)
        r_yes = run_obbt(m, incumbent_cutoff=1000.0)

        np.testing.assert_allclose(r_no.tightened_ub, r_yes.tightened_ub, atol=1e-6)

    def test_cutoff_nonlinear_objective_ignored(self):
        """Nonlinear objectives should be gracefully skipped."""
        m = Model("nonlin")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x**2)
        m.subject_to(x <= 5)

        # Should not crash, just skip cutoff
        result = run_obbt(m, incumbent_cutoff=25.0)
        assert result.tightened_ub[0] <= 5.0 + 1e-6


# ─────────────────────────────────────────────────────────────
# Root OBBT over the McCormick relaxation (range reduction)
# ─────────────────────────────────────────────────────────────


class TestRootObbt:
    """obbt_tighten_root tightens via the nonlinear relaxation polytope.

    Unlike run_obbt (linear constraints only), this sees the bilinear /
    monomial envelopes, so it reduces ranges even when the only constraint is
    nonlinear. Every tightening is a valid outer-approximation deduction, so
    the returned box is always a subset of the input box.
    """

    def _bilinear_model(self):
        # min x+y s.t. x*y <= 5, x in [4,10], y in [0,10].
        # McCormick: x*y >= xlo*y = 4y, so 4y <= 5 => y <= 1.25.
        m = Model("bil")
        m.continuous("x", lb=4, ub=10)
        m.continuous("y", lb=0, ub=10)
        x, y = m._variables[0], m._variables[1]
        m.minimize(x + y)
        m.subject_to(x * y <= 5)
        return m

    def test_tightens_nonlinear_only_bound(self):
        m = self._bilinear_model()
        lb = np.array([4.0, 0.0])
        ub = np.array([10.0, 10.0])
        r = obbt_tighten_root(m, lb, ub)
        assert r.n_tightened >= 1
        assert not r.infeasible
        # y's upper bound is reduced to the envelope-implied 1.25.
        assert r.ub[1] <= 2.0
        assert abs(r.ub[1] - 1.25) <= 1e-4

    def test_never_loosens_bounds(self):
        m = self._bilinear_model()
        lb = np.array([4.0, 0.0])
        ub = np.array([10.0, 10.0])
        r = obbt_tighten_root(m, lb, ub)
        # The returned box must be a subset of the input box (sound).
        assert np.all(r.lb >= lb - 1e-9)
        assert np.all(r.ub <= ub + 1e-9)
        assert np.all(r.lb <= r.ub + 1e-9)

    def test_linear_model_is_noop(self):
        # No relaxable nonlinearity -> nothing for relaxation OBBT to tighten.
        m = Model("lin")
        m.continuous("x", lb=0, ub=10)
        m.continuous("y", lb=0, ub=10)
        x, y = m._variables[0], m._variables[1]
        m.minimize(x + y)
        m.subject_to(x + y <= 5)
        lb = np.array([0.0, 0.0])
        ub = np.array([10.0, 10.0])
        r = obbt_tighten_root(m, lb, ub)
        assert r.n_tightened == 0
        np.testing.assert_allclose(r.lb, lb)
        np.testing.assert_allclose(r.ub, ub)

    def test_integer_bounds_rounded_inward(self):
        # The integer factor's tightened bound must be rounded to an integer.
        # z integer in [0,10], w continuous in [3,10], z*w <= 5 with w>=3
        # => z <= 5/3 = 1.66.., rounded inward to z <= 1.
        m = Model("intbil")
        m.integer("z", lb=0, ub=10)
        m.continuous("w", lb=3, ub=10)
        z, w = m._variables[0], m._variables[1]
        m.minimize(z + w)
        m.subject_to(z * w <= 5)
        lb = np.array([0.0, 3.0])
        ub = np.array([10.0, 10.0])
        r = obbt_tighten_root(m, lb, ub)
        # z's upper bound is an integer and at most 1.
        assert r.ub[0] == np.floor(r.ub[0])
        assert r.ub[0] <= 1.0

    def test_returns_input_box_on_unbounded(self):
        # An open box can't build finite McCormick envelopes; return unchanged
        # rather than raising, so the solve stays sound.
        m = self._bilinear_model()
        lb = np.array([4.0, 0.0])
        ub = np.array([10.0, np.inf])
        r = obbt_tighten_root(m, lb, ub)
        assert not r.infeasible
        # Must not loosen and must not crash.
        assert np.all(r.lb >= lb - 1e-9)


# ─────────────────────────────────────────────────────────────
# OBBT-on-auxiliaries diagnostic (#208 decision gate)
# ─────────────────────────────────────────────────────────────


class TestMeasureDiscardedAuxTightening:
    """The pure-diagnostic measurement of aux-column tightening (#208)."""

    def _bilinear(self):
        m = Model("bil")
        m.continuous("x", lb=0, ub=4)
        m.continuous("y", lb=0, ub=4)
        m.minimize(m._variables[0] * m._variables[1])
        m.subject_to(m._variables[0] + m._variables[1] <= 5)
        return m

    def test_reports_aux_tightening_on_bilinear(self):
        m = self._bilinear()
        lb = np.array([0.0, 0.0])
        ub = np.array([4.0, 4.0])
        rep = measure_discarded_aux_tightening(m, lb, ub)
        assert isinstance(rep, AuxTighteningReport)
        # One product aux w = x*y, and the x+y<=5 facet shrinks its envelope box.
        assert rep.n_aux >= 1
        assert rep.n_aux_tightened >= 1
        assert 0.0 < rep.mean_rel_reduction <= 1.0
        assert rep.max_rel_reduction >= rep.mean_rel_reduction
        assert len(rep.aux_rel_reductions) == rep.n_aux

    def test_cutoff_tightens_at_least_as_much(self):
        m = self._bilinear()
        lb = np.array([0.0, 0.0])
        ub = np.array([4.0, 4.0])
        struct = measure_discarded_aux_tightening(m, lb, ub)
        withcut = measure_discarded_aux_tightening(m, lb, ub, incumbent_cutoff=6.0)
        # Adding an objective cutoff can only add constraints -> >= tightening.
        assert withcut.max_rel_reduction >= struct.max_rel_reduction - 1e-9

    def test_none_on_linear_model(self):
        # No relaxable nonlinearity -> no aux columns -> nothing to measure.
        m = Model("lin")
        m.continuous("x", lb=0, ub=10)
        m.continuous("y", lb=0, ub=10)
        m.minimize(m._variables[0] + m._variables[1])
        m.subject_to(m._variables[0] + m._variables[1] <= 5)
        lb = np.array([0.0, 0.0])
        ub = np.array([10.0, 10.0])
        assert measure_discarded_aux_tightening(m, lb, ub) is None

    def test_none_on_open_box(self):
        # An open box can't build finite McCormick envelopes -> None (no crash).
        m = self._bilinear()
        lb = np.array([0.0, 0.0])
        ub = np.array([4.0, np.inf])
        assert measure_discarded_aux_tightening(m, lb, ub) is None

    def test_full_result_returns_aux_columns(self):
        # full_result=True must expose the aux columns the default path slices off.
        from discopt._jax.mccormick_lp import MccormickLPRelaxer, build_milp_relaxation
        from discopt._jax.obbt import run_obbt_on_relaxation

        m = self._bilinear()
        lb = np.array([0.0, 0.0])
        ub = np.array([4.0, 4.0])
        relaxer = MccormickLPRelaxer(m)
        milp, _ = build_milp_relaxation(
            relaxer._model, relaxer._terms, relaxer._disc, bound_override=(lb, ub)
        )
        n_orig = relaxer._n_orig
        n_total = len(milp._bounds)
        default = run_obbt_on_relaxation(milp, n_orig)
        full = run_obbt_on_relaxation(milp, n_orig, full_result=True)
        assert len(default.tightened_lb) == n_orig
        assert len(full.tightened_lb) == n_total
        # The original-column block is identical between the two return modes.
        np.testing.assert_allclose(full.tightened_lb[:n_orig], default.tightened_lb)
        np.testing.assert_allclose(full.tightened_ub[:n_orig], default.tightened_ub)


# ─────────────────────────────────────────────────────────────
# Reverse-FBBT aux cascade (#208 part 2)
# ─────────────────────────────────────────────────────────────


class TestReverseFbbtFromAux:
    """The reverse-FBBT propagation of tightened aux bounds (#208)."""

    def test_bilinear_hyperbolic_bound(self):
        # w = x*y, y in [2,4], a tightened w <= 6 implies x <= 6/2 = 3 (a bound
        # the linear McCormick rows cannot express). x starts at [0,10].
        from discopt._jax.obbt import reverse_fbbt_from_aux

        lb = np.array([0.0, 2.0, 0.0])  # x, y, w-col placeholder (orig only uses 0,1)
        ub = np.array([10.0, 4.0, 0.0])
        # varmap: bilinear (x=0, y=1) -> aux col 2
        varmap = {"bilinear": {(0, 1): 2}, "monomial": {}}
        aux_lb = np.array([0.0, 0.0, 0.0])
        aux_ub = np.array([0.0, 0.0, 6.0])
        n = reverse_fbbt_from_aux(lb[:2], ub[:2], aux_lb, aux_ub, varmap)
        assert n >= 1
        assert ub[0] <= 3.0 + 1e-9  # x <= w_ub/y_lb = 6/2 = 3
        # y = w/x is NOT tightened: x in [0,10] straddles 0 -> division undefined.
        assert ub[1] == 4.0

    def test_bilinear_skips_zero_straddling_denominator(self):
        from discopt._jax.obbt import reverse_fbbt_from_aux

        lb = np.array([-5.0, -3.0])
        ub = np.array([5.0, 3.0])  # both straddle 0 -> division undefined
        varmap = {"bilinear": {(0, 1): 2}, "monomial": {}}
        aux_lb = np.array([0.0, 0.0, -4.0])
        aux_ub = np.array([0.0, 0.0, 4.0])
        n = reverse_fbbt_from_aux(lb, ub, aux_lb, aux_ub, varmap)
        assert n == 0  # cannot divide by a zero-straddling interval
        assert ub[0] == 5.0 and lb[0] == -5.0

    def test_monomial_square_root_bound(self):
        # w = x**2, x in [0,10], tightened w <= 9 implies x <= 3.
        from discopt._jax.obbt import reverse_fbbt_from_aux

        lb = np.array([0.0])
        ub = np.array([10.0])
        varmap = {"bilinear": {}, "monomial": {(0, 2): 1}}
        aux_lb = np.array([0.0, 0.0])
        aux_ub = np.array([0.0, 9.0])
        n = reverse_fbbt_from_aux(lb, ub, aux_lb, aux_ub, varmap)
        assert n >= 1
        assert abs(ub[0] - 3.0) <= 1e-9

    def test_reverse_fbbt_only_tightens(self):
        # A loose aux bound must never loosen the original box.
        from discopt._jax.obbt import reverse_fbbt_from_aux

        lb = np.array([1.0, 2.0])
        ub = np.array([3.0, 4.0])
        varmap = {"bilinear": {(0, 1): 2}, "monomial": {}}
        aux_lb = np.array([0.0, 0.0, -100.0])
        aux_ub = np.array([0.0, 0.0, 100.0])  # very loose
        reverse_fbbt_from_aux(lb, ub, aux_lb, aux_ub, varmap)
        assert lb[0] >= 1.0 - 1e-12 and ub[0] <= 3.0 + 1e-12
        assert lb[1] >= 2.0 - 1e-12 and ub[1] <= 4.0 + 1e-12

    def test_trilinear_product_of_others_bound(self):
        # w = x*y*z, y in [2,4], z in [1,2] -> y*z in [2,8]; a tightened w <= 12
        # implies x <= 12/2 = 6 (a hyperbolic bound the linear rows can't express).
        from discopt._jax.obbt import reverse_fbbt_from_aux

        lb = np.array([0.0, 2.0, 1.0])
        ub = np.array([10.0, 4.0, 2.0])
        varmap = {"trilinear": {(0, 1, 2): 3}}
        aux_lb = np.array([0.0, 0.0, 0.0, 0.0])
        aux_ub = np.array([0.0, 0.0, 0.0, 12.0])
        n = reverse_fbbt_from_aux(lb, ub, aux_lb, aux_ub, varmap)
        assert n >= 1
        assert ub[0] <= 6.0 + 1e-9  # x <= w_ub / (y*z)_lb = 12/2

    def test_multilinear_product_of_others_bound(self):
        # w = x0*x1*x2*x3, partners in [1,2] -> product-of-others in [1,8];
        # w <= 4 implies x0 <= 4/1 = 4.
        from discopt._jax.obbt import reverse_fbbt_from_aux

        lb = np.array([0.0, 1.0, 1.0, 1.0])
        ub = np.array([10.0, 2.0, 2.0, 2.0])
        varmap = {"multilinear": {(0, 1, 2, 3): 4}}
        aux_lb = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        aux_ub = np.array([0.0, 0.0, 0.0, 0.0, 4.0])
        n = reverse_fbbt_from_aux(lb, ub, aux_lb, aux_ub, varmap)
        assert n >= 1
        assert ub[0] <= 4.0 + 1e-9

    def test_ratio_numerator_and_denominator_bounds(self):
        # w = x / y. y in [2,4], tightened w <= 1 implies x = w*y <= 4.
        from discopt._jax.obbt import reverse_fbbt_from_aux

        lb = np.array([0.0, 2.0])
        ub = np.array([10.0, 4.0])
        varmap = {"ratio": {((0,), (1,)): 2}}
        aux_lb = np.array([0.0, 0.0, 0.0])
        aux_ub = np.array([0.0, 0.0, 1.0])
        n = reverse_fbbt_from_aux(lb, ub, aux_lb, aux_ub, varmap)
        assert n >= 1
        assert ub[0] <= 4.0 + 1e-9  # x <= w_ub * y_ub = 1*4

    def test_ratio_denominator_tightens(self):
        # w = x / y, x fixed at 4, w in [2,10] implies y = x/w in [0.4, 2].
        from discopt._jax.obbt import reverse_fbbt_from_aux

        lb = np.array([4.0, 0.1])
        ub = np.array([4.0, 5.0])
        varmap = {"ratio": {((0,), (1,)): 2}}
        aux_lb = np.array([0.0, 0.0, 2.0])
        aux_ub = np.array([0.0, 0.0, 10.0])
        n = reverse_fbbt_from_aux(lb, ub, aux_lb, aux_ub, varmap)
        assert n >= 1
        assert lb[1] >= 0.4 - 1e-9  # y >= x_lb / w_ub = 4/10
        assert ub[1] <= 2.0 + 1e-9  # y <= x_ub / w_lb = 4/2

    def test_higher_arity_never_loosens_and_is_sound(self):
        # Randomized soundness: every (x,y,z) whose product lands in the aux box
        # must remain inside the tightened box (reverse FBBT only removes points
        # that cannot satisfy w = x*y*z for the given aux bounds).
        from discopt._jax.obbt import reverse_fbbt_from_aux

        rng = np.random.default_rng(0)
        for _ in range(200):
            lb = np.array([rng.uniform(-3, 0), rng.uniform(0.5, 2), rng.uniform(1, 2)])
            ub = np.array([rng.uniform(1, 4), rng.uniform(2, 4), rng.uniform(2, 3)])
            wl, wu = sorted(rng.uniform(-20, 20, size=2))
            varmap = {"trilinear": {(0, 1, 2): 3}}
            aux_lb = np.array([0.0, 0.0, 0.0, wl])
            aux_ub = np.array([0.0, 0.0, 0.0, wu])
            tl, tu = lb.copy(), ub.copy()
            reverse_fbbt_from_aux(tl, tu, aux_lb, aux_ub, varmap)
            # Tightened box is a subset of the original.
            assert np.all(tl >= lb - 1e-9) and np.all(tu <= ub + 1e-9)
            # No feasible (in-box, in-aux) point is cut.
            for _ in range(40):
                p = np.array([rng.uniform(lb[k], ub[k]) for k in range(3)])
                w = p[0] * p[1] * p[2]
                if wl - 1e-12 <= w <= wu + 1e-12:
                    assert np.all(p >= tl - 1e-9) and np.all(p <= tu + 1e-9)

    def test_cascade_preserves_box_subset_and_optimum(self):
        # End to end: cascade_aux must keep the box a subset and never remove the
        # bilinear optimum. min x+y s.t. x*y >= 4, x,y in [0.5,4].
        m = Model("bil")
        m.continuous("x", lb=0.5, ub=4.0)
        m.continuous("y", lb=0.5, ub=4.0)
        x, y = m._variables[0], m._variables[1]
        m.minimize(x + y)
        m.subject_to(x * y >= 4.0)
        lb = np.array([0.5, 0.5])
        ub = np.array([4.0, 4.0])
        off = obbt_tighten_root(m, lb.copy(), ub.copy(), cascade_aux=False)
        on = obbt_tighten_root(m, lb.copy(), ub.copy(), cascade_aux=True)
        # Both are sound subsets of the input box.
        for r in (off, on):
            assert np.all(r.lb >= lb - 1e-9) and np.all(r.ub <= ub + 1e-9)
            assert not r.infeasible
        # The true optimum x=y=2 (obj 4) must remain inside the cascade box.
        assert on.lb[0] <= 2.0 + 1e-6 <= on.ub[0]
        assert on.lb[1] <= 2.0 + 1e-6 <= on.ub[1]


# ─────────────────────────────────────────────────────────────
# #208 aux-cascade budget: only probe reverse-FBBT-reachable aux columns
# ─────────────────────────────────────────────────────────────


class TestCascadeReachableAux:
    """`cascade_reachable_aux` selects exactly the aux columns whose reverse-FBBT
    could tighten an original — a *superset* of the reverse-FBBT deduction guards,
    so restricting the OBBT aux candidate set to it is bound-neutral (#208)."""

    def test_bilinear_reachable_when_partner_sign_definite(self):
        from discopt._jax.obbt import cascade_reachable_aux

        # w = x*y, x in [0,10] straddles 0, y in [2,4] excludes 0. b=w/a needs
        # 0 ∉ [x] (fails) but a=w/b needs 0 ∉ [y] (holds) -> aux reachable.
        lb = np.array([0.0, 2.0])
        ub = np.array([10.0, 4.0])
        varmap = {"bilinear": {(0, 1): 2}}
        assert cascade_reachable_aux(varmap, lb, ub, n_orig=2, n_total=3) == [2]

    def test_bilinear_unreachable_when_both_straddle_zero(self):
        from discopt._jax.obbt import cascade_reachable_aux

        # Both partners straddle 0 -> neither division is defined -> not reachable.
        lb = np.array([-5.0, -3.0])
        ub = np.array([5.0, 3.0])
        varmap = {"bilinear": {(0, 1): 2}}
        assert cascade_reachable_aux(varmap, lb, ub, n_orig=2, n_total=3) == []

    def test_monomial_always_reachable(self):
        from discopt._jax.obbt import cascade_reachable_aux

        # A p>=2 monomial deduces a root box with no divisor -> always reachable,
        # even when the base straddles zero.
        lb = np.array([-5.0])
        ub = np.array([5.0])
        varmap = {"monomial": {(0, 2): 1}}
        assert cascade_reachable_aux(varmap, lb, ub, n_orig=1, n_total=2) == [1]

    def test_ratio_reachable_only_when_a_factor_is_sign_definite(self):
        from discopt._jax.obbt import cascade_reachable_aux

        # w = x/y. Denominator y in [2,4] excludes 0 -> numerator factor x is
        # reachable (x = w*y). Both original.
        lb = np.array([0.0, 2.0])
        ub = np.array([10.0, 4.0])
        varmap = {"ratio": {((0,), (1,)): 2}}
        assert cascade_reachable_aux(varmap, lb, ub, n_orig=2, n_total=3) == [2]

    def test_reachable_set_is_bound_neutral_on_the_corpus_shape(self):
        # The predicate must never *under*-include: any aux from which
        # reverse_fbbt_from_aux actually tightens an original MUST be reachable.
        # Cross-check on a randomized varmap of every term family.
        from discopt._jax.obbt import cascade_reachable_aux, reverse_fbbt_from_aux

        rng = np.random.default_rng(7)
        for _ in range(300):
            n_orig = 3
            lb = np.array([rng.uniform(-3, 1) for _ in range(n_orig)])
            ub = np.array([lb[k] + rng.uniform(0.5, 4) for k in range(n_orig)])
            cw = n_orig  # single aux column
            n_total = n_orig + 1
            kind = rng.integers(0, 4)
            if kind == 0:
                varmap = {"bilinear": {(0, 1): cw}}
            elif kind == 1:
                varmap = {"monomial": {(0, int(rng.integers(2, 4))): cw}}
            elif kind == 2:
                varmap = {"trilinear": {(0, 1, 2): cw}}
            else:
                varmap = {"ratio": {((0,), (1,)): cw}}
            wl, wu = sorted(rng.uniform(-10, 10, size=2))
            aux_lb = np.append(np.zeros(n_orig), wl)
            aux_ub = np.append(np.zeros(n_orig), wu)
            reach = set(cascade_reachable_aux(varmap, lb, ub, n_orig, n_total))
            tl, tu = lb.copy(), ub.copy()
            n = reverse_fbbt_from_aux(tl, tu, aux_lb, aux_ub, varmap)
            if n > 0:  # reverse-FBBT tightened -> the aux MUST have been reachable
                assert cw in reach, (varmap, lb.tolist(), ub.tolist(), wl, wu)

    def test_targeted_candidate_set_matches_full_on_original_box(self):
        # End-to-end bound-neutrality: obbt_tighten_root(cascade_aux=True) with the
        # budgeted candidate set must return the SAME original box as it would with
        # every aux column probed. We assert the shipped (budgeted) path equals a
        # patched full-probe path on a mixed bilinear+monomial model.
        import discopt._jax.obbt as obbt_mod

        m = Model("mix")
        m.continuous("x", lb=0.5, ub=4.0)
        m.continuous("y", lb=1.0, ub=3.0)
        x, y = m._variables[0], m._variables[1]
        m.minimize(x + y)
        m.subject_to(x * y >= 3.0)
        m.subject_to(x * x <= 9.0)
        lb = np.array([0.5, 1.0])
        ub = np.array([4.0, 3.0])

        budgeted = obbt_tighten_root(m, lb.copy(), ub.copy(), cascade_aux=True)

        # Force the full-probe behavior by making the reachability predicate return
        # every aux column, then confirm the propagated original box is identical.
        orig = obbt_mod.cascade_reachable_aux

        def _all_aux(varmap, lo, hi, n_orig, n_total, eps=1e-7):
            return list(range(n_orig, n_total))

        obbt_mod.cascade_reachable_aux = _all_aux
        try:
            full = obbt_tighten_root(m, lb.copy(), ub.copy(), cascade_aux=True)
        finally:
            obbt_mod.cascade_reachable_aux = orig

        assert np.allclose(budgeted.lb, full.lb, atol=1e-9)
        assert np.allclose(budgeted.ub, full.ub, atol=1e-9)


class TestCascadeAuxGraduatedDefault:
    """#208 graduation: the reverse-FBBT aux cascade is default-ON on the real
    solve path (`DISCOPT_OBBT_CASCADE_AUX`, default `1`), with `=0` as the opt-out.
    Guards the flipped default so a future edit can't silently un-graduate it."""

    def _stub_model(self):
        m = Model("bil")
        m.continuous("x", lb=0.5, ub=4.0)
        m.continuous("y", lb=0.5, ub=4.0)
        x, y = m._variables[0], m._variables[1]
        m.minimize(x + y)
        m.subject_to(x * y >= 4.0)
        return m

    def test_default_on_and_optout_off(self, monkeypatch):
        import discopt._jax.obbt as obbt_mod
        from discopt._jax.obbt import RootObbtResult
        from discopt._jax.root_reduce import _stage_obbt

        captured = {}

        def _spy(model, lb, ub, **kw):
            captured["cascade_aux"] = kw.get("cascade_aux")
            return RootObbtResult(lb, ub, 0, 0, 0.0)

        monkeypatch.setattr(obbt_mod, "obbt_tighten_root", _spy)
        m = self._stub_model()
        lb, ub = np.array([0.5, 0.5]), np.array([4.0, 4.0])
        kw = dict(rounds=1, deadline=None, prefer_pounce=False, superposition=False)

        # Default (env unset) -> cascade ON (the graduated default).
        monkeypatch.delenv("DISCOPT_OBBT_CASCADE_AUX", raising=False)
        _stage_obbt(m, lb.copy(), ub.copy(), None, **kw)
        assert captured["cascade_aux"] is True

        # Explicit opt-out -> OFF (legacy path preserved).
        monkeypatch.setenv("DISCOPT_OBBT_CASCADE_AUX", "0")
        _stage_obbt(m, lb.copy(), ub.copy(), None, **kw)
        assert captured["cascade_aux"] is False


# ─────────────────────────────────────────────────────────────
# C-15: run_obbt must clamp the raw LP vertex to the NS-safe bound
# ─────────────────────────────────────────────────────────────


class TestC15NsSafeClamp:
    """`run_obbt` tightens through the Neumaier-Shcherbina safe bound, never the
    raw (possibly optimistic) LP vertex — the C-15 fix. Before the fix the
    model-linear variant applied ``result.objective`` directly, so an
    ill-conditioned solve reporting a vertex *above* the true projection could
    cut off feasible points.
    """

    def _fake_optimistic_solver(self):
        """An LP oracle that reports OPTIMAL with an *optimistic* objective whose
        duals (via NS) prove a much looser true bound. min-x claims obj 5.0 while
        the constraint is inactive (dual 0 ⇒ safe bound at the box lower bound);
        max-x claims max=3 while the box allows 10.
        """

        def _fake(
            c,
            A_ub=None,
            b_ub=None,
            A_eq=None,
            b_eq=None,
            bounds=None,
            warm_basis=None,
            time_limit=None,
            **kw,
        ):
            c = np.asarray(c, dtype=np.float64)
            n_rows = 0 if b_ub is None else int(np.size(b_ub))
            duals = np.zeros(n_rows, dtype=np.float64)  # claim all rows inactive
            # Direction: +1 on some column ⇒ min x_i; -1 ⇒ max x_i (min -x_i).
            if c.max() > 0:
                obj = 5.0  # optimistic min: true min is 0 at the box corner
            else:
                obj = -3.0  # optimistic max: min(-x)=-3 ⇒ max x=3, true max is 8
            return LPResult(
                status=SolveStatus.OPTIMAL,
                x=np.zeros(len(c)),
                objective=obj,
                dual_values=duals,
                reduced_costs=None,
                basis=None,
                iterations=0,
                wall_time=0.0,
                infeasibility_certificate=None,
            )

        return _fake

    def test_optimistic_vertex_is_clamped(self, monkeypatch):
        import discopt._jax.obbt as obbt_mod

        m = Model("c15")
        m.continuous("x", lb=0.0, ub=10.0)
        m.continuous("y", lb=0.0, ub=10.0)
        x, y = m._variables[0], m._variables[1]
        m.minimize(x)
        m.subject_to(x + y <= 8.0)  # a genuine 2-var row (won't fold to a bound)

        fake = self._fake_optimistic_solver()
        # Patch BOTH getters: pre-fix run_obbt used get_exact_lp_solver (no NS
        # clamp → would apply obj 5.0); post-fix uses get_exact_dual_lp_solver
        # (NS clamp → rejects it). Patching both makes this fail-before/pass-after.
        monkeypatch.setattr(obbt_mod, "get_exact_lp_solver", lambda: fake)
        monkeypatch.setattr(obbt_mod, "get_exact_dual_lp_solver", lambda: fake)

        result = run_obbt(m)

        # The optimistic vertex (5.0) is clamped to the NS-safe bound (~0), so the
        # feasible region x ∈ [0, 5) is NOT cut. Pre-fix this asserted 5.0.
        assert result.tightened_lb[0] < 1e-6, (
            f"raw optimistic vertex was trusted: lb={result.tightened_lb[0]}"
        )
        # Likewise the spurious max-tightening (to 3) is rejected; ub stays 10.
        assert result.tightened_ub[0] > 10.0 - 1e-6, (
            f"raw optimistic vertex cut the upper bound: ub={result.tightened_ub[0]}"
        )

    def test_ns_safe_lower_bound_free_equality_multiplier(self):
        """`_ns_safe_lp_lower_bound` treats trailing ``n_eq`` rows as equalities
        with a *free-sign* multiplier — sound and tighter than clamping them.
        LP: min x s.t. x == 2, x ∈ [0, 5] ⇒ true min = 2.
        """
        from discopt._jax.obbt import _ns_safe_lp_lower_bound

        c = np.array([1.0])
        A = np.array([[1.0]])
        b = np.array([2.0])
        lo = np.array([0.0])
        hi = np.array([5.0])
        dual = np.array([1.0])  # HiGHS convention y_eq = +1 (rc = c - Aᵀy_eq = 0)

        # Treated as an inequality (default): the clamp forces y = max(-1, 0) = 0,
        # giving the valid-but-loose bound 0.
        g_ineq = _ns_safe_lp_lower_bound(c, dual, A, b, lo, hi)
        assert g_ineq is not None
        assert abs(g_ineq - 0.0) < 1e-9

        # Treated as an equality (free multiplier): recovers the tight bound 2.
        g_eq = _ns_safe_lp_lower_bound(c, dual, A, b, lo, hi, n_eq=1)
        assert g_eq is not None
        assert abs(g_eq - 2.0) < 1e-6
        # Both must be rigorous under-estimates of the true min (= 2).
        assert g_ineq <= 2.0 + 1e-9 and g_eq <= 2.0 + 1e-9
