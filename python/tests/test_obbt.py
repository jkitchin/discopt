"""Tests for Optimality-Based Bound Tightening (OBBT)."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
from discopt._jax.obbt import (
    ObbtResult,
    _extract_linear_constraints,
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

        monkeypatch.setattr(obbt_mod, "solve_lp", fake_solve_lp)

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
