"""
Tests for warm-start API.

Covers:
  - Valid initial solutions for continuous NLP, MILP, and MINLP models
  - Validation: wrong keys, wrong shapes, bounds violations, integrality violations
  - Warm-start produces results that match or beat cold-start quality
"""

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.warm_start import check_feasibility, validate_initial_solution

# ──────────────────────────────────────────────────────────
# Helpers — build small test models
# ──────────────────────────────────────────────────────────


def _make_nlp():
    """Simple convex NLP: minimize (x-3)^2 + (y-4)^2."""
    m = dm.Model("test_nlp")
    x = m.continuous("x", lb=-10, ub=10)
    y = m.continuous("y", lb=-10, ub=10)
    m.minimize((x - 3) ** 2 + (y - 4) ** 2)
    return m, x, y


def _make_milp():
    """Simple MILP: minimize -x - 2y, s.t. x + y <= 3, x,y binary."""
    m = dm.Model("test_milp")
    x = m.binary("x")
    y = m.binary("y")
    m.minimize(-x - 2 * y)
    m.subject_to(x + y <= 3, name="budget")
    return m, x, y


def _make_minlp():
    """Simple MINLP: minimize (x-2.5)^2 + (y-1)^2 with y binary."""
    m = dm.Model("test_minlp")
    x = m.continuous("x", lb=0, ub=5)
    y = m.binary("y")
    m.minimize((x - 2.5) ** 2 + (y - 1) ** 2)
    m.subject_to(x <= 4 * y + 1, name="linking")
    return m, x, y


def _make_array_model():
    """Model with array variables."""
    m = dm.Model("test_array")
    x = m.continuous("x", shape=(3,), lb=0, ub=10)
    m.minimize(dm.sum([(x[i] - (i + 1)) ** 2 for i in range(3)]))
    return m, x


# ──────────────────────────────────────────────────────────
# TestValidation — validate_initial_solution
# ──────────────────────────────────────────────────────────


class TestValidation:
    """Tests for initial-solution validation logic."""

    def test_valid_scalar_solution(self):
        m, x, y = _make_nlp()
        flat = validate_initial_solution(m, {x: 3.0, y: 4.0})
        assert flat.shape == (2,)
        np.testing.assert_allclose(flat, [3.0, 4.0])

    def test_valid_array_solution(self):
        m, x = _make_array_model()
        flat = validate_initial_solution(m, {x: [1.0, 2.0, 3.0]})
        assert flat.shape == (3,)
        np.testing.assert_allclose(flat, [1.0, 2.0, 3.0])

    def test_valid_numpy_array_solution(self):
        m, x = _make_array_model()
        flat = validate_initial_solution(m, {x: np.array([1.0, 2.0, 3.0])})
        np.testing.assert_allclose(flat, [1.0, 2.0, 3.0])

    def test_partial_solution_fills_defaults(self):
        m, x, y = _make_nlp()
        flat = validate_initial_solution(m, {x: 3.0})
        assert flat.shape == (2,)
        assert flat[0] == 3.0
        # y should be filled with midpoint of [-10, 10] = 0.0
        assert flat[1] == 0.0

    def test_non_dict_raises_type_error(self):
        m, x, y = _make_nlp()
        with pytest.raises(TypeError, match="must be a dict"):
            validate_initial_solution(m, [1.0, 2.0])

    def test_non_variable_key_raises_type_error(self):
        m, x, y = _make_nlp()
        with pytest.raises(TypeError, match="must be Variable objects"):
            validate_initial_solution(m, {"x": 3.0})

    def test_wrong_model_variable_raises_value_error(self):
        m1, x1, y1 = _make_nlp()
        m2, x2, y2 = _make_nlp()
        with pytest.raises(ValueError, match="not part of this model"):
            validate_initial_solution(m1, {x2: 3.0})

    def test_wrong_shape_raises_value_error(self):
        m, x = _make_array_model()
        with pytest.raises(ValueError, match="shape"):
            validate_initial_solution(m, {x: [1.0, 2.0]})

    def test_bounds_violation_warns_and_clamps(self):
        m, x, y = _make_nlp()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            flat = validate_initial_solution(m, {x: 20.0, y: -20.0})
            bound_warns = [ww for ww in w if "outside bounds" in str(ww.message)]
            assert len(bound_warns) >= 1
        # Values should be clamped
        assert flat[0] == 10.0
        assert flat[1] == -10.0

    def test_integrality_violation_warns_and_rounds(self):
        m, x, y = _make_milp()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            flat = validate_initial_solution(m, {x: 0.7, y: 0.3})
            int_warns = [ww for ww in w if "not integer-valued" in str(ww.message)]
            assert len(int_warns) >= 1
        # Values should be rounded
        assert flat[0] == 1.0
        assert flat[1] == 0.0

    def test_binary_valid_values(self):
        m, x, y = _make_milp()
        flat = validate_initial_solution(m, {x: 1.0, y: 0.0})
        np.testing.assert_allclose(flat, [1.0, 0.0])


# ──────────────────────────────────────────────────────────
# TestFeasibilityCheck — check_feasibility
# ──────────────────────────────────────────────────────────


class TestFeasibilityCheck:
    """Tests for feasibility checking of flat solution vectors."""

    def test_feasible_nlp(self):
        m, x, y = _make_nlp()
        flat = np.array([3.0, 4.0])
        is_feas, viols = check_feasibility(m, flat)
        assert is_feas
        assert len(viols) == 0

    def test_infeasible_bounds(self):
        m, x, y = _make_nlp()
        flat = np.array([20.0, 4.0])  # x > ub
        is_feas, viols = check_feasibility(m, flat)
        assert not is_feas
        assert any("above upper bound" in v for v in viols)

    def test_feasible_milp_with_constraint(self):
        m, x, y = _make_milp()
        flat = np.array([1.0, 1.0])  # x+y=2 <= 3, feasible
        is_feas, viols = check_feasibility(m, flat)
        assert is_feas

    def test_infeasible_constraint(self):
        m, x, y = _make_minlp()
        # x=5, y=0: x <= 4*y+1 becomes 5 <= 1, violated
        flat = np.array([5.0, 0.0])
        is_feas, viols = check_feasibility(m, flat)
        assert not is_feas


# ──────────────────────────────────────────────────────────
# TestSolveWarmStart — end-to-end solve with warm start
# ──────────────────────────────────────────────────────────


class TestSolveWarmStart:
    """End-to-end tests that warm-started solve produces valid results."""

    def test_nlp_warm_start(self):
        """Warm-starting a continuous NLP near the optimum."""
        m, x, y = _make_nlp()
        result = m.solve(initial_solution={x: 2.9, y: 3.9})
        assert result.status == "optimal"
        assert result.objective is not None
        np.testing.assert_allclose(result.objective, 0.0, atol=1e-3)

    def test_nlp_cold_start_also_works(self):
        """Cold-start (no initial solution) still works."""
        m, x, y = _make_nlp()
        result = m.solve()
        assert result.status == "optimal"
        np.testing.assert_allclose(result.objective, 0.0, atol=1e-3)

    def test_minlp_warm_start(self):
        """Warm-start a MINLP with a feasible integer solution."""
        m, x, y = _make_minlp()
        # y=1 is optimal (allows x up to 5), x=2.5 is optimal
        result = m.solve(initial_solution={x: 2.5, y: 1.0})
        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        # Optimal: (2.5-2.5)^2 + (1-1)^2 = 0
        np.testing.assert_allclose(result.objective, 0.0, atol=1e-2)

    def test_milp_warm_start(self):
        """Warm-start a MILP with a known feasible solution."""
        m, x, y = _make_milp()
        # Optimal: x=1, y=1, obj = -3
        result = m.solve(initial_solution={x: 1.0, y: 1.0})
        assert result.status in ("optimal", "feasible")
        assert result.objective is not None
        assert result.objective <= -2.0  # at least as good as y=1 alone

    def test_warm_start_with_bad_key_raises(self):
        """Passing a non-Variable key raises TypeError."""
        m, x, y = _make_nlp()
        with pytest.raises(TypeError):
            m.solve(initial_solution={"x": 3.0})

    def test_warm_start_wrong_model_raises(self):
        """Passing a Variable from a different model raises ValueError."""
        m1, x1, y1 = _make_nlp()
        m2, x2, y2 = _make_nlp()
        with pytest.raises(ValueError, match="not part of this model"):
            m1.solve(initial_solution={x2: 3.0})

    @pytest.mark.slow
    def test_warm_start_bounds_violation_warns(self):
        """Out-of-bounds values produce a warning but solve proceeds."""
        m, x, y = _make_nlp()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = m.solve(initial_solution={x: 100.0, y: 4.0})
            bound_warns = [ww for ww in w if "outside bounds" in str(ww.message)]
            assert len(bound_warns) >= 1
        assert result.status == "optimal"
