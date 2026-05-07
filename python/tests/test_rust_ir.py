"""Tests for Rust Expression IR (T2).

Validates:
- Round-trip conversion: Python Model -> Rust ModelRepr
- Structure detection: is_linear, is_quadratic, is_bilinear
- Evaluation: Rust evaluator matches Python within 1e-14
- Variable info: names, bounds, types, shapes
"""

import discopt.modeling as dm
import numpy as np
import pytest

# Import the Rust bindings
from discopt._rust import PyModelRepr, model_to_repr
from discopt.modeling.examples import (
    example_parametric,
    example_pooling_haverly,
    example_reactor_design,
    example_simple_minlp,
)

# ─────────────────────────────────────────────────────────────
# Fixtures: build example models
# ─────────────────────────────────────────────────────────────


@pytest.fixture
def simple_model():
    """Example 1: x1^2 + x2^2 + x3, 2 constraints."""
    m = dm.Model("textbook")
    x1 = m.continuous("x1", lb=0, ub=5)
    x2 = m.continuous("x2", lb=0, ub=5)
    x3 = m.binary("x3")
    m.minimize(x1**2 + x2**2 + x3)
    m.subject_to(x1 + x2 >= 1)
    m.subject_to(x1**2 + x2 <= 3)
    return m


@pytest.fixture
def linear_model():
    """A purely linear model for structure detection."""
    m = dm.Model("linear")
    x = m.continuous("x", shape=(3,), lb=0, ub=10)
    c = np.array([2.0, 3.0, 5.0])
    m.minimize(dm.sum(lambda i: c[i] * x[i], over=range(3)))
    m.subject_to(x[0] + x[1] + x[2] <= 15)
    m.subject_to(x[0] >= 1)
    return m


@pytest.fixture
def nonlinear_model():
    """Model with exp/log nonlinearity."""
    m = dm.Model("nonlinear")
    x = m.continuous("x", lb=0.1, ub=10)
    y = m.continuous("y", lb=0.1, ub=10)
    m.minimize(dm.exp(x) + dm.log(y))
    m.subject_to(x + y <= 5)
    return m


@pytest.fixture
def bilinear_model():
    """Model with bilinear x*y terms."""
    m = dm.Model("bilinear")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    z = m.continuous("z", lb=0, ub=10)
    m.minimize(x * y + z)
    m.subject_to(x + y + z <= 20)
    return m


@pytest.fixture
def parametric_model():
    """Model with parameters (Example 7 simplified)."""
    m = dm.Model("parametric")
    price = m.parameter("price", value=50.0)
    x = m.continuous("x", lb=0, ub=100)
    m.minimize(price * x)
    m.subject_to(x >= 10)
    return m


# ─────────────────────────────────────────────────────────────
# Test: Basic conversion
# ─────────────────────────────────────────────────────────────


class TestConversion:
    def test_simple_model_converts(self, simple_model):
        repr = model_to_repr(simple_model)
        assert isinstance(repr, PyModelRepr)
        assert repr.n_vars == 3  # x1 (scalar) + x2 (scalar) + x3 (scalar)
        assert repr.n_var_blocks == 3
        assert repr.objective_sense == "minimize"

    def test_linear_model_converts(self, linear_model):
        repr = model_to_repr(linear_model)
        assert repr.n_vars == 3
        assert repr.n_var_blocks == 1  # single array variable

    def test_nonlinear_model_converts(self, nonlinear_model):
        repr = model_to_repr(nonlinear_model)
        assert repr.n_vars == 2

    def test_bilinear_model_converts(self, bilinear_model):
        repr = model_to_repr(bilinear_model)
        assert repr.n_vars == 3

    def test_parametric_model_converts(self, parametric_model):
        repr = model_to_repr(parametric_model)
        assert repr.n_vars == 1

    def test_node_count_positive(self, simple_model):
        repr = model_to_repr(simple_model)
        assert repr.n_nodes > 0

    def test_pooling_model_converts(self):
        """Example 2: pooling problem with bilinear constraints."""
        m = example_pooling_haverly()
        repr = model_to_repr(m)
        assert repr.n_vars > 0
        assert repr.n_constraints > 0
        assert repr.objective_sense == "maximize"

    def test_reactor_model_converts(self):
        """Example 5: reactor design with exp() and integer vars."""
        m = example_reactor_design()
        repr = model_to_repr(m)
        assert repr.n_vars > 0
        assert repr.n_constraints > 0


# ─────────────────────────────────────────────────────────────
# Test: Variable info
# ─────────────────────────────────────────────────────────────


class TestVariableInfo:
    def test_var_names(self, simple_model):
        repr = model_to_repr(simple_model)
        names = repr.var_names()
        assert names == ["x1", "x2", "x3"]

    def test_var_types(self, simple_model):
        repr = model_to_repr(simple_model)
        types = repr.var_types()
        assert types == ["continuous", "continuous", "binary"]

    def test_var_shapes_scalar(self, simple_model):
        repr = model_to_repr(simple_model)
        shapes = repr.var_shapes()
        assert shapes == [[], [], []]

    def test_var_shapes_array(self, linear_model):
        repr = model_to_repr(linear_model)
        shapes = repr.var_shapes()
        assert shapes == [[3]]

    def test_var_bounds_continuous(self, simple_model):
        repr = model_to_repr(simple_model)
        lb0 = repr.var_lb(0)
        ub0 = repr.var_ub(0)
        assert lb0 == [0.0]
        assert ub0 == [5.0]

    def test_var_bounds_binary(self, simple_model):
        repr = model_to_repr(simple_model)
        lb2 = repr.var_lb(2)
        ub2 = repr.var_ub(2)
        assert lb2 == [0.0]
        assert ub2 == [1.0]

    def test_var_bounds_array(self, linear_model):
        repr = model_to_repr(linear_model)
        lb = repr.var_lb(0)
        ub = repr.var_ub(0)
        assert lb == [0.0, 0.0, 0.0]
        assert ub == [10.0, 10.0, 10.0]


# ─────────────────────────────────────────────────────────────
# Test: Structure detection
# ─────────────────────────────────────────────────────────────


class TestStructureDetection:
    def test_linear_objective_detected(self, linear_model):
        repr = model_to_repr(linear_model)
        assert repr.is_objective_linear()
        assert repr.is_objective_quadratic()  # linear is also quadratic

    def test_quadratic_objective_detected(self, simple_model):
        repr = model_to_repr(simple_model)
        assert not repr.is_objective_linear()
        assert repr.is_objective_quadratic()

    def test_nonlinear_objective_detected(self, nonlinear_model):
        repr = model_to_repr(nonlinear_model)
        assert not repr.is_objective_linear()
        assert not repr.is_objective_quadratic()

    def test_bilinear_objective_detected(self, bilinear_model):
        repr = model_to_repr(bilinear_model)
        # x*y + z has bilinear term but is not purely bilinear at the top
        assert not repr.is_objective_linear()
        assert repr.is_objective_quadratic()  # bilinear is degree 2

    def test_linear_constraint_detected(self, linear_model):
        repr = model_to_repr(linear_model)
        for i in range(repr.n_constraints):
            assert repr.is_constraint_linear(i)

    def test_quadratic_constraint(self, simple_model):
        repr = model_to_repr(simple_model)
        # Constraint 0: (x1 + x2) - 1 >= 0 (but stored as -(x1+x2-1) <= 0)
        # linear
        assert repr.is_constraint_linear(0)
        # Constraint 1: x1^2 + x2 - 3 <= 0 (quadratic)
        assert not repr.is_constraint_linear(1)
        assert repr.is_constraint_quadratic(1)


# ─────────────────────────────────────────────────────────────
# Test: Evaluation
# ─────────────────────────────────────────────────────────────


class TestEvaluation:
    def test_evaluate_linear_objective(self, linear_model):
        """Sum of c[i]*x[i] with c=[2,3,5], x=[1,2,3] = 2+6+15 = 23."""
        repr = model_to_repr(linear_model)
        x = np.array([1.0, 2.0, 3.0])
        val = repr.evaluate_objective(x)
        assert abs(val - 23.0) < 1e-14

    def test_evaluate_quadratic_objective(self, simple_model):
        """x1^2 + x2^2 + x3 at (3, 4, 1) = 9 + 16 + 1 = 26."""
        repr = model_to_repr(simple_model)
        x = np.array([3.0, 4.0, 1.0])
        val = repr.evaluate_objective(x)
        assert abs(val - 26.0) < 1e-14

    def test_evaluate_nonlinear_objective(self, nonlinear_model):
        """exp(x) + log(y) at (1.0, 2.0)."""
        repr = model_to_repr(nonlinear_model)
        x = np.array([1.0, 2.0])
        val = repr.evaluate_objective(x)
        expected = np.exp(1.0) + np.log(2.0)
        assert abs(val - expected) < 1e-14

    def test_evaluate_bilinear_objective(self, bilinear_model):
        """x*y + z at (2, 3, 5) = 6 + 5 = 11."""
        repr = model_to_repr(bilinear_model)
        x = np.array([2.0, 3.0, 5.0])
        val = repr.evaluate_objective(x)
        assert abs(val - 11.0) < 1e-14

    def test_evaluate_parametric_objective(self, parametric_model):
        """price * x = 50 * 10 = 500."""
        repr = model_to_repr(parametric_model)
        x = np.array([10.0])
        val = repr.evaluate_objective(x)
        assert abs(val - 500.0) < 1e-14

    def test_evaluate_constraint(self, simple_model):
        """Constraint 0 body: 1*(-x1 - x2 + 1) at (3, 4, 1).
        Actually: constraint is x1 + x2 >= 1, stored as (Const(1) - x1 - x2) <= 0.
        So body = -(x1 + x2 - 1) = 1 - x1 - x2 = 1 - 3 - 4 = -6."""
        repr = model_to_repr(simple_model)
        x = np.array([3.0, 4.0, 1.0])
        # The Python Expression for >= is: Constraint(_wrap(other) - self, "<=", 0)
        # So x1 + x2 >= 1 becomes Constraint(Constant(1) - (x1 + x2), "<=", 0)
        val = repr.evaluate_constraint(0, x)
        # body = Constant(1) - (x1 + x2) = 1 - 3 - 4 = -6
        assert abs(val - (-6.0)) < 1e-14


# ─────────────────────────────────────────────────────────────
# Test: Example models from examples.py
# ─────────────────────────────────────────────────────────────


class TestExampleModels:
    def test_simple_minlp(self):
        m = example_simple_minlp()
        repr = model_to_repr(m)
        assert repr.n_vars == 3
        assert repr.objective_sense == "minimize"
        assert repr.is_objective_quadratic()
        assert not repr.is_objective_linear()

    def test_pooling_haverly(self):
        m = example_pooling_haverly()
        repr = model_to_repr(m)
        # 4 variable blocks: y(2), x(2), z(1), p(1) = 6 total
        assert repr.n_vars == 6
        assert repr.objective_sense == "maximize"
        # Objective has only linear terms (revenues - costs)
        assert repr.is_objective_linear()

    def test_reactor_design(self):
        m = example_reactor_design()
        repr = model_to_repr(m)
        # T(3), V(3), F(1), n_stages(1) = 8
        assert repr.n_vars == 8
        assert repr.objective_sense == "minimize"
        # Constraints include exp() so not all are linear

    def test_parametric_full(self):
        m = example_parametric()
        repr = model_to_repr(m)
        assert repr.n_vars == 6
        assert repr.objective_sense == "minimize"

    def test_parametric_simplified(self, parametric_model):
        repr = model_to_repr(parametric_model)
        assert repr.n_vars == 1
        assert repr.objective_sense == "minimize"
        val = repr.evaluate_objective(np.array([10.0]))
        assert abs(val - 500.0) < 1e-14


# ─────────────────────────────────────────────────────────────
# Test: Constraint names
# ─────────────────────────────────────────────────────────────


class TestConstraintNames:
    def test_unnamed_constraints(self, simple_model):
        repr = model_to_repr(simple_model)
        # simple_model constraints are unnamed
        assert repr.constraint_name(0) is None
        assert repr.constraint_name(1) is None

    def test_named_constraints(self):
        m = dm.Model("named")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x + y)
        m.subject_to(x + y <= 15, name="capacity")
        m.subject_to(x >= 1, name="minimum")
        repr = model_to_repr(m)
        assert repr.constraint_name(0) == "capacity"
        assert repr.constraint_name(1) == "minimum"


# ─────────────────────────────────────────────────────────────
# Test: Edge cases
# ─────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_scalar_constant_model(self):
        """Minimize a constant (trivial model)."""
        m = dm.Model("trivial")
        x = m.continuous("x", lb=0, ub=1)
        m.minimize(x + 0.0)
        repr = model_to_repr(m)
        assert repr.n_vars == 1
        val = repr.evaluate_objective(np.array([0.5]))
        assert abs(val - 0.5) < 1e-14

    def test_negation(self):
        """Unary negation."""
        m = dm.Model("neg")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(-x)
        repr = model_to_repr(m)
        val = repr.evaluate_objective(np.array([5.0]))
        assert abs(val - (-5.0)) < 1e-14

    def test_division(self):
        """Division by constant."""
        m = dm.Model("div")
        x = m.continuous("x", lb=1, ub=10)
        m.minimize(x / 2)
        repr = model_to_repr(m)
        val = repr.evaluate_objective(np.array([8.0]))
        assert abs(val - 4.0) < 1e-14

    def test_power(self):
        """Power expression x^3."""
        m = dm.Model("pow")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x**3)
        repr = model_to_repr(m)
        val = repr.evaluate_objective(np.array([2.0]))
        assert abs(val - 8.0) < 1e-14
        # x^3 is degree 3, not quadratic
        assert not repr.is_objective_quadratic()
        assert not repr.is_objective_linear()

    def test_sqrt_function(self):
        """Sqrt function."""
        m = dm.Model("sqrt")
        x = m.continuous("x", lb=0, ub=100)
        m.minimize(dm.sqrt(x))
        repr = model_to_repr(m)
        val = repr.evaluate_objective(np.array([9.0]))
        assert abs(val - 3.0) < 1e-14

    def test_sin_cos(self):
        """Trig functions."""
        m = dm.Model("trig")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(dm.sin(x) + dm.cos(y))
        repr = model_to_repr(m)
        val = repr.evaluate_objective(np.array([np.pi / 2, 0.0]))
        expected = np.sin(np.pi / 2) + np.cos(0.0)
        assert abs(val - expected) < 1e-14

    def test_multiple_math_functions(self):
        """Combined transcendental functions."""
        m = dm.Model("multi_func")
        x = m.continuous("x", lb=0.1, ub=10)
        m.minimize(dm.exp(x) + dm.log(x) + dm.sqrt(x))
        repr = model_to_repr(m)
        val = repr.evaluate_objective(np.array([2.0]))
        expected = np.exp(2.0) + np.log(2.0) + np.sqrt(2.0)
        assert abs(val - expected) < 1e-13


# ─────────────────────────────────────────────────────────────
# Test: Slice indexing (x[i, :], x[:, j], x[:])
# ─────────────────────────────────────────────────────────────


class TestSliceIndexing:
    """Slice indexing converts to and evaluates correctly through the Rust IR.

    The flat layout is row-major (C-order); for a (2, 2) variable, ``x_flat``
    is ``[x[0,0], x[0,1], x[1,0], x[1,1]]``.
    """

    def test_row_slice_evaluates(self):
        m = dm.Model("row_slice")
        x = m.continuous("x", shape=(2, 2), lb=0, ub=10)
        m.minimize(dm.sum(x[0, :]) + 2 * dm.sum(x[1, :]))
        repr = model_to_repr(m)
        # (x00 + x01) + 2 * (x10 + x11) = (1+2) + 2*(3+4) = 17
        val = repr.evaluate_objective(np.array([1.0, 2.0, 3.0, 4.0]))
        assert abs(val - 17.0) < 1e-14

    def test_column_slice_evaluates(self):
        m = dm.Model("col_slice")
        y = m.continuous("y", shape=(2, 2), lb=0, ub=10)
        m.minimize(dm.sum(y[:, 0]) + 10 * dm.sum(y[:, 1]))
        repr = model_to_repr(m)
        # col0 = y00 + y10 = 1 + 3 = 4; col1 = y01 + y11 = 2 + 4 = 6;
        # total = 4 + 60 = 64
        val = repr.evaluate_objective(np.array([1.0, 2.0, 3.0, 4.0]))
        assert abs(val - 64.0) < 1e-14

    def test_full_slice_on_1d_variable(self):
        m = dm.Model("full_slice_1d")
        z = m.continuous("z", shape=(3,), lb=0, ub=10)
        m.minimize(dm.sum(z[:]))
        repr = model_to_repr(m)
        val = repr.evaluate_objective(np.array([5.0, 6.0, 7.0]))
        assert abs(val - 18.0) < 1e-14

    def test_mixed_slice_arithmetic(self):
        m = dm.Model("slice_arith")
        w = m.continuous("w", shape=(2, 2), lb=0, ub=10)
        m.minimize(dm.sum(w[0, :]) - dm.sum(w[1, :]))
        repr = model_to_repr(m)
        # (1+2) - (3+4) = -4
        val = repr.evaluate_objective(np.array([1.0, 2.0, 3.0, 4.0]))
        assert abs(val - (-4.0)) < 1e-14

    def test_partial_slice_start_stop(self):
        """``a[1:4]`` selects elements 1..4 with Python semantics."""
        m = dm.Model("start_stop")
        a = m.continuous("a", shape=(5,), lb=0, ub=100)
        m.minimize(dm.sum(a[1:4]))
        repr = model_to_repr(m)
        val = repr.evaluate_objective(np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        assert abs(val - (20 + 30 + 40)) < 1e-14

    def test_slice_with_step(self):
        """``a[::2]`` strides through every other element."""
        m = dm.Model("step")
        a = m.continuous("a", shape=(6,), lb=0, ub=100)
        m.minimize(dm.sum(a[::2]))
        repr = model_to_repr(m)
        val = repr.evaluate_objective(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        assert abs(val - (1 + 3 + 5)) < 1e-14

    def test_negative_indices(self):
        """``a[-2:]`` indexes from the end."""
        m = dm.Model("neg_idx")
        a = m.continuous("a", shape=(5,), lb=0, ub=100)
        m.minimize(dm.sum(a[-2:]))
        repr = model_to_repr(m)
        val = repr.evaluate_objective(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert abs(val - (4 + 5)) < 1e-14

    def test_reverse_step(self):
        """``a[::-1]`` walks the array backwards (sum is order-independent)."""
        m = dm.Model("reverse")
        a = m.continuous("a", shape=(4,), lb=0, ub=100)
        m.minimize(dm.sum(a[::-1]))
        repr = model_to_repr(m)
        val = repr.evaluate_objective(np.array([1.0, 2.0, 3.0, 4.0]))
        assert abs(val - 10.0) < 1e-14

    def test_partial_slice_2d(self):
        """``X[i, 1:3]`` mixes scalar and partial-slice axes."""
        m = dm.Model("partial_2d")
        x = m.continuous("X", shape=(3, 3), lb=0, ub=100)
        m.minimize(dm.sum(x[1, 1:3]))
        repr = model_to_repr(m)
        # X flat = [1..9]; X[1, 1:3] = {X[1,1], X[1,2]} = {5, 6}
        val = repr.evaluate_objective(np.arange(1.0, 10.0))
        assert abs(val - (5 + 6)) < 1e-14

    def test_empty_slice(self):
        """An empty slice contributes zero to a sum."""
        m = dm.Model("empty")
        a = m.continuous("a", shape=(4,), lb=0, ub=10)
        m.minimize(dm.sum(a[5:5]) + 1.0)
        repr = model_to_repr(m)
        val = repr.evaluate_objective(np.array([1.0, 2.0, 3.0, 4.0]))
        assert abs(val - 1.0) < 1e-14

    def test_step_zero_rejected(self):
        """A zero step is invalid (matches Python's `slice` behavior)."""
        m = dm.Model("step_zero")
        a = m.continuous("a", shape=(4,), lb=0, ub=10)
        m.minimize(dm.sum(a[::0]))
        with pytest.raises(ValueError, match="step cannot be zero"):
            model_to_repr(m)
