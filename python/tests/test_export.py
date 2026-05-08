"""Tests for MPS and LP file export."""

from __future__ import annotations

import tempfile
from pathlib import Path

import discopt.modeling as dm
import pytest
from discopt.export import to_lp, to_mps
from discopt.export._extract import (
    extract_linear_terms,
    extract_quadratic_terms,
    flatten_variables,
)

pytestmark = pytest.mark.unit

# ─────────────────────────────────────────────────────────────
# Helper: build small test models
# ─────────────────────────────────────────────────────────────


def _simple_lp():
    """min 3x + 2y  s.t. x + y >= 5, x <= 8, 0 <= x,y <= 10."""
    m = dm.Model("simple_lp")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(3 * x + 2 * y)
    m.subject_to(x + y >= 5, name="demand")
    m.subject_to(x <= 8, name="cap")
    return m, x, y


def _simple_milp():
    """min x + 5y  s.t. x + 10y >= 6, 0 <= x <= 10, y binary."""
    m = dm.Model("simple_milp")
    x = m.continuous("x", lb=0, ub=10)
    y = m.binary("y")
    m.minimize(x + 5 * y)
    m.subject_to(x + 10 * y >= 6, name="cover")
    return m, x, y


def _simple_qp():
    """min x^2 + y^2 + x*y  s.t. x + y >= 1, 0 <= x,y <= 5."""
    m = dm.Model("simple_qp")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.minimize(x * x + y * y + x * y)
    m.subject_to(x + y >= 1, name="lower")
    return m, x, y


def _maximize_model():
    """max 2x + 3y  s.t. x + y <= 10, 0 <= x,y."""
    m = dm.Model("maximize_test")
    x = m.continuous("x", lb=0, ub=100)
    y = m.continuous("y", lb=0, ub=100)
    m.maximize(2 * x + 3 * y)
    m.subject_to(x + y <= 10, name="budget")
    return m, x, y


def _free_and_fixed_vars():
    """Model with free, fixed, and one-sided bounded variables."""
    m = dm.Model("bounds_test")
    # Free variable (default bounds)
    x_free = m.continuous("x_free")
    # Fixed variable (lb == ub)
    x_fixed = m.continuous("x_fixed", lb=3.0, ub=3.0)
    # Lower-bounded only
    x_lo = m.continuous("x_lo", lb=2.0)
    # Upper-bounded only
    x_up = m.continuous("x_up", ub=7.0)
    m.minimize(x_free + x_fixed + x_lo + x_up)
    m.subject_to(x_free + x_lo >= 0, name="c0")
    return m


def _array_variable_model():
    """Model with array variables."""
    m = dm.Model("array_test")
    x = m.continuous("x", shape=(3,), lb=0, ub=10)
    m.minimize(2 * x[0] + 3 * x[1] + x[2])
    m.subject_to(x[0] + x[1] + x[2] <= 15, name="total")
    m.subject_to(x[0] >= 1, name="min_x0")
    return m, x


def _integer_var_model():
    """Model with general integer variables."""
    m = dm.Model("integer_test")
    x = m.continuous("x", lb=0, ub=10)
    n = m.integer("n", lb=0, ub=5)
    m.minimize(x + 3 * n)
    m.subject_to(x + n >= 2, name="min_total")
    return m, x, n


def _equality_constraint_model():
    """Model with equality constraint."""
    m = dm.Model("equality_test")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x + y)
    m.subject_to(x + y == 5, name="eq_con")
    return m


def _nonlinear_model():
    """Model with nonlinear terms (should raise ValueError on export)."""
    m = dm.Model("nonlinear")
    x = m.continuous("x", lb=0.1, ub=10)
    m.minimize(dm.log(x))
    m.subject_to(x >= 1)
    return m


# ─────────────────────────────────────────────────────────────
# extract_linear_terms tests
# ─────────────────────────────────────────────────────────────


class TestExtractLinearTerms:
    def test_simple_linear(self):
        m, x, y = _simple_lp()
        flat = flatten_variables(m)
        coeffs, const = extract_linear_terms(m._objective.expression, flat, model_vars=m._variables)
        assert const == 0.0
        # x has coefficient 3, y has coefficient 2
        assert coeffs.get(0, 0.0) == pytest.approx(3.0)
        assert coeffs.get(1, 0.0) == pytest.approx(2.0)

    def test_constant_expression(self):
        m = dm.Model("const_test")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x + 5)
        m.subject_to(x >= 0)
        flat = flatten_variables(m)
        coeffs, const = extract_linear_terms(m._objective.expression, flat, model_vars=m._variables)
        assert const == pytest.approx(5.0)
        assert coeffs.get(0, 0.0) == pytest.approx(1.0)

    def test_negation(self):
        m = dm.Model("neg_test")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(-x)
        m.subject_to(x >= 0)
        flat = flatten_variables(m)
        coeffs, const = extract_linear_terms(m._objective.expression, flat, model_vars=m._variables)
        assert coeffs.get(0, 0.0) == pytest.approx(-1.0)

    def test_nonlinear_raises(self):
        m = _nonlinear_model()
        flat = flatten_variables(m)
        with pytest.raises(ValueError, match="cannot be exported"):
            extract_linear_terms(m._objective.expression, flat, model_vars=m._variables)


# ─────────────────────────────────────────────────────────────
# extract_quadratic_terms tests
# ─────────────────────────────────────────────────────────────


class TestExtractQuadraticTerms:
    def test_simple_quadratic(self):
        m, x, y = _simple_qp()
        flat = flatten_variables(m)
        quad, linear, const = extract_quadratic_terms(
            m._objective.expression, flat, model_vars=m._variables
        )
        assert const == 0.0
        # x^2 => (0,0): 1.0, y^2 => (1,1): 1.0, x*y => (0,1): 1.0
        assert quad.get((0, 0), 0.0) == pytest.approx(1.0)
        assert quad.get((1, 1), 0.0) == pytest.approx(1.0)
        assert quad.get((0, 1), 0.0) == pytest.approx(1.0)
        assert len(linear) == 0

    def test_linear_is_also_quadratic(self):
        """A purely linear expression should extract with empty quad terms."""
        m, x, y = _simple_lp()
        flat = flatten_variables(m)
        quad, linear, const = extract_quadratic_terms(
            m._objective.expression, flat, model_vars=m._variables
        )
        assert len(quad) == 0
        assert linear.get(0, 0.0) == pytest.approx(3.0)
        assert linear.get(1, 0.0) == pytest.approx(2.0)


# ─────────────────────────────────────────────────────────────
# MPS export tests
# ─────────────────────────────────────────────────────────────


class TestMPSExport:
    def test_lp_to_mps(self):
        m, _, _ = _simple_lp()
        mps = m.to_mps()
        assert isinstance(mps, str)
        assert "NAME" in mps
        assert "ROWS" in mps
        assert "COLUMNS" in mps
        assert "RHS" in mps
        assert "BOUNDS" in mps
        assert "ENDATA" in mps
        # Check objective row
        assert " N  OBJ" in mps
        # Check constraint rows
        assert "demand" in mps
        assert "cap" in mps

    def test_milp_integer_markers(self):
        m, _, _ = _simple_milp()
        mps = m.to_mps()
        assert "INTORG" in mps
        assert "INTEND" in mps

    def test_qp_quadobj_section(self):
        m, _, _ = _simple_qp()
        mps = m.to_mps()
        assert "QUADOBJ" in mps

    def test_maximize_objsense(self):
        m, _, _ = _maximize_model()
        mps = m.to_mps()
        assert "OBJSENSE MAX" in mps

    def test_variable_bounds(self):
        m = _free_and_fixed_vars()
        mps = m.to_mps()
        assert " FR BND" in mps  # free variable
        assert " FX BND" in mps  # fixed variable
        # Lower-bounded only
        assert " LO BND  x_lo" in mps

    def test_constraint_senses(self):
        m = _equality_constraint_model()
        mps = m.to_mps()
        assert " E  eq_con" in mps

    def test_write_to_file(self):
        m, _, _ = _simple_lp()
        with tempfile.NamedTemporaryFile(suffix=".mps", delete=False) as f:
            path = f.name
        try:
            result = m.to_mps(path)
            assert result is None
            content = Path(path).read_text()
            assert "NAME" in content
            assert "ENDATA" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_nonlinear_raises(self):
        m = _nonlinear_model()
        with pytest.raises(ValueError, match="cannot be exported"):
            m.to_mps()

    def test_array_variables(self):
        m, x = _array_variable_model()
        mps = m.to_mps()
        assert "x_0" in mps
        assert "x_1" in mps
        assert "x_2" in mps

    def test_integer_variable_markers(self):
        m, _, _ = _integer_var_model()
        mps = m.to_mps()
        assert "INTORG" in mps
        assert "INTEND" in mps


# ─────────────────────────────────────────────────────────────
# LP export tests
# ─────────────────────────────────────────────────────────────


class TestLPExport:
    def test_lp_format_basic(self):
        m, _, _ = _simple_lp()
        lp = m.to_lp()
        assert isinstance(lp, str)
        assert "Minimize" in lp
        assert "Subject To" in lp
        assert "Bounds" in lp
        assert "End" in lp

    def test_maximize(self):
        m, _, _ = _maximize_model()
        lp = m.to_lp()
        assert "Maximize" in lp
        assert "Minimize" not in lp

    def test_milp_binaries_section(self):
        m, _, _ = _simple_milp()
        lp = m.to_lp()
        assert "Binaries" in lp
        assert "y" in lp

    def test_integer_generals_section(self):
        m, _, _ = _integer_var_model()
        lp = m.to_lp()
        assert "Generals" in lp
        assert "n" in lp

    def test_qp_quadratic_section(self):
        m, _, _ = _simple_qp()
        lp = m.to_lp()
        assert "^ 2" in lp or "*" in lp

    def test_equality_constraint(self):
        m = _equality_constraint_model()
        lp = m.to_lp()
        # LP format uses = for equality
        assert "= 5" in lp or "= 5.0" in lp

    def test_free_variable_bounds(self):
        m = _free_and_fixed_vars()
        lp = m.to_lp()
        assert "Free" in lp

    def test_write_to_file(self):
        m, _, _ = _simple_lp()
        with tempfile.NamedTemporaryFile(suffix=".lp", delete=False) as f:
            path = f.name
        try:
            result = m.to_lp(path)
            assert result is None
            content = Path(path).read_text()
            assert "Minimize" in content
            assert "End" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_nonlinear_raises(self):
        m = _nonlinear_model()
        with pytest.raises(ValueError, match="cannot be exported"):
            m.to_lp()

    def test_array_variables(self):
        m, _ = _array_variable_model()
        lp = m.to_lp()
        assert "x_0" in lp
        assert "x_1" in lp
        assert "x_2" in lp


# ─────────────────────────────────────────────────────────────
# Round-trip test with HiGHS (if available)
# ─────────────────────────────────────────────────────────────


class TestRoundTrip:
    @pytest.fixture(autouse=True)
    def _check_highspy(self):
        pytest.importorskip("highspy")

    def test_mps_roundtrip_lp(self):
        """Export LP model to MPS, read with HiGHS, verify solution."""
        import highspy

        m, x, y = _simple_lp()
        with tempfile.NamedTemporaryFile(suffix=".mps", delete=False) as f:
            path = f.name
        try:
            m.to_mps(path)
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            h.readModel(path)
            h.run()
            assert h.getInfoValue("primal_solution_status")[1] == 2  # feasible
            obj = h.getInfoValue("objective_function_value")[1]
            # Optimal: min 3x + 2y s.t. x+y>=5, x<=8, 0<=x,y<=10
            # Optimal at y=5, x=0: obj=10
            assert obj == pytest.approx(10.0, abs=1e-6)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_mps_roundtrip_milp(self):
        """Export MILP model to MPS, read with HiGHS, verify solution."""
        import highspy

        m, x, y = _simple_milp()
        with tempfile.NamedTemporaryFile(suffix=".mps", delete=False) as f:
            path = f.name
        try:
            m.to_mps(path)
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            h.readModel(path)
            h.run()
            obj = h.getInfoValue("objective_function_value")[1]
            # min x + 5y s.t. x + 10y >= 6, 0<=x<=10, y in {0,1}
            # y=1: x + 10 >= 6, so x=0, obj=5
            # y=0: x >= 6, obj=6
            assert obj == pytest.approx(5.0, abs=1e-6)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_lp_roundtrip(self):
        """Export LP model to LP format, read with HiGHS, verify solution."""
        import highspy

        m, x, y = _simple_lp()
        with tempfile.NamedTemporaryFile(suffix=".lp", delete=False) as f:
            path = f.name
        try:
            m.to_lp(path)
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            h.readModel(path)
            h.run()
            obj = h.getInfoValue("objective_function_value")[1]
            assert obj == pytest.approx(10.0, abs=1e-6)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_maximize_roundtrip(self):
        """Verify maximization works correctly through MPS round-trip."""
        import highspy

        m, x, y = _maximize_model()
        with tempfile.NamedTemporaryFile(suffix=".mps", delete=False) as f:
            path = f.name
        try:
            m.to_mps(path)
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            h.readModel(path)
            h.run()
            obj = h.getInfoValue("objective_function_value")[1]
            # max 2x + 3y s.t. x+y<=10, 0<=x,y<=100
            # Optimal: x=0, y=10, obj=30
            assert obj == pytest.approx(30.0, abs=1e-6)
        finally:
            Path(path).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────
# Functional import tests
# ─────────────────────────────────────────────────────────────


class TestFunctionalImport:
    def test_to_mps_from_module(self):
        m, _, _ = _simple_lp()
        mps = to_mps(m)
        assert "ENDATA" in mps

    def test_to_lp_from_module(self):
        m, _, _ = _simple_lp()
        lp = to_lp(m)
        assert "End" in lp
