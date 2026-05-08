"""Tests for .nl file parser (T3).

Validates:
- Parsing text-mode .nl files to Rust ModelRepr via PyO3
- Correct variable counts, constraint counts, bounds
- Objective evaluation at known points
- Constraint body evaluation
- Variable types (continuous, binary, integer)
- Structure detection (linear, quadratic, nonlinear)

Requires nl_bindings to be wired into crates/discopt-python/src/lib.rs.
Tests skip gracefully if the binding is not yet available.
"""

import os
import tempfile

import numpy as np
import pytest

pytestmark = pytest.mark.unit

# Try importing the .nl parser binding; skip all tests if unavailable.
try:
    from discopt._rust import parse_nl_file, parse_nl_string

    HAS_NL_BINDINGS = True
except ImportError:
    HAS_NL_BINDINGS = False

pytestmark = pytest.mark.skipif(
    not HAS_NL_BINDINGS,
    reason="nl_bindings not yet wired into discopt._rust",
)


# ─────────────────────────────────────────────────────────────
# .nl content generators (mirror the Rust test helpers)
# ─────────────────────────────────────────────────────────────


def _linear_nl() -> str:
    """min 2*x + 3*y  s.t. x + y <= 10, 0 <= x,y <= 100."""
    lines = [
        "g3 1 1 0",
        " 2 1 1 0 0",
        " 0 0",
        " 0 0 0",
        " 0 0 0",
        " 0 0 0 1",
        " 0 0",
        " 2 2",
        " 0 0",
        " 0 0 0 0 0",
        "O0 0",
        "n0",
        "C0",
        "n0",
        "x2",
        "0 0",
        "1 0",
        "r",
        "1 10",
        "b",
        "0 0 100",
        "0 0 100",
        "k1",
        "1",
        "J0 2",
        "0 1",
        "1 1",
        "G0 2",
        "0 2",
        "1 3",
    ]
    return "\n".join(lines) + "\n"


def _quadratic_nl() -> str:
    """min x^2 + y^2  s.t. x + y >= 1."""
    lines = [
        "g3 1 1 0",
        " 2 1 1 0 0",
        " 1 1",
        " 0 0",
        " 2 2 2",
        " 0 0 0 1",
        " 0 0",
        " 0 0",
        " 0 0",
        " 0 0 0 0 0",
        "O0 0",
        "o0",
        "o5",
        "v0",
        "n2",
        "o5",
        "v1",
        "n2",
        "C0",
        "n0",
        "x2",
        "0 1",
        "1 1",
        "r",
        "2 1",
        "b",
        "0 -10 10",
        "0 -10 10",
        "k1",
        "1",
        "J0 2",
        "0 1",
        "1 1",
        "G0 2",
        "0 0",
        "1 0",
    ]
    return "\n".join(lines) + "\n"


def _nonlinear_nl() -> str:
    """min exp(x) + log(y)  s.t. x + y <= 5."""
    lines = [
        "g3 1 1 0",
        " 2 1 1 0 0",
        " 0 1",
        " 0 0",
        " 0 2 0",
        " 0 0 0 1",
        " 0 0",
        " 2 0",
        " 0 0",
        " 0 0 0 0 0",
        "O0 0",
        "o0",
        "o44",
        "v0",
        "o43",
        "v1",
        "C0",
        "n0",
        "x2",
        "0 1",
        "1 1",
        "r",
        "1 5",
        "b",
        "0 0.1 10",
        "0 0.1 10",
        "k1",
        "1",
        "J0 2",
        "0 1",
        "1 1",
    ]
    return "\n".join(lines) + "\n"


def _mixed_integer_nl() -> str:
    """3 vars: z continuous, x binary, y integer."""
    lines = [
        "g3 1 1 0",
        " 3 1 1 0 0",
        " 0 0",
        " 0 0 0",
        " 0 0 0",
        " 0 0 0 1",
        " 1 1",
        " 3 3",
        " 0 0",
        " 0 0 0 0 0",
        "O0 0",
        "n0",
        "C0",
        "n0",
        "x3",
        "0 0",
        "1 0",
        "2 0",
        "r",
        "2 1",
        "b",
        "0 0 10",
        "0 0 1",
        "0 0 5",
        "k2",
        "1",
        "2",
        "J0 3",
        "0 3",
        "1 1",
        "2 2",
        "G0 3",
        "0 3",
        "1 1",
        "2 2",
    ]
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────
# Helper to write .nl content to a temp file
# ─────────────────────────────────────────────────────────────


def _write_nl(content: str) -> str:
    """Write .nl content to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".nl", prefix="test_nl_")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


# ─────────────────────────────────────────────────────────────
# Tests: parse_nl_string
# ─────────────────────────────────────────────────────────────


class TestParseNlString:
    def test_linear_n_vars(self):
        model = parse_nl_string(_linear_nl())
        assert model.n_vars == 2

    def test_linear_n_constraints(self):
        model = parse_nl_string(_linear_nl())
        assert model.n_constraints == 1

    def test_linear_objective_sense(self):
        model = parse_nl_string(_linear_nl())
        assert model.objective_sense == "minimize"

    def test_linear_objective_eval(self):
        model = parse_nl_string(_linear_nl())
        x = np.array([1.0, 2.0])
        val = model.evaluate_objective(x)
        assert abs(val - 8.0) < 1e-12

    def test_linear_constraint_eval(self):
        model = parse_nl_string(_linear_nl())
        x = np.array([3.0, 4.0])
        val = model.evaluate_constraint(0, x)
        assert abs(val - 7.0) < 1e-12

    def test_quadratic_n_vars(self):
        model = parse_nl_string(_quadratic_nl())
        assert model.n_vars == 2

    def test_quadratic_objective_eval(self):
        model = parse_nl_string(_quadratic_nl())
        x = np.array([3.0, 4.0])
        val = model.evaluate_objective(x)
        assert abs(val - 25.0) < 1e-12

    def test_quadratic_structure(self):
        model = parse_nl_string(_quadratic_nl())
        assert model.is_objective_quadratic()
        assert not model.is_objective_linear()

    def test_nonlinear_objective_eval(self):
        model = parse_nl_string(_nonlinear_nl())
        x = np.array([1.0, np.e])
        val = model.evaluate_objective(x)
        expected = np.e + 1.0
        assert abs(val - expected) < 1e-12

    def test_nonlinear_structure(self):
        model = parse_nl_string(_nonlinear_nl())
        assert not model.is_objective_linear()
        assert not model.is_objective_quadratic()

    def test_mixed_integer_var_types(self):
        model = parse_nl_string(_mixed_integer_nl())
        types = model.var_types()
        assert types[0] == "continuous"
        assert types[1] == "binary"
        assert types[2] == "integer"

    def test_mixed_integer_eval(self):
        model = parse_nl_string(_mixed_integer_nl())
        x = np.array([2.0, 1.0, 3.0])
        val = model.evaluate_objective(x)
        assert abs(val - 13.0) < 1e-12


# ─────────────────────────────────────────────────────────────
# Tests: parse_nl_file
# ─────────────────────────────────────────────────────────────


class TestParseNlFile:
    def test_file_linear(self):
        path = _write_nl(_linear_nl())
        try:
            model = parse_nl_file(path)
            assert model.n_vars == 2
            x = np.array([1.0, 2.0])
            assert abs(model.evaluate_objective(x) - 8.0) < 1e-12
        finally:
            os.unlink(path)

    def test_file_nonexistent(self):
        with pytest.raises(ValueError):
            parse_nl_file("/nonexistent/path.nl")

    def test_file_quadratic(self):
        path = _write_nl(_quadratic_nl())
        try:
            model = parse_nl_file(path)
            assert model.n_vars == 2
            x = np.array([3.0, 4.0])
            assert abs(model.evaluate_objective(x) - 25.0) < 1e-12
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────
# Tests: variable bounds
# ─────────────────────────────────────────────────────────────


class TestVarBounds:
    def test_linear_bounds(self):
        model = parse_nl_string(_linear_nl())
        lb0 = model.var_lb(0)
        ub0 = model.var_ub(0)
        assert abs(lb0[0] - 0.0) < 1e-12
        assert abs(ub0[0] - 100.0) < 1e-12

    def test_mixed_integer_binary_bounds(self):
        model = parse_nl_string(_mixed_integer_nl())
        lb1 = model.var_lb(1)
        ub1 = model.var_ub(1)
        assert abs(lb1[0] - 0.0) < 1e-12
        assert abs(ub1[0] - 1.0) < 1e-12
