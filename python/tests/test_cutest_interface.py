"""
Tests for the CUTEst interface module.

Tests are organized as:
  - Unit tests using mocked pycutest (always run)
  - Integration tests requiring real pycutest installation (marked @requires_pycutest)

To run integration tests: pytest python/tests/test_cutest_interface.py -m cutest
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────
# Fixtures and helpers
# ─────────────────────────────────────────────────────────────


def _check_pycutest_available():
    """Check if pycutest is installed and functional."""
    try:
        import pycutest  # noqa: F401

        return True
    except ImportError:
        return False


requires_pycutest = pytest.mark.skipif(
    not _check_pycutest_available(),
    reason="pycutest not installed",
)


def _make_mock_cutest_problem(n=2, m=0, name="MOCK"):
    """Create a mock pycutest problem for unit testing."""
    mock_prob = MagicMock()
    mock_prob.n = n
    mock_prob.m = m
    mock_prob.x0 = np.zeros(n)
    mock_prob.bl = np.full(n, -10.0)
    mock_prob.bu = np.full(n, 10.0)

    if m > 0:
        mock_prob.cl = np.full(m, -1e20)
        mock_prob.cu = np.zeros(m)
        mock_prob.is_eq_cons = np.zeros(m, dtype=bool)
        mock_prob.is_linear_cons = np.zeros(m, dtype=bool)
    else:
        mock_prob.cl = np.empty(0)
        mock_prob.cu = np.empty(0)
        mock_prob.is_eq_cons = np.empty(0, dtype=bool)
        mock_prob.is_linear_cons = np.empty(0, dtype=bool)

    # Rosenbrock objective for 2D: f(x) = (1-x0)^2 + 100*(x1-x0^2)^2
    def obj_fn(x, gradient=False):
        f = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        if gradient:
            g = np.zeros(n)
            g[0] = -2 * (1 - x[0]) + 200 * (x[1] - x[0] ** 2) * (-2 * x[0])
            g[1] = 200 * (x[1] - x[0] ** 2)
            return f, g
        return f

    mock_prob.obj = MagicMock(side_effect=obj_fn)

    # Hessian
    def hess_fn(x, v=None):
        H = np.zeros((n, n))
        H[0, 0] = 2 + 200 * (2 * x[0] ** 2 - (x[1] - x[0] ** 2) * 2)
        H[0, 1] = -400 * x[0]
        H[1, 0] = -400 * x[0]
        H[1, 1] = 200
        return H

    mock_prob.hess = MagicMock(side_effect=hess_fn)
    mock_prob.ihess = MagicMock(side_effect=hess_fn)

    # Constraints (empty for unconstrained)
    mock_prob.cons = MagicMock(return_value=np.empty(0))
    mock_prob.jac = MagicMock(return_value=np.empty((0, n)))

    # Classification
    mock_prob.getinfo = MagicMock(return_value={"classification": "OUR2-AN-2-0"})

    return mock_prob


# ─────────────────────────────────────────────────────────────
# Unit tests (mocked pycutest)
# ─────────────────────────────────────────────────────────────


class TestNLPEvaluatorFromCUTEst:
    """Test the evaluator interface with mocked CUTEst problem."""

    def setup_method(self):
        """Set up mock pycutest and create evaluator."""
        self.mock_prob = _make_mock_cutest_problem(n=2, m=0)
        self.mock_pycutest = MagicMock()
        self.mock_pycutest.import_problem = MagicMock(return_value=self.mock_prob)
        self.mock_pycutest.find_problems = MagicMock(return_value=["MOCK"])
        self.mock_pycutest.problem_properties = MagicMock(return_value={"n": 2, "m": 0})

    def _create_evaluator(self):
        """Create evaluator with mocked pycutest."""
        with patch.dict(sys.modules, {"pycutest": self.mock_pycutest}):
            # Need to reimport to pick up the mock
            import importlib

            import discopt.interfaces.cutest as cutest_mod

            importlib.reload(cutest_mod)

            prob = cutest_mod.CUTEstProblem("MOCK")
            return prob.to_evaluator(), prob

    def test_n_variables(self):
        evaluator, prob = self._create_evaluator()
        assert evaluator.n_variables == 2
        prob.close()

    def test_n_constraints(self):
        evaluator, prob = self._create_evaluator()
        assert evaluator.n_constraints == 0
        prob.close()

    def test_variable_bounds(self):
        evaluator, prob = self._create_evaluator()
        lb, ub = evaluator.variable_bounds
        np.testing.assert_array_equal(lb, np.full(2, -10.0))
        np.testing.assert_array_equal(ub, np.full(2, 10.0))
        prob.close()

    def test_evaluate_objective(self):
        evaluator, prob = self._create_evaluator()
        x = np.array([1.0, 1.0])
        obj = evaluator.evaluate_objective(x)
        # f(1, 1) = 0 for Rosenbrock
        assert abs(obj) < 1e-10
        prob.close()

    def test_evaluate_objective_at_origin(self):
        evaluator, prob = self._create_evaluator()
        x = np.array([0.0, 0.0])
        obj = evaluator.evaluate_objective(x)
        # f(0, 0) = 1 for Rosenbrock
        assert abs(obj - 1.0) < 1e-10
        prob.close()

    def test_evaluate_gradient(self):
        evaluator, prob = self._create_evaluator()
        x = np.array([1.0, 1.0])
        grad = evaluator.evaluate_gradient(x)
        assert grad.shape == (2,)
        # At optimum, gradient should be zero
        np.testing.assert_allclose(grad, [0.0, 0.0], atol=1e-8)
        prob.close()

    def test_evaluate_hessian(self):
        evaluator, prob = self._create_evaluator()
        x = np.array([1.0, 1.0])
        H = evaluator.evaluate_hessian(x)
        assert H.shape == (2, 2)
        # Hessian from mock at (1,1):
        # H[0,0] = 2 + 200*(2*1 - 0*2) = 402
        # H[0,1] = H[1,0] = -400
        # H[1,1] = 200
        np.testing.assert_allclose(H[0, 0], 402.0, atol=1e-6)
        np.testing.assert_allclose(H[0, 1], -400.0, atol=1e-6)
        np.testing.assert_allclose(H[1, 1], 200.0, atol=1e-6)
        prob.close()

    def test_evaluate_constraints_unconstrained(self):
        evaluator, prob = self._create_evaluator()
        x = np.array([1.0, 1.0])
        c = evaluator.evaluate_constraints(x)
        assert c.shape == (0,)
        prob.close()

    def test_evaluate_jacobian_unconstrained(self):
        evaluator, prob = self._create_evaluator()
        x = np.array([1.0, 1.0])
        J = evaluator.evaluate_jacobian(x)
        assert J.shape == (0, 2)
        prob.close()

    def test_constraint_bounds_unconstrained(self):
        evaluator, prob = self._create_evaluator()
        assert evaluator.constraint_bounds is None
        prob.close()


class TestCUTEstProblemInfo:
    """Test problem metadata extraction."""

    def setup_method(self):
        self.mock_prob = _make_mock_cutest_problem(n=2, m=0)
        self.mock_pycutest = MagicMock()
        self.mock_pycutest.import_problem = MagicMock(return_value=self.mock_prob)

    def test_classification_parsing(self):
        with patch.dict(sys.modules, {"pycutest": self.mock_pycutest}):
            import importlib

            import discopt.interfaces.cutest as cutest_mod

            importlib.reload(cutest_mod)

            prob = cutest_mod.CUTEstProblem("MOCK")
            info = prob.info
            assert info.name == "MOCK"
            assert info.n == 2
            assert info.m == 0
            assert info.objective_type == "other"  # "O" code
            assert info.constraint_type == "unconstrained"  # "U" code
            prob.close()

    def test_repr(self):
        with patch.dict(sys.modules, {"pycutest": self.mock_pycutest}):
            import importlib

            import discopt.interfaces.cutest as cutest_mod

            importlib.reload(cutest_mod)

            prob = cutest_mod.CUTEstProblem("MOCK")
            r = repr(prob)
            assert "MOCK" in r
            assert "n=2" in r
            assert "m=0" in r
            prob.close()


class TestImportGuard:
    """Test that missing pycutest gives helpful error."""

    def test_require_pycutest_error(self):
        with patch.dict(sys.modules, {"pycutest": None}):
            import importlib

            import discopt.interfaces.cutest as cutest_mod

            importlib.reload(cutest_mod)

            # Force _HAS_PYCUTEST to False
            cutest_mod._HAS_PYCUTEST = False

            with pytest.raises(ImportError, match="pycutest is required"):
                cutest_mod._require_pycutest()


class TestProblemDiscovery:
    """Test problem listing/filtering with mocked pycutest."""

    def setup_method(self):
        self.mock_pycutest = MagicMock()
        self.mock_pycutest.find_problems = MagicMock(
            return_value=["ROSENBR", "BEALE", "HS035", "BIGPROBLEM"]
        )
        self.mock_pycutest.problem_properties = MagicMock(
            side_effect=lambda name: {
                "ROSENBR": {"n": 2, "m": 0},
                "BEALE": {"n": 2, "m": 0},
                "HS035": {"n": 3, "m": 1},
                "BIGPROBLEM": {"n": 500, "m": 200},
            }.get(name, {"n": 0, "m": 0})
        )

    def test_list_all(self):
        with patch.dict(sys.modules, {"pycutest": self.mock_pycutest}):
            import importlib

            import discopt.interfaces.cutest as cutest_mod

            importlib.reload(cutest_mod)

            problems = cutest_mod.list_cutest_problems()
            assert len(problems) == 4

    def test_list_with_max_n(self):
        with patch.dict(sys.modules, {"pycutest": self.mock_pycutest}):
            import importlib

            import discopt.interfaces.cutest as cutest_mod

            importlib.reload(cutest_mod)

            problems = cutest_mod.list_cutest_problems(max_n=10)
            assert "BIGPROBLEM" not in problems
            assert "ROSENBR" in problems

    def test_list_with_max_m(self):
        with patch.dict(sys.modules, {"pycutest": self.mock_pycutest}):
            import importlib

            import discopt.interfaces.cutest as cutest_mod

            importlib.reload(cutest_mod)

            problems = cutest_mod.list_cutest_problems(max_m=0)
            assert "HS035" not in problems
            assert "BIGPROBLEM" not in problems


# ─────────────────────────────────────────────────────────────
# Integration tests (real pycutest required)
# ─────────────────────────────────────────────────────────────


@requires_pycutest
class TestCUTEstIntegration:
    """Integration tests with real CUTEst problems. Requires pycutest."""

    def test_rosenbrock_load(self):
        from discopt.interfaces.cutest import load_cutest_problem

        prob = load_cutest_problem("ROSENBR")
        assert prob.n == 2
        assert prob.m == 0
        assert prob.x0.shape == (2,)
        prob.close()

    def test_rosenbrock_evaluate(self):
        from discopt.interfaces.cutest import load_cutest_problem

        prob = load_cutest_problem("ROSENBR")
        evaluator = prob.to_evaluator()

        # f(1, 1) = 0
        obj = evaluator.evaluate_objective(np.array([1.0, 1.0]))
        assert abs(obj) < 1e-10

        # Gradient at optimum should be zero
        grad = evaluator.evaluate_gradient(np.array([1.0, 1.0]))
        np.testing.assert_allclose(grad, 0.0, atol=1e-6)

        prob.close()

    def test_rosenbrock_solve_with_ipopt(self):
        """Full integration: load CUTEst problem, solve with discopt's Ipopt."""
        pytest.importorskip("cyipopt")

        from discopt.interfaces.cutest import load_cutest_problem
        from discopt.solvers.nlp_ipopt import solve_nlp

        prob = load_cutest_problem("ROSENBR")
        evaluator = prob.to_evaluator()

        result = solve_nlp(evaluator, prob.x0, options={"print_level": 0})

        assert result.status.value == "optimal"
        assert abs(result.objective) < 1e-6
        np.testing.assert_allclose(result.x, [1.0, 1.0], atol=1e-4)

        prob.close()

    def test_constrained_problem(self):
        """Test a constrained CUTEst problem (HS071)."""
        from discopt.interfaces.cutest import load_cutest_problem

        prob = load_cutest_problem("HS071")
        assert prob.n == 4
        assert prob.m > 0

        evaluator = prob.to_evaluator()
        assert evaluator.n_variables == 4
        assert evaluator.n_constraints > 0

        # Evaluate at starting point
        x0 = prob.x0
        obj = evaluator.evaluate_objective(x0)
        assert np.isfinite(obj)

        c = evaluator.evaluate_constraints(x0)
        assert c.shape[0] == prob.m

        J = evaluator.evaluate_jacobian(x0)
        assert J.shape == (prob.m, prob.n)

        prob.close()

    def test_problem_info(self):
        from discopt.interfaces.cutest import load_cutest_problem

        prob = load_cutest_problem("ROSENBR")
        info = prob.info
        assert info.name == "ROSENBR"
        assert info.n == 2
        assert info.m == 0
        assert len(info.classification) > 0
        prob.close()

    def test_list_problems(self):
        from discopt.interfaces.cutest import list_cutest_problems

        problems = list_cutest_problems(constraints="U", max_n=5)
        assert len(problems) > 0
        assert "ROSENBR" in problems
