"""
NLP convergence rate validation on CUTEst-style problems.

Tests 20+ NLP problems across standard problem classes:
1. Unconstrained optimization
2. Bound-constrained
3. Equality-constrained
4. Inequality-constrained
5. Mixed constraints
6. Nonlinear least-squares
7. Larger problems (10-50 variables)
8. Problems with multiple local minima

Convergence criterion: >= 80% of problems converge to OPTIMAL.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import builtins

import numpy as np
import pytest

cyipopt = pytest.importorskip("cyipopt")

from discopt.modeling.core import Model, cos, exp, log, sin, sum  # noqa: E402
from discopt.solvers import SolveStatus  # noqa: E402
from discopt.solvers.nlp_ipopt import solve_nlp_from_model  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.integration]

# Default Ipopt options. We use limited-memory Hessian approximation because
# the current _IpoptCallbacks.hessian only provides objective Hessian, not the
# full Lagrangian Hessian (constraint second derivatives are missing).
_DEFAULT_OPTS: dict = {
    "max_iter": 3000,
    "print_level": 0,
    "hessian_approximation": "limited-memory",
}

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

# Collect results across all parametrized tests for the summary.
_CONVERGENCE_RESULTS: dict[str, bool] = {}


def _solve_and_check(
    model: Model,
    name: str,
    x0: np.ndarray | None = None,
    expected_obj: float | None = None,
    expected_x: np.ndarray | None = None,
    atol_obj: float = 1e-3,
    atol_x: float = 1e-3,
    options: dict | None = None,
) -> None:
    """Solve an NLP and record convergence. Assertions use loose tolerances."""
    opts = dict(_DEFAULT_OPTS)
    if options:
        opts.update(options)

    result = solve_nlp_from_model(model, x0=x0, options=opts)
    converged = result.status == SolveStatus.OPTIMAL
    _CONVERGENCE_RESULTS[name] = converged

    assert converged, (
        f"Problem '{name}' did not converge: status={result.status}, obj={result.objective}"
    )

    if expected_obj is not None:
        assert abs(result.objective - expected_obj) < atol_obj, (
            f"Problem '{name}': expected obj={expected_obj}, got {result.objective}"
        )

    if expected_x is not None:
        assert np.allclose(result.x, expected_x, atol=atol_x), (
            f"Problem '{name}': expected x={expected_x}, got {result.x}"
        )


# ─────────────────────────────────────────────────────────────
# 1. Unconstrained optimization
# ─────────────────────────────────────────────────────────────


class TestUnconstrained:
    def test_rosenbrock_2d(self):
        """Rosenbrock: min (1-x)^2 + 100*(y-x^2)^2. Optimal at (1,1), obj=0."""
        m = Model("rosenbrock_2d")
        x = m.continuous("x", lb=-5, ub=5)
        y = m.continuous("y", lb=-5, ub=5)
        m.minimize((1 - x) ** 2 + 100 * (y - x**2) ** 2)
        _solve_and_check(
            m,
            "rosenbrock_2d",
            np.array([-1.0, 1.0]),
            expected_obj=0.0,
            expected_x=np.array([1.0, 1.0]),
        )

    def test_beale(self):
        """Beale function. Optimal at (3, 0.5), obj=0."""
        m = Model("beale")
        x = m.continuous("x", lb=-4.5, ub=4.5)
        y = m.continuous("y", lb=-4.5, ub=4.5)
        m.minimize(
            (1.5 - x + x * y) ** 2 + (2.25 - x + x * y**2) ** 2 + (2.625 - x + x * y**3) ** 2
        )
        _solve_and_check(
            m,
            "beale",
            np.array([1.0, 1.0]),
            expected_obj=0.0,
            expected_x=np.array([3.0, 0.5]),
            atol_x=1e-2,
        )

    def test_booth(self):
        """Booth function: min (x+2y-7)^2 + (2x+y-5)^2. Optimal at (1,3), obj=0."""
        m = Model("booth")
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)
        m.minimize((x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2)
        _solve_and_check(
            m, "booth", np.array([0.0, 0.0]), expected_obj=0.0, expected_x=np.array([1.0, 3.0])
        )

    def test_himmelblau(self):
        """Himmelblau: min (x^2+y-11)^2 + (x+y^2-7)^2. One min near (3,2), obj=0."""
        m = Model("himmelblau")
        x = m.continuous("x", lb=-5, ub=5)
        y = m.continuous("y", lb=-5, ub=5)
        m.minimize((x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2)
        _solve_and_check(m, "himmelblau", np.array([2.0, 2.0]), expected_obj=0.0, atol_obj=1e-6)

    def test_matyas(self):
        """Matyas: min 0.26*(x^2+y^2) - 0.48*x*y. Optimal at (0,0), obj=0."""
        m = Model("matyas")
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)
        m.minimize(0.26 * (x**2 + y**2) - 0.48 * x * y)
        _solve_and_check(
            m, "matyas", np.array([5.0, 5.0]), expected_obj=0.0, expected_x=np.array([0.0, 0.0])
        )


# ─────────────────────────────────────────────────────────────
# 2. Bound-constrained
# ─────────────────────────────────────────────────────────────


class TestBoundConstrained:
    def test_quadratic_box(self):
        """min (x-3)^2 + (y-4)^2 with 0 <= x <= 2, 0 <= y <= 2.
        Optimal at (2, 2), obj=5."""
        m = Model("quad_box")
        x = m.continuous("x", lb=0, ub=2)
        y = m.continuous("y", lb=0, ub=2)
        m.minimize((x - 3) ** 2 + (y - 4) ** 2)
        _solve_and_check(
            m, "quad_box", np.array([1.0, 1.0]), expected_obj=5.0, expected_x=np.array([2.0, 2.0])
        )

    def test_exp_box(self):
        """min exp(x) + exp(y) with 1 <= x,y <= 5. Optimal at (1, 1)."""
        m = Model("exp_box")
        x = m.continuous("x", lb=1, ub=5)
        y = m.continuous("y", lb=1, ub=5)
        m.minimize(exp(x) + exp(y))
        expected = 2 * np.exp(1.0)
        _solve_and_check(m, "exp_box", np.array([3.0, 3.0]), expected_obj=expected)

    def test_log_barrier_like(self):
        """min x^2 - log(x) with 0.01 <= x <= 10. Optimal at x=1/sqrt(2)."""
        m = Model("log_barrier")
        x = m.continuous("x", lb=0.01, ub=10)
        m.minimize(x**2 - log(x))
        x_opt = 1.0 / np.sqrt(2.0)
        obj_opt = x_opt**2 - np.log(x_opt)
        _solve_and_check(
            m,
            "log_barrier",
            np.array([1.0]),
            expected_obj=obj_opt,
            expected_x=np.array([x_opt]),
            atol_x=1e-4,
        )


# ─────────────────────────────────────────────────────────────
# 3. Equality-constrained
# ─────────────────────────────────────────────────────────────


class TestEqualityConstrained:
    def test_circle_min(self):
        """min x + y s.t. x^2 + y^2 == 1.
        Optimal at (-1/sqrt(2), -1/sqrt(2)), obj = -sqrt(2)."""
        m = Model("circle_min")
        x = m.continuous("x", lb=-2, ub=2)
        y = m.continuous("y", lb=-2, ub=2)
        m.minimize(x + y)
        m.subject_to(x**2 + y**2 == 1)
        s = 1.0 / np.sqrt(2.0)
        _solve_and_check(
            m,
            "circle_min",
            np.array([-0.5, -0.5]),
            expected_obj=-np.sqrt(2.0),
            expected_x=np.array([-s, -s]),
        )

    def test_equality_quadratic(self):
        """min x^2 + y^2 + z^2 s.t. x + y + z == 3.
        Optimal at (1, 1, 1), obj=3."""
        m = Model("eq_quad_3d")
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)
        z = m.continuous("z", lb=-10, ub=10)
        m.minimize(x**2 + y**2 + z**2)
        m.subject_to(x + y + z == 3)
        _solve_and_check(
            m,
            "eq_quad_3d",
            np.array([2.0, 0.0, 1.0]),
            expected_obj=3.0,
            expected_x=np.array([1.0, 1.0, 1.0]),
        )

    def test_nonlinear_equality(self):
        """min x^2 + y^2 s.t. x*y == 1, x > 0, y > 0.
        By AM-GM: x^2+y^2 >= 2xy = 2, equality at x=y=1."""
        m = Model("nl_eq")
        x = m.continuous("x", lb=0.1, ub=10)
        y = m.continuous("y", lb=0.1, ub=10)
        m.minimize(x**2 + y**2)
        m.subject_to(x * y == 1)
        _solve_and_check(
            m, "nl_eq", np.array([2.0, 0.5]), expected_obj=2.0, expected_x=np.array([1.0, 1.0])
        )


# ─────────────────────────────────────────────────────────────
# 4. Inequality-constrained
# ─────────────────────────────────────────────────────────────


class TestInequalityConstrained:
    def test_linear_ineq(self):
        """min -x - y s.t. x + 2y <= 4, x + y <= 3, x,y >= 0.
        Optimal obj = -3 (attained along the edge x+y=3, x+2y<=4)."""
        m = Model("lin_ineq")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(-x - y)
        m.subject_to(x + 2 * y <= 4)
        m.subject_to(x + y <= 3)
        _solve_and_check(m, "lin_ineq", np.array([1.0, 1.0]), expected_obj=-3.0)

    def test_nonlinear_ineq(self):
        """min x^2 + y^2 s.t. x + y >= 2. Optimal at (1, 1), obj=2."""
        m = Model("nl_ineq")
        x = m.continuous("x", lb=-10, ub=10)
        y = m.continuous("y", lb=-10, ub=10)
        m.minimize(x**2 + y**2)
        m.subject_to(x + y >= 2)
        _solve_and_check(
            m, "nl_ineq", np.array([3.0, 3.0]), expected_obj=2.0, expected_x=np.array([1.0, 1.0])
        )

    def test_quadratic_ineq(self):
        """min x + y s.t. x^2 + y^2 <= 1, x >= 0, y >= 0.
        Optimal at (0, 0)... actually min on boundary at appropriate point.
        min x+y s.t. x^2+y^2 <= 1 => optimal at (-1/sqrt(2), -1/sqrt(2))
        but with x,y >= 0 => optimal at (0,0), obj=0."""
        m = Model("quad_ineq")
        x = m.continuous("x", lb=0, ub=5)
        y = m.continuous("y", lb=0, ub=5)
        m.minimize(x + y)
        m.subject_to(x**2 + y**2 <= 1)
        _solve_and_check(
            m, "quad_ineq", np.array([0.5, 0.5]), expected_obj=0.0, expected_x=np.array([0.0, 0.0])
        )


# ─────────────────────────────────────────────────────────────
# 5. Mixed constraints (equalities + inequalities)
# ─────────────────────────────────────────────────────────────


class TestMixedConstraints:
    def test_mixed_eq_ineq(self):
        """min x^2 + y^2 + z^2 s.t. x + y + z == 1, x >= 0, y >= 0, z <= 0.5.
        KKT: optimal near (0.25, 0.25, 0.5) with z at bound, obj=0.375."""
        m = Model("mixed")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        z = m.continuous("z", lb=-10, ub=0.5)
        m.minimize(x**2 + y**2 + z**2)
        m.subject_to(x + y + z == 1)
        # By KKT: x=y (symmetry), z=0.5 at bound or z free
        # If z free: x=y=z=1/3, obj=1/3. But z <= 0.5 so z=1/3 is feasible.
        _solve_and_check(
            m, "mixed", np.array([0.5, 0.3, 0.2]), expected_obj=1.0 / 3.0, atol_obj=1e-4
        )

    def test_engineering_design(self):
        """Simple engineering design: min area = x*y s.t. 2x + 2y >= 10 (perimeter),
        x, y in [0.5, 10]. Optimal: x=y=2.5, obj=6.25."""
        m = Model("eng_design")
        x = m.continuous("x", lb=0.5, ub=10)
        y = m.continuous("y", lb=0.5, ub=10)
        m.minimize(x * y)
        m.subject_to(2 * x + 2 * y >= 10)
        _solve_and_check(
            m,
            "eng_design",
            np.array([3.0, 3.0]),
            expected_obj=6.25,
            expected_x=np.array([2.5, 2.5]),
        )


# ─────────────────────────────────────────────────────────────
# 6. Nonlinear least-squares
# ─────────────────────────────────────────────────────────────


class TestLeastSquares:
    def test_linear_regression(self):
        """Fit y = a*x + b to data. Known solution via normal equations."""
        m = Model("linreg")
        a = m.continuous("a", lb=-10, ub=10)
        b = m.continuous("b", lb=-10, ub=10)
        # Data: (0,1), (1,3), (2,5) => perfect fit a=2, b=1
        m.minimize((b - 1) ** 2 + (a + b - 3) ** 2 + (2 * a + b - 5) ** 2)
        _solve_and_check(
            m, "linreg", np.array([0.0, 0.0]), expected_obj=0.0, expected_x=np.array([2.0, 1.0])
        )

    def test_exponential_fit(self):
        """Fit y = a * exp(b * t) to data points.
        Data generated from a=1, b=0.5 at t=0,1,2."""
        m = Model("expfit")
        a = m.continuous("a", lb=0.01, ub=10)
        b = m.continuous("b", lb=-5, ub=5)
        # t=0: y=1.0, t=1: y=exp(0.5)~1.6487, t=2: y=exp(1)~2.7183
        y0 = 1.0
        y1 = np.exp(0.5)
        y2 = np.exp(1.0)
        m.minimize((a - y0) ** 2 + (a * exp(b) - y1) ** 2 + (a * exp(2 * b) - y2) ** 2)
        _solve_and_check(
            m,
            "expfit",
            np.array([1.5, 0.3]),
            expected_obj=0.0,
            expected_x=np.array([1.0, 0.5]),
            atol_x=1e-2,
        )


# ─────────────────────────────────────────────────────────────
# 7. Larger problems (10-50 variables)
# ─────────────────────────────────────────────────────────────


class TestLargerProblems:
    def test_sum_of_squares_20(self):
        """min sum((x_i - i)^2) for i in 0..19. Optimal at x_i=i, obj=0."""
        n = 20
        m = Model("sos_20")
        xs = [m.continuous(f"x{i}", lb=-50, ub=50) for i in range(n)]
        m.minimize(sum([(xs[i] - float(i)) ** 2 for i in range(n)]))
        x0 = np.zeros(n)
        expected_x = np.arange(n, dtype=float)
        _solve_and_check(m, "sos_20", x0, expected_obj=0.0, expected_x=expected_x)

    def test_chained_rosenbrock_10(self):
        """Extended Rosenbrock in 10D: sum (1-x_i)^2 + 100*(x_{i+1}-x_i^2)^2.
        Optimal at all x_i=1, obj=0."""
        n = 10
        m = Model("rosenbrock_10d")
        xs = [m.continuous(f"x{i}", lb=-5, ub=5) for i in range(n)]
        terms = []
        for i in range(n - 1):
            terms.append((1 - xs[i]) ** 2 + 100 * (xs[i + 1] - xs[i] ** 2) ** 2)
        m.minimize(sum(terms))
        x0 = np.full(n, -1.0)
        expected_x = np.ones(n)
        _solve_and_check(
            m,
            "rosenbrock_10d",
            x0,
            expected_obj=0.0,
            expected_x=expected_x,
            atol_x=1e-2,
            atol_obj=1e-2,
        )

    def test_constrained_20var(self):
        """min sum(x_i^2) s.t. sum(x_i) == 10, n=20.
        Optimal: x_i = 0.5 for all i, obj = 20 * 0.25 = 5.0."""
        n = 20
        m = Model("constrained_20")
        xs = [m.continuous(f"x{i}", lb=-10, ub=10) for i in range(n)]
        m.minimize(sum([xs[i] ** 2 for i in range(n)]))
        m.subject_to(sum([xs[i] for i in range(n)]) == 10)
        x0 = np.ones(n)
        expected_x = np.full(n, 0.5)
        _solve_and_check(m, "constrained_20", x0, expected_obj=5.0, expected_x=expected_x)

    def test_portfolio_like_10(self):
        """Markowitz-like: min x'Qx s.t. sum(x) == 1, x >= 0.
        Q = I (identity) => optimal x_i = 1/n, obj = 1/n."""
        n = 10
        m = Model("portfolio_10")
        xs = [m.continuous(f"x{i}", lb=0, ub=1) for i in range(n)]
        # Diagonal Q = I => objective is sum of x_i^2
        m.minimize(sum([xs[i] ** 2 for i in range(n)]))
        m.subject_to(sum([xs[i] for i in range(n)]) == 1)
        x0 = np.full(n, 1.0 / n)
        expected_x = np.full(n, 1.0 / n)
        _solve_and_check(m, "portfolio_10", x0, expected_obj=1.0 / n, expected_x=expected_x)


# ─────────────────────────────────────────────────────────────
# 8. Problems with multiple local minima
# ─────────────────────────────────────────────────────────────


class TestMultipleMinima:
    def test_six_hump_camel(self):
        """Six-hump camel function (bounded). Global min ~ -1.0316.
        Has multiple local minima. We accept any local minimum (just check convergence)."""
        m = Model("six_hump")
        x = m.continuous("x", lb=-3, ub=3)
        y = m.continuous("y", lb=-2, ub=2)
        m.minimize((4 - 2.1 * x**2 + x**4 / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2)
        # Start near a known global min at ~(0.0898, -0.7126)
        result = solve_nlp_from_model(m, x0=np.array([0.1, -0.7]), options=_DEFAULT_OPTS)
        converged = result.status == SolveStatus.OPTIMAL
        _CONVERGENCE_RESULTS["six_hump"] = converged
        assert converged
        # Global minimum is approximately -1.0316
        assert result.objective < 0.0, "Should find a negative objective"

    def test_trig_multimodal(self):
        """min x^2 + sin(5x) on [-3, 3]. Multiple local minima.
        Just verify convergence to some local min."""
        m = Model("trig_multi")
        x = m.continuous("x", lb=-3, ub=3)
        m.minimize(x**2 + sin(5 * x))
        result = solve_nlp_from_model(m, x0=np.array([0.5]), options=_DEFAULT_OPTS)
        converged = result.status == SolveStatus.OPTIMAL
        _CONVERGENCE_RESULTS["trig_multi"] = converged
        assert converged


# ─────────────────────────────────────────────────────────────
# Additional CUTEst-style problems
# ─────────────────────────────────────────────────────────────


class TestAdditionalProblems:
    def test_powell_singular(self):
        """Powell's singular function (4 variables).
        f = (x1+10*x2)^2 + 5*(x3-x4)^2 + (x2-2*x3)^4 + 10*(x1-x4)^4.
        Optimal at origin, obj=0."""
        m = Model("powell_singular")
        x1 = m.continuous("x1", lb=-10, ub=10)
        x2 = m.continuous("x2", lb=-10, ub=10)
        x3 = m.continuous("x3", lb=-10, ub=10)
        x4 = m.continuous("x4", lb=-10, ub=10)
        m.minimize(
            (x1 + 10 * x2) ** 2 + 5 * (x3 - x4) ** 2 + (x2 - 2 * x3) ** 4 + 10 * (x1 - x4) ** 4
        )
        _solve_and_check(
            m, "powell_singular", np.array([3.0, -1.0, 0.0, 1.0]), expected_obj=0.0, atol_obj=1e-2
        )

    def test_wood_function(self):
        """Wood function (4 variables).
        CUTEst standard. Optimal at (1,1,1,1), obj=0."""
        m = Model("wood")
        x1 = m.continuous("x1", lb=-10, ub=10)
        x2 = m.continuous("x2", lb=-10, ub=10)
        x3 = m.continuous("x3", lb=-10, ub=10)
        x4 = m.continuous("x4", lb=-10, ub=10)
        m.minimize(
            100 * (x2 - x1**2) ** 2
            + (1 - x1) ** 2
            + 90 * (x4 - x3**2) ** 2
            + (1 - x3) ** 2
            + 10.1 * ((x2 - 1) ** 2 + (x4 - 1) ** 2)
            + 19.8 * (x2 - 1) * (x4 - 1)
        )
        _solve_and_check(
            m,
            "wood",
            np.array([-3.0, -1.0, -3.0, -1.0]),
            expected_obj=0.0,
            expected_x=np.array([1.0, 1.0, 1.0, 1.0]),
            atol_x=1e-2,
            atol_obj=1e-2,
        )

    def test_trig_equality(self):
        """min x^2 + y^2 s.t. sin(x) + cos(y) == 1.
        Near (0, 0): sin(0)+cos(0)=1 is feasible, obj=0."""
        m = Model("trig_eq")
        x = m.continuous("x", lb=-3, ub=3)
        y = m.continuous("y", lb=-3, ub=3)
        m.minimize(x**2 + y**2)
        m.subject_to(sin(x) + cos(y) == 1)
        _solve_and_check(
            m,
            "trig_eq",
            np.array([0.1, 0.1]),
            expected_obj=0.0,
            expected_x=np.array([0.0, 0.0]),
            atol_x=1e-3,
        )

    def test_hs071(self):
        """HS071 (Hock-Schittkowski #71):
        min x1*x4*(x1+x2+x3) + x3
        s.t. x1*x2*x3*x4 >= 25
             x1^2 + x2^2 + x3^2 + x4^2 == 40
             1 <= x1,x2,x3,x4 <= 5
        Known optimal obj ~ 17.0140."""
        m = Model("hs071")
        x1 = m.continuous("x1", lb=1, ub=5)
        x2 = m.continuous("x2", lb=1, ub=5)
        x3 = m.continuous("x3", lb=1, ub=5)
        x4 = m.continuous("x4", lb=1, ub=5)
        m.minimize(x1 * x4 * (x1 + x2 + x3) + x3)
        m.subject_to(x1 * x2 * x3 * x4 >= 25)
        m.subject_to(x1**2 + x2**2 + x3**2 + x4**2 == 40)
        _solve_and_check(
            m, "hs071", np.array([1.0, 5.0, 5.0, 1.0]), expected_obj=17.014, atol_obj=0.1
        )


# ─────────────────────────────────────────────────────────────
# Summary test: assert overall convergence rate >= 80%
# ─────────────────────────────────────────────────────────────


class TestConvergenceSummary:
    def test_overall_rate(self):
        """At least 80% of the NLP problems above must converge to OPTIMAL.
        This test must run last (it uses results collected by prior tests)."""
        if len(_CONVERGENCE_RESULTS) == 0:
            pytest.skip("No convergence results collected (run full suite)")

        total = len(_CONVERGENCE_RESULTS)
        converged = builtins.sum(1 for v in _CONVERGENCE_RESULTS.values() if v)
        rate = converged / total

        failed = [k for k, v in _CONVERGENCE_RESULTS.items() if not v]
        msg = f"Convergence rate: {converged}/{total} = {rate:.1%}. Failed: {failed}"
        assert rate >= 0.80, msg
        print(f"\nNLP Convergence Summary: {converged}/{total} = {rate:.1%}")
