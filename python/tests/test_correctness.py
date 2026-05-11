"""
T15: Correctness Validation for discopt Solver

Validates Model.solve() against known optimal values across a comprehensive
suite of test problems. Tests cover:

  - Pure continuous NLP (unconstrained, constrained, nonlinear equality)
  - Mixed-Integer Nonlinear Programming (MINLP) with binary/integer variables
  - Various nonlinear functions: quadratic, exponential, logarithmic, trigonometric
  - Constraint types: inequality, equality, nonlinear
  - Edge cases: tight bounds, binding constraints, multiple optima

Acceptance criteria:
  - Zero incorrect results (non-negotiable)
  - Tolerances: abs_tol=1e-4, rel_tol=1e-3 for objective values
  - All instances solved to optimal or feasible status

Reference: Known optima are analytically derived and independently verified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import discopt.modeling as dm
import pytest
from discopt.modeling.core import Model, SolveResult
from discopt.modeling.examples import example_simple_minlp

# ──────────────────────────────────────────────────────────
# Tolerances
# ──────────────────────────────────────────────────────────

pytestmark = pytest.mark.slow

ABS_TOL = 1e-4
REL_TOL = 1e-3
INTEGRALITY_TOL = 1e-5


def assert_optimal_value(
    result: SolveResult,
    expected: float,
    name: str,
    abs_tol: float = ABS_TOL,
    rel_tol: float = REL_TOL,
) -> None:
    """Assert solver found the correct optimal value within tolerance.

    Uses both absolute and relative tolerance:
        |actual - expected| <= abs_tol + rel_tol * |expected|

    This is the same tolerance check used in numpy.isclose.
    """
    assert result.status in ("optimal", "feasible"), (
        f"[{name}] Expected optimal/feasible, got status={result.status}"
    )
    assert result.objective is not None, f"[{name}] No objective value returned"
    actual = result.objective
    tol = abs_tol + rel_tol * abs(expected)
    assert abs(actual - expected) <= tol, (
        f"[{name}] Objective {actual:.8f} differs from expected {expected:.8f} "
        f"by {abs(actual - expected):.2e} (tolerance {tol:.2e})"
    )


def assert_integer_feasible(
    result: SolveResult,
    integer_var_names: list[str],
    name: str,
    tol: float = INTEGRALITY_TOL,
) -> None:
    """Assert all integer/binary variables are integral within tolerance."""
    assert result.x is not None, f"[{name}] No solution"
    for var_name in integer_var_names:
        val = float(result.x[var_name])
        rounded = round(val)
        assert abs(val - rounded) <= tol, (
            f"[{name}] Variable {var_name}={val:.8f} is not integral "
            f"(nearest integer={rounded}, gap={abs(val - rounded):.2e})"
        )


def assert_bounds_satisfied(
    result: SolveResult,
    bounds: dict[str, tuple[float, float]],
    name: str,
    tol: float = 1e-6,
) -> None:
    """Assert all variable values are within their declared bounds."""
    assert result.x is not None, f"[{name}] No solution"
    for var_name, (lb, ub) in bounds.items():
        val = float(result.x[var_name])
        assert val >= lb - tol, f"[{name}] {var_name}={val:.8f} violates lower bound {lb}"
        assert val <= ub + tol, f"[{name}] {var_name}={val:.8f} violates upper bound {ub}"


# ──────────────────────────────────────────────────────────
# Test Instance Registry
# ──────────────────────────────────────────────────────────


@dataclass
class ProblemInstance:
    """A test problem with known optimal value."""

    name: str
    build_fn: Callable[[], Model]
    expected_obj: float
    integer_vars: list[str]
    bounds: dict[str, tuple[float, float]]
    description: str


def _build_simple_minlp() -> Model:
    """min x1^2 + x2^2 + x3, x1+x2>=1, x1^2+x2<=3, x3 binary, x1,x2 in [0,5]."""
    return example_simple_minlp()


def _build_rosenbrock() -> Model:
    """min (1-x)^2 + 100(y-x^2)^2, x,y in [-5,5]. Optimal at (1,1), obj=0."""
    m = dm.Model("rosenbrock")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.minimize((1 - x) ** 2 + 100 * (y - x**2) ** 2)
    return m


def _build_unconstrained_quadratic() -> Model:
    """min (x-2)^2 + (y+1)^2, x,y in [-5,5]. Optimal at (2,-1), obj=0."""
    m = dm.Model("unconstrained_quad")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.minimize((x - 2) ** 2 + (y + 1) ** 2)
    return m


def _build_constrained_quadratic() -> Model:
    """min x^2 + y^2 s.t. x+y>=1, x,y in [-5,5]. Optimal: x=y=0.5, obj=0.5."""
    m = dm.Model("constrained_quad")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.minimize(x**2 + y**2)
    m.subject_to(x + y >= 1)
    return m


def _build_quadratic_equality() -> Model:
    """min x^2 + y^2 s.t. x+y=2, x,y in [0,3]. Optimal: x=y=1, obj=2."""
    m = dm.Model("quad_equality")
    x = m.continuous("x", lb=0, ub=3)
    y = m.continuous("y", lb=0, ub=3)
    m.minimize(x**2 + y**2)
    m.subject_to(x + y == 2)
    return m


def _build_nonlinear_equality() -> Model:
    """min x^2 + y^2 s.t. x*y=1, x,y in [0.1,5]. Optimal: x=y=1, obj=2."""
    m = dm.Model("nonlinear_eq")
    x = m.continuous("x", lb=0.1, ub=5)
    y = m.continuous("y", lb=0.1, ub=5)
    m.minimize(x**2 + y**2)
    m.subject_to(x * y == 1)
    return m


def _build_exp_nlp() -> Model:
    """min exp(x) + y^2 s.t. x+y>=1, x,y in [-2,2].

    KKT: exp(x)=2y, x+y=1. Numerically: x~=0.3149, y~=0.6851, obj~=1.8395.
    """
    m = dm.Model("exp_nlp")
    x = m.continuous("x", lb=-2, ub=2)
    y = m.continuous("y", lb=-2, ub=2)
    m.minimize(dm.exp(x) + y**2)
    m.subject_to(x + y >= 1)
    return m


def _compute_exp_nlp_optimum() -> float:
    """Compute exact optimum for exp_nlp via KKT conditions."""
    from scipy.optimize import fsolve

    def eq(y_val):
        import numpy as np

        return np.exp(1.0 - y_val) - 2.0 * y_val

    y_opt = float(fsolve(eq, 0.7)[0])
    x_opt = 1.0 - y_opt
    return math.exp(x_opt) + y_opt**2


def _build_trig_nlp() -> Model:
    """min sin(x) + cos(y) s.t. x+y>=1, x,y in [0,3].

    Optimal: x=0, y=3, obj = sin(0)+cos(3) = cos(3) ~= -0.98999.
    """
    m = dm.Model("trig_nlp")
    x = m.continuous("x", lb=0, ub=3)
    y = m.continuous("y", lb=0, ub=3)
    m.minimize(dm.sin(x) + dm.cos(y))
    m.subject_to(x + y >= 1)
    return m


def _build_log_nlp() -> Model:
    """min -log(x+1) + y^2 s.t. x+y>=2, x in [0,5], y in [0,5].

    -log(x+1) is decreasing, y^2 is increasing.
    On x+y=2: f(x) = -log(x+1) + (2-x)^2
    f'(x) = -1/(x+1) - 2(2-x) = 0 => -1/(x+1) + 2(x-2) = 0
    => 2(x-2)(x+1) = 1 => 2x^2-2x-4 = 1 => 2x^2-2x-5 = 0
    => x = (2 + sqrt(4+40))/4 = (2+sqrt(44))/4 ~= (2+6.633)/4 ~= 2.158
    y = 2 - 2.158 = -0.158 < 0, outside bounds.
    So constraint x+y>=2 is active at boundary of feasible region.
    y=0: x>=2, f=-log(x+1), minimized at x=5: -log(6) = -1.7918
    y>0: increases y^2, and we need x>=2-y, -log(x+1) >= -log(5-y+1)
    Check y=0, x=5: obj = -log(6) + 0 = -1.7918
    """
    m = dm.Model("log_nlp")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.minimize(-dm.log(x + 1) + y**2)
    m.subject_to(x + y >= 2)
    return m


def _build_sqrt_nlp() -> Model:
    """min sqrt(x+1) + sqrt(y+1) s.t. x+y>=2, x,y in [0,5].

    sqrt is concave and increasing, so the minimum occurs on the active
    constraint at a boundary point. On x+y=2:
    f(x) = sqrt(x+1) + sqrt(3-x). f'(x) = 1/(2*sqrt(x+1)) - 1/(2*sqrt(3-x)) = 0
    => sqrt(3-x) = sqrt(x+1) => 3-x = x+1 => x=1, y=1.
    Since f is concave, that stationary point is the maximum. The minima are
    at (x, y) = (0, 2) and (2, 0), with obj = 1 + sqrt(3) ~= 2.7321.
    """
    m = dm.Model("sqrt_nlp")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.minimize(dm.sqrt(x + 1) + dm.sqrt(y + 1))
    m.subject_to(x + y >= 2)
    return m


def _build_binary_knapsack() -> Model:
    """min -3x1 - 4x2 - 2x3 s.t. 2x1+3x2+x3<=4, all binary.

    Optimal: x1=0, x2=1, x3=1, obj=-6.
    (x1=1,x2=1: 2+3=5>4 infeasible; x1=1,x2=0,x3=1: 2+0+1=3<=4, obj=-5)
    """
    m = dm.Model("binary_knapsack")
    x1 = m.binary("x1")
    x2 = m.binary("x2")
    x3 = m.binary("x3")
    m.minimize(-3 * x1 - 4 * x2 - 2 * x3)
    m.subject_to(2 * x1 + 3 * x2 + x3 <= 4)
    return m


def _build_multi_binary() -> Model:
    """min x1+x2+x3+x4 s.t. x1+x2>=1, x3+x4>=1, all binary.

    Optimal: two variables at 1 (one from each pair), obj=2.
    """
    m = dm.Model("multi_binary")
    x1 = m.binary("x1")
    x2 = m.binary("x2")
    x3 = m.binary("x3")
    x4 = m.binary("x4")
    m.minimize(x1 + x2 + x3 + x4)
    m.subject_to(x1 + x2 >= 1)
    m.subject_to(x3 + x4 >= 1)
    return m


def _build_quadratic_integer() -> Model:
    """min (x-2.7)^2 + (y-3.3)^2, x,y integer in [0,5].

    Optimal: x=3, y=3, obj = 0.09 + 0.09 = 0.18.
    """
    m = dm.Model("quad_integer")
    m.integer("x", lb=0, ub=5)
    m.integer("y", lb=0, ub=5)
    x = m._variables[0]
    y = m._variables[1]
    m.minimize((x - 2.7) ** 2 + (y - 3.3) ** 2)
    return m


def _build_simple_ip() -> Model:
    """min x + 2y s.t. x>=1, y>=1, x+y>=3, x,y integer in [0,10].

    Optimal: x=2, y=1, obj=4.
    """
    m = dm.Model("simple_ip")
    m.integer("x", lb=0, ub=10)
    m.integer("y", lb=0, ub=10)
    x = m._variables[0]
    y = m._variables[1]
    m.minimize(x + 2 * y)
    m.subject_to(x >= 1)
    m.subject_to(y >= 1)
    m.subject_to(x + y >= 3)
    return m


def _build_convex_minlp() -> Model:
    """min (x-3)^2 + (y-2)^2 + z s.t. x+y>=2, x*y<=8, z binary, x,y in [0,5].

    With z=0: unconstrained min at (3,2): 3*2=6<=8, 3+2=5>=2. obj=0.
    """
    m = dm.Model("convex_minlp")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.binary("z")
    z = m._variables[2]
    m.minimize((x - 3) ** 2 + (y - 2) ** 2 + z)
    m.subject_to(x + y >= 2)
    m.subject_to(x * y <= 8)
    return m


def _build_exp_binary_minlp() -> Model:
    """min exp(x) + 10y s.t. x+y>=1, x in [0,3], y binary.

    y=0: min exp(x) s.t. x>=1 => x=1, obj=e ~= 2.71828.
    y=1: min exp(x)+10 s.t. x>=0 => x=0, obj=11.
    Optimal: y=0, x=1, obj=e.
    """
    m = dm.Model("exp_binary")
    x = m.continuous("x", lb=0, ub=3)
    m.binary("y")
    y = m._variables[1]
    m.minimize(dm.exp(x) + 10 * y)
    m.subject_to(x + y >= 1)
    return m


def _build_log_binary_minlp() -> Model:
    """min -log(x+1) + y s.t. x+y>=2, x in [0,5], y binary.

    y=0: min -log(x+1) s.t. x>=2, -log is decreasing so x=5: -log(6) ~= -1.7918.
    y=1: min -log(x+1)+1 s.t. x>=1, x=5: -log(6)+1 ~= -0.7918.
    Optimal: y=0, x=5, obj=-log(6).
    """
    m = dm.Model("log_binary")
    x = m.continuous("x", lb=0, ub=5)
    m.binary("y")
    y = m._variables[1]
    m.minimize(-dm.log(x + 1) + y)
    m.subject_to(x + y >= 2)
    return m


def _build_linear_minlp() -> Model:
    """min 2x + 3y + z s.t. x+y>=2, y+z>=1, x,y in [0,5], z binary.

    z=0: y>=1, x+y>=2 => min 2x+3y = 2(2-y)+3y = 4+y at y=1,x=1: obj=5.
    z=1: y>=0, x+y>=2 => min 2x+3y+1 = 2*2+0+1=5 at y=0,x=2: obj=5.
    Optimal: obj=5.
    """
    m = dm.Model("linear_minlp")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.binary("z")
    z = m._variables[2]
    m.minimize(2 * x + 3 * y + z)
    m.subject_to(x + y >= 2)
    m.subject_to(y + z >= 1)
    return m


def _build_circle_minlp() -> Model:
    """min x + y + z s.t. x^2+y^2>=1, x+z<=2, x,y in [0,3], z binary.

    z=0: min x+y s.t. x^2+y^2>=1, x<=2.
    At (0,1): 0+1=1>=1 satisfied, 0+0<=2. obj=1.
    At (1/sqrt(2), 1/sqrt(2)): obj=sqrt(2)~=1.414 > 1.
    Optimal: x=0, y=1, z=0, obj=1.
    """
    m = dm.Model("circle_minlp")
    x = m.continuous("x", lb=0, ub=3)
    y = m.continuous("y", lb=0, ub=3)
    m.binary("z")
    z = m._variables[2]
    m.minimize(x + y + z)
    m.subject_to(x**2 + y**2 >= 1)
    m.subject_to(x + z <= 2)
    return m


def _build_three_variable_nlp() -> Model:
    """min x^2 + y^2 + z^2 s.t. x+y+z=3, x,y,z in [0,5].

    By symmetry, optimal: x=y=z=1, obj=3.
    """
    m = dm.Model("three_var_nlp")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    z = m.continuous("z", lb=0, ub=5)
    m.minimize(x**2 + y**2 + z**2)
    m.subject_to(x + y + z == 3)
    return m


def _build_weighted_sum_minlp() -> Model:
    """min x1 + 2*x2 + 3*y s.t. x1+x2>=2, y binary, x1+y>=1, x1,x2 in [0,5].

    y=0: x1>=1, x1+x2>=2. On x1+x2=2 active: obj = x1+2*(2-x1) = 4-x1.
    Decreasing in x1, so maximize x1. x1 can go up to 2 (x2=0>=0).
    At x1=2, x2=0: obj=2.
    y=1: x1>=0, x1+x2>=2. obj = x1+2*x2+3 = x1+2*(2-x1)+3 = 7-x1.
    At x1=2, x2=0: obj=5.
    Optimal overall: y=0, x1=2, x2=0, obj=2.
    """
    m = dm.Model("weighted_sum_minlp")
    x1 = m.continuous("x1", lb=0, ub=5)
    x2 = m.continuous("x2", lb=0, ub=5)
    y = m.binary("y")
    m.minimize(x1 + 2 * x2 + 3 * y)
    m.subject_to(x1 + x2 >= 2)
    m.subject_to(x1 + y >= 1)
    return m


def _build_two_integer_nonlinear() -> Model:
    """min (x-1.5)^2 + (y-2.5)^2 + (z-0.5)^2, x,y integer [0,5], z continuous [0,3].

    Optimal integer: x=2 (or 1), y=3 (or 2). z=0.5 continuous.
    x=2,y=3,z=0.5: (0.5)^2+(0.5)^2+(0)^2=0.5
    x=1,y=2,z=0.5: (0.5)^2+(0.5)^2+(0)^2=0.5
    x=2,y=2,z=0.5: (0.5)^2+(0.5)^2+(0)^2=0.5... wait
    x=2,y=3: (0.5)^2+(0.5)^2=0.5
    x=1,y=3: (0.5)^2+(0.5)^2=0.5
    x=2,y=2: (0.5)^2+(0.5)^2=0.5
    x=1,y=2: (0.5)^2+(0.5)^2=0.5
    All give same. z=0.5 for all. obj=0.5.
    """
    m = dm.Model("two_int_nonlinear")
    m.integer("x", lb=0, ub=5)
    m.integer("y", lb=0, ub=5)
    x = m._variables[0]
    y = m._variables[1]
    z = m.continuous("z", lb=0, ub=3)
    m.minimize((x - 1.5) ** 2 + (y - 2.5) ** 2 + (z - 0.5) ** 2)
    return m


def _build_multiple_constraints_nlp() -> Model:
    """min x^2 + y^2 s.t. x>=0.5, y>=0.5, x+y<=3, x-y<=1, x,y in [0,5].

    Feasible region: x>=0.5, y>=0.5, x+y<=3, x-y<=1.
    Minimum of x^2+y^2 at (0.5, 0.5), obj=0.5.
    Check: 0.5+0.5=1<=3, 0.5-0.5=0<=1. Feasible.
    """
    m = dm.Model("multi_constraint_nlp")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.minimize(x**2 + y**2)
    m.subject_to(x >= 0.5)
    m.subject_to(y >= 0.5)
    m.subject_to(x + y <= 3)
    m.subject_to(x - y <= 1)
    return m


def _build_power_nlp() -> Model:
    """min x^3 - 2*x^2 + x + y^2, x in [0,3], y in [-2,2].

    dF/dx = 3x^2-4x+1 = (3x-1)(x-1) = 0 => x=1/3 or x=1.
    dF/dy = 2y = 0 => y=0.
    f(1/3,0) = 1/27 - 2/9 + 1/3 = 1/27 - 6/27 + 9/27 = 4/27 ~= 0.1481
    f(1,0) = 1 - 2 + 1 = 0
    f(0,0) = 0
    f(3,0) = 27 - 18 + 3 = 12
    Optimal: x=1 or x=0, y=0. f(0,0)=0, f(1,0)=0. Both global min.
    """
    m = dm.Model("power_nlp")
    x = m.continuous("x", lb=0, ub=3)
    y = m.continuous("y", lb=-2, ub=2)
    m.minimize(x**3 - 2 * x**2 + x + y**2)
    return m


# ──────────────────────────────────────────────────────────
# Build the test instance registry
# ──────────────────────────────────────────────────────────

# We compute some expected values that need numerical computation
_EXP_NLP_OPT = _compute_exp_nlp_optimum()


INSTANCES: list[ProblemInstance] = [
    # --- Pure continuous NLP ---
    ProblemInstance(
        name="rosenbrock",
        build_fn=_build_rosenbrock,
        expected_obj=0.0,
        integer_vars=[],
        bounds={"x": (-5, 5), "y": (-5, 5)},
        description="Rosenbrock function, optimal at (1,1)",
    ),
    ProblemInstance(
        name="unconstrained_quadratic",
        build_fn=_build_unconstrained_quadratic,
        expected_obj=0.0,
        integer_vars=[],
        bounds={"x": (-5, 5), "y": (-5, 5)},
        description="Unconstrained quadratic, optimal at (2,-1)",
    ),
    ProblemInstance(
        name="constrained_quadratic",
        build_fn=_build_constrained_quadratic,
        expected_obj=0.5,
        integer_vars=[],
        bounds={"x": (-5, 5), "y": (-5, 5)},
        description="min x^2+y^2 s.t. x+y>=1",
    ),
    ProblemInstance(
        name="quadratic_equality",
        build_fn=_build_quadratic_equality,
        expected_obj=2.0,
        integer_vars=[],
        bounds={"x": (0, 3), "y": (0, 3)},
        description="min x^2+y^2 s.t. x+y=2",
    ),
    ProblemInstance(
        name="nonlinear_equality",
        build_fn=_build_nonlinear_equality,
        expected_obj=2.0,
        integer_vars=[],
        bounds={"x": (0.1, 5), "y": (0.1, 5)},
        description="min x^2+y^2 s.t. x*y=1",
    ),
    ProblemInstance(
        name="exp_nlp",
        build_fn=_build_exp_nlp,
        expected_obj=_EXP_NLP_OPT,
        integer_vars=[],
        bounds={"x": (-2, 2), "y": (-2, 2)},
        description="min exp(x)+y^2 s.t. x+y>=1",
    ),
    ProblemInstance(
        name="trig_nlp",
        build_fn=_build_trig_nlp,
        expected_obj=math.cos(3.0),
        integer_vars=[],
        bounds={"x": (0, 3), "y": (0, 3)},
        description="min sin(x)+cos(y) s.t. x+y>=1",
    ),
    ProblemInstance(
        name="log_nlp",
        build_fn=_build_log_nlp,
        expected_obj=-math.log(6.0),
        integer_vars=[],
        bounds={"x": (0, 5), "y": (0, 5)},
        description="min -log(x+1)+y^2 s.t. x+y>=2",
    ),
    ProblemInstance(
        name="sqrt_nlp",
        build_fn=_build_sqrt_nlp,
        expected_obj=1.0 + math.sqrt(3.0),
        integer_vars=[],
        bounds={"x": (0, 5), "y": (0, 5)},
        description="min sqrt(x+1)+sqrt(y+1) s.t. x+y>=2",
    ),
    ProblemInstance(
        name="three_variable_nlp",
        build_fn=_build_three_variable_nlp,
        expected_obj=3.0,
        integer_vars=[],
        bounds={"x": (0, 5), "y": (0, 5), "z": (0, 5)},
        description="min x^2+y^2+z^2 s.t. x+y+z=3",
    ),
    ProblemInstance(
        name="multiple_constraints_nlp",
        build_fn=_build_multiple_constraints_nlp,
        expected_obj=0.5,
        integer_vars=[],
        bounds={"x": (0, 5), "y": (0, 5)},
        description="min x^2+y^2 with 4 linear constraints",
    ),
    ProblemInstance(
        name="power_nlp",
        build_fn=_build_power_nlp,
        expected_obj=0.0,
        integer_vars=[],
        bounds={"x": (0, 3), "y": (-2, 2)},
        description="min x^3-2x^2+x+y^2, multiple local minima",
    ),
    # --- MINLP with binary variables ---
    ProblemInstance(
        name="simple_minlp",
        build_fn=_build_simple_minlp,
        expected_obj=0.5,
        integer_vars=["x3"],
        bounds={"x1": (0, 5), "x2": (0, 5), "x3": (0, 1)},
        description="Textbook MINLP: min x1^2+x2^2+x3",
    ),
    ProblemInstance(
        name="binary_knapsack",
        build_fn=_build_binary_knapsack,
        expected_obj=-6.0,
        integer_vars=["x1", "x2", "x3"],
        bounds={"x1": (0, 1), "x2": (0, 1), "x3": (0, 1)},
        description="Binary knapsack: 3 items",
    ),
    ProblemInstance(
        name="multi_binary",
        build_fn=_build_multi_binary,
        expected_obj=2.0,
        integer_vars=["x1", "x2", "x3", "x4"],
        bounds={"x1": (0, 1), "x2": (0, 1), "x3": (0, 1), "x4": (0, 1)},
        description="Multi-binary cover: 4 binary vars, 2 cover constraints",
    ),
    ProblemInstance(
        name="quadratic_integer",
        build_fn=_build_quadratic_integer,
        expected_obj=0.18,
        integer_vars=["x", "y"],
        bounds={"x": (0, 5), "y": (0, 5)},
        description="Quadratic with integer vars near non-integer optimum",
    ),
    ProblemInstance(
        name="simple_ip",
        build_fn=_build_simple_ip,
        expected_obj=4.0,
        integer_vars=["x", "y"],
        bounds={"x": (0, 10), "y": (0, 10)},
        description="Simple integer program: min x+2y",
    ),
    ProblemInstance(
        name="convex_minlp",
        build_fn=_build_convex_minlp,
        expected_obj=0.0,
        integer_vars=["z"],
        bounds={"x": (0, 5), "y": (0, 5), "z": (0, 1)},
        description="Convex MINLP with bilinear constraint",
    ),
    ProblemInstance(
        name="exp_binary_minlp",
        build_fn=_build_exp_binary_minlp,
        expected_obj=math.e,
        integer_vars=["y"],
        bounds={"x": (0, 3), "y": (0, 1)},
        description="exp(x) + 10y with binary y",
    ),
    ProblemInstance(
        name="log_binary_minlp",
        build_fn=_build_log_binary_minlp,
        expected_obj=-math.log(6.0),
        integer_vars=["y"],
        bounds={"x": (0, 5), "y": (0, 1)},
        description="-log(x+1) + y with binary y",
    ),
    ProblemInstance(
        name="linear_minlp",
        build_fn=_build_linear_minlp,
        expected_obj=5.0,
        integer_vars=["z"],
        bounds={"x": (0, 5), "y": (0, 5), "z": (0, 1)},
        description="Linear MINLP: 2x+3y+z",
    ),
    ProblemInstance(
        name="circle_minlp",
        build_fn=_build_circle_minlp,
        expected_obj=1.0,
        integer_vars=["z"],
        bounds={"x": (0, 3), "y": (0, 3), "z": (0, 1)},
        description="min x+y+z s.t. x^2+y^2>=1",
    ),
    ProblemInstance(
        name="weighted_sum_minlp",
        build_fn=_build_weighted_sum_minlp,
        expected_obj=2.0,
        integer_vars=["y"],
        bounds={"x1": (0, 5), "x2": (0, 5), "y": (0, 1)},
        description="Weighted sum with binary activation",
    ),
    ProblemInstance(
        name="two_integer_nonlinear",
        build_fn=_build_two_integer_nonlinear,
        expected_obj=0.5,
        integer_vars=["x", "y"],
        bounds={"x": (0, 5), "y": (0, 5), "z": (0, 3)},
        description="Quadratic with 2 integers + 1 continuous",
    ),
]


# ──────────────────────────────────────────────────────────
# Parametrized correctness tests
# ──────────────────────────────────────────────────────────


@pytest.mark.correctness
class TestObjectiveCorrectness:
    """Validate optimal objective values against known solutions.

    This is the core correctness test. Zero incorrect results is required.
    """

    @pytest.mark.parametrize(
        "instance",
        INSTANCES,
        ids=[inst.name for inst in INSTANCES],
    )
    def test_optimal_value(self, instance: ProblemInstance) -> None:
        """Verify solver finds correct optimal objective for each instance."""
        model = instance.build_fn()
        result = model.solve(
            time_limit=120.0,
            gap_tolerance=1e-6,
            max_nodes=50_000,
        )
        assert_optimal_value(
            result,
            instance.expected_obj,
            instance.name,
        )

    @pytest.mark.parametrize(
        "instance",
        [inst for inst in INSTANCES if inst.integer_vars],
        ids=[inst.name for inst in INSTANCES if inst.integer_vars],
    )
    def test_integer_feasibility(self, instance: ProblemInstance) -> None:
        """Verify integer/binary variables are integral in the solution."""
        model = instance.build_fn()
        result = model.solve(
            time_limit=120.0,
            gap_tolerance=1e-6,
            max_nodes=50_000,
        )
        assert result.status in ("optimal", "feasible")
        assert_integer_feasible(result, instance.integer_vars, instance.name)

    @pytest.mark.parametrize(
        "instance",
        INSTANCES,
        ids=[inst.name for inst in INSTANCES],
    )
    def test_bounds_satisfied(self, instance: ProblemInstance) -> None:
        """Verify all variables are within declared bounds."""
        model = instance.build_fn()
        result = model.solve(
            time_limit=120.0,
            gap_tolerance=1e-6,
            max_nodes=50_000,
        )
        assert result.status in ("optimal", "feasible")
        assert_bounds_satisfied(result, instance.bounds, instance.name)


# ──────────────────────────────────────────────────────────
# Continuous NLP-specific tests
# ──────────────────────────────────────────────────────────


@pytest.mark.correctness
class TestContinuousNLP:
    """Focused tests for pure continuous NLP problems."""

    def test_rosenbrock_solution_point(self) -> None:
        """Rosenbrock minimum is at (1, 1)."""
        m = _build_rosenbrock()
        r = m.solve()
        assert abs(float(r.x["x"]) - 1.0) < 1e-3
        assert abs(float(r.x["y"]) - 1.0) < 1e-3

    def test_constrained_quadratic_solution_point(self) -> None:
        """min x^2+y^2 s.t. x+y>=1 has optimal at (0.5, 0.5)."""
        m = _build_constrained_quadratic()
        r = m.solve()
        assert abs(float(r.x["x"]) - 0.5) < 1e-3
        assert abs(float(r.x["y"]) - 0.5) < 1e-3

    def test_nonlinear_equality_solution_point(self) -> None:
        """min x^2+y^2 s.t. x*y=1 has optimal at (1, 1)."""
        m = _build_nonlinear_equality()
        r = m.solve()
        assert abs(float(r.x["x"]) - 1.0) < 1e-3
        assert abs(float(r.x["y"]) - 1.0) < 1e-3

    def test_continuous_no_branch_and_bound(self) -> None:
        """Pure continuous problems should not use B&B (node_count=0)."""
        m = _build_constrained_quadratic()
        r = m.solve()
        assert r.node_count == 0, f"Pure continuous problem used B&B with {r.node_count} nodes"

    def test_three_variable_symmetry(self) -> None:
        """min x^2+y^2+z^2 s.t. x+y+z=3 has optimal at (1,1,1)."""
        m = _build_three_variable_nlp()
        r = m.solve()
        for var_name in ["x", "y", "z"]:
            assert abs(float(r.x[var_name]) - 1.0) < 1e-3, (
                f"{var_name} = {float(r.x[var_name]):.6f}, expected 1.0"
            )


# ──────────────────────────────────────────────────────────
# MINLP-specific tests
# ──────────────────────────────────────────────────────────


@pytest.mark.correctness
class TestMINLP:
    """Focused tests for Mixed-Integer Nonlinear Programs."""

    def test_simple_minlp_binary_at_zero(self) -> None:
        """In simple_minlp, optimal x3 should be 0 (not 1)."""
        m = _build_simple_minlp()
        r = m.solve()
        assert abs(float(r.x["x3"])) < 1e-4, f"x3 should be 0, got {float(r.x['x3']):.6f}"

    def test_knapsack_correct_selection(self) -> None:
        """Binary knapsack should select x2=1, x3=1."""
        m = _build_binary_knapsack()
        r = m.solve()
        # x2=1, x3=1 is optimal (obj=-6)
        assert abs(float(r.x["x2"]) - 1.0) < 1e-4
        assert abs(float(r.x["x3"]) - 1.0) < 1e-4

    def test_integer_variables_rounded(self) -> None:
        """Integer variables should be at integer values."""
        m = _build_quadratic_integer()
        r = m.solve()
        x_val = float(r.x["x"])
        y_val = float(r.x["y"])
        assert abs(x_val - round(x_val)) < 1e-4
        assert abs(y_val - round(y_val)) < 1e-4

    def test_exp_binary_correct_branch(self) -> None:
        """Solver should choose y=0 (exp cheaper than fixed cost 10)."""
        m = _build_exp_binary_minlp()
        r = m.solve()
        assert abs(float(r.x["y"])) < 1e-4, f"y should be 0, got {float(r.x['y']):.6f}"

    def test_bnb_uses_nodes(self) -> None:
        """MINLP problems should use at least 1 B&B node."""
        m = _build_simple_minlp()
        r = m.solve()
        assert r.node_count >= 1, "MINLP should use branch-and-bound"


# ──────────────────────────────────────────────────────────
# Constraint satisfaction tests
# ──────────────────────────────────────────────────────────


@pytest.mark.correctness
class TestConstraintSatisfaction:
    """Verify that solutions satisfy all constraints."""

    def test_simple_minlp_constraints(self) -> None:
        """Verify constraint x1+x2>=1 and x1^2+x2<=3."""
        m = _build_simple_minlp()
        r = m.solve()
        x1 = float(r.x["x1"])
        x2 = float(r.x["x2"])
        assert x1 + x2 >= 1.0 - 1e-4, f"x1+x2={x1 + x2:.6f} < 1"
        assert x1**2 + x2 <= 3.0 + 1e-4, f"x1^2+x2={x1**2 + x2:.6f} > 3"

    def test_equality_constraint_satisfied(self) -> None:
        """Verify x+y=2 is satisfied."""
        m = _build_quadratic_equality()
        r = m.solve()
        x = float(r.x["x"])
        y = float(r.x["y"])
        assert abs(x + y - 2.0) < 1e-4, f"x+y={x + y:.6f}, expected 2.0"

    def test_nonlinear_equality_satisfied(self) -> None:
        """Verify x*y=1 is satisfied."""
        m = _build_nonlinear_equality()
        r = m.solve()
        x = float(r.x["x"])
        y = float(r.x["y"])
        assert abs(x * y - 1.0) < 1e-4, f"x*y={x * y:.6f}, expected 1.0"

    def test_knapsack_weight_satisfied(self) -> None:
        """Verify 2x1+3x2+x3<=4 in knapsack."""
        m = _build_binary_knapsack()
        r = m.solve()
        x1 = float(r.x["x1"])
        x2 = float(r.x["x2"])
        x3 = float(r.x["x3"])
        weight = 2 * x1 + 3 * x2 + x3
        assert weight <= 4.0 + 1e-4, f"Weight={weight:.6f} > 4"

    def test_circle_constraint_satisfied(self) -> None:
        """Verify x^2+y^2>=1 in circle_minlp."""
        m = _build_circle_minlp()
        r = m.solve()
        x = float(r.x["x"])
        y = float(r.x["y"])
        assert x**2 + y**2 >= 1.0 - 1e-4, f"x^2+y^2={x**2 + y**2:.6f} < 1"

    def test_weighted_sum_constraints_satisfied(self) -> None:
        """Verify x1+x2>=2 and x1+y>=1 in weighted_sum_minlp."""
        m = _build_weighted_sum_minlp()
        r = m.solve()
        x1 = float(r.x["x1"])
        x2 = float(r.x["x2"])
        y = float(r.x["y"])
        assert x1 + x2 >= 2.0 - 1e-4, f"x1+x2={x1 + x2:.6f} < 2"
        assert x1 + y >= 1.0 - 1e-4, f"x1+y={x1 + y:.6f} < 1"


# ──────────────────────────────────────────────────────────
# Solver behavior tests
# ──────────────────────────────────────────────────────────


@pytest.mark.correctness
class TestSolverBehavior:
    """Test solver properties: determinism, status, profiling."""

    def test_deterministic_objective(self) -> None:
        """Multiple runs should produce identical objective values."""
        objectives = []
        for _ in range(3):
            m = _build_simple_minlp()
            r = m.solve(deterministic=True)
            objectives.append(r.objective)
        assert objectives[0] == objectives[1] == objectives[2], (
            f"Non-deterministic objectives: {objectives}"
        )

    def test_all_instances_solve_successfully(self) -> None:
        """All instances should return optimal or feasible status."""
        failures = []
        for inst in INSTANCES:
            model = inst.build_fn()
            result = model.solve(time_limit=120.0, max_nodes=50_000)
            if result.status not in ("optimal", "feasible"):
                failures.append(f"{inst.name}: status={result.status}")
        assert len(failures) == 0, "Instances failed to solve:\n" + "\n".join(failures)

    def test_profiling_times_positive(self) -> None:
        """Wall time, JAX time should be positive."""
        m = _build_simple_minlp()
        r = m.solve()
        assert r.wall_time > 0
        assert r.jax_time >= 0

    def test_gap_at_optimality(self) -> None:
        """At optimal status, gap should be small."""
        m = _build_constrained_quadratic()
        r = m.solve()
        if r.status == "optimal" and r.gap is not None:
            assert r.gap <= 1e-3, f"Gap={r.gap} at optimal status"


# ──────────────────────────────────────────────────────────
# Aggregate correctness summary
# ──────────────────────────────────────────────────────────


@pytest.mark.correctness
class TestCorrectnessGate:
    """Phase gate: zero incorrect results across all instances.

    This is the single most important test. If any instance gives
    an incorrect optimal value, the phase gate fails.
    """

    def test_zero_incorrect_results(self) -> None:
        """Run all instances and assert zero incorrect results."""
        incorrect = []
        for inst in INSTANCES:
            model = inst.build_fn()
            result = model.solve(
                time_limit=120.0,
                gap_tolerance=1e-6,
                max_nodes=50_000,
            )
            if result.status not in ("optimal", "feasible"):
                incorrect.append(f"{inst.name}: solve failed with status={result.status}")
                continue

            if result.objective is None:
                incorrect.append(f"{inst.name}: no objective value returned")
                continue

            expected = inst.expected_obj
            actual = result.objective
            tol = ABS_TOL + REL_TOL * abs(expected)
            if abs(actual - expected) > tol:
                incorrect.append(
                    f"{inst.name}: obj={actual:.8f}, expected={expected:.8f}, "
                    f"error={abs(actual - expected):.2e}, tol={tol:.2e}"
                )

        assert len(incorrect) == 0, (
            f"CORRECTNESS GATE FAILED: {len(incorrect)} incorrect result(s):\n"
            + "\n".join(f"  - {msg}" for msg in incorrect)
        )


class TestConstraintFeasibilityCheck:
    """Verify that constraint-violating NLP solutions are rejected as incumbents.

    Regression test for the bug where the Rust B&B tree accepted solutions
    based solely on integer feasibility, ignoring constraint violations.
    """

    def test_prob03_constraint_satisfaction(self):
        """prob03: min 3x + 2y s.t. x*y >= 3.5, 1 <= x,y <= 5, integers.

        The correct optimum is x=1, y=4 with obj=11 (or x=2, y=2, obj=10).
        An infeasible point (1,3) has obj=9 but violates x*y >= 3.5.
        """
        m = dm.Model("prob03_constraint_check")
        x = m.integer("x", lb=1, ub=5)
        y = m.integer("y", lb=1, ub=5)
        m.minimize(3 * x + 2 * y)
        m.subject_to(x * y >= 3.5)

        result = m.solve(time_limit=30.0, max_nodes=10000)

        assert result.status in ("optimal", "feasible"), (
            f"Expected optimal/feasible, got {result.status}"
        )
        assert result.objective is not None

        # The correct optimum is 10.0 (x=2, y=2)
        assert result.objective >= 10.0 - 1e-4, (
            f"Objective {result.objective} is below the true optimum 10.0 "
            f"— constraint-violating solution accepted as incumbent"
        )

        # Verify constraint satisfaction at solution
        if result.x is not None:
            x_val = float(result.x["x"])
            y_val = float(result.x["y"])
            assert x_val * y_val >= 3.5 - 1e-6, (
                f"Constraint x*y >= 3.5 violated: {x_val}*{y_val} = {x_val * y_val}"
            )

    def test_feasibility_check_helper(self):
        """Unit test for _check_constraint_feasibility."""
        import numpy as np
        from discopt._jax.nlp_evaluator import NLPEvaluator
        from discopt.solver import _check_constraint_feasibility

        # The modeling API normalizes x + y >= 3 to 3 - x - y <= 0,
        # so _infer_constraint_bounds yields cl=-1e20, cu=0.
        m = dm.Model("feas_check_test")
        x = m.continuous("x", lb=0, ub=5)
        y = m.continuous("y", lb=0, ub=5)
        m.minimize(x + y)
        m.subject_to(x + y >= 3.0)

        ev = NLPEvaluator(m)
        cl = [-1e20]  # <= constraint (normalized from >=)
        cu = [0.0]

        # Feasible point: x=2, y=2 => 3-4 = -1 <= 0 ✓
        assert _check_constraint_feasibility(ev, np.array([2.0, 2.0]), cl, cu)

        # Infeasible point: x=0.5, y=0.5 => 3-1 = 2 > 0 ✗
        assert not _check_constraint_feasibility(ev, np.array([0.5, 0.5]), cl, cu)
