"""MINLPTests.jl standardized NLP/MINLP correctness tests.

Translated from https://github.com/jump-dev/MINLPTests.jl (MIT license).
Each build function produces a discopt Model corresponding to a JuMP test case.

Problem naming convention: {category}_{file_id}
  Categories: nlp_cvx, nlp, nlp_mi
  File IDs match MINLPTests.jl filenames (e.g. 001_010, 101_010, 501_010)

Variable bounds note: JuMP @variable(model, x) (unbounded) is translated using
  discopt's default continuous bounds (lb=-9.999e19, ub=9.999e19). Variables
  with explicit JuMP bounds are translated to matching discopt bounds.

Failure tracking: python/tests/data/known_failures.toml
  Only non-convergence (time_limit, numerical_error, no_convergence) may be
  tracked there. Wrong objectives are never xfailed — they are blocker bugs.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import discopt.modeling as dm
import pytest
from discopt.modeling.core import Model

# ── Failure tracking ─────────────────────────────────────────────────────────

_FAILURES_PATH = Path(__file__).parent / "data" / "known_failures.toml"
_KNOWN_FAILURES: dict[str, dict] = {}

try:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib  # type: ignore[no-redef]

    if _FAILURES_PATH.exists():
        with open(_FAILURES_PATH, "rb") as _f:
            _KNOWN_FAILURES = tomllib.load(_f).get("minlptests", {})
except ImportError:
    pass  # graceful degradation — known_failures not loaded


def _xfail_if_known(problem_id: str, mode: str = "primary") -> None:
    """Apply pytest.xfail if this problem/mode is in known_failures.toml."""
    key = f"{problem_id}/{mode}"
    entry = _KNOWN_FAILURES.get(key)
    if entry is None:
        return
    category = entry["category"]
    assert category != "wrong_objective", (
        f"BUG: {key} tracked as wrong_objective — wrong answers are blocker bugs, "
        f"not expected failures. Fix the solver (issue #{entry.get('issue', '?')})."
    )
    pytest.xfail(
        reason=(f"{category} (issue #{entry.get('issue', '?')}): {entry.get('note', '')}"),
    )


# ── Result assertion helpers ──────────────────────────────────────────────────

_OBJ_TOL = 1e-6  # matches MINLPTests.jl default


def assert_optimal(result, expected_obj: float, name: str) -> None:
    """Assert solver found an optimal solution within MINLPTests tolerance."""
    assert result.status in ("optimal", "feasible"), (
        f"[{name}] Expected optimal/feasible, got status={result.status!r}"
    )
    assert result.objective is not None, f"[{name}] No objective returned"
    tol = _OBJ_TOL + 1e-4 * abs(expected_obj)
    assert abs(result.objective - expected_obj) <= tol, (
        f"[{name}] obj={result.objective:.10g} expected={expected_obj:.10g} "
        f"diff={abs(result.objective - expected_obj):.2e} tol={tol:.2e}"
    )


def assert_infeasible(result, name: str) -> None:
    """Assert solver detected infeasibility."""
    assert result.status == "infeasible", (
        f"[{name}] Expected infeasible, got status={result.status!r}"
    )


# ── Instance descriptor ───────────────────────────────────────────────────────


@dataclass
class MINLPTestInstance:
    problem_id: str
    build_fn: Callable[[], Model]
    expected_obj: float
    expected_status: str = "optimal"
    is_convex: bool = False
    has_integers: bool = False
    tags: list[str] = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════════
# NLP-CVX: Convex NLP problems (nlp-cvx directory, 53 problems)
# ═════════════════════════════════════════════════════════════════════════════

# ── 001-002: Basic LP/QP (4 problems) ─────────────────────────────────────


def _build_nlp_cvx_001_010() -> Model:
    """Min x; 5 linear constraints. Opt: -2.0430107680954848."""
    m = dm.Model("nlp_cvx_001_010")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize(x)
    m.subject_to(x + y <= 5)
    m.subject_to(2 * x - y <= 3)
    m.subject_to(3 * x + 9 * y >= -10)
    m.subject_to(10 * x - y >= -20)
    m.subject_to(-x + 2 * y <= 8)
    return m


def _build_nlp_cvx_001_011() -> Model:
    """Min (x-1)^2 + (y-2)^2; same 5 linear constraints (non-binding). Opt: 0."""
    m = dm.Model("nlp_cvx_001_011")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize((x - 1) ** 2 + (y - 2) ** 2)
    m.subject_to(x + y <= 5)
    m.subject_to(2 * x - y <= 3)
    m.subject_to(3 * x + 9 * y >= -10)
    m.subject_to(10 * x - y >= -20)
    m.subject_to(-x + 2 * y <= 8)
    return m


def _build_nlp_cvx_002_010() -> Model:
    """Min x+y; 6 linear constraints. Opt: 3.9655172067026196."""
    m = dm.Model("nlp_cvx_002_010")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize(x + y)
    m.subject_to(x - 3 * y <= 3)
    m.subject_to(x - 5 * y <= 0)
    m.subject_to(3 * x + 5 * y >= 15)
    m.subject_to(7 * x + 2 * y >= 20)
    m.subject_to(9 * x + y >= 20)
    m.subject_to(3 * x + 7 * y >= 17)
    return m


def _build_nlp_cvx_002_011() -> Model:
    """Min (x-3)^2 + (y-2)^2; same 6 linear constraints (non-binding). Opt: 0."""
    m = dm.Model("nlp_cvx_002_011")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize((x - 3) ** 2 + (y - 2) ** 2)
    m.subject_to(x - 3 * y <= 3)
    m.subject_to(x - 5 * y <= 0)
    m.subject_to(3 * x + 5 * y >= 15)
    m.subject_to(7 * x + 2 * y >= 20)
    m.subject_to(9 * x + y >= 20)
    m.subject_to(3 * x + 7 * y >= 17)
    return m


# ── 101-104: Unit disk constraints (10 problems) ──────────────────────────


def _build_nlp_cvx_101_010() -> Model:
    """Min -x-y; x^2+y^2<=1; x,y in [-2,2]. Opt: -sqrt(2)."""
    m = dm.Model("nlp_cvx_101_010")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.minimize(-x - y)
    m.subject_to(x**2 + y**2 <= 1.0)
    return m


def _build_nlp_cvx_101_011() -> Model:
    """Min -x; x^2+y^2<=1; x,y in [-2,2]. Opt: -1."""
    m = dm.Model("nlp_cvx_101_011")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.minimize(-x)
    m.subject_to(x**2 + y**2 <= 1.0)
    return m


def _build_nlp_cvx_101_012() -> Model:
    """Max x; x^2+y^2<=1; x,y in [-2,2]. Opt: 1."""
    m = dm.Model("nlp_cvx_101_012")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.maximize(x)
    m.subject_to(x**2 + y**2 <= 1.0)
    return m


def _build_nlp_cvx_102_010() -> Model:
    """Min -x; x^2+y^2<=1; x+y>=1.2. Opt: -0.974165743715913."""
    m = dm.Model("nlp_cvx_102_010")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize(-x)
    m.subject_to(x**2 + y**2 <= 1.0)
    m.subject_to(x + y >= 1.2)
    return m


def _build_nlp_cvx_102_011() -> Model:
    """Min x+y; x^2+y^2<=1; x+y>=1.2. Opt: 1.2 (multiple optima)."""
    m = dm.Model("nlp_cvx_102_011")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize(x + y)
    m.subject_to(x**2 + y**2 <= 1.0)
    m.subject_to(x + y >= 1.2)
    return m


def _build_nlp_cvx_102_012() -> Model:
    """Max x+y; x^2+y^2<=1; x+y>=1.2. Opt: sqrt(2)."""
    m = dm.Model("nlp_cvx_102_012")
    x = m.continuous("x")
    y = m.continuous("y")
    m.maximize(x + y)
    m.subject_to(x**2 + y**2 <= 1.0)
    m.subject_to(x + y >= 1.2)
    return m


def _build_nlp_cvx_102_013() -> Model:
    """Min x^2+y^2; x^2+y^2<=1; x+y>=1.2. Opt: 0.72 at [0.6,0.6]."""
    m = dm.Model("nlp_cvx_102_013")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize(x**2 + y**2)
    m.subject_to(x**2 + y**2 <= 1.0)
    m.subject_to(x + y >= 1.2)
    return m


def _build_nlp_cvx_102_014() -> Model:
    """Min (x-0.65)^2+(y-0.65)^2; x^2+y^2<=1; x+y>=1.2. Opt: 0 at [0.65,0.65]."""
    m = dm.Model("nlp_cvx_102_014")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize((x - 0.65) ** 2 + (y - 0.65) ** 2)
    m.subject_to(x**2 + y**2 <= 1.0)
    m.subject_to(x + y >= 1.2)
    return m


def _build_nlp_cvx_103_010() -> Model:
    """Min y; x^2<=y; y<=-x^2+1. Opt: 0 at [0,0]."""
    m = dm.Model("nlp_cvx_103_010")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize(y)
    m.subject_to(x**2 <= y)
    m.subject_to(y <= -(x**2) + 1)
    return m


def _build_nlp_cvx_103_011() -> Model:
    """Min -y; x^2<=y; y<=-x^2+1. Opt: -1 at [0,1]."""
    m = dm.Model("nlp_cvx_103_011")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize(-y)
    m.subject_to(x**2 <= y)
    m.subject_to(y <= -(x**2) + 1)
    return m


def _build_nlp_cvx_103_012() -> Model:
    """Min -x-y; x^2<=y; y<=-x^2+1. Opt: -5/4 at [1/2, 3/4]."""
    m = dm.Model("nlp_cvx_103_012")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize(-x - y)
    m.subject_to(x**2 <= y)
    m.subject_to(y <= -(x**2) + 1)
    return m


def _build_nlp_cvx_103_013() -> Model:
    """Min x+y; x^2<=y; y<=-x^2+1. Opt: -1/4 at [-1/2, 1/4]."""
    m = dm.Model("nlp_cvx_103_013")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize(x + y)
    m.subject_to(x**2 <= y)
    m.subject_to(y <= -(x**2) + 1)
    return m


def _build_nlp_cvx_103_014() -> Model:
    """Min -x; x^2<=y; y<=-x^2+1. Opt: -1/sqrt(2) at [1/sqrt(2), 1/2]."""
    m = dm.Model("nlp_cvx_103_014")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize(-x)
    m.subject_to(x**2 <= y)
    m.subject_to(y <= -(x**2) + 1)
    return m


def _build_nlp_cvx_104_010() -> Model:
    """Min -x; x^2<=y; y<=-x^2+1; x^2+y^2<=1.8. Opt: -1/sqrt(2) (redundant 3rd)."""
    m = dm.Model("nlp_cvx_104_010")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize(-x)
    m.subject_to(x**2 <= y)
    m.subject_to(y <= -(x**2) + 1)
    m.subject_to(x**2 + y**2 <= 1.8)
    return m


# ── 105: Exponential/log constraints (4 problems) ─────────────────────────


def _build_nlp_cvx_105_010() -> Model:
    """Min -x-y; exp(x-2)-0.5<=y; log(x)+0.5>=y. x>0. Opt: -4.176004405036646."""
    m = dm.Model("nlp_cvx_105_010")
    x = m.continuous("x", lb=1e-5)
    y = m.continuous("y")
    m.minimize(-x - y)
    m.subject_to(dm.exp(x - 2.0) - 0.5 <= y)
    m.subject_to(dm.log(x) + 0.5 >= y)
    return m


def _build_nlp_cvx_105_011() -> Model:
    """Min x+y; same constraints as 105_010. Opt: 0.16878271368156372."""
    m = dm.Model("nlp_cvx_105_011")
    x = m.continuous("x", lb=1e-5)
    y = m.continuous("y")
    m.minimize(x + y)
    m.subject_to(dm.exp(x - 2.0) - 0.5 <= y)
    m.subject_to(dm.log(x) + 0.5 >= y)
    return m


def _build_nlp_cvx_105_012() -> Model:
    """Min x-y; same constraints as 105_010. Opt: 1/2 at [1, 1/2]."""
    m = dm.Model("nlp_cvx_105_012")
    x = m.continuous("x", lb=1e-5)
    y = m.continuous("y")
    m.minimize(x - y)
    m.subject_to(dm.exp(x - 2.0) - 0.5 <= y)
    m.subject_to(dm.log(x) + 0.5 >= y)
    return m


def _build_nlp_cvx_105_013() -> Model:
    """Min -x+y; same constraints as 105_010. Opt: -3/2 at [2, 1/2]."""
    m = dm.Model("nlp_cvx_105_013")
    x = m.continuous("x", lb=1e-5)
    y = m.continuous("y")
    m.minimize(-x + y)
    m.subject_to(dm.exp(x - 2.0) - 0.5 <= y)
    m.subject_to(dm.log(x) + 0.5 >= y)
    return m


# ── 106: Trigonometric constraints (2 problems) ───────────────────────────


def _build_nlp_cvx_106_010() -> Model:
    """Min -x-y; trig constraints; x in [-3,3], y in [-1,1]. Opt: -1.857."""
    m = dm.Model("nlp_cvx_106_010")
    x = m.continuous("x", lb=-3.0, ub=3.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    m.minimize(-x - y)
    m.subject_to(dm.sin(-x - 1.0) + x / 2 + 0.5 <= y)
    m.subject_to(dm.cos(x - 0.5) + x / 4 - 0.5 >= y)
    return m


def _build_nlp_cvx_106_011() -> Model:
    """Min x+y; trig constraints; x in [-3,3], y in [-1,1]. Opt: -0.787."""
    m = dm.Model("nlp_cvx_106_011")
    x = m.continuous("x", lb=-3.0, ub=3.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    m.minimize(x + y)
    m.subject_to(dm.sin(-x - 1.0) + x / 2 + 0.5 <= y)
    m.subject_to(dm.cos(x - 0.5) + x / 4 - 0.5 >= y)
    return m


# ── 107: Distance minimization on unit disk (3 problems) ─────────────────


def _build_nlp_cvx_107_010() -> Model:
    """Min (x-0.5)^2+(y-0.5)^2; x^2+y^2<=1. Opt: 0 (feasible interior)."""
    m = dm.Model("nlp_cvx_107_010")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize((x - 0.5) ** 2 + (y - 0.5) ** 2)
    m.subject_to(x**2 + y**2 <= 1.0)
    return m


def _build_nlp_cvx_107_011() -> Model:
    """Min (x-1)^2+(y-1)^2; x^2+y^2<=1. Opt: 0.17157... at [1/sqrt(2),1/sqrt(2)]."""
    m = dm.Model("nlp_cvx_107_011")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize((x - 1) ** 2 + (y - 1) ** 2)
    m.subject_to(x**2 + y**2 <= 1.0)
    return m


def _build_nlp_cvx_107_012() -> Model:
    """Same as 107_011 with different starting point (start=1.5,0.5). Opt: 0.17157."""
    m = dm.Model("nlp_cvx_107_012")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize((x - 1) ** 2 + (y - 1) ** 2)
    m.subject_to(x**2 + y**2 <= 1.0)
    return m


# ── 108: Intersection of nonlinear constraints (4 problems) ──────────────


def _build_nlp_cvx_108_010() -> Model:
    """Min (x-1)^2+(y-0.75)^2; 2x^2-4xy-4x+4<=y; y^2<=-x+2; x,y>=0. Opt: 0."""
    m = dm.Model("nlp_cvx_108_010")
    x = m.continuous("x", lb=0.0)
    y = m.continuous("y", lb=0.0)
    m.minimize((x - 1.0) ** 2 + (y - 0.75) ** 2)
    m.subject_to(2 * x**2 - 4 * x * y - 4 * x + 4 <= y)
    m.subject_to(y**2 <= -x + 2)
    return m


def _build_nlp_cvx_108_011() -> Model:
    """Min (x-3)^2+y^2; same constraints as 108_010. Opt: 1.5240966871955863."""
    m = dm.Model("nlp_cvx_108_011")
    x = m.continuous("x", lb=0.0)
    y = m.continuous("y", lb=0.0)
    m.minimize((x - 3.0) ** 2 + y**2)
    m.subject_to(2 * x**2 - 4 * x * y - 4 * x + 4 <= y)
    m.subject_to(y**2 <= -x + 2)
    return m


def _build_nlp_cvx_108_012() -> Model:
    """Min x^2+(y-2)^2; same constraints as 108_010. Opt: 0.5927195187027438."""
    m = dm.Model("nlp_cvx_108_012")
    x = m.continuous("x", lb=0.0)
    y = m.continuous("y", lb=0.0)
    m.minimize(x**2 + (y - 2) ** 2)
    m.subject_to(2 * x**2 - 4 * x * y - 4 * x + 4 <= y)
    m.subject_to(y**2 <= -x + 2)
    return m


def _build_nlp_cvx_108_013() -> Model:
    """Min x^2+y^2; same constraints as 108_010. Opt: 0.8112507770394088."""
    m = dm.Model("nlp_cvx_108_013")
    x = m.continuous("x", lb=0.0)
    y = m.continuous("y", lb=0.0)
    m.minimize(x**2 + y**2)
    m.subject_to(2 * x**2 - 4 * x * y - 4 * x + 4 <= y)
    m.subject_to(y**2 <= -x + 2)
    return m


# ── 109: Logarithmic objectives (3 problems) ──────────────────────────────


def _build_nlp_cvx_109_010() -> Model:
    """Max log(x); (y-2)^2<=-x+2; x,y>=1e-5. Opt: log(2) at [2,2]."""
    m = dm.Model("nlp_cvx_109_010")
    x = m.continuous("x", lb=1e-5)
    y = m.continuous("y", lb=1e-5)
    m.maximize(dm.log(x))
    m.subject_to((y - 2) ** 2 <= -x + 2)
    return m


def _build_nlp_cvx_109_011() -> Model:
    """Max log(x)+log(y); (y-2)^2<=-x+2; x,y>=1e-5. Opt: 1.4853479762665618."""
    m = dm.Model("nlp_cvx_109_011")
    x = m.continuous("x", lb=1e-5)
    y = m.continuous("y", lb=1e-5)
    m.maximize(dm.log(x) + dm.log(y))
    m.subject_to((y - 2) ** 2 <= -x + 2)
    return m


def _build_nlp_cvx_109_012() -> Model:
    """Max log(x+y); (y-2)^2<=-x+2; x,y>=1e-5. Opt: log(17/4)."""
    m = dm.Model("nlp_cvx_109_012")
    x = m.continuous("x", lb=1e-5)
    y = m.continuous("y", lb=1e-5)
    m.maximize(dm.log(x + y))
    m.subject_to((y - 2) ** 2 <= -x + 2)
    return m


# ── 110: Exponential objectives on unit disk (3 problems) ─────────────────


def _build_nlp_cvx_110_010() -> Model:
    """Min exp(x); x^2+y^2<=1. Opt: exp(-1) at [-1,0]."""
    m = dm.Model("nlp_cvx_110_010")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize(dm.exp(x))
    m.subject_to(x**2 + y**2 <= 1.0)
    return m


def _build_nlp_cvx_110_011() -> Model:
    """Min exp(x)+exp(y); x^2+y^2<=1. Opt: 2*exp(-1/sqrt(2))."""
    m = dm.Model("nlp_cvx_110_011")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize(dm.exp(x) + dm.exp(y))
    m.subject_to(x**2 + y**2 <= 1.0)
    return m


def _build_nlp_cvx_110_012() -> Model:
    """Min exp(x+y); x^2+y^2<=1. Opt: exp(-sqrt(2))."""
    m = dm.Model("nlp_cvx_110_012")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize(dm.exp(x + y))
    m.subject_to(x**2 + y**2 <= 1.0)
    return m


# ── 201: 3D sphere constraints (2 problems) ───────────────────────────────


def _build_nlp_cvx_201_010() -> Model:
    """Min -(x+y+z); x^2+y^2+z^2<=1. Opt: -sqrt(3) at [1/sqrt(3), ...]."""
    m = dm.Model("nlp_cvx_201_010")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z")
    m.minimize(-x - y - z)
    m.subject_to(x**2 + y**2 + z**2 <= 1.0)
    return m


def _build_nlp_cvx_201_011() -> Model:
    """Min -x; x^2+y^2+z^2<=1. Opt: -1 at [1,0,0]."""
    m = dm.Model("nlp_cvx_201_011")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z")
    m.minimize(-x)
    m.subject_to(x**2 + y**2 + z**2 <= 1.0)
    return m


# ── 202: Intersecting paraboloid constraints (5 problems) ─────────────────


def _build_nlp_cvx_202_010() -> Model:
    """Min -z; x^2+y^2<=z; x^2+y^2<=-z+1. Opt: -1 at [0,0,1]."""
    m = dm.Model("nlp_cvx_202_010")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z")
    m.minimize(-z)
    m.subject_to(x**2 + y**2 <= z)
    m.subject_to(x**2 + y**2 <= -z + 1)
    return m


def _build_nlp_cvx_202_011() -> Model:
    """Min z; x^2+y^2<=z; x^2+y^2<=-z+1. Opt: 0 at [0,0,0]."""
    m = dm.Model("nlp_cvx_202_011")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z")
    m.minimize(z)
    m.subject_to(x**2 + y**2 <= z)
    m.subject_to(x**2 + y**2 <= -z + 1)
    return m


def _build_nlp_cvx_202_012() -> Model:
    """Min -(x+y+2z); same constraints. Opt: -9/4 at [1/4,1/4,7/8]."""
    m = dm.Model("nlp_cvx_202_012")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z")
    m.minimize(-x - y - 2 * z)
    m.subject_to(x**2 + y**2 <= z)
    m.subject_to(x**2 + y**2 <= -z + 1)
    return m


def _build_nlp_cvx_202_013() -> Model:
    """Min x+y+2z; same constraints. Opt: -1/4 at [-1/4,-1/4,1/8]."""
    m = dm.Model("nlp_cvx_202_013")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z")
    m.minimize(x + y + 2 * z)
    m.subject_to(x**2 + y**2 <= z)
    m.subject_to(x**2 + y**2 <= -z + 1)
    return m


def _build_nlp_cvx_202_014() -> Model:
    """Min x+y; same constraints. Opt: -1 at [-1/2,-1/2,1/2]."""
    m = dm.Model("nlp_cvx_202_014")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z")
    m.minimize(x + y)
    m.subject_to(x**2 + y**2 <= z)
    m.subject_to(x**2 + y**2 <= -z + 1)
    return m


# ── 203: Second-order cone + quadratic (1 problem) ────────────────────────


def _build_nlp_cvx_203_010() -> Model:
    """Min x+y; sqrt(x^2+y^2)<=z-0.25; x^2+y^2<=-z+1. Opt: -1/sqrt(2)."""
    m = dm.Model("nlp_cvx_203_010")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z")
    m.minimize(x + y)
    m.subject_to(dm.sqrt(x**2 + y**2) <= z - 0.25)
    m.subject_to(x**2 + y**2 <= -z + 1)
    return m


# ── 204: Rotated second-order cone (1 problem) ────────────────────────────


def _build_nlp_cvx_204_010() -> Model:
    """Min -x-y; x^2/z<=y; x^2+y^2<=-z+1; z>=0. Rotated SOC. Opt: -1.2071."""
    m = dm.Model("nlp_cvx_204_010")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z", lb=1e-8)  # z>0 required for x^2/z
    m.minimize(-y - x)
    m.subject_to(x**2 / z <= y)
    m.subject_to(x**2 + y**2 <= -z + 1)
    return m


# ── 205: Exponential cone (1 problem) ─────────────────────────────────────


def _build_nlp_cvx_205_010() -> Model:
    """Max y; y*exp(x/y)<=z; y*exp(-x/y)<=z; x^2+y^2<=-z+5. Opt: 1.7912878."""
    m = dm.Model("nlp_cvx_205_010")
    x = m.continuous("x")
    y = m.continuous("y", lb=1e-8)  # y>0 for y*exp(x/y)
    z = m.continuous("z")
    m.maximize(y)
    m.subject_to(y * dm.exp(x / y) <= z)
    m.subject_to(y * dm.exp(-x / y) <= z)
    m.subject_to(x**2 + y**2 <= -z + 5)
    return m


# ── 206: Power cone (1 problem) ───────────────────────────────────────────


def _build_nlp_cvx_206_010() -> Model:
    """Max 2x+y+z; z<=x^0.3*y^0.7; x<=z^0.7*y^0.3; x^2+y^2<=z+1. Opt: 4."""
    m = dm.Model("nlp_cvx_206_010")
    x = m.continuous("x", lb=1e-8)
    y = m.continuous("y", lb=1e-8)
    z = m.continuous("z", lb=1e-8)
    m.maximize(2 * x + y + z)
    m.subject_to(z <= x**0.3 * y**0.7)
    m.subject_to(x <= z**0.7 * y**0.3)
    m.subject_to(x**2 + y**2 <= z + 1)
    return m


# ── 210: 3D sphere distance problems (3 problems) ─────────────────────────


def _build_nlp_cvx_210_010() -> Model:
    """Min (x-0.5)^2+(y-0.5)^2+(z-0.5)^2; x^2+y^2+z^2<=1. Opt: 0."""
    m = dm.Model("nlp_cvx_210_010")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z")
    m.minimize((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2)
    m.subject_to(x**2 + y**2 + z**2 <= 1.0)
    return m


def _build_nlp_cvx_210_011() -> Model:
    """Min (x-1)^2+(y-1)^2+(z-1)^2; x^2+y^2+z^2<=1. Opt: 0.5359 at [1/sqrt(3),...]."""
    m = dm.Model("nlp_cvx_210_011")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z")
    m.minimize((x - 1) ** 2 + (y - 1) ** 2 + (z - 1) ** 2)
    m.subject_to(x**2 + y**2 + z**2 <= 1.0)
    return m


def _build_nlp_cvx_210_012() -> Model:
    """Same as 210_011 (different starting point variant). Opt: 0.5359."""
    m = dm.Model("nlp_cvx_210_012")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z")
    m.minimize((x - 1) ** 2 + (y - 1) ** 2 + (z - 1) ** 2)
    m.subject_to(x**2 + y**2 + z**2 <= 1.0)
    return m


# ── 501: N-dimensional sphere (factory functions) ─────────────────────────


def _make_nlp_cvx_501_010(n: int) -> Callable[[], Model]:
    """Factory: min -sum(x_i); sum(x_i^2)<=1; n-dimensional. Opt: -sqrt(n)."""

    def _build() -> Model:
        m = dm.Model(f"nlp_cvx_501_010_{n}d")
        vs = [m.continuous(f"x{i}") for i in range(n)]
        m.minimize(dm.sum([-v for v in vs]))
        m.subject_to(dm.sum([v**2 for v in vs]) <= 1.0)
        return m

    return _build


def _make_nlp_cvx_501_011(n: int) -> Callable[[], Model]:
    """Factory: min -sum(x_i); sqrt(sum(x_i^2))<=1; n-dimensional. Opt: -sqrt(n)."""

    def _build() -> Model:
        m = dm.Model(f"nlp_cvx_501_011_{n}d")
        vs = [m.continuous(f"x{i}") for i in range(n)]
        m.minimize(dm.sum([-v for v in vs]))
        m.subject_to(dm.sqrt(dm.sum([v**2 for v in vs])) <= 1.0)
        return m

    return _build


# ── NLP-CVX instances list ─────────────────────────────────────────────────

NLP_CVX_INSTANCES = [
    # ── 001-002: LP/QP ──────────────────────────────────────────────────────
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_001_010", _build_nlp_cvx_001_010, -2.0430107680954848, is_convex=True
        ),
        id="nlp_cvx_001_010",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_001_011", _build_nlp_cvx_001_011, 0.0, is_convex=True),
        id="nlp_cvx_001_011",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_002_010", _build_nlp_cvx_002_010, 3.9655172067026196, is_convex=True
        ),
        id="nlp_cvx_002_010",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_002_011", _build_nlp_cvx_002_011, 0.0, is_convex=True),
        id="nlp_cvx_002_011",
    ),
    # ── 101-104: Unit disk ──────────────────────────────────────────────────
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_101_010",
            _build_nlp_cvx_101_010,
            -math.sqrt(2),
            is_convex=True,
            tags=["smoke"],
        ),
        marks=[pytest.mark.smoke],
        id="nlp_cvx_101_010",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_101_011", _build_nlp_cvx_101_011, -1.0, is_convex=True),
        id="nlp_cvx_101_011",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_101_012", _build_nlp_cvx_101_012, 1.0, is_convex=True),
        id="nlp_cvx_101_012",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_102_010", _build_nlp_cvx_102_010, -0.974165743715913, is_convex=True
        ),
        id="nlp_cvx_102_010",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_102_011", _build_nlp_cvx_102_011, 1.2, is_convex=True),
        id="nlp_cvx_102_011",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_102_012", _build_nlp_cvx_102_012, math.sqrt(2), is_convex=True),
        id="nlp_cvx_102_012",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_102_013", _build_nlp_cvx_102_013, 0.72, is_convex=True),
        id="nlp_cvx_102_013",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_102_014", _build_nlp_cvx_102_014, 0.0, is_convex=True),
        id="nlp_cvx_102_014",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_103_010", _build_nlp_cvx_103_010, 0.0, is_convex=True),
        id="nlp_cvx_103_010",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_103_011", _build_nlp_cvx_103_011, -1.0, is_convex=True),
        id="nlp_cvx_103_011",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_103_012", _build_nlp_cvx_103_012, -5 / 4, is_convex=True),
        id="nlp_cvx_103_012",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_103_013", _build_nlp_cvx_103_013, -1 / 4, is_convex=True),
        id="nlp_cvx_103_013",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_103_014", _build_nlp_cvx_103_014, -1 / math.sqrt(2), is_convex=True
        ),
        id="nlp_cvx_103_014",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_104_010", _build_nlp_cvx_104_010, -1 / math.sqrt(2), is_convex=True
        ),
        id="nlp_cvx_104_010",
    ),
    # ── 105: Exp/log ────────────────────────────────────────────────────────
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_105_010", _build_nlp_cvx_105_010, -4.176004405036646, is_convex=True
        ),
        id="nlp_cvx_105_010",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_105_011", _build_nlp_cvx_105_011, 0.16878271368156372, is_convex=True
        ),
        id="nlp_cvx_105_011",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_105_012", _build_nlp_cvx_105_012, 0.5, is_convex=True),
        id="nlp_cvx_105_012",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_105_013", _build_nlp_cvx_105_013, -1.5, is_convex=True),
        id="nlp_cvx_105_013",
    ),
    # ── 106: Trig ───────────────────────────────────────────────────────────
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_106_010", _build_nlp_cvx_106_010, -1.8572155128552428, is_convex=True
        ),
        id="nlp_cvx_106_010",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_106_011", _build_nlp_cvx_106_011, -0.7868226265935826, is_convex=True
        ),
        id="nlp_cvx_106_011",
    ),
    # ── 107: Distance ───────────────────────────────────────────────────────
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_107_010",
            _build_nlp_cvx_107_010,
            0.0,
            is_convex=True,
            tags=["smoke"],
        ),
        marks=[pytest.mark.smoke],
        id="nlp_cvx_107_010",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_107_011", _build_nlp_cvx_107_011, 0.17157287363083387, is_convex=True
        ),
        id="nlp_cvx_107_011",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_107_012", _build_nlp_cvx_107_012, 0.17157287363083387, is_convex=True
        ),
        id="nlp_cvx_107_012",
    ),
    # ── 108: Nonlinear constraint intersection ───────────────────────────────
    pytest.param(
        MINLPTestInstance("nlp_cvx_108_010", _build_nlp_cvx_108_010, 0.0, is_convex=True),
        id="nlp_cvx_108_010",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_108_011", _build_nlp_cvx_108_011, 1.5240966871955863, is_convex=True
        ),
        id="nlp_cvx_108_011",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_108_012", _build_nlp_cvx_108_012, 0.5927195187027438, is_convex=True
        ),
        id="nlp_cvx_108_012",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_108_013", _build_nlp_cvx_108_013, 0.8112507770394088, is_convex=True
        ),
        id="nlp_cvx_108_013",
    ),
    # ── 109: Log objectives ─────────────────────────────────────────────────
    pytest.param(
        MINLPTestInstance("nlp_cvx_109_010", _build_nlp_cvx_109_010, math.log(2), is_convex=True),
        id="nlp_cvx_109_010",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_109_011", _build_nlp_cvx_109_011, 1.4853479762665618, is_convex=True
        ),
        id="nlp_cvx_109_011",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_109_012",
            _build_nlp_cvx_109_012,
            math.log(7 / 4 + 5 / 2),
            is_convex=True,
        ),
        id="nlp_cvx_109_012",
    ),
    # ── 110: Exp objectives ─────────────────────────────────────────────────
    pytest.param(
        MINLPTestInstance("nlp_cvx_110_010", _build_nlp_cvx_110_010, math.exp(-1), is_convex=True),
        id="nlp_cvx_110_010",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_110_011",
            _build_nlp_cvx_110_011,
            2 * math.exp(-1 / math.sqrt(2)),
            is_convex=True,
        ),
        id="nlp_cvx_110_011",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_110_012",
            _build_nlp_cvx_110_012,
            math.exp(-2 / math.sqrt(2)),
            is_convex=True,
        ),
        id="nlp_cvx_110_012",
    ),
    # ── 201-210: 3D problems ────────────────────────────────────────────────
    pytest.param(
        MINLPTestInstance("nlp_cvx_201_010", _build_nlp_cvx_201_010, -math.sqrt(3), is_convex=True),
        id="nlp_cvx_201_010",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_201_011", _build_nlp_cvx_201_011, -1.0, is_convex=True),
        id="nlp_cvx_201_011",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_202_010", _build_nlp_cvx_202_010, -1.0, is_convex=True),
        id="nlp_cvx_202_010",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_202_011", _build_nlp_cvx_202_011, 0.0, is_convex=True),
        id="nlp_cvx_202_011",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_202_012", _build_nlp_cvx_202_012, -9 / 4, is_convex=True),
        id="nlp_cvx_202_012",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_202_013", _build_nlp_cvx_202_013, -1 / 4, is_convex=True),
        id="nlp_cvx_202_013",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_202_014", _build_nlp_cvx_202_014, -1.0, is_convex=True),
        id="nlp_cvx_202_014",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_203_010", _build_nlp_cvx_203_010, -1 / math.sqrt(2), is_convex=True
        ),
        id="nlp_cvx_203_010",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_204_010", _build_nlp_cvx_204_010, -1.2071067837918394, is_convex=True
        ),
        id="nlp_cvx_204_010",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_205_010", _build_nlp_cvx_205_010, 1.7912878443121907, is_convex=True
        ),
        id="nlp_cvx_205_010",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_206_010", _build_nlp_cvx_206_010, 4.0, is_convex=True),
        id="nlp_cvx_206_010",
    ),
    pytest.param(
        MINLPTestInstance("nlp_cvx_210_010", _build_nlp_cvx_210_010, 0.0, is_convex=True),
        id="nlp_cvx_210_010",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_210_011", _build_nlp_cvx_210_011, 0.535898380052066, is_convex=True
        ),
        id="nlp_cvx_210_011",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_cvx_210_012", _build_nlp_cvx_210_012, 0.535898380052066, is_convex=True
        ),
        id="nlp_cvx_210_012",
    ),
    # ── 501: N-dimensional sphere ────────────────────────────────────────────
    *[
        pytest.param(
            MINLPTestInstance(
                f"nlp_cvx_501_010_{n}d",
                _make_nlp_cvx_501_010(n),
                -math.sqrt(n),
                is_convex=True,
            ),
            id=f"nlp_cvx_501_010_{n}d",
        )
        for n in range(1, 21)
    ],
    *[
        pytest.param(
            MINLPTestInstance(
                f"nlp_cvx_501_011_{n}d",
                _make_nlp_cvx_501_011(n),
                -math.sqrt(n),
                is_convex=True,
            ),
            id=f"nlp_cvx_501_011_{n}d",
        )
        for n in range(1, 21)
    ],
]


# ═════════════════════════════════════════════════════════════════════════════
# NLP: Nonconvex NLP problems (nlp directory, 17 problems)
# ═════════════════════════════════════════════════════════════════════════════


def _build_nlp_001_010() -> Model:
    """Min x*exp(x)+cos(y)+z^3-z^2; z>=1. Opt: -1.3678794486503105 at [-1,pi,1]."""
    m = dm.Model("nlp_001_010")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z", lb=1.0)
    m.minimize(x * dm.exp(x) + dm.cos(y) + z**3 - z**2)
    return m


def _build_nlp_002_010() -> Model:
    """Feasibility: y=log(x)-0.1; x=cos(y)^2+1.5. Opt: 0 (no objective)."""
    m = dm.Model("nlp_002_010")
    x = m.continuous("x", lb=1e-5)
    y = m.continuous("y")
    m.minimize(x * 0.0)  # feasibility — no explicit objective in MINLPTests
    m.subject_to(y - (dm.log(x) - 0.1) == 0.0)
    m.subject_to(x - (dm.cos(y) ** 2 + 1.5) == 0.0)
    return m


def _build_nlp_003_010() -> Model:
    """Max sqrt(x+0.1); y>=exp(x-2)-1.5; y<=sin(x)^2+2; x,y in [0,4]. Opt: 1.832."""
    m = dm.Model("nlp_003_010")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.maximize(dm.sqrt(x + 0.1))
    m.subject_to(y >= dm.exp(x - 2) - 1.5)
    m.subject_to(y <= dm.sin(x) ** 2 + 2)
    return m


def _build_nlp_003_011() -> Model:
    """Max sqrt(x+0.1)+pi; same constraints as 003_010. Opt: 4.973671432569242."""
    m = dm.Model("nlp_003_011")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.maximize(dm.sqrt(x + 0.1) + math.pi)
    m.subject_to(y >= dm.exp(x - 2) - 1.5)
    m.subject_to(y <= dm.sin(x) ** 2 + 2)
    return m


def _build_nlp_003_012() -> Model:
    """Max x (linear); same constraints as 003_010. Opt: 3.256512665824449."""
    m = dm.Model("nlp_003_012")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.maximize(x)
    m.subject_to(y >= dm.exp(x - 2) - 1.5)
    m.subject_to(y <= dm.sin(x) ** 2 + 2)
    return m


def _build_nlp_003_013() -> Model:
    """Max x (declared as NLobjective); same as 003_012. Opt: 3.256512665824449."""
    m = dm.Model("nlp_003_013")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.maximize(x)
    m.subject_to(y >= dm.exp(x - 2) - 1.5)
    m.subject_to(y <= dm.sin(x) ** 2 + 2)
    return m


def _build_nlp_003_014() -> Model:
    """Max x^2+y; same constraints as 003_010. Opt: 12.618023354784961."""
    m = dm.Model("nlp_003_014")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.maximize(x**2 + y)
    m.subject_to(y >= dm.exp(x - 2) - 1.5)
    m.subject_to(y <= dm.sin(x) ** 2 + 2)
    return m


def _build_nlp_003_015() -> Model:
    """Max x^2+y (declared as NLobjective); same as 003_014. Opt: 12.618023354784961."""
    m = dm.Model("nlp_003_015")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.maximize(x**2 + y)
    m.subject_to(y >= dm.exp(x - 2) - 1.5)
    m.subject_to(y <= dm.sin(x) ** 2 + 2)
    return m


def _build_nlp_003_016() -> Model:
    """Max x+pi; same constraints as 003_010. Opt: 6.398105319414242."""
    m = dm.Model("nlp_003_016")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.maximize(x + math.pi)
    m.subject_to(y >= dm.exp(x - 2) - 1.5)
    m.subject_to(y <= dm.sin(x) ** 2 + 2)
    return m


def _build_nlp_004_010() -> Model:
    """Min tan(x)+y+x*z+0.5*abs(y); x in [-1,1]; sphere+linear constraints.
    Opt: -4.87215904079771. Functions: tan, abs.
    """
    m = dm.Model("nlp_004_010")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    y = m.continuous("y")
    z = m.continuous("z")
    m.minimize(dm.tan(x) + y + x * z + 0.5 * dm.abs(y))
    m.subject_to(x**2 + y**2 + z**2 <= 10)
    m.subject_to(2 * x + 3 * y + z >= -10)
    return m


def _build_nlp_005_010() -> Model:
    """Min x+y; 3 division constraints; x,y>=0. Opt: 1.5449760741521967."""
    m = dm.Model("nlp_005_010")
    x = m.continuous("x", lb=0.0)
    y = m.continuous("y", lb=0.0)
    m.minimize(x + y)
    m.subject_to(y >= 1 / (x + 0.1) - 0.5)
    m.subject_to(x >= y ** (-2) - 0.5)
    m.subject_to(4 / (x + y + 0.1) >= 1)
    return m


def _build_nlp_007_010() -> Model:
    """INFEASIBLE: y=exp(x) and x=y^2 have no real solution."""
    m = dm.Model("nlp_007_010")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize(x * 0.0)
    m.subject_to(y - dm.exp(x) == 0.0)
    m.subject_to(x - y**2 == 0.0)
    return m


def _build_nlp_008_010() -> Model:
    """Min x+y^2+z^3; 3 NL constraints; z in [0,1]. Opt: -0.3755859312158738.
    Also tests dual values (not verified here — only objective checked).
    """
    m = dm.Model("nlp_008_010")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z", lb=0.0, ub=1.0)
    m.minimize(x + y**2 + z**3)
    m.subject_to(y >= dm.exp(-x - 2) + dm.exp(-z - 2) - 2)
    m.subject_to(x**2 <= y**2 + z**2)
    m.subject_to(y >= x / 2 + z)
    return m


def _build_nlp_008_011() -> Model:
    """Same as 008_010 (constraints expressed differently in JuMP). Opt: -0.3755859."""
    m = dm.Model("nlp_008_011")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z", lb=0.0, ub=1.0)
    m.minimize(x + y**2 + z**3)
    m.subject_to(y >= dm.exp(-x - 2) + dm.exp(-z - 2) - 2)
    m.subject_to(x**2 <= y**2 + z**2)
    m.subject_to(y >= x / 2 + z)
    return m


def _build_nlp_009_010() -> Model:
    """Min min(0.75+(x-0.5)^3, 0.75-(x-0.5)^2). Opt: 0.75 at x=0.5."""
    m = dm.Model("nlp_009_010")
    x = m.continuous("x")
    m.minimize(dm.minimum(0.75 + (x - 0.5) ** 3, 0.75 - (x - 0.5) ** 2))
    return m


def _build_nlp_009_011() -> Model:
    """Min max(0.75+(x-0.5)^3, 0.75+(x-0.5)^2). Opt: 0.75 at x=0.5."""
    m = dm.Model("nlp_009_011")
    x = m.continuous("x")
    m.minimize(dm.maximum(0.75 + (x - 0.5) ** 3, 0.75 + (x - 0.5) ** 2))
    return m


# ── NLP instances list ─────────────────────────────────────────────────────

NLP_INSTANCES = [
    pytest.param(
        MINLPTestInstance("nlp_001_010", _build_nlp_001_010, -1.3678794486503105),
        id="nlp_001_010",
    ),
    pytest.param(
        MINLPTestInstance("nlp_002_010", _build_nlp_002_010, 0.0),
        id="nlp_002_010",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_003_010",
            _build_nlp_003_010,
            1.8320787790166984,
            tags=["smoke"],
        ),
        marks=[pytest.mark.smoke],
        id="nlp_003_010",
    ),
    pytest.param(
        MINLPTestInstance("nlp_003_011", _build_nlp_003_011, 4.973671432569242),
        id="nlp_003_011",
    ),
    pytest.param(
        MINLPTestInstance("nlp_003_012", _build_nlp_003_012, 3.256512665824449),
        id="nlp_003_012",
    ),
    pytest.param(
        MINLPTestInstance("nlp_003_013", _build_nlp_003_013, 3.256512665824449),
        id="nlp_003_013",
    ),
    pytest.param(
        MINLPTestInstance("nlp_003_014", _build_nlp_003_014, 12.618023354784961),
        id="nlp_003_014",
    ),
    pytest.param(
        MINLPTestInstance("nlp_003_015", _build_nlp_003_015, 12.618023354784961),
        id="nlp_003_015",
    ),
    pytest.param(
        MINLPTestInstance("nlp_003_016", _build_nlp_003_016, 6.398105319414242),
        id="nlp_003_016",
    ),
    pytest.param(
        MINLPTestInstance("nlp_004_010", _build_nlp_004_010, -4.87215904079771),
        id="nlp_004_010",
    ),
    pytest.param(
        MINLPTestInstance("nlp_005_010", _build_nlp_005_010, 1.5449760741521967),
        id="nlp_005_010",
    ),
    pytest.param(
        MINLPTestInstance("nlp_008_010", _build_nlp_008_010, -0.3755859312158738),
        id="nlp_008_010",
    ),
    pytest.param(
        MINLPTestInstance("nlp_008_011", _build_nlp_008_011, -0.3755859312158738),
        id="nlp_008_011",
    ),
    pytest.param(
        MINLPTestInstance("nlp_009_010", _build_nlp_009_010, 0.75),
        id="nlp_009_010",
    ),
    pytest.param(
        MINLPTestInstance("nlp_009_011", _build_nlp_009_011, 0.75),
        id="nlp_009_011",
    ),
]

# ── NLP infeasible instances ───────────────────────────────────────────────

_NLP_007_010 = MINLPTestInstance(
    "nlp_007_010", _build_nlp_007_010, 0.0, expected_status="infeasible"
)


# ═════════════════════════════════════════════════════════════════════════════
# NLP-MI: Nonconvex MINLP problems (nlp-mi directory, 16 problems)
# ═════════════════════════════════════════════════════════════════════════════


def _build_nlp_mi_001_010() -> Model:
    """Min x*exp(x)+cos(y)+z^3-z^2; y integer>=1; z>=1. Opt: -1.35787195018718."""
    m = dm.Model("nlp_mi_001_010")
    x = m.continuous("x")
    y = m.integer("y", lb=1)  # JuMP: y >= 0.1, Int → lb rounds up to 1
    z = m.continuous("z", lb=1.0)
    m.minimize(x * dm.exp(x) + dm.cos(y) + z**3 - z**2)
    return m


def _build_nlp_mi_002_010() -> Model:
    """Feasibility: y<=log(x)-0.1; x<=cos(y)^2+1.5; x integer>=1, y binary.
    Opt: 0 (no objective). Solution: [2, 0].
    """
    m = dm.Model("nlp_mi_002_010")
    x = m.integer("x", lb=1)  # JuMP: x >= 0.9, Int → lb=1
    y = m.binary("y")
    m.minimize(x * 0.0)
    m.subject_to(y <= dm.log(x) - 0.1)
    m.subject_to(x <= dm.cos(y) ** 2 + 1.5)
    return m


def _build_nlp_mi_003_010() -> Model:
    """Max sqrt(x+0.1); y>=exp(x-2)-1.5; y<=sin(x)^2+2; x,y integer in [0,4].
    Opt: 1.7606816937762844 at [3, 2].
    """
    m = dm.Model("nlp_mi_003_010")
    x = m.integer("x", lb=0, ub=4)
    y = m.integer("y", lb=0, ub=4)
    m.maximize(dm.sqrt(x + 0.1))
    m.subject_to(y >= dm.exp(x - 2) - 1.5)
    m.subject_to(y <= dm.sin(x) ** 2 + 2)
    return m


def _build_nlp_mi_003_011() -> Model:
    """Max sqrt(x+0.1)+pi; same constraints as nlp_mi_003_010. Opt: 4.902274."""
    m = dm.Model("nlp_mi_003_011")
    x = m.integer("x", lb=0, ub=4)
    y = m.integer("y", lb=0, ub=4)
    m.maximize(dm.sqrt(x + 0.1) + math.pi)
    m.subject_to(y >= dm.exp(x - 2) - 1.5)
    m.subject_to(y <= dm.sin(x) ** 2 + 2)
    return m


def _build_nlp_mi_003_012() -> Model:
    """Max x (linear); same constraints as nlp_mi_003_010. Opt: 3."""
    m = dm.Model("nlp_mi_003_012")
    x = m.integer("x", lb=0, ub=4)
    y = m.integer("y", lb=0, ub=4)
    m.maximize(x)
    m.subject_to(y >= dm.exp(x - 2) - 1.5)
    m.subject_to(y <= dm.sin(x) ** 2 + 2)
    return m


def _build_nlp_mi_003_013() -> Model:
    """Max x (NLobjective variant); same as 003_012. Opt: 3."""
    m = dm.Model("nlp_mi_003_013")
    x = m.integer("x", lb=0, ub=4)
    y = m.integer("y", lb=0, ub=4)
    m.maximize(x)
    m.subject_to(y >= dm.exp(x - 2) - 1.5)
    m.subject_to(y <= dm.sin(x) ** 2 + 2)
    return m


def _build_nlp_mi_003_014() -> Model:
    """Max x^2+y; same constraints as nlp_mi_003_010. Opt: 11 at [3, 2]."""
    m = dm.Model("nlp_mi_003_014")
    x = m.integer("x", lb=0, ub=4)
    y = m.integer("y", lb=0, ub=4)
    m.maximize(x**2 + y)
    m.subject_to(y >= dm.exp(x - 2) - 1.5)
    m.subject_to(y <= dm.sin(x) ** 2 + 2)
    return m


def _build_nlp_mi_003_015() -> Model:
    """Max x^2+y (NLobjective variant); same as 003_014. Opt: 11."""
    m = dm.Model("nlp_mi_003_015")
    x = m.integer("x", lb=0, ub=4)
    y = m.integer("y", lb=0, ub=4)
    m.maximize(x**2 + y)
    m.subject_to(y >= dm.exp(x - 2) - 1.5)
    m.subject_to(y <= dm.sin(x) ** 2 + 2)
    return m


def _build_nlp_mi_003_016() -> Model:
    """Max x+pi; same constraints as nlp_mi_003_010. Opt: 6.141592682680717."""
    m = dm.Model("nlp_mi_003_016")
    x = m.integer("x", lb=0, ub=4)
    y = m.integer("y", lb=0, ub=4)
    m.maximize(x + math.pi)
    m.subject_to(y >= dm.exp(x - 2) - 1.5)
    m.subject_to(y <= dm.sin(x) ** 2 + 2)
    return m


def _build_nlp_mi_004_010() -> Model:
    """Min tan(x)+y+x*z+0.5*abs(y); x in [-1,1]; z integer; sphere+linear.
    Opt: -4.5768813091909015 at [-0.997, -0.078, 3].
    """
    m = dm.Model("nlp_mi_004_010")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    y = m.continuous("y")
    z = m.integer("z")
    m.minimize(dm.tan(x) + y + x * z + 0.5 * dm.abs(y))
    m.subject_to(x**2 + y**2 + z**2 <= 10)
    m.subject_to(2 * x + 3 * y + z >= -10)
    return m


def _build_nlp_mi_004_011() -> Model:
    """Same as 004_010 with NLconstraints. Opt: -4.5768813091909015."""
    m = dm.Model("nlp_mi_004_011")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    y = m.continuous("y")
    z = m.integer("z")
    m.minimize(dm.tan(x) + y + x * z + 0.5 * dm.abs(y))
    m.subject_to(x**2 + y**2 + z**2 <= 10)
    m.subject_to(2 * x + 3 * y + z >= -10)
    return m


def _build_nlp_mi_004_012() -> Model:
    """Feasibility: x in [-1,1], z integer; 2 constraints. Opt: 0 (no objective)."""
    m = dm.Model("nlp_mi_004_012")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    y = m.continuous("y")
    z = m.integer("z")
    m.minimize(x * 0.0)
    m.subject_to(x**2 + y**2 + z**2 <= 10)
    m.subject_to(2 * x + 3 * y + z >= -10)
    return m


def _build_nlp_mi_005_010() -> Model:
    """Min x+y; 3 division constraints; x integer>=0, y continuous>=0.
    Opt: 1.8164965727459055 at [1, 0.816].
    """
    m = dm.Model("nlp_mi_005_010")
    x = m.integer("x", lb=0)
    y = m.continuous("y", lb=0.0)
    m.minimize(x + y)
    m.subject_to(y >= 1 / (x + 0.1) - 0.5)
    m.subject_to(x >= y ** (-2) - 0.5)
    m.subject_to(4 / (x + y + 0.1) >= 1)
    return m


def _build_nlp_mi_007_010() -> Model:
    """INFEASIBLE: y=exp(x) and x=y^2; both x,y integer."""
    m = dm.Model("nlp_mi_007_010")
    x = m.integer("x")
    y = m.integer("y")
    m.minimize(x * 0.0)
    m.subject_to(y - dm.exp(x) == 0.0)
    m.subject_to(x - y**2 == 0.0)
    return m


def _build_nlp_mi_007_020() -> Model:
    """INFEASIBLE: (x-0.5)^2+(4y-2)^2<=3; x integer in [-2,3], y binary.
    Infeasible because 4 is always added to LHS regardless of y.
    """
    m = dm.Model("nlp_mi_007_020")
    x = m.integer("x", lb=-2, ub=3)
    y = m.binary("y")
    m.minimize(x * 0.0)
    m.subject_to((x - 0.5) ** 2 + (4 * y - 2) ** 2 <= 3)
    return m


# ── NLP-MI instances list ──────────────────────────────────────────────────

NLP_MI_INSTANCES = [
    pytest.param(
        MINLPTestInstance(
            "nlp_mi_001_010", _build_nlp_mi_001_010, -1.35787195018718, has_integers=True
        ),
        id="nlp_mi_001_010",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_mi_002_010",
            _build_nlp_mi_002_010,
            0.0,
            has_integers=True,
            tags=["smoke"],
        ),
        marks=[pytest.mark.smoke],
        id="nlp_mi_002_010",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_mi_003_010", _build_nlp_mi_003_010, 1.7606816937762844, has_integers=True
        ),
        id="nlp_mi_003_010",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_mi_003_011", _build_nlp_mi_003_011, 4.9022743473660775, has_integers=True
        ),
        id="nlp_mi_003_011",
    ),
    pytest.param(
        MINLPTestInstance("nlp_mi_003_012", _build_nlp_mi_003_012, 3.0, has_integers=True),
        id="nlp_mi_003_012",
    ),
    pytest.param(
        MINLPTestInstance("nlp_mi_003_013", _build_nlp_mi_003_013, 3.0, has_integers=True),
        id="nlp_mi_003_013",
    ),
    pytest.param(
        MINLPTestInstance("nlp_mi_003_014", _build_nlp_mi_003_014, 11.0, has_integers=True),
        id="nlp_mi_003_014",
    ),
    pytest.param(
        MINLPTestInstance("nlp_mi_003_015", _build_nlp_mi_003_015, 11.0, has_integers=True),
        id="nlp_mi_003_015",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_mi_003_016", _build_nlp_mi_003_016, 6.141592682680717, has_integers=True
        ),
        id="nlp_mi_003_016",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_mi_004_010", _build_nlp_mi_004_010, -4.5768813091909015, has_integers=True
        ),
        id="nlp_mi_004_010",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_mi_004_011", _build_nlp_mi_004_011, -4.5768813091909015, has_integers=True
        ),
        id="nlp_mi_004_011",
    ),
    pytest.param(
        MINLPTestInstance("nlp_mi_004_012", _build_nlp_mi_004_012, 0.0, has_integers=True),
        id="nlp_mi_004_012",
    ),
    pytest.param(
        MINLPTestInstance(
            "nlp_mi_005_010",
            _build_nlp_mi_005_010,
            1.8164965727459055,
            has_integers=True,
            tags=["smoke"],
        ),
        marks=[pytest.mark.smoke],
        id="nlp_mi_005_010",
    ),
]

# ── Infeasible instances ───────────────────────────────────────────────────

INFEASIBLE_INSTANCES = [
    MINLPTestInstance("nlp_007_010", _build_nlp_007_010, 0.0, expected_status="infeasible"),
    MINLPTestInstance("nlp_mi_007_010", _build_nlp_mi_007_010, 0.0, expected_status="infeasible"),
    MINLPTestInstance("nlp_mi_007_020", _build_nlp_mi_007_020, 0.0, expected_status="infeasible"),
]

# ── xfail registrations: nlp_006_010 and nlp_mi_006_010 ─────────────────
# These use JuMP.register() user-defined functions which have no discopt
# equivalent. They are tracked in known_failures.toml and skipped here.
# When user-defined function support is added, remove from known_failures.toml
# and add the build functions and instances to NLP_INSTANCES / NLP_MI_INSTANCES.


# ═════════════════════════════════════════════════════════════════════════════
# Test classes
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.minlptests
@pytest.mark.slow
class TestMINLPTestsNLPCvx:
    """Convex NLP problems from MINLPTests.jl nlp-cvx directory (53 problems).

    All 53 problems are convex. The discopt convex fast path should fire for all.
    If result.convex_fast_path is False for an instance, add it to known_failures.toml
    with category="no_convergence" and the appropriate issue number.
    """

    @pytest.mark.parametrize("instance", NLP_CVX_INSTANCES)
    def test_optimal_value(self, instance: MINLPTestInstance) -> None:
        _xfail_if_known(instance.problem_id)
        model = instance.build_fn()
        result = model.solve(time_limit=60.0, gap_tolerance=1e-6)
        assert_optimal(result, instance.expected_obj, instance.problem_id)
        if instance.is_convex:
            assert result.convex_fast_path is True, (
                f"[{instance.problem_id}] Expected discopt convex fast path "
                f"(add to known_failures.toml if not yet supported)"
            )


@pytest.mark.minlptests
@pytest.mark.slow
class TestMINLPTestsNLP:
    """Nonconvex NLP problems from MINLPTests.jl nlp directory (15 feasible)."""

    @pytest.mark.parametrize("instance", NLP_INSTANCES)
    def test_optimal_value(self, instance: MINLPTestInstance) -> None:
        _xfail_if_known(instance.problem_id)
        model = instance.build_fn()
        result = model.solve(time_limit=120.0, gap_tolerance=1e-6)
        assert_optimal(result, instance.expected_obj, instance.problem_id)


@pytest.mark.minlptests
@pytest.mark.slow
class TestMINLPTestsMI:
    """Nonconvex MINLP problems from MINLPTests.jl nlp-mi directory (13 feasible)."""

    @pytest.mark.parametrize("instance", NLP_MI_INSTANCES)
    def test_optimal_value(self, instance: MINLPTestInstance) -> None:
        _xfail_if_known(instance.problem_id)
        model = instance.build_fn()
        result = model.solve(time_limit=120.0, gap_tolerance=1e-6)
        assert_optimal(result, instance.expected_obj, instance.problem_id)


@pytest.mark.minlptests
@pytest.mark.slow
class TestMINLPTestsInfeasible:
    """Infeasibility detection tests from MINLPTests.jl (3 problems)."""

    @pytest.mark.parametrize(
        "instance",
        INFEASIBLE_INSTANCES,
        ids=lambda i: i.problem_id,
    )
    def test_detected_infeasible(self, instance: MINLPTestInstance) -> None:
        model = instance.build_fn()
        result = model.solve(time_limit=30.0)
        assert_infeasible(result, instance.problem_id)
