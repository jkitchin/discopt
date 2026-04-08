"""MINLPTests.jl benchmark problem registry.

A representative subset of problems from https://github.com/jump-dev/MINLPTests.jl
registered for performance tracking.  The full correctness test suite lives in
python/tests/test_minlptests.py.

Categories used:
  nlp_convex     — convex continuous NLP (from nlp-cvx/)
  nlp_nonconvex  — nonconvex continuous NLP (from nlp/)
  minlp_nonconvex — nonconvex MINLP (from nlp-mi/)
"""

from __future__ import annotations

import math

from benchmarks.problems.base import TestProblem, register

_NLP_CVX = ["ipm", "ripopt", "ipopt"]
_NLP = ["ipm", "ripopt", "ipopt"]
_MI = ["ipm", "ripopt", "ipopt"]


# ── Convex NLP (nlp-cvx) ──────────────────────────────────────────────────


def _build_nlp_cvx_101_010():
    import discopt.modeling as dm

    m = dm.Model("nlp_cvx_101_010")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.minimize(-x - y)
    m.subject_to(x**2 + y**2 <= 1.0)
    return m


def _build_nlp_cvx_107_010():
    import discopt.modeling as dm

    m = dm.Model("nlp_cvx_107_010")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize((x - 0.5) ** 2 + (y - 0.5) ** 2)
    m.subject_to(x**2 + y**2 <= 1.0)
    return m


def _build_nlp_cvx_201_010():
    import discopt.modeling as dm

    m = dm.Model("nlp_cvx_201_010")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z")
    m.minimize(-x - y - z)
    m.subject_to(x**2 + y**2 + z**2 <= 1.0)
    return m


def _build_nlp_cvx_108_011():
    import discopt.modeling as dm

    m = dm.Model("nlp_cvx_108_011")
    x = m.continuous("x", lb=0.0)
    y = m.continuous("y", lb=0.0)
    m.minimize((x - 3.0) ** 2 + y**2)
    m.subject_to(2 * x**2 - 4 * x * y - 4 * x + 4 <= y)
    m.subject_to(y**2 <= -x + 2)
    return m


def _build_nlp_cvx_105_010():
    import discopt.modeling as dm

    m = dm.Model("nlp_cvx_105_010")
    x = m.continuous("x", lb=1e-5)
    y = m.continuous("y")
    m.minimize(-x - y)
    m.subject_to(dm.exp(x - 2.0) - 0.5 <= y)
    m.subject_to(dm.log(x) + 0.5 >= y)
    return m


register(
    TestProblem(
        name="minlptests_nlp_cvx_101_010",
        category="nlp_convex",
        level="smoke",
        build_fn=_build_nlp_cvx_101_010,
        known_optimum=-math.sqrt(2),
        applicable_solvers=_NLP_CVX,
        n_vars=2,
        n_constraints=1,
        source="programmatic",
        tags=["minlptests", "nlp_cvx", "unit_disk"],
    )
)

register(
    TestProblem(
        name="minlptests_nlp_cvx_107_010",
        category="nlp_convex",
        level="smoke",
        build_fn=_build_nlp_cvx_107_010,
        known_optimum=0.0,
        applicable_solvers=_NLP_CVX,
        n_vars=2,
        n_constraints=1,
        source="programmatic",
        tags=["minlptests", "nlp_cvx", "unit_disk"],
    )
)

register(
    TestProblem(
        name="minlptests_nlp_cvx_201_010",
        category="nlp_convex",
        level="full",
        build_fn=_build_nlp_cvx_201_010,
        known_optimum=-math.sqrt(3),
        applicable_solvers=_NLP_CVX,
        n_vars=3,
        n_constraints=1,
        source="programmatic",
        tags=["minlptests", "nlp_cvx", "3d"],
    )
)

register(
    TestProblem(
        name="minlptests_nlp_cvx_108_011",
        category="nlp_convex",
        level="full",
        build_fn=_build_nlp_cvx_108_011,
        known_optimum=1.5240966871955863,
        applicable_solvers=_NLP_CVX,
        n_vars=2,
        n_constraints=2,
        source="programmatic",
        tags=["minlptests", "nlp_cvx", "nonlinear_constraint"],
    )
)

register(
    TestProblem(
        name="minlptests_nlp_cvx_105_010",
        category="nlp_convex",
        level="full",
        build_fn=_build_nlp_cvx_105_010,
        known_optimum=-4.176004405036646,
        applicable_solvers=_NLP_CVX,
        n_vars=2,
        n_constraints=2,
        source="programmatic",
        tags=["minlptests", "nlp_cvx", "exp_log"],
    )
)


# ── Nonconvex NLP (nlp) ───────────────────────────────────────────────────


def _build_nlp_003_010():
    import discopt.modeling as dm

    m = dm.Model("nlp_003_010")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.maximize(dm.sqrt(x + 0.1))
    m.subject_to(y >= dm.exp(x - 2) - 1.5)
    m.subject_to(y <= dm.sin(x) ** 2 + 2)
    return m


def _build_nlp_001_010():
    import discopt.modeling as dm

    m = dm.Model("nlp_001_010")
    x = m.continuous("x")
    y = m.continuous("y")
    z = m.continuous("z", lb=1.0)
    m.minimize(x * dm.exp(x) + dm.cos(y) + z**3 - z**2)
    return m


def _build_nlp_005_010():
    import discopt.modeling as dm

    m = dm.Model("nlp_005_010")
    x = m.continuous("x", lb=0.0)
    y = m.continuous("y", lb=0.0)
    m.minimize(x + y)
    m.subject_to(y >= 1 / (x + 0.1) - 0.5)
    m.subject_to(x >= y ** (-2) - 0.5)
    m.subject_to(4 / (x + y + 0.1) >= 1)
    return m


register(
    TestProblem(
        name="minlptests_nlp_003_010",
        category="nlp_nonconvex",
        level="smoke",
        build_fn=_build_nlp_003_010,
        known_optimum=1.8320787790166984,
        applicable_solvers=_NLP,
        n_vars=2,
        n_constraints=2,
        source="programmatic",
        tags=["minlptests", "nlp", "maximize", "sqrt", "sin"],
    )
)

register(
    TestProblem(
        name="minlptests_nlp_001_010",
        category="nlp_nonconvex",
        level="full",
        build_fn=_build_nlp_001_010,
        known_optimum=-1.3678794486503105,
        applicable_solvers=_NLP,
        n_vars=3,
        n_constraints=0,
        source="programmatic",
        tags=["minlptests", "nlp", "exp", "cos"],
    )
)

register(
    TestProblem(
        name="minlptests_nlp_005_010",
        category="nlp_nonconvex",
        level="full",
        build_fn=_build_nlp_005_010,
        known_optimum=1.5449760741521967,
        applicable_solvers=_NLP,
        n_vars=2,
        n_constraints=3,
        source="programmatic",
        tags=["minlptests", "nlp", "division"],
    )
)


# ── Nonconvex MINLP (nlp-mi) ─────────────────────────────────────────────


def _build_nlp_mi_005_010():
    import discopt.modeling as dm

    m = dm.Model("nlp_mi_005_010")
    x = m.integer("x", lb=0)
    y = m.continuous("y", lb=0.0)
    m.minimize(x + y)
    m.subject_to(y >= 1 / (x + 0.1) - 0.5)
    m.subject_to(x >= y ** (-2) - 0.5)
    m.subject_to(4 / (x + y + 0.1) >= 1)
    return m


def _build_nlp_mi_001_010():
    import discopt.modeling as dm

    m = dm.Model("nlp_mi_001_010")
    x = m.continuous("x")
    y = m.integer("y", lb=1)
    z = m.continuous("z", lb=1.0)
    m.minimize(x * dm.exp(x) + dm.cos(y) + z**3 - z**2)
    return m


def _build_nlp_mi_003_010():
    import discopt.modeling as dm

    m = dm.Model("nlp_mi_003_010")
    x = m.integer("x", lb=0, ub=4)
    y = m.integer("y", lb=0, ub=4)
    m.maximize(dm.sqrt(x + 0.1))
    m.subject_to(y >= dm.exp(x - 2) - 1.5)
    m.subject_to(y <= dm.sin(x) ** 2 + 2)
    return m


register(
    TestProblem(
        name="minlptests_nlp_mi_005_010",
        category="minlp_nonconvex",
        level="smoke",
        build_fn=_build_nlp_mi_005_010,
        known_optimum=1.8164965727459055,
        applicable_solvers=_MI,
        n_vars=2,
        n_constraints=3,
        source="programmatic",
        tags=["minlptests", "nlp_mi", "division", "integer"],
    )
)

register(
    TestProblem(
        name="minlptests_nlp_mi_002_010",
        category="minlp_nonconvex",
        level="smoke",
        build_fn=_build_nlp_mi_001_010,  # used as a second smoke instance
        known_optimum=-1.35787195018718,
        applicable_solvers=_MI,
        n_vars=3,
        n_constraints=0,
        source="programmatic",
        tags=["minlptests", "nlp_mi", "exp", "cos", "integer"],
    )
)

register(
    TestProblem(
        name="minlptests_nlp_mi_003_010",
        category="minlp_nonconvex",
        level="full",
        build_fn=_build_nlp_mi_003_010,
        known_optimum=1.7606816937762844,
        applicable_solvers=_MI,
        n_vars=2,
        n_constraints=2,
        source="programmatic",
        tags=["minlptests", "nlp_mi", "maximize", "integer"],
    )
)
