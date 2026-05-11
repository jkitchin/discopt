"""M1 regression problems.

Small problems whose objective or constraints apply a univariate transcendental
to a non-trivial inner expression (bilinear, sum-of-squares, etc.).  These
exercise the Tsoukalas & Mitsos 2014 univariate composition rule wired into
``discopt._jax.relaxation_compiler`` (issue #51, item M1).

Each instance has a closed-form known optimum that is straightforward to
verify by inspection, so the regression check is purely about correctness:
``incorrect_count == 0`` on these instances implies the relaxation path is
sound on composed univariates.
"""

from __future__ import annotations

import math

from benchmarks.problems.base import TestProblem, register

_APPLICABLE = ["ipm", "ripopt", "ipopt"]


def _build_exp_bilinear():
    """min exp(x*y) for x,y in [0,1]. Optimum = 1.0 at x*y=0."""
    import discopt.modeling as dm

    m = dm.Model("m1_exp_bilinear")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(dm.exp(x * y))
    return m


def _build_log_sumsq():
    """min log(x^2 + y^2 + 1) for x,y in [-1,1]. Optimum = 0 at (0,0)."""
    import discopt.modeling as dm

    m = dm.Model("m1_log_sumsq")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    m.minimize(dm.log(x**2 + y**2 + 1.0))
    return m


def _build_neg_sqrt_bilinear():
    """min -sqrt(x*y + 1) for x,y in [0,1]. Optimum = -sqrt(2) at x=y=1."""
    import discopt.modeling as dm

    m = dm.Model("m1_neg_sqrt_bilinear")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(-dm.sqrt(x * y + 1.0))
    return m


register(
    TestProblem(
        name="m1_exp_bilinear",
        category="nlp_nonconvex",
        level="smoke",
        build_fn=_build_exp_bilinear,
        known_optimum=1.0,
        applicable_solvers=_APPLICABLE,
        n_vars=2,
        n_constraints=0,
        source="programmatic",
        tags=["m1", "tm2014", "exp", "bilinear"],
    )
)

register(
    TestProblem(
        name="m1_log_sumsq",
        category="nlp_nonconvex",
        level="smoke",
        build_fn=_build_log_sumsq,
        known_optimum=0.0,
        applicable_solvers=_APPLICABLE,
        n_vars=2,
        n_constraints=0,
        source="programmatic",
        tags=["m1", "tm2014", "log", "sumsq"],
    )
)

register(
    TestProblem(
        name="m1_neg_sqrt_bilinear",
        category="nlp_nonconvex",
        level="smoke",
        build_fn=_build_neg_sqrt_bilinear,
        known_optimum=-math.sqrt(2.0),
        applicable_solvers=_APPLICABLE,
        n_vars=2,
        n_constraints=0,
        source="programmatic",
        tags=["m1", "tm2014", "sqrt", "bilinear"],
    )
)
