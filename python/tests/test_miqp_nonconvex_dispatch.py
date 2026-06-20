"""Regression: a nonconvex MIQP must NOT be solved by the convex MIQP solvers.

A model with a quadratic objective and integer variables is classified ``MIQP``
and was routed unconditionally to ``_solve_qp_highs`` / ``_solve_miqp_bb`` — both
of which assume a convex node QP (a convex relaxation solved to global optimality).
On an indefinite or concave-maximize objective they return a local stationary
point and certify it as the global optimum: a *false-optimal*, the worst failure
class.

Concretely ``max x**2`` over integer ``[-3, 3]`` is internally ``min -x**2``; the
stationary point ``x = 0`` is a local min of ``-x**2`` and the convex solver
certified objective ``0`` instead of the true ``9`` (at ``x = ±3``).

The fix mirrors the pure-continuous QP guard: classify convexity (eigenvalue
sound, sense aware) and use the convex MIQP solvers only when the model is known
convex; otherwise route to the sound spatial McCormick Branch-and-Bound, which
branches the integers and bounds each node with a valid outer relaxation.

Ground truth is computed by exhaustive enumeration over the (small) integer box,
so a wrong ``optimal`` verdict is caught directly.
"""

from __future__ import annotations

import itertools
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import pytest

TOL = 1e-4


def _brute(bounds, sense, obj, cons):
    best = None
    feasible = False
    for pt in itertools.product(*[range(int(lo), int(hi) + 1) for lo, hi in bounds]):
        if all(c(pt) for c in cons):
            feasible = True
            v = obj(pt)
            if best is None or (v < best if sense == "min" else v > best):
                best = v
    return best, feasible


# (name, builder -> Model, bounds, sense, obj(pt), [con(pt)], true via brute force)
_CASES = [
    (
        "max x^2 (concave maximization)",
        lambda: _m(lambda m: m.maximize(_x(m, "x", -3, 3) ** 2)),
        [(-3, 3)],
        "max",
        lambda p: p[0] ** 2,
        [],
    ),
    (
        "min -x^2 (concave minimization)",
        lambda: _m(lambda m: m.minimize(-(_x(m, "x", -3, 3) ** 2))),
        [(-3, 3)],
        "min",
        lambda p: -(p[0] ** 2),
        [],
    ),
    (
        "max indefinite x^2-2y^2",
        lambda: _build_indef(),
        [(0, 4), (0, 4)],
        "max",
        lambda p: p[0] ** 2 - 2 * p[1] ** 2,
        [],
    ),
]


def _m(setup):
    m = dm.Model("miqp")
    setup(m)
    return m


def _x(m, name, lo, hi):
    return m.integer(name, lb=lo, ub=hi)


def _build_indef():
    m = dm.Model("indef")
    x = m.integer("x", lb=0, ub=4)
    y = m.integer("y", lb=0, ub=4)
    m.maximize(x * x - 2 * y * y)
    return m


@pytest.mark.parametrize("name, build, bounds, sense, obj, cons", _CASES)
def test_nonconvex_miqp_not_false_optimal(name, build, bounds, sense, obj, cons):
    true_opt, feasible = _brute(bounds, sense, obj, cons)
    assert feasible
    r = build().solve(time_limit=30.0)
    assert r.objective is not None
    if r.status == "optimal":
        # A certified optimum MUST equal the brute-force truth.
        assert abs(float(r.objective) - true_opt) <= TOL * max(1, abs(true_opt)), (
            f"[{name}] FALSE-OPTIMAL: certified {r.objective} but true optimum is {true_opt}"
        )
    else:
        # Uncertified is acceptable, but the reported value must be a genuine
        # feasible value — never better than the true optimum for the sense.
        better = (
            (float(r.objective) < true_opt - TOL)
            if sense == "min"
            else (float(r.objective) > true_opt + TOL)
        )
        assert not better, f"[{name}] reported impossible obj {r.objective} (true {true_opt})"


def test_convex_miqp_still_solves_correctly():
    """The fix must not break convex MIQP: it still routes to the fast convex path
    and certifies. ``min x^2 + y^2 s.t. x + y >= 3`` over integers has optimum 5."""
    m = dm.Model("cvx_miqp")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.minimize(x * x + y * y)
    m.subject_to(x + y >= 3)
    r = m.solve(time_limit=30.0)
    assert r.status == "optimal"
    assert abs(float(r.objective) - 5.0) <= 1e-4
