"""#1774: a convex QP whose IPM point violates an active inequality to tolerance.

The rows pin the feasible set to the single point ``x = (0, 1, 0, 0, 0)``. POUNCE's
qp-ipm returned ``x2 = 1 - 4.7e-9``, where the objective sits 1.8e-9 below the
optimum 0.19; the rigorous tangent bound over the (singleton) implied box was 0.19,
above that "feasible value", and the #1596 certificate refused it as a failed
premise, downgrading a correct ``optimal`` to an uncertified ``feasible``. The
certificate now moves the point onto the inequality sides it violates (as #1655
does for equality rows) before evaluating it.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solver import _project_onto_rows


def _drug2_alone() -> dm.Model:
    beta = np.array([1.2, 0.9, 1.5, 0.8, 1.1])
    v = np.array([0.5, 0.3, 0.6, 0.2, 0.4])
    tau = np.array([0.30, 0.10, 0.60, 0.05, 0.25])
    z = np.array([0, 1, 0, 0, 0.0])
    m = dm.Model("drug2_alone")
    x = [m.continuous(f"x{i + 1}", lb=0, ub=1) for i in range(5)]
    m.minimize(sum(v[i] * x[i] for i in range(5)) ** 2 + sum(tau[i] * x[i] ** 2 for i in range(5)))
    for i in range(5):
        m.subject_to(x[i] <= z[i])
        m.subject_to(-x[i] + 0.3 * z[i] <= 0)
    m.subject_to(-sum(beta[i] * x[i] for i in range(5)) <= -0.9)
    return m


@pytest.mark.parametrize("solver", ["pounce", None])
def test_single_point_qp_is_certified_optimal(solver):
    kw = {"solver": solver} if solver else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _drug2_alone().solve(**kw)
    assert r.status == "optimal" and r.gap_certified, (r.status, r.algorithm_route)
    assert r.objective == pytest.approx(0.19, rel=1e-9)
    assert r.bound is not None and r.bound <= r.objective
    xv = np.array([float(r.x[f"x{i + 1}"]) for i in range(5)])
    # the reported point satisfies every row and bound exactly enough to be feasible
    assert np.all(xv >= 0.0) and np.all(xv <= np.array([0, 1, 0, 0, 0]) + 1e-12)
    assert 0.9 * xv[1] >= 0.9 - 1e-12


def test_projection_repairs_violated_inequality_and_never_worsens():
    # rows: x0 + x1 <= 1 (violated), x0 - x1 <= 0.5 (slack)
    A = np.array([[1.0, 1.0], [1.0, -1.0]])  # noqa: N806
    cl = np.array([-np.inf, -np.inf])
    cu = np.array([1.0, 0.5])
    lb, ub = np.zeros(2), np.ones(2)
    x = np.array([0.4, 0.6 + 3e-9])
    z = _project_onto_rows(x, A, cl, cu, lb, ub)
    assert z is not None
    assert (A @ z)[0] <= 1.0 + 1e-15
    assert (A @ z)[1] <= 0.5
    assert np.all(z >= lb) and np.all(z <= ub)
    # a point that satisfies every inequality needs no repair
    assert _project_onto_rows(np.array([0.4, 0.5]), A, cl, cu, lb, ub) is None


def test_projection_refuses_a_repair_that_worsens_another_row():
    # x0 + x1 <= 1 is violated and x0 >= 0.5 is exactly active. The minimum-norm
    # step on the first row moves x0 below 0.5, so the repair is refused.
    A = np.array([[1.0, 1.0], [-1.0, 0.0]])  # noqa: N806
    cl = np.array([-np.inf, -np.inf])
    cu = np.array([1.0, -0.5])
    lb, ub = np.zeros(2), np.ones(2)
    assert _project_onto_rows(np.array([0.5, 0.5 + 2e-9]), A, cl, cu, lb, ub) is None
