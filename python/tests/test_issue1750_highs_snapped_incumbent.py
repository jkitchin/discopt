"""#1750: the HiGHS MILP route publishes its incumbent at its integral realisation.

The verifier tests a HiGHS incumbent at ``round(x_int)`` (#1380), but the route
published the fractional readback and its objective: 136.99999884 for an
all-integer 137 on a lexicographic facility location, which then made the
``epsilon_constraint`` sub-MILP at that epsilon fail its row check.
"""

from __future__ import annotations

import discopt as do
import numpy as np
import pytest
from discopt.mo import epsilon_constraint

pytestmark = pytest.mark.smoke


def _facility_location():
    rng = np.random.default_rng(3)
    n_s, n_c = 4, 6
    sites, cust = rng.uniform(0, 10, (n_s, 2)), rng.uniform(0, 10, (n_c, 2))
    dist = np.linalg.norm(sites[:, None] - cust[None], axis=2)
    d = rng.integers(5, 15, n_c).astype(float)
    phi = np.array([30.0, 45.0, 60.0, 80.0])
    e = np.array([3.0, 2.0, 1.2, 0.5])
    m = do.Model("fl")
    y = m.binary("y", shape=(n_s,))
    x = m.binary("x", shape=(n_s, n_c))
    for i in range(n_c):
        m.subject_to(sum(x[j, i] for j in range(n_s)) == 1)
    for j in range(n_s):
        m.subject_to(sum(d[i] * x[j, i] for i in range(n_c)) <= 40.0 * y[j])
    cost = sum(phi[j] * y[j] for j in range(n_s)) + sum(
        0.5 * dist[j, i] * d[i] * x[j, i] for j in range(n_s) for i in range(n_c)
    )
    emis = sum(e[j] * d[i] * x[j, i] for j in range(n_s) for i in range(n_c))
    return m, x, y, cost, emis


def test_lexicographic_objective_is_integral():
    m, x, y, cost, emis = _facility_location()
    m.minimize(cost)
    r1 = m.solve()
    m.subject_to(cost <= r1.objective + 1e-6)
    m.minimize(emis)
    r = m.solve()
    assert r.status == "optimal"
    assert r.algorithm_route.startswith("highs-milp")
    xv, yv = r.value(x), r.value(y)
    assert np.array_equal(xv, np.round(xv))
    assert np.array_equal(yv, np.round(yv))
    # Every emissions coefficient e_j * d_i is a multiple of 0.1, so 137 is exact
    # up to the sum's rounding; before the fix this was 136.99999884.
    assert r.objective == pytest.approx(137.0, abs=1e-9)
    assert r.bound <= r.objective


def test_epsilon_constraint_front_is_complete():
    m, _x, _y, cost, emis = _facility_location()
    front = epsilon_constraint(m, [cost, emis], n_points=9)
    assert not front.incomplete
