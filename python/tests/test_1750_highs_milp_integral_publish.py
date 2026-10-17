"""#1750: the HiGHS MILP route published HiGHS's raw column values, not the point it
verified.

HiGHS's tree accepts a binary within ``mip_feasibility_tolerance`` of integral. The
route's incumbent verifier (#1380) tests the *snapped* point, but the ordinary
kOptimal branch then published the raw ``col_value`` and priced the objective there:
on the lexicographic emissions step of a small facility-location model that read
136.99999884 where the claimed assignment is worth exactly 137. ``epsilon_constraint``
uses that value as its first epsilon, and the epsilon sub-MILP at 136.99999884 has no
integral point within row tolerance, so the sweep came back ``incomplete``.

Fix: publish the snapped point and the objective at it, as the repaired and
re-derived branches already did.
"""

from __future__ import annotations

import discopt as do
import numpy as np


def _model():
    rng = np.random.default_rng(3)
    nS, nC = 4, 6
    sites, cust = rng.uniform(0, 10, (nS, 2)), rng.uniform(0, 10, (nC, 2))
    dist = np.linalg.norm(sites[:, None] - cust[None], axis=2)
    d = rng.integers(5, 15, nC).astype(float)
    phi = np.array([30.0, 45.0, 60.0, 80.0])
    e = np.array([3.0, 2.0, 1.2, 0.5])
    u = np.full(nS, 40.0)
    m = do.Model("fl")
    y = m.binary("y", shape=(nS,))
    x = m.binary("x", shape=(nS, nC))
    for i in range(nC):
        m.subject_to(sum(x[j, i] for j in range(nS)) == 1)
    for j in range(nS):
        m.subject_to(sum(d[i] * x[j, i] for i in range(nC)) <= u[j] * y[j])
    cost = sum(phi[j] * y[j] for j in range(nS)) + sum(
        0.5 * dist[j, i] * d[i] * x[j, i] for j in range(nS) for i in range(nC)
    )
    emis = sum(e[j] * d[i] * x[j, i] for j in range(nS) for i in range(nC))
    return m, x, y, cost, emis


def test_lexicographic_step_publishes_integral_point_and_objective():
    m, x, y, cost, emis = _model()
    m.minimize(cost)
    r1 = m.solve()
    assert r1.status == "optimal"
    m.subject_to(cost <= r1.objective + 1e-6)
    m.minimize(emis)
    r = m.solve()
    assert r.status == "optimal"
    assert r.solver_stats.get("route/lp_milp_backend") == 1.0  # the HiGHS route ran
    xv, yv = np.asarray(r.value(x)), np.asarray(r.value(y))
    assert np.array_equal(xv, np.round(xv))
    assert np.array_equal(yv, np.round(yv))
    # Emission coefficients e_j * d_i are multiples of 0.1; the optimum is 137, and the
    # published objective is the objective AT the published point.
    assert abs(r.objective - 137.0) <= 1e-9
    assert r.bound is not None and r.bound <= r.objective


def test_epsilon_constraint_sweep_is_complete():
    from discopt.mo import epsilon_constraint

    m, _, _, cost, emis = _model()
    front = epsilon_constraint(m, [cost, emis], n_points=9, filter=False)
    assert front.incomplete_cells == []
    assert all(p.status == "optimal" for p in front.points)
    assert any(abs(p.objectives[1] - 137.0) <= 1e-9 for p in front.points)
