"""compute_iis sees fast-API (builder-resident) linear rows.

``add_linear_constraints`` rows -- and, since highs-parity S2, every row of a linear
model read by ``from_nl`` / ``from_mps`` / ``from_lp`` -- live only in the builder,
not in ``model._constraints``. Both IIS engines enumerated ``_constraints`` alone, so
on such a model they saw no rows and raised "the model is feasible" on an infeasible
one (caught by the P0.6 IIS baseline panel once file loading went to matrix rows).
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt.infeasibility import compute_iis
from discopt.modeling.core import Model


def _infeasible_fast_api_model(integer: bool) -> Model:
    # x + y >= 3, x + y <= 1 (the IIS), plus a redundant row x - y <= 5.
    m = Model("fast_iis")
    x = (
        m.integer("x", shape=(2,), lb=0, ub=10)
        if integer
        else m.continuous("x", shape=(2,), lb=0, ub=10)
    )
    m.add_linear_constraints(np.array([[1.0, 1.0]]), x, ">=", np.array([3.0]))
    m.add_linear_constraints(np.array([[1.0, 1.0], [1.0, -1.0]]), x, "<=", np.array([1.0, 5.0]))
    m.minimize(x[0])
    assert len(m._constraints) == 0  # the rows really are builder-resident
    return m


@pytest.mark.parametrize("integrality", [False, True])
def test_iis_of_builder_rows(integrality):
    m = _infeasible_fast_api_model(integer=integrality)
    res = compute_iis(m, include_integrality=integrality, time_limit=30.0)
    assert len(res.constraints) == 2, res.constraints
    assert all(c in m._constraints[:2] for c in res.constraints)
