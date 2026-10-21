"""#1834: the box |coeff| auxiliary was capped below its own counterpart.

``t_ub = max|variable bound| + 1`` ignored the coefficient's scale: for
``5 * y * p >= 20`` with ``y`` in [-10, 10] the coefficient of ``xi`` is ``5y``,
reaching 40 at the optimum, against a cap of 11 -- a feasible robust model
reported infeasible. The bound now comes from a sound interval enclosure of the
coefficient (unbounded when none is finite).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.ro import BoxUncertaintySet, RobustCounterpart

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize(
    "row", [lambda y, p: 5.0 * y * p, lambda y, p: y * p * 5.0], ids=["left", "right"]
)
def test_scaled_bilinear_row_is_feasible(row):
    m = dm.Model("ro")
    y = m.continuous("y", lb=-10, ub=10)
    p = m.parameter("p", value=1.0)
    m.subject_to(row(y, p) >= 20.0)
    m.minimize(y)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=0.5)).formulate()
    r = m.solve()
    assert r.status == "optimal"
    # p in [0.5, 1.5]: 5*y*p >= 20 for all p needs y >= 8.
    assert float(r.value(y)) == pytest.approx(8.0, abs=1e-4)
    # Every aux bound covers the coefficient magnitude over the variable box.
    for v in m._variables:
        if v.name.startswith("ro_abs"):
            assert float(np.max(v.ub)) >= 50.0


def test_unbounded_coefficient_gives_unbounded_aux():
    m = dm.Model("ro_free")
    y = m.continuous("y", lb=0)  # no upper bound, sign-definite but unbounded
    z = m.continuous("z", lb=-1, ub=1)
    p = m.parameter("p", value=1.0)
    m.subject_to(y * z * p <= 3.0)
    m.subject_to(y <= 4.0)
    m.minimize(-y)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=0.5)).formulate()
    aux = [v for v in m._variables if v.name.startswith("ro_abs")]
    assert aux
    for v in aux:
        assert float(np.max(v.ub)) >= 1e19
        assert v.name in getattr(m, "_free_by_construction", set())
