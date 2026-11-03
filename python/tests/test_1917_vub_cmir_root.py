"""#1917: regression fixture kept from the retired ``DISCOPT_VUB_CMIR_ROOT``.

The flag added root path-aggregation c-MIR rows with variable-bound substitution
to the spatial McCormick LP. Its section 5 panel (docs/dev/data/1917-vub-cmir-panel/)
was cert-clean, but its node SGM ON/OFF was 1.0132 against a 0.95 bar, so the flag
and its implementation were retired (docs/dev/flag-retirement-audit.md).

The fixture models the structure it targeted, taken from GDPlib ``syngas`` under
``gdp.bigm``: a switch ``f_out <= 5 y`` and a big-M balance
``f_out - f_in <= 5 (1 - y)``. Together they imply ``f_out <= f_in`` at every
mixed-integer point, but the LP relaxation does not. The default solve must
certify the known optimum.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model

pytestmark = pytest.mark.smoke


def _bigm_model() -> Model:
    """Demand 2 is met by one of two units. A unit's outflow needs it switched on
    (``y_k``) and then only passes what flows in. The ``exp`` term keeps the model
    off the native quadratic kernel so the solve takes the McCormick LP route."""
    m = Model("bigm_flow")
    f_in1 = m.continuous("f_in1", lb=0.0, ub=5.0)
    f_out1 = m.continuous("f_out1", lb=0.0, ub=5.0)
    y1 = m.binary("y1")
    f_in2 = m.continuous("f_in2", lb=0.0, ub=5.0)
    f_out2 = m.continuous("f_out2", lb=0.0, ub=5.0)
    y2 = m.binary("y2")
    x = m.continuous("x", lb=1.0, ub=2.0)
    for f_in, f_out, y in ((f_in1, f_out1, y1), (f_in2, f_out2, y2)):
        m.subject_to(f_out <= 5.0 * y)
        m.subject_to(f_out - f_in <= 5.0 * (1.0 - y))
    m.subject_to(f_out1 + f_out2 >= 2.0)
    m.minimize(x * f_in1 + 2.5 * f_in2 + 3.0 * y1 + y2 + dm.exp(x))
    return m


#: optimum: unit 1 on, f_in1 = f_out1 = 2, x = 1 -> 2 + 3 + e (unit 2 costs 6 + e)
_OPT = 5.0 + float(np.e)


def test_default_solve_certifies_the_bigm_flow_optimum():
    r = _bigm_model().solve(time_limit=60)
    assert r.status == "optimal", r.status
    assert r.objective == pytest.approx(_OPT, abs=1e-5)
    assert r.bound is not None and r.bound <= r.objective + 1e-6
