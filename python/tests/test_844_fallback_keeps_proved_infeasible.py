"""A proved ``infeasible`` must survive the #844 no-incumbent fallback.

Found by adversarial testing of #1229: an integer column boxed in ``[-1.6, -1.07]``
holds no integer, and declared-box tightening proves the model infeasible. The #844
LP-spatial fallback fires on "no objective" and so ran anyway; its primal verifier
rounded ``x0`` to ``-1``, clipped it back into the box at ``-1.07`` and accepted that
point, because it checked only the constraints. The solve returned ``optimal``
999980 with ``gap_certified=True`` on both LP backends.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.lp_spatial_bb import solve_lp_spatial_bb


def _no_integer_in_box():
    m = dm.Model("no_integer_in_box")
    x0 = m.integer("x0", lb=-1.600591231915839, ub=-1.0719312778601353)
    x1 = m.integer("x1", lb=-5.0, ub=0.0)
    m.subject_to(2 * x0 + x1 <= -6.596627079311231)
    m.minimize(4 * x1 + 1e6)
    return m


@pytest.mark.parametrize("backend", ["highs", "rust"])
def test_proved_infeasible_is_not_overwritten(monkeypatch, backend):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", backend)
    res = _no_integer_in_box().solve(time_limit=20)
    assert res.status == "infeasible"
    assert res.objective is None


@pytest.mark.parametrize("backend", ["highs", "rust"])
def test_integer_box_without_an_integer_and_no_rows_is_infeasible(monkeypatch, backend):
    """With no row, no tightening rule rounded the box, and ``=rust`` raised
    ``RuntimeError: MILP-BB returned an infeasible point`` on the clipped x = 1.2."""
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", backend)
    m = dm.Model("no_integer_no_rows")
    x = m.integer("x", lb=1.2, ub=1.8)
    m.minimize(x)
    res = m.solve(time_limit=20)
    assert res.status == "infeasible" and res.gap_certified
    assert res.objective is None


def test_integer_box_within_integrality_tolerance_is_not_infeasible():
    """The empty-integer-box check keeps the 1e-5 integrality slack."""
    m = dm.Model("integer_box_touching")
    x = m.integer("x", lb=1.000001, ub=1.8)
    m.minimize(x)
    res = m.solve(time_limit=20)
    assert res.status != "infeasible"


def test_lp_spatial_primal_verifier_refuses_a_clipped_non_integer_point():
    res = solve_lp_spatial_bb(_no_integer_in_box(), time_limit=10.0, gap_tolerance=1e-4)
    assert res is not None  # the engine took the model, so its verifier was exercised
    if res.x is not None:
        x = np.asarray(res.x, dtype=float)
        assert np.all(np.abs(x - np.round(x)) <= 1e-6), x
    assert res.status != "optimal"
