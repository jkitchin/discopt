"""Regression tests for two GDP routing defects found by the GDP/Pyomo parity audit.

D1 -- GDPopt-LOA dropped the objective constant. The linear master minimizes
``c @ x`` only, and ``_solve_master_milp`` discarded the offset, so the master's
dual bound was ``min c @ x`` instead of ``min c @ x + const``. On ``min x - 10``
LOA returned incumbent ``-10`` with ``bound = 0`` and ``gap_certified=True``: a
dual bound above the incumbent. On ``min x + 10`` the bound stayed at ``0`` and
the convex instance could never certify.

D2 -- an unknown ``gdp_method`` silently solved as big-M. Every route maps a name
it does not handle to big-M, so ``gdp_method="gloa"`` (reserved, unimplemented),
``"lbb"`` or a typo reported ``optimal`` with no warning.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt._relax.gdp_reformulate import reformulate_gdp


def _offset_model(const: float, maximize: bool) -> dm.Model:
    m = dm.Model("loa_offset")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 2], [x >= 5]], name="d")
    if maximize:
        m.maximize(-x + const)
    else:
        m.minimize(x + const)
    return m


@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("const", [-10.0, 0.0, 10.0])
def test_loa_bound_includes_objective_constant(const, maximize):
    r = _offset_model(const, maximize).solve(gdp_method="loa", time_limit=30)
    # The optimum is x = 0 in both senses, so the objective is the constant.
    assert r.status == "optimal", (r.status, r.objective, r.bound)
    assert r.objective == pytest.approx(const, abs=1e-6)
    assert r.bound is not None
    assert r.bound == pytest.approx(const, abs=1e-6)
    # The certificate invariant, on the right side for each sense.
    if maximize:
        assert r.bound >= r.objective - 1e-6
    else:
        assert r.bound <= r.objective + 1e-6


def _gdp_model() -> dm.Model:
    m = dm.Model("gdp_method")
    x = m.continuous("x", lb=0, ub=10)
    m.minimize(x)
    m.either_or([[x <= 2], [x >= 5]], name="d")
    return m


@pytest.mark.parametrize("method", ["bigmm", "Big-M", "", "goa"])
def test_unknown_gdp_method_raises(method):
    with pytest.raises(ValueError, match="Unknown gdp_method"):
        _gdp_model().solve(gdp_method=method, time_limit=10)


def test_unknown_gdp_method_raises_without_disjunctions():
    m = dm.Model("plain")
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(x)
    with pytest.raises(ValueError, match="Unknown gdp_method"):
        m.solve(gdp_method="bigmm", time_limit=10)
    # #1775 G3.1's reserved native names have all landed (G4.5); the reservation
    # path is tested in test_1775_g3_1_gdp_routing_and_base.


@pytest.mark.parametrize("method", ["gloa", "typo"])
def test_reformulate_gdp_rejects_unknown_method(method):
    with pytest.raises(ValueError, match="Unknown GDP reformulation method"):
        reformulate_gdp(_gdp_model(), method=method)


@pytest.mark.parametrize("method", ["big-m", "hull", "mbigm", "auto", "simplex", "loa", "gloa"])
def test_known_gdp_methods_still_solve(method):
    r = _gdp_model().solve(gdp_method=method, time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(0.0, abs=1e-6)


# The audit for D1's class (a linear objective's constant dropped where ``c @ x`` is
# compared with an objective *value*) found the same defect in OBBT's incumbent
# cutoff: ``run_obbt`` and ``bootstrap_finite_bounds`` (reached from
# ``obbt_tighten_root`` whenever a bound is infinite) wrote ``sign * c @ x <= z``
# for an objective ``c @ x + k``. With ``sign * k < 0`` that row is tighter than
# valid and excluded the optimum.


def _nonconvex_offset_model(const: float, maximize: bool, ub):
    m = dm.Model("obbt_offset")
    x = m.continuous("x", lb=0, ub=ub)
    m.subject_to(x * x >= 9)  # optimum x = 3; the LP subsystem cannot see it
    if maximize:
        m.maximize(-x + const)
    else:
        m.minimize(x + const)
    # Internal minimization space value of the optimum, which is what every
    # cutoff caller passes.
    return m, (3.0 - const) if maximize else (3.0 + const)


@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("const", [-2.0, 0.0, 2.0])
def test_run_obbt_cutoff_keeps_optimum(const, maximize):
    from discopt._relax.obbt import run_obbt

    m, z = _nonconvex_offset_model(const, maximize, ub=10)
    r = run_obbt(m, incumbent_cutoff=z)
    assert r.tightened_ub[0] >= 3.0 - 1e-6, r.tightened_ub
    # The cutoff is exactly x <= 3 for every constant.
    assert r.tightened_ub[0] == pytest.approx(3.0, abs=1e-5)


@pytest.mark.parametrize("maximize", [False, True])
@pytest.mark.parametrize("const", [-2.0, 0.0, 2.0])
def test_root_obbt_bootstrap_cutoff_keeps_optimum(const, maximize):
    import numpy as np
    from discopt._relax.obbt import bootstrap_finite_bounds, obbt_tighten_root

    m, z = _nonconvex_offset_model(const, maximize, ub=None)
    lb, ub = np.array([0.0]), np.array([np.inf])
    _, bub, _, _ = bootstrap_finite_bounds(m, lb, ub, incumbent_cutoff=z)
    assert bub[0] >= 3.0 - 1e-6, bub
    r = obbt_tighten_root(m, lb, ub, incumbent_cutoff=z)
    assert r.ub[0] >= 3.0 - 1e-6, r.ub
