"""#1917: ``DISCOPT_CONVEX_ROUTE_GDP_LOWERED`` decides the convex route on the lowering.

The convex-MINLP auto-route (#1059) classified the *un-lowered* GDP model, and the
classifier marks every ``_IndicatorConstraint`` unproven, so a GDP model never
routed to LP/NLP-BB even when its big-M lowering is certified convex. Measured on
GDPlib ``batch_processing``: raw 450 of 700 rows unproven, lowered 790 of 790
convex; NLP-BB found no incumbent in 300 s, LP/NLP-BB found 685441.23.

With the flag ON, a GDP model whose only unproven rows are GDP rows is lowered by
the requested ``gdp_method`` and classified again. These tests pin the gate, not
the solve.
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt.solver import _convex_minlp_auto_route

FLAG = "DISCOPT_CONVEX_ROUTE_GDP_LOWERED"


@pytest.fixture(autouse=True)
def _route_env(monkeypatch):
    monkeypatch.setenv("DISCOPT_CONVEX_MINLP_ROUTE", "1")
    monkeypatch.delenv("DISCOPT_CONVEX_ROUTE_METHOD", raising=False)
    monkeypatch.delenv("DISCOPT_CONVEX_ROUTE_OA_MASTER", raising=False)


def _convex_gdp(extra_nonconvex: bool = False):
    m = dm.Model("convex_gdp")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    z = m.binary("z")
    m.minimize((x - 3.0) ** 2 + (y - 4.0) ** 2 + 2.0 * z)
    m.if_then(z, [x + y >= 6.0])
    m.if_then(1 - z, [x**2 + y**2 <= 4.0])
    if extra_nonconvex:
        m.subject_to(x * y >= 1.0)  # bilinear: a non-GDP row that is not convex
    return m


def test_off_does_not_route_a_gdp_model(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    method, reason, _ = _convex_minlp_auto_route(_convex_gdp(), gdp_method="big-m")
    assert method is None, reason


def test_on_routes_the_convex_lowering(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    model = _convex_gdp()
    n_before = len(model._constraints)
    method, reason, _ = _convex_minlp_auto_route(model, gdp_method="big-m")
    assert method == "lp_nlp_bb", reason
    assert "GDP lowered by big-m" in reason
    # The router lowers a copy; the caller's model is untouched.
    assert len(model._constraints) == n_before


def test_on_refuses_when_a_non_gdp_row_is_not_convex(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    method, reason, _ = _convex_minlp_auto_route(
        _convex_gdp(extra_nonconvex=True), gdp_method="big-m"
    )
    assert method is None
    assert reason == "not routed: a non-GDP row is not convex"


@pytest.mark.parametrize("gdp_method", [None, "binary_multiplication"])
def test_on_ignores_methods_outside_the_lowering_set(monkeypatch, gdp_method):
    monkeypatch.setenv(FLAG, "1")
    method, reason, _ = _convex_minlp_auto_route(_convex_gdp(), gdp_method=gdp_method)
    assert method is None, reason
    assert "GDP lowered" not in reason


def test_on_solve_matches_off_solve(monkeypatch):
    """Same certified optimum through the route as through the default path."""
    objs = {}
    for arm in ("0", "1"):
        monkeypatch.setenv(FLAG, arm)
        r = _convex_gdp().solve(gdp_method="big-m", time_limit=60)
        assert r.status == "optimal", (arm, r.status)
        objs[arm] = r.objective
    assert objs["0"] == pytest.approx(objs["1"], abs=1e-5, rel=1e-6)
