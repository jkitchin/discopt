"""Regression for #1699: an unbounded convex MIQP must never be reported infeasible."""

import warnings

import pytest
from discopt import Model


def _unbounded_miqp():
    m = Model("unb")
    x = m.continuous("x", lb=0)
    y = m.continuous("y", lb=0)
    z = m.integer("z", lb=0, ub=3)
    m.minimize(-x - y + z * z)
    m.subject_to(x - y <= 1)  # (0, 0, 0) is feasible by construction
    return m


@pytest.mark.parametrize("route", ["1", "0"])
def test_unbounded_convex_miqp_not_infeasible(route, monkeypatch):
    monkeypatch.setenv("DISCOPT_CONVEX_MINLP_ROUTE", route)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _unbounded_miqp().solve(time_limit=30)
    assert r.status != "infeasible"
    assert not (r.status == "optimal" and r.gap_certified and r.objective is None)
