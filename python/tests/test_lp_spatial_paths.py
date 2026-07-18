"""LP-spatial branch-and-bound path tests (#87).

``lp_spatial=True`` routes nonconvex continuous models through the
LP-relaxation spatial B&B instead of the NLP-based loop. Same rule as every
other engine option: the certificate must match the closed-form optimum.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest
from discopt.modeling.core import Model

pytestmark = pytest.mark.smoke


def _bilinear():
    m = Model("lps")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.subject_to(x * y >= 1.0)
    m.minimize(x + y)
    return m


def test_lp_spatial_certifies_bilinear():
    res = _bilinear().solve(lp_spatial=True, time_limit=60.0)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(2.0, abs=1e-4)
    if res.bound is not None:
        assert res.bound <= res.objective + 1e-6


def test_lp_spatial_with_cut_rounds():
    res = _bilinear().solve(lp_spatial=True, lp_spatial_cut_rounds=3, time_limit=60.0)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(2.0, abs=1e-4)


def test_lp_spatial_maximize_and_equality():
    m = Model("lpsmax")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.subject_to(x + y == 1.0)
    m.maximize(x * y)
    res = m.solve(lp_spatial=True, time_limit=60.0)
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(0.25, abs=1e-4)


def test_lp_spatial_infeasible_never_fabricates():
    # x*y >= 4 is impossible on the unit box. The LP-spatial path currently
    # burns its budget rather than proving infeasibility at the root (the
    # McCormick envelope already refutes it) — a tightness gap, not a
    # soundness one. Whatever the termination status, it must never claim a
    # solution or an optimal certificate.
    m = Model("lpsinf")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.subject_to(x * y >= 4.0)
    m.minimize(x + y)
    res = m.solve(lp_spatial=True, time_limit=3.0)
    assert res.status in ("infeasible", "time_limit", "iteration_limit")
    assert res.objective is None
