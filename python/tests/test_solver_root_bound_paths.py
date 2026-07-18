"""Functional coverage of root-bound strengthening and heuristic options (#87).

Fourth battery: a 8-variable doubly-bilinear MINLP whose optimum is known in
closed form (2 + 2*sqrt(2), all binaries 0) solved under the opt-in root
strengthening / heuristic flags. Every option must reproduce the same
certificate — these flags are tightenings and heuristics, never allowed to
change the certified value.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt.modeling.core import Model

pytestmark = pytest.mark.smoke

_OPT = 2.0 + 2.0 * np.sqrt(2.0)  # x=y=1, u=v=sqrt(2), b=0


def _double_bilinear_model():
    m = Model("dbl")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    u = m.continuous("u", lb=0.0, ub=4.0)
    v = m.continuous("v", lb=0.0, ub=4.0)
    b = m.binary("b", shape=(4,))
    m.subject_to(x * y >= 1.0)
    m.subject_to(u * v >= 2.0)
    m.subject_to(x + u <= 3.0 + b[0] + 2.0 * b[1])
    m.subject_to(y + v <= 3.0 + b[2] + 2.0 * b[3])
    m.minimize(x + y + u + v + b[0] + b[1] + b[2] + b[3])
    return m


def _assert_certified(res, abs_tol=1e-3):
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(_OPT, abs=abs_tol)
    if res.bound is not None:
        # min sense: the dual bound can never exceed the incumbent objective.
        assert res.bound <= res.objective + 1e-6


def test_psd_and_rlt_root_cuts():
    res = _double_bilinear_model().solve(psd_cuts=True, rlt=True, time_limit=60.0)
    _assert_certified(res)


def test_eigenvalue_root_bound():
    res = _double_bilinear_model().solve(eigenvalue_root_bound=True, time_limit=60.0)
    _assert_certified(res)


def test_lagrangian_root_bound():
    res = _double_bilinear_model().solve(lagrangian_bound=True, time_limit=60.0)
    _assert_certified(res)


def test_lns_disabled_recursion_guard():
    # _lns_enabled=False is the sub-MIP recursion guard; the certificate is
    # unchanged with the improvement heuristics off.
    res = _double_bilinear_model().solve(_lns_enabled=False, time_limit=60.0)
    _assert_certified(res)


def test_in_tree_presolve_stride():
    res = _double_bilinear_model().solve(in_tree_presolve_stride=1, time_limit=60.0)
    _assert_certified(res)


def test_obbt_at_root_spatial():
    res = _double_bilinear_model().solve(obbt_at_root=True, time_limit=90.0)
    _assert_certified(res)


def test_subnlp_on_iterating_model():
    res = _double_bilinear_model().solve(
        subnlp_enabled=True, subnlp_frequency=1, subnlp_max_calls=8, time_limit=90.0
    )
    _assert_certified(res)


def test_rens_on_iterating_model():
    res = _double_bilinear_model().solve(rens=True, time_limit=90.0)
    _assert_certified(res)


def test_batch_size_on_iterating_model():
    res = _double_bilinear_model().solve(batch_size=8, time_limit=90.0)
    _assert_certified(res)


def test_max_nodes_terminates_early_without_false_certificate():
    res = _double_bilinear_model().solve(max_nodes=3, time_limit=60.0)
    # With only 3 nodes the solver may or may not close the gap, but it must
    # never report a bound above a feasible incumbent, and any incumbent must
    # be >= the true optimum (min sense).
    if res.objective is not None:
        assert res.objective >= _OPT - 1e-6
    if res.status == "optimal":
        assert res.objective == pytest.approx(_OPT, abs=1e-3)
