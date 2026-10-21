"""#1852: an infeasible nonlinear ``either_or`` ended ``unknown``.

Root FBBT proves the lowered model empty -- ``x >= 8`` forces the ``x <= 3``
disjunct's selector to 0 after integer rounding, which forces ``x**2 <= 4`` --
and the Rust pass reports that as the ``Interval::empty()`` sentinel
``[+inf, -inf]`` on every block. ``_flat_fbbt_bounds`` copied finite values only,
so the proof was dropped and the convex-MINLP NLP-BB route went on to an NLP that
cannot certify infeasibility. The sentinel is now read as a root infeasibility
proof (corpus check: 0 of the 150 in-repo MINLPLib instances, all feasible, are
declared root-infeasible).
"""

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.gdp_reformulate import reformulate_gdp
from discopt.solver import _extract_variable_info
from discopt.solvers._root_presolve import tighten_root_bounds_with_fbbt

pytestmark = pytest.mark.smoke


def _model(lower):
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=10)
    m.subject_to(x >= lower)
    m.either_or([[x**2 <= 4], [x <= 3]])
    m.minimize(x)
    return m


@pytest.mark.parametrize("method", [None, "big-m", "hull", "mbigm"])
def test_infeasible_either_or_is_certified(method):
    kw = {} if method is None else {"gdp_method": method}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _model(8.0).solve(time_limit=60, **kw)
    assert r.status == "infeasible"
    assert r.gap_certified is True
    assert r.bound is None and r.x is None


@pytest.mark.parametrize("method", [None, "big-m", "hull"])
def test_feasible_twin_still_optimal(method):
    kw = {} if method is None else {"gdp_method": method}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _model(2.5).solve(time_limit=60, **kw)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(2.5, abs=1e-6)


def test_root_fbbt_reports_the_empty_sentinel():
    lowered = reformulate_gdp(_model(8.0), "big-m")
    n, lb, ub, io, isz = _extract_variable_info(lowered)
    _lb, _ub, infeasible, _ = tighten_root_bounds_with_fbbt(lowered, lb, ub, io, isz)
    assert infeasible
    feasible = reformulate_gdp(_model(2.5), "big-m")
    n, lb, ub, io, isz = _extract_variable_info(feasible)
    tlb, tub, infeasible, _ = tighten_root_bounds_with_fbbt(feasible, lb, ub, io, isz)
    assert not infeasible
    assert np.all(tlb <= tub)
