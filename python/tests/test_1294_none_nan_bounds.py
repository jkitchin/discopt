"""A ``None`` or NaN variable bound must never reach the solver (#1294).

``np.asarray(None, dtype=float)`` is NaN, so ``continuous("x", ub=None)`` stored
a NaN bound and a trivially feasible model was certified ``infeasible``.
``None`` now means "no bound" (as it already did for ``integer`` and indexed
variables), and a NaN bound is refused at creation.
"""

import discopt.modeling as dm
import numpy as np
import pytest


@pytest.mark.parametrize("kw", [{"ub": None}, {"lb": None}, {"lb": None, "ub": None}])
def test_none_bound_means_unbounded(kw):
    m = dm.Model("t1294_none")
    x = m.continuous("x", **kw)
    assert not np.isnan(x.lb).any() and not np.isnan(x.ub).any()
    ref = m.continuous("ref")
    if "lb" in kw:
        assert np.array_equal(x.lb, ref.lb)
    if "ub" in kw:
        assert np.array_equal(x.ub, ref.ub)


def test_ub_none_model_solves_to_its_optimum():
    m = dm.Model("t1294_solve_ub")
    x = m.continuous("x", lb=0.0, ub=None)
    m.subject_to(x >= 1)
    m.subject_to(x <= 3)
    m.minimize(x)
    r = m.solve()
    assert r.status == "optimal"
    assert r.objective == pytest.approx(1.0, abs=1e-6)


def test_lb_none_lp_route_solves():
    m = dm.Model("t1294_solve_lb")
    x = m.continuous("x", lb=None, ub=5.0)
    m.subject_to(x >= -2)
    m.minimize(x)
    r = m.solve()
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-2.0, abs=1e-6)


@pytest.mark.parametrize(
    "kw",
    [
        {"lb": float("nan")},
        {"ub": np.nan},
        {"shape": (3,), "ub": np.array([1.0, np.nan, 2.0])},
        {"shape": (2,), "lb": [0.0, None]},
    ],
)
def test_nan_bound_is_refused(kw):
    m = dm.Model("t1294_nan")
    with pytest.raises(ValueError, match="NaN"):
        m.continuous("x", **kw)
    # nothing half-registered
    assert m._variables == []


def test_nan_integer_bound_is_refused():
    m = dm.Model("t1294_nan_int")
    with pytest.raises(ValueError, match="NaN"):
        m.integer("n", lb=0, ub=float("nan"))


def test_nan_indexed_bound_is_refused():
    m = dm.Model("t1294_nan_idx")
    s = dm.Set("S", ["a", "b"])
    with pytest.raises(ValueError, match="NaN"):
        m.continuous("x", over=s, ub={"a": 1.0, "b": float("nan")})
