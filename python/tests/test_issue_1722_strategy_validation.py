"""#1722: ``solve(strategy=...)`` is validated, and a route that ignores it says so.

Before the fix ``strategy`` was checked only by ``PyTreeManager``, so a route that
builds no discopt tree -- the default HiGHS MILP route, the LP entry solve, the
monolithic Rust MILP engine -- accepted ``strategy="nonsense_xyz"`` and returned
``optimal`` as if the value had been applied.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest
from discopt.solver import solve_model

pytestmark = pytest.mark.smoke


def _knapsack() -> dm.Model:
    m = dm.Model("knap")
    v, w = [9, 14, 13, 11], [4, 6, 5, 3]
    x = m.binary("x", shape=(4,))
    m.subject_to(sum(w[i] * x[i] for i in range(4)) <= 10)
    m.maximize(sum(v[i] * x[i] for i in range(4)))
    return m


def _lp() -> dm.Model:
    m = dm.Model("lp")
    x = m.continuous("x", lb=0, ub=4)
    m.minimize(-x)
    return m


def _minlp() -> dm.Model:
    m = dm.Model("minlp")
    x = m.continuous("x", lb=-2, ub=3)
    y = m.integer("y", lb=0, ub=4)
    m.subject_to(x * x + y >= 2.5)
    m.minimize(x * x * x - x * y + y)
    return m


def _strategy_warnings(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fn()
    return result, [w for w in caught if "strategy" in str(w.message)]


@pytest.mark.parametrize("build", [_knapsack, _lp, _minlp])
def test_unknown_strategy_raises_on_every_model_class(build):
    with pytest.raises(ValueError, match="strategy='nonsense_xyz'"):
        build().solve(strategy="nonsense_xyz")


def test_unknown_strategy_raises_positionally_through_solve_model():
    # ``strategy`` is the 8th parameter of ``solve_model``; the check must not
    # depend on it being passed by keyword.
    with pytest.raises(ValueError, match="expected one of"):
        solve_model(_knapsack(), 60.0, 1e-4, None, 1, False, 16, "breadth_first")


def test_highs_milp_route_warns_that_strategy_is_unused():
    result, ws = _strategy_warnings(lambda: _knapsack().solve(strategy="depth_first"))
    assert result.status == "optimal" and result.objective == pytest.approx(25.0)
    assert result.algorithm_route.startswith("highs-milp")
    assert len(ws) == 1
    msg = str(ws[0].message)
    assert "strategy='depth_first' was not used" in msg and "highs-milp" in msg
    # Attributed to the caller's line, not to a frame inside discopt.
    assert ws[0].filename == __file__


def test_lp_route_warns_that_strategy_is_unused():
    result, ws = _strategy_warnings(lambda: _lp().solve(strategy="best_estimate"))
    assert result.objective == pytest.approx(-4.0)
    assert len(ws) == 1 and "was not used" in str(ws[0].message)


def test_rust_engine_route_warns_that_strategy_is_unused(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    result, ws = _strategy_warnings(lambda: _knapsack().solve(strategy="depth_first"))
    assert result.objective == pytest.approx(25.0)
    assert [w for w in ws if "was not used" in str(w.message)]


@pytest.mark.parametrize("strategy", ["depth_first", "best_estimate"])
def test_native_milp_tree_honours_strategy_without_warning(strategy):
    result, ws = _strategy_warnings(
        lambda: _knapsack().solve(strategy=strategy, milp_backend="native")
    )
    assert result.objective == pytest.approx(25.0)
    assert f"strategy={strategy!r}" in result.algorithm_route
    assert not [w for w in ws if "was not used" in str(w.message)]


def test_spatial_tree_consumes_strategy_without_unused_warning():
    result, ws = _strategy_warnings(lambda: _minlp().solve(strategy="depth_first"))
    assert result.status == "optimal"
    assert not [w for w in ws if "was not used" in str(w.message)]


def test_default_strategy_never_warns():
    _, ws = _strategy_warnings(lambda: _knapsack().solve())
    assert not ws
    _, ws = _strategy_warnings(lambda: _knapsack().solve(strategy="best_first"))
    assert not ws


def test_route_that_already_declares_strategy_ignored_warns_once():
    # solver='direct' lists ``strategy`` in its own "ignores" warning; the generic
    # #1722 warning must not repeat it.
    m = dm.Model("d")
    x = m.continuous("x", lb=-1, ub=2)
    m.minimize((x - 0.5) ** 2)
    _, ws = _strategy_warnings(
        lambda: m.solve(solver="direct", strategy="depth_first", max_evals=50)
    )
    assert len(ws) == 1 and "solver='direct' ignores" in str(ws[0].message)
