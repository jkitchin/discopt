"""Regression tests for #1678 (certificate, route and warning items).

* (d) LOA: the NLP subproblem's incumbent sat ~1e-5 outside a row at large data
  scale, so the gap against the (valid) master bound read as "crossed" and the
  solve published ``feasible`` -- or, at ``U = 1e4``, a certificate that
  ``Model.solve`` then withheld as UNVERIFIED. The incumbent is now repaired and
  re-verified before it is scored. Before the fix: ``U = 100`` -> ``feasible``,
  ``U = 1e4`` -> ``certificate/incumbent_unverified``; big-M and hull certify the
  same model at every scale.
* (g)/(d) routes: AMP, GDPopt-LOA and the convex kernel return from
  ``Model.solve`` before ``solve_model`` and published ``algorithm_route=None``.
* (e) a solve the native spatial kernel cannot honour (``strategy=``,
  ``node_callback=``) silently moved to the Python tree; it now warns.
* (f) the #815 withhold warning now points at ``result.last_iterate``.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest


def _plant(U: float) -> dm.Model:
    """Convex two-unit process-selection GDP whose data scale with ``U``."""
    m = dm.Model("plant")
    x = m.continuous("x", lb=0, ub=U)
    f = m.continuous("f", lb=0, ub=U)
    c = m.continuous("c", lb=0, ub=10 * U)
    m.either_or(
        [
            [f >= dm.exp(x / U * 2) * 0.3 * U / 3, c == 2.5 * f + 30],
            [f >= 0.5 * x + 0.02 * x**2 / U, c == 1.8 * f + 75],
        ],
        name="unit",
    )
    m.subject_to(x >= 0.4 * U)
    m.minimize(c + 3 * x)
    return m


@pytest.mark.parametrize("U", [100.0, 10000.0])
def test_loa_certifies_where_big_m_does(U):
    ref = _plant(U).solve(gdp_method="big-m", time_limit=60)
    assert ref.status == "optimal", "the big-M reference no longer certifies -- repro drifted"

    r = _plant(U).solve(gdp_method="loa", time_limit=60)
    assert r.status == "optimal", (
        f"LOA published {r.status!r} at U={U} where big-M certifies {ref.objective}"
    )
    assert r.gap_certified
    assert r.objective == pytest.approx(ref.objective, rel=1e-5, abs=1e-6)
    # Soundness: LOA's bound must not exceed big-M's certified optimum.
    assert r.bound <= ref.objective + 1e-6 * max(1.0, abs(ref.objective))
    stats = r.solver_stats or {}
    assert not any("incumbent_unverified" in str(k) for k in stats), stats
    assert r.algorithm_route is not None and r.algorithm_route.startswith("gdpopt-loa")


def test_amp_names_its_route():
    m = dm.Model("amp_route")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    b = m.binary("b")
    m.subject_to(x * y >= 2)
    m.subject_to(x + y <= 3 + b)
    m.minimize(x**2 - y + 3 * b)
    r = m.solve(solver="amp", time_limit=30)
    assert r.algorithm_route is not None and r.algorithm_route.startswith("amp"), r.algorithm_route


def test_convex_kernel_names_its_route():
    from discopt.solvers._convex_kernel import _attempt_convex_solve, build_convex_spec

    m = dm.Model("ck_route")
    x = m.integer("x", lb=-5, ub=5)
    z = m.continuous("z", lb=-10, ub=10)
    m.subject_to(z >= (x - 1.3) ** 2)
    m.minimize(z + 0.1 * x)
    assert build_convex_spec(m) is not None, "not kernel-eligible -- repro drifted"
    r = _attempt_convex_solve(m, time_limit=30.0, gap_tolerance=1e-4)
    assert r is not None, "the kernel declined, so this test measures nothing"
    assert r.algorithm_route is not None and r.algorithm_route.startswith("convex-kernel")


def _bilinear() -> dm.Model:
    m = dm.Model("w")
    x = m.continuous("x", lb=-2, ub=2)
    y = m.continuous("y", lb=-2, ub=2)
    m.subject_to(x * y >= -1)
    m.minimize(x * y + x - y)
    return m


def _bypass_warnings(**kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = _bilinear().solve(time_limit=30, **kw)
    return r, [str(i.message) for i in w if "native spatial" in str(i.message)]


def test_strategy_moving_solve_to_python_tree_warns(monkeypatch):
    monkeypatch.delenv("DISCOPT_NATIVE_SPATIAL_KERNEL", raising=False)
    r, msgs = _bypass_warnings(strategy="depth_first")
    assert msgs, "strategy= moved the solve off the native kernel without a warning"
    assert "strategy" in msgs[0] and "Python spatial tree" in msgs[0]
    assert r.algorithm_route is not None and "Python tree" in r.algorithm_route


def test_plain_solve_does_not_warn(monkeypatch):
    monkeypatch.delenv("DISCOPT_NATIVE_SPATIAL_KERNEL", raising=False)
    _, msgs = _bypass_warnings()
    assert msgs == []


def test_kernel_opt_out_does_not_warn(monkeypatch):
    # With the kernel disabled the Python tree is what the user asked for.
    monkeypatch.setenv("DISCOPT_NATIVE_SPATIAL_KERNEL", "0")
    _, msgs = _bypass_warnings(strategy="depth_first")
    assert msgs == []
