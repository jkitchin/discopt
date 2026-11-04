"""The Rust MILP driver must read the default box (9.999e19) as no bound.

``DEFAULT_VARIABLE_BOUND = 9.999e19`` is the box of a column declared without a
bound. It sits just below the Rust LP layer's ``1e20`` infinity sentinel, so the
MILP std-form marshaling used to hand it to the driver as a FINITE bound. On the
fixture below — a GDP master from clay0203 with 6 of its 30 columns at the
default box — ``solve_milp`` with its defaults (c-MIR on) returned OPTIMAL
41907.46 with bound -752904, against a true optimum of 41420.2719511956 (HiGHS
with the box opened, presolve on and off, and SCIP agree). With
``DISCOPT_MILP_CMIR=0`` it returned ITERATION_LIMIT at 55187.8.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from discopt.constants import DEFAULT_VARIABLE_BOUND
from discopt.solvers import SolveStatus
from discopt.solvers import milp_simplex as ms

_DATA = Path(__file__).parent / "data" / "milp_default_box_clay0203_master.npz"
_TRUE_OPT = 41420.2719511956


def _load():
    d = np.load(_DATA)
    a_ub = sp.csr_matrix(
        (d["A_ub_data"], d["A_ub_ind"], d["A_ub_ptr"]), shape=tuple(d["A_ub_shape"])
    )
    bounds = list(zip(d["lb"].tolist(), d["ub"].tolist()))
    return d["c"], a_ub, d["b_ub"], bounds, d["integ"]


def test_fixture_carries_the_default_box():
    _, _, _, bounds, _ = _load()
    sides = sum(abs(v) == DEFAULT_VARIABLE_BOUND for pair in bounds for v in pair)
    assert sides == 12  # 6 columns, both sides — the defect needs these to fire


def test_open_default_box_maps_only_the_default_box():
    lb = np.array([-DEFAULT_VARIABLE_BOUND, -np.inf, -1e10, 0.0, -2e20])
    ub = np.array([DEFAULT_VARIABLE_BOUND, np.inf, 1e10, 1.0, 2e20])
    lo, hi = ms._open_default_box(lb, ub)
    np.testing.assert_array_equal(lo, [-ms._INF, -np.inf, -1e10, 0.0, -ms._INF])
    np.testing.assert_array_equal(hi, [ms._INF, np.inf, 1e10, 1.0, ms._INF])


@pytest.mark.parametrize("cmir", ["1", "0"])
def test_clay0203_master_no_false_optimum(monkeypatch, cmir):
    monkeypatch.setenv("DISCOPT_MILP_CMIR", cmir)
    c, a_ub, b_ub, bounds, integ = _load()
    r = ms.solve_milp(c, a_ub, b_ub, None, None, bounds, integ, time_limit=60.0)
    tol = 1e-6 * abs(_TRUE_OPT)
    if r.bound is not None:
        assert r.bound <= _TRUE_OPT + tol, f"dual bound {r.bound} crosses {_TRUE_OPT}"
    # Before the fix: c-MIR on -> false OPTIMAL 41907.46; c-MIR off -> ITERATION_LIMIT.
    assert r.status == SolveStatus.OPTIMAL, (r.status, r.objective, r.bound)
    assert r.objective == pytest.approx(_TRUE_OPT, rel=1e-6), (
        f"false OPTIMAL {r.objective} (true optimum {_TRUE_OPT})"
    )
    assert np.max(a_ub @ np.asarray(r.x) - b_ub) <= 1e-6


def test_loa_proposes_a_configuration_when_its_master_is_unbounded(monkeypatch):
    """With the default box opened, an OA master can run off along an unbounded
    column (measured: every master of GDPlib ``spectralog`` under LOA). LOA used
    to read that as "no point, no proof" and stop -- 0.37 s, status unknown, where
    main's finite fake box let it reach 35.41. It now re-solves as a feasibility
    master and keeps iterating; the unbounded solve publishes no bound."""
    import discopt.modeling as dm
    from discopt.gdp.algorithms import master as master_mod

    calls = {"unbounded_injected": 0, "feasibility_resolves": 0}
    orig = master_mod.MasterMILP.solve

    def solve(self, *a, **k):
        if k.get("objective_override") is None and calls["unbounded_injected"] == 0:
            calls["unbounded_injected"] += 1
            return master_mod.MasterResult(None, None, None, False, "unbounded")
        if (
            calls["unbounded_injected"]
            and k.get("objective_override") is not None
            and not np.any(k["objective_override"])
        ):
            calls["feasibility_resolves"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(master_mod.MasterMILP, "solve", solve)
    m = dm.Model("unbounded_master")
    x = m.continuous("x")  # no declared bounds: the default box
    y = m.continuous("y", lb=-3, ub=3)
    m.either_or([[y <= -1.0], [y >= 2.0]], name="side")
    m.subject_to(x >= y**2)
    m.minimize(x)
    r = m.solve(time_limit=30, gdp_method="loa")
    assert calls["unbounded_injected"] == 1
    assert calls["feasibility_resolves"] >= 1
    assert r.objective == pytest.approx(1.0, abs=1e-6)
    assert r.bound is None or r.bound <= 1.0 + 1e-6


@pytest.mark.parametrize("cmir", ["1", "0"])
def test_driver_values_integral_nodes_at_their_point_not_their_bound(monkeypatch, cmir):
    """#2136: the Rust driver must reach the optimum, and certify honestly, even
    when handed the default box as finite bounds.

    `_open_default_box` (above) removes the trigger at the marshaling layer.
    This test keeps the trigger on purpose so the driver itself is tested. With
    the box closed, the NS safe bound of an integral node came out near -752904
    for a vertex whose cᵀx is 41907.46, and the tree recorded that BOUND as the
    incumbent value. It then pruned against a value no point attains, lost the
    optimum, and returned OPTIMAL 41907.46 with bound -752904 (c-MIR on), or
    ITERATION_LIMIT at 55187.8 (c-MIR off).

    The driver now promotes an integral node with its own point's value and keeps
    the safe bound only as a floor on the dual bound (#2067's
    `import_results_with_point_values`). Measured after the fix: 41420.27 under
    both settings, uncertified, with the bound at or below the optimum."""
    monkeypatch.setattr(ms, "_open_default_box", lambda lb, ub: (lb, ub))
    monkeypatch.setenv("DISCOPT_MILP_CMIR", cmir)
    c, a_ub, b_ub, bounds, integ = _load()
    r = ms.solve_milp(c, a_ub, b_ub, None, None, bounds, integ, time_limit=60.0)
    tol = 1e-6 * abs(_TRUE_OPT)
    if r.bound is not None:
        assert r.bound <= _TRUE_OPT + tol, f"dual bound {r.bound} crosses {_TRUE_OPT}"
    assert r.objective == pytest.approx(_TRUE_OPT, rel=1e-6), (
        f"incumbent {r.objective} misses the optimum {_TRUE_OPT}: pruned against a bound"
    )
    if r.status == SolveStatus.OPTIMAL:
        assert r.bound is not None and r.objective - r.bound <= 1e-4 * abs(r.objective)
    assert np.max(a_ub @ np.asarray(r.x) - b_ub) <= 1e-6


# ── HiGHS master over the default box (highs-parity P0.4) ──────────────────


@pytest.mark.parametrize("entry", ["solve_milp", "solve_milp_with_lazy_cuts"])
def test_highs_master_no_false_optimum(entry):
    """The HiGHS master certified a false optimum over the default box.

    Found by the highs-parity P0.4 master replay: ``milp_highs.solve_milp`` returned
    OPTIMAL with bound 41573.26 (primary 41595.42; the #1634 presolve-free cross-solve
    41573.26, so the cross-check confirmed it) against the true 41420.272, a point the
    Rust engine verifies feasible. Every HiGHS master solve now opens the default box.
    """
    pytest.importorskip("highspy")
    from discopt.solvers import milp_highs

    c, a_ub, b_ub, bounds, integ = _load()
    # The lazy master with a separator that accepts every point is the plain master.
    extra = {"lazy_callback": lambda _x: []} if entry == "solve_milp_with_lazy_cuts" else {}
    r = getattr(milp_highs, entry)(
        c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, integrality=integ, time_limit=60.0, **extra
    )
    tol = 1e-6 * abs(_TRUE_OPT)
    assert r.bound is not None and r.bound <= _TRUE_OPT + tol, (r.status, r.bound)
    assert r.status == SolveStatus.OPTIMAL, (r.status, r.objective, r.bound)
    assert r.objective == pytest.approx(_TRUE_OPT, abs=tol)
