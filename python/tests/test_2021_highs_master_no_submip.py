"""#2021 section 2: switch off HiGHS's RINS/RENS sub-MIPs in the ``lp_nlp_bb`` master.

``milp_highs.solve_milp_with_lazy_cuts`` rebuilds the HiGHS tree after every lazy
cut, and HiGHS re-runs its RINS/RENS sub-MIPs on every rebuilt tree; their
incumbents are then vetoed by the separator. ``DISCOPT_HIGHS_MASTER_NO_SUBMIP``
passes ``mip_heuristic_run_rins = mip_heuristic_run_rens = False`` to the master
and to its #1634 presolve-free cross-solve. Heuristics never move a dual bound, so
the change is search-changing, not bound-changing. Entry experiment and panel:
performance-plan section 103.7.
"""

from __future__ import annotations

import numpy as np
import pytest

highspy = pytest.importorskip("highspy", reason="the HiGHS master is a core dependency")

from discopt.solvers import milp_highs  # noqa: E402
from discopt.solvers.oa import solve_lp_nlp_bb  # noqa: E402

NO_SUBMIP = {"mip_heuristic_run_rins": False, "mip_heuristic_run_rens": False}


def _tree_that_lasts():
    """The #1066/#2013 fixture: 45 seeded binaries and a separator that restarts the
    tree three times (``test_2013_master_checkin_running_bound.py``)."""
    rng = np.random.default_rng(7)
    n = 45
    c = -rng.integers(5, 60, size=n).astype(float)
    a_ub = rng.integers(1, 40, size=(6, n)).astype(float)
    b_ub = a_ub.sum(axis=1) * 0.32
    state = {"k": 0}

    def _sep(x):
        if state["k"] < 3 and x.sum() > 12 - state["k"]:
            state["k"] += 1
            return [(np.ones(n), 12.0 - state["k"])]
        return None

    return {
        "c": c,
        "A_ub": a_ub,
        "b_ub": b_ub,
        "bounds": [(0.0, 1.0)] * n,
        "integrality": np.ones(n, dtype=int),
        "lazy_callback": _sep,
    }


def _spy_options(monkeypatch):
    seen: list[tuple[int, str, object]] = []
    orig = highspy.Highs.setOptionValue

    def _set(self, key, val):
        seen.append((id(self), key, val))
        return orig(self, key, val)

    monkeypatch.setattr(highspy.Highs, "setOptionValue", _set)
    return seen


def _instances_with_no_submip(seen):
    per: dict[int, dict[str, object]] = {}
    for inst, key, val in seen:
        per.setdefault(inst, {})[key] = val
    return {i for i, opts in per.items() if all(opts.get(k) is v for k, v in NO_SUBMIP.items())}


@pytest.mark.smoke
def test_the_options_reach_the_master_and_its_cross_solve(monkeypatch):
    seen = _spy_options(monkeypatch)
    res = milp_highs.solve_milp_with_lazy_cuts(**_tree_that_lasts(), highs_options=NO_SUBMIP)
    assert res.callback_stats["restarts"] >= 1
    assert res.callback_stats["presolve_cross_check"]["ran"]
    # the reused master and the fresh presolve-free cross-solve
    assert len(_instances_with_no_submip(seen)) == 2


@pytest.mark.smoke
def test_without_options_no_sub_mip_option_is_touched(monkeypatch):
    seen = _spy_options(monkeypatch)
    milp_highs.solve_milp_with_lazy_cuts(**_tree_that_lasts())
    assert not [k for _, k, _ in seen if k in NO_SUBMIP]


@pytest.mark.smoke
def test_a_rejected_option_is_a_loud_error():
    with pytest.raises(RuntimeError, match="no_such_option"):
        milp_highs.solve_milp_with_lazy_cuts(
            **_tree_that_lasts(), highs_options={"no_such_option": 1}
        )


def _small_convex_minlp():
    import discopt.modeling as dm

    m = dm.Model("convex_mini")
    x = [m.continuous(f"x{i}", lb=-5.0, ub=5.0) for i in range(3)]
    y = [m.binary(f"y{i}") for i in range(3)]
    for i in range(3):
        m.subject_to(x[i] ** 2 <= 16.0 * y[i] + 1.0)
    m.subject_to(sum(y) <= 2)
    m.minimize(sum((x[i] - (i + 2.0)) ** 2 for i in range(3)) + sum(y))
    return m


@pytest.mark.smoke
def test_the_flag_is_on_by_default(monkeypatch):
    """Graduated on its panel (performance-plan section 103.7); ``=0`` opts out."""
    from discopt.solvers.oa import _highs_master_no_submip

    monkeypatch.delenv("DISCOPT_HIGHS_MASTER_NO_SUBMIP", raising=False)
    assert _highs_master_no_submip()
    monkeypatch.setenv("DISCOPT_HIGHS_MASTER_NO_SUBMIP", "0")
    assert not _highs_master_no_submip()


@pytest.mark.smoke
def test_the_flag_changes_the_search_not_the_answer(monkeypatch):
    monkeypatch.setenv("DISCOPT_HIGHS_MASTER_NO_SUBMIP", "0")
    seen = _spy_options(monkeypatch)
    off = solve_lp_nlp_bb(_small_convex_minlp(), time_limit=60.0, milp_solver="highs")
    assert not [k for _, k, _ in seen if k in NO_SUBMIP]
    seen.clear()
    monkeypatch.setenv("DISCOPT_HIGHS_MASTER_NO_SUBMIP", "1")
    on = solve_lp_nlp_bb(_small_convex_minlp(), time_limit=60.0, milp_solver="highs")
    assert _instances_with_no_submip(seen)
    assert off.status == "optimal" and on.status == "optimal", (off.status, on.status)
    assert on.objective == pytest.approx(off.objective, abs=1e-5)
    assert on.bound <= on.objective + 1e-6
    assert (on.solver_stats or {}).get("oa/highs_no_submip") == 1.0
    assert "oa/highs_no_submip" not in (off.solver_stats or {})
