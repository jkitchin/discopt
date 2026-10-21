"""#1775 G4.1: LOA on the GDP framework.

Before: ``solve_gdpopt_loa`` was its own loop over the big-M lowering with a
local NLP per configuration; ``discopt.gdp.algorithms.loa`` did not exist.

After: ``LOA(GDPAlgorithm)``. ``subproblem="nlp"`` is the old loop (bound-neutral:
the 8 panel instances the old LOA solved, solved with the same objectives);
``subproblem="global"`` (the default) certifies each configuration with discopt
and falls back to the NLP step when that proves nothing. Measured on the
gdp_parity panel at 60 s: nlp 8 solved, global 12 (the same 8 plus modprodnet,
constrained_layout_clay0203, ex_633_trespalacios, nonconvex_hen), 0 incorrect.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest
from discopt.gdp.algorithms.loa import LOA
from discopt.gdp.tree import gdp_tree

pytestmark = pytest.mark.smoke


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(*a, **k)


def _nonconvex():
    """min x + y over [x^2 + y^2 >= 9] v [x + y >= 5], x, y in [0, 5].

    The first disjunct is nonconvex: its global optimum is 3 at (3, 0), while a
    local NLP from the box midpoint stops at (2.12, 2.12), 4.24.
    """
    m = dm.Model("nonconvex")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.either_or([[x**2 + y**2 >= 9], [x + y >= 5]], name="d")
    m.minimize(x + y)
    return m


def test_global_subproblems_certify_a_nonconvex_disjunct():
    r = _quiet(LOA(_nonconvex(), time_limit=60).run)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(3.0, abs=1e-5)
    assert r.bound <= r.objective + 1e-6


def test_nlp_subproblems_never_claim_the_local_point():
    r = _quiet(LOA(_nonconvex(), time_limit=60, subproblem="nlp").run)
    assert r.status != "optimal" or r.objective == pytest.approx(3.0, abs=1e-5)
    if r.objective is not None:
        assert r.objective >= 3.0 - 1e-6


def test_global_falls_back_to_the_nlp_step(monkeypatch):
    """A global solve that proves nothing (here: forced) must not lose the
    configuration the NLP step solves (measured: small_batch, positioning)."""
    import discopt.gdp.algorithms.subproblem as sp

    calls = []

    def nothing(sub, limit, tol):
        calls.append(1)
        return sp.SubproblemResult("unresolved", reason="forced")

    monkeypatch.setattr(sp, "_global", nothing)
    m = dm.Model("convex")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 2], [x >= 5]], name="d")
    m.minimize((x - 3) ** 2)
    r = _quiet(LOA(m, time_limit=60).run)
    assert calls, "the global path never ran"
    assert r.status == "optimal" and r.objective == pytest.approx(1.0, abs=1e-6)


def test_a_point_is_offered_when_the_lowering_adds_columns():
    """Logical constraints add auxiliary columns to the lowering; a certified
    point used to be dropped (neither offered nor cut) in that case."""
    m = dm.Model("logic")
    x = m.continuous("x", lb=0, ub=10)
    d = m.either_or([[x <= 2], [x >= 5]], name="d")
    e = m.either_or([[x <= 7], [x >= 8]], name="e")
    m.logical(d.indicators[1] | e.indicators[1])  # x >= 5 or x >= 8
    m.minimize((x - 3) ** 2)
    r = _quiet(LOA(m, time_limit=60).run)
    assert r.status == "optimal" and r.objective == pytest.approx(4.0, abs=1e-6)  # x = 5


def test_empty_disjunct_indicator_is_named():
    """Pyomo's "do nothing" disjunct emits no row; the tree now reads its
    indicator from the selector row (was None, which GDPSpace refused)."""
    m = dm.Model("empty")
    x = m.continuous("x", lb=0, ub=10)
    a = m.make_disjunct("A")
    a.subject_to(x >= 4)
    b = m.make_disjunct("B")
    m.add_disjunction([a, b], name="d")
    m.minimize(x + 3 * a.indicator.variable)
    (node,) = list(gdp_tree(m).walk())
    assert [d.indicator.variable.name for d in node.disjuncts] == ["A_active", "B_active"]
    r = _quiet(LOA(m, time_limit=60).run)
    assert r.status == "optimal" and r.objective == pytest.approx(0.0, abs=1e-6)


def test_the_deadline_reaches_the_nlp_calls(monkeypatch):
    import discopt.solvers.gdpopt_loa as gl

    seen = []
    real_sub, real_rel = gl._solve_nlp_subproblem, gl._solve_nlp_relaxation

    def sub(*a, **k):
        seen.append(("sub", k.get("time_limit")))
        return real_sub(*a, **k)

    def rel(*a, **k):
        seen.append(("rel", k.get("time_limit")))
        return real_rel(*a, **k)

    monkeypatch.setattr(gl, "_solve_nlp_subproblem", sub)
    monkeypatch.setattr(gl, "_solve_nlp_relaxation", rel)
    _quiet(LOA(_nonconvex(), time_limit=20, subproblem="nlp").run)
    assert {kind for kind, _ in seen} == {"sub", "rel"}
    assert all(t is not None and 0 < t <= 20 for _, t in seen)


def test_convexity_slice_is_role2_and_yields_under_deterministic(monkeypatch):
    """#912/#1116: classification gets half of what is left, and under
    ``deterministic`` the solve's own role-1 deadline instead, so the OA cut set
    stops depending on machine speed short of the wall."""
    import dataclasses
    import time

    import discopt._relax.convexity as conv
    import discopt.solver as S

    seen = []
    real = conv.classify_oa_cut_convexity

    def spy(model, **k):
        seen.append(k.get("deadline"))
        return real(model, **k)

    monkeypatch.setattr(conv, "classify_oa_cut_convexity", spy)
    t0 = time.perf_counter()
    _quiet(LOA, _nonconvex(), time_limit=20)
    assert len(seen) == 1 and t0 + 9 <= seen[0] <= time.perf_counter() + 10

    orig = S._tuning
    monkeypatch.setattr(S, "_tuning", lambda: dataclasses.replace(orig(), deterministic=True))
    wall = time.perf_counter() + 1000.0
    token = S._ROLE1_DEADLINE.set(wall)
    try:
        _quiet(LOA, _nonconvex(), time_limit=20)
    finally:
        S._ROLE1_DEADLINE.reset(token)
    assert len(seen) == 2 and seen[1] == wall


def test_gdp_options_drive_loa():
    r = _quiet(
        _nonconvex().solve,
        gdp_method="loa",
        gdp_options={"subproblem": "nlp", "init_algorithm": "set_covering", "iterlim": 5},
        time_limit=60,
    )
    assert r.solver_stats["iterations"] <= 5
    r = _quiet(_nonconvex().solve, gdp_method="loa", time_limit=60)
    assert r.status == "optimal" and r.objective == pytest.approx(3.0, abs=1e-5)
