"""Pre-search work is charged to ``time_limit``, and a spent budget stops it.

The 60 s large-instance sweep that followed #1921-#1929 found the same shape on
most of its 41 slow instances: every whole-model pass before the search runs to
completion whatever the clock says, and part of that work happens before
``solve_model`` even starts its clock. ``telecomsp_nor_sun`` was still classifying
at 135 s; ``parabol5_2_4`` returned at 115 s.

Three defects, one test group each:

* ``Model.solve`` handed ``solve_model`` ``time_limit`` minus only the
  convex-kernel attempt, not the rest of its own pre-work (evaluator build,
  verification snapshot).
* The deep-recursion wrapper sizes every body before ``solve_model`` anchors its
  clock, and that walk was not charged either.
* The mandatory pre-search phases (scaling diagnostics, routing, classification)
  had no gate at all, unlike the optional passes behind ``PresolveDeadline``.

Plus the count's own cost: it ran about four times per solve over the same bodies.
"""

from __future__ import annotations

import time

import discopt.modeling as dm
import discopt.solver as solver_mod
import pytest
from discopt._relax import factorable_reform as fr


def _small_nonconvex():
    m = dm.Model("budget_checkpoints")
    x = m.continuous("x", lb=-2, ub=3)
    y = m.continuous("y", lb=-2, ub=3)
    m.subject_to(x * y >= -1)
    m.subject_to(x + y <= 2.5)
    m.minimize(x * y - x + y**2)
    return m


def test_model_solve_charges_its_pre_work(monkeypatch):
    """Fails before: ``solve_model`` got the full limit after 1 s of pre-work."""
    import discopt._tape_nlp_evaluator as tne

    real_build = tne.build_evaluator
    real_solve = solver_mod.solve_model
    seen: list[float] = []

    def slow_build(*a, **k):
        time.sleep(1.0)  # stands in for a costly evaluator build
        return real_build(*a, **k)

    def spy(*a, **k):
        seen.append(k["time_limit"])
        return real_solve(*a, **k)

    monkeypatch.setattr(tne, "build_evaluator", slow_build)
    monkeypatch.setattr(solver_mod, "solve_model", spy)
    _small_nonconvex().solve(time_limit=20)
    assert seen, "solve_model was never reached, so the assertion below is vacuous"
    assert seen[0] <= 19.0, seen


def test_sizing_walk_is_charged(monkeypatch):
    """Fails before: the wrapper's sizing walk was outside the solve's clock, so
    a 0.6 s walk under a 0.5 s limit still produced a full search."""
    real_count = fr._max_expr_node_count
    reached: list[str] = []

    def slow_count(model):
        time.sleep(0.6)
        return real_count(model)

    def must_not_run(*a, **k):
        reached.append("scaling")
        raise AssertionError("a mandatory phase ran after the budget was spent")

    monkeypatch.setattr(fr, "_max_expr_node_count", slow_count)
    monkeypatch.setattr(solver_mod, "_check_model_scaling", must_not_run)
    t0 = time.perf_counter()
    res = solver_mod.solve_model(_small_nonconvex(), time_limit=0.5)
    wall = time.perf_counter() - t0
    assert reached == []
    assert res.status == "time_limit"
    assert res.objective is None and res.bound is None and res.x is None
    assert res.node_count == 0
    assert wall < 5.0, wall


def test_mandatory_phases_stop_once_the_budget_is_gone(monkeypatch):
    """Fails before: classification ran after the budget was spent."""
    reached: list[str] = []
    real_tighten = solver_mod._declared_box_tightening

    def slow_tighten(model, deadline=None):
        out = real_tighten(model, deadline=deadline)
        time.sleep(0.6)  # the budget runs out inside the box tightening
        return out

    def must_not_run(*a, **k):
        reached.append("scaling")
        raise AssertionError("a mandatory phase ran after the budget was spent")

    monkeypatch.setattr(solver_mod, "_declared_box_tightening", slow_tighten)
    monkeypatch.setattr(solver_mod, "_check_model_scaling", must_not_run)
    res = solver_mod.solve_model(_small_nonconvex(), time_limit=0.5)
    assert reached == []
    assert res.status == "time_limit"
    assert res.objective is None and res.bound is None and not res.bound_valid
    assert not res.gap_certified


def test_checkpoints_follow_the_presolve_deadline_opt_out(monkeypatch):
    """``DISCOPT_PRESOLVE_DEADLINE=0`` turns the checkpoints off with the rest of
    the pre-solve deadline, so the default can be A/B'd in one process."""
    monkeypatch.setenv("DISCOPT_PRESOLVE_DEADLINE", "0")
    reached: list[str] = []
    real_tighten = solver_mod._declared_box_tightening
    real_scaling = solver_mod._check_model_scaling

    def slow_tighten(model, deadline=None):
        out = real_tighten(model, deadline=deadline)
        time.sleep(0.6)
        return out

    def spy_scaling(model):
        reached.append("scaling")
        return real_scaling(model)

    monkeypatch.setattr(solver_mod, "_declared_box_tightening", slow_tighten)
    monkeypatch.setattr(solver_mod, "_check_model_scaling", spy_scaling)
    solver_mod.solve_model(_small_nonconvex(), time_limit=0.5)
    assert reached == ["scaling"]


def test_unspent_budget_solves_normally():
    """The checkpoints are inert while time is left: a small model certifies."""
    res = _small_nonconvex().solve(time_limit=60)
    assert res.status == "optimal"
    assert res.objective is not None


# --- the model-wide node count ----------------------------------------------


def _chain_model(n: int):
    m = dm.Model("count_cache")
    xs = [m.continuous(f"x{i}", lb=0, ub=1) for i in range(n)]
    for k in range(5):
        s = xs[0]
        for v in xs[1 : n - k]:
            s = s + v * xs[0]
        m.subject_to(s <= 3)
    m.minimize(xs[0])
    return m


def _fresh_count(m) -> int:
    largest = fr._expr_node_count(m._objective.expression)
    for body in fr._iter_constraint_bodies(m._constraints):
        largest = max(largest, fr._expr_node_count(body))
    return largest


def test_count_equals_per_body_fresh_count():
    checked = 0
    for n in (3, 40, 900):
        m = _chain_model(n)
        assert fr._max_expr_node_count(m) == _fresh_count(m)
        checked += 1
    assert checked == 3


def test_count_reuses_identical_bodies_and_recounts_new_ones(monkeypatch):
    m = _chain_model(200)
    first = fr._max_expr_node_count(m)
    calls: list[int] = []
    real = fr._expr_node_count

    def spy(expr, *a, **k):
        calls.append(id(expr))
        return real(expr, *a, **k)

    monkeypatch.setattr(fr, "_expr_node_count", spy)
    assert fr._max_expr_node_count(m) == first
    assert calls == []  # every body was sized by the previous call
    # A new, larger body is sized; the untouched ones are not.
    x0 = m._variables[0]
    big = x0
    for _ in range(600):
        big = big + x0 * x0
    m.subject_to(big <= 10)
    got = fr._max_expr_node_count(m)
    assert calls == [id(m._constraints[-1].body)]
    assert got == _fresh_count(m) > first


@pytest.mark.parametrize("depth", [1, 5, 2000])
def test_leaf_and_deep_counts(depth):
    m = dm.Model("leaf")
    x = m.continuous("x", lb=0, ub=1)
    assert fr._expr_node_count(x) == 1
    e = x
    for _ in range(depth):
        e = e + x
    assert fr._expr_node_count(e) == 2 * depth + 1
