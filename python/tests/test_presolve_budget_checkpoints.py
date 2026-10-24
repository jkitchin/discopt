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
    """Fails before: the recursion gate's sizing walk ran before ``solve_model``
    started its clock and was charged to nothing. ``Model.solve`` now sizes the
    model itself (a cache hit for the gate) and deducts the walk."""
    real_count = fr._max_expr_node_count
    real_solve = solver_mod.solve_model
    seen: list[float] = []
    slow_calls: list[int] = []

    def slow_count(model):
        slow_calls.append(1)
        if len(slow_calls) == 1:
            time.sleep(1.0)  # the first (uncached) sizing of this model
        return real_count(model)

    def spy(*a, **k):
        seen.append(k["time_limit"])
        return real_solve(*a, **k)

    monkeypatch.setattr(fr, "_max_expr_node_count", slow_count)
    monkeypatch.setattr(solver_mod, "solve_model", spy)
    from discopt.modeling.core import last_solve_prework_seconds

    _small_nonconvex().solve(time_limit=20)
    assert seen, "solve_model was never reached, so the assertion below is vacuous"
    assert seen[0] <= 19.0, seen
    assert seen[0] == 20 - last_solve_prework_seconds()


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


# --- the whole-model "is there an opaque node?" walk ---------------------------


def _square(v):
    return v * v


def test_custom_call_creation_is_counted_by_every_route():
    import copy
    import pickle

    from discopt.modeling.core import CustomCall

    m = dm.Model("cc_count")
    x = m.continuous("x", lb=0, ub=1)
    before = CustomCall._instances_created
    node = dm.custom(_square, name="sq")(x)
    assert isinstance(node, CustomCall)
    assert CustomCall._instances_created == before + 1
    copy.copy(node)
    assert CustomCall._instances_created == before + 2
    copy.deepcopy(node)
    assert CustomCall._instances_created >= before + 3
    n = CustomCall._instances_created
    pickle.loads(pickle.dumps(node))
    assert CustomCall._instances_created == n + 1


def test_no_custom_call_in_the_process_skips_the_walk(monkeypatch):
    """With none ever created, the answer is exact without walking 21k rows."""
    from discopt.modeling.core import CustomCall

    def boom(*a, **k):
        raise AssertionError("walked although no CustomCall exists")

    monkeypatch.setattr(CustomCall, "_instances_created", 0)
    monkeypatch.setattr(solver_mod, "_iter_custom_calls", boom)
    monkeypatch.setattr(solver_mod, "_expression_contains_custom_call", boom)
    m = _small_nonconvex()
    assert solver_mod._model_contains_custom_call(m) is False
    assert list(solver_mod._model_custom_calls(m)) == []


def test_a_custom_call_is_still_found_when_one_exists():
    m = dm.Model("cc_found")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    m.subject_to(dm.custom(_square, name="sq")(x) + y <= 1)
    m.minimize(y)
    assert solver_mod._model_contains_custom_call(m) is True
    assert len(list(solver_mod._model_custom_calls(m))) == 1


# --- entropy canonicalization abandons mid-run ------------------------------


def _entropy_model(n: int):
    m = dm.Model("entropy_hook")
    xs = [m.continuous(f"x{i}", lb=0.1, ub=2) for i in range(n)]
    for x in xs:
        m.subject_to(x * dm.log(x) <= 1)
    m.minimize(sum(xs))
    return m


def test_entropy_rewrite_unaffected_by_a_hook_that_never_fires():
    m = _entropy_model(4)
    a = fr.canonicalize_entropy(m)
    b = fr.canonicalize_entropy(m, lambda: False)
    assert a is not m and b is not m
    assert [str(c.body) for c in a._constraints] == [str(c.body) for c in b._constraints]
    assert any("entropy" in str(c.body) for c in b._constraints)


def test_entropy_rewrite_expiring_mid_run_returns_the_model_unchanged():
    m = _entropy_model(6)
    calls = [0]

    def hook():
        calls[0] += 1
        return calls[0] > 2  # two rows rewritten, then the budget is gone

    assert fr.canonicalize_entropy(m, hook) is m
    assert calls[0] == 3
    assert all("entropy" not in str(c.body) for c in m._constraints)


def test_solver_rewrites_forward_the_entropy_hook():
    m = _entropy_model(3)
    out, applied = solver_mod._convexity_rewrites(m, epigraph=False, entropy_deadline=lambda: True)
    assert out is m and applied == ()
    out, applied = solver_mod._convexity_rewrites(m, epigraph=False)
    assert applied == ("entropy",)


# --- batched node NLPs end by the caller's remaining budget ---------------------


def test_nlp_batch_runs_in_pool_width_chunks_and_stops_at_the_deadline(monkeypatch):
    import numpy as np

    monkeypatch.setenv("RAYON_NUM_THREADS", "2")
    calls: list[tuple[int, float]] = []

    def fake_batch(problems, x0s, options, share_structure):
        calls.append((len(problems), options["max_wall_time"]))
        time.sleep(0.3)  # the chunk uses up the budget
        return [(np.asarray(x0) + 1.0, {"status": 0, "obj_val": 1.0}) for x0 in x0s]

    problems = list(range(5))
    x0s = [np.full(2, float(i)) for i in range(5)]
    out = solver_mod._solve_nlp_batch_by_deadline(
        fake_batch,
        problems,
        x0s,
        {"max_wall_time": 30.0, "print_level": 0},
        time.perf_counter() + 0.2,
        share_structure=True,
    )
    assert len(calls) == 1  # the second chunk was never started
    assert calls[0][0] == 2 and calls[0][1] <= 0.2
    assert len(out) == 5
    assert [o[1]["status"] for o in out] == [0, 0, -4, -4, -4]
    # an unstarted problem hands back its start, like a timed-out one
    assert np.array_equal(out[3][0], x0s[3])


def test_nlp_batch_with_time_left_is_one_pass_per_chunk(monkeypatch):
    import numpy as np

    monkeypatch.setenv("RAYON_NUM_THREADS", "4")
    sizes: list[int] = []

    def fake_batch(problems, x0s, options, share_structure):
        sizes.append(len(problems))
        assert options["max_wall_time"] <= 30.0
        return [(np.asarray(x0), {"status": 0, "obj_val": 0.0}) for x0 in x0s]

    out = solver_mod._solve_nlp_batch_by_deadline(
        fake_batch,
        list(range(9)),
        [np.zeros(1)] * 9,
        {"max_wall_time": 30.0},
        time.perf_counter() + 60.0,
        share_structure=True,
    )
    assert sizes == [4, 4, 1]
    assert [o[1]["status"] for o in out] == [0] * 9


def test_the_time_out_status_is_not_accepted_as_a_node_bound():
    """The unstarted status must map to the same non-accepted class as a node
    NLP that timed out, so the caller leaves the node open."""
    from discopt.solvers import SolveStatus
    from discopt.solvers.nlp_ipopt import _IPOPT_STATUS_MAP

    status = _IPOPT_STATUS_MAP[solver_mod._NLP_NOT_STARTED_STATUS]
    assert status == SolveStatus.TIME_LIMIT
    assert status not in (SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT)
