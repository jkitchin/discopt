"""#1316: a warm start must never be able to produce a false certified optimum.

Two defects found against #1273:

1. (severe, correctness) ``Model.solve(warm_start=...)`` skipped bound
   validation entirely. ``primal_point_from_result`` did no clamping (unlike
   ``validate_initial_solution``), and the incumbent-injection guards checked
   integrality and the general constraint rows but never the variable box --
   neither does Rust's ``TreeManager::inject_incumbent``, which accepts any
   ``(solution, obj)`` pair that improves. Re-solving after tightening a bound
   therefore reported ``status="optimal"``, ``gap_certified=True`` and an
   objective that is unreachable inside the model's real feasible region, at a
   point plainly outside its bounds. CLAUDE.md §1's hard gate.
2. ``solve_batch(workers>1, warm_start=<SolveResult>)`` crashed with a raw
   ``TypeError: cannot pickle 'module' object`` from inside
   ``concurrent.futures`` -- aborting the whole batch -- instead of the loud,
   actionable refusal ``_UNSENDABLE_KWARGS`` was built for.

The model is the issue's own repro. Its true optimum under the tightened box was
independently brute-forced there at -0.34 (x=0.3), strictly worse than the -0.35
the stale warm start caused to be reported.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest


def _bilinear() -> tuple[dm.Model, dm.Variable, dm.Variable]:
    """``min (x-0.9)^2 + xy - y`` over the unit box with ``x + y <= 1.5``."""
    m = dm.Model("i1316_bilinear")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize((x - 0.9) ** 2 + x * y - y)
    m.subject_to(x + y <= 1.5)
    return m, x, y


def _brute_force_optimum(x_ub: float, n: int = 2001) -> float:
    """Grid the (now smaller) box; independent of every solver path under test."""
    xs = np.linspace(0.0, x_ub, n)
    ys = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    obj = (X - 0.9) ** 2 + X * Y - Y
    obj = np.where(X + Y <= 1.5 + 1e-12, obj, np.inf)
    return float(obj.min())


@pytest.mark.smoke
def test_a_stale_warm_start_cannot_certify_a_point_outside_the_bounds():
    """The headline defect: re-solve after tightening a bound."""
    m, x, _y = _bilinear()
    first = m.solve(time_limit=30)
    assert first.status == "optimal"
    assert float(np.asarray(first.x["x"])) == pytest.approx(0.4, abs=1e-3)

    x.ub = np.array(0.3)  # the previous solution is now infeasible

    with pytest.warns(UserWarning, match="outside this model's current bounds"):
        again = m.solve(time_limit=30, warm_start=first)

    # 1. the reported point is inside the CURRENT box
    x_val = float(np.asarray(again.x["x"]))
    assert x_val <= 0.3 + 1e-6, f"reported x={x_val} violates the current bound x <= 0.3"

    # 2. and the objective is the true optimum of the tightened problem, not the
    #    stale one the warm start carried in.
    truth = _brute_force_optimum(0.3)
    assert again.objective == pytest.approx(truth, abs=1e-3), (
        f"objective {again.objective} is not the tightened problem's optimum {truth}"
    )
    assert again.objective > -0.349, "the pre-tightening objective -0.35 was reported again"

    # 3. a certified result must have a sound bound.
    if again.gap_certified and again.bound is not None:
        assert again.bound <= again.objective + 1e-5


@pytest.mark.unit
def test_primal_point_from_result_clamps_to_the_current_bounds():
    """The flattening step itself, in isolation (the issue's first pointer)."""
    from discopt.warm_start import primal_point_from_result

    m, x, _y = _bilinear()
    first = m.solve(time_limit=30)
    x.ub = np.array(0.3)

    with pytest.warns(UserWarning, match="outside this model's current bounds"):
        clamped = primal_point_from_result(m, first)
    assert clamped[0] <= 0.3 + 1e-12

    # And the escape hatch for a caller that wants the raw point back.
    raw = primal_point_from_result(m, first, clamp=False)
    assert raw[0] == pytest.approx(0.4, abs=1e-3)


@pytest.mark.unit
def test_an_in_bounds_warm_start_is_not_clamped_and_does_not_warn():
    """The guard must be inert on the ordinary case it exists to let through."""
    import warnings

    from discopt.warm_start import primal_point_from_result

    m, _x, _y = _bilinear()
    first = m.solve(time_limit=30)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        point = primal_point_from_result(m, first)
    assert point[0] == pytest.approx(float(np.asarray(first.x["x"])), abs=1e-12)


@pytest.mark.unit
def test_the_box_guard_rejects_an_out_of_box_point():
    """``_point_within_variable_box`` is the guard every injection site shares."""
    from discopt.solver import _make_evaluator, _point_within_variable_box

    m, _x, _y = _bilinear()
    ev = _make_evaluator(m)

    assert _point_within_variable_box(ev, np.array([0.4, 1.0]))
    assert _point_within_variable_box(ev, np.array([1.0 + 1e-9, 0.5]))  # within tol
    assert not _point_within_variable_box(ev, np.array([1.5, 0.5]))
    assert not _point_within_variable_box(ev, np.array([-0.5, 0.5]))

    # A guard that cannot see the box must refuse, not wave the point through.
    with pytest.raises(ValueError, match="entries but the evaluator's box"):
        _point_within_variable_box(ev, np.array([0.4]))


@pytest.mark.unit
def test_warm_start_is_refused_by_solve_batch_with_workers():
    """Part 2: a loud, actionable refusal instead of a pickling crash deep in a pool."""
    m, _x, _y = _bilinear()
    seed = m.solve(time_limit=30)

    with pytest.raises(ValueError, match="cannot forward.*warm_start|warm_start") as exc:
        dm.solve_batch([_bilinear()[0], _bilinear()[0]], workers=2, warm_start=seed)

    message = str(exc.value)
    assert "warm_start" in message
    assert "workers=1" in message, "the refusal must name the way forward"


@pytest.mark.unit
def test_warm_start_still_works_with_a_single_worker():
    """``workers=1`` runs in-process, so there is nothing to pickle and no refusal."""
    m, _x, _y = _bilinear()
    seed = m.solve(time_limit=30)
    results = dm.solve_batch([m], workers=1, time_limit=30, warm_start=seed)
    assert len(results) == 1
    assert results[0].status == "optimal"


@pytest.mark.unit
def test_an_explicit_none_is_not_refused():
    """``warm_start=None`` names no object, so it must not trip the refusal."""
    results = dm.solve_batch(
        [_bilinear()[0], _bilinear()[0]], workers=2, time_limit=30, warm_start=None
    )
    assert len(results) == 2
    assert all(r.status == "optimal" for r in results)
