"""``solver="pounce"`` row-equilibrates its NLP arm (#1768).

POUNCE's gradient-based scaling only scales a row down, so a row with a tiny
gradient reaches the interior-point method as written. On the COPS hanging chain
the link rows ``dx**2 + dy**2 <= L**2`` have gradients ``~2L`` and multipliers
``~1/L``; the multipliers inflate POUNCE's dual scaling factor until its scaled
convergence test passes with the barrier parameter still at ``1.8e-6``. It
reported ``Solve_Succeeded`` -- ``local_optimal`` -- with every link slack.
Scaling each row up to unit gradient at the start point removes that.

From the box-midpoint start every link gradient is 0, so there is nothing to scale
by, and POUNCE stops one barrier update early again. That signature (success with
the barrier parameter and the complementarity both above ``10 * tol``) is now
continued once from the returned point, with the rows rescaled there; if the
continuation does not finish the path the route reports ``local_limit``.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solvers.nlp_pounce as nlp_pounce
import numpy as np
import pytest

pytestmark = [pytest.mark.requires_pounce]


def _chain(n: int):
    L = 1.5 / n
    m = dm.Model("chain")
    x = m.continuous("x", shape=(n + 1,), lb=-10, ub=10)
    y = m.continuous("y", shape=(n + 1,), lb=-10, ub=10)
    m.subject_to([x[0] == 0, y[0] == 0, x[n] == 1, y[n] == 0], name="ends")
    m.subject_to((x[1:] - x[:-1]) ** 2 + (y[1:] - y[:-1]) ** 2 <= L**2, name="link")
    m.minimize(0.5 * dm.sum(y[1:] + y[:-1]))
    return m, x, y, L


def _tiny_row():
    """``min (x-2)^2 + (y-2)^2`` on the disc ``x^2 + y^2 <= 2``, written with 1e-6 rows.

    Optimum ``(1, 1)``; the row's multiplier is ``1e6`` in the model's units. Solve it
    from ``_START``: the scales are taken from the row gradients at the start point,
    and at the default start ``(0, 0)`` the gradient vanishes (no scaling).
    """
    m = dm.Model("tiny")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.minimize((x - 2) ** 2 + (y - 2) ** 2)
    m.subject_to(1e-6 * (x**2 + y**2) <= 2e-6, name="disc")
    return m, x, y


def _start(x, y):
    return {x: 0.5, y: 0.5}


@pytest.fixture
def spy(monkeypatch):
    """Record the evaluator and warm start each POUNCE NLP solve receives."""
    calls: list[dict] = []
    real = nlp_pounce.solve_nlp

    def wrapped(evaluator, x0, **kw):
        calls.append({"evaluator": evaluator, "warm_start": kw.get("warm_start")})
        return real(evaluator, x0, **kw)

    monkeypatch.setattr(nlp_pounce, "solve_nlp", wrapped)
    return calls


def _equilibrated(call) -> bool:
    from discopt.solvers.oa import _RowEquilibratedEvaluator

    return isinstance(call["evaluator"], _RowEquilibratedEvaluator)


def test_chain_with_tiny_link_rows_converges(spy, monkeypatch):
    """n = 20,000 (L**2 = 5.6e-9). Before: ``local_optimal`` at -6053.7496 with every
    link 2e-6..6e-6 short, 19.6 s. After: -6053.78746, links within 1e-8, 3.8 s."""
    monkeypatch.delenv("DISCOPT_POUNCE_ROW_EQUILIBRATE", raising=False)
    n = 20_000
    m, x, y, L = _chain(n)
    r = m.solve(solver="pounce", initial_solution={x: np.linspace(0, 1, n + 1), y: np.zeros(n + 1)})
    assert len(spy) == 1 and _equilibrated(spy[0])
    assert r.status == "local_optimal"
    seg = np.hypot(np.diff(r.value(x)), np.diff(r.value(y)))
    assert np.max(np.abs(seg / L - 1)) < 1e-7, (seg.min() / L - 1, seg.max() / L - 1)
    assert r.objective < -6053.787


def test_duals_are_reported_in_the_models_units(spy, monkeypatch):
    m, x, y = _tiny_row()
    monkeypatch.setenv("DISCOPT_POUNCE_ROW_EQUILIBRATE", "0")
    ref = m.solve(solver="pounce", initial_solution=_start(x, y))
    monkeypatch.delenv("DISCOPT_POUNCE_ROW_EQUILIBRATE")
    r = m.solve(solver="pounce", initial_solution=_start(x, y))
    assert [_equilibrated(c) for c in spy] == [False, True]
    for res in (ref, r):
        assert res.status == "local_optimal"
        assert res.value(x) == pytest.approx(1.0, abs=1e-6)
        assert res.value(y) == pytest.approx(1.0, abs=1e-6)
    lam_ref = np.asarray(ref.constraint_duals["disc"], dtype=float)
    lam = np.asarray(r.constraint_duals["disc"], dtype=float)
    assert np.abs(lam_ref).max() == pytest.approx(1e6, rel=1e-4)
    np.testing.assert_allclose(lam, lam_ref, rtol=1e-4)


def test_warm_start_multipliers_are_scaled_into_the_rows_pounce_sees(spy, monkeypatch):
    monkeypatch.delenv("DISCOPT_POUNCE_ROW_EQUILIBRATE", raising=False)
    m, x, y = _tiny_row()
    first = m.solve(solver="pounce", initial_solution=_start(x, y))
    second = m.solve(solver="pounce", warm_start=first)
    assert len(spy) == 2 and all(_equilibrated(c) for c in spy)
    ws = spy[1]["warm_start"]
    assert ws is not None and ws.lagrange is not None
    s = spy[1]["evaluator"]._s
    # POUNCE sees row ``s * g``, whose multiplier is ``lam / s``.
    lam_model = np.asarray(first.constraint_duals["disc"], dtype=float).ravel()
    np.testing.assert_allclose(np.asarray(ws.lagrange) * s, lam_model, rtol=1e-12)
    assert second.status == "local_optimal"
    assert second.objective == pytest.approx(first.objective, rel=1e-8, abs=1e-10)


def test_explicit_scaling_method_and_opt_out_leave_the_rows_alone(spy, monkeypatch):
    m, x, y = _tiny_row()
    monkeypatch.delenv("DISCOPT_POUNCE_ROW_EQUILIBRATE", raising=False)
    m.solve(solver="pounce", initial_solution=_start(x, y))
    m.solve(
        solver="pounce",
        initial_solution=_start(x, y),
        pounce_options={"nlp_scaling_method": "none"},
    )
    monkeypatch.setenv("DISCOPT_POUNCE_ROW_EQUILIBRATE", "0")
    m.solve(solver="pounce", initial_solution=_start(x, y))
    # The first solve is the control: the same start does trigger the scaling.
    assert [_equilibrated(c) for c in spy] == [True, False, False]


def test_sparse_row_norms_match_the_dense_jacobian():
    """The scales are read through the sparse structure (a dense 100k-link Jacobian is
    149 GiB); they must equal the dense ones exactly, so OA (#1537) is unchanged."""
    from discopt._tape_nlp_evaluator import make_evaluator
    from discopt.solvers.oa import _jacobian_row_inf_norms

    m, x, y, _ = _chain(7)
    ev = make_evaluator(m)
    assert ev.has_sparse_structure()
    rng = np.random.default_rng(0)
    for _ in range(3):
        x0 = rng.uniform(-1, 1, ev.n_variables)
        dense = np.abs(np.asarray(ev.evaluate_jacobian(x0))).max(axis=1)
        sparse = _jacobian_row_inf_norms(ev, x0, ev.n_constraints)
        assert sparse is not None
        np.testing.assert_array_equal(sparse, dense)


def test_cold_start_chain_is_continued_to_convergence(spy, monkeypatch):
    """The same chain from the box midpoint ``x = y = 0``, where every link gradient
    is 0. Before: ``local_optimal`` at -6053.750612, barrier 1.84e-6, links 2e-6..6e-6
    short. After: the first solve still stops there; the continuation from that point
    (rows rescaled at it) takes 2 iterations to -6053.787460, links within 1e-8."""
    monkeypatch.delenv("DISCOPT_POUNCE_ROW_EQUILIBRATE", raising=False)
    n = 20_000
    m, x, y, L = _chain(n)
    r = m.solve(solver="pounce")
    assert len(spy) == 2, "the first solve must still stop mid-path, or this tests nothing"
    assert not _equilibrated(spy[0]) and spy[0]["warm_start"] is None
    assert _equilibrated(spy[1]) and spy[1]["warm_start"] is not None
    assert r.status == "local_optimal"
    seg = np.hypot(np.diff(r.value(x)), np.diff(r.value(y)))
    assert np.max(np.abs(seg / L - 1)) < 1e-7, (seg.min() / L - 1, seg.max() / L - 1)
    assert r.objective < -6053.787


def test_unfinished_midpath_solve_is_local_limit_not_local_optimal(spy, monkeypatch):
    """With the rescaling opted out, nothing can continue the path; the point POUNCE
    calls a success is reported for what it is."""
    monkeypatch.setenv("DISCOPT_POUNCE_ROW_EQUILIBRATE", "0")
    n = 20_000
    m, x, y, L = _chain(n)
    r = m.solve(solver="pounce")
    assert len(spy) == 1
    assert r.status == "local_limit"
    assert r.objective > -6053.76  # the mid-path point, 0.04 above the optimum
    assert r.bound is None and not r.gap_certified


@pytest.mark.parametrize(
    "raw, mu, compl, expected",
    [
        (0, 1.84e-6, 1.85e-6, True),  # the chain: one barrier update short
        (0, 1.84e-6, 0.0, False),  # COPS channel: equalities only, converged
        (0, 2.51e-9, 1.07e-6, False),  # COPS marine, cold start: at the barrier floor
        (0, 9.09e-10, 9.09e-10, False),  # an ordinary converged solve
        (1, 1.84e-6, 1.85e-6, False),  # acceptable level: already local_limit
    ],
)
def test_midpath_signature(raw, mu, compl, expected):
    from types import SimpleNamespace

    from discopt.solver import _pounce_midpath

    res = SimpleNamespace(
        raw_status=raw, kkt={"barrier_parameter": mu, "complementarity_unscaled": compl}
    )
    assert _pounce_midpath(res, 1e-8) is expected


# ---------------------------------------------------------------------------
# The continuation's outcomes (#1768 review). Each test runs the real first solve
# of ``_tiny_row`` (optimum (1, 1), objective 2, 1e-6 rows so the continuation can
# rescale them), stamps the mid-path signature on its KKT record so the
# continuation fires, and scripts what the second POUNCE call returns.
# ---------------------------------------------------------------------------

_MIDPATH_KKT = {"barrier_parameter": 1.84e-6, "complementarity_unscaled": 1.85e-6}


@pytest.fixture
def scripted(monkeypatch):
    """Replace the second POUNCE NLP solve with ``state["second"](first, kw)``."""
    import dataclasses

    monkeypatch.delenv("DISCOPT_POUNCE_ROW_EQUILIBRATE", raising=False)
    real = nlp_pounce.solve_nlp
    state: dict = {"calls": [], "first": None, "second": None, "after_first": None}

    def wrapped(evaluator, x0, **kw):
        import time

        state["calls"].append(kw)
        if len(state["calls"]) == 1:
            state["t_first_call"] = time.perf_counter()
            res = real(evaluator, x0, **kw)
            assert res.raw_status == 0, res.raw_status
            res.kkt = {**(res.kkt or {}), **_MIDPATH_KKT}
            state["first"] = dataclasses.replace(res, x=np.array(res.x, copy=True))
            if state["after_first"] is not None:
                state["after_first"]()
            state["t_first_done"] = time.perf_counter()
            return res
        return state["second"](state["first"], kw)

    monkeypatch.setattr(nlp_pounce, "solve_nlp", wrapped)
    return state


def _continuation(first, *, raw, status, x=None, kkt=None):
    import dataclasses

    return dataclasses.replace(
        first,
        raw_status=raw,
        status=status,
        x=np.array(first.x if x is None else x, dtype=np.float64),
        multipliers=None if first.multipliers is None else np.array(first.multipliers),
        kkt=dict(first.kkt if kkt is None else kkt),
        solve_report={"continuation_marker": True},
    )


def _assert_first_point_kept(r, x, y):
    assert r.status == "local_limit"
    assert r.value(x) == pytest.approx(1.0, abs=1e-6)
    assert r.value(y) == pytest.approx(1.0, abs=1e-6)
    assert r.objective == pytest.approx(2.0, abs=1e-6)
    assert r.bound is None and not r.gap_certified
    assert (r.solve_report or {}).get("continuation_marker") is None


def test_failed_continuation_keeps_first_point_as_local_limit(scripted):
    from discopt.solvers import SolveStatus

    scripted["second"] = lambda first, kw: _continuation(
        first, raw=-1, status=SolveStatus.ITERATION_LIMIT, x=[3.0, 3.0]
    )
    m, x, y = _tiny_row()
    r = m.solve(solver="pounce", initial_solution=_start(x, y))
    assert len(scripted["calls"]) == 2, "the continuation must have been attempted"
    _assert_first_point_kept(r, x, y)


def test_continuation_still_midpath_is_local_limit(scripted):
    from discopt.solvers import SolveStatus

    # Code 0, the same point, the signature still on it: adopted, but not finished.
    scripted["second"] = lambda first, kw: _continuation(first, raw=0, status=SolveStatus.OPTIMAL)
    m, x, y = _tiny_row()
    r = m.solve(solver="pounce", initial_solution=_start(x, y))
    assert len(scripted["calls"]) == 2
    assert r.status == "local_limit"
    assert r.solve_report.get("continuation_marker") is True  # it was adopted


def test_no_time_left_skips_the_continuation(scripted, monkeypatch):
    """Less than the continuation floor is left: no second POUNCE call (before: it was
    launched whenever any time at all was left, with ``max_wall_time = 0.1``).

    The floor is raised above the whole budget instead of sleeping the budget away,
    so the outcome does not depend on how long the model setup takes under load."""
    import discopt.solver as solver_mod

    T = 3.0
    monkeypatch.setattr(solver_mod, "_POUNCE_MIDPATH_MIN_CONTINUATION_TIME", 10 * T)
    scripted["second"] = lambda first, kw: pytest.fail("no continuation without time left")
    m, x, y = _tiny_row()
    r = m.solve(solver="pounce", initial_solution=_start(x, y), time_limit=T)
    assert len(scripted["calls"]) == 1
    _assert_first_point_kept(r, x, y)


def test_continuation_budget_never_exceeds_the_deadline(scripted, monkeypatch):
    """With the floor at 0, the continuation is given exactly what is left -- never the
    0.1 s pad the first version added on top of it.

    The deadline is the solver's own (``t_start + time_limit`` as ``_solve_continuous``
    receives them), so neither the sleep nor the bound depends on the setup time."""
    import time

    import discopt.solver as solver_mod
    from discopt.solvers import SolveStatus

    monkeypatch.setattr(solver_mod, "_POUNCE_MIDPATH_MIN_CONTINUATION_TIME", 0.0)
    real_continuous = solver_mod._solve_continuous
    deadline: list[float] = []

    def capture(model, time_limit, options, t_start, *args, **kwargs):
        deadline.append(t_start + time_limit)
        return real_continuous(model, time_limit, options, t_start, *args, **kwargs)

    monkeypatch.setattr(solver_mod, "_solve_continuous", capture)
    left = 0.05  # under the old 0.1 s pad
    scripted["after_first"] = lambda: time.sleep(max(0.0, deadline[0] - left - time.perf_counter()))
    scripted["second"] = lambda first, kw: _continuation(first, raw=0, status=SolveStatus.OPTIMAL)
    m, x, y = _tiny_row()
    m.solve(solver="pounce", initial_solution=_start(x, y), time_limit=3.0)
    assert len(deadline) == 1 and len(scripted["calls"]) == 2
    budget = scripted["calls"][1]["options"]["max_wall_time"]
    # The budget is computed after the first solve returns, so it can be at most what
    # was left then.
    bound = deadline[0] - scripted["t_first_done"]
    assert bound <= left  # so the old pad (0.1) fails the next line
    assert 0.0 < budget <= bound, (budget, bound)


@pytest.mark.parametrize(
    "point, why",
    [
        ((1.1, 1.1), "less feasible"),  # objective 1.62 < 2, but the disc is violated
        ((0.9, 0.9), "worse objective"),  # feasible, objective 2.42 > 2
    ],
)
def test_acceptable_continuation_worse_than_first_point_is_rejected(scripted, point, why):
    """Code 1 meets only the acceptable tolerances; it may not replace the first point
    with a less feasible or a worse one. Before: adopted unconditionally."""
    from discopt.solvers import SolveStatus

    scripted["second"] = lambda first, kw: _continuation(
        first, raw=1, status=SolveStatus.OPTIMAL, x=point
    )
    m, x, y = _tiny_row()
    r = m.solve(solver="pounce", initial_solution=_start(x, y))
    assert len(scripted["calls"]) == 2
    _assert_first_point_kept(r, x, y)


def test_acceptable_continuation_no_worse_is_adopted_as_local_limit(scripted):
    """The control for the test above: a code-1 point no worse on both counts is taken,
    and is still ``local_limit`` -- code 1 never certifies a local optimum."""
    from discopt.solvers import SolveStatus

    scripted["second"] = lambda first, kw: _continuation(first, raw=1, status=SolveStatus.OPTIMAL)
    m, x, y = _tiny_row()
    r = m.solve(solver="pounce", initial_solution=_start(x, y))
    assert len(scripted["calls"]) == 2
    assert r.status == "local_limit"
    assert r.solve_report.get("continuation_marker") is True
