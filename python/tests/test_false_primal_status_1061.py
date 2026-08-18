"""The false-primal guard must not leave the status claiming optimality (#1061).

Found by the §5 graduation panel for ``DISCOPT_PRESOLVE_BOUND_PROPAGATION``.
On ``nvs05`` with that flag on, the guard fired correctly -- the incumbent was
infeasible in the original problem, so it was withheld and decertified -- but
``status`` stayed ``"optimal"``. The returned object therefore read

    status='optimal'  objective=None  x=None  gap_certified=False

which is a *proven optimum with no solution*: every gate, panel and script in
this repo keys on ``status``, so the withdrawal was invisible to all of them.
Reproduced 3/3 interleaved reps, so it is a code path, not a timing artifact.

These tests drive the guard directly rather than through that flag, because the
invariant belongs to the guard and must hold whatever made the incumbent bad
(CLAUDE.md §2: fix the class, not the instance).
"""

import inspect
import warnings

import numpy as np
import pytest
from discopt import Model


def _nonlinear_model(name="false_primal_guard"):
    """A model that takes the constrained nonlinear path, so the guard is armed.

    The guard only builds its verification snapshot for a model with constraints
    that is not in the fast linear/quadratic family; on a linear model the code
    under test is never reached and every assertion below would pass vacuously
    (CLAUDE.md §6).
    """
    m = Model(name)
    x = m.continuous("x", lb=0.1, ub=4.0)
    y = m.continuous("y", lb=0.1, ub=4.0)
    z = m.integer("z", lb=0, ub=3)
    # The ``- z`` matters: the posynomial version of this model classifies as a
    # geometric program and is solved by ``solve_gp`` through a *nested*
    # ``Model.solve`` on the log-space model, so the guard fires on the inner
    # result and these assertions would inspect the wrong object.
    m.subject_to(x * y * y - z >= 1.0)
    m.subject_to(x + y + z <= 6.0)
    m.minimize(x + y + 0.5 * z)
    return m


def _force_guard_verdict(monkeypatch, feasible: bool):
    """Force the GUARD's feasibility verdict, leaving every other caller real.

    Patching ``_check_constraint_feasibility`` (~35 call sites) wholesale also
    rewires the solver's own primal screens, which withhold the incumbent earlier
    and leave the guard nothing to act on -- the first draft of this test did
    exactly that and never reached the code under test. ``passes_false_primal_screen``
    is the narrow entry point instead: only the final guard and the single-NLP
    source screen call it, and dispatching on the calling frame keeps the latter
    real, so the solver still screens its own points normally.

    Returns a dict whose ``n`` counts guard calls actually intercepted (§6).
    """
    import discopt._relax.primal_heuristics as ph

    real = ph.passes_false_primal_screen
    calls = {"n": 0}
    here = "modeling/core.py"

    def _dispatch(evaluator, x):
        caller = inspect.currentframe().f_back
        if caller is not None and caller.f_code.co_filename.replace("\\", "/").endswith(here):
            calls["n"] += 1
            return feasible
        return real(evaluator, x)

    monkeypatch.setattr(ph, "passes_false_primal_screen", _dispatch)
    return calls


def _solve_with_guard_verdict(monkeypatch, feasible: bool):
    calls = _force_guard_verdict(monkeypatch, feasible)
    res = _nonlinear_model().solve(time_limit=30)
    # §6: without this, "status is not optimal" could pass because the solve
    # failed for an unrelated reason and the guard never ran at all.
    assert calls["n"] > 0, "the false-primal guard never ran; this test proves nothing"
    return res


def test_guard_runs_and_passes_a_genuine_incumbent(monkeypatch):
    """Control arm: verdict 'feasible' withholds nothing."""
    res = _solve_with_guard_verdict(monkeypatch, feasible=True)
    assert res.incumbent_verification_failed is False
    assert res.x is not None
    assert res.objective is not None
    assert res.status != "error"


def test_withheld_false_primal_does_not_report_optimal(monkeypatch):
    """The regression. Before the fix this returned status='optimal'."""
    res = _solve_with_guard_verdict(monkeypatch, feasible=False)

    assert res.incumbent_verification_failed is True
    assert res.x is None
    assert res.objective is None
    assert res.gap is None
    assert res.gap_certified is False
    assert res.status != "optimal", (
        "a withheld false primal was still reported as status='optimal' with no "
        "incumbent -- a proven optimum that does not exist"
    )
    assert res.status == "error"


def test_the_dual_bound_survives_the_withhold(monkeypatch):
    """Only the PRIMAL was shown invalid; the dual bound stays rigorous."""
    ok = _solve_with_guard_verdict(monkeypatch, feasible=True)
    monkeypatch.undo()
    bad = _solve_with_guard_verdict(monkeypatch, feasible=False)
    assert ok.bound is not None, "control arm produced no dual bound; nothing to compare"
    assert bad.bound is not None, "the withhold must not discard the valid dual bound"
    assert bad.bound == pytest.approx(ok.bound, rel=1e-6, abs=1e-9)


def test_a_broken_guard_is_loud_rather_than_silent(monkeypatch):
    """CLAUDE.md §7: a swallowed exception here DELETES the soundness guard.

    It was logged at DEBUG, so a solve whose incumbent was never screened looked
    exactly like one that passed screening.
    """
    import discopt._relax.primal_heuristics as ph

    real = ph.passes_false_primal_screen
    seen = {"n": 0}

    def _explode(evaluator, x):
        caller = inspect.currentframe().f_back
        if caller is not None and caller.f_code.co_filename.replace("\\", "/").endswith(
            "modeling/core.py"
        ):
            seen["n"] += 1
            raise RuntimeError("guard is broken")
        return real(evaluator, x)

    monkeypatch.setattr(ph, "passes_false_primal_screen", _explode)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = _nonlinear_model().solve(time_limit=30)

    assert seen["n"] > 0, "the guard never ran; this test proves nothing"
    # The solve is still returned -- a broken checker must not break a good solve.
    assert res is not None
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("unscreened" in m for m in msgs), (
        f"a guard that raised produced no RuntimeWarning; warnings seen: {msgs}"
    )


def test_status_error_is_reachable_only_through_the_guard():
    """Guard against a hardwired pass: an unpatched solve of this model is fine."""
    res = _nonlinear_model("false_primal_control").solve(time_limit=30)
    assert res.status != "error"
    assert res.incumbent_verification_failed is False
    assert isinstance(res.objective, float) and np.isfinite(res.objective)


def test_a_gp_does_not_hide_the_guards_verdict(monkeypatch):
    """``solve_gp`` must propagate the withhold flag, not just the status.

    A posynomial model is routed to ``solve_gp``, which runs a *nested*
    ``Model.solve`` on the log-space model. The guard therefore fires on the
    inner result, and the x-space result handed back to the caller is rebuilt
    field by field -- so the flag has to be carried across explicitly or a
    detected false primal reads as an ordinary empty result.
    """
    m = Model("false_primal_gp")
    x = m.continuous("x", lb=0.1, ub=4.0)
    y = m.continuous("y", lb=0.1, ub=4.0)
    m.subject_to(x * y * y >= 1.0)
    m.subject_to(x + y <= 5.0)
    m.minimize(x + y)

    calls = _force_guard_verdict(monkeypatch, feasible=False)
    res = m.solve(time_limit=30)
    assert calls["n"] > 0, "the guard never ran; this test proves nothing"

    assert res.x is None
    assert res.objective is None
    assert res.status != "optimal"
    assert res.incumbent_verification_failed is True, (
        "solve_gp dropped the false-primal flag, so the caller cannot tell a "
        "withheld unsound incumbent from an ordinary no-solution result"
    )
