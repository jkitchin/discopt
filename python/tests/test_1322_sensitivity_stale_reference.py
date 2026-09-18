"""#1322: ``matches_reference=True`` must mean the derivatives ARE the reference solution.

#1313 made ``sensitivity()`` warm-start from ``Model._last_solve_result`` and
cross-check its KKT point against that result. Nothing invalidated the recorded
result when the model changed, and the cross-check compared objective *values*
only -- so after a parameter change moved the global optimum to another well,
the call warm-started into the old well, found the old objective, and reported
``matches_reference=True``. That is the #1313 failure again, with a stamp saying
it had been checked.

The fix fingerprints the problem state with the recorded result: structure
(objective, constraints, variables, parameters), variable bounds, and parameter
values. A reference that does not match the model as it is now is stale; a stale
reference is neither warm-started from nor allowed to report a match.
"""

import warnings

import discopt.modeling as dm
import numpy as np
import pytest


def _double_well(name="i1322_dw"):
    """The issue's model: a deep well at x=-0.5 and a ``p``-scaled one at x=3.

    At ``p=0`` the global is the deep well (obj -5.005). At ``p=10`` it is the
    other one (obj -11.970, per the issue's 6e5-point brute-force oracle).
    """
    m = dm.Model(name)
    p = m.parameter("p", value=0.0)
    x = m.continuous("x", lb=-1.0, ub=5.0)
    m.minimize(
        -5 * dm.exp(-((x + 0.5) ** 2) / 0.18) - (2 + p) * dm.exp(-((x - 3) ** 2) / 0.18) + 0.01 * x
    )
    return m, p, x


# ── the headline: a parameter change moves the global optimum ────────────


def test_a_parameter_change_since_the_solve_makes_the_reference_stale():
    """The issue's repro verbatim. Before the fix: x=-0.50018, objective -5.005,
    ``matches_reference=True``, no warning -- while the global at ``p=10`` is
    x≈3.0 with objective -11.970."""
    m, p, _x = _double_well()
    r = m.solve()
    assert r.objective == pytest.approx(-5.005, abs=1e-2)

    p.value = 10.0
    with pytest.warns(RuntimeWarning, match="DIFFERENT problem"):
        s = m.sensitivity()

    assert s.matches_reference is False
    # Not warm-started into the old basin any more: the stale point is not used.
    assert float(np.asarray(s.x).ravel()[0]) == pytest.approx(3.0, abs=0.2)
    assert s.objective == pytest.approx(-11.970, abs=1e-2)


def test_a_stale_reference_is_never_reported_as_a_match_even_when_the_objectives_agree():
    """The mechanism, isolated: the whole defect is that two objective values
    agreeing across a change of problem meant nothing, and was reported as a
    match anyway."""
    m = dm.Model("i1322_shift")
    p = m.parameter("p", value=0.0)
    x = m.continuous("x", lb=-5.0, ub=5.0)
    # min (x - 1 - p)^2 -- optimum 0.0 at x = 1 + p for every p.
    m.minimize((x - 1.0 - p) ** 2)
    r = m.solve()
    assert r.objective == pytest.approx(0.0, abs=1e-6)

    p.value = 3.0  # the optimum is still 0.0, but now at x = 4
    with pytest.warns(RuntimeWarning, match="DIFFERENT problem"):
        s = m.sensitivity()

    assert s.objective == pytest.approx(0.0, abs=1e-6)
    assert s.reference_objective == pytest.approx(0.0, abs=1e-6)
    assert s.matches_reference is False, "equal objectives across two problems is not a match"
    assert float(np.asarray(s.x).ravel()[0]) == pytest.approx(4.0, abs=1e-3)


# ── a reference from another model ───────────────────────────────────────


def test_a_solveresult_from_another_model_cannot_report_a_match():
    """``at=`` used to accept any result whose variable names lined up, and then
    cross-check against that other problem's objective. Both models below have
    optimum 0.0, so the objective check alone called it a match."""
    ma = dm.Model("i1322_a")
    pa = ma.parameter("p", value=0.0)
    xa = ma.continuous("x", lb=-5.0, ub=5.0)
    ma.minimize((xa - 1.0) ** 2 + pa * 0.0)
    ra = ma.solve()

    mb = dm.Model("i1322_b")
    _pb = mb.parameter("p", value=0.0)
    xb = mb.continuous("x", lb=-5.0, ub=5.0)
    mb.minimize((xb - 2.0) ** 2)
    rb = mb.solve()
    assert ra.objective == pytest.approx(rb.objective, abs=1e-6) == pytest.approx(0.0, abs=1e-6)

    with pytest.warns(RuntimeWarning, match="DIFFERENT problem|another model"):
        s = mb.sensitivity(at=ra)

    assert s.matches_reference is False
    # An explicit at= is still honoured as the start point -- that is what it is for.
    assert s.status == "optimal"


def test_a_hand_built_solveresult_cannot_report_a_match():
    """A result that never came from `Model.solve` in this process carries no
    fingerprint, so there is nothing to verify and no basis for claiming it
    describes this problem."""
    m = dm.Model("i1322_handbuilt")
    _p = m.parameter("p", value=0.0)
    x = m.continuous("x", lb=-5.0, ub=5.0)
    m.minimize((x - 1.0) ** 2)
    r = m.solve()
    assert getattr(r, "_problem_fingerprint", None) is not None

    stripped = m.solve()
    delattr(stripped, "_problem_fingerprint")
    with pytest.warns(RuntimeWarning, match="DIFFERENT problem"):
        s = m.sensitivity(at=stripped)
    assert s.matches_reference is False


# ── structural changes ───────────────────────────────────────────────────


def test_a_constraint_added_since_the_solve_makes_the_reference_stale():
    """The issue's third case: after `m.subject_to(x >= 1)` the mismatch warning
    called the new point 'a strictly worse point' than a reference that is now
    infeasible. That comparison is meaningless, and now says so instead."""
    m = dm.Model("i1322_rowadd")
    p = m.parameter("p", value=0.0)
    x = m.continuous("x", lb=-5.0, ub=5.0)
    m.minimize((x - 4.0) ** 2 + p * x)
    m.solve()

    m.subject_to(x <= 1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        s = m.sensitivity()
    messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert messages, "the probe fired no RuntimeWarning at all"
    assert any("DIFFERENT problem" in msg for msg in messages)
    assert not any("strictly worse point" in msg for msg in messages)
    assert s.matches_reference is False


def test_a_bound_changed_since_the_solve_makes_the_reference_stale():
    """The case #1313's own test pins, now reported as what it is."""
    m = dm.Model("i1322_bound")
    p = m.parameter("p", value=0.0)
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize((x - 0.9) ** 2 + p * x)
    m.solve()

    x.ub = np.array(0.5)
    with pytest.warns(RuntimeWarning, match="DIFFERENT problem"):
        s = m.sensitivity()
    assert s.matches_reference is False
    assert s.objective == pytest.approx(0.16, rel=1e-3)


# ── the objective value alone does not identify a point ──────────────────


def test_the_same_objective_at_a_different_point_is_not_a_match():
    """The objective value does not identify a point: a symmetric model has two
    global optima with exactly equal objectives, and the derivatives belong to
    whichever one the inner solve returned.

    Constructed rather than stumbled upon: a live (non-stale) reference for this
    very model is edited to claim the optimal objective at a point that is not
    the optimum. The objective check passes, and only the point check can tell
    that the KKT point the derivatives were taken at is somewhere else.
    """
    m = dm.Model("i1322_symmetric")
    p = m.parameter("p", value=0.0)
    x = m.continuous("x", lb=-3.0, ub=3.0)
    m.minimize((x**2 - 1.0) ** 2 + p * 0.0)  # minima at x = -1 and x = +1, both 0
    r = m.solve()
    assert r.objective == pytest.approx(0.0, abs=1e-6)
    assert abs(float(np.asarray(r.x["x"]).ravel()[0])) == pytest.approx(1.0, abs=1e-4)

    r.x = {"x": np.asarray(0.01)}  # the same objective claimed at a different point
    with pytest.warns(RuntimeWarning, match="DIFFERENT point"):
        s = m.sensitivity(at=r)

    assert s.matches_reference is False
    assert s.objective == pytest.approx(0.0, abs=1e-6)
    assert abs(float(np.asarray(s.x).ravel()[0])) == pytest.approx(1.0, abs=1e-3)


# ── no false positives ───────────────────────────────────────────────────


def test_an_unchanged_model_still_reports_a_match_and_does_not_warn():
    """The whole point of the guard is that it fires only on a real change."""
    m, _p, _x = _double_well("i1322_clean")
    r = m.solve()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        s = m.sensitivity()
    assert s.matches_reference is True
    assert s.reference_objective == pytest.approx(r.objective, abs=1e-9)


def test_a_second_solve_refreshes_the_reference():
    """Re-solving after the change is the documented remedy, and it works."""
    m, p, _x = _double_well("i1322_resolve")
    m.solve()
    p.value = 10.0
    m.solve()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        s = m.sensitivity()
    assert s.matches_reference is True
    assert s.objective == pytest.approx(-11.970, abs=1e-2)


def test_reading_bounds_or_re_fixing_and_restoring_does_not_make_it_stale():
    """A scope that restores what it changed leaves the fingerprint identical --
    the comparison is on values, so a round trip is not a change."""
    m, _p, x = _double_well("i1322_roundtrip")
    m.solve()
    with x.fixed(0.0):
        pass
    with m.saved_bounds():
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        s = m.sensitivity()
    assert s.matches_reference is True


def test_an_unsolved_model_still_has_no_reference():
    m, _p, _x = _double_well("i1322_unsolved")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        s = m.sensitivity()
    assert s.matches_reference is None
    assert s.reference_objective is None


# ── the fingerprint itself ───────────────────────────────────────────────


def test_the_fingerprint_moves_with_every_thing_that_moves_the_optimum():
    """Directly, so the staleness rule cannot silently lose a dimension."""
    from discopt._evaluator_cache import solution_state_fingerprint

    m = dm.Model("i1322_fp")
    p = m.parameter("p", value=1.0)
    x = m.continuous("x", lb=0.0, ub=5.0)
    m.minimize((x - p) ** 2)
    base = solution_state_fingerprint(m)
    assert solution_state_fingerprint(m) == base, "must be stable with nothing changed"

    checks = 0
    p.value = 2.0
    assert solution_state_fingerprint(m) != base
    checks += 1
    p.value = 1.0
    assert solution_state_fingerprint(m) == base
    checks += 1

    x.ub = np.asarray(4.0)
    assert solution_state_fingerprint(m) != base
    checks += 1
    x.ub = np.asarray(5.0)
    assert solution_state_fingerprint(m) == base
    checks += 1

    m.subject_to(x <= 3.0)
    assert solution_state_fingerprint(m) != base
    checks += 1

    m.minimize((x - p) ** 2)  # a fresh objective object
    assert solution_state_fingerprint(m) != base
    checks += 1

    assert checks == 6, "the probe did not execute every comparison it reports"


def test_the_fingerprint_does_not_travel_through_serialization():
    """It holds process-local object identities, and `_last_solve_result` is
    already classified as not-carried; nothing may reintroduce it."""
    m = dm.Model("i1322_ser")
    _p = m.parameter("p", value=1.0)
    x = m.continuous("x", lb=0.0, ub=5.0)
    m.minimize((x - 1.0) ** 2)
    r = m.solve()
    assert getattr(r, "_problem_fingerprint", None) is not None

    from discopt.result_io import deserialize_result, serialize_result

    doc = serialize_result(r)
    assert not any("fingerprint" in str(k) for k in doc)
    assert getattr(deserialize_result(doc), "_problem_fingerprint", None) is None
