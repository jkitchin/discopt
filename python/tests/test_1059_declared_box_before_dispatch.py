"""The declared-box check must be a property of the model, not of the family.

``_check_finite_bounds`` (the "very large or infinite declared bounds" warning)
and ``_detect_nonlinear_bound_infeasibility`` (which can prove infeasibility
outright) used to run ~1700 lines into ``solve_model`` -- past the point where
six of the eight solver families have already returned. Measured on a convex
MINLP with an unbounded ``x``:

===========================  =======
call                         warned
===========================  =======
``solve()`` (auto-routed)    no
``solve(solver="mip-nlp")``  no
``solve(solver="bb")``       yes
``solve(nlp_bb=True)``       yes
===========================  =======

So whether a user heard about their own unbounded variable depended on which
engine happened to run. The #1059 auto-route is what surfaced it -- a bare
``.solve()`` on a convex model started going to ``mip-nlp`` -- but
``solver="mip-nlp"``, ``"amp"``, ``"gp"``, ``"gp-minlp"``, ``"direct"`` and
``"surrogate"`` had the same hole before the route existed. The check now runs
once, before the dispatch.

``test_863_shared_declared_box_tightening.py`` is the other half of this
contract: it pins that moving the call did not turn one pass into two.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest

WARN_MATCH = "very large or infinite declared bounds"


def _unbounded_convex_minlp():
    """Convex MINLP whose ``x`` has no declared bounds -- the warning's subject.

    ``x <= 20`` is a *constraint*, not a bound, so the declared box stays
    ``(-inf, +inf)`` and the model is still solvable to optimality. The warning
    must fire even though nothing goes wrong numerically; that is the point of
    warning rather than failing.
    """
    m = dm.Model("unbounded_x")
    m.continuous("x")
    m.binary("y")
    x, y = m._variables[0], m._variables[1]
    m.minimize(dm.exp(x) + 3 * y)
    m.subject_to(x + y >= 1)
    m.subject_to(x <= 20)
    return m


def _warned(**solve_kwargs) -> tuple[bool, str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _unbounded_convex_minlp().solve(time_limit=20, **solve_kwargs)
    hit = any(WARN_MATCH in str(w.message) for w in caught)
    return hit, result.status


@pytest.mark.parametrize(
    "label,kwargs",
    [
        ("default (auto-routed to mip-nlp)", {}),
        ("solver='mip-nlp'", {"solver": "mip-nlp"}),
        ("solver='bb'", {"solver": "bb"}),
        ("nlp_bb=True", {"nlp_bb": True}),
        ("an option that declines the route", {"node_callback": lambda *a, **k: None}),
    ],
)
def test_every_entry_point_warns_about_an_unbounded_declared_box(label, kwargs):
    """The first two rows failed before the check was hoisted."""
    warned, status = _warned(**kwargs)
    assert status == "optimal", f"{label}: model did not solve ({status})"
    assert warned, f"{label}: no unbounded-declared-box warning"


def test_the_warning_stays_quiet_on_a_properly_bounded_model():
    """The complement: a bounded model must not learn to warn.

    Without this the test above is satisfiable by warning unconditionally.
    """
    m = dm.Model("bounded")
    x = m.continuous("x", lb=0, ub=20)
    y = m.binary("y")
    m.minimize(dm.exp(x) + 3 * y)
    m.subject_to(x + y >= 1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = m.solve(time_limit=20)
    assert result.status == "optimal"
    assert not any(WARN_MATCH in str(w.message) for w in caught)


def test_the_check_runs_before_the_family_dispatch_not_inside_one():
    """Structural pin: the routed path must reach the check.

    The behavioural tests above would also pass if someone re-implemented the
    warning separately inside ``solve_mip_nlp``. Spying on the shared pass
    itself pins that the *one* hoisted call site is what the routed solve hits
    -- which is what keeps #863's run-once property and this one from drifting
    apart.
    """
    import discopt.solver as _solver

    real = _solver._declared_box_tightening
    calls: list[int] = []

    def _counting(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    _solver._declared_box_tightening = _counting
    try:
        with pytest.warns(UserWarning, match=WARN_MATCH):
            result = _unbounded_convex_minlp().solve(time_limit=20)
    finally:
        _solver._declared_box_tightening = real

    assert result.status == "optimal"
    assert result.algorithm_route is not None and "mip-nlp" in result.algorithm_route, (
        "this model is no longer auto-routed, so it no longer covers the routed path"
    )
    assert len(calls) == 1, f"declared-box pass ran {len(calls)} times (expected exactly 1)"


# ── Soundness: the pass now sees rows it never saw before ─────────────────────
#
# Running before the dispatch also means running before ``reformulate_gdp``, so
# the tightening pass is handed ``_DisjunctiveConstraint`` / ``_IndicatorConstraint``
# rows that previously reached it only in already-reformulated big-M form. A
# disjunction means "at least ONE disjunct holds" and an indicator binds only at
# its active value; reading either unconditionally would let the pass prove an
# infeasibility that is not real. A false ``infeasible`` is the one outcome
# CLAUDE.md §1 allows no slack for, so it gets its own tests rather than being
# argued about in a comment.


@pytest.mark.parametrize("build_name", ["disjunction", "indicator", "either_or"])
def test_gdp_rows_never_yield_a_false_infeasibility(build_name):
    """Each model is feasible, and each is a trap for a conjunctive reading."""
    import discopt.solver as _solver

    if build_name == "disjunction":
        # x <= 1 AND x >= 9 is infeasible; "x <= 1 OR x >= 9" is not.
        m = dm.Model("contradictory_disjuncts")
        x = m.continuous("x", lb=0, ub=10)
        low = m.make_disjunct("low")
        low.subject_to(x <= 1)
        high = m.make_disjunct("high")
        high.subject_to(x >= 9)
        m.add_disjunction([low, high], name="mode")
        m.minimize(x)
        expected = 0.0
    elif build_name == "indicator":
        # w <= 1 binds only at z = 1, and contradicts w >= 2 if read always.
        m = dm.Model("indicator")
        z = m.binary("z")
        w = m.continuous("w", lb=0, ub=10)
        m.subject_to(w >= 2)
        m.if_then(z, [w <= 1], name="only_if_z")
        m.minimize(w)
        expected = 2.0
    else:
        m = dm.Model("either_or")
        v = m.continuous("v", lb=0, ub=10)
        m.either_or([[v <= 1], [v >= 9]], name="mode")
        m.minimize(v)
        expected = 0.0

    tightening = _solver._declared_box_tightening(m)
    if tightening is not None:
        assert not tightening[2].infeasible, (
            f"{build_name}: tightening claimed infeasible on a feasible model "
            f"({tightening[2].infeasibility_reason})"
        )
    assert _solver._detect_nonlinear_bound_infeasibility(m) is None

    result = m.solve(time_limit=30)
    assert result.status == "optimal", f"{build_name}: {result.status}"
    assert result.objective == pytest.approx(expected, abs=1e-6)


def test_an_infeasibility_proof_still_short_circuits_from_the_new_position():
    """The early-``infeasible`` return must survive the move.

    The real pass proves infeasibility only on models too large for a unit test,
    so the proof is injected -- the same idiom as
    ``test_863...::test_infeasibility_proof_is_still_reported``. What is under
    test here is the *call site*: that a proof produced before the dispatch still
    ends the solve, including on the auto-routed path that never reached the old
    position.
    """
    import discopt.solver as _solver

    class _Stats:
        infeasible = True
        infeasibility_reason = "synthetic empty interval"
        n_tightened = 0
        applied_rules = ()

    real = _solver._declared_box_tightening

    def _infeasible(model, *a, **kw):
        lb, ub, _ = real(model, *a, **kw)
        return lb, ub, _Stats()

    _solver._declared_box_tightening = _infeasible
    try:
        result = _unbounded_convex_minlp().solve(time_limit=20)
    finally:
        _solver._declared_box_tightening = real

    assert result.status == "infeasible"
    assert result.gap_certified is True
