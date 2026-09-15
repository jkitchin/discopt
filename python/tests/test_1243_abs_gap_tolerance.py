"""``Model.solve(abs_gap_tolerance=...)`` on every branch-and-bound route (#1243).

The solver has always converged on a DISJUNCTION -- relative gap ``<=
gap_tolerance`` OR absolute gap ``<= abs_gap_tol`` -- but only the relative half
was reachable from :meth:`Model.solve`. The absolute half was the module
constant ``_DEFAULT_ABS_GAP_TOL = 1e-6``, so no caller could tighten it.

That blocks any certificate stated as an absolute test. A CALPHAD phase-stability
certificate is exactly one: every phase's pricing lower bound must satisfy
``LB >= -eps`` in units of G/RT, and at a converged equilibrium the pricing
optimum is approximately 0, where a relative gap carries no information at all.

Two properties are pinned here:

1. **It binds.** An explicit ``abs_gap_tolerance`` is honored on the route the
   solve takes, and a solve that reports ``optimal`` under it really did close
   the absolute gap to it.
2. **It is inert when omitted.** Every default is byte-identical to the constant
   it replaced, so an omitted argument cannot move a bound. (The corpus panel
   backing this is in the commit message: 10/10 non-time-limited instances
   reproduce status/objective/bound/node_count exactly.)
"""

from __future__ import annotations

import inspect
import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt.solver import (
    _DEFAULT_ABS_GAP_TOL,
    _gap_criterion,
    _gap_values_converged,
    _resolve_abs_gap_tolerance,
    solve_model,
    solve_model_accepted_kwargs,
)

# ──────────────────────────────────────────────────────────────────────
# 1. Surface
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_option_is_on_both_public_signatures_and_is_forwardable():
    assert "abs_gap_tolerance" in inspect.signature(Model.solve).parameters
    assert "abs_gap_tolerance" in inspect.signature(solve_model).parameters
    # ``Model.solve`` rejects an unknown keyword loudly, so the option must also
    # be in the forwardable set or a caller could never pass it through.
    assert "abs_gap_tolerance" in solve_model_accepted_kwargs()


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0.0, -1e-6, float("inf"), float("nan")])
def test_a_nonpositive_or_nonfinite_tolerance_is_refused(bad):
    """Refuse loudly rather than degrade to the relative arm alone.

    A zero or negative absolute tolerance can never be met by a floating-point
    gap, so accepting it would silently turn the disjunction back into the
    relative criterion — the exact mode this option exists to escape.
    """
    with pytest.raises(ValueError, match="abs_gap_tolerance"):
        _resolve_abs_gap_tolerance(bad)


@pytest.mark.unit
def test_omitting_the_option_resolves_to_the_constant_it_replaced():
    assert _resolve_abs_gap_tolerance(None) == _DEFAULT_ABS_GAP_TOL == 1e-6


# ──────────────────────────────────────────────────────────────────────
# 2. The criterion itself
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_a_near_zero_optimum_is_not_declared_converged_at_a_tightened_tolerance():
    """The issue's acceptance case, at the arithmetic that decides it.

    True optimum ~1e-8, absolute gap 1e-7. The RELATIVE arm cannot rescue it:
    the denominator ``max(|ub|, |lb|, 1e-10)`` is itself ~1e-7 there, so the
    relative gap is ~0.9, not ~1e-7. (A denominator floored at 1.0 — which is
    what the tree's own ``gap()`` uses — would report 1e-7 and certify; that is
    the gear pathology the decoupled absolute tolerance exists to prevent.)
    """
    lb, ub = 1e-8, 1e-8 + 1e-7
    assert not _gap_values_converged(ub, lb, gap_tolerance=1e-4, abs_gap_tol=1e-10)
    # And the default tolerance *would* have certified it, which is the whole
    # reason a caller needs control of this number.
    assert _gap_values_converged(ub, lb, gap_tolerance=1e-4, abs_gap_tol=1e-6)


@pytest.mark.unit
def test_the_criterion_is_a_disjunction_in_both_directions():
    # Absolute arm alone: a huge relative gap, a tiny absolute one.
    assert _gap_values_converged(1e-9, 0.0, gap_tolerance=1e-12, abs_gap_tol=1e-6)
    # Relative arm alone: a large absolute gap on a large objective.
    assert _gap_values_converged(1e6, 1e6 - 1.0, gap_tolerance=1e-4, abs_gap_tol=1e-9)
    # Neither.
    assert not _gap_values_converged(1e6, 0.0, gap_tolerance=1e-4, abs_gap_tol=1e-9)


@pytest.mark.unit
def test_reported_criterion_never_disagrees_with_the_test_that_stopped_the_search():
    """``_gap_criterion`` and ``_gap_values_converged`` must agree everywhere."""
    rng = np.random.default_rng(1243)
    checked = 0
    for _ in range(2000):
        scale = 10.0 ** rng.integers(-9, 7)
        ub = float(rng.normal() * scale)
        ub_lb_gap = float(abs(rng.normal()) * scale * 10.0 ** rng.integers(-9, 1))
        lb = ub - ub_lb_gap
        rel = float(10.0 ** rng.integers(-10, -2))
        abs_ = float(10.0 ** rng.integers(-12, -2))
        converged = _gap_values_converged(ub, lb, rel, abs_)
        crit = _gap_criterion(ub, lb, rel, abs_)
        assert converged == (crit is not None), (ub, lb, rel, abs_, converged, crit)
        if crit == "absolute":
            assert max(0.0, ub - lb) <= abs_
        elif crit == "relative":
            assert max(0.0, ub - lb) > abs_
            assert max(0.0, ub - lb) / max(abs(ub), abs(lb), 1e-10) <= rel
        checked += 1
    assert checked == 2000, f"probe ran {checked} comparisons"


@pytest.mark.unit
def test_criterion_is_none_when_neither_arm_holds_or_a_bound_is_missing():
    assert _gap_criterion(1.0, 0.0, 1e-4, 1e-6) is None
    assert _gap_criterion(float("inf"), 0.0, 1e-4, 1e-6) is None
    assert _gap_criterion(1.0, float("-inf"), 1e-4, 1e-6) is None


# ──────────────────────────────────────────────────────────────────────
# 3. End to end
# ──────────────────────────────────────────────────────────────────────


def _entropy_model() -> Model:
    """#1242's acceptance model: nonconvex, certifies, optimum ~ -0.0583."""
    m = Model()
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(dm.xlogx(y) + dm.xlogx(1 - y) + 3 * y * (1 - y))
    return m


@pytest.mark.slow
@pytest.mark.correctness
def test_tightening_the_absolute_tolerance_actually_closes_the_gap():
    """#1242's acceptance criterion, which needed this option to be reachable.

    At the default the solve stops at an absolute gap of ~9e-7 no matter how
    small ``gap_tolerance`` is made — the 1e-6 constant is what binds, and
    before #1243 there was no way to say otherwise.
    """
    loose = _entropy_model().solve(solver="bb", time_limit=300, gap_tolerance=1e-9)
    assert loose.status == "optimal"
    assert abs(loose.objective - loose.bound) > 1e-9  # the constant binds

    tight = _entropy_model().solve(
        solver="bb", time_limit=300, gap_tolerance=1e-9, abs_gap_tolerance=1e-9
    )
    assert tight.status == "optimal"
    assert abs(tight.objective - tight.bound) <= 1e-9
    # Same optimum, certified harder — not a different answer.
    assert tight.objective == pytest.approx(loose.objective, abs=1e-9)
    assert tight.bound <= tight.objective + 1e-12


@pytest.mark.slow
@pytest.mark.correctness
def test_an_optimal_verdict_honors_the_tolerance_it_was_given():
    """The invariant the CALPHAD certificate rests on.

    Whenever the solver returns ``optimal`` under an explicit
    ``abs_gap_tolerance``, one of the two arms must actually hold on the
    reported pair — otherwise "optimal to eps" is a claim the solve did not
    earn.
    """
    checked = 0
    for tol in (1e-6, 1e-8, 1e-10):
        res = _entropy_model().solve(
            solver="bb", time_limit=300, gap_tolerance=1e-4, abs_gap_tolerance=tol
        )
        if res.status != "optimal":
            continue
        assert res.bound is not None and math.isfinite(res.bound)
        assert _gap_values_converged(res.objective, res.bound, 1e-4, tol), (
            f"reported optimal at abs_gap_tolerance={tol} with "
            f"objective={res.objective!r} bound={res.bound!r}"
        )
        checked += 1
    assert checked > 0, "no solve reached `optimal`; the probe asserted nothing"


@pytest.mark.slow
def test_solver_stats_names_the_criterion_that_stopped_the_search():
    res = _entropy_model().solve(
        solver="bb", time_limit=300, gap_tolerance=1e-9, abs_gap_tolerance=1e-9
    )
    assert res.status == "optimal"
    assert res.solver_stats is not None
    assert res.solver_stats.get("gap_criterion") == "absolute"


@pytest.mark.smoke
def test_a_budget_stop_reports_no_criterion():
    """``gap_criterion`` is absent when the solve stopped on a budget.

    Reporting one there would claim a convergence that never happened.
    """
    m = Model()
    x = m.continuous("x", shape=6, lb=-5.0, ub=5.0)
    m.minimize(dm.sum(x * x[::-1]) + dm.sum(x))
    res = m.solve(solver="bb", time_limit=300, max_nodes=3)
    if res.status == "optimal":
        pytest.skip("model certified within the node budget; nothing to assert")
    assert (res.solver_stats or {}).get("gap_criterion") is None


# ──────────────────────────────────────────────────────────────────────
# 4. Route coverage
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_the_native_spatial_kernel_takes_a_relative_arm():
    """The kernel's ``gap_tol`` is applied ABSOLUTELY, so honoring the option
    there means also giving it the relative arm — otherwise the two routes
    would certify to different tolerances on the same model.

    ``solver.py`` passes ``rel_gap_tol=`` unconditionally, so a binding without
    it would raise ``TypeError`` on every native-kernel solve. Asserted against
    the live binding rather than the source: the two are maintained separately
    and drifting apart is the failure mode.
    """
    from discopt._rust import solve_spatial_tree_py

    with pytest.raises(TypeError) as exc:
        # Deliberately arity-incomplete: the call must fail for the MISSING
        # required arguments, never for an unknown ``rel_gap_tol`` keyword.
        solve_spatial_tree_py(rel_gap_tol=0.0)
    msg = str(exc.value)
    assert "rel_gap_tol" not in msg, f"solve_spatial_tree_py rejected rel_gap_tol: {msg}"
    assert "argument" in msg.lower(), f"unexpected TypeError from the binding: {msg}"


@pytest.mark.unit
def test_amp_receives_the_tolerance_as_its_own_abs_tol():
    """AMP spells the same knob ``abs_tol``; the portable name must reach it."""
    import discopt.solver as _solver

    captured = {}

    def _fake_solve_amp(model, **kw):
        captured.update(kw)
        raise RuntimeError("stop after the option mapping")

    import discopt.solvers.amp as amp_mod

    real = amp_mod.solve_amp
    amp_mod.solve_amp = _fake_solve_amp
    try:
        m = Model()
        x = m.continuous("x", lb=0.0, ub=1.0)
        y = m.continuous("y", lb=0.0, ub=1.0)
        m.minimize(x * y - x)
        with pytest.raises(RuntimeError, match="stop after the option mapping"):
            _solver.solve_model(m, solver="amp", abs_gap_tolerance=1e-9, time_limit=5)
    finally:
        amp_mod.solve_amp = real
    assert captured.get("abs_tol") == 1e-9
    assert captured.get("rel_gap") == 1e-4


@pytest.mark.unit
def test_an_explicit_amp_abs_tol_still_wins():
    import discopt.solver as _solver
    import discopt.solvers.amp as amp_mod

    captured = {}

    def _fake_solve_amp(model, **kw):
        captured.update(kw)
        raise RuntimeError("stop")

    real = amp_mod.solve_amp
    amp_mod.solve_amp = _fake_solve_amp
    try:
        m = Model()
        x = m.continuous("x", lb=0.0, ub=1.0)
        y = m.continuous("y", lb=0.0, ub=1.0)
        m.minimize(x * y - x)
        with pytest.raises(RuntimeError):
            _solver.solve_model(m, solver="amp", abs_gap_tolerance=1e-9, abs_tol=1e-3, time_limit=5)
    finally:
        amp_mod.solve_amp = real
    assert captured.get("abs_tol") == 1e-3
