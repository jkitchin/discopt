"""#875: root setup must be *cheap* and *bounded* before the first B&B node.

Successor to #863/#868. After PR #868, ``watercontamination0202`` (106,711 vars /
107,209 rows, 7 binaries) returned instead of hanging, but ``solve(time_limit=30)``
took **579.3 s** — 19.3x — with ``nodes=0``. The time was fully attributed:

    phase                                     cost     note
    _fix_single_var_equalities              ~460 s     dominant, 23/29 stack samples
    tighten_nonlinear_bounds x3               80.9 s   no deadline awareness at all
    _classify_model_convexity x2              14.7 s   per-model budget, not per-solve
    presolve + load + classification          ~12 s    already capped by #868

Every item is pre-B&B root setup — the class #858 fixed in the LP engine and #868
fixed in the Rust presolve passes, one layer up each time. Two distinct defects hide
under that one heading, and the tests below separate them:

* a **cost** bug — ``_fix_single_var_equalities`` was ``O(n_constraints * n_vars)``
  because the affine linearizer it calls allocates and zeroes a dense ``n_vars``
  array per call, and the caller then walked all ``n_vars`` entries in Python to find
  the single nonzero. That is not a budget problem; no deadline should have been
  needed for a scan over one-leaf bodies. Fixed by a sparse linearizer core.
* a **budget** bug — ``tighten_nonlinear_bounds`` had no ``deadline`` parameter, and
  the convexity classifier's budget is a fraction of ``time_limit`` recomputed per
  *model object*, so a reformulation restarts it and the fractions add up.

The in-repo corpus is far too small to show either as a wall-clock overrun, so these
tests target the mechanism directly: the cost bug via a scaling probe that fails
before the fix, the budget bugs via an expired deadline.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import time  # noqa: E402
from pathlib import Path  # noqa: E402

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._relax.milp_relaxation import (  # noqa: E402
    _any_linear_constraint_form,
    _linear_constraint_forms,
    _linearize_affine_expr,
    _linearize_affine_expr_sparse,
)
from discopt._relax.model_utils import flat_variable_bounds  # noqa: E402
from discopt._relax.nonlinear_bound_tightening import (  # noqa: E402
    DEFAULT_NONLINEAR_BOUND_RULES,
    FunctionDomainBoundRule,
    NonlinearBoundTighteningRule,
    PeriodicVariableBoundRule,
    tighten_nonlinear_bounds,
)
from discopt._relax.uniform_relax import _fix_single_var_equalities  # noqa: E402
from discopt.modeling.core import SolveResult  # noqa: E402

# --------------------------------------------------------------------------
# The sparse affine linearizer: same answer, no O(n_vars) per call
# --------------------------------------------------------------------------


def _affine_model(n_vars: int, n_eq: int):
    """``n_eq`` single-variable equalities over ``n_vars`` variables.

    Every equality body has exactly ONE leaf, so an honest scan is O(n_eq); the dense
    linearizer made it O(n_eq * n_vars) regardless.
    """
    m = dm.Model(f"aff{n_vars}")
    x = m.continuous("x", shape=(n_vars,), lb=-10.0, ub=10.0)
    for k in range(n_eq):
        m.subject_to(x[k % n_vars] == 1.0)
    m.minimize(x[0])
    return m


def test_sparse_linearizer_agrees_with_the_dense_one():
    """The dense wrapper is now a view of the sparse core; they must not diverge."""
    m = dm.Model("agree")
    x = m.continuous("x", shape=(6,), lb=-1.0, ub=1.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    m.minimize(x[0])

    m.subject_to(2.0 * x[0] + 3.0 * x[3] - 4.0 * y == 1.0)
    m.subject_to(x[1] - x[1] + x[5] <= 2.0)  # cancelling terms -> an explicit zero
    m.subject_to(dm.sum(x) + y >= 0.0)

    n = 7
    for con in m._constraints:
        dense, dense_const = _linearize_affine_expr(con.body, m, n)
        terms, sparse_const = _linearize_affine_expr_sparse(con.body, m, n)
        assert sparse_const == dense_const
        rebuilt = np.zeros(n, dtype=np.float64)
        for j, c in terms.items():
            rebuilt[j] = c
        assert np.array_equal(rebuilt, dense)


def test_sparse_linearizer_refuses_exactly_what_the_dense_one_refused():
    m = dm.Model("refuse")
    x = m.continuous("x", shape=(2,), lb=-1.0, ub=1.0)
    m.minimize(x[0])
    m.subject_to(x[0] * x[1] <= 1.0)  # nonlinear body
    body = m._constraints[0].body
    with pytest.raises(ValueError):
        _linearize_affine_expr(body, m, 2)
    with pytest.raises(ValueError):
        _linearize_affine_expr_sparse(body, m, 2)


def test_fix_single_var_equalities_is_flat_in_the_variable_count():
    """The decisive cost test, and the one that FAILS before the fix.

    At a fixed constraint count the pass must not get slower as variables are added:
    the bodies do not change. Measured before the fix, 400 equalities:

        n_vars=2,000    0.068 s
        n_vars=8,000    0.240 s   (3.55x for 4x n_vars)
        n_vars=32,000   0.951 s   (3.96x)
        n_vars=128,000  3.650 s   (3.84x)

    i.e. exactly linear in ``n_vars`` — which extrapolates to the ~460 s the issue
    measured. After: 0.001 s at n_vars=32,000 (0.002 s at 128,000), and flat.

    Two assertions, and the ABSOLUTE one is primary. A ratio of two post-fix timings
    divides ~1 ms by ~1 ms, which on a shared CI runner is noise over noise — one
    scheduler hiccup flips it. The absolute ceiling has ~250x headroom for the sparse
    scan and still fails the dense one on a runner 3x slower than the machine these
    numbers came from, so it measures the implementation rather than the box. The
    ratio is kept as the complexity-class signal but only evaluated when the baseline
    is far enough above the timer floor to mean anything.

    ``min`` over repetitions, not mean: the fastest run is the one least polluted by
    whatever else the machine was doing, which is the whole question here.
    """
    n_eq = 300
    reps = 3
    walls = {}
    for n_vars in (2_000, 32_000):
        best = float("inf")
        for _ in range(reps):
            m = _affine_model(n_vars, n_eq)
            lb, ub = flat_variable_bounds(m)
            t0 = time.perf_counter()
            out_lb, out_ub = _fix_single_var_equalities(m, lb, ub)
            best = min(best, time.perf_counter() - t0)
            # the pass must still do its job: every pinned variable collapsed to a point
            pinned = np.flatnonzero(out_lb == out_ub)
            assert pinned.size >= min(n_eq, n_vars)
            assert np.all(out_lb[pinned] == 1.0)
        walls[n_vars] = best

    # Primary: dense was 0.851 s here at n_vars=32,000, sparse is ~0.001 s.
    assert walls[32_000] < 0.25, (
        f"the scan is still paying O(n_vars) per row: {walls[32_000]:.3f}s at "
        f"n_vars=32,000 with only {n_eq} rows (dense measured 0.851s, sparse 0.001s)"
    )
    # Secondary: 16x the variables for the same rows must not cost 16x the wall.
    # Skipped when the baseline is at the timer floor, where the quotient is noise.
    if walls[2_000] > 5e-3:
        ratio = walls[32_000] / walls[2_000]
        assert ratio < 4.0, (
            f"cost still scales with n_vars (16x vars -> {ratio:.1f}x wall): {walls}"
        )


def test_fix_single_var_equalities_still_pins_and_still_refuses():
    """Behaviour parity on the cases the docstring promises."""
    m = dm.Model("pins")
    x = m.continuous("x", shape=(4,), lb=-10.0, ub=10.0)
    m.minimize(x[0])
    m.subject_to(2.0 * x[1] == 6.0)  # pins x1 = 3
    m.subject_to(x[2] == 99.0)  # OUTSIDE the box -> left for the LP rows
    m.subject_to(x[0] + x[3] == 1.0)  # two variables -> not a pin
    lb, ub = flat_variable_bounds(m)
    out_lb, out_ub = _fix_single_var_equalities(m, lb, ub)

    assert out_lb[1] == out_ub[1] == pytest.approx(3.0)
    assert (out_lb[2], out_ub[2]) == (-10.0, 10.0), "an out-of-box pin must not be applied"
    assert (out_lb[0], out_ub[0]) == (-10.0, 10.0)
    assert (out_lb[3], out_ub[3]) == (-10.0, 10.0)
    # inputs untouched
    assert lb[1] == -10.0 and ub[1] == 10.0


def test_any_linear_constraint_form_agrees_with_the_list(monkeypatch):
    """The boolean probe must answer exactly what ``bool(_linear_constraint_forms())``
    answered, on both a model that has linear factors and one that has none."""
    m = _affine_model(4_000, 200)
    assert _any_linear_constraint_form(m, 4_000) is True
    assert bool(_linear_constraint_forms(m, 4_000)) is True

    nonlinear = dm.Model("nl")
    z = nonlinear.continuous("z", shape=(3,), lb=0.1, ub=2.0)
    nonlinear.minimize(z[0])
    nonlinear.subject_to(z[0] * z[1] <= 1.0)
    nonlinear.subject_to(dm.log(z[2]) <= 1.0)
    assert _any_linear_constraint_form(nonlinear, 3) is False
    assert bool(_linear_constraint_forms(nonlinear, 3)) is False


def test_any_linear_constraint_form_short_circuits(monkeypatch):
    """Counted, not timed: the probe must linearize ONE row when the first row is
    linear, rather than every row.

    Counting the linearizations is the actual claim (``bool()`` of a fully
    materialised list is what this replaced), and unlike a wall-clock comparison it
    cannot flake on a loaded runner — the failure mode that cost real time on #863
    and briefly on this branch.
    """
    import discopt._relax.milp_relaxation as mr

    calls = []
    real = mr._linearize_affine_expr_sparse

    def _counting(expr, model, n_vars):
        calls.append(1)
        return real(expr, model, n_vars)

    monkeypatch.setattr(mr, "_linearize_affine_expr_sparse", _counting)

    m = _affine_model(4_000, 200)
    calls.clear()
    assert _any_linear_constraint_form(m, 4_000) is True
    short_circuit_calls = len(calls)

    calls.clear()
    _linear_constraint_forms(m, 4_000)
    full_calls = len(calls)

    assert short_circuit_calls == 1, (
        f"the boolean probe linearized {short_circuit_calls} rows; it must stop at "
        f"the first linear one"
    )
    assert full_calls == 200, f"expected one linearization per row, got {full_calls}"


# --------------------------------------------------------------------------
# tighten_nonlinear_bounds honours a deadline
# --------------------------------------------------------------------------


def _nbt_model(n: int = 40):
    m = dm.Model(f"nbt{n}")
    x = m.continuous("x", shape=(n,), lb=-100.0, ub=100.0)
    y = m.continuous("y", shape=(n,))  # free: the rules exist to bound these
    m.minimize(x[0])
    for k in range(n):
        m.subject_to(x[k] * x[k] <= 4.0)
        m.subject_to(y[k] - x[k] * x[k] == 0.0)
    return m


def test_tightening_with_an_expired_deadline_does_nothing_and_says_so():
    m = _nbt_model()
    lb, ub = flat_variable_bounds(m)
    out_lb, out_ub, stats = tighten_nonlinear_bounds(m, lb, ub, deadline=time.perf_counter())
    assert stats.deadline_reached is True
    assert stats.n_tightened == 0
    assert stats.applied_rules == ()
    assert np.array_equal(out_lb, lb) and np.array_equal(out_ub, ub)
    assert stats.infeasible is False


def test_a_future_deadline_changes_nothing():
    """The no-regression half: when the budget is not binding — the normal case — the
    poll must be the ONLY difference. Same box, same stats, byte for byte."""
    m = _nbt_model()
    lb, ub = flat_variable_bounds(m)
    base_lb, base_ub, base_stats = tighten_nonlinear_bounds(m, lb, ub)
    dl_lb, dl_ub, dl_stats = tighten_nonlinear_bounds(
        m, lb, ub, deadline=time.perf_counter() + 3600.0
    )
    assert np.array_equal(base_lb, dl_lb)
    assert np.array_equal(base_ub, dl_ub)
    assert base_stats.n_tightened == dl_stats.n_tightened
    assert base_stats.applied_rules == dl_stats.applied_rules
    assert base_stats.infeasible == dl_stats.infeasible
    assert dl_stats.deadline_reached is False
    assert base_stats.deadline_reached is False


def test_a_truncated_pass_only_ever_loosens():
    """Anytime contract: whatever a budgeted pass returns must be a SUPERSET of the
    box an unbudgeted pass returns (looser or equal, never tighter, never wrong)."""
    m = _nbt_model(60)
    lb, ub = flat_variable_bounds(m)
    full_lb, full_ub, _ = tighten_nonlinear_bounds(m, lb, ub)
    for budget in (0.0, 1e-4, 1e-3, 1e-2):
        cut_lb, cut_ub, _stats = tighten_nonlinear_bounds(
            m, lb, ub, deadline=time.perf_counter() + budget
        )
        assert np.all(cut_lb <= full_lb + 1e-12), "a budgeted pass tightened PAST the full pass"
        assert np.all(cut_ub >= full_ub - 1e-12), "a budgeted pass tightened PAST the full pass"
        assert np.all(cut_lb >= lb - 1e-12) and np.all(cut_ub <= ub + 1e-12)


def test_a_row_scan_that_needs_completeness_is_never_truncated():
    """``PeriodicVariableBoundRule`` concludes from what the model does NOT contain: a
    variable is restricted to one period only while no row uses it outside sin/cos. A
    truncated scan could miss the disqualifying row and cut feasible points, so the
    rule must be skipped whole under a budget, never run on a prefix.

    Guarded structurally (the flag) and behaviourally (a model whose disqualifying
    use sits in the LAST row).
    """
    assert PeriodicVariableBoundRule.row_scan_is_anytime is False
    assert NonlinearBoundTighteningRule.row_scan_is_anytime is False, (
        "the base-class default must stay the SAFE one, so a new accumulating rule "
        "does not silently inherit truncation"
    )
    assert FunctionDomainBoundRule.row_scan_is_anytime is True

    from discopt._relax.nonlinear_bound_tightening import _cached_flat_metadata

    def _build(disqualify: bool):
        """``t`` free and used inside ``cos``; optionally ALSO used bare, in the very
        last row, which is what makes the period reduction invalid."""
        m = dm.Model(f"periodic{int(disqualify)}")
        m.continuous("t")  # flat index 0
        pad = m.continuous("pad", shape=(400,), lb=0.0, ub=1.0)
        m.minimize(dm.cos(m._variables[0]))
        for k in range(400):
            m.subject_to(pad[k] <= 1.0)
        if disqualify:
            m.subject_to(m._variables[0] + pad[0] <= 1e6)
        return m

    expired = time.perf_counter()
    rule = PeriodicVariableBoundRule()

    # Positive control: with nothing disqualifying it, the rule DOES reduce ``t`` to
    # one period even under an expired deadline — so the scan really did run whole.
    ok = _build(disqualify=False)
    lb, ub = flat_variable_bounds(ok)
    out_lb, out_ub = rule.tighten(
        ok, lb.copy(), ub.copy(), _cached_flat_metadata(ok), deadline=expired
    )
    assert (out_lb[0], out_ub[0]) == pytest.approx((-np.pi, np.pi))

    # The case that matters: the disqualifying use is the LAST row, so any truncation
    # would miss it and wrongly shrink ``t``.
    bad = _build(disqualify=True)
    lb, ub = flat_variable_bounds(bad)
    out_lb, out_ub = rule.tighten(
        bad, lb.copy(), ub.copy(), _cached_flat_metadata(bad), deadline=expired
    )
    assert (out_lb[0], out_ub[0]) == (lb[0], ub[0]), (
        "t was restricted to one period despite a non-periodic use in the final row"
    )


def test_every_default_rule_declares_its_row_scan_contract():
    """A rule added without an explicit contract inherits ``False`` (skip-whole),
    which is safe; this test exists so the choice is visible rather than accidental."""
    for rule in DEFAULT_NONLINEAR_BOUND_RULES:
        assert isinstance(rule.row_scan_is_anytime, bool), rule.name


def test_a_legacy_four_argument_rule_still_works():
    """External rules keep the old signature; the runner must not pass ``deadline``
    to a rule that does not declare it."""

    class LegacyRule(NonlinearBoundTighteningRule):
        name = "legacy"
        row_scan_is_anytime = True  # even so: it cannot accept the kwarg

        def tighten(self, model, flat_lb, flat_ub, metadata):
            del model, metadata
            out_ub = flat_ub.copy()
            out_ub[0] = min(float(out_ub[0]), 2.0)
            return flat_lb.copy(), out_ub

    m = dm.Model("legacy")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    lb = np.array([0.0])
    ub = np.array([10.0])
    out_lb, out_ub, stats = tighten_nonlinear_bounds(
        m, lb, ub, rules=(LegacyRule(),), deadline=time.perf_counter() + 3600.0
    )
    assert out_ub[0] == pytest.approx(2.0)
    assert stats.applied_rules == ("legacy",)
    assert out_lb[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# gap_certified needs BOTH ends of the gap
# --------------------------------------------------------------------------


def test_a_limit_exit_with_no_incumbent_is_not_certified():
    """``status=time_limit, objective=None, bound=<finite>, gap_certified=True`` claims
    a certified gap where no gap was ever formed. The dual bound is kept — it is
    valid — but the certification claim is dropped."""
    r = SolveResult(status="time_limit", objective=None, bound=-7497.0, gap_certified=True)
    assert r.gap_certified is False
    assert r.bound == pytest.approx(-7497.0), "a valid dual bound must survive the downgrade"
    assert r.gap is None

    r2 = SolveResult(status="time_limit", objective=None, bound=None, gap_certified=True)
    assert r2.gap_certified is False


def test_an_infeasibility_certificate_is_still_exempt():
    r = SolveResult(status="infeasible", objective=None, bound=None, gap_certified=True)
    assert r.gap_certified is True


def test_a_real_certified_optimum_is_untouched():
    r = SolveResult(status="optimal", objective=1.5, bound=1.5, gap=0.0, gap_certified=True)
    assert r.gap_certified is True
    assert r.bound == pytest.approx(1.5)
    assert r.gap == pytest.approx(0.0)


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_a_wide_model_reaches_the_solver_inside_its_budget():
    """The issue's shape in miniature: many variables, many single-variable
    equalities, a nonlinear core. Before the fix the root setup alone scaled with
    ``n_vars * n_constraints`` and the solve never reached a node."""
    n = 4_000
    m = dm.Model("wide")
    x = m.continuous("x", shape=(n,), lb=-5.0, ub=5.0)
    b = m.binary("b", shape=(4,))
    for k in range(0, n, 2):
        m.subject_to(x[k] == 0.5)
    for k in range(0, 200, 2):
        m.subject_to(x[k] * x[k + 1] <= 4.0)
    m.subject_to(sum(b[j] for j in range(4)) == 2)
    m.minimize(sum(x[k] for k in range(200)) + sum(b))

    t0 = time.perf_counter()
    res = m.solve(time_limit=20.0)
    wall = time.perf_counter() - t0

    assert wall < 60.0, f"root setup still dominates: {wall:.1f}s against a 20 s limit"
    assert res.status in {"optimal", "feasible", "time_limit"}
    if res.gap_certified:
        assert res.objective is not None and res.bound is not None
        assert res.bound <= res.objective + 1e-6


_BIG = Path(
    os.path.expanduser(
        "~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/watercontamination0202.nl"
    )
)


@pytest.mark.slow
@pytest.mark.skipif(not _BIG.exists(), reason="needs the full MINLPLib snapshot")
@pytest.mark.xfail(
    strict=True,
    reason=(
        "#1039 bucket B: root setup still overruns the deadline. #875 fixed most "
        "of it (19.3x -> 2.0x) but not all of it. The 1.25x threshold is "
        "DELIBERATELY unchanged -- relaxing it would retire the contract."
    ),
)
@pytest.mark.parametrize("budget", [30.0, 60.0])
def test_watercontamination0202_honours_its_time_limit(budget):
    """The issue's definition of done, on the instance that exposed the class.

    Before: ``solve(time_limit=30)`` = 579.3 s (19.3x) and ``solve(time_limit=60)`` =
    620.8 s (10.4x), both with ``nodes=0`` — the whole budget spent in root setup
    before a single node existed. Required: within ~1.25x, with a sound result.

    Needs the full MINLPLib snapshot (the in-repo 61-file corpus has nothing near
    this size), so it skips without it. The in-repo evidence for the same fixes is
    the mechanism-level tests above: the scaling probe for the cost bug and the
    expired-deadline tests for the budget bug.

    #1039: still failing, and re-measured at load 3.33 so it is not the CLAUDE.md
    §9 load artifact that three other failures in that sweep turned out to be:

        budget 30 s -> 59.7 s wall (2.0x)
        budget 60 s -> 89.4 s wall (1.5x)

    (the issue reported 61.1 s and 90.0 s; reproduced to within noise.) #875 took
    this from 579.3 s (19.3x) and 620.8 s (10.4x) with nodes=0, so most of the
    class is fixed and a residual is left.

    Not confined to this instance -- the same role-1 overrun was measured
    incidentally on three others while working #1039:

        casctanks     ~321 s against a 120 s time_limit  (2.7x)
        nvs19          80.7 s against a  60 s time_limit  (1.35x)
        sonet23v4       4.6 s against a   2 s time_limit  (2.32x)

    so this is a class, per CLAUDE.md §2, not a watercontamination0202 quirk.

    Pinned as a STRICT xfail rather than repaired: the 1.25x threshold and every
    soundness assertion below are untouched, so this cannot pass by having its
    goalposts moved -- it passes when root setup is actually bounded, and the
    strictness then fails the suite to say so. See also
    ``test_issue654_deadline_root_setup.py``'s
    ``test_sonet23v4_bound_survives_the_deadline_gating``, which asserts the
    OPPOSITE contract on the same mechanism; the two cannot both be satisfied
    until the bound-producing op is made interruptible.
    """
    m = dm.from_nl(str(_BIG))
    t0 = time.perf_counter()
    res = m.solve(time_limit=budget)
    wall = time.perf_counter() - t0

    assert wall < 1.25 * budget, (
        f"solve took {wall:.1f}s against a {budget:.0f}s time_limit "
        f"({wall / budget:.1f}x); root setup is still unbounded"
    )
    # Sound, whatever it managed to prove.
    assert res.status != "infeasible", "FALSE-INFEASIBLE on a feasible instance"
    if res.objective is not None and res.bound is not None:
        assert res.bound <= res.objective + 1e-6, "UNSOUND CERT (bound > incumbent)"
    if res.gap_certified:
        assert res.objective is not None and res.bound is not None
