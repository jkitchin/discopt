"""Issue #946: GBD must not depend on the recourse NLP returning a *finite*
multiplier.

The Lagrangian optimality cut in ``decomposition/benders/gbd.py`` is built from
whatever multipliers the recourse NLP returns. When the recourse subproblem is
**degenerate at its solution** (Slater's condition fails: the feasible set has
empty interior) no finite multiplier exists, the returned one diverges, and the
cut — still sound — becomes tight at its own first-stage point and vacuous one
step away. The master bound then stalls and GBD exits ``iteration_limit``
instead of ``optimal``: a lost *certificate*, not a lost soundness guarantee.

The dependency that hides this is Ipopt's ``bound_relax_factor`` (default 1e-8),
which gives a degenerate feasible set an artificial interior and keeps the
multiplier finite. These tests force ``bound_relax_factor = 0`` on the recourse
solve, which is exactly the arm issue #946 measured:

    recourse x at y=0   multiplier   cut slope    outcome (before the fix)
    7.07e-05 (relaxed)  7.07e+03     -5.66e+04    optimal, bound -1
    6.93e-09 (exact)    9.82e+07     -7.86e+08    iteration_limit, bound -2

The fix is the multiplier-free **integer L-shaped optimality cut** (Laporte &
Louveaux 1993) added alongside the Lagrangian cut at an all-0/1 first stage.

Every test here first asserts that the degeneracy it is meant to exercise is
actually present (a multiplier above ``DEGENERATE_MU``), so the file cannot
silently pass by failing to reproduce (CLAUDE.md §6).
"""

import logging

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.decomposition.benders import solve_benders

try:
    from discopt.solvers.lp_pounce import POUNCE_AVAILABLE
except ImportError:
    POUNCE_AVAILABLE = False
try:
    import highspy  # noqa: F401

    HAS_HIGHS = True
except ImportError:
    HAS_HIGHS = False

pytestmark = pytest.mark.skipif(
    not (POUNCE_AVAILABLE or HAS_HIGHS), reason="no LP/MILP backend available"
)

#: A multiplier above this is unusable at the objective scale of these models
#: (the measured degenerate value is 9.8e7; the non-degenerate arm is 7.1e3).
DEGENERATE_MU = 1e6


class _Taps:
    """Records every recourse-NLP return so a test can prove what it exercised."""

    def __init__(self):
        self.mu_max = 0.0
        self.calls = 0

    @property
    def saw_degenerate_multiplier(self) -> bool:
        return self.mu_max > DEGENERATE_MU


@pytest.fixture
def exact_recourse_bounds(monkeypatch):
    """Force ``bound_relax_factor = 0`` on GBD's recourse NLP and tap its returns.

    This removes the artificial interior that keeps a degenerate multiplier
    finite — the dependency #946 is about. Yields the tap record.
    """
    import discopt.solvers.nlp_pounce as nlp_pounce

    taps = _Taps()
    real = nlp_pounce.solve_nlp

    def patched(problem, x0, options=None):
        opts = dict(options or {})
        opts["bound_relax_factor"] = 0.0
        res = real(problem, x0, options=opts)
        taps.calls += 1
        if res.multipliers is not None and len(res.multipliers):
            taps.mu_max = max(taps.mu_max, float(np.max(np.abs(res.multipliers))))
        return res

    monkeypatch.setattr(nlp_pounce, "solve_nlp", patched)
    return taps


def _degenerate_model():
    """``min 3y - x0 - x1`` s.t. ``x0^2 + x1^2 <= 8y``, y binary (optimum -1).

    At the master proposal ``y = 0`` the recourse constraint collapses to
    ``x0^2 + x1^2 <= 0``: the feasible set is the single point ``x = 0`` and its
    Jacobian ``2x`` vanishes there too, so the multiplier diverges like ``1/x``.
    """
    m = dm.Model("linnl")
    y = m.binary("y")
    x = m.continuous("x", shape=(2,), lb=0, ub=5)
    m.first_stage(y)
    m.minimize(3 * y - x[0] - x[1])
    m.subject_to(x[0] * x[0] + x[1] * x[1] <= 8 * y)
    return m


def test_degenerate_recourse_still_certifies(exact_recourse_bounds):
    """The headline regression: a diverging multiplier must not cost the
    certificate. Before the integer L-shaped cut this returned
    ``iteration_limit`` with bound -2 against an optimum of -1."""
    r = solve_benders(_degenerate_model(), time_limit=60)

    assert exact_recourse_bounds.saw_degenerate_multiplier, (
        "reproducer no longer degenerate (max |mu| = "
        f"{exact_recourse_bounds.mu_max:.3e}); this test would pass for the wrong reason"
    )
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-1.0, abs=1e-3)
    assert r.bound is not None
    assert r.bound <= r.objective + 1e-3
    # Soundness: the bound may not exceed the true optimum.
    assert r.bound <= -1.0 + 1e-6


def test_degenerate_recourse_converges_without_exhausting_the_budget(exact_recourse_bounds):
    """The exact cut closes the gap immediately; it must not merely arrive on
    the last of the 100 default iterations. Two first-stage points (y=0, y=1)
    suffice, so the recourse NLP is solved a handful of times, not ~100."""
    r = solve_benders(_degenerate_model(), time_limit=60)
    assert exact_recourse_bounds.saw_degenerate_multiplier
    assert r.status == "optimal"
    assert exact_recourse_bounds.calls <= 10, (
        f"{exact_recourse_bounds.calls} recourse solves: the master is still stalling"
    )


def test_bound_relaxed_arm_unchanged():
    """Control: with the backend default ``bound_relax_factor`` (the arm that
    always worked) the answer is the same. The fix must not move it."""
    r = solve_benders(_degenerate_model(), time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-1.0, abs=1e-3)
    assert r.bound is not None and r.bound <= -1.0 + 1e-6


@pytest.mark.correctness
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_lshaped_cut_soundness_panel(seed, exact_recourse_bounds):
    """The integer L-shaped cut is added on every all-0/1 master point, so its
    validity is checked here on models that are *not* degenerate too: the GBD
    bound must never exceed the monolithic optimum."""
    rng = np.random.default_rng(seed)
    ny, nx = 2, 3
    m = dm.Model("panel")
    y = m.binary("y", shape=(ny,))
    x = m.continuous("x", shape=(nx,), lb=0, ub=4)
    m.first_stage(y)
    a = rng.uniform(0.5, 2.0, nx)
    cy = rng.uniform(1, 3, ny)
    tgt = rng.uniform(2, 5)
    m.minimize(
        sum(float(a[j]) * x[j] * x[j] for j in range(nx))
        + sum(float(cy[i]) * y[i] for i in range(ny))
    )
    m.subject_to(sum(x[j] for j in range(nx)) >= tgt)
    for j in range(nx):
        m.subject_to(x[j] <= 4 * y[j % ny])

    r = solve_benders(m, time_limit=60)
    mono = m.solve(time_limit=60)
    assert mono.objective is not None
    if r.bound is not None:
        assert r.bound <= mono.objective + 1e-3, (
            f"unsound GBD bound {r.bound} > monolithic optimum {mono.objective}"
        )
    if r.objective is not None:
        assert r.objective == pytest.approx(mono.objective, abs=1e-2)


def _nonbinary_degenerate_model():
    """``_degenerate_model`` with an *integer* first stage ``y in [0, 3]``.

    Same degeneracy at ``y = 0``; no all-0/1 first stage, so the multiplier-free
    integer L-shaped cut is unavailable.
    """
    m = dm.Model("linnl_int")
    y = m.integer("y", lb=0, ub=3)
    x = m.continuous("x", shape=(2,), lb=0, ub=5)
    m.first_stage(y)
    m.minimize(3 * y - x[0] - x[1])
    m.subject_to(x[0] * x[0] + x[1] * x[1] <= 8 * y)
    return m


def test_nonbinary_first_stage_explains_the_lost_certificate(exact_recourse_bounds, caplog):
    """Item 3 of #946: a *non-binary* first stage has no cheap exact cut, so when
    the budget runs out before the master bound catches up the honest outcome is
    the uncertified one — and GBD must say why rather than exhausting the budget
    silently.

    The budget is capped at one iteration to reach that exit deterministically.
    It used to be reached with the default 100-iteration budget, because the MILP
    master promoted an incumbent that was outside its own declared rows and the
    resulting cut sequence stalled; #952 stops that promotion and the uncapped
    run now certifies (pinned by the companion test below). Capping is what keeps
    this test about the *explanation* rather than about a stall that has since
    been fixed — the assertion below would otherwise pass vacuously or not at all.

    #977 note: this used to assert ``r.objective == approx(-1.0)`` — that a single
    iteration already lands on the optimum. That held only by an arbitrary
    tie-break. The whole cost here is nonlinear and lives in the recourse, so the
    first master solve has a **zero objective** (measured: ``master_obj = 0.0``
    under both engines) and *every* feasible first-stage point is optimal for it.
    The old POUNCE-IPM-backed master happened to return ``y = 1`` — the eventual
    optimum — while the exact-vertex simplex #977 pins the master to returns
    ``y = 0``; both are optima of that master. One iteration is not enough to reach
    ``-1`` and must not be asserted to be. What *is* invariant is that the run
    brackets the true optimum, so that is what is asserted now: the bound never
    rises above -1 and the incumbent never falls below it. The full-budget
    companion below still pins the optimum itself, and still reaches it in <= 10
    recourse solves.
    """
    with caplog.at_level(logging.WARNING, logger="discopt.decomposition.benders.gbd"):
        r = solve_benders(_nonbinary_degenerate_model(), time_limit=60, max_iterations=1)

    assert exact_recourse_bounds.saw_degenerate_multiplier, (
        "reproducer no longer degenerate (max |mu| = "
        f"{exact_recourse_bounds.mu_max:.3e}); this test would pass for the wrong reason"
    )
    assert r.status == "iteration_limit", (
        f"expected the uncertified exit at max_iterations=1, got {r.status!r}"
    )
    # Sound, just uncertified: the run brackets the true optimum (-1) from both
    # sides. The bound never rises above it...
    assert r.bound is None or r.bound <= -1.0 + 1e-6
    # ...and the incumbent, being a genuinely feasible point, never falls below it.
    # (Its *distance* from -1 after one iteration is the tie-break artifact
    # described above, so it is deliberately not pinned.)
    assert r.objective is not None
    assert r.objective >= -1.0 - 1e-6, (
        f"incumbent {r.objective!r} beats the true optimum -1.0: not a feasible point"
    )
    text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "degenerate" in text, f"no explanation logged; got: {text!r}"


def test_nonbinary_first_stage_certifies_with_the_full_budget(exact_recourse_bounds):
    """The other half of the above: with its default budget the non-binary arm
    closes the gap, and its bound stays sound while doing so.

    This is what #952 bought — before it, the master's incumbent could sit
    outside the master's own rows, the cut added there carried no usable eta
    information, and GBD burned its budget re-proposing the same point.
    """
    r = solve_benders(_nonbinary_degenerate_model(), time_limit=60)

    assert exact_recourse_bounds.saw_degenerate_multiplier
    assert r.bound is None or r.bound <= -1.0 + 1e-6
    assert r.objective == pytest.approx(-1.0, abs=1e-3)
    assert r.status == "optimal"
    # And it got there without burning all 100 iterations.
    assert exact_recourse_bounds.calls <= 10, (
        f"{exact_recourse_bounds.calls} recourse solves: the master is stalling"
    )


@pytest.mark.correctness
@pytest.mark.parametrize("shape", ["unbounded_above", "free"])
def test_lshaped_floor_is_sound_with_unbounded_recourse_variables(shape):
    """The floor ``L`` is a closed-form box minimum, so an *unbounded* recourse
    column is the shape that can break it: ``_box_min_linear`` skips a column
    whose gradient component is below ``_STATIONARY_TOL`` before it looks at
    that column's bound, which reports 0 where the exact minimum is -inf. That
    is sound only because the threshold sits below the NLP's dual tolerance
    (see the constant's comment) — this pins the consequence rather than the
    argument: every floor GBD registers must stay at or below the monolithic
    optimum, and so must the bound built from it.

    ``test_lshaped_cut_soundness_panel`` covers the cut on *bounded* recourse
    boxes, where the skip cannot fire at all.
    """
    from discopt.decomposition.benders import gbd

    def build():
        m = dm.Model(f"unb_{shape}")
        y = m.binary("y", shape=(2,))
        if shape == "unbounded_above":
            x = m.continuous("x", shape=(3,), lb=0)  # no upper bound
            m.first_stage(y)
            m.minimize(sum(x[j] * x[j] for j in range(3)) + 2.0 * y[0] + 3.0 * y[1])
            m.subject_to(x[0] + x[1] + x[2] >= 3.0)
            for j in range(3):
                m.subject_to(x[j] <= 4 * y[j % 2])
        else:
            x = m.continuous("x", shape=(2,))  # free in both directions
            m.first_stage(y)
            m.minimize((x[0] - 1.0) ** 2 + (x[1] + 2.0) ** 2 + 2.0 * y[0] + 3.0 * y[1])
            m.subject_to(x[0] + x[1] >= 0.5 + y[0])
            m.subject_to(x[0] - x[1] <= 3.0 * y[1])
        return m

    floors: list[float] = []
    real_cls = gbd._Recourse

    class _Tapped(real_cls):  # type: ignore[misc,valid-type]
        def __new__(cls, *a, **kw):
            obj = super().__new__(cls, *a, **kw)
            if obj.floor is not None and np.isfinite(obj.floor):
                floors.append(float(obj.floor))
            return obj

    gbd._Recourse = _Tapped
    try:
        r = solve_benders(build(), time_limit=60)
    finally:
        gbd._Recourse = real_cls

    mono = build().solve(time_limit=60)
    assert mono.objective is not None
    # The tap must have fired, or this asserts nothing (CLAUDE.md §6).
    assert floors, "no objective floor was registered; the cut was never built"
    worst = max(floors)
    assert worst <= mono.objective + 1e-6, (
        f"objective floor {worst} exceeds the monolithic optimum {mono.objective}: "
        "it is not a valid global lower bound"
    )
    if r.bound is not None:
        assert r.bound <= mono.objective + 1e-3, (
            f"unsound GBD bound {r.bound} > monolithic optimum {mono.objective}"
        )


def test_stationary_tolerance_stays_below_the_nlp_dual_tolerance():
    """``_box_min_linear``'s skip is sound on an unbounded column only because a
    component that small is roundoff on a direction the NLP drove to
    stationarity. Raising the threshold above the NLP's own dual tolerance
    (1e-8) breaks that argument and inflates the anchor and the floor by up to
    ``tol * 1e20`` in the unsafe direction."""
    from discopt.decomposition.benders.gbd import _STATIONARY_TOL

    assert _STATIONARY_TOL < 1e-8


def test_box_min_linear_closed_form():
    """Unit test for the closed-form box minimum both the recourse anchor and the
    global objective floor are built from."""
    from discopt.decomposition.benders.gbd import _box_min_linear

    lb = np.array([0.0, 0.0, -1.0])
    ub = np.array([1.0, 5.0, 5.0])
    x_ref = np.array([0.0, 0.0, 0.0])

    # g > 0 -> minimized at the lower bound; g < 0 -> at the upper bound.
    val, finite = _box_min_linear(np.array([3.0, -1.0, -1.0]), x_ref, range(3), lb, ub)
    assert finite
    assert val == pytest.approx(3.0 * (0.0 - 0.0) + (-1.0) * 5.0 + (-1.0) * 5.0)

    # Stationary components contribute nothing (and do not need a finite bound).
    val, finite = _box_min_linear(np.array([0.0, 0.0, 0.0]), x_ref, range(3), lb, ub)
    assert finite and val == 0.0

    # A non-stationary component unbounded in its descent direction: the true
    # minimum is -inf, so there is no finite underestimator.
    val, finite = _box_min_linear(
        np.array([0.0, -1.0, 0.0]), x_ref, range(3), lb, np.array([1.0, np.inf, 5.0])
    )
    assert not finite
    # The 1e20 sentinel counts as infinite too (INF in the LP layer).
    val, finite = _box_min_linear(
        np.array([0.0, -1.0, 0.0]), x_ref, range(3), lb, np.array([1.0, 1e20, 5.0])
    )
    assert not finite
