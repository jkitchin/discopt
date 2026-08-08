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


def test_nonbinary_first_stage_explains_the_lost_certificate(exact_recourse_bounds, caplog):
    """Item 3 of #946: a *non-binary* first stage has no cheap exact cut, so the
    honest outcome is the uncertified one — but GBD must say why rather than
    silently exhausting its iteration budget."""
    m = dm.Model("linnl_int")
    y = m.integer("y", lb=0, ub=3)
    x = m.continuous("x", shape=(2,), lb=0, ub=5)
    m.first_stage(y)
    m.minimize(3 * y - x[0] - x[1])
    m.subject_to(x[0] * x[0] + x[1] * x[1] <= 8 * y)

    with caplog.at_level(logging.WARNING, logger="discopt.decomposition.benders.gbd"):
        r = solve_benders(m, time_limit=60)

    assert exact_recourse_bounds.saw_degenerate_multiplier
    # Sound, just uncertified: the bound never exceeds the true optimum (-1).
    assert r.bound is None or r.bound <= -1.0 + 1e-6
    assert r.objective == pytest.approx(-1.0, abs=1e-3)
    text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "degenerate" in text, f"no explanation logged; got: {text!r}"
    # And it stopped early instead of burning all 100 iterations on a master
    # that keeps re-proposing the same point.
    assert exact_recourse_bounds.calls <= 10, (
        f"{exact_recourse_bounds.calls} recourse solves: the stall was not detected"
    )


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
