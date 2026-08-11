"""#966: a heuristic sub-NLP must be capped by the deadline, not just polled.

THE DEFECT. ``bchoco07``, ``bchoco08`` and ``heatexch_gen3`` overran a 20 s
``solve(time_limit=...)`` by 2.4–4.7 s in EVERY arm of the #966 coupled panel —
all three flags ON and all three OFF, within ~0.5 s of each other. That makes it
a defect of the DEFAULT configuration rather than a flag effect. A post-deadline
stack sampler attributed 100 % of its 73 post-deadline samples to a single
``nlp_pounce.solve_nlp`` call: 91.8 % reached from ``feasibility_pump``, 8.2 %
from ``integer_local_search``.

THE CAUSE. Every deadline guard in ``primal_heuristics`` gated whether a solve
*starts*. None bounded how long the started solve *runs*. Polling therefore caps
the NUMBER of overshooting solves at one and says nothing about its duration; the
pump's round 0 is not polled at all. ``_HEURISTIC_NLP_MAX_ITER`` does not help —
an iteration cap is not a time cap, and one IPM iteration carrying an exact
Hessian is seconds long on the no-relaxation flowsheet class.

THE FIX, and what these tests pin:
  * every deadline-aware heuristic derives ``max_wall_time`` for its sub-NLP from
    the remaining budget (``_deadline_wall_cap``);
  * a caller that passes NO deadline is bit-for-bit unchanged (no key added);
  * an explicit caller ``max_wall_time`` still wins;
  * ``feasibility_pump`` ACCEPTS a TIME_LIMIT point — a cap whose own truncated
    points are discarded merely trades an overrun for a lost incumbent — but
    accepts it only through the SAME independent re-verification that already
    licenses ``subnlp``'s ITERATION_LIMIT, never by relaxing a feasibility check.

The last bullet is the soundness pin: ``test_time_limited_but_INFEASIBLE_point_is_still_rejected``
must fail loudly if anyone ever widens the acceptance set without the
verification behind it.

Deterministic by construction: every test drives a recording fake backend, so
nothing here depends on machine speed or on POUNCE being installed.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._relax import primal_heuristics as ph  # noqa: E402
from discopt._tape_nlp_evaluator import make_evaluator as cached_evaluator  # noqa: E402
from discopt.solvers import NLPResult, SolveStatus  # noqa: E402

pytestmark = pytest.mark.unit


def _small_minlp() -> dm.Model:
    """A 3-binary/1-continuous MINLP whose rounding is trivially repairable."""
    m = dm.Model("c")
    y = m.binary("y", shape=(3,))
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x + y[0] + 2 * y[1] + 3 * y[2])
    m.subject_to(x + y[0] + y[1] + y[2] >= 1.0)
    return m


def _recorder(status=SolveStatus.OPTIMAL, x_of=None):
    """A fake NLP backend that records the options dict it was handed."""
    captured: list[dict] = []

    def backend(evaluator, x0, options=None):
        captured.append(dict(options or {}))
        x = np.asarray(x0, dtype=float) if x_of is None else x_of(np.asarray(x0, dtype=float))
        return NLPResult(
            status=status,
            x=x,
            objective=float(np.sum(x)),
            multipliers=None,
            bound_multipliers_lower=None,
            bound_multipliers_upper=None,
            iterations=1,
            wall_time=0.0,
        )

    return backend, captured


# ── the cap helper itself ──────────────────────────────────────────────────────


def test_no_deadline_means_no_cap():
    """A caller that never passed a deadline keeps its unclamped behaviour."""
    assert ph._deadline_wall_cap(None) is None
    assert ph._deadline_wall_cap(float("inf")) is None


def test_cap_shrinks_with_the_remaining_budget_and_never_exceeds_the_ceiling():
    now = ph._now()
    # Plenty of budget left: the constant ceiling binds, not the deadline.
    assert ph._deadline_wall_cap(now + 600.0) == pytest.approx(ph._DEADLINE_NLP_CAP_S)
    # Little budget left: the deadline binds.
    tight = ph._deadline_wall_cap(now + 1.0)
    assert 0.0 < tight <= 1.0


def test_expired_deadline_yields_a_positive_floor_not_zero_or_negative():
    """A zero/negative ``max_wall_time`` has backend-defined meaning; never emit one."""
    assert ph._deadline_wall_cap(ph._now() - 50.0) == pytest.approx(ph._DEADLINE_NLP_FLOOR_S)
    assert ph._DEADLINE_NLP_FLOOR_S > 0.0


# ── feasibility_pump ───────────────────────────────────────────────────────────


def test_pump_caps_each_solve_from_the_deadline():
    """THE REGRESSION: fails before the fix — the pump forwarded no wall cap."""
    m = _small_minlp()
    ev = cached_evaluator(m)
    backend, captured = _recorder(status=SolveStatus.INFEASIBLE)

    ph.feasibility_pump(
        m,
        np.array([0.4, 0.6, 0.5, 1.0]),
        backend=backend,
        evaluator=ev,
        deadline=ph._now() + 1.0,
    )

    assert captured, "pump never invoked the NLP backend (vacuous pass)"
    for o in captured:
        assert "max_wall_time" in o, "a pump solve ran with no wall cap under a deadline"
        assert 0.0 < o["max_wall_time"] <= ph._DEADLINE_NLP_CAP_S


def test_pump_without_a_deadline_is_unchanged():
    m = _small_minlp()
    ev = cached_evaluator(m)
    backend, captured = _recorder(status=SolveStatus.INFEASIBLE)

    ph.feasibility_pump(m, np.array([0.4, 0.6, 0.5, 1.0]), backend=backend, evaluator=ev)

    assert captured, "pump never invoked the NLP backend (vacuous pass)"
    assert all("max_wall_time" not in o for o in captured)


def test_explicit_caller_wall_time_still_wins():
    m = _small_minlp()
    ev = cached_evaluator(m)
    backend, captured = _recorder(status=SolveStatus.INFEASIBLE)

    ph.feasibility_pump(
        m,
        np.array([0.4, 0.6, 0.5, 1.0]),
        backend=backend,
        evaluator=ev,
        deadline=ph._now() + 1.0,
        ipopt_options={"max_wall_time": 99.0},
    )

    assert captured, "pump never invoked the NLP backend (vacuous pass)"
    assert all(o["max_wall_time"] == 99.0 for o in captured)


def test_time_limited_but_feasible_point_is_accepted():
    """Capping is only an improvement if truncated-but-feasible points survive.

    Fails before the fix: the pump's gate accepted OPTIMAL alone, so a clamped
    solve's point was discarded and the cap would have traded an overrun for a
    lost incumbent.
    """
    m = _small_minlp()
    ev = cached_evaluator(m)
    # y = (1,0,0), x = 1 satisfies x + y0 + y1 + y2 >= 1 comfortably.
    feasible = np.array([1.0, 0.0, 0.0, 1.0])
    backend, captured = _recorder(status=SolveStatus.TIME_LIMIT, x_of=lambda _x: feasible.copy())

    out = ph.feasibility_pump(
        m,
        np.array([0.6, 0.4, 0.4, 1.0]),
        backend=backend,
        evaluator=ev,
        deadline=ph._now() + 1.0,
    )

    assert captured, "pump never invoked the NLP backend (vacuous pass)"
    assert out is not None, "a feasible TIME_LIMIT point was discarded"
    np.testing.assert_allclose(out, feasible)


def test_time_limited_but_INFEASIBLE_point_is_still_rejected():
    """SOUNDNESS PIN. Accepting TIME_LIMIT relocates a status hint; it must never
    relax the feasibility evidence. This point violates ``x + y0 + y1 + y2 >= 1``
    (all-zero), and the constraint check — not the status — must reject it."""
    m = _small_minlp()
    ev = cached_evaluator(m)
    infeasible = np.zeros(4)
    backend, captured = _recorder(status=SolveStatus.TIME_LIMIT, x_of=lambda _x: infeasible.copy())

    out = ph.feasibility_pump(
        m,
        np.array([0.6, 0.4, 0.4, 1.0]),
        backend=backend,
        evaluator=ev,
        deadline=ph._now() + 1.0,
    )

    assert captured, "pump never invoked the NLP backend (vacuous pass)"
    assert out is None, "an INFEASIBLE point was accepted on the strength of its status"


def test_pump_does_not_widen_the_shared_feasibility_gate():
    """``_is_nlp_feasible`` is shared with call sites that do NOT re-verify; the
    #966 acceptance widening must stay local to the pump."""
    res = NLPResult(
        status=SolveStatus.TIME_LIMIT,
        x=np.zeros(4),
        objective=0.0,
        multipliers=None,
        bound_multipliers_lower=None,
        bound_multipliers_upper=None,
        iterations=1,
        wall_time=0.0,
    )
    assert ph._is_nlp_feasible(res) is False


# ── the other two attributed paths ─────────────────────────────────────────────


def test_integer_local_search_caps_its_subnlp_solves():
    """8.2 % of the measured post-deadline samples came from this path."""
    m = _small_minlp()
    ev = cached_evaluator(m)
    backend, captured = _recorder(status=SolveStatus.INFEASIBLE)

    ph.integer_local_search(
        m,
        np.array([0.4, 0.6, 0.5, 1.0]),
        backend=backend,
        evaluator=ev,
        deadline=ph._now() + 1.0,
    )

    assert captured, "integer_local_search never invoked the NLP backend (vacuous pass)"
    for o in captured:
        assert "max_wall_time" in o
        assert 0.0 < o["max_wall_time"] <= ph._DEADLINE_NLP_CAP_S


def test_diving_caps_each_dive_step():
    """``diving``'s own comment records heatexch_gen3 running tens of seconds past
    the deadline; the entry poll it gained bounds the step COUNT, not the step."""
    m = _small_minlp()
    ev = cached_evaluator(m)
    backend, captured = _recorder(status=SolveStatus.OPTIMAL)

    ph.diving(
        m,
        np.array([0.4, 0.6, 0.5, 1.0]),
        backend=backend,
        evaluator=ev,
        deadline=ph._now() + 1.0,
    )

    assert captured, "diving never invoked the NLP backend (vacuous pass)"
    for o in captured:
        assert "max_wall_time" in o
        assert 0.0 < o["max_wall_time"] <= ph._DEADLINE_NLP_CAP_S
