"""Issue #911 — a DECLINED convex-kernel attempt must be deducted from the budget.

``Model.solve`` hands the convex kernel ``min(time_limit,
DISCOPT_CONVEX_KERNEL_BUDGET)`` seconds, adopts the result ONLY when it certifies
optimality, and then runs the default path. Before this fix the default path was
handed the caller's **full** ``time_limit`` again, so an eligible-but-uncertifiable
model paid the attempt on top of its whole budget: ``solve(time_limit=T)`` ran for
~2T. Measured on ``clay0303hfsg`` at a 10 s budget, the replicate whose attempt did
not certify ran **22.35 s** against an OFF arm of 10.88 s.

The tests below pin, in order of what actually protects the contract:

1. **the arithmetic** — a declined attempt of a known duration is subtracted from the
   budget the default path receives (deterministic; no wall-clock threshold);
2. **flag-off exactness** — with ``DISCOPT_CONVEX_KERNEL`` off the deduction is
   *literally* ``0.0`` and the forwarded budget is bit-identical, so the default path
   is untouched. This is asserted rather than assumed: a nonzero reading here would
   perturb every deadline-sensitive decision in the solver;
3. **end to end** — a real eligible instance under a budget the kernel provably
   cannot meet stays inside its stated limit, and the test refuses to pass vacuously
   (it asserts the attempt really ran and really declined).
"""

from __future__ import annotations

import os
import time

import discopt.modeling as dm
import pytest

_ck = pytest.importorskip("discopt.solvers._convex_kernel")

_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")

# Long enough to dwarf the sub-millisecond noise of the surrounding bookkeeping, short
# enough to keep the deterministic tests fast.
_FAKE_ATTEMPT_S = 0.75
_TIME_LIMIT = 20.0


def _box_model():
    """A model with no constraints: the #844 fallback reserve and the #772
    verification snapshot both key off ``self._constraints``, so a box-only model
    isolates the budget arithmetic under test from both."""
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize(x)
    return m


def _reserve_model():
    """A model the #844 LP-spatial fallback is IN SCOPE for (pure-integer, minimize,
    with an integer product), so ``Model.solve`` carves out its 35% reserve. #911
    changed that reserve from 35% of the stated limit to 35% of what remains after
    the kernel attempt; this model is how that path gets exercised."""
    m = dm.Model()
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.constraint(dm.RangeSet(1), lambda _i: x * y >= 4.0, name="prod", fast=False)
    m.minimize(x + y)
    return m


def _patched_solve(monkeypatch, *, kernel_on: bool, attempt_s: float, model_fn=None):
    """Solve ``_box_model()`` with a declining attempt of ``attempt_s`` seconds and
    return ``(time_limit the default path was handed, number of attempts made)``."""
    monkeypatch.setenv("DISCOPT_CONVEX_KERNEL", "1" if kernel_on else "0")

    calls = {"spec": 0, "solve_model": 0}

    def _slow_decline(model):
        # Stands in for a real ``build_convex_spec`` + kernel run that burns the
        # budget and then declines. Declining through ``build_convex_spec`` exercises
        # the real ``try_convex_solve`` clock rather than a stubbed one.
        calls["spec"] += 1
        time.sleep(attempt_s)
        return None

    monkeypatch.setattr(_ck, "build_convex_spec", _slow_decline)

    import discopt.solver as _solver
    from discopt.modeling.core import SolveResult

    seen: dict[str, float] = {}

    def _fake_solve_model(model, **kw):
        calls["solve_model"] += 1
        seen["time_limit"] = kw["time_limit"]
        return SolveResult(status="optimal", objective=0.0, bound=0.0, gap=0.0)

    monkeypatch.setattr(_solver, "solve_model", _fake_solve_model)

    (model_fn or _box_model)().solve(time_limit=_TIME_LIMIT)

    # CLAUDE.md §6: a probe that never fired must not read as a pass.
    assert calls["solve_model"] == 1, "the default path was never reached"
    assert calls["spec"] == (1 if kernel_on else 0), (
        f"expected {'an' if kernel_on else 'no'} attempt, got {calls['spec']}"
    )
    return seen["time_limit"], calls["spec"]


def test_declined_attempt_is_deducted_from_the_default_path_budget(monkeypatch):
    """The wall a declined attempt spent is gone from the caller's budget."""
    forwarded, attempts = _patched_solve(monkeypatch, kernel_on=True, attempt_s=_FAKE_ATTEMPT_S)
    assert attempts == 1
    # Deducted, and not over-deducted: the attempt slept at least _FAKE_ATTEMPT_S, so
    # the forwarded budget sits just below ``time_limit - _FAKE_ATTEMPT_S``.
    assert forwarded <= _TIME_LIMIT - _FAKE_ATTEMPT_S, (
        f"attempt of {_FAKE_ATTEMPT_S}s not deducted: default path got {forwarded}s "
        f"of a {_TIME_LIMIT}s budget"
    )
    assert forwarded > _TIME_LIMIT - _FAKE_ATTEMPT_S - 1.0, (
        f"over-deducted: default path got {forwarded}s"
    )


def test_flag_off_deducts_exactly_zero(monkeypatch):
    """Flag-off must forward the caller's budget bit-identically."""
    forwarded, attempts = _patched_solve(monkeypatch, kernel_on=False, attempt_s=_FAKE_ATTEMPT_S)
    assert attempts == 0, "no attempt may be made with the flag off"
    # Exact equality, deliberately: `==`, not `approx`. The deduction is subtracted
    # from every downstream deadline, so "close to zero" is not the contract.
    assert forwarded == _TIME_LIMIT, f"flag-off budget drifted: {forwarded!r} != {_TIME_LIMIT!r}"
    assert _ck.last_attempt_seconds() == 0.0, (
        f"flag-off attempt clock is not literally 0.0: {_ck.last_attempt_seconds()!r}"
    )


def test_flag_off_is_exact_on_the_fallback_reserve_path_too(monkeypatch):
    """The #844 reserve now takes 35% of the REMAINING budget, not of the stated
    limit. With the flag off the remainder IS the stated limit, so the primary budget
    must come out bit-identical to the pre-#911 ``time_limit - 0.35*time_limit``."""
    from discopt._jax.lp_spatial_bb import _is_in_scope
    from discopt.modeling.core import _lp_spatial_fallback_enabled

    # Non-vacuity: if the reserve is not actually carved out here, this test would
    # silently be re-checking the box-only path (CLAUDE.md §6).
    if not _lp_spatial_fallback_enabled() or not _is_in_scope(_reserve_model()):
        pytest.skip("the #844 reserve does not apply to this model -- nothing to pin")

    forwarded, attempts = _patched_solve(
        monkeypatch, kernel_on=False, attempt_s=0.0, model_fn=_reserve_model
    )
    assert attempts == 0
    expected = _TIME_LIMIT - 0.35 * _TIME_LIMIT
    assert forwarded == expected, f"reserve arithmetic drifted: {forwarded!r} != {expected!r}"


def test_attempt_clock_is_zero_before_any_attempt():
    """A thread that has never run an attempt reads 0.0, not garbage."""
    import threading

    seen: list[float] = []
    t = threading.Thread(target=lambda: seen.append(_ck.last_attempt_seconds()))
    t.start()
    t.join()
    assert seen == [0.0]


@pytest.mark.slow
def test_real_instance_does_not_pay_a_second_full_budget(monkeypatch):
    """End to end on a real eligible instance under a budget the kernel cannot meet.

    ``clay0303hfsg`` is convex-kernel eligible and needs ~8 s to certify on this
    corpus, so a 6 s budget makes the attempt decline. The assertion is the contract
    this fix delivers -- *the default path does not get a second full budget* -- not
    ``wall <= time_limit``: the attempt has its own granularity (the kernel polls its
    deadline between LP re-solves, so it can overshoot by one re-solve), and folding
    that into the threshold would make this test measure the LP layer instead.

    Pre-fix the wall was ``attempt + budget + overhead``; post-fix it is
    ``attempt + overhead``.
    """
    from discopt.modeling.core import from_nl

    nl = os.path.join(_DATA, "clay0303hfsg.nl")
    if not os.path.exists(nl):  # pragma: no cover - corpus always present in-repo
        pytest.skip(f"missing corpus instance {nl}")
    monkeypatch.setenv("DISCOPT_CONVEX_KERNEL", "1")
    budget = 6.0

    # Non-vacuity guard, checked FIRST: the attempt must really run and really
    # decline at this budget, else the timing assertion below means nothing. A
    # skip (not a pass) if it does not, so a vacuous run is visible in the report.
    if _ck.try_convex_solve(from_nl(nl), time_limit=budget) is not None:
        pytest.skip("kernel certified within the budget -- no declined attempt to test")
    probe_attempt = _ck.last_attempt_seconds()
    assert probe_attempt > 0.5 * budget, (
        f"attempt consumed only {probe_attempt:.2f}s of a {budget}s budget -- it did "
        f"not spend the budget, so this instance no longer exercises the hazard"
    )

    t0 = time.perf_counter()
    from_nl(nl).solve(time_limit=budget)
    wall = time.perf_counter() - t0
    attempt = _ck.last_attempt_seconds()
    assert attempt > 0.5 * budget, "the timed solve did not make a real attempt"

    # 0.8, not 1.0: pre-fix the default path got the WHOLE budget on top of the
    # attempt, so anything at or above `attempt + budget` is the old behaviour and
    # the margin below it absorbs presolve/teardown overhead.
    assert wall < attempt + 0.8 * budget, (
        f"solve(time_limit={budget}) ran {wall:.2f}s with a {attempt:.2f}s convex-"
        f"kernel attempt -- the default path was handed a second full budget "
        f"instead of the {budget - attempt:.2f}s that remained"
    )
