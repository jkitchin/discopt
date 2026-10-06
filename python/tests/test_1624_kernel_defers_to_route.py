"""#1624 -- the convex kernel stands aside for a model the convex-MINLP route takes.

``Model.solve`` ran ``try_convex_solve`` before ``solve_model`` with
``min(time_limit, DISCOPT_CONVEX_KERNEL_BUDGET)`` -- the whole budget for any
``time_limit <= 120``. On a kernel-eligible model the #1059 router would divert,
a kernel that could not certify left ``solve_model`` ~0 s, the router's
convexity proof hit the expired deadline, and the route never fired: on MINLPLib
``syn``/``rsyn`` 30 of 51 instances came back with no incumbent at all.

These tests assert *which path ran*, not seconds, so they are not a speed test.
The kernel's native tree is wrapped (and still called through) to record whether
it ran.
"""

from __future__ import annotations

import logging
import os

import pytest
from discopt.modeling import from_nl
from discopt.solvers import _convex_kernel as ck

_NL = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "clay0303hfsg.nl")

pytestmark = pytest.mark.skipif(
    not os.path.exists(_NL), reason="vendored clay0303hfsg.nl not present"
)


@pytest.fixture
def tree_calls(monkeypatch):
    """Record every call into the kernel's native tree, delegating to the real one."""
    calls: list[float] = []
    real = ck.solve_convex_tree

    def spy(spec, *, time_limit_s=None, **cfg):
        calls.append(time_limit_s)
        return real(spec, time_limit_s=time_limit_s, **cfg)

    monkeypatch.setattr(ck, "solve_convex_tree", spy)
    return calls


def _known_optimum() -> float:
    from _optima import known_optimum

    return float(known_optimum("clay0303hfsg"))


def test_routed_kernel_eligible_model_goes_to_the_route(tree_calls, monkeypatch):
    """clay0303hfsg is kernel-eligible AND routed: the route, not the kernel, runs.

    Fails before #1624: the kernel's tree ran first (and at this budget it
    certifies or spends the budget, so ``algorithm_route`` never names the route).
    """
    monkeypatch.delenv("DISCOPT_CONVEX_KERNEL_DEFER_TO_ROUTE", raising=False)
    monkeypatch.delenv("DISCOPT_CONVEX_KERNEL", raising=False)
    m = from_nl(_NL)
    assert ck.build_convex_spec(from_nl(_NL)) is not None, "probe must be kernel-eligible"

    r = m.solve(time_limit=30)

    assert tree_calls == [], "the kernel's tree ran on a model the route takes"
    assert ck.last_deferred_reason() is not None
    assert (r.algorithm_route or "").startswith("mip-nlp/"), r.algorithm_route
    # Soundness: the route's answer is the oracle's, and its bound does not cross it.
    opt = _known_optimum()
    assert r.objective is not None
    assert r.objective == pytest.approx(opt, rel=1e-4)
    assert r.bound is not None and r.bound <= opt + 1e-6 * max(1.0, abs(opt))


def test_opt_out_restores_kernel_first(tree_calls, monkeypatch):
    monkeypatch.setenv("DISCOPT_CONVEX_KERNEL_DEFER_TO_ROUTE", "0")
    from_nl(_NL).solve(time_limit=2)
    assert len(tree_calls) == 1
    assert ck.last_deferred_reason() is None


def test_option_the_route_ignores_keeps_the_kernel(tree_calls, monkeypatch):
    """``solve_model`` refuses the route when the caller set an option the MIP-NLP
    family drops; the kernel must not defer to a route that will not be taken."""
    monkeypatch.delenv("DISCOPT_CONVEX_KERNEL_DEFER_TO_ROUTE", raising=False)
    r = from_nl(_NL).solve(time_limit=2, abs_gap_tolerance=1e-6)
    assert len(tree_calls) == 1
    assert ck.last_deferred_reason() is None
    assert "mip-nlp" not in (r.algorithm_route or "")


def test_router_raise_falls_back_to_kernel_loudly(monkeypatch, caplog):
    """A raise inside the router is a defect, not a verdict: no deferral, logged."""
    import discopt.solver as S

    monkeypatch.delenv("DISCOPT_CONVEX_KERNEL_DEFER_TO_ROUTE", raising=False)

    def boom(model):
        raise RuntimeError("router defect 1624")

    monkeypatch.setattr(S, "_convex_minlp_auto_route", boom)
    with caplog.at_level(logging.WARNING):
        assert S._convex_route_preempts_kernel(from_nl(_NL), {}) is None
    assert "router defect 1624" in caplog.text


def test_probe_classification_is_held_to_the_callers_time_limit(monkeypatch):
    """Review of #1650: the probe ran the router's classification on its default 15 s
    cap, outside ``solve_model``'s clamp, so it could overrun ``time_limit`` and prove
    a verdict ``solve_model``'s shorter classification could not. It now sees the
    budget ``solve_model`` would set, and leaves the model's attributes as it found
    them."""
    import discopt.solver as S

    seen: list = []

    def router(model):
        seen.append(
            (
                getattr(model, "_convexity_time_budget", None),
                getattr(model, "_solve_deadline", None),
            )
        )
        model._convexity_classification_cache = (False, False, None)
        return None, None, None

    monkeypatch.setattr(S, "_convex_minlp_auto_route", router)
    m = from_nl(_NL)
    import time

    t0 = time.perf_counter()
    assert S._convex_route_preempts_kernel(m, {}, time_limit=2.0) is None
    assert len(seen) == 1
    budget, deadline = seen[0]
    assert budget == pytest.approx(0.5)  # min(max(0.2 * 2, 0.5), 20)
    assert deadline is not None and t0 < deadline <= time.perf_counter() + 2.0
    for attr in ("_convexity_time_budget", "_solve_deadline", "_convexity_classification_cache"):
        assert not hasattr(m, attr), attr
