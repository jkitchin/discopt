"""#2006: the OBBT sweep's ``deadline`` bounds each persistent-path probe LP.

``run_obbt_on_relaxation`` polled its ``deadline`` only *between* probes, and the
persistent probe LP (``_PersistentProbeLP.solve``) called
``solve_lp_warm_csc_py`` with no ``time_limit_s``. One probe that started just
inside the deadline therefore ran to convergence however long that took. Measured
on QPLIB_5875 (QBN, 200 binaries, 15 771 bilinear terms) at ``time_limit=30``: a
periodic-OBBT probe started at 24.8 s inside a <= 5 s phase budget and ran 49.6 s,
so the solve returned at 75-88 s.

The fix forwards the remaining time to the binding. A probe cut short is
non-optimal and tightens nothing, and the cap is the remaining time only, so a
sweep that finishes inside its deadline is bound-identical (checked below).
"""

from __future__ import annotations

import inspect
import time

import discopt._relax.obbt as obbt_mod
import numpy as np
import pytest
from discopt._relax.obbt import run_obbt_on_relaxation
from test_obbt_warm_probes import (
    PANEL,
    _build_relaxation,
    _maxdiff,
    record_binding_time_limits,
)

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _record_budgets(monkeypatch):
    """Record ``(time_limit_s, t_before, t_call)`` per probe.

    The sweep computes each probe's budget as ``deadline - perf_counter()`` at some
    instant ``t_budget`` that this instrument cannot observe. It can observe two
    instants that bracket it: ``t_call``, read here at the binding call, which comes
    after the budget is computed, and ``t_before``, the previous probe's ``t_call``
    (or ``state["t_start"]``, read before the sweep starts, for the first probe),
    which comes before it. ``perf_counter`` is monotonic, so a budget equal to the
    remaining time satisfies ``deadline - t_call <= tl <= deadline - t_before``
    exactly. The earlier one-sided check ``tl <= (deadline - t_call) + 1e-3`` charged
    the time between the sweep's clock read and this one to a 1 ms tolerance, and a
    slow CI runner exceeded it by 3.1 ms (``59.99585 <= 59.99276 + 0.001``, #2327).
    """
    state: dict = {"deadline": None, "t_start": None, "t_last": None}

    def bracket(tl):
        t_call = time.perf_counter()
        t_before = state["t_last"] if state["t_last"] is not None else state["t_start"]
        state["t_last"] = t_call
        return (tl, t_before, t_call)

    return record_binding_time_limits(monkeypatch, on_call=bracket), state


@pytest.mark.parametrize("name", list(PANEL))
def test_probe_lp_receives_remaining_deadline(monkeypatch, name):
    assert obbt_mod._simplex_available(), "Rust simplex binding required for this test"
    seen, state = _record_budgets(monkeypatch)
    model = PANEL[name]()
    n_orig = sum(v.size for v in model._variables)
    rel = _build_relaxation(model)
    deadline = time.perf_counter() + 60.0
    state["deadline"] = deadline
    state["t_start"] = time.perf_counter()
    run_obbt_on_relaxation(rel, n_orig=n_orig, time_limit_per_lp=5.0, deadline=deadline)

    assert seen, "probe binding never called: the instrument measured nothing"
    for tl, t_before, t_call in seen:
        # Before #2006 ``tl`` was None: the probe ran unbounded past the deadline.
        assert tl is not None and np.isfinite(tl)
        # The remaining time at an instant between the two clock reads; no tolerance.
        assert 0.0 <= deadline - t_call <= tl <= deadline - t_before, (
            tl,
            deadline - t_before,
            deadline - t_call,
        )
    print(f"EXECUTED {len(seen)} probe-budget assertions")


@pytest.mark.parametrize("name", list(PANEL))
def test_deadline_cap_is_bound_neutral_when_not_hit(name):
    """A sweep that finishes inside its deadline tightens exactly as with none."""
    model = PANEL[name]()
    n_orig = sum(v.size for v in model._variables)
    free = run_obbt_on_relaxation(_build_relaxation(model), n_orig=n_orig, time_limit_per_lp=5.0)
    capped = run_obbt_on_relaxation(
        _build_relaxation(model),
        n_orig=n_orig,
        time_limit_per_lp=5.0,
        deadline=time.perf_counter() + 60.0,
    )
    assert capped.n_lp_solves == free.n_lp_solves > 0
    assert capped.n_tightened == free.n_tightened
    assert _maxdiff(capped.tightened_lb, free.tightened_lb) == 0.0
    assert _maxdiff(capped.tightened_ub, free.tightened_ub) == 0.0


def test_deadline_cut_probe_tightens_nothing(monkeypatch):
    """A probe the budget stops is non-optimal and must not tighten any bound."""
    import discopt._rust as rust

    orig = rust.solve_lp_warm_csc_py
    calls = []

    sig = inspect.signature(orig)

    def expired(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.arguments["time_limit_s"] = 0.0  # an already-elapsed budget
        calls.append(1)
        return orig(*bound.args, **bound.kwargs)

    monkeypatch.setattr(rust, "solve_lp_warm_csc_py", expired)
    model = PANEL["bilinear"]()
    n_orig = sum(v.size for v in model._variables)
    rel = _build_relaxation(model)
    box = np.asarray(rel._bounds[:n_orig], dtype=np.float64)
    res = run_obbt_on_relaxation(
        rel, n_orig=n_orig, time_limit_per_lp=5.0, deadline=time.perf_counter() + 60.0
    )
    assert calls
    assert res.n_tightened == 0
    assert np.array_equal(res.tightened_lb[:n_orig], box[:, 0])
    assert np.array_equal(res.tightened_ub[:n_orig], box[:, 1])
