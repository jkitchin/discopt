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

import time

import discopt._relax.obbt as obbt_mod
import numpy as np
import pytest
from discopt._relax.obbt import run_obbt_on_relaxation
from test_obbt_warm_probes import PANEL, _build_relaxation, _maxdiff

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

_TIME_LIMIT_POS = 13  # positional index of ``time_limit_s`` in solve_lp_warm_csc_py


def _record_budgets(monkeypatch):
    """Wrap the binding; record ``(time_limit_s, seconds_left_at_call)`` per probe."""
    import discopt._rust as rust

    orig = rust.solve_lp_warm_csc_py
    seen: list[tuple[object, float]] = []
    state = {"deadline": None}

    def wrapped(*args, **kwargs):
        positional = args[_TIME_LIMIT_POS] if len(args) > _TIME_LIMIT_POS else None
        tl = kwargs.get("time_limit_s", positional)
        left = state["deadline"] - time.perf_counter() if state["deadline"] else float("nan")
        seen.append((tl, left))
        return orig(*args, **kwargs)

    monkeypatch.setattr(rust, "solve_lp_warm_csc_py", wrapped)
    return seen, state


@pytest.mark.parametrize("name", list(PANEL))
def test_probe_lp_receives_remaining_deadline(monkeypatch, name):
    assert obbt_mod._simplex_available(), "Rust simplex binding required for this test"
    seen, state = _record_budgets(monkeypatch)
    model = PANEL[name]()
    n_orig = sum(v.size for v in model._variables)
    rel = _build_relaxation(model)
    state["deadline"] = time.perf_counter() + 60.0
    run_obbt_on_relaxation(rel, n_orig=n_orig, time_limit_per_lp=5.0, deadline=state["deadline"])

    assert seen, "probe binding never called: the instrument measured nothing"
    for tl, left in seen:
        # Before #2006 ``tl`` was None: the probe ran unbounded past the deadline.
        assert tl is not None and np.isfinite(tl)
        assert 0.0 <= tl <= left + 1e-3
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

    def expired(*args, **kwargs):
        args = list(args)
        args[_TIME_LIMIT_POS] = 0.0  # an already-elapsed budget
        calls.append(1)
        return orig(*args, **kwargs)

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
