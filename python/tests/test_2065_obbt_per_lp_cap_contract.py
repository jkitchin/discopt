"""#2065: the persistent OBBT probe path ignores ``time_limit_per_lp`` on purpose.

``run_obbt_on_relaxation``'s docstring states the contract and the measurement
behind it. The seam-oracle probes honor the per-LP cap. The default persistent
probe path (``_PersistentProbeLP``) is bounded only by the sweep's remaining
``deadline`` (#2006). Forwarding the per-LP cap there would change bounds, and the
#2065 entry experiment found under 0.1% of probes exceed it, so that was not done.
These tests pin the contract so that a change to it also updates the docstring.
"""

from __future__ import annotations

import time

import discopt._relax.obbt as obbt_mod
import pytest
from discopt._relax.obbt import run_obbt_on_relaxation
from test_obbt_warm_probes import (
    PANEL,
    _build_relaxation,
    _maxdiff,
    record_binding_time_limits,
)

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

_TINY = 1e-9  # a per-LP cap no LP can meet


def _n_orig(model) -> int:
    return sum(v.size for v in model._variables)


@pytest.mark.parametrize("name", list(PANEL))
def test_persistent_path_does_not_forward_per_lp_cap(monkeypatch, name):
    assert obbt_mod._simplex_available(), "Rust simplex binding required for this test"
    seen = record_binding_time_limits(monkeypatch)
    model = PANEL[name]()
    run_obbt_on_relaxation(_build_relaxation(model), n_orig=_n_orig(model), time_limit_per_lp=_TINY)
    assert seen, "probe binding never called: the instrument measured nothing"
    # No deadline: every probe runs uncapped, whatever ``time_limit_per_lp`` says.
    assert all(tl is None for tl in seen), seen
    n = len(seen)

    seen.clear()
    deadline = time.perf_counter() + 60.0
    run_obbt_on_relaxation(
        _build_relaxation(model), n_orig=_n_orig(model), time_limit_per_lp=_TINY, deadline=deadline
    )
    assert seen
    # With a deadline: the cap is the remaining time, never the per-LP limit.
    assert all(tl is not None and tl > 1.0 for tl in seen), seen
    print(f"EXECUTED {n + len(seen)} probe-limit assertions")


def test_persistent_path_bounds_independent_of_per_lp_cap():
    """The tightenings are bit-identical whatever per-LP cap the caller passes."""
    n_tightened = 0
    for make in PANEL.values():
        model = make()
        tiny = run_obbt_on_relaxation(
            _build_relaxation(model), n_orig=_n_orig(model), time_limit_per_lp=_TINY
        )
        free = run_obbt_on_relaxation(
            _build_relaxation(model), n_orig=_n_orig(model), time_limit_per_lp=None
        )
        assert tiny.n_lp_solves == free.n_lp_solves > 0
        assert tiny.n_tightened == free.n_tightened
        assert _maxdiff(tiny.tightened_lb, free.tightened_lb) == 0.0
        assert _maxdiff(tiny.tightened_ub, free.tightened_ub) == 0.0
        n_tightened += free.n_tightened
    # A panel that tightens nothing would make the comparison vacuous.
    assert n_tightened > 0


def test_seam_path_honors_per_lp_cap(monkeypatch):
    """The cold seam path (no Rust binding) still forwards ``time_limit_per_lp``."""
    monkeypatch.setattr(obbt_mod, "_simplex_available", lambda: False)
    seen: list = []
    dual = obbt_mod.get_exact_dual_lp_solver()
    inner = dual or obbt_mod.get_exact_lp_solver()
    assert inner is not None, "an exact LP oracle is required for this test"

    def recording(**kwargs):
        seen.append(kwargs.get("time_limit"))
        return inner(**kwargs)

    name = "get_exact_dual_lp_solver" if dual is not None else "get_exact_lp_solver"
    monkeypatch.setattr(obbt_mod, name, lambda: recording)
    model = PANEL["bilinear"]()
    run_obbt_on_relaxation(_build_relaxation(model), n_orig=_n_orig(model), time_limit_per_lp=0.7)
    assert seen, "seam oracle never called: the instrument measured nothing"
    assert all(tl == 0.7 for tl in seen), seen
