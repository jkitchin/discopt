"""#2065: ``time_limit_per_lp`` on the persistent OBBT probe path.

The seam-oracle probes honor the per-LP cap. The default persistent probe path
(``_PersistentProbeLP``) is bounded only by the sweep's remaining ``deadline``
(#2006). ``DISCOPT_OBBT_PROBE_PER_LP_CAP=1`` (default OFF, bound-changing, pending
its section 5 panel) forwards ``min(budget, time_limit_per_lp)`` instead. These
tests pin both arms: OFF is unchanged, ON caps every probe and is bound-identical
whenever the cap does not bind.
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


@pytest.fixture(autouse=True)
def _flag_off(monkeypatch):
    """Every test starts from the shipped default (flag unset)."""
    monkeypatch.delenv("DISCOPT_OBBT_PROBE_PER_LP_CAP", raising=False)


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


@pytest.mark.parametrize("name", list(PANEL))
def test_flag_on_forwards_per_lp_cap(monkeypatch, name):
    monkeypatch.setenv("DISCOPT_OBBT_PROBE_PER_LP_CAP", "1")
    seen = record_binding_time_limits(monkeypatch)
    model = PANEL[name]()
    run_obbt_on_relaxation(_build_relaxation(model), n_orig=_n_orig(model), time_limit_per_lp=0.7)
    assert seen, "probe binding never called: the instrument measured nothing"
    # No deadline: the per-LP cap itself.
    assert all(tl == 0.7 for tl in seen), seen
    n = len(seen)

    seen.clear()
    deadline = time.perf_counter() + 60.0
    run_obbt_on_relaxation(
        _build_relaxation(model), n_orig=_n_orig(model), time_limit_per_lp=0.7, deadline=deadline
    )
    assert seen
    # A deadline far away: still the per-LP cap (the smaller of the two).
    assert all(tl == 0.7 for tl in seen), seen

    seen.clear()
    run_obbt_on_relaxation(
        _build_relaxation(model),
        n_orig=_n_orig(model),
        time_limit_per_lp=60.0,
        deadline=time.perf_counter() + 5.0,
    )
    assert seen
    # A deadline nearer than the cap: the remaining time.
    assert all(tl is not None and tl <= 5.0 for tl in seen), seen
    print(f"EXECUTED {n + 2 * len(seen)} flag-ON probe-limit assertions")


def test_flag_on_is_bound_identical_when_cap_does_not_bind(monkeypatch):
    n_tightened = 0
    for make in PANEL.values():
        model = make()
        monkeypatch.setenv("DISCOPT_OBBT_PROBE_PER_LP_CAP", "0")
        off = run_obbt_on_relaxation(
            _build_relaxation(model), n_orig=_n_orig(model), time_limit_per_lp=60.0
        )
        monkeypatch.setenv("DISCOPT_OBBT_PROBE_PER_LP_CAP", "1")
        on = run_obbt_on_relaxation(
            _build_relaxation(model), n_orig=_n_orig(model), time_limit_per_lp=60.0
        )
        assert on.n_lp_solves == off.n_lp_solves > 0
        assert on.n_tightened == off.n_tightened
        assert _maxdiff(on.tightened_lb, off.tightened_lb) == 0.0
        assert _maxdiff(on.tightened_ub, off.tightened_ub) == 0.0
        n_tightened += off.n_tightened
    assert n_tightened > 0


def test_flag_on_cut_short_probe_tightens_nothing(monkeypatch):
    """A cap no LP can meet stops every probe; no bound moves (sound, weaker)."""
    import numpy as np

    monkeypatch.setenv("DISCOPT_OBBT_PROBE_PER_LP_CAP", "1")
    model = PANEL["bilinear"]()
    rel = _build_relaxation(model)
    n = _n_orig(model)
    box = np.asarray(rel._bounds[:n], dtype=np.float64)
    res = run_obbt_on_relaxation(rel, n_orig=n, time_limit_per_lp=0.0)
    assert res.n_lp_solves > 0
    assert res.n_tightened == 0
    assert np.array_equal(res.tightened_lb[:n], box[:, 0])
    assert np.array_equal(res.tightened_ub[:n], box[:, 1])
