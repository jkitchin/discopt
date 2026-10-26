"""#1959: the lift race's lift slice ends when the MILP engine stalls with no incumbent.

Inside the #1940 race's opening lift slice, the native MILP engine's #291 stall
guard deferring with no incumbent used to hand the rest of the slice to the Python
MILP tree on the lifted model. Measured, that tree never produced an incumbent or a
certificate there (``scripts/panel_1959_race_stall_exit.py`` counts it), so on every
instance where the lift loses the race the slice ran its full 5 s for nothing.
``DISCOPT_IPX_RACE_STALL_EXIT`` (default ON, ``=0`` opt-out) ends the slice at the
deferral and keeps the engine's valid bound. These tests pin:

* that only the two opening SLICES are marked as such, never the continuation;
* that a stall inside a lift slice skips the Python tree and returns the engine's
  bound, and that it does not under the opt-out, outside a slice, or in an
  ordinary solve (an engine stub decides the stall, so no clock is involved);
* on the #1940 dense integer QP, that the race still certifies the optimum and
  the lift slice now ends on the stall.
"""

from __future__ import annotations

import os
import time
import types

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import solver as S
from discopt.modeling.core import from_nl

DATA = os.path.join(os.path.dirname(__file__), "data")

STALL_ROUTE = "ended on the engine's stall"


def _nl(name):
    for d in ("minlplib", "minlplib_nl"):
        p = os.path.join(DATA, d, f"{name}.nl")
        if os.path.exists(p):
            return from_nl(p)
    raise AssertionError(f"{name}.nl not found")


@pytest.mark.smoke
def test_default_on_with_an_opt_out(monkeypatch):
    monkeypatch.delenv("DISCOPT_IPX_RACE_STALL_EXIT", raising=False)
    assert S._ipx_race_stall_exit_enabled() is True
    for off in ("0", "false", "no", "off"):
        monkeypatch.setenv("DISCOPT_IPX_RACE_STALL_EXIT", off)
        assert S._ipx_race_stall_exit_enabled() is False, off


def _fake(obj, bound, certified):
    return S.SolveResult(
        status="optimal" if certified else "time_limit",
        objective=obj,
        bound=bound,
        gap_certified=certified,
        node_count=10,
        bound_valid=bound is not None,
    )


@pytest.mark.smoke
def test_only_the_opening_slices_are_marked_as_slices(monkeypatch):
    seen = []
    script = {
        "lift": [_fake(None, 1.0, False)],
        "unlifted": [_fake(10.0, 5.0, False), _fake(10.0, 10.0, True)],
    }

    def stub(**kw):
        arm = S._IPX_RACE_ARM.get()
        seen.append((arm, S._IPX_RACE_IN_SLICE.get()))
        return script[arm].pop(0)

    monkeypatch.setattr(S, "solve_model", stub)
    entry = dict(
        model=types.SimpleNamespace(name="m"),
        time_limit=60.0,
        max_nodes=1000,
        gap_tolerance=1e-4,
        kwargs={},
    )
    r = S._ipx_lift_race(entry, time.perf_counter(), False)
    assert r.gap_certified
    assert seen == [("lift", True), ("unlifted", True), ("unlifted", False)]
    assert S._IPX_RACE_IN_SLICE.get() is False, "the mark leaked out of the race"


def _stalling_engine(monkeypatch, bound):
    """Make the MILP engine defer as its #291 stall guard does: no result, a bound."""
    calls = {"engine": 0, "pytree": 0}
    real_bb = S._solve_milp_bb

    def engine(*a, deferred=None, **k):
        calls["engine"] += 1
        if deferred is not None:
            deferred["bound"] = bound
            deferred["stats"] = {"milp_driver/iters": 50.0, "milp_driver/nodes": 7.0}
        return None

    def pytree(*a, **k):
        calls["pytree"] += 1
        return real_bb(*a, **k)

    monkeypatch.setattr(S, "_solve_milp_simplex", engine)
    monkeypatch.setattr(S, "_solve_milp_bb", pytree)
    return calls


def _solve_as(arm, in_slice):
    tok_arm = S._IPX_RACE_ARM.set(arm)
    tok_slice = S._IPX_RACE_IN_SLICE.set(in_slice)
    try:
        return S.solve_model(_nl("prob03"), time_limit=60.0, incumbent_time_extension=0.0)
    finally:
        S._IPX_RACE_IN_SLICE.reset(tok_slice)
        S._IPX_RACE_ARM.reset(tok_arm)


@pytest.mark.smoke
def test_a_stall_in_a_lift_slice_ends_the_slice_with_the_engine_bound(monkeypatch):
    monkeypatch.delenv("DISCOPT_IPX_RACE_STALL_EXIT", raising=False)
    calls = _stalling_engine(monkeypatch, bound=3.0)  # prob03's optimum is 10
    r = _solve_as("lift", True)
    assert calls == {"engine": 1, "pytree": 0}
    assert r.status == "time_limit" and r.objective is None and not r.gap_certified
    assert r.bound == pytest.approx(3.0)
    assert r.node_count == 7
    assert STALL_ROUTE in r.algorithm_route


@pytest.mark.smoke
@pytest.mark.parametrize(
    "arm,in_slice,env",
    [
        ("lift", True, "0"),  # the opt-out keeps the legacy fallback
        ("lift", False, None),  # the continuation keeps it too
        (None, False, "0"),  # an ordinary solve (race off, so the lift is adopted)
    ],
    ids=["opt-out", "continuation", "ordinary-solve"],
)
def test_the_python_tree_still_runs_outside_a_lift_slice(monkeypatch, arm, in_slice, env):
    monkeypatch.delenv("DISCOPT_IPX_RACE_STALL_EXIT", raising=False)
    if env is not None:
        monkeypatch.setenv("DISCOPT_IPX_RACE_STALL_EXIT", env)
    if arm is None:
        monkeypatch.setenv("DISCOPT_IPX_RACE", "0")
    calls = _stalling_engine(monkeypatch, bound=3.0)
    r = _solve_as(arm, in_slice)
    assert calls == {"engine": 1, "pytree": 1}
    assert STALL_ROUTE not in (r.algorithm_route or "")
    assert r.status == "optimal" and r.objective == pytest.approx(10.0)


def _dense_qp(n, seed):
    rng = np.random.default_rng(seed)
    m = dm.Model(f"dense_qp_n{n}_s{seed}")
    x = [m.integer(f"x{i}", lb=0, ub=10) for i in range(n)]
    m.minimize(
        sum(float(rng.normal()) * x[i] * x[j] for i in range(n) for j in range(i + 1, n)) + sum(x)
    )
    m.subject_to(sum(x) >= n)
    return m


@pytest.mark.slow
@pytest.mark.correctness
def test_dense_qp_race_ends_its_lift_slice_on_the_stall_and_certifies(monkeypatch):
    """#1940's dense integer QP, n=15 seed 0. The lift slice's engine stalls with
    no incumbent; the slice must end there, and the un-lifted arm certifies the
    same optimum as before."""
    monkeypatch.delenv("DISCOPT_IPX_CHEAP_FIRST", raising=False)
    monkeypatch.delenv("DISCOPT_IPX_RACE", raising=False)
    monkeypatch.delenv("DISCOPT_IPX_RACE_STALL_EXIT", raising=False)
    slices = []
    real = S.solve_model

    def spy(*a, **k):
        r = real(*a, **k)
        if S._IPX_RACE_ARM.get() is not None:
            slices.append((S._IPX_RACE_ARM.get(), r.algorithm_route or ""))
        return r

    monkeypatch.setattr(S, "solve_model", spy)
    r = _dense_qp(15, 0).solve(time_limit=60.0)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(-1060.5575443498, rel=1e-6)
    assert r.bound <= r.objective + 1e-6 * abs(r.objective)
    assert slices and slices[0][0] == "lift"
    assert STALL_ROUTE in slices[0][1], slices
