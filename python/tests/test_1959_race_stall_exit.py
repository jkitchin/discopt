"""#1959: the lift race's lift slice ends when the MILP engine stalls with no incumbent.

Inside the #1940 race's opening lift slice, the native MILP engine's #291 stall guard
deferring with no incumbent used to hand the rest of the slice to the Python MILP tree
on the lifted model, which never produced an incumbent or a certificate there
(``scripts/panel_1959_race_stall_exit.py``). ``DISCOPT_IPX_RACE_STALL_EXIT`` ended the
slice at the deferral. Since the Python MILP tree was retired, the engine itself returns
its no-incumbent result with its valid bound at the end of its budget, so the flag and
its branch are gone. These tests pin:

* that only the two opening SLICES are marked as such, never the continuation;
* on the #1940 dense integer QP, that the race still certifies the optimum and the
  lift slice ends on the engine's own no-incumbent result.
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

STALL_ROUTE = "no incumbent; searched the whole budget"


def _nl(name):
    for d in ("minlplib", "minlplib_nl"):
        p = os.path.join(DATA, d, f"{name}.nl")
        if os.path.exists(p):
            return from_nl(p)
    raise AssertionError(f"{name}.nl not found")


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
