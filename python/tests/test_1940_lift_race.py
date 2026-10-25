"""#1940: the integer-bilinear lift must be adopted because it wins, not because it is possible.

The lift (``integer.bilinear``) turns an integer-bilinear MINLP into an exact pure
MILP. It wins on most of what it applies to and loses 5-30x on dense integer QPs,
and no structural separator survived measurement, so ``solve_model`` now RACES the
two exact formulations (``solver._ipx_lift_race``): a lifted slice, an un-lifted
slice, then the arm with the smaller gap, all merged. These tests pin:

* the control flow, against a stubbed ``solve_model`` (fast and clock-free);
* that the race is invisible where the lift certifies inside its first slice --
  the same tree and the same certificate as the legacy rule (CLAUDE.md §5,
  bound-neutral regime);
* that it declines the lift on an instance where the lift loses;
* the guards that keep the legacy rule (opt-out, ``deterministic``, short limits,
  callbacks / warm starts, cheap-first, nested arms).
"""

from __future__ import annotations

import math
import os
import types

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import solver as S
from discopt.modeling.core import from_nl

DATA = os.path.join(os.path.dirname(__file__), "data")


def _nl(name):
    for d in ("minlplib", "minlplib_nl"):
        p = os.path.join(DATA, d, f"{name}.nl")
        if os.path.exists(p):
            return from_nl(p)
    raise AssertionError(f"{name}.nl not found")


# --------------------------------------------------------------------------
# Control flow, against a stub: which arms run, with what budget, and what is
# returned. No real solve, so no clock decides anything here.
# --------------------------------------------------------------------------


def _fake(obj, bound, certified, nodes=10):
    return S.SolveResult(
        status="optimal" if certified else "time_limit",
        objective=obj,
        bound=bound,
        gap_certified=certified,
        node_count=nodes,
        bound_valid=bound is not None,
    )


class _Stub:
    """Stands in for ``solve_model``; answers per arm from a script."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    def __call__(self, **kw):
        arm = S._IPX_RACE_ARM.get()
        self.calls.append((arm, kw))
        return self.script[arm].pop(0)


def _entry(**extra):
    model = types.SimpleNamespace(name="m")
    args = dict(model=model, time_limit=60.0, max_nodes=1000, gap_tolerance=1e-4, kwargs={})
    args.update(extra)
    return args


def _race(monkeypatch, script, **entry):
    stub = _Stub(script)
    monkeypatch.setattr(S, "solve_model", stub)
    import time

    result = S._ipx_lift_race(_entry(**entry), time.perf_counter(), False)
    return result, stub


@pytest.mark.smoke
def test_a_lift_that_certifies_in_its_slice_is_the_answer(monkeypatch):
    lifted = _fake(5.0, 5.0, True)
    result, stub = _race(monkeypatch, {"lift": [lifted]})
    assert result is lifted
    assert [arm for arm, _ in stub.calls] == ["lift"], "no un-lifted slice may run"
    assert stub.calls[0][1]["time_limit"] == pytest.approx(S._IPX_RACE_SLICE_S)


@pytest.mark.smoke
def test_an_unlifted_slice_that_certifies_wins_and_keeps_its_certificate(monkeypatch):
    result, stub = _race(
        monkeypatch,
        {"lift": [_fake(7.0, 1.0, False, nodes=40)], "unlifted": [_fake(5.0, 5.0, True, 3)]},
    )
    assert [arm for arm, _ in stub.calls] == ["lift", "unlifted"]
    assert result.objective == 5.0 and result.gap_certified and result.status == "optimal"
    # Work is a statistic of the call, not of the winning arm.
    assert result.node_count == 43
    # The second slice is charged the first one's nodes.
    assert stub.calls[1][1]["max_nodes"] == 1000 - 40


@pytest.mark.smoke
@pytest.mark.parametrize(
    "lift_gap_pair,unlifted_gap_pair,expected",
    [
        ((10.0, 9.9), (10.0, 5.0), "lift"),  # lift has the smaller gap
        ((10.0, 5.0), (10.0, 9.9), "unlifted"),
        ((10.0, 5.0), (10.0, 5.0), "lift"),  # a tie keeps the legacy choice
        ((None, 5.0), (None, 4.0), "lift"),  # neither has an incumbent: tie
        ((None, 5.0), (10.0, 4.0), "unlifted"),  # only one has a gap at all
    ],
)
def test_neither_slice_certifies_so_the_smaller_gap_continues(
    monkeypatch, lift_gap_pair, unlifted_gap_pair, expected
):
    final = _fake(9.0, 9.0, True)
    result, stub = _race(
        monkeypatch,
        {
            "lift": [_fake(*lift_gap_pair, False)] + ([final] if expected == "lift" else []),
            "unlifted": [_fake(*unlifted_gap_pair, False)]
            + ([final] if expected == "unlifted" else []),
        },
    )
    arms = [arm for arm, _ in stub.calls]
    assert arms == ["lift", "unlifted", expected]
    # The continuation gets the rest of the caller's budget, not another slice.
    assert stub.calls[2][1]["time_limit"] > 60.0 - 2 * S._IPX_RACE_SLICE_S - 1.0
    assert result.objective == 9.0 and result.gap_certified


@pytest.mark.smoke
def test_the_merge_keeps_the_best_incumbent_and_the_tightest_bound(monkeypatch):
    """#1940's "mixed" rows: the lift has the tighter bound, the spatial route the
    better incumbent. Both are valid for the caller's model, so both are kept."""
    result, _ = _race(
        monkeypatch,
        {
            "lift": [_fake(-70.21, -81.12, False)],
            "unlifted": [_fake(-80.08, -97.05, False), _fake(-80.08, -90.0, False)],
        },
    )
    assert result.objective == pytest.approx(-80.08)
    assert result.bound == pytest.approx(-81.12)
    assert not result.gap_certified


@pytest.mark.smoke
def test_every_arm_receives_the_callers_options_verbatim(monkeypatch):
    lifted = _fake(5.0, 5.0, True)
    _, stub = _race(
        monkeypatch,
        {"lift": [lifted]},
        gap_tolerance=3e-3,
        kwargs={"pounce_scaling": "x"},
    )
    kw = stub.calls[0][1]
    assert kw["gap_tolerance"] == 3e-3
    assert kw["pounce_scaling"] == "x", "a **kwargs option was dropped"
    assert "kwargs" not in kw


@pytest.mark.smoke
def test_only_the_continuation_may_take_the_incumbent_extension(monkeypatch):
    """``Model.solve`` grants the primary an extension past its limit while it
    holds an incumbent (#917). A slice that inherited it ran 15.5 s on a 5 s
    budget (``arrqp_n15``) and starved the continuation."""
    _, stub = _race(
        monkeypatch,
        {
            "lift": [_fake(10.0, 5.0, False)],
            "unlifted": [_fake(10.0, 9.0, False), _fake(10.0, 10.0, True)],
        },
        incumbent_time_extension=10.5,
    )
    ext = [kw["incumbent_time_extension"] for _, kw in stub.calls]
    assert ext == [0.0, 0.0, 10.5]


@pytest.mark.smoke
def test_the_entry_snapshot_holds_every_parameter():
    """The race re-enters with ``solve_model``'s arguments as received; the snapshot
    is taken from ``locals()`` on the first line of the body, so it must be the
    parameter set exactly (anything else would be forwarded as a bogus option)."""
    import inspect

    src = inspect.getsource(S.solve_model)
    body = src.split('"""', 2)[2]
    first = next(
        line.strip()
        for line in body.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )
    assert first == "_ipx_entry_args = dict(locals())", first


@pytest.mark.smoke
def test_default_on_with_an_opt_out(monkeypatch):
    monkeypatch.delenv("DISCOPT_IPX_RACE", raising=False)
    assert S._ipx_race_enabled() is True
    for off in ("0", "false", "no", "off"):
        monkeypatch.setenv("DISCOPT_IPX_RACE", off)
        assert S._ipx_race_enabled() is False, off
    monkeypatch.setenv("DISCOPT_IPX_RACE", "1")
    assert S._ipx_race_enabled() is True


# --------------------------------------------------------------------------
# Guards: each keeps the legacy rule, i.e. the race never runs.
# --------------------------------------------------------------------------


def _spy_race(monkeypatch):
    calls = []
    real = S._ipx_lift_race

    def spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(S, "_ipx_lift_race", spy)
    return calls


@pytest.mark.smoke
@pytest.mark.parametrize(
    "env,solve_kw",
    [
        ({"DISCOPT_IPX_RACE": "0"}, {}),
        ({}, {"deterministic": True}),
        ({}, {"time_limit": S._IPX_RACE_MIN_LIMIT_S - 1.0}),
        ({}, {"incumbent_callback": lambda *a, **k: None}),
        ({"DISCOPT_IPX_CHEAP_FIRST": "1"}, {}),
    ],
    ids=["opt-out", "deterministic", "short-limit", "callback", "cheap-first"],
)
def test_guards_keep_the_legacy_rule(monkeypatch, env, solve_kw):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    calls = _spy_race(monkeypatch)
    kw = {"time_limit": 60.0, **solve_kw}
    r = _nl("prob03").solve(**kw)
    assert r.status == "optimal" and r.objective == pytest.approx(10.0)
    assert not calls, "the race ran under a guard that must keep the legacy rule"


@pytest.mark.smoke
def test_the_race_runs_on_an_adopted_instance(monkeypatch):
    """§6: the guard test above must not pass because the race never runs at all."""
    monkeypatch.delenv("DISCOPT_IPX_RACE", raising=False)
    monkeypatch.delenv("DISCOPT_IPX_CHEAP_FIRST", raising=False)
    calls = _spy_race(monkeypatch)
    r = _nl("prob03").solve(time_limit=60.0)
    assert r.status == "optimal" and r.gap_certified
    assert calls == [1]


# --------------------------------------------------------------------------
# Real solves.
# --------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("name", ["ex1264", "nvs02", "nvs14", "prob02", "prob03"])
def test_bound_neutral_where_the_lift_certifies_in_its_slice(name, monkeypatch):
    """The lift certifies these well inside one slice, so the race must return
    exactly the legacy answer: same certified objective, same tree."""
    monkeypatch.delenv("DISCOPT_IPX_CHEAP_FIRST", raising=False)
    monkeypatch.setenv("DISCOPT_IPX_RACE", "0")
    legacy = _nl(name).solve(time_limit=60.0)
    monkeypatch.setenv("DISCOPT_IPX_RACE", "1")
    raced = _nl(name).solve(time_limit=60.0)
    assert legacy.status == "optimal" and legacy.gap_certified
    assert raced.status == "optimal" and raced.gap_certified
    assert raced.objective == legacy.objective
    assert raced.node_count == legacy.node_count


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
def test_the_race_declines_a_lift_that_loses(monkeypatch):
    """The #1940 dense integer QP, n=15 seed 0: the spatial route certifies in
    ~3 s, the lift in ~25 s. The race must hand the answer to the un-lifted arm
    and certify the same optimum."""
    monkeypatch.delenv("DISCOPT_IPX_CHEAP_FIRST", raising=False)
    monkeypatch.delenv("DISCOPT_IPX_RACE", raising=False)
    arms = []
    real = S.solve_model

    def spy(*a, **k):
        arms.append(S._IPX_RACE_ARM.get())
        return real(*a, **k)

    monkeypatch.setattr(S, "solve_model", spy)
    r = _dense_qp(15, 0).solve(time_limit=60.0)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(-1060.5575443498, rel=1e-6)
    assert r.bound <= r.objective + 1e-6 * abs(r.objective)
    assert "unlifted" in arms, f"the un-lifted arm never ran: {arms}"
    assert not math.isnan(r.wall_time)
