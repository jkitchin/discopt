"""Regression tests for #698 — the Rust MILP engine's truncated time budget.

``_solve_milp_simplex`` bounds the engine to ``max(0.5, min(0.5 * remaining,
_SIMPLEX_MILP_BUDGET_CAP_S))``. That slice is a *stall* bound (#291: defer fast
to the robust fallback when the engine wedges), but on an uncertified
``feasible`` the function RETURNS rather than falling back — so the rest of the
user's ``time_limit`` was simply discarded. At ``time_limit=40`` the engine used
~10 s and returned uncertified; at the default 3600 s it used ~10 s.

An engine that produced an incumbent is demonstrably not wedged, so the fix
re-enters it with the remaining budget, seeded with the best point so far. The
engine is monotone in its budget, so this tightens the bound and can certify.

These tests pin the *contract* (budget is spent; the re-entry never worsens the
answer; no pointless re-entry when there is no extra budget), not a specific
objective — the engine is wall-clock-limited and therefore load-nondeterministic,
so asserting exact bounds/nodes here would be flaky.
"""

from __future__ import annotations

import time

import discopt.modeling as dm
import numpy as np
import pytest


def _hard_partition_model(N, K, per, seed=7):
    """An assignment-structured binary-product QP big enough not to certify fast.

    Scalar ``integer(0, 1)`` columns match how the ``.nl`` reader encodes this
    family (``_int_factor_range`` excludes declared ``BINARY``), so the
    integer-bilinear reform lifts it to an exact MILP and it reaches the Rust
    MILP engine — the path under test.
    """
    rng = np.random.default_rng(seed)
    m = dm.Model("gp_hard")
    x = [[m.integer(f"x_{i}_{k}", lb=0, ub=1) for k in range(K)] for i in range(N)]
    for i in range(N):
        m.subject_to(dm.sum(x[i][k] for k in range(K)) == 1, name=f"assign_{i}")
    for k in range(K):
        m.subject_to(dm.sum(x[i][k] for i in range(N)) == per, name=f"bal_{k}")
    edges = [(i, j, float(rng.integers(1, 9))) for i in range(N) for j in range(i + 1, N)]
    m.minimize(dm.sum(w * dm.sum(x[i][k] * x[j][k] for k in range(K)) for (i, j, w) in edges))
    return m


def _engine_budgets(monkeypatch):
    """Record the ``time_limit_s`` handed to each Rust MILP driver call."""
    import discopt._rust as rust

    budgets: list[float] = []
    real = rust.solve_milp_py

    def spy(*args, **kwargs):
        budgets.append(float(kwargs.get("time_limit_s", 0.0)))
        return real(*args, **kwargs)

    monkeypatch.setattr(rust, "solve_milp_py", spy)
    return budgets


@pytest.mark.slow
def test_uncertified_milp_spends_the_user_budget(monkeypatch):
    """A solve that comes back uncertified must not leave most of the limit unused.

    Pre-fix this returned at ~min(0.5*TL, 10 s) regardless of ``time_limit``.
    """
    budgets = _engine_budgets(monkeypatch)
    m = _hard_partition_model(18, 3, 6)

    tl = 30.0
    t0 = time.perf_counter()
    r = m.solve(time_limit=tl)
    wall = time.perf_counter() - t0

    assert wall <= tl + 5.0, "the solve must still respect the user's time_limit"
    if r.status != "optimal":
        # Uncertified: the budget should have been spent, not discarded at ~10 s.
        assert wall > 12.0, f"returned uncertified after only {wall:.1f}s of a {tl}s limit"
        assert max(budgets) > 10.0 + 1e-6, (
            "no engine call was given more than the stall-slice cap, so the "
            f"remaining budget was discarded (budgets={budgets})"
        )


@pytest.mark.slow
def test_no_pointless_reentry_when_no_extra_budget(monkeypatch):
    """With a small limit there is no budget left to beat run 1, so do not re-enter
    on the dual lever — a restart with *less* time than run 1 cannot improve it and
    would only burn wall. (A swap-improved seed is a separate, tested lever.)"""
    budgets = _engine_budgets(monkeypatch)
    m = _hard_partition_model(18, 3, 6)

    tl = 4.0
    t0 = time.perf_counter()
    m.solve(time_limit=tl)
    wall = time.perf_counter() - t0

    assert wall <= tl + 3.0, f"overran a {tl}s limit ({wall:.1f}s)"
    # Every engine budget must stay within what the limit could justify.
    assert all(b <= tl + 1e-6 for b in budgets), f"budget exceeds the limit: {budgets}"


@pytest.mark.slow
def test_reentry_never_returns_a_worse_answer():
    """The re-entry keeps its result only when strictly better (or when it
    certifies the same incumbent), so a longer limit must never yield a worse
    objective than a shorter one on the same model."""
    short = _hard_partition_model(18, 3, 6).solve(time_limit=5)
    long = _hard_partition_model(18, 3, 6).solve(time_limit=25)

    assert short.objective is not None and long.objective is not None
    # Minimize sense: more budget must not raise the objective.
    assert long.objective <= short.objective + 1e-6, (
        f"more budget produced a worse incumbent: {short.objective} -> {long.objective}"
    )
    # And the dual bound must never cross the incumbent it certifies.
    assert long.bound <= long.objective + 1e-6
