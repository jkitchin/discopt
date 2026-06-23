"""Regression: ``_solve_milp_highs`` must charge already-elapsed time against the
budget, not run a *fresh* full ``time_limit``.

Sequel to issue #291. On a MINLP that the #285 reformulation turns into a pure
MILP, the routing first tries the fast Rust simplex engine
(``_solve_milp_simplex``) under a bounded slice of the budget. When that engine
stalls and defers (returns ``None``), control falls through to the HiGHS MILP
path. ``_solve_milp_highs`` was handed the *full* ``time_limit`` and forwarded it
verbatim to HiGHS — so the deferred attempt's wall time stacked on top of HiGHS's
own full budget, overrunning the user's ``time_limit`` by the upstream cost
(tln6 / rsyn0810m03hfsg ran ~40 s on a 30 s limit: ~10 s simplex + a fresh 30 s
HiGHS).

Fix: ``_solve_milp_highs`` subtracts ``perf_counter() - t_start`` from
``time_limit`` (floored at a small positive value) before forwarding it, so HiGHS
finishes by the *shared* deadline ``t_start + time_limit``.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import discopt.solver as S  # noqa: E402


def _knapsack_milp() -> dm.Model:
    m = dm.Model("milp")
    x1, x2, x3 = m.binary("x1"), m.binary("x2"), m.binary("x3")
    m.minimize(-3 * x1 - 4 * x2 - 5 * x3)
    m.subject_to(2 * x1 + 3 * x2 + 4 * x3 <= 6)
    return m  # optimum x1=x3=1 -> -8


def test_highs_charges_elapsed_time_against_budget(monkeypatch):
    """With ``t_start`` 10 s in the past and a 30 s ``time_limit``, HiGHS must be
    handed ~20 s (the *remaining* budget), not a fresh 30 s."""
    captured: dict[str, float | None] = {}

    import discopt.solvers.milp_highs as milp_highs

    real_solve = milp_highs.solve_milp

    def _spy(*args, **kwargs):
        captured["time_limit"] = kwargs.get("time_limit")
        return real_solve(*args, **kwargs)

    # _solve_milp_highs imports solve_milp lazily from the module, so patch there.
    monkeypatch.setattr(milp_highs, "solve_milp", _spy)

    model = _knapsack_milp()
    t_start = time.perf_counter() - 10.0  # pretend 10 s already spent upstream
    result = S._solve_milp_highs(model, t_start, time_limit=30.0, gap_tolerance=1e-4)

    # The fix must have been exercised (HiGHS is available in this build).
    assert "time_limit" in captured, "HiGHS solve_milp was not called"
    forwarded = captured["time_limit"]
    assert forwarded is not None
    # Remaining budget ~= 30 - 10 = 20 s, never the fresh 30 s, never <= 0.
    assert forwarded < 30.0, f"HiGHS got a fresh budget ({forwarded}s), elapsed not charged"
    assert 18.0 <= forwarded <= 21.0, f"expected ~20 s remaining, got {forwarded}s"
    # Correctness is unaffected — the tiny model still solves to the optimum.
    assert result is not None and result.objective is not None
    assert abs(result.objective - (-8.0)) < 1e-6


def test_highs_floors_exhausted_budget_positive(monkeypatch):
    """When the budget is already exhausted, HiGHS must still receive a small
    *positive* limit (not 0.0, which HiGHS reads as 'stop immediately', nor a
    negative value)."""
    captured: dict[str, float | None] = {}

    import discopt.solvers.milp_highs as milp_highs

    real_solve = milp_highs.solve_milp

    def _spy(*args, **kwargs):
        captured["time_limit"] = kwargs.get("time_limit")
        return real_solve(*args, **kwargs)

    monkeypatch.setattr(milp_highs, "solve_milp", _spy)

    model = _knapsack_milp()
    t_start = time.perf_counter() - 100.0  # budget blown long ago
    S._solve_milp_highs(model, t_start, time_limit=30.0, gap_tolerance=1e-4)

    forwarded = captured.get("time_limit")
    assert forwarded is not None
    assert forwarded > 0.0, f"HiGHS got a non-positive budget ({forwarded}s)"
