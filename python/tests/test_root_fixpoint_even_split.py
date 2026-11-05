"""``DISCOPT_ROOT_FIXPOINT_EVEN_SPLIT``: the root fixpoint's rounds share its OBBT budget.

Legacy hands round 1 ``obbt_stage_frac`` (85%) of the WHOLE loop budget, and round 2
gets 85% of the remainder. On GDPlib modprodnet hull that left round 2 1.4 s of an
11.6 s loop and the solve took 249 nodes; with the same total split evenly it took 5.
Pinned here: legacy budgets unchanged with the flag off; with it on each round gets
``frac * remaining / rounds_left``; the loop stays tighten-only either way.
"""

from __future__ import annotations

import time

import discopt._relax.root_reduce as RR
import discopt.modeling as dm
import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _model():
    m = dm.Model("even_split")
    x = m.continuous("x", lb=-4.0, ub=4.0)
    y = m.continuous("y", lb=-4.0, ub=4.0)
    m.subject_to(x * y >= 1.0)
    m.minimize(x + y)
    return m


def _budgets(monkeypatch, flag: str | None, total: float = 10.0):
    if flag is None:
        monkeypatch.delenv("DISCOPT_ROOT_FIXPOINT_EVEN_SPLIT", raising=False)
    else:
        monkeypatch.setenv("DISCOPT_ROOT_FIXPOINT_EVEN_SPLIT", flag)
    seen: list[float] = []

    def stage(model, lb, ub, cutoff, *, rounds, deadline, prefer_pounce):
        seen.append(deadline - time.perf_counter())
        # Tighten one bound a little each round, so the loop runs every round.
        lb = lb.copy()
        lb[0] = min(lb[0] + 0.1, ub[0])
        return lb, ub, False, 1

    monkeypatch.setattr(RR, "_stage_obbt", stage)
    res = RR.run_root_fixpoint(
        _model(),
        np.array([-4.0, -4.0]),
        np.array([4.0, 4.0]),
        deadline=time.perf_counter() + total,
        stages=("obbt",),
        max_rounds=2,
        measure_bound=False,
    )
    assert np.all(res.lb >= -4.0) and np.all(res.ub <= 4.0)
    assert len(seen) == 2, seen
    return seen


def test_flag_off_keeps_the_legacy_split(monkeypatch):
    r1, r2 = _budgets(monkeypatch, None)
    assert r1 == pytest.approx(8.5, abs=0.05)
    assert r2 == pytest.approx(8.5, abs=0.05)  # the stub returns at once: 85% again
    r1z, _ = _budgets(monkeypatch, "0")
    assert r1z == pytest.approx(8.5, abs=0.05)


def test_flag_on_divides_by_the_rounds_still_to_run(monkeypatch):
    r1, r2 = _budgets(monkeypatch, "1")
    assert r1 == pytest.approx(4.25, abs=0.05)  # 0.85 * 10 / 2
    assert r2 == pytest.approx(8.5, abs=0.05)  # last round: 0.85 * remaining / 1


def test_flag_on_never_exceeds_the_loop_deadline(monkeypatch):
    for total in (0.5, 3.0, 20.0):
        r1, r2 = _budgets(monkeypatch, "1", total)
        assert 0 < r1 <= total and 0 < r2 <= total
