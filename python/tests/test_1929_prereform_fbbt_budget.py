"""#1929: the FBBT before the factorable reform is capped at the solve's time left.

``solve_model`` ran ``tighten_root_bounds_with_fbbt`` before the factorable reform
without ``time_limit_ms``; ``_presolve_deadline.afford`` only decides whether that
block starts. On ``transswitch2383wpp`` (19,850 vars / 39,701 rows) the call ran
175 s uncapped against a 60 s solve. It now gets the remaining budget, as the
root-presolve call does (#863).
"""

import discopt.modeling as dm
import discopt.solvers._root_presolve as rp
import pytest


def _model():
    # A fractional power of a product: factorable work, so the pre-reform FBBT runs.
    m = dm.Model("prereform_fbbt")
    x = m.continuous("x", lb=0.5, ub=4)
    y = m.continuous("y", lb=0.5, ub=4)
    z = m.continuous("z", lb=0, ub=100)
    m.subject_to((x * y) ** 0.5 + x <= z)
    m.subject_to(x + y >= 2)
    m.minimize(z + (x * y) ** 1.5)
    return m


@pytest.mark.unit
def test_prereform_fbbt_receives_the_remaining_budget(monkeypatch):
    seen = []
    real = rp.tighten_root_bounds_with_fbbt

    def spy(*args, **kw):
        seen.append(kw.get("time_limit_ms"))
        return real(*args, **kw)

    monkeypatch.setattr(rp, "tighten_root_bounds_with_fbbt", spy)
    r = _model().solve(time_limit=20)
    assert r.status in ("optimal", "feasible")
    assert len(seen) >= 2, "the pre-reform and root FBBT calls should both run"
    # Every call is capped, and no cap exceeds the whole limit.
    assert all(ms is not None and 0 < ms <= 20_000 for ms in seen), seen


@pytest.mark.unit
def test_an_uncapped_solve_passes_no_cap(monkeypatch):
    seen = []
    real = rp.tighten_root_bounds_with_fbbt

    def spy(*args, **kw):
        seen.append(kw.get("time_limit_ms"))
        return real(*args, **kw)

    monkeypatch.setattr(rp, "tighten_root_bounds_with_fbbt", spy)
    _model().solve(time_limit=float("inf"))
    assert seen and all(ms is None for ms in seen), seen


@pytest.mark.unit
def test_fbbt_budget_ms_converts_a_deadline():
    import time

    assert rp.fbbt_budget_ms(None) is None
    assert rp.fbbt_budget_ms(float("inf")) is None
    # A spent deadline is 1 ms, never 0 (which the Rust binding reads as "no cap").
    assert rp.fbbt_budget_ms(time.perf_counter() - 5.0) == 1
    ms = rp.fbbt_budget_ms(time.perf_counter() + 2.0)
    assert 1500 < ms <= 2000
