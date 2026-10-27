"""#1990: a node whose level-1-RLT LP gives no certified verdict is re-solved
without the RLT rows (``DISCOPT_RLT_STRIP_RETRY``, default on; ``=0`` opts out).

Measured on ex1252a: the RLT-augmented node LP of the box
``x2 in [2, 3], x3 = 3, x19 = 1`` returned no certified verdict, the #671 row
filter then dropped 144 rows, and the node bounded at 14 522. The same box
without the RLT rows certifies 47 742. RLT rows only tighten a relaxation that is
valid without them, so (as C-42 does for the inherited pool) they must never be
what costs a node its bound.

The RLT failure is forced here by making every solve of an RLT-built relaxation
report ``numerical``, so the test pins the mechanism, not one LP's conditioning.
"""

import dataclasses

import discopt._relax.mccormick_lp as mlp
import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.model_utils import flat_variable_bounds

_FLAG = "DISCOPT_RLT_STRIP_RETRY"


def _model():
    m = dm.Model("rlt_strip")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.subject_to(x + y >= 3)
    m.subject_to(x * y >= 1)
    m.minimize(x * y - x)
    return m


def _box(m):
    lb, ub = flat_variable_bounds(m)
    return np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64)


@pytest.fixture
def rlt_solves_fail(monkeypatch):
    """Every solve of a relaxation built with ``rlt_level1=True`` reports
    ``numerical``; RLT-free builds are untouched. Returns the count of forced
    failures so a test can assert the stub fired (CLAUDE.md §6)."""
    forced = {"n": 0}
    orig = mlp.build_milp_relaxation

    def build(*args, **kwargs):
        milp, varmap = orig(*args, **kwargs)
        if kwargs.get("rlt_level1"):
            solve = milp.solve

            def failing_solve(*a, **k):
                forced["n"] += 1
                return dataclasses.replace(solve(*a, **k), status="numerical")

            milp.solve = failing_solve
        return milp, varmap

    monkeypatch.setattr(mlp, "build_milp_relaxation", build)
    return forced


def test_flag_defaults_on(monkeypatch):
    """Graduated default ON (#1990, performance-plan §97); ``=0`` opts out."""
    monkeypatch.delenv(_FLAG, raising=False)
    assert mlp._rlt_strip_retry_enabled() is True
    monkeypatch.setenv(_FLAG, "0")
    assert mlp._rlt_strip_retry_enabled() is False


def test_failed_rlt_node_keeps_the_rlt_free_bound(monkeypatch, rlt_solves_fail):
    m = _model()
    lb, ub = _box(m)
    plain = mlp.MccormickLPRelaxer(m, rlt_level1=False, build_incremental=False)
    ref = plain.solve_at_node(lb.copy(), ub.copy())
    assert ref.status == "optimal" and rlt_solves_fail["n"] == 0

    monkeypatch.setenv(_FLAG, "0")
    off = mlp.MccormickLPRelaxer(m, rlt_level1=True, build_incremental=False)
    assert off._rlt_applicable
    off.solve_at_node(lb.copy(), ub.copy())
    assert rlt_solves_fail["n"] > 0, "the forced RLT failure never fired"
    # OFF keeps the legacy failure handling (row filter, vertex fallback), untouched.
    assert off._rlt_strip_stats == {"retries": 0, "certified": 0}

    monkeypatch.setenv(_FLAG, "1")
    on = mlp.MccormickLPRelaxer(m, rlt_level1=True, build_incremental=False)
    r_on = on.solve_at_node(lb.copy(), ub.copy())
    assert r_on.status == "optimal"
    assert r_on.lower_bound == pytest.approx(ref.lower_bound, abs=1e-9)
    assert on._rlt_strip_stats == {"retries": 1, "certified": 1}
    # The retry restores the relaxer: the next node is built with RLT again.
    assert on._rlt_applicable and not on._rlt_strip_active


def test_certified_rlt_node_is_untouched(monkeypatch):
    """An RLT node LP that certifies never enters the retry, so its bound is
    byte-identical with the flag on or off."""
    m = _model()
    lb, ub = _box(m)
    bounds = {}
    for flag in ("0", "1"):
        monkeypatch.setenv(_FLAG, flag)
        r = mlp.MccormickLPRelaxer(m, rlt_level1=True, build_incremental=False)
        res = r.solve_at_node(lb.copy(), ub.copy())
        assert res.status == "optimal"
        assert r._rlt_strip_stats["retries"] == 0
        bounds[flag] = res.lower_bound
    assert bounds["0"] == bounds["1"]
