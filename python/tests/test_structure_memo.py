"""``discopt._structure_memo``: one linearity / classification answer per model
structure, and a miss whenever anything the answer reads has changed.

Plus the vectorised ``_recentre.plan_shifts``, checked against the per-variable
reference it replaced (2026-11-14: 1.1 s -> 0.06 s on a 72,747-scalar file model).
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as S
import numpy as np
import pytest
from discopt import Model
from discopt._relax import problem_classifier as pc
from discopt._relax.problem_classifier import ProblemClass
from discopt.modeling import _recentre


def _milp() -> tuple[Model, object, object]:
    m = Model("memo")
    x = m.integer("x", shape=3, lb=0, ub=5)
    y = m.continuous("y", lb=-1, ub=4)
    m.subject_to(x[0] + 2 * x[1] - y <= 6)
    m.minimize(dm.sum(x) + y)
    return m, x, y


def _counting(monkeypatch):
    calls = {"classify": 0, "linear": 0}
    orig_c, orig_l = pc._classify_problem_uncached, S._milp_is_exactly_linear_uncached

    def c(m):
        calls["classify"] += 1
        return orig_c(m)

    def lin(m):
        calls["linear"] += 1
        return orig_l(m)

    monkeypatch.setattr(pc, "_classify_problem_uncached", c)
    monkeypatch.setattr(S, "_milp_is_exactly_linear_uncached", lin)
    return calls


def test_repeated_queries_compute_once(monkeypatch):
    calls = _counting(monkeypatch)
    m, _, _ = _milp()
    for _ in range(4):
        assert pc.classify_problem(m) == ProblemClass.MILP
        assert S._milp_is_exactly_linear(m)
    assert calls == {"classify": 1, "linear": 1}


def test_structural_changes_miss(monkeypatch):
    calls = _counting(monkeypatch)
    m, x, y = _milp()
    assert pc.classify_problem(m) == ProblemClass.MILP and S._milp_is_exactly_linear(m)
    # A nonlinear row: both answers change.
    m.subject_to(x[0] * y <= 3)
    assert pc.classify_problem(m) == ProblemClass.MIQCP
    assert not S._milp_is_exactly_linear(m)
    # A new objective.
    m2, x2, y2 = _milp()
    assert pc.classify_problem(m2) == ProblemClass.MILP
    m2.minimize(y2 * y2 + dm.sum(x2))
    assert pc.classify_problem(m2) == ProblemClass.MIQP
    # A type change in place (implied-integer detection does this): integrality.
    m3 = Model("t")
    z = m3.continuous("z", lb=0, ub=3)
    m3.subject_to(2 * z <= 5)
    m3.minimize(z)
    assert pc.classify_problem(m3) == ProblemClass.LP
    from discopt.modeling.core import VarType

    z.var_type = VarType.INTEGER
    assert pc.classify_problem(m3) == ProblemClass.MILP
    assert calls["classify"] == 6 and calls["linear"] == 2


def test_bounds_do_not_miss(monkeypatch):
    calls = _counting(monkeypatch)
    m, x, _ = _milp()
    pc.classify_problem(m)
    x.ub = np.array([1.0, 2.0, 3.0])
    pc.classify_problem(m)
    assert calls["classify"] == 1


def test_parameter_value_is_in_the_key(monkeypatch):
    calls = _counting(monkeypatch)
    m = Model("p")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    p = m.parameter("p", value=0.0)
    m.subject_to(x + y <= 1)
    m.minimize(x + p * x * y)
    first = pc.classify_problem(m)
    p.value = 1.0
    pc.classify_problem(m)
    assert calls["classify"] == 2, first


def _plan_reference(model: Model, ratio: float) -> dict:
    """The per-variable ``plan_shifts`` as it was before vectorising (verbatim logic)."""
    plan = {}
    for v in model._variables:
        lb = np.asarray(v.lb, dtype=np.float64)
        ub = np.asarray(v.ub, dtype=np.float64)
        lb_fin = np.isfinite(lb) & (np.abs(lb) < _recentre._OPEN)
        ub_fin = np.isfinite(ub) & (np.abs(ub) < _recentre._OPEN)
        width = np.where(lb_fin & ub_fin, ub - lb, np.inf)
        narrow = lb_fin & ub_fin & (np.abs(lb) >= ratio * np.maximum(1.0, width))
        from_lb = lb_fin & ~narrow & (lb >= ratio) & (width >= lb)
        from_ub = ub_fin & ~lb_fin & (ub <= -ratio)
        if np.any(narrow | from_lb | from_ub):
            plan[id(v)] = np.where(
                narrow | from_lb, np.round(lb), np.where(from_ub, np.round(ub), 0.0)
            )
    return plan


@pytest.mark.parametrize("seed", range(20))
def test_vectorised_plan_shifts_matches_the_per_variable_reference(seed):
    rng = np.random.default_rng(seed)
    pool = [0.0, 1.0, -3.0, 150.25, 1e6, -1e6, 1e6 + 3, 2e10, np.inf, -np.inf, -150.0, 99.0]
    m = Model("r")
    compared = 0
    for k in range(int(rng.integers(1, 9))):
        shape = [(), (3,), (2, 2)][int(rng.integers(0, 3))]
        size = int(np.prod(shape)) if shape else 1
        lo = rng.choice(pool, size=size)
        hi = np.array([lo_i + rng.choice([0.0, 2.0, 1e3, 1e7, np.inf]) for lo_i in lo])
        lo = np.where(np.isinf(lo) & (lo > 0), -np.inf, lo)
        hi = np.where(np.isnan(hi), np.inf, hi)  # -inf + inf
        lb, ub = (lo.reshape(shape), hi.reshape(shape)) if shape else (float(lo[0]), float(hi[0]))
        maker = m.integer if rng.integers(0, 2) else m.continuous
        maker(f"v{k}", shape=shape, lb=lb, ub=ub)
    got = _recentre.plan_shifts(m, 100.0)
    want = _plan_reference(m, 100.0)
    assert got.keys() == want.keys()
    for key in want:
        assert np.shape(got[key]) == np.shape(want[key])
        assert np.array_equal(got[key], want[key])
        compared += 1
    assert compared == len(want)
