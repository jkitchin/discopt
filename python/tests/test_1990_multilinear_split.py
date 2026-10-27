"""#1990: products of three or more variables reach the native kernel as bilinear
chains (``_relax/multilinear_split.py``, ``DISCOPT_KERNEL_MULTILINEAR_SPLIT``).

The kernel's spec builder declines a relaxation with trilinear or multilinear
terms. On ex1252a (``aux * x0 * x3 * x18`` per line after the factorable lift) that
left the solve on the Python tree, ~280 nodes in 600 s; split, the kernel certifies
it in 2,048 nodes (``docs/dev/performance-plan.md`` §97).
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.model_utils import flat_variable_bounds
from discopt._relax.multilinear_split import extend_point, split_multilinear_products
from discopt._relax.spatial_producer import build_spatial_kernel_spec
from discopt._tape_nlp_evaluator import make_evaluator

_FLAG = "DISCOPT_KERNEL_MULTILINEAR_SPLIT"


def _model():
    m = dm.Model("ml")
    a = m.integer("a", lb=0, ub=3)
    b = m.integer("b", lb=1, ub=3)
    y = m.binary("y")
    f = m.continuous("f", lb=0.5, ub=4)
    g = m.continuous("g", lb=-2, ub=1)
    m.subject_to(a <= 3 * y)
    m.subject_to(a + b >= 2)
    m.subject_to(f + g >= 1)
    m.minimize(2.0 * f * a * b * y + a * b * g + 0.5 * f)
    return m


def _box(m):
    lb, ub = flat_variable_bounds(m)
    return np.asarray(lb, dtype=np.float64), np.asarray(ub, dtype=np.float64)


def test_split_is_exact_and_bounded():
    m = _model()
    lb, ub = _box(m)
    out = split_multilinear_products(m, lb, ub)
    assert out is not None
    sm, slb, sub = out
    n = lb.size
    # Products y*a*b*f (binary, integers, continuous) and a*b*g: y*a, y*a*b,
    # y*a*b*f, a*b, a*b*g.
    assert slb.size == n + 5
    names = [v.name for v in sm._variables]
    assert names[:n] == [v.name for v in m._variables]
    ev0 = make_evaluator(m)
    ev1 = make_evaluator(sm)
    rng = np.random.default_rng(1990)
    checks = 0
    for _ in range(200):
        x = rng.uniform(lb, ub)
        x[:3] = np.round(x[:3])
        a, b, y, f, g = x
        xs = extend_point(sm, x)
        assert sorted(np.round(xs[n:], 9)) == sorted(
            np.round([y * a, y * a * b, y * a * b * f, a * b, a * b * g], 9)
        )
        # The aux bounds are interval products, so they enclose every value.
        assert np.all(xs[n:] >= slb[n:] - 1e-12) and np.all(xs[n:] <= sub[n:] + 1e-12)
        assert float(ev1.evaluate_objective(xs)) == pytest.approx(
            float(ev0.evaluate_objective(x)), rel=1e-12, abs=1e-12
        )
        # Every aux row holds with equality at the extended point.
        g_split = np.asarray(ev1.evaluate_constraints(xs))
        assert np.max(np.abs(g_split[-5:])) < 1e-12
        checks += 1
    assert checks == 200


def test_split_admits_the_kernel_spec():
    m = _model()
    lb, ub = _box(m)
    assert build_spatial_kernel_spec(m, bounds=(lb, ub)) is None
    sm, slb, sub = split_multilinear_products(m, lb, ub)
    assert build_spatial_kernel_spec(sm, bounds=(slb, sub)) is not None


def test_flag_defaults_on(monkeypatch):
    from discopt.solver import _kernel_multilinear_split_enabled

    monkeypatch.delenv(_FLAG, raising=False)
    assert _kernel_multilinear_split_enabled() is True
    monkeypatch.setenv(_FLAG, "0")
    assert _kernel_multilinear_split_enabled() is False


def test_no_product_no_split():
    m = dm.Model("bil")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    m.minimize(x * y - x)
    assert split_multilinear_products(m, *_box(m)) is None


def test_unbounded_factor_declines():
    m = dm.Model("unb")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    z = m.continuous("z", lb=0)
    m.subject_to(z <= 5)
    m.minimize(x * y * z)
    assert split_multilinear_products(m, *_box(m)) is None


@pytest.mark.parametrize("flag", ["0", "1"])
def test_solve_matches_and_routes(monkeypatch, flag):
    monkeypatch.setenv(_FLAG, flag)
    r = _model().solve(time_limit=60)
    assert r.status == "optimal"
    # min over the box: a*b*g with g = -2 and a*b = 9 needs y = 1, f >= 3 -> f = 3
    # 2*3*9 - 18 + 1.5 = 37.5; a = 0 (b >= 2) gives 0.25 with f = 0.5, g = 0.5.
    assert r.objective == pytest.approx(0.25, abs=1e-5)
    assert math.isfinite(r.bound) and r.bound <= r.objective + 1e-6
    assert set(r.x) == {"a", "b", "y", "f", "g"}
    split = (r.solver_stats or {}).get("tree/multilinear_split_cols")
    if flag == "1":
        assert r.algorithm_route.startswith("native-spatial")
        assert split == 5.0
    else:
        assert split is None


@pytest.mark.slow
@pytest.mark.correctness
def test_ex1252a_certifies_by_default(monkeypatch):
    """#1990's Done criterion: ex1252a certifies at 120 s on the default path.

    Before: the Python tree, bound 19,389 (incumbent 131,124) at 120 s. After: the
    kernel on the split, with the integral-point primal step, about 20 s here.
    """
    import os

    for flag in (_FLAG, "DISCOPT_NATIVE_NLP_PRIMAL", "DISCOPT_HOOK_INTEGRAL_POINTS"):
        monkeypatch.delenv(flag, raising=False)
    path = os.path.join(os.path.dirname(__file__), "data", "minlplib", "ex1252a.nl")
    r = dm.from_nl(path).solve(time_limit=120)
    opt = 128893.741
    assert r.status == "optimal", (r.status, r.objective, r.bound)
    assert r.objective == pytest.approx(opt, rel=1e-4)
    assert r.bound <= opt * (1 + 1e-6)
    assert r.algorithm_route.startswith("native-spatial")
