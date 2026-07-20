"""Issue #798 / K4 — the convex-kernel producer + soundness gate.

Verifies (1) a convex composite-of-affine MINLP is routed to the native kernel and
certified soundly (dual bound ≥ incumbent AND ≥ the analytic optimum), and (2) the
gate falls back (returns ``None``) on models it cannot prove convex — a bilinear
product, a wrong-curvature nonlinear row, a nonlinear equality, and a nonlinear
objective — so an unsound outer-approximation is never applied.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest

_ck = pytest.importorskip("discopt.solvers._convex_kernel")
build_convex_spec = _ck.build_convex_spec
solve_convex_tree = _ck.solve_convex_tree


def _scalar(m, expr_fn, name):
    """Add a single scalar constraint via the indexed API (singleton set)."""
    m.constraint(dm.RangeSet(1), lambda _i: expr_fn(), name=name, fast=False)


def _convex_minlp():
    # max x + k  s.t.  k ≤ x,  exp(x) ≤ 5 (→ x ≤ ln 5),  x∈[0,10], k∈{0..3} int.
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    k = m.integer("k", lb=0, ub=3)
    _scalar(m, lambda: k - x <= 0, "kx")
    _scalar(m, lambda: dm.exp(x) <= 5.0, "expc")
    m.maximize(x + k)
    return m


def test_convex_minlp_is_routed_and_certified_soundly():
    # max x + k  s.t.  k ≤ x,  exp(x) ≤ 5 (→ x ≤ ln 5),  x∈[0,10], k∈{0..3} int.
    # Optimum: x = ln 5, k = 1 → ln 5 + 1.
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=10.0)
    k = m.integer("k", lb=0, ub=3)
    _scalar(m, lambda: k - x <= 0, "kx")
    _scalar(m, lambda: dm.exp(x) <= 5.0, "expc")
    m.maximize(x + k)

    spec = build_convex_spec(m)
    assert spec is not None, "convex composite-of-affine model must be routed"

    r = solve_convex_tree(spec, initial_incumbent=None, time_limit_s=30.0)
    assert r["status"] == "optimal"
    truth = float(np.log(5.0)) + 1.0
    inc, bound = r["incumbent"], r["bound"]
    # Certified to tolerance, and the certificate is CONSISTENT + SOUND:
    assert abs(inc - truth) < 1e-3, f"incumbent {inc} != {truth}"
    assert bound >= inc - 1e-6 * max(1.0, abs(inc)), "cert invariant: bound ≥ incumbent"
    assert bound >= truth - 1e-6, "sound: dual bound never below the true optimum"


def test_bilinear_falls_back():
    # a*b ≤ 4 is a nonconvex (bilinear) feasible region → must NOT be routed.
    m = dm.Model()
    a = m.continuous("a", lb=0.0, ub=5.0)
    b = m.continuous("b", lb=0.0, ub=5.0)
    z = m.binary("z")
    _scalar(m, lambda: a * b <= 4.0, "bilin")
    m.maximize(a + b + z)
    assert build_convex_spec(m) is None, "bilinear model must fall back to NLP-BB"


def test_wrong_curvature_falls_back():
    # exp(x) ≥ 3 : exp is convex, so the ≥ row's ≤-normal-form (−exp) is CONCAVE →
    # nonconvex feasible region → must fall back.
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=5.0)
    z = m.binary("z")
    _scalar(m, lambda: dm.exp(x) >= 3.0, "expge")
    m.maximize(x + z)
    assert build_convex_spec(m) is None, "wrong-curvature row must fall back"


def test_nonlinear_equality_falls_back():
    m = dm.Model()
    x = m.continuous("x", lb=0.1, ub=5.0)
    z = m.binary("z")
    _scalar(m, lambda: dm.log(x) == -0.5, "logeq")
    m.maximize(x + z)
    assert build_convex_spec(m) is None, "nonlinear equality must fall back"


def test_nonlinear_objective_falls_back():
    m = dm.Model()
    x = m.continuous("x", lb=0.1, ub=5.0)
    z = m.binary("z")
    _scalar(m, lambda: x + z <= 4.0, "lin")
    m.maximize(dm.log(x) + z)  # nonlinear objective → cannot be an LP objective
    assert build_convex_spec(m) is None, "nonlinear objective must fall back"


def test_model_solve_routes_to_kernel_when_flag_on(monkeypatch):
    """DISCOPT_CONVEX_KERNEL=1 → Model.solve() routes a convex MINLP to the kernel
    and returns the certified optimum, verified feasible against the pristine model."""
    monkeypatch.setenv("DISCOPT_CONVEX_KERNEL", "1")
    truth = float(np.log(5.0)) + 1.0
    r = _convex_minlp().solve(time_limit=30)
    assert r.status == "optimal"
    assert r.gap_certified
    assert abs(r.objective - truth) < 1e-3, f"objective {r.objective} != {truth}"
    assert r.bound >= r.objective - 1e-6 * max(1.0, abs(r.objective))  # cert invariant


def test_model_solve_default_path_when_flag_off(monkeypatch):
    """Flag off → the convex fast path is not taken; the default path still solves
    the same model to the same optimum (both paths must agree)."""
    monkeypatch.setenv("DISCOPT_CONVEX_KERNEL", "0")
    truth = float(np.log(5.0)) + 1.0
    r = _convex_minlp().solve(time_limit=30)
    assert r.status in ("optimal", "feasible")
    assert r.objective is not None and abs(r.objective - truth) < 1e-2


def test_nonconvex_uses_default_path_even_with_flag_on(monkeypatch):
    """Flag on but non-convex (bilinear) → the gate declines, so the default
    (spatial) path handles it and still returns a feasible/optimal result."""
    monkeypatch.setenv("DISCOPT_CONVEX_KERNEL", "1")
    m = dm.Model()
    a = m.continuous("a", lb=0.0, ub=2.0)
    b = m.continuous("b", lb=0.0, ub=2.0)
    z = m.binary("z")
    _scalar(m, lambda: a * b <= 1.0, "bilin")
    _scalar(m, lambda: a + b + z <= 5.0, "lin")
    m.maximize(a + b + z)
    r = m.solve(time_limit=30)
    assert r.status in ("optimal", "feasible")
    assert r.objective is not None
