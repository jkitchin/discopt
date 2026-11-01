"""#1917: root path-aggregation c-MIR with variable-bound substitution.

The separator (``discopt._relax.vub_cmir``) targets big-M rows whose continuous
columns carry variable upper bounds. The canonical case, from GDPlib ``syngas``
under ``gdp.bigm``: a switch ``f_out <= 5 y`` and a big-M balance
``f_out - f_in <= 5 (1 - y)``. Together they imply ``f_out <= f_in`` at every
mixed-integer point, but at ``y = 0.4`` the LP lets ``f_out = 2`` with ``f_in = 0``.

Tests:
* the separator finds a cut that removes that LP point, and every cut it returns
  holds at every sampled mixed-integer feasible point (counted, CLAUDE.md §6);
* on a small big-M MINLP the root driver raises the McCormick LP bound and its
  rows hold at every sampled feasible point of the MODEL;
* the flag is OFF by default and OFF is a no-op: the driver is never called;
* ON and OFF certify the same optimum.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt import Model
from discopt._relax import vub_cmir
from discopt._relax.model_utils import flat_variable_bounds

pytestmark = pytest.mark.smoke


def _bigm_model() -> Model:
    """Demand 2 is met by one of two units. A unit's outflow needs it switched on
    (``y_k``) and then only passes what flows in. The ``exp`` term keeps the model
    off the native quadratic kernel so the solve takes the McCormick LP route."""
    m = Model("bigm_flow")
    f_in1 = m.continuous("f_in1", lb=0.0, ub=5.0)
    f_out1 = m.continuous("f_out1", lb=0.0, ub=5.0)
    y1 = m.binary("y1")
    f_in2 = m.continuous("f_in2", lb=0.0, ub=5.0)
    f_out2 = m.continuous("f_out2", lb=0.0, ub=5.0)
    y2 = m.binary("y2")
    x = m.continuous("x", lb=1.0, ub=2.0)
    for f_in, f_out, y in ((f_in1, f_out1, y1), (f_in2, f_out2, y2)):
        m.subject_to(f_out <= 5.0 * y)
        m.subject_to(f_out - f_in <= 5.0 * (1.0 - y))
    m.subject_to(f_out1 + f_out2 >= 2.0)
    m.minimize(x * f_in1 + 2.5 * f_in2 + 3.0 * y1 + y2 + dm.exp(x))
    return m


#: optimum: unit 1 on, f_in1 = f_out1 = 2, x = 1 -> 2 + 3 + e (unit 2 costs 6 + e)
_OPT = 5.0 + float(np.e)


def _model_feasible(p: np.ndarray) -> bool:
    f_in1, f_out1, y1, f_in2, f_out2, y2, x = p
    tol = 1e-12
    ok = 1 <= x <= 2 and f_out1 + f_out2 >= 2 - tol
    for f_in, f_out, y in ((f_in1, f_out1, y1), (f_in2, f_out2, y2)):
        ok = ok and 0 <= f_in <= 5 and 0 <= f_out <= 5 and y in (0.0, 1.0)
        ok = ok and f_out <= 5 * y + tol and f_out - f_in <= 5 * (1 - y) + tol
    return bool(ok)


def _sample_points() -> list[np.ndarray]:
    grid = np.linspace(0.0, 5.0, 6)
    pts = [
        np.array([a, b, y1, c, d, y2, x])
        for a, b, c, d, y1, y2, x in itertools.product(
            grid, grid, grid, grid, (0.0, 1.0), (0.0, 1.0), (1.0, 1.5, 2.0)
        )
    ]
    pts = [p for p in pts if _model_feasible(p)]
    assert len(pts) > 100
    return pts


def test_separator_finds_the_flow_cut_and_it_is_valid():
    # columns: f_in, f_out, y ; rows A x <= b
    A = sp.csr_matrix(
        np.array(
            [
                [0.0, 1.0, -5.0],  # f_out - 5 y <= 0          (VUB)
                [-1.0, 1.0, 5.0],  # f_out - f_in + 5 y <= 5   (big-M balance)
            ]
        )
    )
    b = np.array([0.0, 5.0])
    lb = np.zeros(3)
    ub = np.array([5.0, 5.0, 1.0])
    is_int = np.array([False, False, True])
    vub, vlb = vub_cmir.detect_variable_bounds(A, b, lb, ub, is_int)
    assert vub == {1: [(2, 5.0)]} and vlb == {}
    xstar = np.array([0.0, 2.0, 0.4])  # LP point: flow fabricated from nothing
    cuts = vub_cmir.separate_path_cmir(A, b, xstar, lb, ub, is_int, vub, vlb)
    assert cuts, "the separator found no cut at the fabricated-flow point"
    assert max(float(a @ xstar - r) for a, r in cuts) > 1e-3

    grid = np.linspace(0.0, 5.0, 21)
    checks = 0
    for fi, fo, y in itertools.product(grid, grid, (0.0, 1.0)):
        p = np.array([fi, fo, y])
        if np.any(A @ p > b + 1e-12):
            continue
        for a, r in cuts:
            checks += 1
            assert a @ p <= r + 1e-9, f"cut {a} <= {r} removes feasible {p}"
    assert checks > 100, checks


def test_root_driver_raises_the_bound_with_valid_rows():
    m = _bigm_model()
    lb, ub = flat_variable_bounds(m)
    n = len(lb)
    res = vub_cmir.root_vub_cmir_cuts(m, lb, ub, n, deadline=float("inf"))
    assert res.A is not None and res.A.shape[1] == n, res.stop_reason
    assert res.n_vub >= 1
    assert len(res.bound_trace) >= 2 and res.bound_trace[-1] > res.bound_trace[0] + 1e-3

    checks = 0
    for p in _sample_points():
        for a, r in zip(res.A, res.b):
            checks += 1
            assert a @ p <= r + 1e-9, f"root row {a} <= {r} removes feasible {p}"
    assert checks > 10, checks


def test_flag_off_by_default_and_a_no_op(monkeypatch):
    monkeypatch.delenv("DISCOPT_VUB_CMIR_ROOT", raising=False)
    assert not vub_cmir.vub_cmir_root_enabled()

    def _boom(*a, **k):
        raise AssertionError("root_vub_cmir_cuts called with the flag OFF")

    monkeypatch.setattr(vub_cmir, "root_vub_cmir_cuts", _boom)
    r = _bigm_model().solve(time_limit=60)
    assert r.status == "optimal"
    assert not any(k.startswith("vub_cmir/") for k in (r.solver_stats or {}))


def test_on_and_off_certify_the_same_optimum(monkeypatch):
    out = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("DISCOPT_VUB_CMIR_ROOT", flag)
        r = _bigm_model().solve(time_limit=60)
        assert r.status == "optimal", (flag, r.status)
        out[flag] = r
    for r in out.values():
        assert r.objective == pytest.approx(_OPT, abs=1e-5)
    stats = out["1"].solver_stats or {}
    assert stats.get("vub_cmir/rows", 0.0) >= 1.0, stats
    assert stats.get("vub_cmir/rejected", 0.0) == 0.0
