"""#1287: the in-house Rust MILP kernel must round integer bounds and verify rows.

Two false ``optimal`` results from the kernel behind ``DISCOPT_LP_MILP_BACKEND=rust``
and ``solve_milp_with_lazy_cuts``:

1. A fractional bound on an integer column was a legal LP vertex value, so
   ``x ∈ [0, 1.5]`` integer came back as ``x = 1.5``.
2. On the ``±9.999e19`` default box an integral LP vertex with
   ``y = 9.999e19, w = -9.999e19`` was promoted to the incumbent: ``y + w``
   cancels to ``0`` in floating point, so the violated row reads as satisfied and
   an infeasible model was reported ``optimal``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("discopt._rust")

import discopt.modeling as dm  # noqa: E402
from discopt.solvers import SolveStatus  # noqa: E402
from discopt.solvers import milp_simplex as MS  # noqa: E402


@pytest.mark.parametrize("with_row", [False, True])
def test_fractional_integer_upper_bound_is_rounded(with_row):
    kw = dict(A_ub=np.array([[1.0]]), b_ub=np.array([5.0])) if with_row else {}
    r = MS.solve_milp(np.array([-1.0]), bounds=[(0.0, 1.5)], integrality=np.array([1]), **kw)
    assert r.status == SolveStatus.OPTIMAL
    assert r.x is not None
    assert r.x[0] == pytest.approx(1.0, abs=1e-9)
    assert r.objective == pytest.approx(-1.0, abs=1e-9)


def test_integer_box_without_an_integer_is_infeasible():
    r = MS.solve_milp(
        np.array([1.0]),
        A_ub=np.array([[1.0]]),
        b_ub=np.array([5.0]),
        bounds=[(0.2, 0.8)],
        integrality=np.array([1]),
    )
    assert r.status == SolveStatus.INFEASIBLE


def test_lazy_master_respects_fractional_integer_bounds():
    # max x0 + x1 with x0 + x1 <= 10, both integer in [0, 3.5] / [0, 4.5]: the
    # optimum is 3 + 4 = 7. The lazy callback accepts every point.
    calls = []

    def accept(x):
        calls.append(np.array(x))
        return None

    r = MS.solve_milp_with_lazy_cuts(
        np.array([-1.0, -1.0]),
        A_ub=np.array([[1.0, 1.0]]),
        b_ub=np.array([10.0]),
        bounds=[(0.0, 3.5), (0.0, 4.5)],
        integrality=np.array([1, 1]),
        lazy_callback=accept,
    )
    assert calls, "the lazy callback never ran"
    assert r.x is not None
    assert r.x[0] <= 3.0 + 1e-9 and r.x[1] <= 4.0 + 1e-9, r.x
    assert np.allclose(r.x, np.round(r.x), atol=1e-6), r.x
    assert r.objective == pytest.approx(-7.0, abs=1e-6)


def _default_box_model(x_lb: float) -> tuple[dm.Model, dict]:
    m = dm.Model("default_box")
    v = {
        "x": m.continuous("x", lb=x_lb),
        "y": m.continuous("y"),
        "w": m.continuous("w"),
        "z": m.binary("z"),
    }
    m.subject_to(v["x"] + v["y"] + v["w"] - v["z"] <= 1)
    m.subject_to(v["y"] + v["w"] >= -1)
    m.minimize(v["x"])
    return m, v


def _exact_rows_hold(x: dict) -> bool:
    from fractions import Fraction

    f = {k: Fraction(float(np.asarray(val))) for k, val in x.items()}
    tol = Fraction(1, 10**5)
    return (f["x"] + f["y"] + f["w"] - f["z"] <= 1 + tol) and (f["y"] + f["w"] >= -1 - tol)


def test_default_box_infeasible_model_is_not_optimal(monkeypatch):
    # x <= 1 + z - (y + w) <= 3 < 3.5: infeasible.
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    m, _ = _default_box_model(3.5)
    r = m.solve(time_limit=60)
    assert r.status == "infeasible", (r.status, r.objective, r.x)


def test_default_box_feasible_model_returns_a_point_that_satisfies_its_rows(monkeypatch):
    # With x >= 3 the optimum is 3 at z = 1, y + w = -1.
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    m, _ = _default_box_model(3.0)
    r = m.solve(time_limit=60)
    assert r.status == "optimal", (r.status, r.objective, r.x)
    assert r.objective == pytest.approx(3.0, abs=1e-5)
    assert _exact_rows_hold(r.x), r.x
