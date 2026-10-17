"""Single-variable rows of a continuous QP reach the backend as column bounds.

COPS ``torsion`` writes ``-D <= v <= D`` as two rows per cell on a free ``v``,
and the cells on the boundary have ``D = 0``, so their two rows pin ``v`` and
leave the interior-point backend no interior. At 50x50 the QP route returned an
uncertified -0.4178480 after 13.6 s (reference -0.418087), while the same model
with the rows declared as bounds certified in 0.9 s. The rows are now folded
into the box before the solve and their duals restored after it, so every
check and every reported dual is on the model's own rows.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solver import _fold_singleton_rows, _unfold_singleton_duals

pytest.importorskip("pounce")


def _torsion(nx: int, as_rows: bool):
    hx, hy = 1.0 / (nx + 1), 1.0 / (nx + 1)
    D = np.array(
        [
            [min(min(i, nx - i + 1) * hx, min(j, nx - j + 1) * hy) for j in range(nx + 2)]
            for i in range(nx + 2)
        ]
    )
    m = dm.Model("torsion_shape")
    if as_rows:
        v = m.continuous("v", shape=(nx + 2, nx + 2), lb=-np.inf, ub=np.inf)
    else:
        v = m.continuous("v", shape=(nx + 2, nx + 2), lb=-D, ub=D)
    quad = dm.sum(((v[1:, :] - v[:-1, :]) / hx) ** 2) + dm.sum(((v[:, 1:] - v[:, :-1]) / hy) ** 2)
    m.minimize(0.5 * hx * hy * quad - 5.0 * hx * hy * dm.sum(v))
    if as_rows:
        for i in range(nx + 2):
            for j in range(nx + 2):
                m.subject_to(v[i, j] >= -float(D[i, j]))
                m.subject_to(v[i, j] <= float(D[i, j]))
    return m, v, D


def test_torsion_row_form_certifies_like_the_bound_form():
    m_rows, v, D = _torsion(10, as_rows=True)
    r = m_rows.solve(time_limit=120, initial_solution={v: D})
    m_box, vb, _ = _torsion(10, as_rows=False)
    ref = m_box.solve(time_limit=120, initial_solution={vb: D})
    assert ref.status == "optimal" and ref.gap_certified
    # Before the fold: status "feasible", objective -0.4099260, no bound.
    assert r.status == "optimal" and r.gap_certified, (r.status, r.objective, r.bound)
    assert r.bound <= ref.objective + 1e-9
    assert ref.bound <= r.objective + 1e-9
    assert abs(r.objective - ref.objective) <= 1e-4 * (1 + abs(ref.objective))
    x = np.asarray(r.x["v"])
    assert np.all(np.abs(x) <= D + 1e-6)


def test_folded_row_keeps_its_named_dual():
    # min (x - 2)^2 + (y + 1)^2  s.t.  x <= 1 (a row),  y >= 0 (a row),  x + y <= 5.
    # At (1, 0): mu_x = 2, mu_y = 2 in the reported mu >= 0 convention.
    m = dm.Model("fold_duals")
    x = m.continuous("x", lb=-10, ub=10)
    y = m.continuous("y", lb=-10, ub=10)
    m.minimize((x - 2) ** 2 + (y + 1) ** 2)
    m.subject_to(x <= 1, name="xcap")
    m.subject_to(y >= 0, name="yfloor")
    m.subject_to(x + y <= 5, name="sum")
    r = m.solve(time_limit=60)
    assert r.status == "optimal" and r.gap_certified
    np.testing.assert_allclose([r.x["x"], r.x["y"]], [1.0, 0.0], atol=1e-6)
    cd = r.constraint_duals
    assert cd is not None
    np.testing.assert_allclose(float(np.ravel(cd["xcap"])[0]), 2.0, atol=1e-5)
    np.testing.assert_allclose(float(np.ravel(cd["yfloor"])[0]), 2.0, atol=1e-5)
    np.testing.assert_allclose(float(np.ravel(cd["sum"])[0]), 0.0, atol=1e-5)
    # The bound multipliers belong to the rows, not the (inactive) declared box.
    for fam in (r.bound_duals_lower, r.bound_duals_upper):
        if fam is not None:
            for val in fam.values():
                np.testing.assert_allclose(np.ravel(val), 0.0, atol=1e-5)


def test_fold_moves_rows_into_the_box_and_back():
    A_ub = np.array(
        [
            [2.0, 0.0, 0.0],  # x0 <= 1.5
            [0.0, -1.0, 0.0],  # x1 >= -0.25
            [1.0, 1.0, 1.0],  # kept
            [3.0, 0.0, 0.0],  # x0 <= 2, looser than row 0: dropped, dual 0
        ]
    )
    b_ub = np.array([3.0, 0.25, 4.0, 6.0])
    A_eq = np.array([[0.0, 0.0, 4.0], [1.0, 0.0, 1.0]])  # x2 == 0.5 ; kept
    b_eq = np.array([2.0, 2.0])
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (0.0, 5.0)]
    out = _fold_singleton_rows(A_ub, b_ub, A_eq, b_eq, bounds)
    assert out is not None
    A_ub_f, b_ub_f, A_eq_f, b_eq_f, bounds_f, fold = out
    np.testing.assert_array_equal(A_ub_f, A_ub[[2]])
    np.testing.assert_array_equal(b_ub_f, b_ub[[2]])
    np.testing.assert_array_equal(A_eq_f, A_eq[[1]])
    (l0, u0), (l1, u1), (l2, u2) = bounds_f
    assert (l0, u0) == (-5.0, 1.5)
    assert (l1, u1) == (-0.25, 5.0)
    # An equality row PINS its column: lb == ub, never a box a few ulps wide.
    assert l2 == u2 == 0.5

    # Folded-problem multipliers: x0 at its upper (rc < 0), x1 at its lower (rc > 0),
    # x2 fixed with rc > 0, kept rows' duals 0.1 / -0.2.
    rc_f = np.array([-4.0, 1.0, 8.0])
    y, rc = _unfold_singleton_duals(fold, np.array([-0.1, -0.2]), rc_f)
    np.testing.assert_array_equal(rc, 0.0)
    np.testing.assert_allclose(y, [-2.0, -1.0, -0.1, 0.0, 2.0, -0.2])
    # The same stationarity on the original rows: A' y equals what rc held.
    A = np.vstack([A_ub, A_eq])
    kept = np.zeros(6)
    kept[[2, 5]] = [-0.1, -0.2]
    np.testing.assert_allclose(A.T @ (y - kept), rc_f)
    assert np.all(y[:4] <= 0.0)  # <= rows keep the HiGHS sign


def test_no_fold_without_singleton_rows_or_with_an_empty_box():
    A = np.array([[1.0, 1.0]])
    assert _fold_singleton_rows(A, np.array([1.0]), None, None, [(0, 1), (0, 1)]) is None
    # x0 <= -1 against a declared lb of 0: left for the backend to call infeasible.
    A = np.array([[1.0, 0.0]])
    assert _fold_singleton_rows(A, np.array([-1.0]), None, None, [(0, 1), (0, 1)]) is None


def _random_row_bounded_qp(seed: int, as_rows: bool):
    """A convex QP whose box is written as (scaled) single-variable rows.

    15% of the columns are pinned (``lo == hi``), half of those by one equality
    row; 30% of the rows carry a non-unit scale, so ``b/a`` is inexact.
    """
    rng = np.random.default_rng(seed)
    n = int(rng.integers(10, 120))
    B = rng.normal(size=(max(1, n // 3), n))
    Q = B.T @ B + 1e-3 * np.eye(n)
    c = rng.normal(size=n) * 5
    lo, hi = -rng.uniform(0, 2, n), rng.uniform(0, 2, n)
    pin = rng.random(n) < 0.15
    lo[pin] = hi[pin] = rng.normal(size=pin.sum()) * 0.1 * (rng.random(pin.sum()) < 0.5)
    scale = np.where(rng.random(n) < 0.3, rng.uniform(0.2, 7.0, n), 1.0)
    G = rng.normal(size=(n // 4, n))
    h = np.abs(rng.normal(size=n // 4)) + 0.5
    by_eq = rng.random(n) < 0.5
    m = dm.Model("row_bounded_qp")
    if as_rows:
        x = m.continuous("x", shape=(n,), lb=-np.inf, ub=np.inf)
        for j in range(n):
            if pin[j] and by_eq[j]:
                m.subject_to(scale[j] * x[j] == scale[j] * float(lo[j]))
            else:
                m.subject_to(scale[j] * x[j] <= scale[j] * float(hi[j]))
                m.subject_to(-scale[j] * x[j] <= -scale[j] * float(lo[j]))
    else:
        x = m.continuous("x", shape=(n,), lb=lo, ub=hi)
    m.minimize(0.5 * x @ Q @ x + c @ x)
    for i in range(G.shape[0]):
        m.subject_to(dm.sum(G[i] * x) <= float(h[i]))
    return m


@pytest.mark.parametrize("seed", [1, 7, 19, 22])
def test_scaled_and_pinned_row_bounds_certify(seed):
    """These four failed (3 ``error`` from the #145 KKT guard, 1 uncertified) when
    the fold widened each quotient by an ulp: a pinned column became a box two
    ulps wide. The row form must certify the bound form's optimum."""
    ref = _random_row_bounded_qp(seed, as_rows=False).solve(time_limit=60)
    r = _random_row_bounded_qp(seed, as_rows=True).solve(time_limit=60)
    assert ref.status == "optimal" and ref.gap_certified
    assert r.status == "optimal" and r.gap_certified, (r.status, r.error)
    tol = 1e-6 * (1 + abs(ref.objective))
    assert r.bound <= ref.objective + tol and ref.bound <= r.objective + tol


def test_a_bound_the_ipm_would_drop_stays_a_row():
    # x0 <= 1e16 as a column bound is >= the 1e15 threshold at which POUNCE is
    # handed its infinity (#1319); as a row it is kept.
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = _fold_singleton_rows(A, np.array([1e16, 2.0]), None, None, [(0, np.inf)] * 2)
    assert out is not None
    A_ub_f, b_ub_f, _, _, bounds_f, _ = out
    np.testing.assert_array_equal(A_ub_f, A[[0]])
    assert bounds_f == [(0.0, np.inf), (0.0, 2.0)]
