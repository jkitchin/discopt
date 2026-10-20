"""#1609: builder objectives on the pounce routes, and two recentring defects.

* ``add_linear_objective`` / ``add_quadratic_objective`` leave a zero placeholder
  in ``Model._objective``. The tape and JAX evaluators lowered that placeholder,
  so the #1537 E incumbent repair re-evaluated every point at 0, called a valid
  bound "refuted" and withdrew the certificate (``feasible 0.0``), and a singular
  PSD bulk QP routed to ``pounce:nlp`` minimised 0 and returned an arbitrary
  feasible point as ``local_optimal``.
* Recentring (default-on since #1594) moved boxes that contain 0, anchoring
  ``j in [-1e3, 1e3]`` at -1e3; the optimum j* = 2.5e-4 then sat at
  z* = 1000.00025 and came back 19 % off, labelled ``optimal``.
* Recentring crashed on a negative index (``T[-1]``).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt.modeling._recentre import plan_shifts, recentre


def _certified_optimal(r, opt, tol=1e-6):
    assert r.status == "optimal", (r.status, r.objective)
    assert r.gap_certified
    assert r.objective == pytest.approx(opt, abs=tol, rel=1e-7)
    assert r.bound <= r.objective + tol


# ── builder objectives on solver="pounce" ─────────────────────────────────────


def test_linear_builder_objective_on_pounce():
    m = dm.Model("a13n")
    x = m.continuous("x", shape=(2,), lb=0, ub=10)
    m.subject_to(x[0] + x[1] == 4)
    m.add_linear_objective(np.array([2.0, 3.0]), x)
    r = m.solve(solver="pounce")
    _certified_optimal(r, 8.0)


def test_linear_builder_objective_with_bulk_rows_on_pounce():
    m = dm.Model("a13n2")
    x = m.continuous("x", shape=(2,), lb=0, ub=10)
    m.add_linear_constraints(np.array([[1.0, 1.0]]), x, "==", np.array([1.0]))
    m.add_linear_objective(np.array([-1.0, -2.0]), x)
    r = m.solve(solver="pounce")
    _certified_optimal(r, -2.0)
    np.testing.assert_allclose(r.value(x), [0.0, 1.0], atol=1e-6)


def test_quadratic_builder_objective_on_pounce():
    c = np.array([1.0, 2.0, 3.0, 4.0])
    m = dm.Model("b19q")
    z = m.continuous("z", shape=(4,), lb=0, ub=20)
    m.add_linear_constraints(np.ones((1, 4)), z, "==", np.array([10.0]))
    m.add_quadratic_objective(np.eye(4), c, z)
    r = m.solve(solver="pounce")
    zv = np.asarray(r.value(z))
    _certified_optimal(r, 35.0)
    assert r.objective == pytest.approx(c @ zv + 0.5 * zv @ zv, abs=1e-6)


def test_singular_psd_builder_qp_on_pounce_nlp_minimises_the_real_objective():
    n = 40
    rng = np.random.default_rng(0)
    cc = rng.uniform(1, 5, n)
    S = sp.diags([np.ones(n - 1), -np.ones(n - 1)], [0, 1], shape=(n - 1, n))
    P = (S.T @ S).tocsr()

    def build():
        m = dm.Model("b19s")
        z = m.continuous("z", shape=(n,), lb=0, ub=20)
        m.add_linear_constraints(sp.csr_matrix(np.ones((1, n))), z, "==", np.array([100.0]))
        m.add_linear_constraints(sp.eye(n).tocsr(), z, "<=", np.full(n, 10.0))
        m.add_quadratic_objective(P, cc, z)
        return m, z

    ref = build()[0].solve()
    assert ref.status == "optimal"
    m, z = build()
    r = m.solve(solver="pounce")
    x = np.asarray(r.value(z))
    true_obj = cc @ x + 0.5 * x @ (P @ x)
    assert r.objective == pytest.approx(true_obj, rel=1e-8, abs=1e-8)
    assert true_obj == pytest.approx(ref.objective, rel=1e-6)


@pytest.mark.parametrize("sym", [False, True])
def test_objective_expression_matches_the_builder_convention(sym):
    """``_objective_expression`` is the builder's ``0.5 x'Sx + c'x + k``. For a
    non-symmetric Q that is ``0.5 x'Qx`` (#1818: Q is stored as its symmetric
    part; it used to be ``S = triu(Q) + striu(Q)'``, a different objective)."""
    rng = np.random.default_rng(3)
    Q = rng.normal(size=(3, 3))
    if sym:
        Q = Q + Q.T
    c = rng.normal(size=3)
    m = dm.Model("conv")
    x = m.continuous("x", shape=(3,), lb=-5, ub=5)
    m.add_quadratic_objective(Q, c, x, constant=1.25)
    Sm = 0.5 * Q + 0.5 * Q.T
    from discopt._tape_nlp_evaluator import make_evaluator

    ev = make_evaluator(m)
    n_checked = 0
    for _ in range(5):
        p = rng.uniform(-5, 5, 3)
        want = 0.5 * p @ Sm @ p + c @ p + 1.25
        assert float(ev.evaluate_objective(p)) == pytest.approx(want, rel=1e-12, abs=1e-12)
        g = np.asarray(ev.evaluate_gradient(p), dtype=float)
        np.testing.assert_allclose(g, 0.5 * (Sm + Sm.T) @ p + c, rtol=1e-12, atol=1e-12)
        n_checked += 1
    assert n_checked == 5


def test_placeholder_without_a_builder_block_refuses():
    m = dm.Model("bad")
    x = m.continuous("x", shape=(2,), lb=0, ub=1)
    m.add_linear_objective(np.array([1.0, 1.0]), x)
    m._builder_linear_objective = None
    with pytest.raises(ValueError, match="placeholder"):
        m._objective_expression()


# ── recentring ────────────────────────────────────────────────────────────────


def test_box_containing_zero_is_not_recentred():
    m = dm.Model("rb")
    x = m.continuous("x", lb=0, ub=1)
    j = m.continuous("j", lb=-1e3, ub=1e3)
    m.subject_to(x - j == 0.4)
    m.minimize(0.5 * (64 / 0.01) * j * j + (x - 0.4) * (x - 0.4) - 1.6 * j)
    assert plan_shifts(m, 100.0) == {}
    r = m.solve(solver="pounce")
    assert r.status == "optimal"
    assert float(r.value(j)) == pytest.approx(1.6 / 6402, rel=1e-5)
    assert not (r.solver_stats or {}).get("recentre/variables_moved")


def test_negative_index_is_recentred():
    m = dm.Model("rc")
    T = m.continuous("T", shape=(3,), lb=298.0, ub=299.0)
    m.subject_to(T[0] == 298.5)
    m.minimize(T[-1] - 2.0 * T[-3])
    assert recentre(m) is not None
    r = m.solve()
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(298.0 - 2 * 298.5, abs=1e-6)
    np.testing.assert_allclose(r.value(T)[[0, 2]], [298.5, 298.0], atol=1e-6)
