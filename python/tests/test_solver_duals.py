"""End-to-end tests for solver-supplied duals on the LP / QP / B&B paths.

Each test runs a real solve and asserts that ``constraint_duals`` /
``bound_duals_lower`` / ``bound_duals_upper`` are populated with the
expected sign and magnitude, then optionally re-runs through the
Examiner to confirm the duals satisfy KKT.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest


def _scalar(name: str, vec: dict[str, np.ndarray]) -> float:
    return float(np.asarray(vec[name]).ravel()[0])


def _assert_cs_against_declared_box(model, res, tol: float = 1e-7) -> int:
    """Every reported bound multiplier must satisfy complementary slackness
    against the bounds the MODEL DECLARES (#1037).

    This is the general contract, not a per-instance expectation: a nonzero
    multiplier on a bound the point is not sitting on prices a constraint the
    user never wrote. Returns the number of products actually checked so the
    caller can assert the probe fired (CLAUDE.md §6).
    """
    if res.bound_duals_lower is None and res.bound_duals_upper is None:
        return 0
    checked = 0
    for v in model._variables:
        if v.var_type in (dm.VarType.BINARY, dm.VarType.INTEGER):
            continue  # integer bound duals price the fixing, not the box
        xv = np.asarray(res.x[v.name], dtype=float).ravel()
        for bnd, duals, tag in (
            (v.lb, res.bound_duals_lower, "lb"),
            (v.ub, res.bound_duals_upper, "ub"),
        ):
            if duals is None or bnd is None or v.name not in duals:
                continue
            lam = np.asarray(duals[v.name], dtype=float).ravel()
            slack = np.abs(xv - float(bnd))
            prod = np.abs(lam) * slack
            worst = int(np.argmax(prod))
            assert prod[worst] <= tol, (
                f"CS violated on {v.name}[{worst}].{tag}: declared bound={float(bnd):.6g}, "
                f"x={xv[worst]:.6g}, lambda={lam[worst]:.6e}, product={prod[worst]:.6e}"
            )
            checked += lam.size
    return checked


# ── LP fast path ────────────────────────────────────────────────────────


def test_lp_path_returns_constraint_and_bound_duals():
    """min x + 2y s.t. x + y >= 4, 0<=x,y<=10 → x*=4, y*=0."""
    m = dm.Model("lp")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize(x + 2 * y)
    m.subject_to(x + y >= 4, name="c1")

    res = m.solve()

    assert res.status == "optimal"
    assert res.constraint_duals is not None
    assert res.bound_duals_lower is not None
    assert res.bound_duals_upper is not None

    assert _scalar("x", res.x) == pytest.approx(4.0, abs=1e-6)
    assert _scalar("y", res.x) == pytest.approx(0.0, abs=1e-6)

    # ">=" row binding from below → μ ≥ 0 in the discopt convention.
    mu = _scalar("c1", res.constraint_duals)
    assert mu == pytest.approx(1.0, abs=1e-6)

    # Bound on y at lb=0 is active; reduced cost is 2 - mu = 1 (>= 0).
    lam_lb_y = _scalar("y", res.bound_duals_lower)
    assert lam_lb_y == pytest.approx(1.0, abs=1e-6)

    # Bound on x is inactive; multiplier ≈ 0.
    assert _scalar("x", res.bound_duals_lower) == pytest.approx(0.0, abs=1e-6)
    assert _scalar("x", res.bound_duals_upper) == pytest.approx(0.0, abs=1e-6)


def test_lp_equality_constraint_dual_sign():
    """Equality row dual is free; verify magnitude matches reduced LP."""
    m = dm.Model("lp_eq")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize(x + 3 * y)
    m.subject_to(x + y == 4, name="ceq")

    res = m.solve()
    assert res.status == "optimal"
    assert res.constraint_duals is not None
    # x*=4, y*=0; ∇f = [1, 3]; KKT (Examiner convention ∇f + ∇body·μ = 0)
    # gives 1 + 1·μ = 0 → μ = -1 (equality multipliers are free in sign).
    mu = _scalar("ceq", res.constraint_duals)
    assert mu == pytest.approx(-1.0, abs=1e-6)


# ── QP fast path ────────────────────────────────────────────────────────


def test_qp_path_returns_duals_at_active_constraint():
    """min (x-0.5)^2 + (y-0.5)^2 s.t. x+y >= 5, 0<=x,y<=10."""
    m = dm.Model("qp_active")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize((x - 0.5) ** 2 + (y - 0.5) ** 2)
    m.subject_to(x + y >= 5, name="c1")

    res = m.solve()
    assert res.status == "optimal"
    assert res.constraint_duals is not None
    assert res.bound_duals_lower is not None

    # Symmetric → x* = y* = 2.5; ∇f = [2(x-0.5), 2(y-0.5)] = [4, 4].
    # ">=" with μ ≥ 0 gives 4 - μ = 0 → μ = 4.
    mu = _scalar("c1", res.constraint_duals)
    assert mu == pytest.approx(4.0, abs=1e-4)


# ── MILP B&B path ───────────────────────────────────────────────────────


def test_milp_returns_relaxation_duals_at_incumbent():
    """min x + 2y, integer, s.t. x+y >= 5. Optimum x=5,y=0 with all-integer
    fix-and-resolve degenerating to zero free continuous columns; the
    recovery should still attach a dict (possibly all zeros) without
    raising."""
    m = dm.Model("milp")
    x = m.integer("x", lb=0, ub=10)
    y = m.integer("y", lb=0, ub=10)
    m.minimize(x + 2 * y)
    m.subject_to(x + y >= 5, name="c1")

    res = m.solve()
    assert res.status == "optimal"
    assert _scalar("x", res.x) == pytest.approx(5.0, abs=1e-6)
    assert _scalar("y", res.x) == pytest.approx(0.0, abs=1e-6)
    # Recovery returns dicts (even if all zero — fully integer fix gives
    # no free columns; what matters is the plumbing wires through).
    assert res.constraint_duals is not None
    assert "c1" in res.constraint_duals
    assert res.bound_duals_lower is not None
    assert "x" in res.bound_duals_lower
    assert "y" in res.bound_duals_lower


def test_miqp_returns_relaxation_duals_at_incumbent():
    """min (x-0.5)^2 + (y-2.0)^2 s.t. x+y >= 5, x cont, y int.

    The y-center is offset to 2.0 (not 0.5) so the optimum is *unique*
    (y=3, x=2): with the symmetric 0.5 center, y=2,x=3 and y=3,x=2 are both
    optimal at 8.5 and the recovered dual depends on which tied incumbent the
    search keeps (purification can pick either). With y fixed at the
    incumbent (=3), the QP relaxation is min (x-0.5)^2 s.t. x >= 2,
    x in [0,10] → x*=2, μ=3 on the ">=" row.
    """
    m = dm.Model("miqp")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.integer("y", lb=0, ub=10)
    m.minimize((x - 0.5) ** 2 + (y - 2.0) ** 2)
    m.subject_to(x + y >= 5, name="c1")

    res = m.solve()
    assert res.status == "optimal"
    assert res.constraint_duals is not None
    mu = _scalar("c1", res.constraint_duals)
    assert mu == pytest.approx(3.0, abs=1e-3)


# ── validate=True hook ──────────────────────────────────────────────────


def test_solve_with_validate_attaches_examiner_report():
    m = dm.Model("lp_validate")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize(x + 2 * y)
    m.subject_to(x + y >= 4, name="c1")

    res = m.solve(validate=True)
    assert res.validation_report is not None
    rep = res.validation_report
    assert rep.passed, rep.summary(verbose=True)
    # The solver-duals branch should have run, since the LP fast path
    # populated constraint_duals.
    assert rep.solver_duals_used


def test_solve_without_validate_leaves_report_none():
    m = dm.Model("lp_no_validate")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    m.subject_to(x >= 1, name="c1")
    res = m.solve()
    assert res.validation_report is None


def test_validate_true_on_milp_attaches_report():
    m = dm.Model("milp_validate")
    x = m.integer("x", lb=0, ub=10)
    y = m.integer("y", lb=0, ub=10)
    m.minimize(x + 2 * y)
    m.subject_to(x + y >= 5, name="c1")

    res = m.solve(validate=True)
    assert res.validation_report is not None
    # We don't assert .passed — pure-integer fix-and-resolve degenerates;
    # the contract is that the report exists and primal checks pass.
    rep = res.validation_report
    primal = [c for c in rep.checks if c.name.startswith("primal_")]
    assert primal, "expected at least one primal check"
    assert all(c.passed for c in primal), rep.summary(verbose=True)


# ── duals belong to the DECLARED model, not the presolved one (#1037) ────


def _circle_model(scale: float = 1.0, radius: float = 1.0, box: float = 2.0):
    """min -x  s.t.  s*x^2 + s*y^2 <= s*r^2,  x,y in [-box, box].

    Optimum x*=r, y*=0. Stationarity -1 + mu*2*s*r = 0 gives mu = 1/(2*s*r),
    independent of the box. When box > r, presolve derives x in [-r, r] and the
    derived bound is active at x*, making the split between the row multiplier
    and the bound multiplier degenerate on the TIGHTENED box — but uniquely
    determined on the declared one, where x* is strictly interior.

    The scale is DISTRIBUTED over the terms on purpose. Writing the row as
    ``s*(x**2 + y**2) <= s*r**2`` puts a multiply above the sum, FBBT cannot
    derive x in [-r, r] through it, no tightening happens and the defect does
    not arise — a model shaped that way is silently not a test of #1037.
    """
    m = dm.Model(f"circle_s{scale}_r{radius}_b{box}")
    x = m.continuous("x", lb=-box, ub=box)
    y = m.continuous("y", lb=-box, ub=box)
    m.minimize(-x)
    m.subject_to(scale * x**2 + scale * y**2 <= scale * radius**2, name="ball")
    return m


def test_derived_bound_does_not_steal_the_row_multiplier():
    """The #1037 reproduction verbatim: mu must be 0.5, not the degenerate
    split the backend returns against the presolve-tightened box."""
    m = _circle_model()
    res = m.solve()

    assert res.status == "optimal"
    assert _scalar("x", res.x) == pytest.approx(1.0, abs=1e-6)
    assert res.constraint_duals is not None, "duals were withheld, not refitted"

    assert _scalar("ball", res.constraint_duals) == pytest.approx(0.5, abs=1e-6)
    # x* = 1 is strictly interior to the DECLARED box [-2, 2], so CS forces the
    # upper-bound multiplier to zero. Before the fix this carried 5.92e-01; the
    # tolerance is the examiner's CS budget, not the refit's noise floor (~1e-9).
    assert _scalar("x", res.bound_duals_upper) == pytest.approx(0.0, abs=1e-7)
    assert _assert_cs_against_declared_box(m, res) > 0


@pytest.mark.parametrize("scale", [1e-3, 0.1, 1.0, 10.0, 1000.0, 1e5])
def test_declared_box_duals_hold_across_row_scaling(scale):
    """Same defect at every row scale — mu = 1/(2*scale). Fixing the class,
    not the instance: the split the backend picks is arbitrary, and pre-fix it
    varied with scale (mu/mu_true measured at 0.408 and 0.431 for scale 1 and
    10). Multiplying a row through by a constant changes nothing about the
    problem, so nothing about the duals may depend on it."""
    m = _circle_model(scale=scale)
    res = m.solve()

    assert res.status == "optimal"
    assert res.constraint_duals is not None, "duals were withheld, not refitted"
    expected = 1.0 / (2.0 * scale)
    mu = _scalar("ball", res.constraint_duals)
    assert mu == pytest.approx(expected, rel=1e-5)
    assert _assert_cs_against_declared_box(m, res) > 0


def test_genuinely_active_declared_bound_keeps_its_multiplier():
    """Control: the refit must not destroy a real bound multiplier. With
    r = 10 the ball is slack at x* = 2 and the DECLARED upper bound is what
    binds, so -1 + z_U = 0 → z_U = 1."""
    m = _circle_model(radius=10.0, box=2.0)
    res = m.solve()

    assert res.status == "optimal"
    assert _scalar("x", res.x) == pytest.approx(2.0, abs=1e-6)
    assert res.bound_duals_upper is not None
    assert _scalar("x", res.bound_duals_upper) == pytest.approx(1.0, abs=1e-5)
    assert _scalar("ball", res.constraint_duals) == pytest.approx(0.0, abs=1e-6)
    assert _assert_cs_against_declared_box(m, res) > 0


def test_unbounded_variable_carries_no_bound_multiplier():
    """The second cause the issue did not name: the +/-1e20 sentinel standing
    in for "no bound at all". A variable with no declared bound has no bound to
    price, so any residue the backend leaves there is a CS violation of ~1e15.

    A guard, not a regression test — this small model gets clean zeros on the
    pre-fix code too. The residue is backend- and instance-dependent (measured
    lam_ub = 3.84e-05 on nlp_cvx_102_010); the regression coverage for it is
    test_minlptests.py, where that bucket went from failing to passing.
    """
    m = dm.Model("free_vars")
    x = m.continuous("x")
    y = m.continuous("y")
    m.minimize((x - 3.0) ** 2 + (y + 1.0) ** 2)
    m.subject_to(x + y >= 1.0, name="c1")

    res = m.solve()
    assert res.status == "optimal"
    assert _scalar("x", res.x) == pytest.approx(3.0, abs=1e-5)
    assert _scalar("y", res.x) == pytest.approx(-1.0, abs=1e-5)
    assert _assert_cs_against_declared_box(m, res) > 0


def test_examiner_accepts_the_reported_duals_on_the_declared_box():
    """End-to-end: the examiner's primal_cs check on the solver's own duals is
    what exposed #1037. It must now pass on the reproduction."""
    m = _circle_model()
    res = m.solve(validate=True)

    rep = res.validation_report
    assert rep is not None
    assert rep.solver_duals_used, "the refit must feed the solver-duals branch"
    assert rep.passed, rep.summary(verbose=True)
