"""#1380 — every feasibility arbiter must test the INTEGRAL realisation of a point.

The defect
----------

``min -x + 3z`` subject to ``x <= M z``, ``x in [0, 10]``, ``z`` binary has exactly
two cases: ``z=0`` forces ``x=0`` (objective 0) and ``z=1`` allows ``x=10``
(objective -7). The true optimum is **-7.0**.

At ``M = 1e7`` three of discopt's four routes certified ``optimal`` at
**-9.999997**, at a point with ``z = 1e-6``. Both of the tests that should have
caught it passed, because they were run *independently* on the point as computed:

* integrality:  ``|1e-6 - 0| = 1e-6 < INT_TOL = 1e-5``  ✓
* the row:      ``10 <= 1e7 * 1e-6`` holds exactly       ✓

But the point being *claimed* is the integral one, and at ``z = 0`` that row is
violated by 10.0. An integer column inside ``INT_TOL`` buys up to
``|a_ij| * INT_TOL`` of slack on row ``i``; at ``|a_ij| = 1e7`` that is 100 units,
eight orders past ``abs = 1e-6``.

The contract these tests pin
----------------------------

Feasibility is decided at the integral realisation: integrality is established
first, then the rows and bounds are tested at the snapped point. The snap is a
bit-for-bit no-op on a genuinely integral point, so the only points it rejects are
those whose feasibility rests on fractionality the solver has already declared
absent.

Nothing here is keyed to this instance (CLAUDE.md §2): the model is a three-line
big-M formulation and the arbiters are exercised directly, on points constructed
in the test.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.validation.feasibility import (
    snap_integer_columns,
    snap_integers,
    verify_point,
)

#: The true optimum of :func:`_big_m_model`, by enumeration over the one binary.
TRUE_OPTIMUM = -7.0


def _big_m_model(M: float):
    """``min -x + 3z``  s.t. ``x <= M z``, ``x in [0,10]``, ``z`` binary."""
    m = dm.Model("bigm")
    x = m.continuous("x", lb=0.0, ub=10.0)
    z = m.binary("z")
    m.minimize(-x + 3.0 * z)
    m.subject_to(x <= M * z)
    return m


def _true_optimum_by_enumeration(M: float) -> float:
    """The oracle, computed outside discopt: enumerate the binary."""
    best = None
    for zv in (0, 1):
        # for fixed z the continuous part is  min -x  over  0 <= x <= min(10, M*z)
        x_hi = min(10.0, M * zv)
        val = -x_hi + 3.0 * zv
        best = val if best is None else min(best, val)
    return best


def test_the_oracle_is_minus_seven():
    """Guard the guard: the reference below is arithmetic, not a pinned constant."""
    assert _true_optimum_by_enumeration(1e7) == pytest.approx(TRUE_OPTIMUM)


# ── the verifier ───────────────────────────────────────────────────────────


def test_verify_point_refuses_a_point_its_integral_realisation_violates():
    """The #1380 entry experiment, as a regression.

    Before the fix this returned ``ok=True, objective=-9.999997`` — the single
    incumbent verifier vouching for a value the model cannot achieve.
    """
    m = _big_m_model(1e7)
    assert [v.name for v in m._variables] == ["x", "z"]

    res = verify_point(m, np.array([10.0, 1e-6]), with_objective=True)
    assert not res.ok, (
        f"verify_point vouched for x=10, z=1e-6 (objective {res.objective}); at the "
        "integral z=0 that row is violated by 10.0"
    )

    # ...and the integral point it is a claim about is itself refused, for the
    # same reason and with the same arithmetic.
    assert not verify_point(m, np.array([10.0, 0.0])).ok


def test_verify_point_still_accepts_the_genuinely_feasible_points():
    """The fix must not cost a single legitimate incumbent."""
    m = _big_m_model(1e7)
    for point, obj in (([10.0, 1.0], -7.0), ([0.0, 0.0], 0.0), ([4.0, 1.0], -1.0)):
        res = verify_point(m, np.array(point), with_objective=True)
        assert res.ok, f"{point} is feasible but was refused: {res.reason}"
        assert res.objective == pytest.approx(obj)


def test_snapping_is_a_no_op_on_an_exactly_integral_point():
    """Why the fix is free: ``round`` of an exact integer is that integer."""
    m = _big_m_model(1e7)
    exact = np.array([3.25, 1.0])
    np.testing.assert_array_equal(snap_integers(m, exact), exact)
    np.testing.assert_array_equal(snap_integer_columns(exact, [1]), exact)
    # and it moves an in-tolerance column exactly onto its integer
    np.testing.assert_array_equal(
        snap_integer_columns(np.array([3.25, 1e-6]), [1]), np.array([3.25, 0.0])
    )
    # ...while leaving the continuous columns alone
    assert snap_integer_columns(np.array([3.25, 1e-6]), [1])[0] == 3.25


def test_snap_helpers_tolerate_an_empty_integer_set():
    m = dm.Model("cont")
    m.continuous("x", lb=0, ub=1)
    np.testing.assert_array_equal(snap_integers(m, np.array([0.4])), np.array([0.4]))
    np.testing.assert_array_equal(snap_integer_columns(np.array([0.4]), []), np.array([0.4]))


# ── the HiGHS LP/MILP route's own arbiter ──────────────────────────────────


def test_highs_route_arbiter_refuses_the_fractional_binary():
    """``feasibility_problem`` is the gate on the default pure-MILP route."""
    pytest.importorskip("highspy")
    from discopt.solver import _highs_std_form
    from discopt.solvers.lp_milp_highs import feasibility_problem

    m = _big_m_model(1e7)
    _lp_data, _n_orig, sf = _highs_std_form(m)
    assert sf.int_idx.size == 1, "the model has exactly one integer column"

    def _with_slacks(xv, zv):
        """``StdForm`` is ``A x = b``; solve the added slack columns for the row."""
        x = np.zeros(sf.xl.shape[0], dtype=np.float64)
        x[0], x[1] = xv, zv
        A = sf.A.toarray() if hasattr(sf.A, "toarray") else np.asarray(sf.A)
        resid = np.asarray(sf.b, dtype=np.float64) - A[:, :2] @ np.array([xv, zv])
        for i in range(A.shape[0]):
            (cols,) = np.nonzero(A[i, 2:])
            assert cols.size == 1, f"row {i} has {cols.size} slack columns"
            x[2 + cols[0]] = resid[i] / A[i, 2 + cols[0]]
        np.testing.assert_allclose(A @ x, sf.b, atol=1e-9)
        return x

    why = feasibility_problem(_with_slacks(10.0, 1e-6), sf, check_integrality=True)
    assert why is not None, (
        "the route's arbiter accepted x=10, z=1e-6 — the point whose integral "
        "realisation violates its row by 10.0"
    )

    assert feasibility_problem(_with_slacks(10.0, 1.0), sf, check_integrality=True) is None, (
        "the genuinely feasible integral point must still pass"
    )


# ── end to end, on every route ─────────────────────────────────────────────


@pytest.mark.parametrize("backend", ["highs", "rust"])
@pytest.mark.parametrize("M", [1e3, 1e6, 1e7])
def test_no_route_reports_an_objective_below_the_true_optimum(monkeypatch, M, backend):
    """The invariant that actually matters: never a value the model cannot reach.

    A refusal is an acceptable outcome (CLAUDE.md §3 — at a big-M this large the
    model is not decidable at the declared tolerances, and saying so loudly beats
    certifying a number). Reporting -9.999997 is not.
    """
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", backend)
    oracle = _true_optimum_by_enumeration(M)
    try:
        r = _big_m_model(M).solve(time_limit=30)
    except RuntimeError as exc:
        assert "integral realisation" in str(exc) or "infeasible point" in str(exc), exc
        return

    if r.objective is not None:
        assert r.objective >= oracle - 1e-6, (
            f"M={M:g} backend={backend}: reported objective {r.objective} is below the "
            f"true optimum {oracle} — no feasible point of this model attains it"
        )
    if r.bound is not None and np.isfinite(r.bound):
        assert r.bound <= oracle + 1e-6, (
            f"M={M:g} backend={backend}: dual bound {r.bound} crosses the true optimum {oracle}"
        )
    if r.status == "optimal":
        assert r.objective == pytest.approx(oracle, abs=1e-5), (
            f"M={M:g} backend={backend}: certified optimal at {r.objective}, true optimum {oracle}"
        )
        zv = float(np.asarray(r.x["z"]))
        assert abs(zv - round(zv)) < 1e-9, (
            f"M={M:g} backend={backend}: certified optimal at a non-integral z={zv}"
        )


@pytest.mark.parametrize("M", [1e3, 1e6])
def test_a_solvable_big_m_still_solves(M):
    """The fix must not turn a decidable model into a refusal."""
    r = _big_m_model(M).solve(time_limit=30)
    assert r.status == "optimal", f"M={M:g} regressed to status={r.status}"
    assert r.objective == pytest.approx(TRUE_OPTIMUM, abs=1e-6)
    assert float(np.asarray(r.x["z"])) == pytest.approx(1.0, abs=1e-9)
