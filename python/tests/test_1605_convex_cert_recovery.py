"""#1605: the #1596 convex-route certificate, recovered without giving up soundness.

#1596 replaced ``bound := f(x)`` on the convex fast path with a dual bound that
charges the backend's unconverged complementarity. Measured over the MINLPTests
convex panel under the invariance transforms (rows x1e-4, rows x1e4, shift 1e3),
that sound bound withheld 69 of 364 certificates. These tests pin the mechanisms
that recover them, each of which leaves the bound valid:

* the tangent bound is taken over the declared box intersected with the FBBT box
  (a valid bound over any box containing the feasible set), so a rounding-sized
  residual slope is no longer multiplied by the 9.999e19 default box;
* a Newton support point is adopted as the incumbent only when it is as feasible
  as the backend's point (it heads for the Lagrangian's minimizer, which can sit
  outside the user's rows);
* a multiplier candidate with the inactive rows zeroed (``_active_row_multipliers``);
* the final bound is rounded outward by its own arithmetic and by the evaluation
  of ``f``, ``c`` and the slope at the support point;
* the QP route recomputes its reduced costs at an adopted point, reports ``gap``
  ``None`` at a ~0 objective, and shape errors in the certificate are raised, not
  read as "no certificate".

Every test runs with ``DISCOPT_RECENTRE=0`` (as #1596's do), so the fix holds
without recentring.
"""

from __future__ import annotations

import math
from decimal import Decimal

import numpy as np
import pytest
from _invariance import rescale_rows, translate
from test_minlptests import MINLPTESTS_CVX_BY_ID

# Exact optima (mpmath, 40 digits). The MINLPTests reference for 102_010 is
# -0.974165743715913, 5e-9 BELOW the true optimum, so a correct bound may sit above
# it; soundness is judged against the exact value.
_EXACT = {
    "nlp_cvx_102_010": Decimal("-0.97416573867739414"),
    "nlp_cvx_103_013": Decimal("-0.25"),
    "nlp_cvx_103_014": Decimal("-0.70710678118654752440"),
    "nlp_cvx_104_010": Decimal("-0.70710678118654752440"),
    "nlp_cvx_002_011": Decimal("0"),
    # x + exp(x - 2) - 1/2 at the left root of exp(x - 2) = log(x) + 1.
    "nlp_cvx_105_011": Decimal("0.16878273090316615765"),
}

_VARIANTS = {
    "rows1e-4": lambda m: rescale_rows(m, 1e-4),
    "rows1e4": lambda m: rescale_rows(m, 1e4),
    "shift1e3": lambda m: translate(m, 1e3),
}


@pytest.fixture(autouse=True)
def _no_recentre(monkeypatch):
    monkeypatch.setenv("DISCOPT_RECENTRE", "0")


def _solve(pid: str, variant: str):
    m = _VARIANTS[variant](MINLPTESTS_CVX_BY_ID[pid].build_fn())
    return m.solve(time_limit=30)


def _assert_certified_and_sound(r, exact: Decimal) -> None:
    assert r.status == "optimal" and r.gap_certified, (r.status, r.objective, r.bound)
    assert r.bound is not None
    # Compared exactly: a certified bound may never sit above the true optimum,
    # not even by an ulp.
    assert Decimal(r.bound) <= exact, (r.bound, exact)
    # The incumbent is feasible, so it can never beat the optimum.
    assert Decimal(r.objective) >= exact - Decimal("1e-9"), (r.objective, exact)
    assert r.objective - r.bound <= 1e-4 * (1.0 + abs(r.objective))


@pytest.mark.parametrize(
    "pid,variant",
    [
        ("nlp_cvx_102_010", "rows1e4"),
        ("nlp_cvx_102_010", "rows1e-4"),
        ("nlp_cvx_103_014", "rows1e4"),
        ("nlp_cvx_104_010", "rows1e4"),
        ("nlp_cvx_104_010", "shift1e3"),
    ],
)
def test_fbbt_box_recovers_free_variable_certificates(pid, variant):
    """Free variables (default box 9.999e19), feasible set bounded by its rows:
    withheld by the #1596 bound over the declared box, certified over the FBBT box."""
    _assert_certified_and_sound(_solve(pid, variant), _EXACT[pid])


def test_support_point_incumbent_must_be_as_feasible_as_the_backend_point():
    """rows x1e-4: a Newton support point violating a row by the old 1e-9 floor
    (1e-5 in the user's units) was adopted as a super-optimal incumbent, so the
    valid bound exceeded it and the premise check withheld the certificate."""
    _assert_certified_and_sound(_solve("nlp_cvx_103_013", "rows1e-4"), _EXACT["nlp_cvx_103_013"])


def test_active_row_multipliers_certify_an_unbounded_feasible_set():
    """002_011 shifted by 1e3: all six rows inactive, feasible set unbounded (FBBT
    cannot help), and a ~1e-8 multiplier on a row with slack ~1e3 costs 6e-5."""
    r = _solve("nlp_cvx_002_011", "shift1e3")
    _assert_certified_and_sound(r, _EXACT["nlp_cvx_002_011"])
    # #1605: the QP route's near-zero contract -- a relative gap is undefined at a
    # ~0 objective, so ``None`` (as on the NLP route), not ``0.0``.
    assert abs(r.objective) <= 1e-10
    assert r.gap is None


def test_active_row_multipliers_unit():
    from discopt.solver import _active_row_multipliers

    lam = np.array([1e-8, -2.0, 3.0, 0.0])
    cons = np.array([-1e3, 1.0, 5.0 - 1e-9, 7.0])
    cl = np.array([-np.inf, 1.0, -np.inf, 0.0])
    cu = np.array([0.0, np.inf, 5.0, 10.0])
    out = _active_row_multipliers(lam, cons, cl, cu)
    assert out is not None
    np.testing.assert_array_equal(out, [0.0, -2.0, 3.0, 0.0])
    # Nothing inactive -> None; no rows -> None.
    assert _active_row_multipliers(out, cons, cl, cu) is None
    assert _active_row_multipliers(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)) is None


@pytest.mark.parametrize("variant", ["rows1e4", "shift1e3"])
def test_bound_is_rounded_outward(variant):
    """105_011 is tight to rounding: with rows x1e4 the unrounded bound sat 3 ulp
    above the exact optimum, and with a 1e3 shift (``exp(x - 1002)`` at x ~ 1e3)
    the evaluation error put it 3e-14 above."""
    _assert_certified_and_sound(_solve("nlp_cvx_105_011", variant), _EXACT["nlp_cvx_105_011"])


def test_outward_tangent_bound_charges_evaluation_magnitude():
    from discopt.solver import _outward_tangent_bound, _tangent_box_bound

    g = np.array([1.0, -2.0])
    s = np.array([1e3, -1e3])
    lo, hi = np.array([0.0, -2e3]), np.array([2e3, 0.0])
    plain = _tangent_box_bound(0.5, g, s, lo, hi)
    own = _outward_tangent_bound(0.5, 0.5, 2, g, s, lo, hi)
    charged = _outward_tangent_bound(0.5, 0.5, 2, g, s, lo, hi, eval_mag=3e3)
    assert charged < own < plain
    eps = np.finfo(np.float64).eps
    k = 2 + 2 + 4
    assert own - charged == pytest.approx(2 * k * eps * 3e3, rel=1e-6)


def test_qp_reduced_costs_at_follow_the_highs_convention():
    """``Qx + c - A^T y`` with ``y`` laid out as the ``A_ub`` rows, then ``A_eq``."""
    import scipy.sparse as sps
    from discopt.solver import _qp_reduced_costs_at

    Q = np.array([[2.0, 0.0], [0.0, 4.0]])  # noqa: N806
    c = np.array([-1.0, 1.0])
    A_ub = sps.csr_matrix([[1.0, 1.0]])  # noqa: N806
    A_eq = sps.csr_matrix([[1.0, -1.0]])  # noqa: N806
    x = np.array([0.5, 0.25])
    y = np.array([0.3, -0.7])
    rc = _qp_reduced_costs_at(x, Q, c, A_ub, A_eq, y)
    expected = Q @ x + c - np.array([1.0, 1.0]) * 0.3 - np.array([1.0, -1.0]) * -0.7
    np.testing.assert_allclose(rc, expected, rtol=0, atol=1e-15)
    # Row duals of the wrong length are not reinterpreted.
    assert _qp_reduced_costs_at(x, Q, c, A_ub, A_eq, np.array([0.3])) is None
    assert _qp_reduced_costs_at(x, Q, c, A_ub, A_eq, None) is None


def test_qp_route_refreshes_reduced_costs_after_adopting_a_better_point(monkeypatch):
    """When the certificate adopts a better point than the backend's, the published
    bound duals must describe that point, not the one the backend returned."""
    import discopt.solver as S

    seen = {}
    orig = S._qp_reduced_costs_at

    def spy(x, *a, **k):
        seen["x"] = np.asarray(x, dtype=np.float64).copy()
        return orig(x, *a, **k)

    orig_cert = S._qp_convex_certificate

    def adopting(*a, **k):
        cert = orig_cert(*a, **k)
        if cert is None or cert.bound is None:
            return cert
        # Hand back a different point as the "better" one: what is checked is
        # only that the reduced costs are recomputed there.
        seen["adopted"] = True
        return cert._replace(
            better_x=np.array([0.75]),
            better_obj=0.75**2 - 0.75,
            bound=min(cert.bound, 0.75**2 - 0.75),
        )

    monkeypatch.setattr(S, "_qp_reduced_costs_at", spy)
    monkeypatch.setattr(S, "_qp_convex_certificate", adopting)

    import discopt.modeling as dm

    m = dm.Model("qp_swap")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize(x**2 - x)
    r = m.solve(time_limit=30)
    assert seen.get("adopted"), f"model did not take the QP route (status={r.status})"
    np.testing.assert_array_equal(seen["x"], [0.75])
    np.testing.assert_allclose(r.x["x"], 0.75)


def test_certificate_shape_errors_are_raised(monkeypatch):
    """A shape mismatch inside the certificate is a bug, not "no certificate": it
    must reach the caller instead of being swallowed by the route's fallback."""
    import discopt.modeling as dm
    import discopt.solver as S

    calls = {"n": 0}

    def wrong_size(model, n):
        calls["n"] += 1
        return np.zeros(n + 1), np.ones(n + 1)

    monkeypatch.setattr(S, "_fbbt_implied_box", wrong_size)
    m = dm.Model("shape")
    x = m.continuous("x", lb=0.5, ub=3.0)
    y = m.continuous("y", lb=0.5, ub=3.0)
    m.minimize((x - 1) ** 4 + math.e * y)
    m.subject_to(x * x + y * y <= 4)
    with pytest.raises(S._CertificateShapeError):
        m.solve(time_limit=30)
    assert calls["n"] >= 1


def test_refreshed_reduced_costs_price_only_active_bounds():
    """A refreshed reduced cost on a bound inactive at ``x`` is stationarity residue,
    not a price: it is zeroed, and kept on the active side its sign selects."""
    from discopt.solver import _qp_reduced_costs_at

    Q = np.zeros((3, 3))  # noqa: N806
    c = np.array([2.5e-9, 1.0, -1.0])
    x = np.array([3.0, 0.0, 5.0])
    bounds = [(-9.999e19, 9.999e19), (0.0, 10.0), (0.0, 5.0)]
    rc = _qp_reduced_costs_at(x, Q, c, None, None, None, bounds)
    # Column 0 is free: its 2.5e-9 residue would price a 9.999e19 default bound.
    np.testing.assert_array_equal(rc, [0.0, 1.0, -1.0])
    # Signs that point at an INACTIVE side are zeroed too.
    rc2 = _qp_reduced_costs_at(x, Q, -c, None, None, None, bounds)
    np.testing.assert_array_equal(rc2, [0.0, 0.0, 0.0])


def test_estimated_qp_multipliers_are_sign_consistent():
    """With no backend duals the QP route estimates them; only active rows get a
    multiplier, with the sign of the side that is active (equality rows free)."""
    from discopt.solver import _qp_estimated_multipliers

    # min (x-1)^2 + (y-2)^2  s.t.  x + y <= 2 (active),  x - y <= 5 (inactive),
    # x - y = -1 (active equality). Optimum (0.5, 1.5).
    A = np.array([[1.0, 1.0], [1.0, -1.0], [1.0, -1.0]])  # noqa: N806
    cl = np.array([-np.inf, -np.inf, -1.0])
    cu = np.array([2.0, 5.0, -1.0])
    x = np.array([0.5, 1.5])
    grad = 2.0 * (x - np.array([1.0, 2.0]))
    lb = np.array([0.0, 0.0])
    ub = np.array([10.0, 10.0])
    lam = _qp_estimated_multipliers(A, cl, cu, grad, x, lb, ub)
    assert lam[0] >= 0.0 and lam[1] == 0.0
    np.testing.assert_allclose(grad + A.T @ lam, 0.0, atol=1e-10)
    # No active row: zero multipliers (sign-consistent, so still a valid bound).
    lam0 = _qp_estimated_multipliers(A[1:2], cl[1:2], cu[1:2], grad, x, lb, ub)
    np.testing.assert_array_equal(lam0, [0.0])


def test_qp_route_certifies_without_backend_duals():
    """A backend that returns a converged optimum but no row duals still gets a
    certified answer: the multipliers are estimated, and the bound is the rigorous
    tangent bound under them, at most the exact optimum 0.5."""
    import time

    import discopt.modeling as dm
    import discopt.solver as S
    from discopt.solvers import QPResult, SolveStatus

    m = dm.Model("no_duals")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize((x - 1) ** 2 + (y - 2) ** 2)
    m.subject_to(x + y <= 2)

    def engine(*args, **kwargs):
        return QPResult(
            status=SolveStatus.OPTIMAL, x=np.array([0.5, 1.5]), objective=0.5, kkt_error=1e-9
        )

    out = S._solve_qp_matrix(m, time.perf_counter(), None, engine, "fake")
    assert out is not None and out.status == "optimal" and out.gap_certified
    assert out.bound <= 0.5
    assert out.bound >= 0.5 - 1e-9
