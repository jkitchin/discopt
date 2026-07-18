"""Tests for the Tier-1 feasibility certificate emitter + reference checker.

The reference checker (:mod:`discopt.certificate.refcheck`) is the executable
specification of the Lean ``checkFeasible`` decision procedure; these tests pin
its accept/reject behaviour so the Lean port has a fixed oracle to match. They
also guard the emitter's "refuse loudly" contract for unsupported nodes.
"""

from __future__ import annotations

import json

import discopt.modeling as dm
import pytest
from discopt.certificate import (
    CertificateError,
    build_convex_certificate,
    build_feasibility_certificate,
    check_certificate,
    write_certificate,
)


def _convex_qp(ub_x=10):
    """min (x-2)^2 + (y-1)^2 s.t. x+y<=2 -- convex QP with an active constraint."""
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=ub_x)
    y = m.continuous("y", lb=0, ub=10)
    m.subject_to(x + y <= 2, name="c1")
    m.minimize((x - 2) ** 2 + (y - 1) ** 2)
    return m, m.solve()


def _nlp():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.subject_to(x + y <= 5, name="c1")
    m.subject_to(x * y >= 3, name="c2")
    m.minimize((x - 2) ** 2 + (y - 1) ** 2)
    return m, m.solve()


def _milp():
    m = dm.Model()
    a = m.integer("a", lb=0, ub=10)
    b = m.integer("b", lb=0, ub=10)
    m.subject_to(a + b <= 7, name="cap")
    m.subject_to(2 * a + b <= 10, name="res")
    m.maximize(3 * a + 2 * b)
    return m, m.solve()


@pytest.mark.smoke
def test_nlp_certificate_accepts_valid():
    m, r = _nlp()
    cert = build_feasibility_certificate(m, r)
    ok, reason = check_certificate(cert)
    assert ok, reason
    # Schema shape.
    body = cert["certificate"]
    assert body["tier"] == "feasibility"
    assert body["model"]["n_columns"] == 2
    assert len(body["incumbent"]["x"]) == 2
    assert all(isinstance(v, list) and len(v) == 2 for v in body["incumbent"]["x"])


@pytest.mark.smoke
def test_milp_certificate_accepts_valid():
    m, r = _milp()
    cert = build_feasibility_certificate(m, r)
    ok, reason = check_certificate(cert)
    assert ok, reason
    assert all(c["type"] == "integer" for c in cert["certificate"]["model"]["columns"])


@pytest.mark.smoke
def test_rejects_inflated_objective():
    m, r = _nlp()
    cert = build_feasibility_certificate(m, r)
    cert["certificate"]["incumbent"]["objectiveValue"] = [10**6, 1]
    ok, reason = check_certificate(cert)
    assert not ok and "objective" in reason.lower()


@pytest.mark.smoke
def test_rejects_infeasible_point():
    m, r = _nlp()
    cert = build_feasibility_certificate(m, r)
    # x=y=0 violates the bilinear constraint x*y >= 3.
    cert["certificate"]["incumbent"]["x"][0] = [0, 1]
    cert["certificate"]["incumbent"]["x"][1] = [0, 1]
    ok, reason = check_certificate(cert)
    assert not ok


@pytest.mark.smoke
def test_rejects_out_of_bounds_point():
    m, r = _nlp()
    cert = build_feasibility_certificate(m, r)
    num, den = cert["certificate"]["incumbent"]["x"][0]
    cert["certificate"]["incumbent"]["x"][0] = [num + 100 * den, den]  # x + 100, past ub=4
    ok, reason = check_certificate(cert)
    assert not ok


@pytest.mark.smoke
def test_rejects_non_integral_on_integer_column():
    m, r = _milp()
    cert = build_feasibility_certificate(m, r)
    cert["certificate"]["incumbent"]["x"][0] = [1, 2]  # 0.5 on an integer column
    ok, reason = check_certificate(cert)
    assert not ok and "integral" in reason.lower()


@pytest.mark.smoke
def test_json_round_trip_is_stable():
    m, r = _nlp()
    cert = build_feasibility_certificate(m, r)
    reloaded = json.loads(json.dumps(cert))
    assert check_certificate(reloaded)[0]
    assert reloaded == cert  # pure JSON scalars, no numpy leakage


@pytest.mark.smoke
def test_write_certificate(tmp_path):
    m, r = _nlp()
    cert = build_feasibility_certificate(m, r)
    path = tmp_path / "cert.json"
    write_certificate(cert, path)
    assert check_certificate(json.loads(path.read_text()))[0]


@pytest.mark.smoke
def test_emitter_refuses_transcendental_for_exact_check():
    # exp() is emitted (for schema completeness) but the exact-rational checker
    # must refuse a certificate whose checked expressions contain it.
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=2)
    m.subject_to(dm.exp(x) <= 5, name="c")
    m.minimize(x)
    r = m.solve()
    cert = build_feasibility_certificate(m, r)
    # The exp node is present in the encoding...
    assert '"fn"' in json.dumps(cert)
    # ...and the exact checker refuses rather than guesses.
    ok, reason = check_certificate(cert)
    assert not ok and "transcendental" in reason.lower()


@pytest.mark.smoke
def test_emitter_refuses_without_incumbent():
    m, _ = _nlp()
    from discopt.modeling.core import SolveResult

    empty = SolveResult(status="infeasible")
    with pytest.raises(CertificateError):
        build_feasibility_certificate(m, empty)


# ── Tier 2: convex / KKT global optimality ───────────────────────────────────
@pytest.mark.smoke
def test_convex_certificate_accepts_constraint_active():
    m, r = _convex_qp()
    cert = build_convex_certificate(m, r)
    body = cert["certificate"]
    assert body["tier"] == "convex"
    assert "dualBound" in body and "kkt" in body
    ok, reason = check_certificate(cert)
    assert ok, reason
    assert "global optimum" in reason


@pytest.mark.smoke
def test_convex_certificate_accepts_bound_active():
    # ub=1 forces the upper bound active at the optimum (x*=1): pins bound-dual sign.
    m, r = _convex_qp(ub_x=1)
    cert = build_convex_certificate(m, r)
    assert check_certificate(cert)[0]


@pytest.mark.smoke
def test_convex_rejects_tampered_multiplier():
    m, r = _convex_qp()
    cert = build_convex_certificate(m, r)
    cert["certificate"]["kkt"]["constraint_multipliers"][0] = [5, 1]  # breaks stationarity
    ok, reason = check_certificate(cert)
    assert not ok and "stationarity" in reason.lower()


@pytest.mark.smoke
def test_convex_rejects_open_gap():
    m, r = _convex_qp()
    cert = build_convex_certificate(m, r)
    cert["certificate"]["dualBound"] = [0, 1]  # claim a bound below the optimum
    ok, reason = check_certificate(cert)
    assert not ok and "gap" in reason.lower()


@pytest.mark.smoke
def test_convex_checker_rejects_nonconvex_objective():
    # Swap the objective for a NON-convex bilinear body x*y (indefinite Hessian),
    # keeping the reported value equal to its value at x* so feasibility still
    # passes -- so the ONLY thing that can reject is the convexity test.
    from fractions import Fraction

    m, r = _convex_qp()
    cert = build_convex_certificate(m, r)
    body = cert["certificate"]
    xv = [Fraction(n, d) for n, d in body["incumbent"]["x"]]
    body["model"]["objective"]["body"] = {
        "k": "mul",
        "l": {"k": "var", "i": 0},
        "r": {"k": "var", "i": 1},
    }
    val = xv[0] * xv[1]
    body["incumbent"]["objectiveValue"] = [val.numerator, val.denominator]
    body["dualBound"] = [val.numerator, val.denominator]
    ok, reason = check_certificate(cert)
    assert not ok and "convex" in reason.lower()


@pytest.mark.smoke
def test_convex_checker_rejects_negated_dual_feasibility():
    m, r = _convex_qp()
    cert = build_convex_certificate(m, r)
    # A negative inequality multiplier is dual-infeasible (and breaks stationarity).
    num, den = cert["certificate"]["kkt"]["constraint_multipliers"][0]
    cert["certificate"]["kkt"]["constraint_multipliers"][0] = [-num, den]
    assert not check_certificate(cert)[0]


@pytest.mark.smoke
def test_convex_emitter_refuses_maximize():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1)
    m.maximize(-((x - 0.5) ** 2))  # concave max == convex min, but emitter is min-only
    r = m.solve()
    with pytest.raises(CertificateError):
        build_convex_certificate(m, r)


@pytest.mark.smoke
def test_convex_certificate_json_round_trip():
    m, r = _convex_qp()
    cert = build_convex_certificate(m, r)
    reloaded = json.loads(json.dumps(cert))
    assert check_certificate(reloaded)[0]
    assert reloaded == cert
