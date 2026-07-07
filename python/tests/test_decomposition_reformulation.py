"""Tests for the Decomposition Advisor reformulation IR (Phase 5).

Covers ``advisor.decompose()``, the ``DecomposedModel`` partition/certificate/
variable-mapping assembly, and ``solve()`` dispatch to the shipping drivers. The
solve *backend* needs the compiled extension, so dispatch is verified by
monkeypatching the driver functions — no solver is actually run here.
"""

import discopt.modeling as dm
import pytest
from discopt.decomposition import (
    DecomposedModel,
    MethodKind,
    SoundnessCertificate,
    analyze_decomposition,
    build_decomposition,
)
from discopt.decomposition.advisor import Candidate, Soundness
from discopt.decomposition.ir import reformulation as refmod

# ── fixtures ───────────────────────────────────────────────────


def _independent_blocks_model():
    m = dm.Model("indep")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    u = m.continuous("u", lb=0, ub=1)
    v = m.continuous("v", lb=0, ub=1)
    m.subject_to(x + y <= 1)
    m.subject_to(u + v <= 1)
    m.minimize(x + y + u + v)
    return m


def _coupled_model():
    m = dm.Model("coupled")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    u = m.continuous("u", lb=0, ub=1)
    v = m.continuous("v", lb=0, ub=1)
    m.subject_to(x + y <= 1)
    m.subject_to(u + v <= 1)
    m.subject_to(x + u <= 1)
    m.minimize(x + y + u + v)
    return m


def _benders_model():
    m = dm.Model("benders")
    z = m.binary("z")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.subject_to(x <= 5 * z)
    m.subject_to(y <= 5 * z)
    m.minimize(x + y - z)
    return m


# ── decompose() assembly ───────────────────────────────────────


def test_decompose_recommended_is_benders():
    dcmp = analyze_decomposition(_benders_model()).decompose()
    assert isinstance(dcmp, DecomposedModel)
    assert dcmp.method is MethodKind.BENDERS
    # master holds the complicating (integer) variable; recourse splits in two
    assert dcmp.master is not None
    assert "z" in dcmp.master.variables
    assert dcmp.num_blocks == 2
    assert dcmp.certificate.level is Soundness.PROVEN_EQUIVALENT


def test_decompose_variable_mapping_roles():
    dcmp = analyze_decomposition(_benders_model()).decompose()
    assert dcmp.var_map.role("z") == "master"
    assert dcmp.var_map.role("x").startswith("block")
    assert dcmp.var_map.role("y").startswith("block")
    # x and y are in different recourse blocks
    assert dcmp.var_map.role("x") != dcmp.var_map.role("y")


def test_decompose_independent_blocks():
    dcmp = analyze_decomposition(_independent_blocks_model()).decompose()
    assert dcmp.method is MethodKind.INDEPENDENT_BLOCKS
    assert dcmp.master is None
    assert dcmp.num_blocks == 2


def test_decompose_lagrangian_on_coupled():
    dcmp = analyze_decomposition(_coupled_model()).decompose()
    assert dcmp.method is MethodKind.LAGRANGIAN
    assert dcmp.certificate.level is Soundness.RELAXATION
    assert len(dcmp.master.coupling_constraints) >= 1


def test_decompose_explicit_method_override():
    # force NONE even though Benders is recommended
    dcmp = analyze_decomposition(_benders_model()).decompose(method="none")
    assert dcmp.method is MethodKind.NONE
    assert dcmp.num_blocks == 0


def test_decompose_unavailable_method_raises():
    with pytest.raises(ValueError):
        analyze_decomposition(_independent_blocks_model()).decompose(method="benders")


def test_decompose_summary_string():
    s = analyze_decomposition(_benders_model()).decompose().summary()
    assert "DecomposedModel" in s
    assert "subproblem" in s


# ── certificate ────────────────────────────────────────────────


def test_certificate_refuses_heuristic():
    cert = SoundnessCertificate(MethodKind.BENDERS, Soundness.HEURISTIC, "no guarantee")
    assert not cert.is_sound()
    with pytest.raises(ValueError):
        cert.assert_sound()


def test_certificate_relaxation_is_sound():
    cert = SoundnessCertificate(MethodKind.LAGRANGIAN, Soundness.RELAXATION, "dual bound")
    assert cert.is_sound()
    cert.assert_sound()  # does not raise


# ── #391 item 1: certificate post-condition + authoritative driver signal ──


class _FakeResult:
    """A driver result that *bypasses* ``SolveResult.__post_init__`` — the way a
    non-self-policing future driver could smuggle an unearned certificate past the
    permissive ``is_sound()`` gate."""

    def __init__(self, status, gap_certified, bound):
        self.status = status
        self.gap_certified = gap_certified
        self.bound = bound


def test_check_result_raises_on_unearned_certified_gap():
    # A RELAXATION certificate returning gap_certified=True with NO finite bound
    # is an unearned certificate: check_result must refuse it (#391 item 1).
    cert = SoundnessCertificate(MethodKind.LAGRANGIAN, Soundness.RELAXATION, "dual bound")
    with pytest.raises(AssertionError, match="without a finite dual bound"):
        cert.check_result(_FakeResult(status="optimal", gap_certified=True, bound=None))
    with pytest.raises(AssertionError):
        cert.check_result(_FakeResult(status="optimal", gap_certified=True, bound=float("-inf")))


def test_check_result_accepts_proven_certified_gap():
    # UNKNOWN certificate + driver-proved certified gap (finite bound) → allowed:
    # this is exactly GBD's UNKNOWN→exact resolution.
    cert = SoundnessCertificate(MethodKind.GENERALIZED_BENDERS, Soundness.UNKNOWN, "convex?")
    cert.check_result(_FakeResult(status="optimal", gap_certified=True, bound=3.14))  # no raise


def test_check_result_ignores_uncertified_and_infeasible():
    cert = SoundnessCertificate(MethodKind.LAGRANGIAN, Soundness.RELAXATION, "dual bound")
    # gap_certified=False → nothing to check.
    cert.check_result(_FakeResult(status="feasible", gap_certified=False, bound=None))
    # infeasibility certificate legitimately carries bound=None.
    cert.check_result(_FakeResult(status="infeasible", gap_certified=True, bound=None))
    # PROVEN_EQUIVALENT certificates may certify freely.
    ok = SoundnessCertificate(MethodKind.BENDERS, Soundness.PROVEN_EQUIVALENT, "exact")
    ok.check_result(_FakeResult(status="optimal", gap_certified=True, bound=None))


def test_effective_level_surfaces_driver_verdict():
    # UNKNOWN certificate whose driver certified optimality → authoritative
    # PROVEN_EQUIVALENT (GBD UNKNOWN→exact disagreement fixed).
    cert = SoundnessCertificate(MethodKind.GENERALIZED_BENDERS, Soundness.UNKNOWN, "convex?")
    proved = _FakeResult(status="optimal", gap_certified=True, bound=1.0)
    unproved = _FakeResult(status="time_limit", gap_certified=False, bound=None)
    assert cert.effective_level(proved) is Soundness.PROVEN_EQUIVALENT
    assert cert.effective_level(unproved) is Soundness.UNKNOWN
    # A relaxation not certified stays a relaxation.
    lag = SoundnessCertificate(MethodKind.LAGRANGIAN, Soundness.RELAXATION, "dual bound")
    assert lag.effective_level(unproved) is Soundness.RELAXATION


def test_solve_runs_postcondition_and_raises_on_bad_driver(monkeypatch):
    # A driver that (wrongly) reports a certified gap without a bound must be
    # caught by solve()'s post-condition, not silently trusted. Lagrangian is a
    # RELAXATION certificate.
    monkeypatch.setattr(
        refmod,
        "solve_lagrangian",
        lambda model, **kw: _FakeResult(status="optimal", gap_certified=True, bound=None),
    )
    dcmp = analyze_decomposition(_coupled_model()).decompose()
    assert dcmp.certificate.level is Soundness.RELAXATION
    with pytest.raises(AssertionError, match="#391 item 1"):
        dcmp.solve()


def test_solve_records_last_result_and_summary_surfaces_verdict(monkeypatch):
    # After a GBD solve that certified optimality, summary() must report the
    # driver's authoritative verdict, not the stale UNKNOWN certificate.
    monkeypatch.setattr(
        refmod,
        "solve_gbd",
        lambda model, **kw: _FakeResult(status="optimal", gap_certified=True, bound=2.0),
    )
    m = _benders_model()
    gbd_cand = Candidate(
        MethodKind.GENERALIZED_BENDERS,
        None,
        "test",
        Soundness.UNKNOWN,
    )
    dcmp = build_decomposition(m, gbd_cand)
    # Force UNKNOWN (a nonconvex-ish path) so the driver-upgrade is observable.
    dcmp.certificate = SoundnessCertificate(
        MethodKind.GENERALIZED_BENDERS, Soundness.UNKNOWN, "recourse not verified convex"
    )
    res = dcmp.solve()
    assert dcmp.last_result is res
    s = dcmp.summary()
    assert "driver certified optimality" in s
    assert "proven_equivalent" in s


# ── solve() dispatch (monkeypatched drivers) ───────────────────


def test_solve_dispatches_to_benders_driver(monkeypatch):
    captured = {}

    def fake_benders(model, *, structure=None, **cfg):
        captured["model"] = model
        captured["structure"] = structure
        captured["cfg"] = cfg
        return "BENDERS_RESULT"

    monkeypatch.setattr(refmod, "solve_benders", fake_benders)
    m = _benders_model()
    dcmp = analyze_decomposition(m).decompose()
    result = dcmp.solve(max_iterations=7)
    assert result == "BENDERS_RESULT"
    assert captured["model"] is m
    assert captured["structure"] is dcmp.structure
    assert captured["cfg"]["max_iterations"] == 7


def test_solve_dispatches_to_lagrangian_driver(monkeypatch):
    monkeypatch.setattr(refmod, "solve_lagrangian", lambda model, **kw: "LAGR_RESULT")
    dcmp = analyze_decomposition(_coupled_model()).decompose()
    assert dcmp.solve() == "LAGR_RESULT"


def test_solve_none_falls_back_to_model_solve(monkeypatch):
    m = _benders_model()
    monkeypatch.setattr(type(m), "solve", lambda self, **kw: "MONO_RESULT", raising=False)
    dcmp = build_decomposition(
        m, Candidate(MethodKind.NONE, None, "baseline", Soundness.PROVEN_EQUIVALENT)
    )
    assert dcmp.solve() == "MONO_RESULT"


def test_solve_independent_blocks_falls_back_to_model_solve(monkeypatch):
    m = _independent_blocks_model()
    monkeypatch.setattr(type(m), "solve", lambda self, **kw: "MONO_RESULT", raising=False)
    dcmp = analyze_decomposition(m).decompose()
    assert dcmp.solve() == "MONO_RESULT"


# ── Phase 5 (T5.3): the certificate records a real convexity check ──


def test_gbd_certificate_reflects_convexity():
    import discopt.modeling as dm
    from discopt.decomposition import MethodKind, analyze_decomposition, build_decomposition
    from discopt.decomposition.advisor.types import Soundness

    # Convex MINLP: GBD/OA certificate becomes proven-equivalent.
    m = dm.Model("cvx")
    b = [m.binary(f"b{i}") for i in range(2)]
    x = [m.continuous(f"x{i}", lb=0, ub=5) for i in range(2)]
    for i in range(2):
        m.subject_to(x[i] * x[i] <= 4 * b[i])
    m.subject_to(b[0] + b[1] >= 1)
    m.minimize(sum(x) + 2 * sum(b))
    adv = analyze_decomposition(m)
    gbd = next(c for c in adv.candidates() if c.method is MethodKind.GENERALIZED_BENDERS)
    cert = build_decomposition(m, gbd).certificate
    assert cert.level is Soundness.PROVEN_EQUIVALENT
    assert "convex" in cert.rationale.lower()
