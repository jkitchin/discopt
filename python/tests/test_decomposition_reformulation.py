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
