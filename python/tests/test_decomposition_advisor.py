"""Tests for the Decomposition Advisor analysis surface (Phase 2).

Covers ``StructureAnalyzer``/``StructureReport``, the candidate generators, the
``DecompositionAdvisor`` façade, and the ``Model.analyze_decomposition()`` hook.
"""

import discopt.modeling as dm
import pytest
from discopt.decomposition import (
    Candidate,
    DecompositionAdvisor,
    MethodKind,
    Soundness,
    analyze_decomposition,
)
from discopt.decomposition.advisor import StructureAnalyzer, generate_candidates

# ── model fixtures ─────────────────────────────────────────────


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
    m.subject_to(x + u <= 1)  # the coupling row
    m.minimize(x + y + u + v)
    return m


def _benders_model():
    """Two continuous blocks tied together only through a binary variable."""
    m = dm.Model("benders")
    z = m.binary("z")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.subject_to(x <= 5 * z)
    m.subject_to(y <= 5 * z)
    m.minimize(x + y - z)
    return m


# ── StructureReport ────────────────────────────────────────────


def test_report_independent_blocks():
    report = StructureAnalyzer().analyze(_independent_blocks_model())
    assert report.num_vars == 4
    assert report.num_constraints == 2
    assert report.is_block_diagonal
    assert report.num_blocks == 2
    assert report.num_integer == 0
    assert report.coupling_density == 0.0


def test_report_coupled_has_coupling_and_bridge():
    report = StructureAnalyzer().analyze(_coupled_model())
    assert not report.is_block_diagonal
    assert report.num_blocks == 1
    assert len(report.coupling_constraints) >= 1
    assert report.coupling_density > 0.0
    assert 2 in report.bridge_constraints  # the x+u<=1 row


def test_report_integer_projection_localizes():
    report = StructureAnalyzer().analyze(_benders_model())
    assert report.num_integer == 1
    assert "z" in report.integer_vars
    # removing z disconnects x-block from y-block
    assert report.blocks_after_integer_projection >= 2
    assert report.integer_localizes


def test_report_summary_is_multiline_string():
    s = StructureAnalyzer().analyze(_coupled_model()).summary()
    assert "StructureReport" in s
    assert "coupling:" in s


# ── candidate generation ───────────────────────────────────────


def test_candidates_always_include_monolithic_baseline():
    report = StructureAnalyzer().analyze(_independent_blocks_model())
    cands = generate_candidates(_independent_blocks_model(), report)
    assert any(c.method is MethodKind.NONE for c in cands)


def test_candidates_independent_blocks():
    m = _independent_blocks_model()
    cands = generate_candidates(m, StructureAnalyzer().analyze(m))
    methods = {c.method for c in cands}
    assert MethodKind.INDEPENDENT_BLOCKS in methods
    ib = next(c for c in cands if c.method is MethodKind.INDEPENDENT_BLOCKS)
    assert ib.est_soundness is Soundness.PROVEN_EQUIVALENT
    assert ib.num_blocks == 2


def test_candidates_benders_from_integer_projection():
    m = _benders_model()
    cands = generate_candidates(m, StructureAnalyzer().analyze(m))
    methods = {c.method for c in cands}
    # linear recourse -> classical Benders, proven-equivalent
    assert MethodKind.BENDERS in methods
    b = next(c for c in cands if c.method is MethodKind.BENDERS)
    assert b.est_soundness is Soundness.PROVEN_EQUIVALENT
    assert "z" in b.structure.complicating_vars


def test_candidates_lagrangian_from_coupling():
    m = _coupled_model()
    cands = generate_candidates(m, StructureAnalyzer().analyze(m))
    methods = {c.method for c in cands}
    assert MethodKind.LAGRANGIAN in methods
    lag = next(c for c in cands if c.method is MethodKind.LAGRANGIAN)
    assert lag.est_soundness is Soundness.RELAXATION
    assert len(lag.structure.coupling_constraints) >= 1


def test_candidates_deduplicated():
    m = _benders_model()
    cands = generate_candidates(m, StructureAnalyzer().analyze(m))
    keys = [(c.method, c.provenance) for c in cands]
    assert len(keys) == len(set(keys))


def test_candidate_summary_string():
    c = Candidate(
        method=MethodKind.BENDERS,
        structure=None,
        provenance="test",
        est_soundness=Soundness.PROVEN_EQUIVALENT,
    )
    assert "Classical Benders" in c.summary()


# ── advisor façade ─────────────────────────────────────────────


def test_advisor_via_function_and_method_agree():
    m = _benders_model()
    a1 = analyze_decomposition(m)
    a2 = m.analyze_decomposition()
    assert isinstance(a1, DecompositionAdvisor)
    assert isinstance(a2, DecompositionAdvisor)
    assert a1.structure().num_vars == a2.structure().num_vars


def test_advisor_caches_report_and_candidates():
    adv = analyze_decomposition(_benders_model())
    assert adv.structure() is adv.structure()
    # candidates() returns a fresh list each call but of equal content
    c1, c2 = adv.candidates(), adv.candidates()
    assert [c.method for c in c1] == [c.method for c in c2]


def test_advisor_blocks_returns_structure():
    adv = analyze_decomposition(_coupled_model())
    blocks = adv.blocks()
    assert blocks.num_blocks >= 1
    assert len(blocks.coupling_constraints) >= 1


def test_advisor_graph_and_export():
    adv = analyze_decomposition(_independent_blocks_model())
    g = adv.graph()
    assert g.num_vars == 4
    js = adv.export_graph(fmt="json")
    assert "nodes" in js


def test_advisor_summary_lists_candidates():
    adv = analyze_decomposition(_benders_model())
    s = adv.summary()
    assert "Candidates" in s
    assert "Benders" in s


def test_later_phase_methods_raise_not_implemented():
    adv = analyze_decomposition(_benders_model())
    with pytest.raises(NotImplementedError):
        adv.recommendation()
    with pytest.raises(NotImplementedError):
        adv.explain()
    with pytest.raises(NotImplementedError):
        adv.decompose()
