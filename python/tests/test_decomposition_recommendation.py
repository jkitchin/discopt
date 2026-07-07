"""Tests for the Decomposition Advisor recommendation engine (Phase 4).

Covers the selection policy (decision tree + soundness tie-break), the
``Explanation`` data model and its renderers, and ``advisor.recommendation()`` /
``advisor.explain()`` including counterfactual "why not X?" queries.
"""

import json

import discopt.modeling as dm
from discopt.decomposition import Explanation, analyze_decomposition
from discopt.decomposition.advisor import (
    Candidate,
    MethodKind,
    RuleBasedPolicy,
    ScoreVector,
    SelectionContext,
    Severity,
    Soundness,
    StructureAnalyzer,
)
from discopt.decomposition.advisor.scoring import PerformanceEstimate

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


def _monolithic_model():
    # fully dense: every variable in one constraint → no exploitable structure
    m = dm.Model("mono")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    z = m.continuous("z", lb=0, ub=1)
    m.subject_to(x + y + z <= 1)
    m.minimize(x + y + z)
    return m


# ── recommendation ─────────────────────────────────────────────


def test_recommendation_benders():
    exp = analyze_decomposition(_benders_model()).recommendation()
    assert isinstance(exp, Explanation)
    assert exp.recommendation is MethodKind.BENDERS
    assert "Benders" in exp.headline
    assert exp.reasons  # evidence-backed
    assert exp.score is not None and exp.score.performance is not None


def test_recommendation_independent_blocks():
    exp = analyze_decomposition(_independent_blocks_model()).recommendation()
    assert exp.recommendation is MethodKind.INDEPENDENT_BLOCKS


def test_recommendation_lagrangian_on_coupled():
    exp = analyze_decomposition(_coupled_model()).recommendation()
    assert exp.recommendation is MethodKind.LAGRANGIAN


def test_recommendation_none_when_no_structure():
    exp = analyze_decomposition(_monolithic_model()).recommendation()
    assert exp.recommendation is MethodKind.NONE
    assert exp.reasons  # explains why nothing beats the baseline


def test_recommendation_lists_alternatives():
    exp = analyze_decomposition(_benders_model()).recommendation()
    alt_methods = {a.method for a in exp.alternatives}
    assert MethodKind.NONE in alt_methods  # baseline always shown as an alternative


# ── selection policy ───────────────────────────────────────────


def _sv(aggregate, confidence=0.8, perf=True):
    p = PerformanceEstimate(2, 2.0, 1.0, 2.0, 5, confidence) if perf else None
    return ScoreVector(metrics={}, aggregate=aggregate, confidence=confidence, performance=p)


def test_policy_prefers_exact_on_near_tie():
    # a proven-equivalent Benders slightly below an unknown-soundness GBD:
    # correctness-first tie-break should pick Benders.
    report = StructureAnalyzer().analyze(_benders_model())
    benders = Candidate(MethodKind.BENDERS, None, "x", Soundness.PROVEN_EQUIVALENT)
    gbd = Candidate(MethodKind.GENERALIZED_BENDERS, None, "x", Soundness.UNKNOWN)
    none = Candidate(MethodKind.NONE, None, "baseline", Soundness.PROVEN_EQUIVALENT)
    scored = [(gbd, _sv(1.10)), (benders, _sv(1.00)), (none, _sv(0.0, 1.0, perf=False))]
    ranked = RuleBasedPolicy().rank(scored, SelectionContext(report))
    rec = next(r for r in ranked if r.recommended)
    assert rec.candidate.method is MethodKind.BENDERS


def test_policy_recommends_none_when_nothing_beats_baseline():
    report = StructureAnalyzer().analyze(_monolithic_model())
    weak = Candidate(MethodKind.LAGRANGIAN, None, "x", Soundness.RELAXATION)
    none = Candidate(MethodKind.NONE, None, "baseline", Soundness.PROVEN_EQUIVALENT)
    scored = [(none, _sv(0.0, 1.0, perf=False)), (weak, _sv(-0.5))]
    ranked = RuleBasedPolicy().rank(scored, SelectionContext(report))
    rec = next(r for r in ranked if r.recommended)
    assert rec.candidate.method is MethodKind.NONE


# ── rendering ──────────────────────────────────────────────────


def test_explain_text_render():
    txt = analyze_decomposition(_benders_model()).explain()
    assert "Recommended:" in txt
    assert "Why:" in txt
    assert "Estimated:" in txt


def test_explain_markdown_render():
    md = analyze_decomposition(_benders_model()).explain("markdown")
    assert md.startswith("### Recommended:")
    assert "**Why:**" in md


def test_explain_json_render_roundtrips():
    js = analyze_decomposition(_benders_model()).explain("json")
    payload = json.loads(js)
    assert payload["recommendation"] == "benders"
    assert payload["reasons"]
    assert payload["performance"]["num_blocks"] == 2


def test_explanation_concern_for_gbd_unknown_soundness():
    m = dm.Model("gbd")
    z = m.binary("z")
    x = m.continuous("x", lb=0.1, ub=5)
    y = m.continuous("y", lb=0.1, ub=5)
    m.subject_to(x * x <= 5 * z)
    m.subject_to(y * y <= 5 * z)
    m.minimize(x + y - z)
    exp = analyze_decomposition(m).recommendation()
    if exp.recommendation is MethodKind.GENERALIZED_BENDERS:
        assert any(c.severity is Severity.WARN for c in exp.concerns)


# ── counterfactual ─────────────────────────────────────────────


def test_counterfactual_for_generated_method():
    msg = analyze_decomposition(_benders_model()).explain(method="benders")
    assert "Benders" in msg
    assert "score" in msg


def test_counterfactual_for_inapplicable_method():
    # the independent-blocks model has no integer vars → Benders inapplicable
    msg = analyze_decomposition(_independent_blocks_model()).explain(method=MethodKind.BENDERS)
    assert "not applicable" in msg
    assert "integer" in msg.lower()


def test_counterfactual_lagrangian_without_coupling():
    msg = analyze_decomposition(_benders_model()).explain(method="lagrangian")
    assert "not applicable" in msg
    assert "coupling" in msg.lower()
