"""Tests for the Decomposition Advisor scoring stage (Phase 3).

Covers the analytic performance model, the scalarized score, gatekeeping/penalty
behavior, and ranking via ``advisor.scores()``.
"""

import discopt.modeling as dm
from discopt.decomposition import analyze_decomposition
from discopt.decomposition.advisor import (
    MethodKind,
    ScoringWeights,
    Soundness,
    StructureAnalyzer,
)
from discopt.decomposition.advisor.candidates import generate_candidates
from discopt.decomposition.advisor.scoring import Scorer
from discopt.decomposition.advisor.types import Candidate

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


def _top(adv):
    return adv.scores()[0]


# ── baseline & ranking ─────────────────────────────────────────


def test_monolithic_baseline_scores_zero():
    adv = analyze_decomposition(_independent_blocks_model())
    none = next(sv for c, sv in adv.scores() if c.method is MethodKind.NONE)
    assert none.aggregate == 0.0
    assert none.confidence == 1.0
    assert none.performance is None


def test_independent_blocks_beats_baseline_and_ranks_first():
    adv = analyze_decomposition(_independent_blocks_model())
    top_cand, top_sv = _top(adv)
    assert top_cand.method is MethodKind.INDEPENDENT_BLOCKS
    assert top_sv.aggregate > 0.0
    # a balanced 2-block split is ~2x with full efficiency
    assert top_sv.performance.num_blocks == 2
    assert top_sv.performance.parallel_efficiency == 1.0
    assert top_sv.performance.estimated_speedup > 1.5


def test_benders_ranks_first_on_integer_coupled_model():
    adv = analyze_decomposition(_benders_model())
    top_cand, top_sv = _top(adv)
    assert top_cand.method is MethodKind.BENDERS
    assert top_sv.aggregate > 0.0
    # recourse splits into 2 blocks once z is fixed
    assert top_sv.performance.num_blocks == 2
    assert top_sv.metrics["integer_localization"] == 1.0


def test_lagrangian_beats_baseline_on_lightly_coupled_model():
    adv = analyze_decomposition(_coupled_model())
    scores = {c.method: sv for c, sv in adv.scores()}
    assert MethodKind.LAGRANGIAN in scores
    assert scores[MethodKind.LAGRANGIAN].aggregate > scores[MethodKind.NONE].aggregate


def test_scores_are_sorted_descending():
    adv = analyze_decomposition(_benders_model())
    aggs = [sv.aggregate for _, sv in adv.scores()]
    assert aggs == sorted(aggs, reverse=True)


# ── gatekeeping & penalties ────────────────────────────────────


def test_dense_coupling_penalized_below_baseline():
    # Annotate most constraints as coupling → coupling density above tau → the
    # Lagrangian candidate is pushed below the no-decomposition baseline.
    m = _coupled_model()
    for i, c in enumerate(m._constraints):
        m.mark_coupling(c)
    adv = analyze_decomposition(m)
    scores = {c.method: sv for c, sv in adv.scores()}
    assert scores[MethodKind.NONE].aggregate == 0.0
    if MethodKind.LAGRANGIAN in scores:
        assert scores[MethodKind.LAGRANGIAN].aggregate < 0.0
    # baseline should therefore rank first
    assert adv.scores()[0][0].method is MethodKind.NONE


def test_heuristic_soundness_is_vetoed():
    m = _benders_model()
    report = StructureAnalyzer().analyze(m)
    bogus = Candidate(
        method=MethodKind.BENDERS,
        structure=None,
        provenance="test",
        est_soundness=Soundness.HEURISTIC,
    )
    sv = Scorer().score(m, bogus, report)
    assert sv.aggregate == float("-inf")


def test_gbd_unknown_soundness_trims_confidence():
    # a nonlinear recourse model → GBD candidate with UNKNOWN soundness
    m = dm.Model("gbd")
    z = m.binary("z")
    x = m.continuous("x", lb=0.1, ub=5)
    y = m.continuous("y", lb=0.1, ub=5)
    m.subject_to(x * x <= 5 * z)
    m.subject_to(y * y <= 5 * z)
    m.minimize(x + y - z)
    cands = generate_candidates(m, StructureAnalyzer().analyze(m))
    gbd = [c for c in cands if c.method is MethodKind.GENERALIZED_BENDERS]
    # if GBD was generated, its score confidence is discounted vs its base
    if gbd:
        sv = Scorer().score(m, gbd[0], StructureAnalyzer().analyze(m))
        assert sv.aggregate != float("-inf")
        assert 0.0 < sv.confidence < 0.6


# ── weights & performance model ────────────────────────────────


def test_custom_weights_change_scores():
    m = _independent_blocks_model()
    adv_default = analyze_decomposition(m)
    default_top = adv_default.scores()[0][1].aggregate

    from discopt.decomposition.advisor import DecompositionAdvisor

    heavy = DecompositionAdvisor(m, scorer=Scorer(ScoringWeights(w_parallel=5.0)))
    heavy_top = heavy.scores()[0][1].aggregate
    assert heavy_top > default_top


def test_performance_estimate_summary_string():
    adv = analyze_decomposition(_benders_model())
    _, sv = adv.scores()[0]
    assert "speedup" in sv.performance.summary()
    assert "score=" in sv.summary()


def test_summary_lists_ranked_scores():
    adv = analyze_decomposition(_benders_model())
    s = adv.summary()
    assert "Ranked candidates" in s
    assert "speedup" in s
