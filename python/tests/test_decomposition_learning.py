"""Tests for the Decomposition Advisor learning foundation (Phase 7).

Covers feature extraction + fingerprinting, the JSONL/in-memory RecordStore and
its nearest-neighbor retrieval, SolveRecord (de)serialization, the record
builder, and the InstanceBasedPolicy (override + safe fallback).
"""

import discopt.modeling as dm
from discopt.decomposition import (
    InstanceBasedPolicy,
    RecordStore,
    SolveRecord,
    analyze_decomposition,
    record_outcome,
)
from discopt.decomposition.advisor import DecompositionAdvisor, MethodKind
from discopt.decomposition.advisor.analyzer import StructureAnalyzer
from discopt.decomposition.learning import (
    ObservedPerformance,
    Outcome,
    build_record,
    extract_features,
    fingerprint,
)

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


def _benders_model():
    m = dm.Model("benders")
    z = m.binary("z")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.subject_to(x <= 5 * z)
    m.subject_to(y <= 5 * z)
    m.minimize(x + y - z)
    return m


# ── features & fingerprint ─────────────────────────────────────


def test_extract_features_from_report():
    report = StructureAnalyzer().analyze(_benders_model())
    f = extract_features(report)
    assert f.num_vars == 3
    assert f.num_constraints == 2
    assert 0.0 < f.integer_fraction <= 1.0
    assert f.blocks_after_integer_projection == 2
    assert len(f.vector()) == len(f.__class__.names())


def test_fingerprint_is_stable_and_structural():
    r1 = StructureAnalyzer().analyze(_benders_model())
    r2 = StructureAnalyzer().analyze(_benders_model())
    assert fingerprint(r1) == fingerprint(r2)
    r3 = StructureAnalyzer().analyze(_independent_blocks_model())
    assert fingerprint(r1) != fingerprint(r3)


# ── SolveRecord (de)serialization ──────────────────────────────


def test_solve_record_roundtrip():
    report = StructureAnalyzer().analyze(_benders_model())
    rec = build_record(
        analyze_decomposition(_benders_model()).scores(),
        report,
        MethodKind.BENDERS,
        observed=ObservedPerformance(wall_clock_s=1.5, iterations=7),
        outcome=Outcome.OPTIMAL,
        timestamp=123.0,
    )
    d = rec.to_dict()
    back = SolveRecord.from_dict(d)
    assert back.chosen == "benders"
    assert back.observed.wall_clock_s == 1.5
    assert back.outcome == "optimal"
    assert back.timestamp == 123.0
    assert back.features.num_vars == 3
    # predicted was filled from the Benders score
    assert back.predicted_speedup is not None


# ── RecordStore ────────────────────────────────────────────────


def test_store_in_memory_append_and_all():
    store = RecordStore()
    report = StructureAnalyzer().analyze(_benders_model())
    rec = build_record(analyze_decomposition(_benders_model()).scores(), report, MethodKind.BENDERS)
    store.append(rec)
    assert len(store) == 1
    assert store.all()[0].chosen == "benders"


def test_store_jsonl_persists_and_reloads(tmp_path):
    path = str(tmp_path / "records.jsonl")
    report = StructureAnalyzer().analyze(_benders_model())
    rec = build_record(analyze_decomposition(_benders_model()).scores(), report, MethodKind.BENDERS)
    RecordStore(path).append(rec)
    # a fresh store loads what was persisted
    reloaded = RecordStore(path)
    assert len(reloaded) == 1
    assert reloaded.all()[0].chosen == "benders"


def test_store_nearest_orders_by_distance():
    store = RecordStore()
    bre = StructureAnalyzer().analyze(_benders_model())
    ind = StructureAnalyzer().analyze(_independent_blocks_model())
    store.append(build_record([], bre, MethodKind.BENDERS))
    store.append(build_record([], ind, MethodKind.INDEPENDENT_BLOCKS))
    # querying with the benders features → the benders record is nearest (dist 0)
    nearest = store.nearest(extract_features(bre), k=2)
    assert nearest[0][0].chosen == "benders"
    assert nearest[0][1] == 0.0


def test_store_nearest_empty_returns_empty():
    assert (
        RecordStore().nearest(extract_features(StructureAnalyzer().analyze(_benders_model()))) == []
    )


# ── record_outcome convenience ─────────────────────────────────


def test_record_outcome_defaults_to_recommendation():
    store = RecordStore()
    adv = analyze_decomposition(_benders_model())
    rec = record_outcome(adv, store, observed=ObservedPerformance(wall_clock_s=2.0), timestamp=1.0)
    assert rec.chosen == "benders"  # the recommendation
    assert len(store) == 1


def test_advisor_features_method():
    f = analyze_decomposition(_benders_model()).features()
    assert f.num_vars == 3


# ── InstanceBasedPolicy ────────────────────────────────────────


def test_instance_policy_defers_when_insufficient_data():
    # empty store → behaves exactly like the rule-based default
    store = RecordStore()
    adv = DecompositionAdvisor(_benders_model(), policy=InstanceBasedPolicy(store, min_records=3))
    assert adv.recommendation().recommendation is MethodKind.BENDERS


def test_instance_policy_overrides_toward_neighbor_winners():
    # seed the store with LAGRANGIAN winners on instances structurally identical
    # to the query; the learner should override the rule-based BENDERS pick
    # *only if* Lagrangian is a viable candidate for the query. Since it is not
    # for the pure-Benders model, we instead verify the vote is computed and the
    # override is gated on viability (falls back to Benders).
    store = RecordStore()
    report = StructureAnalyzer().analyze(_benders_model())
    for i in range(4):
        store.append(build_record([], report, MethodKind.LAGRANGIAN, timestamp=float(i)))
    adv = DecompositionAdvisor(_benders_model(), policy=InstanceBasedPolicy(store, min_records=3))
    # Lagrangian is not a viable candidate here → safe fallback to Benders
    assert adv.recommendation().recommendation is MethodKind.BENDERS


def test_instance_policy_promotes_viable_neighbor_winner():
    # On a model where BENDERS and (via annotation) INDEPENDENT/other candidates
    # both exist, a store voting for the lower-ranked-but-viable method promotes
    # it. We simulate with the benders model: seed votes for BENDERS (the viable
    # pick) and confirm it stays recommended.
    store = RecordStore()
    report = StructureAnalyzer().analyze(_benders_model())
    for i in range(5):
        store.append(build_record([], report, MethodKind.BENDERS, timestamp=float(i)))
    adv = DecompositionAdvisor(_benders_model(), policy=InstanceBasedPolicy(store, min_records=3))
    ranked = adv.ranked()
    rec = next(r for r in ranked if r.recommended)
    assert rec.candidate.method is MethodKind.BENDERS


def test_instance_policy_flips_to_voted_viable_candidate():
    # #391 item 3: the *positive* override — the learner's voted method IS a
    # viable candidate (present, sound, score > 0), so the recommendation must
    # actually FLIP from the rule-based pick to the voted one. Exercised at the
    # policy seam with two synthetic viable candidates so the flip is deterministic
    # and independent of what the generator cascade happens to produce.
    from discopt.decomposition.advisor.scoring import ScoreVector
    from discopt.decomposition.advisor.selection import RuleBasedPolicy, SelectionContext
    from discopt.decomposition.advisor.types import Candidate, Soundness

    report = StructureAnalyzer().analyze(_benders_model())
    ctx = SelectionContext(report=report)

    # Two viable candidates: BENDERS scores higher (the rule-based pick),
    # LAGRANGIAN is viable but lower-scored.
    benders = Candidate(MethodKind.BENDERS, None, "test", Soundness.PROVEN_EQUIVALENT)
    lagr = Candidate(MethodKind.LAGRANGIAN, None, "test", Soundness.RELAXATION)
    scored = [
        (benders, ScoreVector(aggregate=1.5, confidence=1.0)),
        (lagr, ScoreVector(aggregate=0.8, confidence=1.0)),
    ]

    # Baseline: the rule-based policy recommends BENDERS.
    base_reco = next(r for r in RuleBasedPolicy().rank(scored, ctx) if r.recommended)
    assert base_reco.candidate.method is MethodKind.BENDERS

    # Seed the store with LAGRANGIAN winners on the identical instance → the
    # learner votes LAGRANGIAN, which IS viable here → the pick flips.
    store = RecordStore()
    for i in range(4):
        store.append(build_record([], report, MethodKind.LAGRANGIAN, timestamp=float(i)))
    policy = InstanceBasedPolicy(store, min_records=3)
    ranked = policy.rank(scored, ctx)
    reco = next(r for r in ranked if r.recommended)
    assert reco.candidate.method is MethodKind.LAGRANGIAN
    # Exactly one candidate is recommended after the override.
    assert sum(r.recommended for r in ranked) == 1


# ── Phase 5 (T5.2): store auto-wires the learned policy ───────


def test_store_autowires_instance_based_policy():
    from discopt.decomposition import analyze_decomposition
    from discopt.decomposition.learning.policies import InstanceBasedPolicy
    from discopt.decomposition.learning.store import RecordStore

    m = dm.Model("m")
    x = m.continuous("x", lb=0, ub=1)
    m.subject_to(x <= 1)
    m.minimize(x)
    # No store -> rule-based; with a store -> instance-based (learned) policy.
    plain = analyze_decomposition(m)
    assert type(plain._policy).__name__ == "RuleBasedPolicy"
    learned = analyze_decomposition(m, store=RecordStore(path=None))
    assert isinstance(learned._policy, InstanceBasedPolicy)
