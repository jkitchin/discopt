"""Separator program (Kalamaris & Maravelias 2026 §4) and its consumers.

Pins: the program recovers the design variables of a capacity-expansion LP as
the Benders first stage; every separator is lifted and its split re-measured
on the element graph (a disagreement raises); coarsening never merges vertices
of different types; Benders accepts a column-level master; the advisor
recommends the separator candidate, ``decompose()`` builds the recommended one,
and its solve reproduces the monolithic optimum.
"""

import dataclasses

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.decomposition import MethodKind, analyze_decomposition, solve_benders, solve_gbd
from discopt.decomposition.graph import separator as sepmod
from discopt.decomposition.graph.element import ElementGraph
from discopt.decomposition.graph.separator import (
    SeparatorError,
    coarsen_by_matching,
    find_separators,
)
from discopt.decomposition.structure import detect_decomposition

G, P, S, T = 2, 2, 3, 3


def _capex(seed=0):
    """Linear capacity expansion: design K[g,p], L[p] shared by S scenarios."""
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(0.6, 1.0, size=(G, P, T))
    demand = rng.uniform(2.0, 6.0, size=(P, S, T)) * (1 + 0.3 * np.arange(S))[None, :, None]
    m = dm.Model("capex")
    K = m.continuous("K", shape=(G, P), lb=0, ub=20)
    L = m.continuous("L", shape=(P,), lb=0, ub=20)
    Q = m.continuous("Q", shape=(G, P, S, T), lb=0, ub=40)
    I = m.continuous("I", shape=(P, S, T), lb=0, ub=40)  # noqa: E741 (the paper's name)
    U = m.continuous("U", shape=(P, S, T), lb=0, ub=40)
    for p in range(P):
        for s in range(S):
            for t in range(T):
                prev = (I[p - 1, s, T - 1] if p > 0 else 1.0) if t == 0 else I[p, s, t - 1]
                m.subject_to(
                    prev - I[p, s, t] + sum(Q[g, p, s, t] for g in range(G)) + U[p, s, t]
                    == float(demand[p, s, t])
                )
                for g in range(G):
                    m.subject_to(Q[g, p, s, t] <= float(alpha[g, p, t]) * K[g, p])
                m.subject_to(I[p, s, t] <= L[p])
    for p in range(1, P):
        for g in range(G):
            m.subject_to(K[g, p] >= K[g, p - 1])
        m.subject_to(L[p] >= L[p - 1])
    opcost = sum(
        (sum(0.2 * Q[g, p, s, t] for g in range(G)) + 5.0 * U[p, s, t]) / S
        for p in range(P)
        for s in range(S)
        for t in range(T)
    )
    m.minimize(3.0 * dm.sum(K) + dm.sum(L) + opcost)
    return m


def _design_columns(g: ElementGraph) -> set[int]:
    return set(g.columns_of("K").tolist()) | set(g.columns_of("L").tolist())


# ── the program ─────────────────────────────────────────────────


@pytest.mark.requires_highs
def test_variable_separator_recovers_design_variables():
    g = ElementGraph.from_model(_capex())
    seps = find_separators(g, "variables", max_components=3)
    by_c = {s.num_components: s for s in seps}
    assert 3 in by_c
    three = by_c[3]
    assert set(three.columns) == _design_columns(g)
    assert three.blocks == S  # one block per scenario, measured on the element graph
    assert three.imbalance == 0.0
    for s in seps:  # every result was lifted and re-measured
        assert s.blocks >= s.num_components
        assert sum(len(b) for b in s.block_columns) + s.weight <= g.num_cols


@pytest.mark.requires_highs
def test_constraint_separator_splits_and_is_verified():
    g = ElementGraph.from_model(_capex())
    seps = find_separators(g, "constraints", max_components=2)
    assert seps
    for s in seps:
        assert s.rows and not s.columns
        assert g.bearing_blocks(drop_rows=s.rows) == s.blocks >= 2


def test_lifted_separator_that_does_not_split_is_refused(monkeypatch):
    """The program's answer is never trusted: a non-split raises."""
    g = ElementGraph.from_model(_capex())
    monkeypatch.setattr(sepmod, "_solve_partition", lambda *a, **k: ([], "optimal"))
    with pytest.raises(SeparatorError, match="disagree"):
        find_separators(g, "variables", max_components=2, max_fraction=1.0)


def test_coarsening_preserves_types_and_pins():
    rng = np.random.default_rng(7)
    merged = 0
    for _ in range(50):
        n = int(rng.integers(4, 30))
        edges = {}
        for _ in range(3 * n):
            a, b = sorted(rng.integers(0, n, size=2).tolist())
            if a != b:
                edges[(a, b)] = edges.get((a, b), 0) + 1
        types = rng.integers(0, 3, size=n)
        pinned = sorted(set(rng.integers(0, n, size=3).tolist()))
        group, nv, _, weight = coarsen_by_matching(n, edges, types, target=2, pinned=pinned)
        assert int(weight.sum()) == n and nv == len(set(group.tolist()))
        pin = np.zeros(n, dtype=bool)
        pin[pinned] = True
        for c in set(group.tolist()):
            members = np.flatnonzero(group == c)
            assert len(set(types[members].tolist())) == 1
            assert len(set(pin[members].tolist())) == 1
            merged += members.size > 1
    assert merged > 0


# ── Benders with a column-level master ──────────────────────────


@pytest.mark.requires_highs
def test_benders_accepts_partial_array_master():
    mono = _capex().solve()
    assert mono.status == "optimal"
    m = _capex()
    g = ElementGraph.from_model(m)
    # Design columns plus the end-of-period inventory of period 0, scenario 0:
    # a first stage that is part of array I.
    cols = sorted(_design_columns(g) | {int(g.columns_of("I")[T - 1])})
    st = dataclasses.replace(
        detect_decomposition(m), complicating_vars=["I", "K", "L"], master_columns=tuple(cols)
    )
    res = solve_benders(m, structure=st)
    assert res.status == "optimal" and res.gap_certified
    assert res.objective == pytest.approx(mono.objective, abs=1e-6)


def test_master_columns_out_of_range_and_gbd_refusal():
    m = _capex()
    st = dataclasses.replace(detect_decomposition(m), master_columns=(10**6,))
    with pytest.raises(ValueError, match="outside the model"):
        solve_benders(m, structure=st)
    with pytest.raises(NotImplementedError, match="master_columns"):
        solve_gbd(m, structure=st)


# ── advisor ─────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.requires_highs
def test_advisor_recommends_separator_benders_and_decomposes_it():
    mono = _capex().solve()
    adv = analyze_decomposition(_capex(), separator_search=True)
    rec = adv.recommendation()
    assert rec.recommendation is MethodKind.BENDERS
    picked = next(r.candidate for r in adv.ranked() if r.recommended)
    assert picked.provenance.startswith("separator program")
    # ``decompose()`` builds the recommended candidate, not the first generated.
    decomposed = adv.decompose()
    assert decomposed.structure.master_columns == picked.structure.master_columns
    assert decomposed.num_blocks == picked.num_blocks
    # The design-only first stage is among the candidates (the scorer may rank
    # a larger master with more blocks above it).
    g = ElementGraph.from_model(_capex())
    masters = [
        set(c.structure.master_columns)
        for c in adv.candidates()
        if c.structure is not None and c.structure.master_columns is not None
    ]
    assert _design_columns(g) in masters
    assert "separator program" in adv.explain()
    result = decomposed.solve()
    assert result.status == "optimal" and result.gap_certified
    assert result.objective == pytest.approx(mono.objective, abs=1e-6)


def test_separator_generator_silent_on_nonlinear_or_structured_models():
    from discopt.decomposition.advisor.candidates import SeparatorGenerator

    gen = SeparatorGenerator()
    nonlinear = _capex()
    nonlinear.subject_to(nonlinear._variables[0][0, 0] ** 2 <= 100)
    assert not gen.applicable(analyze_decomposition(nonlinear).structure())

    m = dm.Model("benders_fixture")
    z = m.binary("z")
    a = m.continuous("a", lb=0, ub=5)
    b = m.continuous("b", lb=0, ub=5)
    m.subject_to(a <= 5 * z)
    m.subject_to(b <= 5 * z)
    m.minimize(a + b - z)
    report = analyze_decomposition(m).structure()
    assert report.integer_localizes and not gen.applicable(report)


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.requires_highs
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_separator_candidate_is_sound(seed):
    """Benders candidates reproduce the optimum; Lagrangian bounds stay below it."""
    from discopt.decomposition.ir.reformulation import build_decomposition

    mono = _capex(seed).solve()
    adv = analyze_decomposition(_capex(seed), separator_search=True)
    cands = [c for c in adv.candidates() if c.provenance.startswith("separator program")]
    assert cands
    for c in cands:
        res = build_decomposition(adv._model, c).solve(max_iterations=60, time_limit=60.0)
        if c.method is MethodKind.BENDERS:
            assert res.status == "optimal"
            assert res.objective == pytest.approx(mono.objective, abs=1e-6)
        else:
            assert res.bound is None or res.bound <= mono.objective + 1e-6


@pytest.mark.requires_highs
def test_highs_node_limit_is_forwarded():
    """``node_limit`` reaches HiGHS (``max_nodes`` never has): an invalid value
    is rejected by HiGHS itself, a valid one solves."""
    from discopt.solvers.milp_highs import solve_milp

    c = -np.ones(3)
    a = np.array([[2.0, 2.0, 2.0]])
    kw = dict(A_ub=a, b_ub=np.array([3.0]), bounds=[(0, 1)] * 3, integrality=np.ones(3, int))
    assert solve_milp(c, node_limit=5, **kw).objective == pytest.approx(-1.0)
    with pytest.raises(RuntimeError, match="mip_max_nodes"):
        solve_milp(c, node_limit=-1, **kw)


@pytest.mark.requires_highs
def test_separators_are_deterministic():
    """A node budget, not a wall clock: the same graph gives the same answer."""
    g = ElementGraph.from_model(_capex())
    first = [
        (s.num_components, s.columns) for s in find_separators(g, "variables", max_components=3)
    ]
    second = [
        (s.num_components, s.columns) for s in find_separators(g, "variables", max_components=3)
    ]
    assert first and first == second


def test_separator_search_is_opt_in(monkeypatch):
    """Neither the default advisor nor ``solve(decomposition="auto")`` runs the
    separator program (it measured 39 s against a 0.8 s solve at 774 columns)."""
    from discopt.decomposition.advisor import candidates as cand_mod

    def boom(*args, **kwargs):
        raise AssertionError("the separator program ran without being requested")

    monkeypatch.setattr(sepmod, "find_separators", boom)
    assert not any(isinstance(g, cand_mod.SeparatorGenerator) for g in cand_mod.DEFAULT_GENERATORS)
    adv = analyze_decomposition(_capex())
    assert all(not c.provenance.startswith("separator program") for c in adv.candidates())
    assert adv.recommendation().recommendation is MethodKind.NONE
    result = _capex().solve(decomposition="auto")
    assert result.status == "optimal"
    with pytest.raises(AssertionError, match="without being requested"):
        analyze_decomposition(_capex(), separator_search=True).candidates()
