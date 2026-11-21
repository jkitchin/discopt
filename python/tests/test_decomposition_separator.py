"""Separator program (Kalamaris & Maravelias 2026 §4) and its consumers.

Pins: the program recovers the design variables of a capacity-expansion LP as
the Benders first stage; every separator is lifted and its split re-measured
on the element graph (a disagreement raises); coarsening never merges vertices
of different types; Benders accepts a column-level master and is exact on
every separator the program returns. The advisor's separator search is
retired (it measured slower than the monolith on two-stage LPs); a guard here
keeps the advisor from calling the program.
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
    seps = find_separators(g, "variables", max_components=3, skip_covered=False)
    by_c = {s.num_components: s for s in seps}
    assert 3 in by_c
    three = by_c[3]
    assert set(three.columns) == _design_columns(g)
    assert three.blocks == S  # one block per scenario, measured on the element graph
    assert three.imbalance == 0.0
    for s in seps:  # every result was lifted and re-measured
        assert s.blocks >= s.num_components
        assert sum(len(b) for b in s.block_columns) + s.weight <= g.num_cols
    # The default skips a |C| an earlier separator already covers: here the
    # |C| = 2 split leaves more than 3 blocks, so |C| = 3 is not solved.
    default = find_separators(g, "variables", max_components=3)
    assert [s.num_components for s in default] == [2]
    assert default[0].blocks > 3
    assert default[0].columns == by_c[2].columns


@pytest.mark.requires_highs
def test_constraint_separator_splits_and_is_verified():
    g = ElementGraph.from_model(_capex())
    seps = find_separators(g, "constraints", max_components=2, skip_covered=False)
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


# ── soundness of the library path ───────────────────────────────


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.requires_highs
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_every_separator_benders_is_exact(seed):
    """Every variable separator, used as a Benders master, reproduces the
    monolithic optimum (Benders is exact on any linear partition)."""
    mono = _capex(seed).solve()
    assert mono.status == "optimal"
    g = ElementGraph.from_model(_capex(seed))
    seps = find_separators(g, "variables", max_components=3, skip_covered=False)
    assert seps
    for sep in seps:
        m = _capex(seed)
        arrays = sorted({g.var_names[int(g.col_var[j])] for j in sep.columns})
        st = dataclasses.replace(
            detect_decomposition(m), complicating_vars=arrays, master_columns=sep.columns
        )
        res = solve_benders(m, structure=st)
        assert res.status == "optimal"
        assert res.objective == pytest.approx(mono.objective, rel=1e-4)


@pytest.mark.requires_highs
def test_highs_max_nodes_is_forwarded():
    """``max_nodes`` reaches HiGHS (it used to be accepted and dropped): a
    value outside HiGHS's range is clamped, not rejected, and a node budget too
    small to close the tree stops it short of a certificate."""
    from discopt.solvers import SolveStatus
    from discopt.solvers.milp_highs import solve_milp

    c = -np.ones(3)
    a = np.array([[2.0, 2.0, 2.0]])
    kw = dict(A_ub=a, b_ub=np.array([3.0]), bounds=[(0, 1)] * 3, integrality=np.ones(3, int))
    assert solve_milp(c, max_nodes=10**12, **kw).objective == pytest.approx(-1.0)
    # A 5-dimensional knapsack HiGHS needs ~80 nodes for (seed 2, measured):
    # capped at 5 it must stop short of a certificate, with a valid bound.
    rng = np.random.default_rng(2)
    n, m = 35, 5
    w = rng.integers(10, 100, size=(m, n)).astype(float)
    v = rng.integers(10, 100, size=n).astype(float)
    hard = dict(A_ub=w, b_ub=w.sum(1) / 2, bounds=[(0, 1)] * n, integrality=np.ones(n, int))
    full = solve_milp(-v, **hard)
    assert full.status == SolveStatus.OPTIMAL
    assert full.node_count > 10  # the instance really branches (the probe fires)
    capped = solve_milp(-v, max_nodes=5, **hard)
    assert capped.status == SolveStatus.ITERATION_LIMIT
    assert capped.bound <= full.objective + 1e-6 <= capped.objective + 2e-6


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


def test_advisor_never_runs_the_separator_program(monkeypatch):
    """The advisor's separator search is retired: neither the advisor nor
    ``solve(decomposition="auto")`` calls the program, and the opt-in keyword
    is gone (it measured slower than the monolith on two-stage LPs)."""

    def boom(*args, **kwargs):
        raise AssertionError("the separator program ran from the advisor")

    monkeypatch.setattr(sepmod, "find_separators", boom)
    adv = analyze_decomposition(_capex())
    assert all(c.structure is None or c.structure.master_columns is None for c in adv.candidates())
    assert adv.recommendation().recommendation is MethodKind.NONE
    assert _capex().solve(decomposition="auto").status == "optimal"
    with pytest.raises(TypeError):
        analyze_decomposition(_capex(), separator_search=True)
    from discopt.decomposition.advisor import candidates as cand_mod

    assert not hasattr(cand_mod, "SeparatorGenerator")


# ── user index coarsening, epsilon ───────────────────────────────


@pytest.mark.requires_highs
def test_user_index_coarsening_drives_the_program():
    """``coarse=`` (the paper's index coarsening) replaces the automatic one."""
    g = ElementGraph.from_model(_capex())
    coarse = g.coarsen({"Q": (0, 3), "I": (2,), "U": (2,)})
    seps = find_separators(g, "variables", max_components=3, skip_covered=False, coarse=coarse)
    assert seps
    for s in seps:
        assert s.blocks >= s.num_components
        members = set(s.columns)
        # The separator is a union of the user's super-columns: each one is
        # wholly in or wholly out.
        for sc in range(coarse.num_cols):
            cols = set(np.flatnonzero(coarse.col_of == sc).tolist())
            assert cols <= members or not (cols & members)


def test_coarse_graph_from_another_model_is_refused():
    g = ElementGraph.from_model(_capex())
    other = ElementGraph.from_model(_small_other())
    with pytest.raises(ValueError, match="different graph"):
        find_separators(g, "variables", coarse=other.coarsen({}))


def _small_other():
    m = dm.Model("other")
    x = m.continuous("x", shape=(4,), lb=0, ub=1)
    m.subject_to(x[0] + x[1] <= 1)
    m.minimize(dm.sum(x))
    return m


@pytest.mark.requires_highs
def test_epsilon_trades_separator_size_for_balance():
    g = ElementGraph.from_model(_capex())

    def weight(eps):
        seps = find_separators(g, "variables", max_components=2, epsilon=eps)
        assert seps
        return seps[0].weight

    # A larger penalty on separator weight never yields a larger separator.
    assert weight(100.0) <= weight(0.01)


# ── unbounded first-stage variables, driver diagnostics ─────────


def _unbounded_link():
    """Scenarios coupled only through a FREE variable t (as CVaR's VaR is)."""
    m = dm.Model("free_link")
    t = m.continuous("t", lb=-1e20, ub=1e20)
    q = m.continuous("q", lb=0, ub=10)
    for s in range(4):
        u = m.continuous(f"u_{s}", lb=0, ub=50)
        m.subject_to(u >= 3.0 + s - q - t)
    m.minimize(t + q + sum(m._variables[2 + s] for s in range(4)) / 4)
    return m


def test_unbounded_columns_are_never_in_the_master():
    g = ElementGraph.from_model(_unbounded_link())
    t = g.columns_of("t").tolist()
    for s in find_separators(
        g, "variables", max_components=3, skip_covered=False, excluded_columns=t, max_fraction=1.0
    ):
        assert not set(t) & set(s.columns)


def test_benders_names_an_unbounded_first_stage():
    m = _unbounded_link()
    st = dataclasses.replace(detect_decomposition(m), complicating_vars=["t"], master_columns=(0,))
    res = solve_benders(m, structure=st)
    assert res.status == "error"
    assert "'t'" in res.error and "bounded first stage" in res.error
