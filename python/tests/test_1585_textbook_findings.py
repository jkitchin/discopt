"""Regression tests for the six textbook findings of #1585.

1. ``solver="pounce"`` validated ``pounce_options`` against the convex QP engine
   before deciding the route, so an indefinite QP (which runs on the NLP engine)
   could never receive an NLP option.
2. A POUNCE ``Invalid_Option`` (-12) surfaced as ``status="error"`` with the
   refusal only on stderr; it now raises ``ValueError`` carrying POUNCE's message.
   ``solve_report`` always has ``iterations``.
3. ``SolveResult.termination`` says why a branch-and-bound / MILP search stopped,
   so a ``feasible`` exit at ``max_nodes`` is distinguishable from others.
4. ``SolveResult.gap`` is ``|o-b| / max(|o|, |b|, 1e-10)`` on every exit, one
   helper; the node-limit exit used to report a ``max(1, |o|)``-floored number.
5. The HiGHS route's ``node_count`` is HiGHS's ``mip_node_count`` (the docstring
   said it was always 0; it is 1 on the issue's knapsack).
6. The ``_pounce_report`` docstring (documentation; covered by review, and by the
   ``iterations`` default it documents, tested here).
"""

from __future__ import annotations

import json

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import status as st
from discopt.solvers import _gap


def _formula(o: float, b: float) -> float:
    return abs(o - b) / max(abs(o), abs(b), 1e-10)


# ── 1 + 2: pounce route option handling ────────────────────────────────────


def _indefinite_qp():
    m = dm.Model("indef_qp")
    x = m.continuous("x", shape=(2,), lb=0, ub=4)
    m.minimize(x[0] * x[1])
    m.subject_to(x[0] + x[1] == 2, name="sum")
    return m


@pytest.mark.requires_pounce
def test_an_indefinite_qp_accepts_an_nlp_engine_option():
    """Before: ValueError 'not options of POUNCE's convex LP/QP interior-point
    method'. The model runs on the NLP engine, where the option is valid."""
    r = _indefinite_qp().solve(solver="pounce", pounce_options={"neg_curv_escapes": 0})
    assert r.status == "local_optimal"
    assert r.objective is not None and np.isfinite(r.objective)


@pytest.mark.requires_pounce
def test_a_convex_engine_still_refuses_an_nlp_only_option():
    """Scope: the route that DOES run the convex engine keeps its refusal."""
    m = dm.Model("lp")
    x = m.continuous("x", shape=2, lb=0, ub=10)
    m.minimize(-x[0] - 2 * x[1])
    m.subject_to(x[0] + x[1] <= 4)
    with pytest.raises(ValueError, match="neg_curv_escapes"):
        m.solve(solver="pounce", pounce_options={"neg_curv_escapes": 0})

    q = dm.Model("psd_qp")
    y = q.continuous("y", shape=2, lb=-5, ub=5)
    q.minimize(y[0] ** 2 + y[1] ** 2 - 3 * y[0])
    q.subject_to(y[0] + y[1] <= 1)
    with pytest.raises(ValueError, match="neg_curv_escapes"):
        q.solve(solver="pounce", pounce_options={"neg_curv_escapes": 0})


@pytest.mark.requires_pounce
def test_an_option_pounce_refuses_raises_with_pounces_message():
    """Before: status='error', the reason only on stderr."""
    m = dm.Model("refused")
    y = m.continuous("y", lb=-3, ub=3)
    m.minimize((y - 1) ** 2 + dm.sin(3 * y))
    with pytest.raises(ValueError, match="POUNCE rejected a solver option") as ei:
        m.solve(solver="pounce", initial_solution={y: 0.0}, pounce_options={"theta_min": 1e-4})
    assert "theta_min" in str(ei.value)
    assert "Invalid_Option" in str(ei.value)


def test_a_report_without_iterations_reads_back_with_an_empty_list(tmp_path):
    """A solve that never iterated wrote no ``iterations`` key; the schema now
    always has it."""
    from discopt.solvers._pounce_report import _read_report

    p = tmp_path / "report.json"
    p.write_text(json.dumps({"solver": "pounce", "status": "Invalid_Option"}))
    report = _read_report(str(p))
    assert report is not None
    assert report["iterations"] == []


# ── 3 + 4: termination reason and one gap formula ─────────────────────────


def _knap():
    rng = np.random.default_rng(3)
    n = 30
    v = rng.integers(10, 60, n)
    w = rng.integers(5, 40, n)
    k = dm.Model("knap")
    b = k.binary("b", shape=(n,))
    k.maximize(dm.sum(lambda i: int(v[i]) * b[i], over=range(n)))
    k.subject_to(dm.sum(lambda i: int(w[i]) * b[i], over=range(n)) <= int(w.sum() // 3), name="cap")
    return k


def test_a_node_limit_exit_names_its_reason_and_reports_the_documented_gap():
    """The issue's exact reproduction. Before: gap 0.0052140 = |b-o|/|o|, no
    reason anywhere; the documented formula gives 0.0051870."""
    r = _knap().solve(milp_backend="native", milp_cuts=False, max_nodes=5, batch_size=1)
    assert r.status == "feasible"  # the #933 vocabulary is unchanged
    assert r.termination == st.TERMINATION_NODE_LIMIT
    assert r.objective is not None and r.bound is not None
    assert r.bound >= r.objective - 1e-9  # maximize: the bound is an upper bound
    assert r.gap == pytest.approx(_formula(r.objective, r.bound), rel=1e-12)
    assert r.gap != pytest.approx(abs(r.bound - r.objective) / abs(r.objective), rel=1e-6)


def test_a_converged_exit_names_the_gap_and_reports_the_same_formula():
    r = _knap().solve(milp_backend="native", milp_cuts=False, gap_tolerance=0.06, batch_size=1)
    assert r.status == "optimal"
    assert r.termination == st.TERMINATION_GAP
    assert r.gap == pytest.approx(_formula(r.objective, r.bound), rel=1e-12)


def test_a_drained_tree_reports_exhausted():
    r = _knap().solve(milp_backend="native", milp_cuts=False, batch_size=1)
    assert r.status == "optimal"
    assert r.termination in (st.TERMINATION_EXHAUSTED, st.TERMINATION_GAP)
    # The documented formula INCLUDING its absolute arm (default abs_gap_tolerance
    # 1e-6): since #1953 the tree's bound is a rigorous NS bound, which can sit
    # ~1e-12 off the incumbent rather than equal to it, and such a pair reports 0.0.
    expected = _gap.reported_gap(r.objective, r.bound, abs_tol=1e-6)
    assert r.gap == pytest.approx(expected, rel=1e-12, abs=1e-15)


def _mkp():
    """A multi-dimensional knapsack HiGHS has to branch on (53 nodes measured)."""
    rng = np.random.default_rng(7)
    N, M = 60, 8
    W = rng.integers(5, 100, (M, N))
    V = rng.integers(10, 100, N)
    k = dm.Model("mkp")
    b = k.binary("b", shape=(N,))
    k.maximize(dm.sum(lambda i: int(V[i]) * b[i], over=range(N)))
    for j in range(M):
        k.subject_to(dm.sum(lambda i: int(W[j, i]) * b[i], over=range(N)) <= int(W[j].sum() // 2))
    return k


def test_the_highs_route_names_a_node_limit():
    r = _mkp().solve(milp_backend="highs", max_nodes=2)
    assert r.status == "feasible"
    assert r.termination == st.TERMINATION_NODE_LIMIT
    assert r.gap == pytest.approx(_formula(r.objective, r.bound), rel=1e-12)


def test_the_highs_route_names_a_closed_gap():
    r = _knap().solve(milp_backend="highs")
    assert r.status == "optimal"
    assert r.termination == st.TERMINATION_GAP


def test_every_termination_value_is_in_the_documented_vocabulary():
    for r in (
        _knap().solve(milp_backend="native", max_nodes=5),
        _knap().solve(milp_backend="highs"),
        _mkp().solve(milp_backend="highs", max_nodes=2),
    ):
        assert r.termination in st.TERMINATION_REASONS


def test_termination_round_trips_through_result_io():
    from discopt.result_io import deserialize_result, serialize_result

    r = _knap().solve(milp_backend="native", milp_cuts=False, max_nodes=5, batch_size=1)
    back = deserialize_result(serialize_result(r))
    assert back.termination == r.termination == "node_limit"


@pytest.mark.smoke
def test_the_reported_gap_helper_is_the_documented_formula():
    assert _gap.reported_gap(563.0, 565.9354838709678) == pytest.approx(
        2.9354838709678 / 565.9354838709678
    )
    assert _gap.reported_gap(None, 1.0) is None
    assert _gap.reported_gap(1.0, float("inf")) is None
    assert _gap.reported_gap(0.0, 0.0) == 0.0
    assert _gap.reported_gap(1.0, 1.0 + 1e-9, abs_tol=1e-6) == 0.0
    # symmetric in the pair, so sense never matters
    assert _gap.reported_gap(6.0, 7.0) == _gap.reported_gap(7.0, 6.0) == pytest.approx(1 / 7)


# ── 5: HiGHS node_count agrees with the documentation ──────────────────────


def test_the_highs_node_count_is_highs_own_count():
    """The docstring said ``node_count == 0``; the issue's knapsack reports 1.
    It is HiGHS's ``mip_node_count`` (the root counts), published as-is."""
    r = _knap().solve(milp_backend="highs")
    assert r.algorithm_route.startswith("highs-milp")
    assert r.node_count == int(r.solver_stats["lp/driver_nodes"])
    assert r.node_count == 1
    from discopt.modeling.core import Model

    doc = Model.solve.__doc__
    assert "node_count == 0" not in doc
    assert "mip_node_count" in doc


# ── #1590 review follow-ups (N1, N2, N5, N6) ──────────────────────────────


@pytest.mark.requires_pounce
def test_a_refused_option_escapes_the_default_branch_and_bound(monkeypatch):
    """N1: the B&B node handler used to catch the refusal as a node failure, log
    it, and keep going -- re-probing every option on every failing node NLP (73
    probe solves on a model like this one) and silently dropping the option. It
    now propagates, so the default path refuses as loudly as ``solver="pounce"``."""
    import discopt.solvers.nlp_pounce as np_mod
    from discopt.solvers import PounceOptionError

    calls = {"n": 0}
    real = np_mod._probe_one_option

    def counting(k, v):
        calls["n"] += 1
        return real(k, v)

    monkeypatch.setattr(np_mod, "_probe_one_option", counting)
    m = dm.Model("minlp")
    x = m.continuous("x", lb=-3, ub=3)
    k = m.integer("k", lb=0, ub=3)
    m.minimize((x - 1) ** 2 + dm.sin(3 * x) + 0.1 * (k - 1.5) ** 2 + 0.05 * x * k)
    m.subject_to(x + k <= 3.5)
    with pytest.raises(PounceOptionError, match="theta_min"):
        m.solve(pounce_options={"theta_min": 1e-4}, time_limit=60)
    assert calls["n"] > 0  # the probe ran: the refusal is POUNCE's own
    # one failing NLP call's worth of probes, not one batch per node
    assert calls["n"] <= 20
    assert issubclass(PounceOptionError, ValueError)


@pytest.mark.requires_pounce
def test_a_convex_only_option_on_an_indefinite_qp_is_refused_loudly():
    """N6: the indefinite QP runs on the NLP engine, which does not know the
    convex engine's ``tau`` -- POUNCE refuses it and the refusal is raised."""
    with pytest.raises(ValueError, match="OPTION_INVALID|Invalid_Option"):
        _indefinite_qp().solve(solver="pounce", pounce_options={"tau": 0.99})


class _Res:
    def __init__(self, gap):
        self.gap = gap


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("o", "b", "is_max"),
    [(10.0, 11.0, False), (11.0, 10.0, True)],  # bound crosses the incumbent by ~10%
)
def test_a_materially_inverted_pair_keeps_the_routes_gap(o, b, is_max):
    """N2: OA/GDPopt report 1.0 for an inverted pair ("nothing proved"); the
    symmetric formula would rewrite it to ~0.09 -- an honest-looking open gap."""
    from discopt.solver import _stamp_reported_gap

    r = _Res(1.0)
    _stamp_reported_gap(r, o, b, 1e-6, is_maximize=is_max)
    assert r.gap == 1.0


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("o", "b", "is_max"),
    [(11.0, 10.0, False), (10.0, 11.0, True)],  # an ordinary open gap
)
def test_an_ordinary_pair_is_restamped_with_the_formula(o, b, is_max):
    from discopt.solver import _stamp_reported_gap

    r = _Res(1.0)
    _stamp_reported_gap(r, o, b, 1e-6, is_maximize=is_max)
    assert r.gap == pytest.approx(_formula(o, b), rel=1e-12)


@pytest.mark.smoke
def test_an_inversion_within_rounding_is_still_reconciled():
    from discopt.solver import _stamp_reported_gap

    r = _Res(1.0)
    _stamp_reported_gap(r, 10.0, 10.0 + 1e-9, 1e-6, is_maximize=False)
    assert r.gap == 0.0


@pytest.mark.smoke
def test_an_unknown_sense_leaves_the_routes_gap():
    from discopt.solver import _stamp_reported_gap

    r = _Res(0.25)
    _stamp_reported_gap(r, 11.0, 10.0, 1e-6, is_maximize=None)
    assert r.gap == 0.25


def _through_chokepoint(result, model=None, tols=(1e-4, 1e-6)):
    """Run ``result`` through the solve-level chokepoint as a route's return."""
    from discopt import solver as S

    def fake(model):
        S._GAP_TOLERANCES.append(tols)
        return result

    return S._stamp_layer_timing(fake)(model)


@pytest.mark.smoke
def test_the_chokepoint_keeps_an_inverted_pairs_sentinel():
    """N2 end-to-end: the chokepoint knows the model's sense and passes it."""
    from discopt.modeling.core import SolveResult

    m = dm.Model("min")
    x = m.continuous("x", lb=0, ub=20)
    m.minimize(x)
    r = SolveResult(status="feasible", objective=10.0, bound=11.0, gap=1.0)
    assert _through_chokepoint(r, m).gap == 1.0
    r2 = SolveResult(status="feasible", objective=11.0, bound=10.0, gap=1.0)
    assert _through_chokepoint(r2, m).gap == pytest.approx(1 / 11, rel=1e-12)


@pytest.mark.smoke
@pytest.mark.parametrize("status", ["time_limit", "node_limit", "iteration_limit"])
def test_the_chokepoint_names_a_limit_status_as_the_termination(status):
    """N6: a route that set no reason but whose status names a budget."""
    from discopt.modeling.core import SolveResult

    r = _through_chokepoint(SolveResult(status=status))
    assert r.termination == status


@pytest.mark.smoke
def test_the_chokepoint_infers_nothing_else_and_overrides_nothing():
    from discopt.modeling.core import SolveResult

    assert _through_chokepoint(SolveResult(status="feasible")).termination is None
    assert _through_chokepoint(SolveResult(status="optimal")).termination is None
    preset = SolveResult(status="time_limit", termination="interrupted")
    assert _through_chokepoint(preset).termination == "interrupted"


@pytest.mark.smoke
def test_a_result_file_without_termination_reads_back_as_none():
    """N5: files written before #1585 have no ``termination`` key."""
    from discopt.modeling.core import SolveResult
    from discopt.result_io import deserialize_result, serialize_result

    d = serialize_result(SolveResult(status="optimal", objective=1.0, termination="gap"))
    assert d.pop("termination") == "gap"
    back = deserialize_result(d)
    assert back.termination is None
    assert back.status == "optimal"
