"""#1775 G4.4: logic-based branch and bound, stage 1 (spec §5.3, §3.4).

Before this task ``gdp_method="lbb"`` was reserved and refused. :class:`LBB`
branches on disjunctions best-first; a node is the work model with its partial
selection applied, solved by discopt under a per-node budget, so a certified node
solve closes its subtree and anything less only bounds it. Logic propagation
(G5.3) prunes refuted children; OR disjunctions branch on their subsets (Pyomo
raises ``NotImplementedError``); and the time limit returns the open nodes'
bound (Pyomo's time-limit branch raises ``AttributeError``, spec §3.4 item 4).

``node_time_limit`` far below a node solve's cost forces branching on these
small models, so the tree code runs, not just the root solve.
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import discopt.modeling as dm
import pytest
from discopt.gdp.algorithms.init import GDPSpace
from discopt.gdp.algorithms.lbb import LBB, drop_undecided_nonlinear_rows
from discopt.gdp.state import transform_current_disjunctive_state
from discopt.gdp.tree import gdp_tree
from test_1775_g4_2_ric_enumerate import _nested, _nonconvex, _nonlinear_objective, _or_model
from test_1775_g5_3_logic_propagation import _chain

pytestmark = pytest.mark.smoke

#: Small enough that no node solve finishes inside it: every interior node branches.
TINY = 1e-3


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(*a, **k)


def _mixed():
    """Nonlinear rows in every disjunction representation the node relaxation
    drops from: an ``either_or`` disjunct and an ``add_disjunction`` disjunct that
    are left empty, a nested block, and an ``if_then``."""
    m = dm.Model("mixed")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    z = m.binary("z")
    m.either_or([[x**2 + y**2 <= 4, x >= 0.5], [x * y >= 3]], name="d1")
    a = m.make_disjunct("A")
    a.subject_to(x + y >= 3)
    a.subject_to((x - 3) ** 2 <= 1)
    a1 = m.make_disjunct("A1")
    a1.subject_to(x * y <= 2.5)
    a2 = m.make_disjunct("A2")
    a2.subject_to(y <= 0.5)
    a.add_disjunction([a1, a2], name="inner")
    b = m.make_disjunct("B")
    b.subject_to(y**2 >= 2)
    m.add_disjunction([a, b], name="outer")
    m.if_then(z, [x**2 >= 1])
    m.minimize((x - 1.5) ** 2 + (y - 1.2) ** 2 - 0.5 * z)
    return m


MODELS = [
    _nonconvex,
    _or_model,
    _nested,
    _nonlinear_objective,
    _chain,
    # #1913 item 6: its big-M reference solve plus LBB took ~11 s on a dev machine and
    # ran over the fast lane's 15 s call budget on CI (#1901, #1904).
    pytest.param(_mixed, marks=[pytest.mark.slow, pytest.mark.correctness]),
]


@pytest.mark.parametrize("mk", MODELS)
@pytest.mark.parametrize("node_time_limit", [None, TINY])
def test_certified_agreement_with_the_spatial_bb(mk, node_time_limit):
    ref = _quiet(mk().solve, gdp_method="big-m", time_limit=60)
    r = _quiet(LBB(mk(), time_limit=60, node_time_limit=node_time_limit).run)
    assert ref.status == "optimal" and r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(ref.objective, abs=1e-5)
    assert r.bound <= r.objective + 1e-9


def test_the_route():
    r = _quiet(_nonconvex().solve, gdp_method="lbb", time_limit=60)
    assert r.status == "optimal" and r.gap_certified
    assert r.algorithm_route.startswith("gdpopt-lbb")


def test_a_root_whose_relaxed_optimum_verifies_closes_the_tree():
    """The dropped row is slack at the relaxation's optimum (0, 0), which then
    verifies on the model: its objective equals the root bound, nothing branches."""
    m = dm.Model("slack")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.either_or([[x**2 + y**2 <= 100], [x + y >= 5]], name="d")
    m.minimize(x + y)
    r = _quiet(LBB(m, time_limit=60).run)
    assert r.status == "optimal" and r.objective == pytest.approx(0.0, abs=1e-6)
    assert r.solver_stats["n_nodes"] == 1 and r.solver_stats["n_closed"] == 1
    assert r.solver_stats["n_rows_dropped"] == 1


def test_or_branches_on_every_nonempty_subset():
    """{A}, {B}, {A, B}: the root and its three children."""
    r = _quiet(LBB(_or_model(), time_limit=60, node_time_limit=TINY).run)
    assert r.status == "optimal" and r.objective == pytest.approx(2.0, abs=1e-6)
    assert r.solver_stats["n_nodes"] == 4


def test_nested_disjunctions_become_branchable():
    """Root, then A and B; A's nested disjunction then gives A+A1 and A+A2."""
    r = _quiet(LBB(_nested(), time_limit=60, node_time_limit=TINY).run)
    assert r.status == "optimal" and r.objective == pytest.approx(0.25, abs=1e-6)
    assert r.solver_stats["n_nodes"] == 5


def test_logic_propagation_prunes_children_and_changes_nothing_else():
    on = _quiet(LBB(_chain(), time_limit=60, node_time_limit=TINY).run)
    off = _quiet(LBB(_chain(), time_limit=60, node_time_limit=TINY, propagate=False).run)
    assert on.status == off.status == "optimal"
    assert on.objective == pytest.approx(off.objective, abs=1e-6)
    assert on.solver_stats["n_logic_pruned"] > 0 == off.solver_stats["n_logic_pruned"]
    assert on.solver_stats["n_nodes"] < off.solver_stats["n_nodes"]


@pytest.mark.parametrize("node_time_limit", [None, TINY])
def test_an_infeasible_gdp_is_certified_infeasible(node_time_limit):
    m = dm.Model("infeasible")
    x = m.continuous("x", lb=0, ub=1)
    m.either_or([[x**2 >= 4], [x >= 2]], name="d")
    m.minimize(x)
    r = _quiet(LBB(m, time_limit=60, node_time_limit=node_time_limit).run)
    assert r.status == "infeasible" and r.gap_certified and r.x is None


def test_children_the_logic_refutes_are_never_solved():
    """``y0 & y1`` over an exactly-one disjunction: undecided at the root (nothing
    assigned), refuted in both children. The proof of infeasibility is the
    refutation; no child is solved."""
    m = dm.Model("contradiction")
    x = m.continuous("x", lb=0, ub=1)
    d = m.either_or([[x >= 0.5], [x <= 0.5]], name="d")
    m.logical(d.indicators[0] & d.indicators[1])
    m.minimize(x)
    r = _quiet(LBB(m, time_limit=60, node_time_limit=TINY).run)
    assert r.status == "infeasible" and r.gap_certified
    assert r.solver_stats["n_nodes"] == 1 and r.solver_stats["n_logic_pruned"] == 2


def test_maximize_is_in_model_units():
    r = _quiet(LBB(_nonlinear_objective("max"), time_limit=60, node_time_limit=TINY).run)
    assert r.status == "optimal" and r.objective == pytest.approx(-0.5625, abs=1e-5)
    assert r.bound >= r.objective - 1e-9


def test_the_time_limit_returns_the_open_nodes_bound():
    """Spec §3.4: stopping mid-search is a result, not a crash. The bound is the
    minimum over the open nodes, which is below the optimum, and nothing is
    certified."""
    a = LBB(_nested(), time_limit=60, node_time_limit=TINY)
    _quiet(a.iterate)  # the root: not closed, so it branches
    assert len(a._heap) == 2
    a.deadline.time_limit = 0.0
    r = _quiet(a.run)
    assert r.solver_stats["stop_reason"] == "time_limit"
    assert not r.gap_certified and r.status in ("time_limit", "feasible")
    assert r.solver_stats["n_open"] == 2
    assert r.bound is None or r.bound <= 0.25 + 1e-9


def test_an_unresolved_leaf_blocks_optimal(monkeypatch):
    """A leaf whose solve proves nothing stays in the LB with its parent's bound
    and blocks ``optimal``."""
    real = LBB._solve_node
    leaves = []

    def first_leaf_proves_nothing(self, node, leaf):
        r = real(self, node, leaf)
        if leaf and not leaves:
            leaves.append(node)
            return dataclasses.replace(
                r, status="feasible", gap_certified=False, bound=None, bound_valid=False, x=None
            )
        return r

    monkeypatch.setattr(LBB, "_solve_node", first_leaf_proves_nothing)
    r = _quiet(LBB(_nonconvex(), time_limit=60, node_time_limit=TINY).run)
    assert leaves and r.solver_stats["n_unresolved_leaves"] in (0, 1)
    if r.solver_stats["n_unresolved_leaves"] == 1:
        assert r.status == "feasible" and not r.gap_certified
        assert r.bound is None or r.bound <= 3.0 + 1e-9
    else:  # the incumbent closed the gap to the leaf's bound
        assert r.status == "optimal"


def test_the_node_relaxation_drops_only_undecided_nonlinear_rows():
    """At the root every disjunction is undecided: each nonlinear disjunct row
    goes, the global rows and the objective stay, the result still reads as a GDP
    tree, its certified optimum bounds the true one from below, and the work
    model is untouched."""
    space = GDPSpace.from_model(_mixed())
    before = list(space.work._constraints)
    node = space.work.clone()
    dropped = drop_undecided_nonlinear_rows(node)
    assert dropped == 6  # x²+y², x·y >= 3, (x-3)², x·y <= 2.5, y², the if_then's x²
    gdp_tree(node)
    assert space.work._constraints == before
    relaxed = _quiet(node.solve, time_limit=60)
    exact = _quiet(_mixed().solve, gdp_method="big-m", time_limit=60)
    assert relaxed.status == exact.status == "optimal" and relaxed.gap_certified
    assert relaxed.objective <= exact.objective + 1e-6


def test_a_leaf_is_solved_exactly(monkeypatch):
    """Nothing is dropped from a leaf: its certified optimum closes it."""
    import discopt.gdp.algorithms.lbb as lbb

    calls = []
    real = lbb.drop_undecided_nonlinear_rows

    def spy(model):
        calls.append(1)
        return real(model)

    monkeypatch.setattr(lbb, "drop_undecided_nonlinear_rows", spy)
    a = LBB(_nonconvex(), time_limit=60, node_time_limit=TINY)
    r = _quiet(a.run)
    assert r.status == "optimal" and len(calls) == 1  # the root only; two leaves
    assert r.solver_stats["n_nodes"] == 3


def test_exact_or_reads_a_partial_or_selection_exactly():
    """G4.4's one change to G1.3: an OR child that chose {A} has B off."""
    space = GDPSpace.from_model(_or_model())
    a, b = space.tree.disjunctions[0].disjuncts
    with pytest.raises(ValueError, match="OR disjunction"):
        transform_current_disjunctive_state(space.work.clone(), [a])
    m = space.work.clone()
    transform_current_disjunctive_state(m, [a], exact_or=True)
    r = _quiet(m.solve, time_limit=60)
    assert r.status == "optimal" and r.objective == pytest.approx(7.0, abs=1e-6)  # x=4, A only


def test_gdp_options():
    r = _quiet(_chain().solve, gdp_method="lbb", time_limit=60, gdp_options={"check_sat": False})
    assert r.status == "optimal" and r.solver_stats["n_logic_pruned"] == 0
    r = _quiet(_nonconvex().solve, gdp_method="lbb", time_limit=60, gdp_options={"iterlim": 1})
    assert r.solver_stats["iterations"] <= 1
    with pytest.raises(ValueError, match="check_sat"):
        _nonconvex().solve(gdp_method="lbb", gdp_options={"check_sat": 1})
    with pytest.raises(ValueError, match="does not read gdp_options"):
        _nonconvex().solve(gdp_method="lbb", gdp_options={"max_configurations": 3})


def test_node_time_limit_is_validated():
    with pytest.raises(ValueError, match="node_time_limit"):
        LBB(_nonconvex(), node_time_limit=0)
    assert math.isclose(LBB(_nonconvex(), time_limit=100).node_time_limit, 10.0)
    # #1153: the default node budget saturates instead of growing with the run's.
    assert math.isclose(LBB(_nonconvex(), time_limit=3600).node_time_limit, 30.0)
    assert math.isclose(LBB(_nonconvex(), time_limit=2).node_time_limit, 1.0)
    assert math.isclose(LBB(_nonconvex()).node_time_limit, 30.0)
