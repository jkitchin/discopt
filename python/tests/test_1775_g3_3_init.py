"""#1775 G3.3: the native GDP algorithms' initialization strategies.

Before: ``discopt.gdp.algorithms.init`` did not exist; LOA seeded its master only
from the continuous NLP relaxation (spec A7).

After (NC): ``GDPSpace`` (a working clone with every disjunct's indicator a master
column; the user's model is untouched) and ``initialize`` with Pyomo GDPopt's
strategies: no_init, fix_disjuncts, custom_disjuncts, max_binary, set_covering.
The strategies choose configurations through the caller's ``solve`` callback and
never change the caller's master.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.gdp.algorithms.base import Deadline
from discopt.gdp.algorithms.init import INIT_STRATEGIES, GDPSpace, initialize
from discopt.gdp.algorithms.master import MasterMILP
from discopt.gdp.algorithms.subproblem import solve_subproblem
from discopt.gdp.tree import gdp_tree

pytestmark = pytest.mark.smoke


def _two(infeasible_d1=False):
    """d: [x <= 2] v [x >= 5, x*y >= k];  e: [y <= 1] v [y >= 4, exp(y) <= 100].

    d[1] and e[1] hold nonlinear rows. With k = 1000 > 10*10, d[1] is infeasible.
    """
    m = dm.Model("two")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    k = 1000.0 if infeasible_d1 else 3.0
    m.either_or([[x <= 2], [x >= 5, x * y >= k]], name="d")
    m.either_or([[y <= 1], [y >= 4, dm.exp(y) <= 100]], name="e")
    m.minimize((x - 3) ** 2 + y)
    return m


def _by_name(space):
    return {d.name: d for d in space.columns}


def _global(space, deadline):
    calls = []

    def solve(sel):
        calls.append(sorted(d.name for d in sel))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return solve_subproblem(space.work, sel, deadline, mode="global")

    return solve, calls


# ── GDPSpace ─────────────────────────────────────────────────────────────────


def test_space_works_on_a_clone_with_indicator_columns():
    m = _two()
    before = ([v.name for v in m._variables], len(m._constraints))
    sp = GDPSpace.from_model(m)
    assert ([v.name for v in m._variables], len(m._constraints)) == before
    names = [v.name for v in sp.work._variables]
    assert names == ["x", "y", "d_0_active", "d_1_active", "e_0_active", "e_1_active"]
    assert [v.name for v in sp.lowered._variables[:6]] == names
    assert sorted(sp.columns.values()) == [2, 3, 4, 5]
    assert {d.name for d in sp.nonlinear} == {"d[1]", "e[1]"}


def test_selection_from_x_walks_nested_blocks():
    m = dm.Model("nested")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    a = m.make_disjunct("A")
    a.subject_to(x <= 3)
    a1 = m.make_disjunct("A1")
    a1.subject_to(y >= 6)
    a2 = m.make_disjunct("A2")
    a2.subject_to([y >= 8, x >= 2])
    a.add_disjunction([a1, a2], name="inner")
    b = m.make_disjunct("B")
    b.subject_to(x >= 7)
    m.add_disjunction([a, b], name="outer")
    m.minimize(x + y)
    sp = GDPSpace.from_model(m)
    by_ind = {d.indicator.variable.name: d for d in sp.columns}
    assert set(by_ind) == {"A_active", "A1_active", "A2_active", "B_active"}
    x0 = np.zeros(sum(v.size for v in sp.work._variables))
    for name in ("A_active", "A2_active"):
        x0[sp.columns[by_ind[name]]] = 1
    assert [d.name for d in sp.selection_from_x(x0)] == ["A_active", "A2_active"]
    x1 = np.zeros_like(x0)
    x1[sp.columns[by_ind["B_active"]]] = 1
    x1[sp.columns[by_ind["A1_active"]]] = 1  # under an inactive A: not selected
    assert [d.name for d in sp.selection_from_x(x1)] == ["B_active"]


# ── strategies ───────────────────────────────────────────────────────────────


def test_no_init_tries_nothing():
    sp = GDPSpace.from_model(_two())
    solve, calls = _global(sp, Deadline(60))
    rep = initialize("no_init", sp, solve, Deadline(60))
    assert rep.tried == [] and calls == [] and rep.stop_reason == "no_init"


def test_fix_disjuncts_uses_the_fixed_indicators():
    m = _two()
    d, e = (c for c in m._constraints if hasattr(c, "disjuncts"))
    for ind in (d.indicators[1], e.indicators[0]):
        ind.variable.lb = ind.variable.ub = 1
    sp = GDPSpace.from_model(m)
    dl = Deadline(60)
    solve, calls = _global(sp, dl)
    rep = initialize("fix_disjuncts", sp, solve, dl)
    assert calls == [["d[1]", "e[0]"]]
    ((sel, r),) = rep.tried
    assert r.status == "optimal" and r.objective == pytest.approx(
        4.0 + 0.6, abs=1e-6
    )  # x = 5, 5y >= 3


def test_fix_disjuncts_without_fixed_indicators_is_refused():
    sp = GDPSpace.from_model(_two())
    with pytest.raises(ValueError, match="no indicator is fixed"):
        initialize("fix_disjuncts", sp, lambda s: None, Deadline(60))


def test_custom_disjuncts_in_order_resolving_the_callers_nodes():
    m = _two()
    sp = GDPSpace.from_model(m)
    n = _by_name(sp)
    dl = Deadline(60)
    solve, calls = _global(sp, dl)
    rep = initialize(
        "custom_disjuncts", sp, solve, dl, custom=[[n["d[0]"], n["e[0]"]], [n["d[1]"], n["e[1]"]]]
    )
    assert calls == [["d[0]", "e[0]"], ["d[1]", "e[1]"]] and rep.stop_reason == "done"
    # the caller's own tree resolves by address onto the working copy
    own = {d.name: d for node in gdp_tree(m).walk() for d in node.disjuncts}
    rep = initialize("custom_disjuncts", sp, solve, dl, custom=[[own["d[1]"], own["e[0]"]]])
    assert calls[-1] == ["d[1]", "e[0]"] and rep.tried[0][0][0] is n["d[1]"]
    # a node of a different model is refused
    other = dm.Model("other")
    z = other.continuous("z", lb=0, ub=1)
    other.either_or([[z <= 0.5], [z >= 0.7]], name="q")
    foreign = [d for node in gdp_tree(other).walk() for d in node.disjuncts]
    with pytest.raises(ValueError, match="not a disjunct of this model's current GDP tree"):
        initialize("custom_disjuncts", sp, solve, dl, custom=[foreign[:1]])
    with pytest.raises(ValueError, match="needs at least one"):
        initialize("custom_disjuncts", sp, solve, dl, custom=[])


def test_max_binary_turns_on_every_disjunct_of_an_or():
    m = dm.Model("or")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x >= 1], [x >= 2], [x <= 9]], name="o", semantics="or")
    m.minimize(x)
    sp = GDPSpace.from_model(m)
    dl = Deadline(60)
    solve, calls = _global(sp, dl)
    rep = initialize("max_binary", sp, solve, dl)
    assert calls == [["o[0]", "o[1]", "o[2]"]]
    assert rep.tried[0][1].status == "optimal" and rep.tried[0][1].objective == pytest.approx(2.0)


def test_set_covering_covers_every_nonlinear_disjunct():
    sp = GDPSpace.from_model(_two())
    dl = Deadline(120)
    solve, calls = _global(sp, dl)
    master = sp.master()
    state = (len(master.cuts), list(master.no_good_configs), master.n_solves)
    rep = initialize("set_covering", sp, solve, dl, master=master)
    assert rep.stop_reason == "covered" and rep.uncovered == set()
    assert {"d[1]", "e[1]"} <= {name for c in calls for name in c}
    assert len(calls) == 1  # one configuration covers both
    # the caller's master is not changed (only its solve counter moves)
    assert (len(master.cuts), master.no_good_configs) == state[:2]
    assert master.n_solves == state[2] + 1


def test_set_covering_gives_up_on_an_infeasible_disjunct():
    """d[1] is infeasible: both configurations holding it fail, are excluded for
    the strategy's own master, and the loop ends with d[1] uncovered."""
    sp = GDPSpace.from_model(_two(infeasible_d1=True))
    dl = Deadline(120)
    solve, calls = _global(sp, dl)
    master = sp.master()
    rep = initialize("set_covering", sp, solve, dl, master=master)
    assert {d.name for d in rep.uncovered} == {"d[1]"}
    assert rep.stop_reason in ("nothing new to cover", "master infeasible")
    tried_d1 = [c for c in calls if "d[1]" in c]
    assert len(tried_d1) == 2 and len({tuple(c) for c in calls}) == len(calls)
    assert all(r.status == "infeasible" for s, r in rep.tried if "d[1]" in {d.name for d in s})
    assert master.no_good_configs == [] and master.cuts == []


def test_set_covering_respects_iterlim_and_the_deadline():
    sp = GDPSpace.from_model(_two(infeasible_d1=True))
    dl = Deadline(120)
    solve, calls = _global(sp, dl)
    rep = initialize("set_covering", sp, solve, dl, iterlim=1)
    assert rep.stop_reason == "iterlim" and len(calls) == 1
    t = [0.0]
    expired = Deadline(1.0, clock=lambda: t[0])
    t[0] = 5.0
    for strategy in ("set_covering", "max_binary", "fix_disjuncts", "custom_disjuncts"):
        if strategy == "fix_disjuncts":
            continue  # needs fixed indicators; covered by its own test
        n = _by_name(sp)
        custom = [[n["d[0]"], n["e[0]"]]] if strategy == "custom_disjuncts" else None
        rep = initialize(strategy, sp, lambda s: pytest.fail("solved"), expired, custom=custom)
        assert rep.tried == [] and rep.stop_reason == "deadline", strategy


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"strategy": "set_cover"}, "init strategy must be one of"),
        ({"strategy": "max_binary", "custom": [[]]}, "read only by 'custom_disjuncts'"),
        ({"strategy": "set_covering", "iterlim": -1}, "iterlim must be >= 0"),
    ],
)
def test_bad_arguments_are_refused(kwargs, match):
    sp = GDPSpace.from_model(_two())
    strategy = kwargs.pop("strategy")
    with pytest.raises(ValueError, match=match):
        initialize(strategy, sp, lambda s: None, Deadline(10), **kwargs)


def test_strategy_names_are_pyomos():
    assert set(INIT_STRATEGIES) == {
        "no_init",
        "fix_disjuncts",
        "custom_disjuncts",
        "max_binary",
        "set_covering",
    }


# ── the master's per-call override ───────────────────────────────────────────


def test_master_override_solve_leaves_the_master_alone():
    m = MasterMILP(2, [0, 0], [1, 1], [1, 1], objective=(np.array([3.0, 2.0]), 10.0))
    m.add_row([1.0, 1.0], ">=", 1.0)
    r = m.solve(10, objective_override=[-1.0, -1.0])
    assert r.config == (1, 1) and r.bound is None
    r = m.solve(10, objective_override=[-1.0, -1.0], exclude_also=[(1, 1)])
    assert r.config in ((0, 1), (1, 0)) and m.cuts == []
    assert m.solve(10).config == (0, 1) and m.solve(10).bound == pytest.approx(12.0)
    with pytest.raises(ValueError, match="only for an objective_override"):
        m.solve(10, exclude_also=[(1, 1)])
    with pytest.raises(ValueError, match="2 finite numbers"):
        m.solve(10, objective_override=[1.0])


def test_set_covering_excludes_on_the_indicators_only():
    """Review of #1865: a failed selection is excluded on its indicator columns.
    The full integer configuration would include this unbounded integer, whose
    exact no-good refuses (needs finite bounds), and would let the master
    propose the same selection again with another value of it."""
    m = _two(infeasible_d1=True)
    k = m.integer("k", lb=0)  # unbounded general integer, unused by the disjuncts
    m.subject_to(k >= 0)
    sp = GDPSpace.from_model(m)
    dl = Deadline(120)
    solve, calls = _global(sp, dl)
    rep = initialize("set_covering", sp, solve, dl)
    assert {d.name for d in rep.uncovered} == {"d[1]"}
    assert len([c for c in calls if "d[1]" in c]) == 2  # each failed selection once
