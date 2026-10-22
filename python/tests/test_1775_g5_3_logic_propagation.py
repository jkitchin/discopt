"""#1775 G5.3: logic propagation over the indicators (spec §7).

Enumerate used to generate every configuration and let the subproblem discover
that the logic forbade it. ``discopt.gdp.propagate`` refutes a partial indicator
assignment that no completion can satisfy -- the ``logical()`` constraints read
three-valued, and linear rows over indicators alone bounded over the unassigned
ones -- and enumerate never generates a refuted configuration. Measured on the
panel: syngas 8388608 configurations -> 1440, gdp_col 32768 -> 35; on
ex1_linan_2023 and eight_process_logical every one of the 237 refuted complete
configurations was solved globally and came back infeasible.
"""

from __future__ import annotations

import itertools
import warnings

import discopt.modeling as dm
import pytest
from discopt.gdp.algorithms.base import Deadline
from discopt.gdp.algorithms.enumerate import Enumerate, configurations
from discopt.gdp.algorithms.init import GDPSpace
from discopt.gdp.algorithms.subproblem import solve_subproblem
from discopt.gdp.propagate import LogicChecker, _kleene, count_pruned, pruned_configurations
from discopt.modeling.core import BooleanVar, atleast, atmost, exactly

pytestmark = pytest.mark.smoke


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(*a, **k)


def _chain(with_logic: bool = True):
    """Three two-way disjunctions; the logic leaves few configurations:
    d1[0] -> d2[1], at most one of the first disjuncts, and a linear row over
    indicators (y(d3[1]) <= y(d1[1]))."""
    m = dm.Model("chain")
    x = m.continuous("x", lb=0, ub=10)
    d = [m.either_or([[x >= 1 + k], [x <= 8 - k]], name=f"d{k}") for k in range(3)]
    if with_logic:
        m.logical(d[0].indicators[0].implies(d[1].indicators[1]))
        m.logical(atmost(1, [d[0].indicators[0], d[1].indicators[0], d[2].indicators[0]]))
        m.subject_to(d[2].indicators[1].variable <= d[0].indicators[1].variable)
    m.minimize((x - 4.5) ** 2 + sum((k + 1) * dd.indicators[0].variable for k, dd in enumerate(d)))
    return m


def _complete(space, selection):
    """The indicator assignment a configuration fixes; an implication's own
    selector is not part of a configuration and stays unassigned."""
    implication = {d for n in space.tree.walk() if n.kind == "implication" for d in n.disjuncts}
    chosen = {id(d) for d in selection}
    return {
        col: (d.active_value if id(d) in chosen else 1 - d.active_value)
        for d, col in space.columns.items()
        if d not in implication
    }


def test_kleene_on_partial_assignments():
    m = dm.Model("k")
    y = [m.boolean(f"y{i}") for i in range(3)]
    col = {id(b.variable): i for i, b in enumerate(y)}

    def at(assign):
        return lambda bv: (
            None if col[id(bv.variable)] not in assign else bool(assign[col[id(bv.variable)]])
        )

    assert _kleene(y[0].implies(y[1]), at({0: 1})) is None
    assert _kleene(y[0].implies(y[1]), at({0: 1, 1: 0})) is False
    assert _kleene(y[0].implies(y[1]), at({0: 0})) is True
    assert _kleene(atleast(2, y), at({0: 0, 1: 0})) is False
    assert _kleene(atleast(2, y), at({0: 1})) is None
    assert _kleene(atmost(1, y), at({0: 1, 1: 1})) is False
    assert _kleene(exactly(1, y), at({0: 1, 1: 0, 2: 0})) is True
    assert _kleene(exactly(1, y), at({0: 0, 1: 0})) is None
    assert _kleene(y[0].equivalent_to(y[1]), at({0: 1})) is None
    assert _kleene(~y[0] | y[2], at({0: 1, 2: 0})) is False


def test_refutation_is_exact_on_complete_configurations():
    """On a complete configuration the checker refutes exactly what a direct
    evaluation of the logic and the indicator rows rejects (brute force)."""
    space = GDPSpace.from_model(_chain())
    ck = LogicChecker(space)
    refuted = kept = 0
    for sel in configurations(space):
        a = _complete(space, sel)
        y = {name: a[space.columns[d]] for d, name in _names(space).items()}
        truth = (
            (not y["d0_0"] or y["d1_1"])
            and (y["d0_0"] + y["d1_0"] + y["d2_0"] <= 1)
            and (y["d2_1"] <= y["d0_1"])
        )
        assert ck.refutes(a) is (not truth)
        refuted += not truth
        kept += truth
    assert refuted > 0 and kept > 0


def _names(space):
    out = {}
    for node in space.tree.walk():
        for k, d in enumerate(node.disjuncts):
            out[d] = f"{node.name}_{k}"
    return out


def test_every_refuted_configuration_is_infeasible():
    """Soundness (CLAUDE.md §5): a refuted configuration has no feasible point."""
    space = GDPSpace.from_model(_chain())
    ck = LogicChecker(space)
    solved = 0
    for sel in configurations(space):
        if ck.refutes(_complete(space, sel)):
            r = _quiet(solve_subproblem, space.work, sel, Deadline(30), mode="global")
            assert r.status == "infeasible", r.reason
            solved += 1
    assert solved > 0


def test_pruned_walk_keeps_exactly_the_unrefuted_configurations():
    space = GDPSpace.from_model(_chain())
    ck = LogicChecker(space)
    full = [s for s in configurations(space) if not ck.refutes(_complete(space, s))]
    walked = list(pruned_configurations(space, LogicChecker(space)))
    assert sorted(tuple(sorted(id(d) for d in s)) for s in walked) == sorted(
        tuple(sorted(id(d) for d in s)) for s in full
    )
    assert count_pruned(space, 100) == len(full) < 8
    assert count_pruned(space, len(full) - 1) is None


def test_enumerate_agrees_with_and_without_propagation():
    on = _quiet(Enumerate(_chain(), time_limit=60).run)
    off = _quiet(Enumerate(_chain(), time_limit=60, propagate=False).run)
    assert on.status == off.status == "optimal"
    assert on.objective == pytest.approx(off.objective, abs=1e-6)
    assert on.bound == pytest.approx(off.bound, abs=1e-6)
    assert on.solver_stats["n_configurations"] < off.solver_stats["n_configurations"] == 8
    assert on.solver_stats["n_configurations_raw"] == 8


def test_the_refusal_counts_surviving_configurations():
    """8 raw configurations, fewer after propagation: a ceiling between the two
    runs instead of refusing (it refused before G5.3)."""
    space = GDPSpace.from_model(_chain())
    kept = count_pruned(space, 100)
    r = _quiet(Enumerate(_chain(), time_limit=60, max_configurations=kept).run)
    assert r.status == "optimal"
    with pytest.raises(ValueError, match="more than max_configurations"):
        Enumerate(_chain(), max_configurations=kept, propagate=False)


def test_an_implication_selector_stays_unknown():
    """An ``if_then`` selector is not part of a configuration: logic over it
    refutes nothing, whatever the disjunctions select."""
    m = dm.Model("impl")
    x = m.continuous("x", lb=0, ub=10)
    y = m.binary("y")
    m.if_then(y, [x >= 5])
    d = m.either_or([[x >= 1], [x <= 2]], name="d")
    m.logical(d.indicators[0].implies(BooleanVar(y)))
    m.minimize(x)
    space = GDPSpace.from_model(m)
    ck = LogicChecker(space)
    assert all(not ck.refutes(_complete(space, s)) for s in configurations(space))
    assert list(itertools.islice(pruned_configurations(space, ck), 3))
