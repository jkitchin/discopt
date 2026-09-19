"""Issue #1344: the duplicated helpers in solver.py now have one definition each.

`_affine_reduce` is the only one whose signature changed (it took `model` from a
closure and now takes it as a parameter), so it carries the behavioural tests.
`_record_improver_run` and `_validate_injected_candidate` were lifted verbatim and
are covered by their binders' existing call sites plus the bound-neutrality panel
recorded in the PR; the tests here pin the properties a future edit could break.
"""

from __future__ import annotations

import logging

import discopt.modeling as dm
import numpy as np
from discopt.modeling.core import Constant
from discopt.solver import (
    _affine_reduce,
    _improver_within_contingent,
    _record_improver_run,
    _verify_and_inject_candidate,
)

# ---------------------------------------------------------------------------
# _affine_reduce
# ---------------------------------------------------------------------------


def _model_with(expr_builder):
    m = dm.Model("affine")
    x = m.continuous("x", shape=(3,), lb=0, ub=10)
    m.minimize(x[0])
    return m, x, expr_builder(x)


def test_constant_reduces_to_an_empty_map_and_the_value():
    m, _x, e = _model_with(lambda _x: Constant(4.0))
    assert _affine_reduce(e, m) == ({}, 4.0)


def test_the_scale_applies_to_a_bare_constant():
    m, _x, e = _model_with(lambda _x: Constant(4.0))
    assert _affine_reduce(e, m, -2.0) == ({}, -8.0)


def test_single_variable_reduces_to_unit_coefficient():
    m, x, e = _model_with(lambda x: x[1])
    coeffs, const = _affine_reduce(e, m)
    assert coeffs == {1: 1.0}
    assert const == 0.0


def test_negation_flips_the_sign():
    m, x, e = _model_with(lambda x: -x[2])
    coeffs, const = _affine_reduce(e, m)
    assert coeffs == {2: -1.0}
    assert const == 0.0


def test_sum_and_difference_accumulate_per_index():
    m, x, e = _model_with(lambda x: x[0] + x[1] - x[0])
    coeffs, const = _affine_reduce(e, m)
    assert coeffs == {0: 0.0, 1: 1.0}
    assert const == 0.0


def test_constant_scaling_on_either_side():
    m, x, left = _model_with(lambda x: 3.0 * x[0])
    coeffs, _ = _affine_reduce(left, m)
    assert coeffs == {0: 3.0}

    m2, x2, right = _model_with(lambda x: x[0] * 3.0)
    coeffs2, _ = _affine_reduce(right, m2)
    assert coeffs2 == {0: 3.0}


def test_the_outer_scale_parameter_multiplies_through():
    m, x, e = _model_with(lambda x: x[0] + 2.0)
    coeffs, const = _affine_reduce(e, m, -1.0)
    assert coeffs == {0: -1.0}
    assert const == -2.0


def test_a_nonlinear_expression_is_declined():
    m, x, e = _model_with(lambda x: x[0] * x[1])
    assert _affine_reduce(e, m) is None


def test_a_nonlinear_subexpression_declines_the_whole_sum():
    m, x, e = _model_with(lambda x: x[0] + x[1] * x[2])
    assert _affine_reduce(e, m) is None


def test_model_is_a_parameter_not_a_closure():
    """The signature change #1344 makes: two models, one function, no rebinding."""
    m1 = dm.Model("a")
    a = m1.continuous("a", shape=(2,), lb=0, ub=1)
    m1.minimize(a[0])
    m2 = dm.Model("b")
    b = m2.continuous("b", shape=(4,), lb=0, ub=1)
    m2.minimize(b[0])

    assert _affine_reduce(a[1], m1) == ({1: 1.0}, 0.0)
    assert _affine_reduce(b[3], m2) == ({3: 1.0}, 0.0)


# ---------------------------------------------------------------------------
# _record_improver_run
# ---------------------------------------------------------------------------


def test_recording_charges_cost_and_counts_the_call():
    state = {"calls": 0, "found": 0, "cost": 0.0}
    _record_improver_run(state, 5.0, False)
    assert state == {"calls": 1, "found": 0, "cost": 5.0}


def test_recording_an_improvement_also_counts_it():
    state = {"calls": 0, "found": 0, "cost": 0.0}
    _record_improver_run(state, 2.5, True)
    _record_improver_run(state, 2.5, True)
    assert state == {"calls": 2, "found": 2, "cost": 5.0}


def test_the_state_dict_is_mutated_in_place_not_replaced():
    """The binders close over their dict, so the shared function must mutate it."""
    state = {"calls": 0, "found": 0, "cost": 0.0}
    same = state
    _record_improver_run(state, 1.0, True)
    assert same is state
    assert same["calls"] == 1


# ---------------------------------------------------------------------------
# _improver_within_contingent
# ---------------------------------------------------------------------------


class _FakeTree:
    """Minimal stand-in exposing the two members the contingent test reads."""

    def __init__(self, incumbent, total_nodes):
        self._incumbent = incumbent
        self._nodes = total_nodes

    def incumbent(self):
        return self._incumbent

    def stats(self):
        return {"total_nodes": self._nodes}


_KNOBS = dict(success_gain=3.0, offset=0.0, quot=0.5)


def test_the_governor_off_switch_allows_everything():
    state = {"calls": 999, "found": 0, "cost": 1e9}
    assert _improver_within_contingent(
        1e9, budget_on=False, tree=_FakeTree(0.0, 10_000), heur_state=state, **_KNOBS
    )


def test_the_finder_role_is_never_gated():
    """No incumbent yet -> securing the first one wins, whatever the budget says."""
    state = {"calls": 999, "found": 0, "cost": 1e9}
    assert _improver_within_contingent(
        1e9, budget_on=True, tree=_FakeTree(None, 10_000), heur_state=state, **_KNOBS
    )


def test_an_improver_is_refused_once_it_exceeds_the_contingent():
    # contingent = 0 + 0.5 * 100 * (3 * (0+1)/(0+1)) = 150
    state = {"calls": 0, "found": 0, "cost": 0.0}
    tree = _FakeTree(1.0, 100)
    assert _improver_within_contingent(150.0, budget_on=True, tree=tree, heur_state=state, **_KNOBS)
    assert not _improver_within_contingent(
        150.1, budget_on=True, tree=tree, heur_state=state, **_KNOBS
    )


def test_success_widens_the_contingent():
    """The weighting is 3*(found+1)/(calls+1), so a hit buys more budget."""
    tree = _FakeTree(1.0, 100)
    cold = {"calls": 3, "found": 0, "cost": 0.0}  # weight 3*1/4  -> contingent 37.5
    warm = {"calls": 3, "found": 3, "cost": 0.0}  # weight 3*4/4  -> contingent 150
    assert not _improver_within_contingent(
        50.0, budget_on=True, tree=tree, heur_state=cold, **_KNOBS
    )
    assert _improver_within_contingent(50.0, budget_on=True, tree=tree, heur_state=warm, **_KNOBS)


def test_spent_cost_counts_against_the_contingent():
    tree = _FakeTree(1.0, 100)  # contingent 150 with a fresh state
    spent = {"calls": 0, "found": 0, "cost": 145.0}
    assert _improver_within_contingent(5.0, budget_on=True, tree=tree, heur_state=spent, **_KNOBS)
    assert not _improver_within_contingent(
        5.1, budget_on=True, tree=tree, heur_state=spent, **_KNOBS
    )


def test_a_zero_node_tree_allows_only_the_offset():
    """At the root the contingent is the offset alone, so quot cannot matter."""
    state = {"calls": 0, "found": 0, "cost": 0.0}
    tree = _FakeTree(1.0, 0)
    assert not _improver_within_contingent(
        0.1, budget_on=True, tree=tree, heur_state=state, **_KNOBS
    )
    assert _improver_within_contingent(
        0.1, budget_on=True, tree=tree, heur_state=state, success_gain=3.0, offset=1.0, quot=0.5
    )


# ---------------------------------------------------------------------------
# _verify_and_inject_candidate  (the #952 guard, shared by both funnels)
# ---------------------------------------------------------------------------


class _RecordingTree:
    """Captures what the funnel tried to inject, without a real B&B tree."""

    def __init__(self):
        self.injected = []

    def inject_incumbent(self, x, obj):
        self.injected.append((np.asarray(x, dtype=float).copy(), obj))


def _one_row_problem():
    """`x0 + x1 <= 1`, no equalities, node box [0,1]^2."""
    A_ub = np.array([[1.0, 1.0]])
    b_ub = np.array([1.0])
    A_eq = np.zeros((0, 2))
    b_eq = np.zeros(0)
    return A_ub, b_ub, A_eq, b_eq, np.zeros(2), np.ones(2)


def _inject(tree, x, obj, lb, ub, A_ub, b_ub, A_eq, b_eq, how="snapped"):
    return _verify_and_inject_candidate(
        tree,
        np.asarray(x, dtype=float),
        obj,
        n_orig=2,
        node_lb_i=lb,
        node_ub_i=ub,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        log_prefix="TEST-BB",
        how=how,
    )


def test_a_feasible_candidate_is_injected_with_its_objective():
    A_ub, b_ub, A_eq, b_eq, lb, ub = _one_row_problem()
    tree = _RecordingTree()

    assert _inject(tree, [1.0, 0.0], -7.5, lb, ub, A_ub, b_ub, A_eq, b_eq) is True
    assert len(tree.injected) == 1
    assert tree.injected[0][1] == -7.5


def test_a_candidate_violating_a_row_is_refused():
    """The #952 shape: exactly integral, but outside a declared inequality row."""
    A_ub, b_ub, A_eq, b_eq, lb, ub = _one_row_problem()
    tree = _RecordingTree()

    assert _inject(tree, [1.0, 1.0], -99.0, lb, ub, A_ub, b_ub, A_eq, b_eq) is False
    assert tree.injected == []


def test_a_candidate_outside_the_node_box_is_refused():
    """An off-box point can be integral and still pass every row (a binary at -1)."""
    A_ub, b_ub, A_eq, b_eq, lb, ub = _one_row_problem()
    tree = _RecordingTree()

    # satisfies x0 + x1 <= 1, but x0 = -1 is outside the node box
    assert _inject(tree, [-1.0, 0.0], -99.0, lb, ub, A_ub, b_ub, A_eq, b_eq) is False
    assert tree.injected == []


def test_the_node_box_is_what_is_checked_not_the_declared_one():
    """Same point, two node boxes: accepted under the wider one, refused under the tighter."""
    A_ub, b_ub, A_eq, b_eq, _lb, _ub = _one_row_problem()
    x = [1.0, 0.0]

    wide = _RecordingTree()
    assert _inject(wide, x, 1.0, np.zeros(2), np.ones(2), A_ub, b_ub, A_eq, b_eq) is True

    # node has since branched x0 down to 0
    tight = _RecordingTree()
    assert (
        _inject(tight, x, 1.0, np.zeros(2), np.array([0.0, 1.0]), A_ub, b_ub, A_eq, b_eq) is False
    )
    assert tight.injected == []


def test_the_how_label_reaches_the_rejection_log(caplog):
    """MILP passes 'snapped'; MIQP passes 'snapped' or 'rounded' (#1064)."""
    A_ub, b_ub, A_eq, b_eq, lb, ub = _one_row_problem()
    with caplog.at_level(logging.DEBUG, logger="discopt.solver"):
        _inject(_RecordingTree(), [1.0, 1.0], 0.0, lb, ub, A_ub, b_ub, A_eq, b_eq, how="rounded")
    assert any("rounded" in r.getMessage() for r in caplog.records)
