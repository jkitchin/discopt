"""``DISCOPT_CONVEX_NODE_CLIP_POINT_VALUE``: an integral convex node records its point
clipped into its box, valued there.

Measured on the vendored ``nvs15`` (``nlp_bb=True``; NLP-BB's lifted model has 127
columns and 410 rows): a node whose box is a single point returned an NLP point 9.87e-9
outside it -- the interior-point solver's bound relaxation -- and the objective's
gradient reaches 3.28e4 there, so the point was valued 0.99778029, 2.2e-3 below the
true optimum 1.0. The box point is valued exactly 1.000000000000 and satisfies every
row to 8.9e-16. The tree recorded the understated value as the incumbent, and the
point sat below a proved ancestor bound, so #1296's contradiction guard withdrew the
certificate: 5 of 10 runs ended ``feasible`` with no bound. With the flag, 10 of 10
certified (bound <= 1.0), and the guard never fired.
"""

from __future__ import annotations

from pathlib import Path

import discopt.modeling as dm
import discopt.solver as S
import numpy as np
import pytest
from discopt._tape_nlp_evaluator import make_evaluator
from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

FLAG = "DISCOPT_CONVEX_NODE_CLIP_POINT_VALUE"
NVS15 = Path(__file__).parent / "data" / "minlplib_nl" / "nvs15.nl"
G = 3.3e4  # the objective gradient scale measured on nvs15


def _steep_model(row_ub: float = 4.0):
    """``min G*y + z`` over integer ``y`` and continuous ``z``, with ``y + z <= row_ub``."""
    m = dm.Model("steep")
    y = m.integer("y", lb=0, ub=5)
    z = m.continuous("z", lb=0, ub=1)
    m.minimize(G * y + z)
    m.subject_to(y + z <= row_ub)
    ev = make_evaluator(m)
    cl, cu = _infer_constraint_bounds(ev)
    return ev, np.asarray(cl, float), np.asarray(cu, float)


def _clip(x, lo, hi, feas=False, row_ub=4.0):
    ev, cl, cu = _steep_model(row_ub)
    return S._clipped_integral_point(
        ev,
        np.asarray(x, float),
        np.asarray(lo, float),
        np.asarray(hi, float),
        (0,),
        (1,),
        feas,
        cl,
        cu,
    ), ev


def test_a_point_outside_a_fixed_box_is_valued_at_the_box_point():
    x = np.array([2.0 - 1e-8, 0.0])  # integral within 1e-5, 1e-8 below the fixed box
    out, ev = _clip(x, [2.0, 0.0], [2.0, 0.0])
    assert out is not None
    xc, val = out
    np.testing.assert_array_equal(xc, [2.0, 0.0])
    assert val == 2.0 * G
    # The unclipped value is understated by gradient * distance.
    assert float(ev.evaluate_objective(x)) < val - 1e-4


def test_inside_the_box_nothing_changes():
    out, _ = _clip([2.0, 0.5], [2.0, 0.0], [2.0, 1.0])
    assert out is None


def test_a_fractional_node_is_left_alone():
    out, _ = _clip([2.4, 0.0], [2.0, 0.0], [3.0, 0.0])  # 2.4 is not integral
    assert out is None
    out, _ = _clip([2.4, -1e-8], [2.0, 0.0], [3.0, 0.0], feas=True)  # unless the tree says so
    assert out is not None


def test_a_clipped_point_the_constraint_gate_refuses_is_not_used():
    # y + z <= 1.5 with y fixed at 2: the clipped point violates the row by 0.5.
    out, _ = _clip([2.0 - 1e-8, 0.0], [2.0, 0.0], [2.0, 0.0], row_ub=1.5)
    assert out is None


def test_flag_is_default_on_with_an_opt_out(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert S._convex_node_clip_point_value_enabled()
    monkeypatch.setenv(FLAG, "0")
    assert not S._convex_node_clip_point_value_enabled()


@pytest.mark.parametrize("on", [False, True])
def test_the_tree_records_the_exact_incumbent_value(monkeypatch, on):
    """Through the real import path: a node whose integer column is fixed at 2 returns
    an NLP point 1e-8 outside its box (the interior-point bound relaxation). Without
    the flag the tree records the understated value as the incumbent; with it, the
    exact value at the box point."""
    from discopt._rust import PyTreeManager

    monkeypatch.setenv(FLAG, "1" if on else "0")
    ev, cl, cu = _steep_model()
    lo, hi = np.array([2.0, 0.0]), np.array([2.0, 1.0])
    tree = PyTreeManager(2, list(lo), list(hi), [0], [1], "best_first")
    tree.initialize()
    blb, bub, ids, _ = tree.export_batch(1)
    x_hat = np.array([[2.0 - 1e-8, 0.0]])
    lbs = np.array([float(ev.evaluate_objective(x_hat[0]))])
    counts: dict = {}
    points, n_proved, _, _ = S._apply_convex_node_safe_bounds(
        ev,
        [lo],
        [hi],
        lbs,
        x_hat,
        cl,
        cu,
        deadline=None,
        int_offsets=(0,),
        int_sizes=(1,),
        result_feas=np.array([False]),
        counts=counts,
    )
    assert n_proved == 1
    tree.import_results(
        np.asarray(ids, dtype=np.int64), lbs, x_hat, np.array([False]), point_values=points
    )
    tree.process_evaluated()
    value = tree.incumbent()[1]
    if on:
        assert counts == {"clipped": 1}  # rule 6: the arm fired
        assert value == 2.0 * G
        np.testing.assert_array_equal(tree.incumbent()[0], [2.0, 0.0])
    else:
        assert counts == {}
        assert value < 2.0 * G - 1e-4  # understated by gradient * distance


@pytest.mark.slow
@pytest.mark.correctness
def test_nvs15_certifies_in_every_run_with_the_flag(monkeypatch):
    """Unflagged, about half the runs end ``feasible`` (5 of 10 measured; the
    out-of-box node appears on some search paths only). With the flag every run
    certifies. Whether a given run meets that node is not controllable here, so the
    arm's firing is asserted deterministically by
    ``test_the_tree_records_the_exact_incumbent_value``."""
    monkeypatch.setenv(FLAG, "1")
    for _ in range(3):
        r = dm.from_nl(str(NVS15)).solve(time_limit=20, nlp_bb=True)
        stats = r.solver_stats or {}
        assert r.status == "optimal" and r.gap_certified, (r.status, r.bound, stats)
        assert r.objective == pytest.approx(1.0, abs=1e-9)
        assert r.bound is not None and r.bound <= 1.0 + 1e-9
        assert "tree/bound_contradicted" not in stats


# ── abstaining nodes (review of #2276): no bound step, same clip-and-price ────


def test_an_abstaining_candidate_is_clipped_and_priced(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    ev, cl, cu = _steep_model()
    x = np.array([2.0 - 1e-8, 0.0])
    raw = float(ev.evaluate_objective(x))
    xc, val = S._abstaining_node_candidate(ev, x, raw, [2.0, 0.0], [2.0, 1.0], (0,), (1,), cl, cu)
    np.testing.assert_array_equal(xc, [2.0, 0.0])
    assert val == 2.0 * G
    monkeypatch.setenv(FLAG, "0")
    xr, vr = S._abstaining_node_candidate(ev, x, raw, [2.0, 0.0], [2.0, 1.0], (0,), (1,), cl, cu)
    np.testing.assert_array_equal(xr, x)
    assert vr == raw


class _RecordInjections:
    def __init__(self, tree):
        self._t, self.injected = tree, []

    def inject_incumbent(self, x, v):
        self.injected.append((np.asarray(x, dtype=np.float64).copy(), float(v)))
        return self._t.inject_incumbent(x, v)

    def __getattr__(self, name):
        return getattr(self._t, name)


def test_nlpbb_injects_an_abstaining_node_at_its_clipped_point(monkeypatch):
    """Every node abstains (no safe bound) and every node NLP point is pushed 1e-8
    below its box on the integer column. NLP-BB still injects each integral point as
    an incumbent; before the review fix it injected the raw point at its raw value."""
    import discopt.solvers.convex_node_bound as CNB

    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(CNB, "convex_node_safe_bound", lambda *a, **k: (None, "test"))
    orig_apply = S._apply_convex_node_safe_bounds
    shifted = []

    def shift_then_apply(ev, blb, bub, lbs, sols, *a, **k):
        for i in range(len(sols)):
            sols[i][0] = float(np.asarray(blb[i])[0]) - 1e-8  # integral within 1e-5
            shifted.append(i)
        return orig_apply(ev, blb, bub, lbs, sols, *a, **k)

    monkeypatch.setattr(S, "_apply_convex_node_safe_bounds", shift_then_apply)
    trees = []
    orig_tree = S._tree_manager

    def make(*a, **k):
        trees.append(_RecordInjections(orig_tree(*a, **k)))
        return trees[-1]

    monkeypatch.setattr(S, "_tree_manager", make)
    m = dm.Model("steep_convex")
    y = m.integer("y", lb=0, ub=5)
    z = m.continuous("z", lb=0, ub=1)
    m.minimize(G * y + (z - 0.5) ** 2)
    m.subject_to(y + z <= 4)
    m.solve(time_limit=10, nlp_bb=True, deterministic=True, max_nodes=8)
    assert shifted and len(trees) == 1, (shifted, trees)  # rule 6: the shift fired
    nodes = [(x, v) for x, v in trees[0].injected if abs(x[0] - round(x[0])) < 1e-5]
    assert nodes, trees[0].injected  # rule 6: an abstaining node was injected
    for x, v in nodes:
        assert x[0] == round(x[0]) and x[0] >= 0.0, x  # inside the box, not 1e-8 below
        assert v == pytest.approx(G * x[0] + (x[1] - 0.5) ** 2, abs=1e-12)


# ── batches imported without a proved bound: a nonconvex model (heuristic NLP-BB),
# or a convex one with DISCOPT_CONVEX_NODE_SAFE_BOUND=0. The node's raw NLP objective
# went in unchecked as the incumbent value of every integral node. ─────────────────


def test_unproved_batch_clips_integral_nodes_only():
    ev, cl, cu = _steep_model()
    sols = np.array(
        [
            [2.0 - 1e-8, 0.0],  # integral, 1e-8 below its fixed box -> clipped
            [2.0, 0.5],  # integral, inside its box -> untouched
            [2.4, 0.0],  # fractional -> untouched
            [2.0 - 1e-8, 0.0],  # sentinel (infeasible) -> untouched
            [2.0 - 1e-8, 0.0],  # -inf import -> untouched
        ]
    )
    lbs = np.array(
        [float(ev.evaluate_objective(s)) for s in sols[:3]] + [S._INFEASIBILITY_SENTINEL, -np.inf]
    )
    lbs_before, sols_before = lbs.copy(), sols.copy()
    lo = [[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]
    hi = [[2.0, 0.0], [2.0, 1.0], [3.0, 0.0], [2.0, 0.0], [2.0, 0.0]]
    counts: dict = {}
    pts = S._clip_unproved_integral_points(
        ev, lo, hi, lbs, sols, np.zeros(5, bool), (0,), (1,), cl, cu, counts=counts
    )
    assert counts == {"clipped": 1}  # rule 6: the arm fired, exactly once
    assert pts is not None and pts[0] == 2.0 * G and np.isnan(pts[1:]).all()
    np.testing.assert_array_equal(sols[0], [2.0, 0.0])
    np.testing.assert_array_equal(sols[1:], sols_before[1:])
    np.testing.assert_array_equal(lbs, lbs_before)  # no imported bound moves


def test_unproved_batch_with_nothing_to_clip_keeps_the_legacy_import():
    ev, cl, cu = _steep_model()
    sols = np.array([[2.0, 0.5]])
    lbs = np.array([float(ev.evaluate_objective(sols[0]))])
    assert (
        S._clip_unproved_integral_points(
            ev, [[2.0, 0.0]], [[2.0, 1.0]], lbs, sols, None, (0,), (1,), cl, cu
        )
        is None
    )


@pytest.mark.parametrize("on", [False, True])
def test_nonconvex_nlpbb_records_the_exact_incumbent_value(monkeypatch, on):
    """End to end on a NONCONVEX model under ``nlp_bb=True`` (heuristic mode). Every node
    NLP returns its point 1e-8 below its box on the integer column -- the interior-point
    bound relaxation -- valued there, which on this gradient (3.3e4) understates the
    objective by ~3.3e-4. Before the fix the tree recorded that raw value as the
    incumbent value, with no box check; with the clip it records the box point, valued
    exactly. Flag off reproduces the legacy behaviour (the opt-out is honest)."""
    monkeypatch.setenv(FLAG, "1" if on else "0")
    trees: list = []
    orig_tree = S._tree_manager

    def make(*a, **k):
        trees.append(orig_tree(*a, **k))
        return trees[-1]

    shifted: list = []

    def shifting(fn, lb_pos):
        def wrapped(evaluator, *a, **k):
            r = fn(evaluator, *a, **k)
            node_lb = np.asarray(a[lb_pos], dtype=np.float64)
            if r.x is not None and abs(r.x[0] - round(r.x[0])) < 1e-6:
                x = np.asarray(r.x, dtype=np.float64).copy()
                x[0] = node_lb[0] - 1e-8  # integral within 1e-5, outside the box
                r.x, r.objective = x, float(evaluator.evaluate_objective(x))
                shifted.append(x[0])
            return r

        return wrapped

    monkeypatch.setattr(S, "_tree_manager", make)
    # (evaluator, node_lb, ...) for the root; (evaluator, x0, node_lb, ...) after it.
    monkeypatch.setattr(
        S, "_solve_root_node_multistart", shifting(S._solve_root_node_multistart, 0)
    )
    monkeypatch.setattr(S, "_solve_node_nlp", shifting(S._solve_node_nlp, 1))
    m = dm.Model("steep_nonconvex")
    y = m.integer("y", lb=1, ub=5)
    z = m.continuous("z", lb=0, ub=1)
    m.minimize(G * y - (z - 0.5) ** 2)  # concave in z: nonconvex
    m.subject_to(y + z <= 4)
    r = m.solve(time_limit=10, nlp_bb=True, deterministic=True, max_nodes=8)
    assert shifted and len(trees) == 1, (shifted, trees)  # rule 6: the shift fired
    x_inc, v_inc = trees[0].incumbent()
    x_inc = np.asarray(x_inc, dtype=np.float64)
    exact = G * x_inc[0] - (x_inc[1] - 0.5) ** 2
    if on:
        assert x_inc[0] == round(x_inc[0]) and x_inc[0] >= 1.0, x_inc  # inside the box
        assert v_inc == pytest.approx(exact, abs=1e-12)
        stats = r.solver_stats or {}
        assert stats.get("nlpbb/unproved_point_clipped", 0) >= 1, stats  # rule 6
    else:
        assert x_inc[0] < 1.0  # the raw point, outside the box
        assert v_inc < G * round(x_inc[0]) - (x_inc[1] - 0.5) ** 2 - 1e-4  # understated
