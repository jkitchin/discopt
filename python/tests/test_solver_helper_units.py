"""Unit tests for pure-logic helpers in ``discopt.solver`` (#87).

Covers the SOS1-selector detector, alphaBB finite-difference estimation,
non-smooth model detection, the pre-import callback gate (lazy constraints /
incumbent veto, including the documented soft-fail semantics for user-code
exceptions), and the certified-callback-bound scrubbing rules. All direct
calls on tiny models; no solves.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.callbacks import CutResult
from discopt.modeling.core import Model
from discopt.solver import (
    _INFEASIBILITY_SENTINEL,
    _certified_callback_bound,
    _estimate_alpha_fd,
    _invoke_pre_import_callbacks,
    _model_contains_nonsmooth_node,
    _select_priority_branch_var,
    _sos1_selector_vars,
    _unpack_solution,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# SOS1 selector detection (issue #196 metadata)
# ---------------------------------------------------------------------------


def _selector_model():
    m = Model("sos1")
    s = m.continuous("s", lb=0.0, ub=1.0, shape=(2,))
    y = m.binary("y", shape=(2,))
    x = m.continuous("x", lb=0.0, ub=5.0)
    m.subject_to(s[0] + s[1] == 1.0)  # selection row, k != 0
    m.subject_to(s[0] - y[0] <= 0.0)  # indicator coupling
    m.subject_to(s[1] - y[1] <= 0.0)
    m.minimize(x * s[0] + x * s[1])  # selectors gate bilinear products
    return m


def test_sos1_selector_vars_detects_ex1252_structure():
    m = _selector_model()
    selectors = _sos1_selector_vars(m)
    assert selectors == frozenset({0, 1})


def test_sos1_selector_vars_absent_structure_is_empty():
    # No selection row -> no candidates at all.
    m = Model("plain")
    x = m.continuous("x", lb=0.0, ub=5.0, shape=(2,))
    m.subject_to(x[0] * x[1] >= 1.0)
    m.minimize(x[0] + x[1])
    assert _sos1_selector_vars(m) == frozenset()

    # Selection row exists but selectors never touch the nonlinear core.
    m2 = Model("nolink")
    s = m2.continuous("s", lb=0.0, ub=1.0, shape=(2,))
    z = m2.continuous("z", lb=0.0, ub=5.0, shape=(2,))
    m2.subject_to(s[0] + s[1] == 1.0)
    m2.subject_to(z[0] * z[1] >= 1.0)
    m2.minimize(z[0] + z[1] + s[0])
    assert _sos1_selector_vars(m2) == frozenset()

    # Integer members disqualify the row (not a convex-combination row).
    m3 = Model("introw")
    i = m3.integer("i", lb=0, ub=3, shape=(2,))
    m3.subject_to(i[0] + i[1] == 2)
    m3.minimize(i[0] * i[1])
    assert _sos1_selector_vars(m3) == frozenset()


def test_select_priority_branch_var_prefers_most_fractional_viable():
    priority = frozenset({0, 2})
    node_lb = np.array([0.0, 0.0, 0.0])
    node_ub = np.array([1.0, 1.0, 1.0])
    sol = np.array([0.5, 0.5, 0.9])
    # Index 0 is the most fractional priority var.
    assert _select_priority_branch_var(sol, node_lb, node_ub, priority) == 0
    # A pinned interval is not branchable.
    node_pinned = np.array([0.5, 0.0, 0.0])
    assert _select_priority_branch_var(sol, node_pinned, np.array([0.5, 1.0, 1.0]), priority) == 2
    assert _select_priority_branch_var(sol, node_lb, node_ub, frozenset()) is None


# ---------------------------------------------------------------------------
# alphaBB finite-difference alpha estimation
# ---------------------------------------------------------------------------


def test_estimate_alpha_fd_zero_for_convex_objective():
    m = Model("cvx")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    m.minimize(x**2)
    ev = NLPEvaluator(m)
    alpha = _estimate_alpha_fd(ev, np.array([-2.0]), np.array([2.0]), n_samples=3)
    # Convex objective: no negative curvature, alpha stays at the epsilon floor.
    assert alpha.shape == (1,)
    assert alpha[0] == pytest.approx(1e-6, rel=1e-3)


def test_estimate_alpha_fd_scales_with_negative_curvature():
    m = Model("ccv")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    m.minimize(-(x**2))
    ev = NLPEvaluator(m)
    alpha = _estimate_alpha_fd(ev, np.array([-2.0]), np.array([2.0]), n_samples=3)
    # d2/dx2 of -x^2 is -2 everywhere: alpha = 2/2 * 1.5 + 1e-6.
    assert alpha[0] == pytest.approx(1.5, rel=1e-2)


# ---------------------------------------------------------------------------
# Non-smooth model detection
# ---------------------------------------------------------------------------


def test_model_contains_nonsmooth_node_variants():
    m = Model("smooth")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    m.minimize(x**2)
    assert not _model_contains_nonsmooth_node(m)

    m2 = Model("absobj")
    y = m2.continuous("y", lb=-1.0, ub=1.0)
    m2.minimize(abs(y))
    assert _model_contains_nonsmooth_node(m2)

    m3 = Model("abscon")
    z = m3.continuous("z", lb=-1.0, ub=1.0)
    m3.subject_to(abs(z) <= 0.5)
    m3.minimize(z)
    assert _model_contains_nonsmooth_node(m3)

    # Nested inside a smooth wrapper still counts.
    m4 = Model("nested")
    w = m4.continuous("w", lb=-1.0, ub=1.0)
    m4.minimize((abs(w) + 1.0) ** 2)
    assert _model_contains_nonsmooth_node(m4)


# ---------------------------------------------------------------------------
# Certified callback bound scrubbing
# ---------------------------------------------------------------------------


def test_certified_callback_bound_scrubs_sentinels():
    assert _certified_callback_bound(1.5, True, False) == 1.5
    # Maximize sense reports the original-space (negated) bound.
    assert _certified_callback_bound(1.5, True, True) == -1.5
    # Tainted tree, missing, non-finite, or sentinel bounds are scrubbed.
    assert _certified_callback_bound(1.5, False, False) is None
    assert _certified_callback_bound(None, True, False) is None
    assert _certified_callback_bound(-np.inf, True, False) is None
    assert _certified_callback_bound(1e30, True, False) is None


def test_unpack_solution_reshapes_by_variable():
    m = Model("unpack")
    m.continuous("x", lb=0.0, ub=1.0, shape=(2,))
    m.continuous("z", lb=0.0, ub=1.0)
    out = _unpack_solution(m, np.array([0.1, 0.2, 0.3]))
    np.testing.assert_allclose(out["x"], [0.1, 0.2])
    assert out["z"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Pre-import callback gate
# ---------------------------------------------------------------------------


class _FakeTree:
    def __init__(self, incumbent=None, stats=None):
        self._incumbent = incumbent
        self._stats = stats or {"total_nodes": 5, "global_lower_bound": 0.25, "gap": 0.5}

    def incumbent(self):
        return self._incumbent

    def stats(self):
        return dict(self._stats)


class _RecordingCutPool:
    def __init__(self):
        self.cuts = []

    def add(self, cut):
        self.cuts.append(cut)


def _binary_model():
    m = Model("cb")
    b = m.binary("b", shape=(2,))
    m.minimize(b[0] + b[1])
    return m, b


def _run_gate(model, *, lazy=None, incumbent_cb=None, lbs=None, sols=None):
    n_batch = 3
    result_lbs = np.array([0.5, 0.6, 1e31]) if lbs is None else lbs
    result_sols = np.array([[0.0, 1.0], [0.5, 0.3], [0.0, 0.0]]) if sols is None else sols
    pool = _RecordingCutPool()
    _invoke_pre_import_callbacks(
        model=model,
        tree=_FakeTree(),
        t_start=0.0,
        result_ids=np.array([1, 2, 3]),
        result_lbs=result_lbs,
        result_sols=result_sols,
        result_feas=np.ones(n_batch, dtype=bool),
        n_batch=n_batch,
        int_offsets=[0],
        int_sizes=[2],
        n_vars=2,
        lazy_constraints=lazy,
        incumbent_callback=incumbent_cb,
        _cut_pool=pool,
    )
    return result_lbs, pool


def test_pre_import_lazy_cut_rejects_node_and_adds_cut():
    m, b = _binary_model()
    calls = []

    def lazy(ctx, model):
        calls.append(ctx)
        return [CutResult(terms=[(b[0], 1.0)], sense="<=", rhs=0.0)]

    lbs, pool = _run_gate(m, lazy=lazy)
    # Row 0 (integer feasible) is cut: rejected + cut pooled. Row 1 is
    # fractional and row 2 already infeasible -> untouched, callback not
    # invoked for them.
    assert lbs[0] == _INFEASIBILITY_SENTINEL
    assert lbs[1] == 0.6
    assert len(pool.cuts) == 1
    assert len(calls) == 1
    ctx = calls[0]
    assert ctx.node_count == 5 and ctx.best_bound == 0.25
    assert ctx.node_bound == pytest.approx(0.5)


def test_pre_import_incumbent_veto_and_soft_fail():
    m, _b = _binary_model()

    lbs, _ = _run_gate(m, incumbent_cb=lambda ctx, model, sol: False)
    assert lbs[0] == _INFEASIBILITY_SENTINEL

    # Accepting (True) leaves the node alone.
    lbs, _ = _run_gate(m, incumbent_cb=lambda ctx, model, sol: True)
    assert lbs[0] == pytest.approx(0.5)

    # A raising user callback fails SOFT: logged, node kept.
    def boom(ctx, model, sol):
        raise RuntimeError("user bug")

    lbs, _ = _run_gate(m, incumbent_cb=boom)
    assert lbs[0] == pytest.approx(0.5)


def test_pre_import_lazy_soft_fail_keeps_node_and_pool_untouched():
    m, _b = _binary_model()

    def boom(ctx, model):
        raise RuntimeError("user bug")

    lbs, pool = _run_gate(m, lazy=boom)
    assert lbs[0] == pytest.approx(0.5)
    assert pool.cuts == []


def test_pre_import_callbacks_receive_solution_dict():
    m, _b = _binary_model()
    seen = {}

    def incumbent_cb(ctx, model, sol):
        seen.update(sol)
        return True

    _run_gate(m, incumbent_cb=incumbent_cb)
    np.testing.assert_allclose(seen["b"], [0.0, 1.0])
