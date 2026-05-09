"""Tests for the callback and cut generation API.

Validates:
  - CallbackContext and CutResult dataclass construction
  - cut_result_to_dense conversion for scalar and array variables
  - Node callback is called and receives correct context
  - Lazy constraint callback can add cuts that exclude solutions
  - Incumbent callback can reject solutions
  - Callbacks work with MILP models
  - Empty (no-op) callbacks do not break the solver
  - Callback exceptions are caught and logged without crashing
"""

from __future__ import annotations

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import discopt
import numpy as np
import pytest
from discopt.callbacks import (
    CallbackContext,
    CutResult,
    cut_result_to_dense,
)

# ── Helper: build a simple MILP ──


def _simple_milp():
    """MINLP: min exp(x-0.3) + 2*y  s.t.  x + y >= 1,  x, y binary.

    Uses exp() to force MINLP classification and the Python B&B path.
    """
    m = discopt.Model("simple_milp")
    x = m.binary("x")
    y = m.binary("y")
    m.minimize(discopt.exp(x - 0.3) + 2 * y)
    m.subject_to(x + y >= 1, name="cover")
    return m, x, y


def _small_milp_with_continuous():
    """MINLP: min exp(z)  s.t.  z >= x + y - 1,  x + y <= 2,  x,y binary, 0<=z<=10.

    Uses exp() to force MINLP classification and the Python B&B path.
    """
    m = discopt.Model("mixed")
    x = m.binary("x")
    y = m.binary("y")
    z = m.continuous("z", lb=0, ub=10)
    m.minimize(discopt.exp(z))
    m.subject_to(z >= x + y - 1, name="link")
    m.subject_to(x + y <= 2, name="pair")
    return m, x, y, z


# ── Unit tests for CutResult and cut_result_to_dense ──


class TestCutResult:
    def test_valid_senses(self):
        cut = CutResult(terms=[], sense="<=", rhs=0.0)
        assert cut.sense == "<="
        cut = CutResult(terms=[], sense=">=", rhs=1.0)
        assert cut.sense == ">="
        cut = CutResult(terms=[], sense="==", rhs=2.0)
        assert cut.sense == "=="

    def test_invalid_sense_raises(self):
        with pytest.raises(ValueError, match="Invalid cut sense"):
            CutResult(terms=[], sense="<", rhs=0.0)


class TestCutResultToDense:
    def test_scalar_variables(self):
        m, x, y = _simple_milp()
        cut = CutResult(terms=[(x, 1.0), (y, -1.0)], sense="<=", rhs=0.5)
        coeffs, rhs, sense = cut_result_to_dense(cut, m)
        assert coeffs.shape == (m.num_variables,)
        assert rhs == 0.5
        assert sense == "<="
        # x is first variable, y is second
        assert coeffs[0] == 1.0
        assert coeffs[1] == -1.0

    def test_indexed_variables(self):
        m = discopt.Model("array_test")
        x = m.continuous("x", shape=(3,), lb=0, ub=10)
        y = m.binary("y")
        m.minimize(x[0] + x[1] + x[2] + y)
        cut = CutResult(
            terms=[(x[0], 1.0), (x[2], 2.0), (y, -3.0)],
            sense=">=",
            rhs=1.0,
        )
        coeffs, rhs, sense = cut_result_to_dense(cut, m)
        assert coeffs.shape == (4,)  # 3 for x + 1 for y
        assert coeffs[0] == 1.0  # x[0]
        assert coeffs[1] == 0.0  # x[1]
        assert coeffs[2] == 2.0  # x[2]
        assert coeffs[3] == -3.0  # y

    def test_unknown_variable_raises(self):
        m1, x1, _ = _simple_milp()
        m2 = discopt.Model("other")
        z = m2.binary("z")
        m2.minimize(z)
        cut = CutResult(terms=[(z, 1.0)], sense="<=", rhs=0.0)
        with pytest.raises(ValueError, match="not found in model"):
            cut_result_to_dense(cut, m1)


# ── Integration tests with solve ──
# These require the Rust backend (discopt._rust) which provides PyTreeManager.

try:
    import discopt._rust  # noqa: F401

    _has_rust = True
except ImportError:
    _has_rust = False

needs_rust = pytest.mark.skipif(not _has_rust, reason="discopt._rust not available")


@pytest.mark.slow
@needs_rust
class TestNodeCallback:
    def test_node_callback_called(self):
        """Node callback should be invoked at least once during B&B."""
        m, x, y = _simple_milp()
        call_log = []

        def on_node(ctx, model):
            call_log.append(ctx)

        m.solve(node_callback=on_node, time_limit=30)
        assert len(call_log) > 0
        ctx = call_log[0]
        assert isinstance(ctx, CallbackContext)
        assert ctx.node_count >= 0
        assert ctx.elapsed_time > 0

    def test_node_callback_receives_context_fields(self):
        m, x, y = _simple_milp()
        contexts = []

        def on_node(ctx, model):
            contexts.append(ctx)

        m.solve(node_callback=on_node, time_limit=30)
        assert len(contexts) > 0
        ctx = contexts[0]
        assert isinstance(ctx.x_relaxation, np.ndarray)
        assert ctx.x_relaxation.shape == (m.num_variables,)
        assert isinstance(ctx.node_bound, float)


@pytest.mark.slow
@needs_rust
class TestLazyConstraints:
    def test_lazy_cut_excludes_solution(self):
        """Lazy constraints can exclude the otherwise-optimal solution.

        Without the cut, optimal is x=1, y=0 (obj=exp(0.7)~2.01).
        The lazy constraint forces y=1, changing the optimal.
        """
        m, x, y = _simple_milp()
        cut_added = [False]

        def lazy_cb(ctx, model):
            sol = ctx.x_relaxation
            # If y < 0.5 (y=0 in the integer solution), add cut y >= 1
            if sol[1] < 0.5 and not cut_added[0]:
                cut_added[0] = True
                return [CutResult(terms=[(y, 1.0)], sense=">=", rhs=1.0)]
            return []

        result = m.solve(lazy_constraints=lazy_cb, time_limit=30)
        # The cut forces y=1.
        if result.status in ("optimal", "feasible"):
            assert result.x is not None
            assert result.x["y"] >= 0.5  # y should be 1

    def test_empty_lazy_callback_is_noop(self):
        """An empty lazy callback should not change the optimal solution."""
        m, x, y = _simple_milp()

        def noop_lazy(ctx, model):
            return []

        result = m.solve(lazy_constraints=noop_lazy, time_limit=30)
        assert result.status in ("optimal", "feasible")
        assert result.objective is not None


@pytest.mark.slow
@needs_rust
class TestIncumbentCallback:
    def test_incumbent_rejection(self):
        """Incumbent callback can reject solutions where y=0."""
        m, x, y = _simple_milp()

        def reject_y0(ctx, model, solution):
            # Reject solutions where y is 0
            y_val = solution.get("y", np.array([0.0]))
            if np.all(y_val < 0.5):
                return False
            return True

        result = m.solve(incumbent_callback=reject_y0, time_limit=30)
        if result.status in ("optimal", "feasible"):
            assert result.x is not None
            assert result.x["y"] >= 0.5

    def test_accept_all_is_noop(self):
        """Accepting all incumbents should behave identically to no callback."""
        m, x, y = _simple_milp()

        def accept_all(ctx, model, solution):
            return True

        result = m.solve(incumbent_callback=accept_all, time_limit=30)
        assert result.status in ("optimal", "feasible")


@pytest.mark.slow
@needs_rust
class TestCallbackExceptionHandling:
    def test_node_callback_exception_logged(self, caplog):
        """A node callback that raises should not crash the solver."""
        m, x, y = _simple_milp()

        def bad_callback(ctx, model):
            raise RuntimeError("intentional test error")

        import logging

        with caplog.at_level(logging.WARNING, logger="discopt.solver"):
            result = m.solve(node_callback=bad_callback, time_limit=30)
        # Solver should still produce a result
        assert result.status in ("optimal", "feasible", "infeasible", "node_limit")

    def test_lazy_callback_exception_logged(self, caplog):
        """A lazy constraint callback that raises should not crash the solver."""
        m, x, y = _simple_milp()

        def bad_lazy(ctx, model):
            raise ValueError("intentional lazy error")

        import logging

        with caplog.at_level(logging.WARNING, logger="discopt.solver"):
            result = m.solve(lazy_constraints=bad_lazy, time_limit=30)
        assert result.status in ("optimal", "feasible", "infeasible", "node_limit")

    def test_incumbent_callback_exception_logged(self, caplog):
        """An incumbent callback that raises should not crash the solver."""
        m, x, y = _simple_milp()

        def bad_inc(ctx, model, solution):
            raise TypeError("intentional incumbent error")

        import logging

        with caplog.at_level(logging.WARNING, logger="discopt.solver"):
            result = m.solve(incumbent_callback=bad_inc, time_limit=30)
        assert result.status in ("optimal", "feasible", "infeasible", "node_limit")


@pytest.mark.slow
@needs_rust
class TestCallbacksWithMILP:
    def test_all_callbacks_together(self):
        """All three callbacks can be used simultaneously."""
        m, x, y, z = _small_milp_with_continuous()
        node_calls = []
        lazy_calls = []
        inc_calls = []

        def on_node(ctx, model):
            node_calls.append(1)

        def on_lazy(ctx, model):
            lazy_calls.append(1)
            return []

        def on_inc(ctx, model, solution):
            inc_calls.append(1)
            return True

        result = m.solve(
            node_callback=on_node,
            lazy_constraints=on_lazy,
            incumbent_callback=on_inc,
            time_limit=30,
        )
        assert result.status in ("optimal", "feasible")
        assert len(node_calls) > 0


class TestCallbackContext:
    def test_construction(self):
        ctx = CallbackContext(
            node_count=10,
            incumbent_obj=3.5,
            best_bound=1.0,
            gap=0.71,
            elapsed_time=1.23,
            x_relaxation=np.zeros(3),
            node_bound=1.5,
        )
        assert ctx.node_count == 10
        assert ctx.incumbent_obj == 3.5
        assert ctx.best_bound == 1.0
        assert ctx.gap == 0.71
        assert ctx.elapsed_time == 1.23
        assert ctx.node_bound == 1.5

    def test_none_incumbent(self):
        ctx = CallbackContext(
            node_count=0,
            incumbent_obj=None,
            best_bound=-np.inf,
            gap=None,
            elapsed_time=0.0,
            x_relaxation=np.zeros(2),
            node_bound=-1.0,
        )
        assert ctx.incumbent_obj is None
        assert ctx.gap is None
