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


def _nonlinear_knapsack():
    """MINLP knapsack whose lazy callback actually fires at branch nodes.

    max 4*x0 + 5*x1 + 6*x2 + 7*x3  s.t.  3*x0+4*x1+5*x2+6*x3 <= 9,  xi binary,
    plus a genuine nonlinear term exp(z) (z>=0) to force the MINLP / NLP-BB
    route. The LP relaxation is fractional, so B&B branches and integer-feasible
    nodes reach the lazy-constraint callback (unlike a convex model whose primal
    heuristics find the optimum at the root and never invoke the callback). The
    unconstrained optimum is x=(0,1,1,0) (value 11); a lazy cut x1+x2<=1 excludes
    it, making the true optimum x=(1,0,0,1) (value 11 as well but a distinct
    point, so accepting the excluded (0,1,1,0) is observably wrong).
    """
    m = discopt.Model("nl_knap")
    n = 4
    xs = [m.binary(f"x{i}") for i in range(n)]
    z = m.continuous("z", lb=0.0, ub=1.0)
    v = [4, 5, 6, 7]
    w = [3, 4, 5, 6]
    m.minimize(-sum(v[i] * xs[i] for i in range(n)) + discopt.exp(z))
    m.subject_to(sum(w[i] * xs[i] for i in range(n)) <= 9, name="cap")
    m.subject_to(z >= 0.0, name="zpos")
    return m, xs


def _reject_0110_cut(xs):
    """Lazy callback: exclude the integer point x=(0,1,1,0) via x1+x2<=1."""

    def cb(ctx, model):
        xr = [int(round(ctx.x_relaxation[i])) for i in range(4)]
        if xr == [0, 1, 1, 0]:
            return [CutResult(terms=[(xs[1], 1.0), (xs[2], 1.0)], sense="<=", rhs=1.0)]
        return []

    return cb


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
@pytest.mark.slow
@pytest.mark.integration
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
@pytest.mark.slow
@pytest.mark.integration
class TestLazyConstraints:
    def test_lazy_cut_excludes_solution(self):
        """Lazy constraints exclude the otherwise-optimal integer point.

        Uses the nonlinear-knapsack fixture whose LP relaxation is fractional so
        B&B branches and the lazy callback actually fires at integer-feasible
        nodes (a convex model's primal heuristics find the optimum at the root
        and never invoke the callback). The cut x1+x2<=1 excludes (0,1,1,0), so
        the returned solution must not be that point.
        """
        m, xs = _nonlinear_knapsack()
        result = m.solve(lazy_constraints=_reject_0110_cut(xs), time_limit=30)
        if result.status in ("optimal", "feasible"):
            assert result.x is not None
            sol = [round(float(result.x[f"x{i}"])) for i in range(4)]
            assert sol != [0, 1, 1, 0]  # the excluded point must not be accepted

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
@pytest.mark.integration
class TestInt1NlpBbLazyRejection:
    """INT-1 (#413): on the NLP-BB path a lazy-constraint / incumbent-callback
    rejection was silently swallowed (``_cut_pool`` was ``None``, so
    ``_cut_pool.add`` raised an ``AttributeError`` caught by a broad ``except``
    that ALSO dropped the node-rejection line that followed it), so the excluded
    integer-feasible point was accepted as the incumbent — a wrong optimum.

    The fix refuses these callbacks loudly on the NLP-BB path (the path cannot
    enforce them: no per-node cut application and heuristics inject incumbents
    without consulting the callback), and routes them to spatial B&B where the
    rejection is honored. It also reorders ``_invoke_pre_import_callbacks`` so the
    node-rejection precedes any fallible cut insertion and narrows the ``except``
    so a programming error can no longer be swallowed.
    """

    def test_nlp_bb_true_with_lazy_constraints_refuses_loudly(self):
        """Explicit ``nlp_bb=True`` + ``lazy_constraints`` must raise, not
        silently drop the rejection and return the excluded point."""
        m, xs = _nonlinear_knapsack()
        with pytest.raises(ValueError, match="nlp_bb=True"):
            m.solve(nlp_bb=True, lazy_constraints=_reject_0110_cut(xs), time_limit=30)

    def test_nlp_bb_true_with_incumbent_callback_refuses_loudly(self):
        """Explicit ``nlp_bb=True`` + ``incumbent_callback`` must raise: the
        NLP-BB path cannot honor an incumbent rejection either."""
        m, xs = _nonlinear_knapsack()
        with pytest.raises(ValueError, match="nlp_bb=True"):
            m.solve(
                nlp_bb=True,
                incumbent_callback=lambda ctx, model, sol: True,
                time_limit=30,
            )

    def test_spatial_path_honors_lazy_rejection(self):
        """``nlp_bb=False`` (spatial B&B) must EXCLUDE the rejected point — the
        pre-fix NLP-BB bug returned exactly this excluded point."""
        m, xs = _nonlinear_knapsack()
        result = m.solve(nlp_bb=False, lazy_constraints=_reject_0110_cut(xs), time_limit=30)
        assert result.status in ("optimal", "feasible")
        sol = [round(float(result.x[f"x{i}"])) for i in range(4)]
        assert sol != [0, 1, 1, 0], f"rejected point accepted as incumbent: {sol}"

    def test_auto_select_with_lazy_routes_to_spatial_and_honors(self):
        """Auto-select (no ``nlp_bb``) with a lazy callback must fall through to
        spatial B&B (not the callback-blind NLP-BB auto path) and exclude the
        rejected point."""
        m, xs = _nonlinear_knapsack()
        result = m.solve(lazy_constraints=_reject_0110_cut(xs), time_limit=30)
        assert result.status in ("optimal", "feasible")
        sol = [round(float(result.x[f"x{i}"])) for i in range(4)]
        assert sol != [0, 1, 1, 0], f"rejected point accepted as incumbent: {sol}"

    def test_nlp_bb_true_without_callbacks_still_works(self):
        """The refusal must be narrow: ``nlp_bb=True`` without any rejecting
        callback still solves normally."""
        m, xs = _nonlinear_knapsack()
        result = m.solve(nlp_bb=True, time_limit=30)
        assert result.status in ("optimal", "feasible")
        assert result.objective is not None


@pytest.mark.slow
@needs_rust
@pytest.mark.slow
@pytest.mark.integration
class TestIncumbentCallback:
    def test_incumbent_rejection(self):
        """Incumbent callback can reject an integer-feasible point.

        Uses the nonlinear-knapsack fixture (fractional relaxation → the callback
        actually fires at branch nodes). Reject the point x=(0,1,1,0); the
        returned solution must therefore differ from it.
        """
        m, xs = _nonlinear_knapsack()

        def reject_0110(ctx, model, solution):
            xr = [round(float(np.ravel(solution[f"x{i}"])[0])) for i in range(4)]
            return xr != [0, 1, 1, 0]

        result = m.solve(incumbent_callback=reject_0110, time_limit=30)
        if result.status in ("optimal", "feasible"):
            assert result.x is not None
            sol = [round(float(result.x[f"x{i}"])) for i in range(4)]
            assert sol != [0, 1, 1, 0]

    def test_accept_all_is_noop(self):
        """Accepting all incumbents should behave identically to no callback."""
        m, x, y = _simple_milp()

        def accept_all(ctx, model, solution):
            return True

        result = m.solve(incumbent_callback=accept_all, time_limit=30)
        assert result.status in ("optimal", "feasible")


@pytest.mark.slow
@needs_rust
@pytest.mark.slow
@pytest.mark.integration
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
@pytest.mark.slow
@pytest.mark.integration
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
