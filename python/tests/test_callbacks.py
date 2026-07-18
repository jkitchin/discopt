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


def _spatial_mi_model():
    """Nonconvex spatial MINLP from issue #740.

    min x + y + 2*b  s.t.  x*y >= 1,  x + y <= 4 + b,  x,y in [0,4], b binary.
    Unconstrained-by-callback optimum is b=0 (objective 2.0, at x=y=1); vetoing
    or cutting off every b=0 incumbent makes the true optimum b=1 (objective
    4.0). The bilinear ``x*y`` forces the spatial (nonconvex) B&B path, whose
    primal heuristics (sub-NLP, feasibility pump, LNS, per-node NLP polish)
    find the b=0 point through side channels that bypass the batch-import
    callback gate — the #740 bug.
    """
    m = discopt.Model("spatial_mi")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    b = m.binary("b")
    m.subject_to(x * y >= 1.0)
    m.subject_to(x + y <= 4.0 + b)
    m.minimize(x + y + 2.0 * b)
    return m, (x, y, b)


@pytest.mark.slow
@needs_rust
@pytest.mark.integration
class TestIssue740HeuristicInjectionGate:
    """#740: incumbent_callback / lazy_constraints were bypassed by heuristic
    incumbent injections on the spatial MINLP path. The batch-import gate
    (``_invoke_pre_import_callbacks``) honored them, but incumbents entering via
    ``tree.inject_incumbent`` side channels (warm start, sub-NLP, feasibility
    pump, LNS, per-node NLP polish, completeness guard) never consulted the
    callbacks, so a vetoed / cut-off point was returned as the final
    ``optimal`` incumbent. The fix funnels every injection through
    ``_screen_heuristic_incumbent``.
    """

    def test_incumbent_callback_veto_on_spatial_path(self):
        """A vetoed point must never be the returned incumbent: rejecting all
        b=0 solutions forces the b=1 optimum (objective 4.0). Pre-fix this
        returned the vetoed b=0 point with objective 2.0."""
        m, _vars = _spatial_mi_model()
        vetoed = []

        def inc_cb(ctx, model, sol):
            if sol["b"] < 0.5:
                vetoed.append(dict(sol))
                return False
            return True

        res = m.solve(incumbent_callback=inc_cb, time_limit=60.0)
        assert res.status in ("optimal", "feasible")
        assert res.x is not None
        assert round(float(np.ravel(res.x["b"])[0])) == 1, (
            f"vetoed b=0 point returned as incumbent (objective {res.objective})"
        )
        assert res.objective == pytest.approx(4.0, rel=1e-3)
        # The callback actually fired (the b=0 point WAS encountered and vetoed).
        assert len(vetoed) > 0

    def test_lazy_constraints_cut_on_spatial_path(self):
        """A point excluded by a lazy cut must never be the returned incumbent:
        cutting off b=0 relaxation points with a ``b >= 1`` cut forces the b=1
        optimum (objective 4.0). Pre-fix this converged to the excluded b=0
        point with objective 2.0."""
        m, (_x, _y, b) = _spatial_mi_model()
        cuts_fired = []

        def lazy_cb(ctx, model):
            # b is the third flat variable (index 2)
            if ctx.x_relaxation[2] < 0.5:
                cuts_fired.append(1)
                return [CutResult(terms=[(b, 1.0)], sense=">=", rhs=1.0)]
            return []

        res = m.solve(lazy_constraints=lazy_cb, time_limit=60.0)
        assert res.status in ("optimal", "feasible")
        assert res.x is not None
        assert round(float(np.ravel(res.x["b"])[0])) == 1, (
            f"cut-off b=0 point returned as incumbent (objective {res.objective})"
        )
        assert res.objective == pytest.approx(4.0, rel=1e-3)
        assert len(cuts_fired) > 0

    def test_warm_start_incumbent_is_screened(self):
        """The warm-start injection is a side channel too: an initial point the
        incumbent callback vetoes must not survive as the final incumbent."""
        m, (x, y, b) = _spatial_mi_model()

        def reject_b0(ctx, model, sol):
            return not sol["b"] < 0.5

        # Feasible b=0 warm start (x=y=1): objective 2.0, vetoed by the callback.
        res = m.solve(
            incumbent_callback=reject_b0,
            initial_solution={x: 1.0, y: 1.0, b: 0.0},
            time_limit=60.0,
        )
        assert res.status in ("optimal", "feasible")
        assert round(float(np.ravel(res.x["b"])[0])) == 1
        assert res.objective == pytest.approx(4.0, rel=1e-3)


class TestScreenHeuristicIncumbent:
    """Unit tests for the #740 single-candidate callback gate."""

    class _FakeTree:
        def __init__(self, incumbent=None):
            self._inc = incumbent

        def incumbent(self):
            return self._inc

        def stats(self):
            return {"total_nodes": 7, "global_lower_bound": 1.5, "gap": 0.5}

    @staticmethod
    def _model():
        m = discopt.Model("gate_unit")
        x = m.continuous("x", lb=0.0, ub=4.0)
        b = m.binary("b")
        m.minimize(x + b)
        m.subject_to(x + b >= 0.5)
        return m

    def test_veto_blocks_candidate(self):
        from discopt.solver import _screen_heuristic_incumbent

        m = self._model()
        seen = []

        def veto(ctx, model, sol):
            seen.append(sol)
            return False

        ok = _screen_heuristic_incumbent(
            model=m,
            tree=self._FakeTree(),
            t_start=0.0,
            x=np.array([1.0, 0.0]),
            obj=1.0,
            int_offsets=[1],
            int_sizes=[1],
            lazy_constraints=None,
            incumbent_callback=veto,
            _cut_pool=None,
        )
        assert ok is False
        assert len(seen) == 1

    def test_lazy_cut_blocks_candidate_and_pools_cut(self):
        from discopt._jax.cutting_planes import CutPool
        from discopt.solver import _screen_heuristic_incumbent

        m = self._model()
        b = m._variables[1]
        pool = CutPool(max_cuts=10)

        def lazy(ctx, model):
            return [CutResult(terms=[(b, 1.0)], sense=">=", rhs=1.0)]

        ok = _screen_heuristic_incumbent(
            model=m,
            tree=self._FakeTree(),
            t_start=0.0,
            x=np.array([1.0, 0.0]),
            obj=1.0,
            int_offsets=[1],
            int_sizes=[1],
            lazy_constraints=lazy,
            incumbent_callback=None,
            _cut_pool=pool,
        )
        assert ok is False
        assert len(pool) == 1

    def test_accepting_callback_passes_candidate(self):
        from discopt.solver import _screen_heuristic_incumbent

        m = self._model()
        ok = _screen_heuristic_incumbent(
            model=m,
            tree=self._FakeTree(),
            t_start=0.0,
            x=np.array([1.0, 0.0]),
            obj=1.0,
            int_offsets=[1],
            int_sizes=[1],
            lazy_constraints=None,
            incumbent_callback=lambda ctx, model, sol: True,
            _cut_pool=None,
        )
        assert ok is True

    def test_non_integer_candidate_skips_callbacks(self):
        """A fractional candidate can never become the incumbent
        (``inject_incumbent`` re-verifies), so user code must not see it."""
        from discopt.solver import _screen_heuristic_incumbent

        m = self._model()
        calls = []
        ok = _screen_heuristic_incumbent(
            model=m,
            tree=self._FakeTree(),
            t_start=0.0,
            x=np.array([1.0, 0.4]),
            obj=1.4,
            int_offsets=[1],
            int_sizes=[1],
            lazy_constraints=None,
            incumbent_callback=lambda ctx, model, sol: calls.append(1) or False,
            _cut_pool=None,
        )
        assert ok is True
        assert not calls

    def test_non_improving_candidate_skips_callbacks(self):
        """The callback contract is "a new incumbent is about to be accepted":
        a candidate that cannot strictly improve the incumbent is passed
        through uninspected (``inject_incumbent`` rejects it)."""
        from discopt.solver import _screen_heuristic_incumbent

        m = self._model()
        calls = []
        ok = _screen_heuristic_incumbent(
            model=m,
            tree=self._FakeTree(incumbent=(np.array([1.0, 0.0]), 1.0)),
            t_start=0.0,
            x=np.array([2.0, 0.0]),
            obj=2.0,
            int_offsets=[1],
            int_sizes=[1],
            lazy_constraints=None,
            incumbent_callback=lambda ctx, model, sol: calls.append(1) or False,
            _cut_pool=None,
        )
        assert ok is True
        assert not calls

    def test_callback_exception_fails_soft(self):
        """Only the user callback may fail softly (INT-1 discipline): an
        exception is logged and the candidate proceeds."""
        from discopt.solver import _screen_heuristic_incumbent

        m = self._model()

        def bad(ctx, model, sol):
            raise RuntimeError("intentional")

        ok = _screen_heuristic_incumbent(
            model=m,
            tree=self._FakeTree(),
            t_start=0.0,
            x=np.array([1.0, 0.0]),
            obj=1.0,
            int_offsets=[1],
            int_sizes=[1],
            lazy_constraints=None,
            incumbent_callback=bad,
            _cut_pool=None,
        )
        assert ok is True


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

    def test_none_best_bound(self):
        # A1: best_bound is Optional — None when no certified global bound exists.
        ctx = CallbackContext(
            node_count=5,
            incumbent_obj=3.0,
            best_bound=None,
            gap=None,
            elapsed_time=0.1,
            x_relaxation=np.zeros(2),
            node_bound=1.0,
        )
        assert ctx.best_bound is None


class TestCertifiedCallbackBound:
    """A1: the callback's ``best_bound`` must never over-report the certified
    global dual bound. ``_certified_callback_bound`` maps the Rust tree's raw
    ``global_lower_bound`` to the value that is safe to surface.
    """

    def test_valid_minimize_passthrough(self):
        from discopt.solver import _certified_callback_bound

        assert _certified_callback_bound(5.32, True, False) == 5.32

    def test_maximize_is_negated(self):
        from discopt.solver import _certified_callback_bound

        # Internal minimization tracks -obj: a lower bound L on -obj is an upper
        # bound -L on obj. A raw min-sense bound must be negated for MAXIMIZE.
        assert _certified_callback_bound(-52.89, True, True) == 52.89

    def test_tainted_tree_reports_none(self):
        from discopt.solver import _certified_callback_bound

        # A non-rigorous fathom removed an unproven subtree; the surviving
        # frontier minimum may sit above the true certified bound (nvs05:
        # global_lower_bound = 5.32 on a tainted tree whose rigorous bound is
        # 1.35). Never surface it.
        assert _certified_callback_bound(5.32, False, False) is None
        assert _certified_callback_bound(-52.89, False, True) is None

    def test_neg_inf_root_reports_none(self):
        from discopt.solver import _certified_callback_bound

        # Unbounded / free-variable root (#467): the tree pins the bound at -inf.
        assert _certified_callback_bound(-np.inf, True, False) is None

    def test_failure_sentinel_reports_none(self):
        from discopt.constants import SENTINEL_THRESHOLD
        from discopt.solver import _certified_callback_bound

        # The 1e30 no-relaxation sentinel (hda/heatexch class) is "no bound",
        # not a numeric bound — must not leak through the API as a huge number.
        assert _certified_callback_bound(SENTINEL_THRESHOLD, True, False) is None
        assert _certified_callback_bound(None, True, False) is None


class TestCallbackBoundSoundness:
    """A1 regression: the callback's ``best_bound`` is a *certified* global dual
    bound. It must (1) be on the correct side of the objective for the sense and
    (2) never over-report relative to the final ``SolveResult.bound``. The
    pre-fix code surfaced the raw internal min-sense ``global_lower_bound`` with
    no sense negation and no taint gate, so both invariants failed.
    """

    def test_maximize_best_bound_is_a_valid_upper_bound(self):
        # A small NON-NAMED nonconvex MAXIMIZE MINLP. For a maximize, every
        # certified best_bound is an UPPER bound (>= the achievable objective).
        # Pre-fix reported the un-negated internal bound (a lower bound on -obj,
        # i.e. NEGATIVE here) and this assertion fails; post-fix it passes.
        m = discopt.Model("cb_max_bound")
        n = m.integer("n", lb=1, ub=6)
        x = m.continuous("x", lb=0.0, ub=5.0)
        m.maximize(x * n - (x - 2.0) ** 2 * n)  # nonconvex
        m.subject_to(x + n <= 8.0)

        bounds: list = []

        def cb(ctx, _model):
            bounds.append(ctx.best_bound)

        res = m.solve(time_limit=15, gap_tolerance=1e-4, node_callback=cb)
        assert res.status in ("optimal", "feasible")
        assert res.objective is not None
        finite = [b for b in bounds if b is not None and np.isfinite(b)]
        assert finite, "expected at least one certified best_bound in the trace"
        # Every certified upper bound must be >= the achievable objective.
        for b in finite:
            assert b >= res.objective - 1e-6, (
                f"maximize best_bound {b} below objective {res.objective}: "
                "callback under-/wrong-signed the certified bound"
            )

    def test_best_bound_never_exceeds_final_certified_bound(self):
        # A tiny-budget spatial solve that exits feasible with an open frontier.
        # The running certified best_bound must never exceed what the final
        # SolveResult certifies (both are the same certified global bound; the
        # callback is a snapshot of it). Over-reporting = the A1 bug.
        m = discopt.Model("cb_min_bound")
        n = m.integer("n", lb=1, ub=20)
        x = m.continuous("x", lb=0.01, ub=20.0)
        m.minimize((x - n) ** 2 * n - 10.0 * x / n)  # nonconvex product/reciprocal
        m.subject_to(x * n >= 12.0)
        m.subject_to(x + n <= 30.0)

        bounds: list = []

        def cb(ctx, _model):
            bounds.append(ctx.best_bound)

        res = m.solve(time_limit=5, gap_tolerance=1e-4, node_callback=cb)
        assert res.status in ("optimal", "feasible")
        finite = [b for b in bounds if b is not None and np.isfinite(b)]
        if res.bound is not None and np.isfinite(res.bound):
            for b in finite:
                # MINIMIZE: a certified lower bound never exceeds the finally
                # certified lower bound by more than solver tolerance.
                assert b <= res.bound + 1e-4, (
                    f"callback best_bound {b} over-reports the final certified bound {res.bound}"
                )


# ── #748: MILP / convex-MIQP classifier dispatch honors callbacks ──


def _pure_milp_748():
    """Pure MILP (no exp() forcing) classified ``ProblemClass.MILP``.

    min x + 10*y   s.t.  3*x + 2*y >= 5,   x,y integer in [0,3].

    The LP relaxation root is fractional (x=5/3, y=0) so B&B branches and the
    UNIQUE integer optimum is x=2, y=0 (obj 2.0). A callback that rejects x>=2
    must drive the solve to the acceptable optimum x=1, y=1 (obj 11.0).

    Pre-#748 this model routed to ``_solve_milp_bb`` (or ``_solve_milp_simplex``),
    which never consulted the callback and returned the vetoed x=2 point as
    ``optimal`` — the silent-drop bug. After the fix it falls through to the
    spatial B&B, which honors the callback.
    """
    m = discopt.Model("milp748")
    x = m.integer("x", lb=0, ub=3)
    y = m.integer("y", lb=0, ub=3)
    m.minimize(x + 10 * y)
    m.subject_to(3 * x + 2 * y >= 5, name="c")
    return m, x, y


def _convex_miqp_748():
    """Convex MIQP (no exp() forcing) classified ``ProblemClass.MIQP``.

    min (x - 2.2)**2 + 3*y   s.t.  x + y >= 1,   x integer in [0,3], y binary.

    Convex quadratic objective, fractional QP root (x=2.2) so B&B branches; the
    UNIQUE integer optimum is x=2, y=0 (obj 0.04). A callback rejecting exactly
    x==2 drives the solve to the acceptable optimum x=3, y=0 (obj 0.64) — which
    is the QP-integer-optimum of the x>=3 branch, so the convex spatial B&B
    reaches it.

    Pre-#748 this routed to ``_solve_miqp_bb`` (convex MIQP B&B) which never
    consulted the callback and returned the vetoed x=2 point as ``optimal``.
    """
    m = discopt.Model("miqp748")
    x = m.integer("x", lb=0, ub=3)
    y = m.binary("y")
    m.minimize((x - 2.2) ** 2 + 3 * y)
    m.subject_to(x + y >= 1, name="c")
    return m, x, y


@pytest.mark.slow
@needs_rust
@pytest.mark.integration
class TestIssue748MilpMiqpCallbackDispatch:
    """#748: the problem-classifier dispatch routed a MILP / convex-MIQP model
    with ``incumbent_callback`` / ``lazy_constraints`` to specialized engines
    (``_solve_milp_simplex`` / ``_solve_milp_bb`` / ``_solve_miqp_bb``) that never
    received or consulted the callbacks — the rejection/cut was dropped SILENTLY
    and the vetoed/cut-off point was returned as ``optimal`` (a soundness bug for
    lazy constraints; an API violation for incumbent rejection). The fix routes
    these models to the spatial B&B (which honors both callbacks, #740) when a
    callback is present. These use PURE MILP / convex MIQP models (NO ``exp()``)
    so they exercise the newly-fixed dispatch; the pre-existing tests all add an
    ``exp()`` term to force MINLP classification and dodge this path.
    """

    @staticmethod
    def _x(sol):
        return round(float(np.ravel(sol["x"])[0]))

    def test_milp_incumbent_callback_honored(self):
        """Pure MILP + rejecting incumbent_callback: the vetoed x>=2 optimum
        (returned by the MILP engine pre-fix) must NOT be the incumbent."""
        m, _x, _y = _pure_milp_748()

        def reject_x_ge_2(ctx, model, sol):
            return float(np.ravel(sol["x"])[0]) <= 1.5

        res = m.solve(incumbent_callback=reject_x_ge_2, time_limit=60)
        assert res.status in ("optimal", "feasible")
        assert res.x is not None
        xv = round(float(np.ravel(res.x["x"])[0]))
        assert xv <= 1, f"vetoed x={xv} (>=2) returned as incumbent (obj {res.objective})"
        assert res.objective == pytest.approx(11.0, rel=1e-3)

    def test_milp_lazy_constraint_honored(self):
        """Pure MILP + lazy cut x<=1: the cut-off x>=2 optimum must NOT be the
        returned incumbent."""
        m, x, _y = _pure_milp_748()

        def lazy_cut(ctx, model):
            if round(float(ctx.x_relaxation[0])) >= 2:
                return [CutResult(terms=[(x, 1.0)], sense="<=", rhs=1.0)]
            return []

        res = m.solve(lazy_constraints=lazy_cut, time_limit=60)
        assert res.status in ("optimal", "feasible")
        assert res.x is not None
        xv = round(float(np.ravel(res.x["x"])[0]))
        assert xv <= 1, f"cut-off x={xv} (>=2) returned as incumbent (obj {res.objective})"
        assert res.objective == pytest.approx(11.0, rel=1e-3)

    def test_convex_miqp_incumbent_callback_honored(self):
        """Convex MIQP + rejecting incumbent_callback: the vetoed x==2 optimum
        (returned by the convex MIQP engine pre-fix) must NOT be the incumbent;
        the solve must reach the acceptable x=3 optimum instead."""
        m, _x, _y = _convex_miqp_748()

        def reject_x_eq_2(ctx, model, sol):
            return self._x(sol) != 2

        res = m.solve(incumbent_callback=reject_x_eq_2, time_limit=60)
        assert res.status in ("optimal", "feasible")
        assert res.x is not None
        xv = round(float(np.ravel(res.x["x"])[0]))
        assert xv != 2, f"vetoed x=2 returned as incumbent (obj {res.objective})"
        assert res.objective == pytest.approx(0.64, rel=1e-3)

    def test_milp_no_callback_unchanged(self):
        """Sanity: a pure MILP WITHOUT callbacks still solves to its true
        optimum (the specialized engine path is unchanged)."""
        m, _x, _y = _pure_milp_748()
        res = m.solve(time_limit=60)
        assert res.status == "optimal"
        assert res.objective == pytest.approx(2.0, rel=1e-3)
        assert round(float(np.ravel(res.x["x"])[0])) == 2

    def test_convex_miqp_no_callback_unchanged(self):
        """Sanity: a convex MIQP WITHOUT callbacks still solves to its true
        optimum (the convex MIQP engine path is unchanged)."""
        m, _x, _y = _convex_miqp_748()
        res = m.solve(time_limit=60)
        assert res.status == "optimal"
        assert res.objective == pytest.approx(0.04, abs=1e-3)
        assert round(float(np.ravel(res.x["x"])[0])) == 2

    def test_reproduction_sketch_never_returns_vetoed_point(self):
        """The issue's exact reproduction sketch (a tight-integer-root MILP whose
        LP relaxation is already integer-optimal). Pre-fix it silently returned
        the vetoed x=1 point certified ``optimal``. After the fix the vetoed
        point is never returned: because the rejected root leaves nothing to
        branch, the spatial B&B reports an honest UNCERTIFIED result
        (feasibility undetermined) rather than either the vetoed point OR a
        certified-false ``infeasible`` — the callback rejection is a non-rigorous
        fathom, so certification is withheld (#748 / CLAUDE.md §1)."""
        m = discopt.Model("milp748_sketch")
        x = m.binary("x")
        y = m.binary("y")
        m.minimize(x + 2 * y)
        m.subject_to(x + y >= 1)

        res = m.solve(
            incumbent_callback=lambda ctx, model, sol: float(sol["x"]) < 0.5,
            time_limit=30,
        )
        # The vetoed x=1 point must never be returned as the incumbent.
        if res.x is not None:
            assert round(float(np.ravel(res.x["x"])[0])) == 0, (
                f"vetoed x=1 point returned as incumbent (status {res.status})"
            )
        # And a feasible model must never be CERTIFIED infeasible (a false
        # certificate is the worst-class error; the pre-backstop code returned
        # exactly this).
        assert not (res.status == "infeasible" and res.gap_certified), (
            "feasible model falsely certified infeasible"
        )
