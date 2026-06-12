"""P0.3 trusted-mask decoupling: non-KKT convex bounds decertify the gap.

For a convex model the node relaxation objective is used as the node lower
bound. An interior-point solve that stops short of KKT (codes 3/4 / Ipopt
ITERATION_LIMIT) yields ``f(x~) >= f*`` — not a valid lower bound (issue #39).
The batch solvers return a ``trusted`` mask flagging such nodes; the B&B loops
decertify the optimality gap on them instead of corrupting the bound /
incumbent (which ``-inf`` or a sentinel would). These tests pin both halves:

  - solver side: ``_solve_batch_ipm`` / ``_solve_batch_pounce`` return
    ``trusted=False`` for non-KKT convex nodes, ``True`` once converged, and
    always ``True`` for nonconvex models (objective discarded by the caller);
  - caller side: an untrusted node (batch) or an ITERATION_LIMIT node
    (serial) makes ``Model.solve`` report ``gap_certified=False`` /
    ``"feasible"`` while still returning the correct incumbent (bounds are
    left untouched).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pounce")

import discopt.modeling as dm  # noqa: E402
import discopt.solver as S  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from discopt.solver import (  # noqa: E402
    _make_evaluator,
    _solve_batch_ipm,
    _solve_batch_pounce,
)
from discopt.solvers import SolveStatus  # noqa: E402
from discopt.solvers.nlp_ipopt import _infer_constraint_bounds  # noqa: E402


def _convex_eval():
    """A convex constrained NLP evaluator plus one node box."""
    m = dm.Model("cvx")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.minimize((x - 1) ** 2 + (y - 2) ** 2)
    m.subject_to(x + y >= 1)
    ev = _make_evaluator(m)
    cl, cu = _infer_constraint_bounds(ev)
    cb = list(zip(cl.tolist(), cu.tolist()))
    return ev, cb, jnp.array(cl), jnp.array(cu)


_LB = [[-5.0, -5.0], [-5.0, -5.0]]
_UB = [[5.0, 5.0], [5.0, 5.0]]


# ---------------------------------------------------------------------------
# Solver side: the trusted mask
# ---------------------------------------------------------------------------
class TestTrustedMaskIPM:
    def test_non_kkt_convex_is_untrusted(self):
        ev, cb, gl, gu = _convex_eval()
        *_, trusted = _solve_batch_ipm(
            ev,
            _LB,
            _UB,
            [0, 1],
            ev.n_variables,
            cb,
            {"max_iter": 1},
            gl,
            gu,
            multistart=True,
            convex=True,
        )
        assert not trusted.any()  # max_iter=1 -> non-KKT, polish also capped

    def test_converged_convex_is_trusted(self):
        ev, cb, gl, gu = _convex_eval()
        *_, trusted = _solve_batch_ipm(
            ev,
            _LB,
            _UB,
            [0, 1],
            ev.n_variables,
            cb,
            {"max_iter": 200},
            gl,
            gu,
            multistart=True,
            convex=True,
        )
        assert trusted.all()

    def test_nonconvex_always_trusted(self):
        ev, cb, gl, gu = _convex_eval()
        *_, trusted = _solve_batch_ipm(
            ev,
            _LB,
            _UB,
            [0, 1],
            ev.n_variables,
            cb,
            {"max_iter": 1},
            gl,
            gu,
            multistart=True,
            convex=False,
        )
        assert trusted.all()  # objective discarded by caller -> trust irrelevant


class TestTrustedMaskPOUNCE:
    def test_stalled_convex_is_rescued_by_polish_retry(self):
        """A max_iter=1 stall is no longer just untrusted: the boosted polish
        re-solve (P0.3 polish-retry) reaches KKT, so the node keeps a valid,
        trusted bound (the true optimum here is 0)."""
        ev, cb, _, _ = _convex_eval()
        _, lbs, _, _, trusted = _solve_batch_pounce(
            ev, _LB, _UB, [0, 1], ev.n_variables, cb, {"max_iter": 1}, convex=True
        )
        assert trusted.all()
        assert np.allclose(lbs, 0.0, atol=1e-4)

    def test_unpolishable_convex_is_untrusted(self, monkeypatch):
        """When even the polish re-solve cannot reach KKT, trust is withheld."""
        from discopt.solvers import NLPResult

        def stalled(evaluator, x0, node_lb, node_ub, constraint_bounds, options, convex=False):
            return NLPResult(
                status=SolveStatus.ITERATION_LIMIT,
                x=np.asarray(x0, dtype=np.float64),
                objective=1.234,
            )

        monkeypatch.setattr(S, "_solve_node_nlp_pounce", stalled)
        ev, cb, _, _ = _convex_eval()
        *_, trusted = _solve_batch_pounce(
            ev, _LB, _UB, [0, 1], ev.n_variables, cb, {"max_iter": 1}, convex=True
        )
        assert not trusted.any()

    def test_converged_convex_is_trusted(self):
        ev, cb, _, _ = _convex_eval()
        *_, trusted = _solve_batch_pounce(
            ev, _LB, _UB, [0, 1], ev.n_variables, cb, {"max_iter": 200}, convex=True
        )
        assert trusted.all()

    def test_nonconvex_always_trusted(self):
        ev, cb, _, _ = _convex_eval()
        *_, trusted = _solve_batch_pounce(
            ev, _LB, _UB, [0, 1], ev.n_variables, cb, {"max_iter": 1}, convex=False
        )
        assert trusted.all()


# ---------------------------------------------------------------------------
# Serial-node retry: failed solves retry from alternative starts (P0.2)
# ---------------------------------------------------------------------------
class TestSerialNodeRetry:
    def _run(self, monkeypatch, outcomes):
        """Drive _solve_node_nlp_pounce with a scripted solve_nlp sequence."""
        import discopt.solvers.nlp_pounce as nlp_pounce
        from discopt.solvers import NLPResult

        ev, cb, _, _ = _convex_eval()
        starts: list[np.ndarray] = []

        def scripted(evaluator, x0, constraint_bounds=None, options=None):
            starts.append(np.asarray(x0, dtype=np.float64).copy())
            status = outcomes[min(len(starts), len(outcomes)) - 1]
            obj = 0.5 if status == SolveStatus.OPTIMAL else np.nan
            return NLPResult(status=status, x=np.asarray(x0, dtype=np.float64), objective=obj)

        monkeypatch.setattr(nlp_pounce, "solve_nlp", scripted)
        x0 = np.array([4.0, 4.0])
        res = S._solve_node_nlp_pounce(
            ev, x0, np.array([-5.0, -5.0]), np.array([5.0, 5.0]), cb, {"max_iter": 100}
        )
        return res, starts

    def test_failed_solve_retries_alternative_starts(self, monkeypatch):
        """ERROR on the warm start -> retry from the midpoint succeeds."""
        res, starts = self._run(monkeypatch, [SolveStatus.ERROR, SolveStatus.OPTIMAL])
        assert res.status == SolveStatus.OPTIMAL
        assert len(starts) == 2
        assert not np.allclose(starts[0], starts[1])  # genuinely different start

    def test_all_starts_failing_reports_failure(self, monkeypatch):
        """If every start fails, the failure is reported (3 attempts max)."""
        res, starts = self._run(
            monkeypatch, [SolveStatus.ERROR, SolveStatus.ERROR, SolveStatus.ERROR]
        )
        assert res.status == SolveStatus.ERROR
        assert len(starts) == 3  # warm + midpoint + off-center, then give up

    def test_successful_solve_does_not_retry(self, monkeypatch):
        res, starts = self._run(monkeypatch, [SolveStatus.OPTIMAL])
        assert res.status == SolveStatus.OPTIMAL
        assert len(starts) == 1


# ---------------------------------------------------------------------------
# Caller side: decertification
# ---------------------------------------------------------------------------
def _convex_minlp():
    """Convex MINLP (exp keeps it out of the MIQP class) routed to _solve_nlp_bb.

    Optimum x=1, y=2 -> exp(0.5)+0 ~ 1.6487.
    """
    m = dm.Model("cvx_minlp")
    x = m.integer("x", lb=0, ub=8)
    y = m.continuous("y", lb=0, ub=5)
    m.minimize(dm.exp(0.5 * x) + (y - 2) ** 2)
    m.subject_to(x + y >= 3)
    return m


_OPT = 1.6487212707


class TestCallerDecertifies:
    def test_control_normal_solve_certifies(self):
        """Without interference the convex MINLP certifies optimality."""
        r = _convex_minlp().solve(nlp_solver="ipm", time_limit=60, batch_size=8)
        assert r.status == "optimal"
        assert r.gap_certified is True
        assert abs(r.objective - _OPT) < 1e-3

    def test_batch_untrusted_node_decertifies(self, monkeypatch):
        """An untrusted batch node decertifies the gap but keeps the answer."""
        orig = S._solve_batch_ipm

        def wrap(*a, **k):
            ids, lbs, sols, feas, trusted = orig(*a, **k)
            trusted = np.asarray(trusted).copy()
            trusted[0] = False  # simulate an unpolishable non-KKT convex node
            return ids, lbs, sols, feas, trusted

        monkeypatch.setattr(S, "_solve_batch_ipm", wrap)
        r = _convex_minlp().solve(nlp_solver="ipm", time_limit=60, batch_size=8)
        assert r.gap_certified is False
        assert r.status == "feasible"
        # Bounds were untouched, so the incumbent is still correct.
        assert abs(r.objective - _OPT) < 1e-2

    def test_serial_iteration_limit_decertifies(self, monkeypatch):
        """A convex serial node returning ITERATION_LIMIT decertifies the gap."""
        orig = S._solve_node_nlp

        def wrap(*a, **k):
            res = orig(*a, **k)
            if res.status == SolveStatus.OPTIMAL:
                res.status = SolveStatus.ITERATION_LIMIT  # simulate non-KKT
            return res

        monkeypatch.setattr(S, "_solve_node_nlp", wrap)
        # batch_size=1 forces the serial node path (n_batch == 1).
        r = _convex_minlp().solve(nlp_solver="ipm", time_limit=60, batch_size=1)
        assert r.gap_certified is False
