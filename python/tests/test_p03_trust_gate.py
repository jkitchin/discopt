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
    def test_non_kkt_convex_is_untrusted(self):
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
