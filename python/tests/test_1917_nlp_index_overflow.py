"""#1917: a GDP sub-NLP must keep its sparse structure, and an NLP whose KKT
triplet count overflows the solver's 32-bit index must be refused in Python.

GDPlib ``stranded_gas`` under ``gdp_method='ric'`` died with a POUNCE FERAL
``PanicException`` (``assertion left == right``: 2553513213 vs the same count
wrapped through i32). The GDP ``_BoundsProxy`` did not forward
``has_sparse_structure``, so ``_IpoptCallbacks`` fell back to a dense Jacobian
(m*n) and dense Hessian triangle (n(n+1)/2): 2.55e9 triplets for a problem with
374k structural nonzeros. A ``PanicException`` is a ``BaseException``, so no
``except Exception`` in the caller saw it.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._tape_nlp_evaluator import make_evaluator
from discopt.solvers.gdpopt_loa import _BoundsProxy, _solve_nlp_subproblem
from discopt.solvers.nlp_ipopt import (
    _INT32_MAX,
    NLPIndexOverflowError,
    _check_kkt_index_capacity,
    _IpoptCallbacks,
)

pytestmark = pytest.mark.smoke


def _sparse_model():
    m = dm.Model("sparse")
    x = m.continuous("x", shape=(6,), lb=0.1, ub=4.0)
    m.minimize(dm.sum([(x[i] - 1.0) ** 2 for i in range(6)]))
    for i in range(5):
        m.subject_to(x[i] * x[i + 1] <= 3.0)
    return m


def test_bounds_proxy_keeps_sparse_structure():
    ev = make_evaluator(_sparse_model())
    assert ev.has_sparse_structure()
    lb, ub = ev.variable_bounds
    cb = _IpoptCallbacks(_BoundsProxy(ev, np.asarray(lb) - 0.05, np.asarray(ub) + 0.05))
    assert cb._use_sparse, "proxy dropped sparse structure: dense fallback would be used"
    r, c = cb.jacobianstructure()
    er, ec = ev.jacobian_structure()
    np.testing.assert_array_equal(r, er)
    np.testing.assert_array_equal(c, ec)
    assert len(r) < ev.n_constraints * ev.n_variables  # genuinely sparse
    hr, _ = cb.hessianstructure()
    assert len(hr) < ev.n_variables * (ev.n_variables + 1) // 2
    x = np.full(ev.n_variables, 1.5)
    lam = np.ones(ev.n_constraints)
    np.testing.assert_allclose(cb.jacobian(x), ev.evaluate_jacobian_values(x))
    np.testing.assert_allclose(cb.hessian(x, lam, 1.0), ev.evaluate_hessian_values(x, 1.0, lam))


class _HugeDenseEvaluator:
    """Dense-only evaluator the size of stranded_gas's lowered model. Nothing of
    size m*n may be built: any structure/derivative call fails the test."""

    n_variables = 57828
    n_constraints = 15241
    timing_bucket = "tape"
    variable_bounds = (np.zeros(57828), np.ones(57828))

    # Row senses, read by solve_nlp's constraint-bound inference (O(m), no m*n).
    _source_constraints = [type("_Row", (), {"sense": "<="})()]
    _constraint_flat_sizes = [15241]

    def has_sparse_structure(self):
        return False

    def __getattr__(self, name):
        raise AssertionError(f"{name} called on an NLP that must be refused first")


def test_dense_fallback_over_int32_is_refused():
    with pytest.raises(NLPIndexOverflowError, match="32-bit"):
        _IpoptCallbacks(_HugeDenseEvaluator())


def test_gdp_subproblem_propagates_refusal():
    ev = _HugeDenseEvaluator()
    lb, ub = ev.variable_bounds
    with pytest.raises(NLPIndexOverflowError):
        _solve_nlp_subproblem(ev, lb.copy(), ub.copy(), "pounce")


def test_capacity_boundary():
    # Exactly at the limit is accepted; one past it is refused.
    n, m = 10, 5
    room = _INT32_MAX - (n + 3 * m)
    _check_kkt_index_capacity(n, m, room, 0, dense=False)
    with pytest.raises(NLPIndexOverflowError):
        _check_kkt_index_capacity(n, m, room + 1, 0, dense=False)
