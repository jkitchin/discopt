"""#1846: ``if_else`` with a ``Parameter`` branch certified a wrong optimum.

``if_else`` lowers to a hull disjunction. The linear hull row ``a^T x + b`` becomes
``a^T v_k + b * y_k``, and ``_hull_linear_substitute`` scaled ``b`` by ``y_k`` only
for a ``Constant`` -- an indexed ``Parameter`` (``p[0]``) was returned unscaled, so
the de-selected disjunct (variables pinned to 0) was forced to ``w = p[0]``: the
"else" branch became infeasible and the model below certified 10.001 instead of
11. A bare scalar ``Parameter`` takes the perspective path, whose ``g(0)``
evaluator refused it; and that evaluator read an indexed constant ``c[0]`` as
``max(c)``. A Parameter branch now solves exactly like the equal constant.
"""

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.gdp_reformulate import _body_at_zero
from discopt.modeling.core import Constant

pytestmark = pytest.mark.smoke


def _solve(m, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.solve(time_limit=60, **kw)


def _model(branch, objective_y=0.001):
    """max x + c*y s.t. x + if_else(y >= 1, branch, 0) <= 13, x + y <= 11."""
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=20)
    y = m.continuous("y", lb=0, ub=5)
    p = m.parameter("p", value=np.array([3.0, 1.0]))
    q = m.parameter("q", value=3.0)
    w = dm.if_else(y >= 1, branch(p, q, y), 0)
    m.subject_to(x + w <= 13)
    m.subject_to(x + y <= 11)
    m.maximize(x + objective_y * y)
    return m, p, q


BRANCHES = {
    "p[0]": (lambda p, q, y: p[0], lambda c, y: c),
    "q": (lambda p, q, y: q, lambda c, y: c),
    "2*q - p[1]": (lambda p, q, y: 2 * q - p[1], lambda c, y: 2 * c - 1.0),
    "p[0]*y**2": (lambda p, q, y: p[0] * y**2, lambda c, y: c * y**2),
    "q + exp(y) - 1": (lambda p, q, y: q + dm.exp(y) - 1, lambda c, y: c + dm.exp(y) - 1),
}


@pytest.mark.parametrize("name", sorted(BRANCHES))
@pytest.mark.parametrize("method", [None, "big-m", "hull"])
def test_parameter_branch_matches_constant(name, method):
    par, const = BRANCHES[name]
    kw = {} if method is None else {"gdp_method": method}
    m_p, _, _ = _model(par)
    m_c, _, _ = _model(lambda p, q, y: const(3.0, y))
    r_p, r_c = _solve(m_p, **kw), _solve(m_c, **kw)
    assert r_c.status == "optimal"
    assert r_p.status == "optimal"
    assert r_p.objective == pytest.approx(r_c.objective, abs=1e-6)
    if name == "p[0]":
        assert r_p.objective == pytest.approx(11.0, abs=1e-6)  # the issue's truth


def test_resolve_after_parameter_change():
    # max x + 2y: the "then" branch (y >= 1) gives x <= 13 - p0; the else branch
    # (y <= 1) caps the objective at 12 (x=10, y=1).
    m, p, _ = _model(lambda p, q, y: p[0], objective_y=2.0)
    for p0, truth in ((3.0, 16.0), (12.0, 12.0)):
        p.value = np.array([p0, 1.0])
        r = _solve(m)
        assert r.status == "optimal"
        assert r.objective == pytest.approx(truth, abs=1e-6), p0


def test_body_at_zero_reads_the_indexed_entry():
    m = dm.Model()
    x = m.continuous("x", lb=-1, ub=1)
    p = m.parameter("p", value=np.array([2.0, 7.0]))
    c = Constant(np.array([1.0, 5.0]))
    vars_ = {"x": x}
    assert _body_at_zero(dm.exp(x) + c[0], vars_) == pytest.approx(2.0)  # was 1 + max(c)
    assert _body_at_zero(dm.exp(x) + p[1], vars_) == pytest.approx(8.0)  # was a refusal
