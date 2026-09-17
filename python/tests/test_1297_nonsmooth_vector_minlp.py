"""#1297: a feasible convex MINLP with a zero-slack elementwise ``max`` row.

``v_i >= max(s, 1)`` written as ``max(max(v, s), 1) - v <= 0`` (and three other
spellings) with an integer ``s``. The optimum is ``n*max(s, 1) - 1.5 s``: 0.5, 1.5 and
2.5 for n = 2, 3, 4. Before the fix the solve returned ``infeasible`` for n >= 3, and
two defects remained under that:

* The OA/LOA/AMP decompositions indexed the convexity mask and the senses per
  ``Constraint`` object, but cut generation reads them per evaluator row, so any
  array-valued constraint ran ``convex_mask[k]`` past its end (IndexError).
* Every gradient route (OA, NLP-BB, the spatial loop's convex mode) trusts a smooth
  NLP at the ``max`` kink, where it stalls. Such models now take the spatial B&B's
  exact envelopes, as nonsmooth continuous models already did.
"""

import logging

import discopt.modeling as dm
import pytest
from discopt.solver import _convex_minlp_auto_route
from discopt.solvers.oa import _decompose_model, _oa_rows

OPT = {2: 0.5, 3: 1.5, 4: 2.5}
SPELLINGS = ("nested", "nary", "max_le_v", "scalar_rows")


def _max_model(n, spelling="nested"):
    m = dm.Model("i1297")
    v = m.continuous("v", shape=(n,), lb=0, ub=4)
    s = m.integer("s", lb=0, ub=3)
    m.minimize(v.sum() - 1.5 * s)
    if spelling == "nested":
        m.subject_to(dm.maximum(dm.maximum(v, s), 1.0) - v <= 0)
    elif spelling == "nary":
        m.subject_to(dm.maximum(v, s, 1.0) - v <= 0)
    elif spelling == "max_le_v":
        m.subject_to(dm.maximum(v, s, 1.0) <= v)
    else:
        for i in range(n):
            m.subject_to(dm.maximum(v[i], s, 1.0) - v[i] <= 0)
    return m


def _vector_convex_minlp():
    """Smooth convex MINLP whose only nonlinear row is array-valued."""
    m = dm.Model("vec")
    x = m.continuous("x", shape=(3,), lb=-3, ub=3)
    y = m.continuous("y", lb=-10, ub=10)
    z = m.binary("z")
    m.subject_to(x**2 <= 4)
    m.subject_to(y**2 <= 1)
    m.subject_to(x[0] >= -3 + 2 * z)
    m.minimize(x.sum() + y + z)
    return m


def _slack(v):
    return 1e-6 * (1 + abs(v))


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("spelling", SPELLINGS)
def test_default_solve_certifies_the_optimum(n, spelling):
    r = _max_model(n, spelling).solve(time_limit=60)
    assert r.status == "optimal", (r.status, r.objective, r.bound)
    assert r.gap_certified
    assert r.objective == pytest.approx(OPT[n], abs=1e-5)
    assert r.bound <= OPT[n] + _slack(OPT[n])
    assert r.bound >= OPT[n] - 1e-4
    v = r.x["v"]
    s = float(r.x["s"])
    assert abs(s - round(s)) <= 1e-6
    assert all(vi >= max(s, 1.0) - 1e-6 for vi in v)


def test_nonsmooth_minlp_is_not_auto_routed():
    method, reason, _ = _convex_minlp_auto_route(_max_model(3))
    assert method is None
    assert "nonsmooth" in reason


def test_smooth_convex_minlp_is_still_auto_routed():
    method, reason, _ = _convex_minlp_auto_route(_vector_convex_minlp())
    assert method == "oa", reason


def test_oa_rows_are_per_evaluator_row():
    m = _vector_convex_minlp()
    d = _decompose_model(m)
    assert d.n_cons == 5  # 3 rows of x**2 <= 4, then y**2 <= 1, then the linear row
    assert len(d.oa_constraint_mask) == d.n_cons
    assert len(d.constraint_senses) == d.n_cons
    assert d.oa_constraint_mask == [True] * 5
    assert d.constraint_senses == ["<="] * 5
    assert d.nonlinear_indices == [0, 1, 2, 3]
    assert len(d.linear_A_rows) == 1
    assert not d.oa_has_unclassified_constraints


def test_oa_rows_keeps_a_nonconvex_vector_row_nonconvex():
    m = dm.Model("mixed")
    x = m.continuous("x", shape=(2,), lb=-1, ub=1)
    y = m.continuous("y", lb=-1, ub=1)
    m.subject_to(y**2 <= 1)
    m.subject_to(-(x**2) <= -0.25)  # nonconvex, two rows
    m.subject_to(x.sum() + y <= 1)
    m.minimize(y)
    d = _decompose_model(m)
    assert d.oa_constraint_mask == [True, False, False, True]
    # Row 3 (``x.sum() + y``) is linear; the extractor declines the vector ``sum``,
    # which only keeps it out of the master's linear block.
    assert d.nonlinear_indices[:3] == [0, 1, 2]
    assert len(d.constraint_senses) == d.n_cons == 4


def test_oa_rows_rejects_a_mask_of_the_wrong_length():
    m = _vector_convex_minlp()
    with pytest.raises(ValueError, match="constraint_mask"):
        _oa_rows(_decompose_model(m).evaluator, m, [True])


def test_explicit_mip_nlp_runs_on_a_vector_row():
    # Raised IndexError out of generate_oa_cuts_from_evaluator before the fix.
    r = _vector_convex_minlp().solve(solver="mip-nlp", time_limit=30)
    assert r.status == "optimal", r.status
    assert r.objective == pytest.approx(-7.0, abs=1e-5)


def test_loa_runs_on_a_vector_row():
    m = dm.Model("loa")
    x = m.continuous("x", shape=(3,), lb=-3, ub=3)
    y = m.continuous("y", lb=-10, ub=10)
    m.subject_to(x**2 <= 4)
    m.subject_to(y**2 <= 1)
    m.either_or([[x[0] >= 1], [x[0] <= -1]])
    m.minimize(x.sum() + y)
    r = m.solve(gdp_method="loa", time_limit=30)
    assert r.status == "optimal", r.status
    assert r.objective == pytest.approx(-7.0, abs=1e-5)


def test_amp_direct_oa_runs_on_a_vector_row(caplog):
    m = dm.Model("amp")
    x = m.continuous("x", shape=(3,), lb=-2, ub=2)
    y = m.continuous("y", lb=-2, ub=2)
    z = m.binary("z")
    m.subject_to(x**2 <= 1)
    m.subject_to(x[0] * y >= -1 + z)  # nonconvex, so AMP has work to do
    m.minimize(x.sum() + y - z)
    with caplog.at_level(logging.DEBUG, logger="discopt.solvers.amp"):
        r = m.solve(solver="amp", time_limit=30)
    assert r.status in ("optimal", "feasible"), r.status
    failures = [
        rec.getMessage()
        for rec in caplog.records
        if "OA cut computation failed" in rec.getMessage()
    ]
    assert not failures, failures


def test_model_constraint_bounds_are_per_evaluator_row():
    # The Model form counted one row per Constraint object (2 here, not 4), so AMP's
    # feasibility check raised a broadcast error and rejected every point.
    from discopt._tape_nlp_evaluator import make_evaluator
    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    m = dm.Model("rows")
    x = m.continuous("x", shape=(3,), lb=-2, ub=2)
    y = m.continuous("y", lb=-2, ub=2)
    m.subject_to(x**2 <= 1)
    m.subject_to(x[0] + y == 0)
    m.minimize(x.sum() + y)
    cl, cu = _infer_constraint_bounds(m)
    assert cl.size == cu.size == make_evaluator(m).n_constraints == 4
    assert list(cu) == [0.0] * 4
    assert list(cl) == [-1e20] * 3 + [0.0]
