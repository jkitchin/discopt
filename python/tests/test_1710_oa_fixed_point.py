"""#1710: OA stops at a fixed point instead of re-solving one fixed NLP 51 times.

Measured before the fix on #1708's 2-variable MINLP under ``solver="mip-nlp"``:
every OA iteration added zero cuts, so the master returned the same assignment,
OA re-solved the same fixed-integer NLP from the same warm start, and the loop
only ended at the 50-iteration no-progress exit -- 51 bit-identical fixed-NLP
solves and 51 master solves for one answer. Two changes, both bound-neutral:

* the loop stops as soon as an iteration's full start state (every master input
  and the loop state the iteration reads) repeats the previous iteration's, with
  the same ``"stalling"`` reason the 50-iteration exit gave;
* a converged fixed-NLP solve is reused for a bit-identical later request.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solvers.oa as oa
import numpy as np
import pytest
from discopt.solvers import SolveStatus


def _issue_model():
    def f(u):
        return np.array([u[0] * np.exp(-0.1 * u[1]), u[0] + u[1] ** 2])

    def jac(u):
        e = np.exp(-0.1 * u[1])
        return np.array([[e, -0.1 * u[0] * e], [1.0, 2 * u[1]]])

    def hess(u):
        e = np.exp(-0.1 * u[1])
        return np.array([[[0, -0.1 * e], [-0.1 * e, 0.01 * u[0] * e]], [[0, 0], [0, 2.0]]])

    # memoize=False: the fixed-NLP count asserted below must not lean on #1709.
    blk = dm.external(f, jac=jac, hess=hess, shape=(2,), name="unit", memoize=False)
    m = dm.Model("toy")
    u = m.continuous("u", shape=(2,), lb=[1.0, 0.0], ub=[5.0, 3.0])
    on = m.binary("on")
    y = blk(u)
    ye = m.continuous("ye", shape=(2,), lb=0.0, ub=20.0)
    ylo, yhi = np.array([0.5, 1.0]), np.array([5.0, 14.0])
    for k in range(2):  # ye = on * y, exact for binary on
        m.subject_to(ye[k] <= yhi[k] * on)
        m.subject_to(ye[k] >= ylo[k] * on)
        m.subject_to(ye[k] <= y[k] - ylo[k] * (1 - on))
        m.subject_to(ye[k] >= y[k] - yhi[k] * (1 - on))
    m.subject_to(ye[1] <= 6.0)
    m.maximize(3 * ye[0] - 0.5 * ye[1] - 1.0 * on)
    return m


@pytest.fixture
def fixed_nlp_calls(monkeypatch):
    calls = []
    real = oa._solve_fixed_nlp_subproblem_attempt

    def counting(*args, **kwargs):
        attempt = real(*args, **kwargs)
        calls.append(attempt)
        return attempt

    monkeypatch.setattr(oa, "_solve_fixed_nlp_subproblem_attempt", counting)
    return calls


def test_issue_model_solves_its_one_fixed_nlp_once(fixed_nlp_calls):
    m = _issue_model()
    r = m.solve(time_limit=60, solver="mip-nlp")

    # Same answer as before the fix (objective 11.5, bound 13.5, uncertified).
    assert r.objective == pytest.approx(11.5, abs=1e-6)
    assert r.bound == pytest.approx(13.5, abs=1e-6)
    assert r.status == "feasible"

    assert len(fixed_nlp_calls) == 1  # was 51
    trace = r.mip_nlp_trace
    assert trace["termination_reason"] == "stalling"  # unchanged reason
    summary = trace["summary"]
    assert summary["fixed_nlp_call_count"] == 1
    assert summary["fixed_nlp_cache_hit_count"] >= 1
    iterations = trace["iterations"]
    assert iterations[-1].get("fixed_point") is True
    assert iterations[-1]["master_status"] == "not_run"
    assert r.mip_count <= 3  # was 51


def test_cycling_check_keeps_its_own_exit(fixed_nlp_calls):
    """The fixed-point exit stands aside when ``cycling_check`` is on."""
    r = oa.solve_oa(_issue_model(), time_limit=60, cycling_check=True)
    assert r.mip_nlp_trace["termination_reason"] == "cycling"
    assert not any(it.get("fixed_point") for it in r.mip_nlp_trace["iterations"])


def _attempt(x, status):
    return oa._NLPAttempt(
        x=None if x is None else np.asarray(x, dtype=np.float64),
        objective=None if x is None else 1.0,
        multipliers=None,
        status=status,
    )


def test_only_converged_attempts_are_cacheable():
    assert oa._fixed_nlp_attempt_cacheable(_attempt([1.0], SolveStatus.OPTIMAL))
    # A failure, or a point accepted at a resource limit, can depend on the
    # shrinking wall budget, so it is not a property of the subproblem.
    assert not oa._fixed_nlp_attempt_cacheable(_attempt(None, SolveStatus.ERROR))
    assert not oa._fixed_nlp_attempt_cacheable(_attempt([1.0], SolveStatus.TIME_LIMIT))
    assert not oa._fixed_nlp_attempt_cacheable(_attempt([1.0], None))


def test_cached_attempt_is_not_aliased():
    a = _attempt([1.0, 2.0], SolveStatus.OPTIMAL)
    b = oa._copy_nlp_attempt(a)
    b.x[0] = 99.0
    assert a.x[0] == 1.0


def test_state_fingerprint_is_exact():
    fp = oa._state_fingerprint
    checked = 0
    for left, right in [
        (np.array([0.0]), np.array([-0.0])),  # bitwise, so a repeat is never invented
        (np.array([1.0]), np.array([1.0 + 1e-16 * 2])),
        (np.array([1, 2], dtype=np.int64), np.array([1, 2], dtype=np.int32)),
        (np.zeros((2, 1)), np.zeros((1, 2))),
        ([1.0, 2.0], [[1.0], 2.0]),
        (None, 0.0),
        ("a", "b"),
        (True, 1),
    ]:
        assert fp(left) != fp(right), (left, right)
        checked += 1
    assert fp(np.arange(3.0), {1, 2}) == fp(np.arange(3.0), {2, 1})
    assert fp(frozenset({(1.0,), (2.0,)})) == fp(frozenset({(2.0,), (1.0,)}))
    assert checked == 8
    with pytest.raises(TypeError):
        fp(object())
    with pytest.raises(TypeError):
        fp(np.array([object()]))
