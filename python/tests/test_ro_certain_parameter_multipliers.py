"""A certain Parameter multiplying an uncertain one must stay robustifiable.

#1839 (merged here for #1831) froze a constant-valued multiplier that held a
certain :class:`Parameter` into a :class:`Constant`. A second
``RobustCounterpart`` pass over that parameter -- what the error messages
recommend for mixed uncertainty -- then found nothing to robustify:
``x + q*p <= 10`` with a box on ``p`` and then on ``q`` certified ``x = 7``
where the robust optimum is ``5.5``, and a ``q.value`` changed after
``formulate()`` was silently ignored. The Parameter now stays symbolic and the
sign its direction was read from is a recorded assumption, checked by a later
pass over its set and by ``Model.validate`` before every solve.

Every optimum is compared with a grid oracle over the joint box of ``(p, q)``.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.ro import (
    BoxUncertaintySet,
    EllipsoidalUncertaintySet,
    PolyhedralUncertaintySet,
    RobustCounterpart,
)
from discopt.ro.formulations._common import RobustSignAssumptionError

pytestmark = pytest.mark.smoke

P_BAR, P_DELTA = 2.0, 1.0
Q_BAR, Q_DELTA = 1.0, 0.5


def _first_set(kind, p):
    if kind == "box":
        return BoxUncertaintySet(p, delta=P_DELTA)
    if kind == "polyhedral":
        return PolyhedralUncertaintySet(p, A=np.array([[1.0], [-1.0]]), b=np.array([1.0, 1.0]))
    return EllipsoidalUncertaintySet(p, rho=P_DELTA)


# (name, row(x, p, q) for ``row <= 10``, numeric twin f(p, q) of the uncertain part)
ROWS = [
    ("q_times_p", lambda x, p, q: x + q * p, lambda p, q: q * p),
    ("p_times_q", lambda x, p, q: x + p * q, lambda p, q: p * q),
    ("neg_q_times_p", lambda x, p, q: x - q * p, lambda p, q: -q * p),
    ("two_q_times_p", lambda x, p, q: x + (2.0 * q) * p, lambda p, q: 2.0 * q * p),
    ("p_over_q", lambda x, p, q: x + p / q, lambda p, q: p / q),
]


def _oracle(f):
    ps = np.linspace(P_BAR - P_DELTA, P_BAR + P_DELTA, 401)
    qs = np.linspace(Q_BAR - Q_DELTA, Q_BAR + Q_DELTA, 401)
    P, Q = np.meshgrid(ps, qs)
    return 10.0 - float(np.max(f(P, Q)))


@pytest.mark.parametrize("order", ["pq", "qp"])
@pytest.mark.parametrize("name,row,f", ROWS, ids=[r[0] for r in ROWS])
def test_sequential_box_passes_match_joint_oracle(order, name, row, f):
    m = dm.Model(name)
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=P_BAR)
    q = m.parameter("q", value=Q_BAR)
    m.subject_to(row(x, p, q) <= 10)
    m.maximize(x)
    sets = {"p": BoxUncertaintySet(p, delta=P_DELTA), "q": BoxUncertaintySet(q, delta=Q_DELTA)}
    if name == "p_over_q" and order == "qp":
        # q first: the certain p sits over an uncertain divisor (not monotone in
        # q with a known direction) and cannot be kept symbolic -- refused.
        with pytest.raises(NotImplementedError, match="certain Parameter"):
            RobustCounterpart(m, sets["q"]).formulate()
        return
    for key in order:
        RobustCounterpart(m, sets[key]).formulate()
    r = m.solve()
    assert r.status == "optimal"
    assert float(r.value(x)) == pytest.approx(_oracle(f), abs=1e-4)


def test_polyhedral_dual_then_pass_on_q_refused():
    """The dual equality ``q = lam1 - lam2`` holds ``q``: one ``lam`` for every ``q``
    would be a false infeasibility, so a later pass over ``q`` is refused."""
    m = dm.Model("poly")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=P_BAR)
    q = m.parameter("q", value=Q_BAR)
    m.subject_to(x + q * p <= 10)
    m.maximize(x)
    RobustCounterpart(m, _first_set("polyhedral", p)).formulate()
    q.value = 1.5  # a value change is fine for the dual: it is exact for any q
    r = m.solve()
    assert r.status == "optimal"
    assert float(r.value(x)) == pytest.approx(10.0 - 1.5 * 3.0, abs=1e-4)
    with pytest.raises(RobustSignAssumptionError, match="certain"):
        RobustCounterpart(m, BoxUncertaintySet(q, delta=Q_DELTA)).formulate()


@pytest.mark.parametrize("kind", ["box", "ellipsoidal"])
def test_first_pass_any_kind_then_box_on_q(kind):
    """The first pass over ``p`` keeps ``q`` symbolic."""
    m = dm.Model(kind)
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=P_BAR)
    q = m.parameter("q", value=Q_BAR)
    m.subject_to(x + q * p <= 10)
    m.maximize(x)
    RobustCounterpart(m, _first_set(kind, p)).formulate()
    RobustCounterpart(m, BoxUncertaintySet(q, delta=Q_DELTA)).formulate()
    r = m.solve()
    assert r.status == "optimal"
    assert float(r.value(x)) == pytest.approx(10.0 - 1.5 * 3.0, abs=1e-4)


def test_vector_mixed_sign_parameter_multiplier():
    """``sum(q * p)`` with ``q = (1, -1)``: split by sign, both parts symbolic."""
    m = dm.Model("vec")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=np.array([2.0, 2.0]))
    q = m.parameter("q", value=np.array([1.0, -1.0]))
    m.subject_to(x + dm.sum(q * p) <= 10)
    m.maximize(x)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=np.ones(2))).formulate()
    RobustCounterpart(m, BoxUncertaintySet(q, delta=np.array([0.5, 0.5]))).formulate()
    r = m.solve()
    assert r.status == "optimal"
    best = max(
        a * b + c * d
        for a, c in itertools.product((0.5, 1.5), (-1.5, -0.5))
        for b, d in itertools.product((1.0, 3.0), (1.0, 3.0))
    )
    assert float(r.value(x)) == pytest.approx(10.0 - best, abs=1e-4)


def test_second_pass_straddling_the_assumed_sign_is_refused():
    m = dm.Model("straddle")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=P_BAR)
    q = m.parameter("q", value=Q_BAR)
    m.subject_to(x + q * p <= 10)
    m.maximize(x)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=P_DELTA)).formulate()
    with pytest.raises(RobustSignAssumptionError):
        RobustCounterpart(m, BoxUncertaintySet(q, delta=2.0)).formulate()


def test_value_change_keeping_sign_is_honoured():
    m = dm.Model("value_same_sign")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=P_BAR)
    q = m.parameter("q", value=Q_BAR)
    m.subject_to(x + q * p <= 10)
    m.maximize(x)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=P_DELTA)).formulate()
    q.value = 2.0
    r = m.solve()
    assert r.status == "optimal"
    assert float(r.value(x)) == pytest.approx(10.0 - 2.0 * 3.0, abs=1e-4)


def test_value_change_flipping_sign_is_refused_at_solve():
    m = dm.Model("value_flip")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=P_BAR)
    q = m.parameter("q", value=Q_BAR)
    m.subject_to(x + q * p <= 10)
    m.maximize(x)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=P_DELTA)).formulate()
    q.value = -1.0
    with pytest.raises(RobustSignAssumptionError):
        m.solve()


def test_certain_parameter_under_non_monotone_node_refused():
    m = dm.Model("nonmono")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=1.0)
    q = m.parameter("q", value=1.0)
    m.subject_to(x + q / p <= 10)
    m.maximize(x)
    with pytest.raises(NotImplementedError, match="certain Parameter"):
        RobustCounterpart(m, BoxUncertaintySet(p, delta=0.5)).formulate()


def test_joint_pass_handles_the_same_rows():
    """Robustifying both parameters at once needs no assumption at all."""
    m = dm.Model("joint")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=1.0)
    q = m.parameter("q", value=1.0)
    m.subject_to(x + q / p <= 10)
    m.maximize(x)
    RobustCounterpart(
        m, [BoxUncertaintySet(p, delta=0.5), BoxUncertaintySet(q, delta=0.5)]
    ).formulate()
    r = m.solve()
    assert r.status == "optimal"
    assert float(r.value(x)) == pytest.approx(10.0 - 1.5 / 0.5, abs=1e-4)


def test_serialize_refuses_a_guarded_model():
    """The guard is not written by this format; dropping it on reload would let a
    sign-flipping ``q.value`` through, so the save is refused instead."""
    from discopt.serialize import SerializationError, dumps

    m = dm.Model("ser")
    x = m.continuous("x", lb=0, ub=20)
    p = m.parameter("p", value=P_BAR)
    q = m.parameter("q", value=Q_BAR)
    m.subject_to(x + q * p <= 10)
    m.maximize(x)
    RobustCounterpart(m, BoxUncertaintySet(p, delta=P_DELTA)).formulate()
    with pytest.raises(SerializationError, match="_ro_sign_preconditions"):
        dumps(m)
