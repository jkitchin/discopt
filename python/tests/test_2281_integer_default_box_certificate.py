"""#2281 -- no certificate may rest on the default integer ``ub = 1e6``.

``Model.integer`` fills an unwritten ``ub`` with the finite fallback ``1e6`` (C-6
made that loud with a declaration-time ``UserWarning``). #1678 (b) withheld any
certificate whose point sits on the continuous default box (``±9.999e19``) but did
not recognise the integer fallback, so ``min -x - y`` s.t. ``x - 2y >= -3``
(``x >= 0`` integer, no ``ub``) -- unbounded below -- was published ``optimal`` with
``gap_certified=True`` at ``x = 1e6`` by the HiGHS MILP route, and the nonconvex
``min -x*z - z**2`` by the native spatial kernel. Both are now ``feasible`` with no
dual bound, and a warning names the variable.

A ``1e6`` the user *wrote* is honoured as posed, and a model whose answer does not
touch the default side is unaffected.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest

pytestmark = pytest.mark.smoke

_MATCH = "sits on the default variable bound"


def _solve(m):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = m.solve(time_limit=20)
    return r, [str(w.message) for w in caught]


def _milp():
    m = dm.Model("u1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = m.integer("x", lb=0)
    y = m.continuous("y", lb=0, ub=1)
    m.minimize(-x - y)
    m.subject_to(x - 2 * y >= -3)
    return m


def _nonconvex():
    m = dm.Model("u1b")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = m.integer("x", lb=0)
    z = m.continuous("z", lb=-1, ub=1)
    m.minimize(-x * z - z**2)
    m.subject_to(z >= 0.5)
    return m


def _indexed():
    m = dm.Model("u1i")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = m.integer("x", shape=3, lb=0)
    m.minimize(-x[1])
    m.subject_to(x[0] + x[2] <= 4)
    return m


@pytest.mark.parametrize("build", [_milp, _nonconvex, _indexed])
def test_point_on_default_integer_ub_is_not_certified(build):
    r, msgs = _solve(build())
    assert r.status == "feasible", r.status
    assert r.gap_certified is False
    assert r.bound is None
    assert r.solver_stats["certificate/default_box_withheld"] == 1.0
    assert any(_MATCH in s and "ub=1e+06" in s for s in msgs), msgs


def test_point_on_default_integer_lb_is_not_certified():
    """The C-6 reproduction: ``min n`` with ``n`` integer and no bounds is
    unbounded, but the default ``lb = 0`` made it ``optimal`` at ``n = 0``."""
    m = dm.Model("c6")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n = m.integer("n")
    m.minimize(n)
    r, msgs = _solve(m)
    assert r.status == "feasible" and r.gap_certified is False and r.bound is None
    assert any(_MATCH in s and "lb=0" in s for s in msgs), msgs


def test_bound_assigned_after_declaration_is_honoured():
    m = dm.Model("later")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = m.integer("x", lb=0)
    x.ub = 10
    m.minimize(-x)
    r, msgs = _solve(m)
    assert r.status == "optimal" and r.gap_certified is True
    assert r.objective == pytest.approx(-10)
    assert not any(_MATCH in s for s in msgs)


def test_clone_keeps_the_default_side_record():
    r, _ = _solve(_indexed().clone())
    assert r.status == "feasible" and r.gap_certified is False and r.bound is None


def test_explicit_1e6_is_honoured_as_posed():
    m = dm.Model("explicit")
    x = m.integer("x", lb=0, ub=1e6)
    m.minimize(-x)
    r, msgs = _solve(m)
    assert r.status == "optimal" and r.gap_certified is True
    assert r.objective == pytest.approx(-1e6)
    assert not any(_MATCH in s for s in msgs)


def test_interior_answer_with_default_ub_keeps_certificate():
    m = dm.Model("interior")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = m.integer("x", lb=0)
    m.minimize((x - 3.4) ** 2)
    r, msgs = _solve(m)
    assert r.status == "optimal" and r.gap_certified is True
    assert r.objective == pytest.approx(0.16, abs=1e-6)
    assert not any(_MATCH in s for s in msgs)
