"""#1723: ``compute_iis`` must not leak the large-bound warning from its own probes.

The deletion filter drops a candidate bound by setting it to the ``±1e20``
sentinel and re-solving. Each such probe used to warn "Variables with very large
or infinite declared bounds: y[0] (lb=-1e+20, ...)" about a model whose every
declared bound was finite.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest

pytestmark = pytest.mark.smoke

_LARGE_BOUND = "very large or infinite declared bounds"


def _iis_with_warnings(m):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        iis = m.compute_iis()
    return iis, [str(w.message) for w in caught if _LARGE_BOUND in str(w.message)]


def test_issue_repro_vector_variable_no_warning():
    m = dm.Model("inf")
    y = m.continuous("y", shape=(2,), lb=0, ub=10)
    m.subject_to(y[0] + y[1] >= 5, name="a")
    m.subject_to(y[0] + y[1] <= 3, name="b")
    m.minimize(y[0])
    iis, msgs = _iis_with_warnings(m)
    assert sorted(c.name for c in iis.constraints) == ["a", "b"]
    assert iis.n_solves > 1  # the probes that used to warn did run
    assert msgs == []


def test_bound_driven_iis_no_warning_and_bounds_restored():
    m = dm.Model("bnd")
    x = m.continuous("x", lb=5, ub=8)
    z = m.integer("z", shape=(3,), lb=0, ub=4)
    m.subject_to(x + z[0] <= 2, name="cap")
    m.minimize(x + z.sum())
    iis, msgs = _iis_with_warnings(m)
    assert msgs == []
    assert [c.name for c in iis.constraints] == ["cap"]
    assert any(v is x and side == "lower" for v, side in iis.variable_bounds)
    # The probes' sentinel bounds never outlive the probe.
    assert float(x.lb) == 5 and float(x.ub) == 8
    assert list(z.lb) == [0, 0, 0] and list(z.ub) == [4, 4, 4]


def test_user_declared_free_variable_is_still_reported():
    # Only the bounds a probe relaxed are excused: ``z`` is declared with no
    # bounds, so the warning about it is about the user's model and must stay.
    m = dm.Model("free")
    y = m.continuous("y", shape=(2,), lb=0, ub=10)
    z = m.continuous("z")
    m.subject_to(y[0] + y[1] >= 5, name="a")
    m.subject_to(y[0] + y[1] <= 3, name="b")
    m.subject_to(z >= y[0], name="c")
    m.minimize(y[0])
    _, msgs = _iis_with_warnings(m)
    assert msgs, "the warning about the user's own free variable was suppressed"
    assert all("z (lb=" in s and "y[" not in s for s in msgs)


def test_plain_solve_still_warns_after_compute_iis():
    # The marker is scoped to the probe: a later ordinary solve of a model with a
    # genuinely huge bound still warns.
    m = dm.Model("after")
    y = m.continuous("y", shape=(2,), lb=0, ub=10)
    m.subject_to(y[0] + y[1] >= 5, name="a")
    m.subject_to(y[0] + y[1] <= 3, name="b")
    m.minimize(y[0])
    m.compute_iis()
    m2 = dm.Model("huge")
    w = m2.continuous("w", lb=-1e18, ub=1)
    m2.minimize(w * w)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m2.solve()
    assert any(_LARGE_BOUND in str(c.message) for c in caught)
