"""``x[-1]`` must name ``x``'s last element in the convex kernel and the interval Hessian.

Two more sites of the #1952 index pattern (``offset + int(idx)``), found by the
adjacent-bug sweep:

* ``_convex_kernel._col_of`` decomposed ``exp(x[-1])`` onto the variable declared
  before ``x``; the kernel certified ``~0`` on a model whose optimum is ``-10``.
* ``interval_ad._impl`` put ``x[-1]``'s gradient and Hessian on that same slot, so
  ``x[-1]*z`` got the Hessian of ``z**2`` and was proved convex; the model went to
  a single local NLP and certified a non-global point.
"""

from __future__ import annotations

import discopt
import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity.interval_ad import interval_hessian


def _kernel_model(neg):
    m = discopt.Model("ck_neg")
    z = m.continuous("z", lb=0, ub=10)  # declared before x
    x = m.continuous("x", shape=(2,), lb=0, ub=10)
    y = m.binary("y", shape=(2,))
    last = x[-1] if neg else x[1]
    m.subject_to(dm.exp(last) <= 20 * y[0] + 1)
    m.subject_to(x[0] + y[1] <= 3)
    m.minimize(-z + 4 * y[0] + y[1])
    return m


@pytest.mark.smoke
@pytest.mark.parametrize("neg", [False, True], ids=["x[1]", "x[-1]"])
def test_convex_kernel_resolves_negative_index(neg):
    r = _kernel_model(neg).solve(time_limit=30, threads=2)
    assert r.bound <= -10 + 1e-6, (r.status, r.objective, r.bound, r.algorithm_route)
    assert r.objective == pytest.approx(-10.0, abs=1e-6)


@pytest.mark.smoke
def test_interval_hessian_puts_negative_index_on_its_own_slot():
    m = discopt.Model("iad")
    z = m.continuous("z", lb=-1, ub=1)
    x = m.continuous("x", shape=(2,), lb=-1, ub=1)
    want = interval_hessian(x[1] * z, m)
    got = interval_hessian(x[-1] * z, m)
    for a, b in ((want.hess.lo, got.hess.lo), (want.hess.hi, got.hess.hi)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # the bilinear z*x[1] entry (outward-rounded enclosure of 1); z's diagonal is 0
    assert np.asarray(got.hess.lo)[0, 2] <= 1.0 <= np.asarray(got.hess.hi)[0, 2]
    assert np.asarray(got.hess.hi)[0, 0] == 0.0


@pytest.mark.smoke
def test_bilinear_in_negative_index_is_not_proved_convex():
    m = discopt.Model("iad_solve")
    z = m.continuous("z", lb=-1, ub=3)
    x = m.continuous("x", shape=(2,), lb=-2, ub=1)
    a = x[-1]
    m.subject_to(x[0] + a <= 2)
    m.minimize(dm.exp(0.5 * a * z) + 0.2 * z)
    r = m.solve(time_limit=30, threads=1)
    assert "convex-nlp" not in (r.algorithm_route or "")
    assert r.bound <= 0.40653 + 1e-4, (r.status, r.objective, r.bound, r.algorithm_route)
    assert r.objective == pytest.approx(0.40653, abs=1e-4)
