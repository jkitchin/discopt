"""The convexity certificate builds a row's interval Hessian over its support.

``certify_convex`` densified each constraint's interval Hessian to ``n x n`` over
the whole model. On GDPlib biofuel lowered by big-M (36804 columns) that is
10.8 GB per bound for a row of a few variables; once the sparse linear context
(same PR) stopped eating the solve's budget first, the convexity walk reached
the certificate and the solve was killed for memory. ``compact=True`` builds the
Hessian over the row's support. The rows and columns it leaves out are exact
zeros, whose Gershgorin bound (0 against a slack of 0) decides nothing, so the
verdict is the same: 12594 verdicts on 87 models, identical to the n x n form.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity import certificate
from discopt._relax.convexity.certificate import certify_convex
from discopt._relax.convexity.interval_ad import interval_hessian

pytestmark = pytest.mark.smoke


def _wide(n_extra: int = 200):
    m = dm.Model("wide")
    pad = m.continuous("pad", lb=-1, ub=1, shape=(n_extra,))
    x = m.continuous("x", lb=0.5, ub=3)
    y = m.continuous("y", lb=-2, ub=2)
    z = m.continuous("z", lb=0.1, ub=4)
    m.minimize(dm.sum(pad))
    return m, x, y, z, pad


def test_compact_spans_the_support_only():
    m, x, y, z, _ = _wide()
    ad = interval_hessian(x**2 + x * y + dm.exp(z), m, compact=True)
    n = sum(int(v.size) for v in m._variables)
    assert ad.support is not None and ad.support.tolist() == [n - 3, n - 2, n - 1]
    assert ad.hess.lo.shape == (3, 3) and ad.grad.lo.shape == (3,)
    full = interval_hessian(x**2 + x * y + dm.exp(z), m)
    idx = np.ix_(ad.support, ad.support)
    np.testing.assert_array_equal(ad.hess.lo, full.hess.lo[idx])
    np.testing.assert_array_equal(ad.hess.hi, full.hess.hi[idx])
    np.testing.assert_array_equal(ad.grad.lo, full.grad.lo[ad.support])


def test_certify_convex_never_builds_the_model_width_matrix(monkeypatch):
    m, x, y, z, _ = _wide(2000)
    shapes = []
    real = certificate.interval_hessian

    def spy(*a, **k):
        ad = real(*a, **k)
        shapes.append(ad.hess.lo.shape)
        return ad

    monkeypatch.setattr(certificate, "interval_hessian", spy)
    assert certify_convex(x**2 + dm.exp(z), m) is not None
    assert shapes == [(2, 2)]


def _exprs(x, y, z, pad):
    return [
        x**2 + y**2,
        -(x**2) - z**2,
        x * y,
        dm.exp(z) + x**2,
        dm.log(z) + x,
        x**3,
        (x - y) ** 2,
        y**2 * 1e12 - x * 1e-12,
        pad[0] ** 2 + x,
        z * 0 + x,  # affine: an all-zero Hessian
        dm.sqrt(z),
    ]


def _two():
    m = dm.Model("two")
    x = m.continuous("x", lb=0.5, ub=3)
    y = m.continuous("y", lb=-2, ub=2)
    m.minimize(x + y)
    return m, [x**2 + y**2, -(x**2), x * y, x**2 + 0.1 * x * y + y**2, dm.exp(x), x + y, x**3]


def _verdicts_agree(monkeypatch, m, exprs) -> int:
    real = certificate.interval_hessian

    def full_width(ex, mo, box=None, compact=False):
        return real(ex, mo, box)

    decided = 0
    for e in exprs:
        compact = certify_convex(e, m)
        monkeypatch.setattr(certificate, "interval_hessian", full_width)
        full = certify_convex(e, m)
        monkeypatch.setattr(certificate, "interval_hessian", real)
        assert compact == full, e
        decided += compact is not None
    return decided


def test_the_verdict_is_the_full_forms_on_two_variables(monkeypatch):
    """The 2x2 Sylvester test applies here, with and without compaction."""
    m, exprs = _two()
    assert 0 < _verdicts_agree(monkeypatch, m, exprs) < len(exprs)


@pytest.mark.parametrize("n_extra", [1, 50])
def test_the_verdict_is_the_full_forms(monkeypatch, n_extra):
    m, x, y, z, pad = _wide(n_extra)
    exprs = _exprs(x, y, z, pad)
    assert 0 < _verdicts_agree(monkeypatch, m, exprs) < len(exprs)


def test_two_variable_model_keeps_the_sylvester_test():
    for lo_x, lo_y in itertools.product([0.5, 1.0], [0.5, 2.0]):
        m = dm.Model("two")
        x = m.continuous("x", lb=lo_x, ub=3)
        y = m.continuous("y", lb=lo_y, ub=3)
        m.minimize(x + y)
        assert certify_convex(x**2 + y**2 + 0.1 * x * y, m) is not None
