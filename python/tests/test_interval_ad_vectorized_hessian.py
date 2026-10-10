"""The interval-AD Hessian is carried as arrays, not one scalar interval per entry.

``test_1544_wide_reconstruct.py::test_shallow_model_with_wide_sum_certifies`` took
13-20 s locally and hit its 60 s limit on CI three times: ``sin(sum(x)/100) * y``
over 600 ``x`` reaches the box-curvature certificate, whose interval Hessian has
``601*602/2`` entries, and the sparse walker held each one as its own scalar
:class:`Interval` -- ~1.1 M Python-level interval multiplies per solve. The Hessian
map is now an :class:`_HMap` of parallel arrays driven by element-wise kernels.

The change is bound-neutral by construction: each kernel reproduces the scalar
operator **bit for bit, entry by entry** (pinned below on an adversarial grid --
zeros, signed zeros, infinities, underflowing products, subnormals), so every
Hessian is identical to the one the dict walker produced.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
from discopt._relax.convexity import interval_ad as iad
from discopt._relax.convexity.interval import Interval

_VALUES = [
    -np.inf,
    -1e200,
    -3.0,
    -1.0,
    -1e-160,
    -5e-324,
    -0.0,
    0.0,
    5e-324,
    1e-170,
    1e-160,
    0.5,
    2.0,
    1e200,
    np.inf,
]
_IVS = [(lo, hi) for lo, hi in itertools.product(_VALUES, _VALUES) if lo <= hi]


def _bits(x) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64)).view(np.int64)


def _same(scalar: Interval, lo, hi) -> bool:
    return bool(
        np.array_equal(_bits(scalar.lo), _bits(lo)) and np.array_equal(_bits(scalar.hi), _bits(hi))
    )


def test_elementwise_kernels_are_bit_identical_to_scalar_interval_ops():
    pairs = list(itertools.product(_IVS, _IVS))
    alo = np.array([p[0][0] for p in pairs])
    ahi = np.array([p[0][1] for p in pairs])
    blo = np.array([p[1][0] for p in pairs])
    bhi = np.array([p[1][1] for p in pairs])
    with np.errstate(all="ignore"):
        kernels = {
            "mul": iad._v_mul(alo, ahi, blo, bhi),
            "add": iad._v_add(alo, ahi, blo, bhi),
            "sub": iad._v_sub(alo, ahi, blo, bhi),
        }
        compared = 0
        for k, ((al, ah), (bl, bh)) in enumerate(pairs):
            a, b = Interval(al, ah), Interval(bl, bh)
            for name, want in (("mul", a * b), ("add", a + b), ("sub", a - b)):
                lo, hi = kernels[name]
                assert _same(want, lo[k], hi[k]), (name, a, b, lo[k], hi[k])
                compared += 1
        sq_lo, sq_hi = iad._v_sq(np.array([p[0] for p in _IVS]), np.array([p[1] for p in _IVS]))
        for k, (lo, hi) in enumerate(_IVS):
            assert _same(Interval(lo, hi) ** 2, sq_lo[k], sq_hi[k]), (lo, hi)
            compared += 1
    assert compared == 3 * len(pairs) + len(_IVS)


def test_mixed_array_keeps_the_per_entry_exact_zero_rule():
    """One underflowing entry must not nudge another entry's exact zero (#957).

    ``Interval.__mul__`` on an array decides the #957 rule once for the whole
    array; the sparse walker multiplied entries one at a time, so the kernel must
    decide per entry. Entry 0 underflows (1e-200 * 1e-200); entry 1 is an exact
    ``[0, 0] * [-3, 7]``, which must stay exactly ``[0, 0]``.
    """
    lo, hi = iad._v_mul(
        np.array([1e-200, 0.0]),
        np.array([1e-200, 0.0]),
        np.array([1e-200, -3.0]),
        np.array([1e-200, 7.0]),
    )
    assert lo[0] < 0.0 < hi[0]  # underflow: nudged outward to enclose 1e-400
    assert _bits(lo[1]) == _bits(0.0) and _bits(hi[1]) == _bits(0.0)


def _wide(n: int):
    m = dm.Model("wide_sin")
    x = m.continuous("x", shape=(n,), lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    xs = [x[i] for i in range(n)]
    return m, xs, y, dm.sin(dm.sum(xs) / 100.0) * y + y


def test_wide_outer_product_does_no_per_entry_scalar_work(monkeypatch):
    """Scalar interval multiplies stay O(n), not O(n^2), on a wide ``sin(sum)``.

    Deterministic stand-in for the CI timeout: the dict walker did ~0.54 M scalar
    multiplies per Hessian here (n = 600); the gradient chain rule alone needs a
    few per variable.
    """
    n = 600
    m, _, _, body = _wide(n)
    calls = 0
    real_mul = Interval.__mul__

    def counting_mul(self, other):
        nonlocal calls
        calls += 1
        return real_mul(self, other)

    monkeypatch.setattr(Interval, "__mul__", counting_mul)
    ad = iad.interval_hessian(body, m)
    assert ad.hess.lo.shape == (n + 1, n + 1)
    assert calls > 0  # the probe saw the walk
    assert calls < 20 * n, calls


def test_wide_hessian_encloses_the_true_hessian():
    """Soundness on the regression class: enclose ``∇²f`` at sampled box points.

    ``f = y (1 + sin(s/100))``, ``s = Σ x``: ``f_xi_xj = -y sin(s/100)/1e4``,
    ``f_xi_y = cos(s/100)/100``, ``f_yy = 0``.
    """
    n = 40
    m, xs, y, body = _wide(n)
    ad = iad.interval_hessian(body, m)
    h_lo, h_hi = np.asarray(ad.hess.lo), np.asarray(ad.hess.hi)
    assert np.array_equal(h_lo, h_lo.T) and np.array_equal(h_hi, h_hi.T)
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(50):
        xv = rng.uniform(0.0, 1.0, n)
        yv = rng.uniform(0.0, 1.0)
        s = xv.sum() / 100.0
        true = np.zeros((n + 1, n + 1))
        true[:n, :n] = -yv * np.sin(s) / 1e4
        true[:n, n] = true[n, :n] = np.cos(s) / 100.0
        assert np.all(h_lo <= true) and np.all(true <= h_hi)
        checked += 1
    assert checked == 50
