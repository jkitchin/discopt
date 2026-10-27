"""#1941 item 4A: carry a root cut pool wider than the incremental structure.

The root pool is captured on the cold build, which lifts more columns than the
incremental fast path (its RLT products). On a Rosenbrock model the pool is 12
columns wide and the structure 7, so every pool-inheriting node used to cold-build.
``DISCOPT_INCREMENTAL_POOL_EXTEND`` appends the missing product / even-power columns
with their per-node envelopes (``PoolColumnExtension``) so the fast path serves those
nodes. These pin that the envelope rows are valid (no lifted feasible point is cut),
that unsupported columns refuse the extension, and that the switch fires end to end
with the same certified optimum.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.incremental_mccormick import PoolColumnExtension

pytestmark = pytest.mark.smoke


class _Inc:
    """The two attributes ``PoolColumnExtension.build`` reads."""

    def __init__(self, n, idents):
        self.n = n
        self.ncol = len(idents)
        self.col_identities = tuple(idents)


_BASE = [("orig", 0), ("orig", 1), ("orig", 2)]
_ROOT = _BASE + [
    ("bilinear", (0, 2)),
    ("univariate_square", (("orig", 2), 2)),
    ("monomial", (1, 4)),
]


def _pool_using(cols, width):
    a = np.zeros((1, width))
    a[0, cols] = 1.0
    return a


def test_extension_rows_never_cut_a_lifted_feasible_point():
    ext = PoolColumnExtension.build(_Inc(3, _BASE), _pool_using([3, 4, 5], 6), tuple(_ROOT))
    assert ext is not None and len(ext.terms) == 3
    rng = np.random.default_rng(1941)
    checked = 0
    for _ in range(200):
        lo = rng.uniform(-5.0, 2.0, 3)
        hi = lo + rng.uniform(0.0, 4.0, 3)
        A0 = np.zeros((0, 3))
        A, b, bounds = ext.extend(A0, np.zeros(0), np.column_stack([lo, hi]), lo, hi)
        assert A.shape == (4 + 4 + 4, 6)
        for _ in range(20):
            x = rng.uniform(lo, hi)
            z = np.concatenate([x, [x[0] * x[2], x[2] ** 2, x[1] ** 4]])
            assert np.all(A @ z <= b + 1e-9 * (1.0 + np.abs(b)))
            assert np.all(z >= bounds[:, 0] - 1e-9) and np.all(z <= bounds[:, 1] + 1e-9)
            checked += 1
    assert checked == 4000


@pytest.mark.parametrize(
    "key",
    [("opaque", 3), ("monomial", (1, 3)), ("bilinear", (0, 7)), ("trilinear", (0, 1, 2))],
)
def test_unsupported_column_refuses_the_extension(key):
    root = tuple(_BASE + [key])
    assert PoolColumnExtension.build(_Inc(3, _BASE), _pool_using([0, 3], 4), root) is None


def test_columns_the_structure_has_are_not_added():
    root = tuple(_BASE + [("bilinear", (0, 2))])
    inc = _Inc(3, root)
    ext = PoolColumnExtension.build(inc, _pool_using([0, 3], 4), root)
    assert ext is not None and ext.terms == []


def _rosenbrock():
    m = dm.Model("rb")
    x0 = m.continuous("x0", lb=-10, ub=5)
    x1 = m.continuous("x1", lb=-10, ub=10)
    m.minimize(100 * (x1 - x0**2) ** 2 + (1 - x0) ** 2)
    return m


@pytest.mark.parametrize("on", ["0", "1"])
def test_switch_fires_and_keeps_the_bound_sound(monkeypatch, on):
    # #2015: the native kernel now takes this model; the test is about the Python tree.
    monkeypatch.setenv("DISCOPT_NATIVE_SPATIAL_KERNEL", "0")
    monkeypatch.setenv("DISCOPT_INCREMENTAL_POOL_EXTEND", on)
    # A short time limit, not ``max_nodes``: a node cap sends this model to the
    # native kernel, which has no McCormick pool. The switch fires at the first
    # pool-inheriting node, so the limit only bounds the test's cost.
    r = _rosenbrock().solve(time_limit=8, gap_tolerance=1e-4)
    assert r.bound is not None and r.bound <= 1e-9  # the optimum is 0
    extended = r.solver_stats.get("pool/extended_nodes", 0)
    if on == "1":
        assert extended > 0, "the wider pool was never carried"
    else:
        assert extended == 0
