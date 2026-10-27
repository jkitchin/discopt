"""#2015: a lifted model constraint over a product's own columns is a fixed row, not
part of the product's envelope (``DISCOPT_KERNEL_BLF_FULL_SUPPORT``).

``x*y == 2`` lifts to ``w <= 2`` and ``-w <= -2``, rows whose support lies inside
the product's columns, so #861's containment test counted them as the product's
envelope ("claims 6 rows") and declined the whole spec (ex8_1_7). On the probe box
every McCormick row has a nonzero coefficient on each factor, so a row with a
smaller support is not one; under the flag it stays a fixed row.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.spatial_producer import build_spatial_kernel_spec


def _model():
    """min x + 2y  s.t.  x*y == 2, x in [-1, 4], y in [0.5, 4]  ->  x = 2, y = 1, objective 4.

    ``x`` may be negative so the model is not a posynomial program (the GP route
    would take it); ``x*y == 2`` with ``y > 0`` still forces ``x > 0``."""
    m = dm.Model("blf_full_support")
    x = m.continuous("x", lb=-1.0, ub=4.0)
    y = m.continuous("y", lb=0.5, ub=4.0)
    m.subject_to(x * y == 2.0)
    m.minimize(x + 2.0 * y)
    return m


@pytest.mark.smoke
def test_the_constraint_rows_stay_fixed_rows(monkeypatch):
    monkeypatch.setenv("DISCOPT_KERNEL_BLF_FULL_SUPPORT", "0")
    assert build_spatial_kernel_spec(_model()) is None, "the #861 decline is the opt-out"
    monkeypatch.delenv("DISCOPT_KERNEL_BLF_FULL_SUPPORT", raising=False)
    spec = build_spatial_kernel_spec(_model())
    assert spec is not None
    w = int(spec["blf_w"][0])
    ptr, cols = spec["fixed_row_ptr"], spec["fixed_cols"]
    only_w = [r for r in range(len(ptr) - 1) if set(cols[ptr[r] : ptr[r + 1]].tolist()) == {w}]
    assert len(only_w) == 2, "x*y == 2 must reach every node LP as two fixed rows"
    rhs = sorted(spec["fixed_rhs"][only_w].tolist())
    assert np.allclose(rhs, [-2.0, 2.0])


def test_the_kernel_certifies_it(monkeypatch):
    monkeypatch.delenv("DISCOPT_KERNEL_BLF_FULL_SUPPORT", raising=False)
    r = _model().solve(time_limit=20)
    assert (r.algorithm_route or "").startswith("native-spatial"), r.algorithm_route
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(4.0, abs=1e-5)
    assert r.bound <= 4.0 + 1e-6
