"""#1917: the native spatial kernel's in-node OBBT never crosses a box, and never panics.

Measured on GDPlib ``cstr`` (big-M lowering, the kernel spec saved in
``data/spatial_spec_cstr_bigm_1917.npz``): with ``run_obbt=True`` the tree panicked
within 30 nodes with ``min > max, or either was NaN. min = 1.0, max =
0.9999999999999996``. The OBBT probe took its tightened edge from the LP *vertex
objective*, which drifted an ulp inside the true maximum of a binary whose node box
was ``[1, 1]``; the intersected box crossed, and ``f64::clamp`` aborted. Each OBBT
edge is now the probe's Neumaier--Shcherbina safe bound, and a box crossed within
round-off is widened outward rather than collapsed or handed to ``clamp``.

``max_nodes=15`` is the smallest node limit at which the pre-fix build panics
(measured against main 54c7f944: 10 nodes clean, 15 nodes panic). 60 nodes cost
62 s on CI, over the 15 s fast-lane budget; 15 nodes still fail before the fix.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

R = pytest.importorskip("discopt._rust")

_SPEC = Path(__file__).parent / "data" / "spatial_spec_cstr_bigm_1917.npz"
# Global optimum of GDPlib cstr (MINLPLib-style oracle used by the #1917 panel).
_CSTR_OPT = 3.0620145717261655


def _spec() -> dict:
    with np.load(_SPEC) as z:
        kw = {k: z[k] for k in z.files if k != "__scalars__"}
        kw.update(json.loads(str(z["__scalars__"])))
    return kw


@pytest.mark.parametrize("run_obbt", [False, True])
def test_cstr_kernel_with_obbt_runs_and_bounds_soundly(run_obbt):
    kw = _spec()
    kw.update(max_nodes=15, time_limit_s=1e6, bound_time_extension_s=None, run_obbt=run_obbt)
    r = R.solve_spatial_tree_py(**kw)
    assert r["node_count"] == 15
    assert r["status"] == "node_limit"
    # The bound is a lower bound on the known optimum: OBBT tightened nothing away.
    assert np.isfinite(r["bound"])
    assert r["bound"] <= _CSTR_OPT + 1e-6
