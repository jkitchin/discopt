"""#2225: the path separator does not spin on an entry the aggregator hands back.

``LpAggregator::current_aggregation`` keeps a dropped entry whose bound on that side is
infinite (the fac2 soundness fix). A column cancelled by its substitution row then came
back with a roundoff coefficient, was substituted again, and the path loop never
advanced. HiGHS never sees such an entry; the path now skips any entry at most
``small_matrix_value``.
"""

from __future__ import annotations

import pytest

_PATH_LOOP = r"""
import sys, time, numpy as np, scipy.sparse as sp
from discopt._iis_backend import make_backend
z = np.load(sys.argv[1])
A = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
b = make_backend("rust", A, z["rlo"], z["rhi"], z["lb"], z["ub"], z["is_int"])
t = time.perf_counter()
st = b.run(sys.argv[2], t + 5.0)
print(st, time.perf_counter() - t)
"""


@pytest.mark.parametrize("presolve", ["choose", "off"])
def test_the_path_separator_does_not_spin_on_a_dropped_entry(presolve):
    """A MIPcc27 ``dell`` IIS probe (``data/iis_2225_dell_path_loop.npz``: the model and
    the probe's rows, bounds and integrality) on which the root's node separation spun in
    the path separator: a column whose aggregated coefficient was roundoff, handed back
    by ``current_aggregation``, was substituted again and again without the path
    advancing. Measured before the fix: over 120 s against a 5 s budget, either presolve
    setting; after: Infeasible in ~2.5 s. Run in a subprocess so a regression fails on
    the timeout instead of hanging the suite."""
    import subprocess
    import sys
    from pathlib import Path

    data = Path(__file__).parent / "data" / "iis_2225_dell_path_loop.npz"
    out = subprocess.run(
        [sys.executable, "-c", _PATH_LOOP, str(data), presolve],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    status, wall = out.stdout.split()
    assert status == "infeasible", out.stdout
    assert float(wall) < 5.0 + 1.0, f"probe overran its 5 s budget: {wall} s"
