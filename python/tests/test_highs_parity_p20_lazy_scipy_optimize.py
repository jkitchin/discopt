"""highs-parity P2.0: a pure LP solve does not import ``scipy.optimize``.

``discopt._dual_recovery`` imported ``scipy.optimize.lsq_linear`` at module load, and every
solve imports that module (``discopt.solver`` -> ``discopt.validation``): 0.16 s of a
fresh process's first solve, measured with ``-X importtime``, on a route that never calls
the refit (0 ``lsq_linear`` calls over four L-panel LPs). The import is now taken on first
use. Fails before the change: ``scipy.optimize`` is in ``sys.modules`` after the solve.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("highspy")

_CHILD = textwrap.dedent(
    """
    import sys
    import discopt.modeling as dm
    m = dm.Model("lp")
    x = m.continuous("x", shape=(3,), lb=0, ub=10)
    m.subject_to(x[0] + 2 * x[1] - x[2] <= 4)
    m.subject_to(x[0] + x[1] + x[2] >= 2)
    m.minimize(3 * x[0] + x[1] + 2 * x[2])
    r = m.solve(time_limit=20)
    assert r.status == "optimal", r.status
    print("LOADED" if "scipy.optimize" in sys.modules else "CLEAN")
    """
)


def test_pure_lp_solve_does_not_import_scipy_optimize():
    p = subprocess.run([sys.executable, "-c", _CHILD], capture_output=True, text=True, timeout=240)
    assert p.returncode == 0, p.stderr[-2000:]
    assert p.stdout.strip().splitlines()[-1] == "CLEAN"


def test_lazy_lsq_linear_is_scipys():
    import numpy as np
    from discopt import _dual_recovery as DR
    from scipy.optimize import lsq_linear

    A, b = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]), np.array([1.0, 2.0, 3.0])
    got = DR.lsq_linear(A, b, bounds=(0.0, 1.5)).x
    assert np.array_equal(got, lsq_linear(A, b, bounds=(0.0, 1.5)).x)
