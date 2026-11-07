"""highs-parity P3.3: IIS time bounding on the Rust engine.

The HiGHS backend stops an overrunning run with an interrupt callback (``_GRACE_S`` past
its budget, single-sourced from ``discopt._iis_backend.GRACE_S``); the Rust MILP driver
polls its own deadline, so the guard is route-specific. Measured at P3.3's entry: a probe
with a 0.5 s budget on enlight-class lights-out models (8x8 .. 14x14) returned in <=
0.58 s, presolve on and off.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import scipy.sparse as sp
from discopt import _iis_backend as _bk
from discopt import _iis_milp


def _enlight(n):
    """Lights-out on an ``n x n`` grid: every cell toggled an odd number of times,
    ``sum(neighbourhood x) - 2 k = 1``; ``x`` binary, ``k`` integer in [0, 2]."""
    N = n * n
    A = sp.lil_matrix((N, 2 * N))
    for i in range(n):
        for j in range(n):
            r = i * n + j
            for di, dj in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < n and 0 <= b < n:
                    A[r, a * n + b] = 1.0
            A[r, N + r] = -2.0
    lb = np.zeros(2 * N)
    ub = np.concatenate([np.ones(N), np.full(N, 2.0)])
    return sp.csr_matrix(A), np.ones(N), np.ones(N), lb, ub, np.ones(2 * N, dtype=bool)


def test_grace_is_single_sourced():
    assert _iis_milp._GRACE_S == _bk.GRACE_S


@pytest.mark.parametrize("presolve", ["choose", "off"])
def test_rust_probe_returns_within_one_second_of_a_half_second_budget(presolve):
    """P3.3 Done: an enlight-class probe with a 0.5 s budget returns within 1.0 s."""
    be = _bk.RustBackend(*_enlight(12))
    t0 = time.perf_counter()
    st = be.run(presolve, time.perf_counter() + 0.5)
    wall = time.perf_counter() - t0
    assert wall < 1.0, wall
    assert st == _bk.INTERRUPTED  # stopped by the budget, reported as such (no proof)


def test_rust_engine_honours_the_total_budget():
    """A whole IIS run on the Rust engine ends within its ``total_time_limit`` (plus a
    CI margin): pigeonhole PHP(10,9) cannot be proven in 2 s, so it raises
    ``TimeoutError`` -- an honest "nothing confirmed", on time."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_iis_sat_candidate import _pigeonhole_model

    budget = 2.0
    t0 = time.perf_counter()
    with pytest.raises(TimeoutError):
        _iis_milp.milp_iis(
            _pigeonhole_model(10, 9), time_limit=10, total_time_limit=budget, iis_engine="rust"
        )
    assert time.perf_counter() - t0 < budget + 1.5
