"""Regression for issue #291: the native simplex MILP engine must honor the time
limit and never hang.

nvs12 (=opt= -481.2, ~0.4 s on the prior build) has integer-bilinear/monomial
terms that the #285 reformulation turns into a pure MILP, routed to the Rust
``solve_milp_py`` engine. That engine was called with no time limit
(``time_limit_s`` defaulting to 0.0 -> no deadline), so a node whose simplex failed
to converge looped unbounded, ignoring the user's ``time_limit`` (>40 s on a 15 s
limit; only an external kill stopped it).

Fix: pass the remaining wall-clock budget (capped, so a stall defers quickly
regardless of the overall limit), and defer (fall back to the robust spatial/POUNCE
path) when the engine exhausts its budget without a usable incumbent — which solves
nvs12 instead of returning a bare ``node_limit``.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib")


@pytest.mark.slow
@pytest.mark.requires_pounce
def test_nvs12_does_not_hang_and_solves():
    """nvs12 must return within the time limit (no unbounded hang) and solve to the
    known optimum via the fallback after the simplex engine stalls."""
    path = os.path.join(_DATA, "nvs12.nl")
    if not os.path.exists(path):
        pytest.skip("nvs12 instance unavailable")
    t0 = time.perf_counter()
    r = dm.from_nl(path).solve(time_limit=40, gap_tolerance=1e-4)
    elapsed = time.perf_counter() - t0
    # honored the time limit (with generous margin for the fallback handoff) —
    # the bug ran unbounded past any limit
    assert elapsed < 90, f"solve ran {elapsed:.0f}s — time limit not honored (hang)"
    assert r.status != "node_limit"  # must not surface a bare no-solution result
    assert r.objective is not None and r.objective == pytest.approx(-481.2, abs=1e-2)
