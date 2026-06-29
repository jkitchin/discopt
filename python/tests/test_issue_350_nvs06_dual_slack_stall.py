"""Regression for issue #350: the dual-slack root solve must not stall on an
ill-conditioned relaxation MILP and burn the whole time budget.

nvs06 (=opt= 1.7703125, ~0.3 s on the pre-#334 build) finds its optimum at the
root in ~1 s, then certifies in 3 nodes. #334's dual-slack root solve
(``solve_lp_root`` -> ``solve_lp_warm``) is a large win on covering/packing
relaxations, but on nvs06's geometric-mean-equilibrated McCormick relaxation MILP
the dual simplex from the all-slack basis degenerate-cycles to ``max_iter`` and
consumed the *entire* remaining per-node budget — wall time scaled with the limit
(10 s cap -> 10 s, 30 s cap -> 24 s) while the bound stayed pinned, never
certifying within a sane budget.

Fix (``milp_driver.rs``): the dual-slack warm start is only ever an optimization,
so cap it to a size-proportional pivot budget and fall back to the cold primal
when it stalls (``IterLimit`` / ``Numerical``). The covering-LP win is preserved
(it converges in O(m+n) pivots, far under the cap); the stall trips the cap almost
immediately and cold-solves.

This guards budget-independence: with the bug, wall time tracks the limit, so a
generous limit on a problem that genuinely needs ~2 s exposes it.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib")
_NVS06_OPT = 1.7703125


@pytest.mark.slow
@pytest.mark.requires_pounce
def test_nvs06_does_not_burn_budget_on_dual_slack_stall():
    """nvs06 must certify quickly and *independently of the time budget*. With the
    #334 stall it consumed the full limit; the fix bounds it to a few seconds."""
    path = os.path.join(_DATA, "nvs06.nl")
    if not os.path.exists(path):
        pytest.skip("nvs06 instance unavailable")

    # A generous budget: the solve genuinely needs ~2 s, so anything near the limit
    # means the dual-slack stall is back (wall used to scale with the limit).
    t0 = time.perf_counter()
    r = dm.from_nl(path).solve(time_limit=30, gap_tolerance=1e-4)
    elapsed = time.perf_counter() - t0

    assert elapsed < 8.0, (
        f"nvs06 took {elapsed:.1f}s on a 30s budget — dual-slack root solve is "
        f"stalling and burning the budget again (#350)"
    )
    assert r.status == "optimal", f"nvs06 did not certify (status={r.status})"
    assert r.gap_certified, "nvs06 reached optimum but gap was not certified"
    assert r.objective == pytest.approx(_NVS06_OPT, abs=1e-3)
