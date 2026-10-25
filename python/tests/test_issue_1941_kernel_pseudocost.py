"""#1941 (#1913 item 1): pseudocost integer branching in the native spatial kernel.

The kernel branched on the lowest-index fractional integer column, which made the
tree a chaotic function of the root box (nvs17: 17,229 to >= 100,000 nodes over
valid root boxes). ``DISCOPT_KERNEL_PSEUDOCOST`` picks the column by pseudocost,
``DISCOPT_KERNEL_PSEUDOCOST_RELIABILITY`` strong-branches a column until it has that
many observations per side. Only the branching order changes, so both arms must
certify the same optimum; the probe counter says whether the rule ran.
"""

from __future__ import annotations

from pathlib import Path

import discopt.modeling as dm
import pytest

pytestmark = pytest.mark.smoke

NVS13 = Path(__file__).parent / "data" / "minlplib_nl" / "nvs13.nl"
NVS13_OPT = -585.2


def _solve(monkeypatch, on: bool, reliability: int = 1):
    monkeypatch.setenv("DISCOPT_KERNEL_PSEUDOCOST", "1" if on else "0")
    monkeypatch.setenv("DISCOPT_KERNEL_PSEUDOCOST_RELIABILITY", str(reliability))
    r = dm.from_nl(str(NVS13)).solve(time_limit=60, gap_tolerance=1e-4)
    assert r.algorithm_route.startswith("native-spatial"), r.algorithm_route
    return r


@pytest.mark.parametrize("reliability", [0, 1])
def test_both_rules_certify_the_same_optimum(monkeypatch, reliability):
    off = _solve(monkeypatch, on=False)
    on = _solve(monkeypatch, on=True, reliability=reliability)
    for r in (off, on):
        assert r.status == "optimal" and r.gap_certified
        assert r.objective == pytest.approx(NVS13_OPT, abs=1e-4)
        assert r.bound <= r.objective + 1e-9
    assert off.solver_stats["tree/pseudocost_picks"] == 0
    assert on.solver_stats["tree/pseudocost_picks"] > 0, "the pseudocost rule never ran"
