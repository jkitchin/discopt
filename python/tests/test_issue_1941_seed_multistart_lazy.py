"""#1941 item 4: the native kernel's seed runs ``continuous_multistart`` only when no
earlier seed stage yielded a candidate (``DISCOPT_KERNEL_SEED_MULTISTART_LAZY``).

Before, a pure-continuous model's multistart ran even after the relaxation-point
sub-NLP had produced a verified seed; on one-node instances it was most of the solve
(st_m2 8.0 of 8.8 s). Primal only, so both arms must certify the same optimum; the
probe counter says whether the rule skipped anything.
"""

from __future__ import annotations

from pathlib import Path

import discopt.modeling as dm
import discopt.solver as SV
import pytest

pytestmark = pytest.mark.smoke

DISPATCH = Path(__file__).parent / "data" / "minlplib_nl" / "dispatch.nl"
DISPATCH_OPT = 3155.2879


@pytest.mark.parametrize("lazy", ["0", "1"])
def test_both_arms_certify_the_same_optimum(monkeypatch, lazy):
    monkeypatch.setenv("DISCOPT_KERNEL_SEED_MULTISTART_LAZY", lazy)
    SV.KERNEL_SEED_MULTISTART_STATS.clear()
    r = dm.from_nl(str(DISPATCH)).solve(time_limit=60, gap_tolerance=1e-4)
    assert r.algorithm_route.startswith("native-spatial"), r.algorithm_route
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(DISPATCH_OPT, rel=1e-4)
    assert r.bound <= r.objective + 1e-9
    capped = SV.KERNEL_SEED_MULTISTART_STATS["capped"]
    assert (capped > 0) == (lazy == "1"), dict(SV.KERNEL_SEED_MULTISTART_STATS)
