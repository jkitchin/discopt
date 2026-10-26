"""#1970: the native kernel's spatial branching rule.

The kernel split the widest operand of the single term with the largest McCormick
gap, at the LP value. ``DISCOPT_KERNEL_SPATIAL_BRANCHING`` scores every operand of
every violated term (its share of the gaps plus a spatial pseudocost) and pulls the
split point toward the midpoint, after SCIP's nonlinear-handler rule. Only the
branching order changes, so both arms must certify the same optimum; the probe
counter ``tree/spatial_score_picks`` says whether the rule ran.
"""

from __future__ import annotations

from pathlib import Path

import discopt.modeling as dm
import pytest

NOUS1 = Path(__file__).parent / "data" / "issue_1871" / "nous1.nl"
NOUS1_OPT = 1.567071935677213  # SCIP 10, and MINLPLib's =opt=


def _chain_model() -> dm.Model:
    m = dm.Model("chain")
    x, y, z, w, v = (m.continuous(n, lb=0, ub=2) for n in "xyzwv")
    m.subject_to(x + y + z + w + v <= 4)
    m.minimize(-x * y - y * z - z * w - w * v + 0.3 * x * v)
    return m


@pytest.mark.smoke
def test_both_rules_certify_the_same_optimum(monkeypatch):
    res = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("DISCOPT_KERNEL_SPATIAL_BRANCHING", flag)
        r = _chain_model().solve(time_limit=60)
        assert r.algorithm_route.startswith("native-spatial"), r.algorithm_route
        assert r.status == "optimal" and r.gap_certified
        assert r.objective == pytest.approx(-4.0, abs=1e-3)
        assert r.bound <= r.objective + 1e-9
        res[flag] = r
    assert res["0"].solver_stats["tree/spatial_score_picks"] == 0
    assert res["1"].solver_stats["tree/spatial_score_picks"] > 0, "the scored rule never ran"


@pytest.mark.smoke
def test_flag_parsing(monkeypatch):
    from discopt.solver import _kernel_spatial_branching_kwargs

    for off in ("0", "", "off", "false"):
        monkeypatch.setenv("DISCOPT_KERNEL_SPATIAL_BRANCHING", off)
        assert _kernel_spatial_branching_kwargs() == {}
    monkeypatch.delenv("DISCOPT_KERNEL_SPATIAL_BRANCHING", raising=False)
    assert _kernel_spatial_branching_kwargs() == {}
    monkeypatch.setenv("DISCOPT_KERNEL_SPATIAL_BRANCHING", "1")
    assert _kernel_spatial_branching_kwargs() == {
        "spatial_score": True,
        "spatial_pscost_weight": 1.0,
        "spatial_midpull": 0.75,
    }


@pytest.mark.slow
@pytest.mark.correctness
def test_nous1_certifies(monkeypatch):
    """The issue's probe: OFF ends at the time limit (bound ~1.21 after ~64k nodes
    at 120 s); ON certifies in ~12k nodes, ~33 s on a dev machine."""
    monkeypatch.setenv("DISCOPT_KERNEL_SPATIAL_BRANCHING", "1")
    r = dm.from_nl(str(NOUS1)).solve(time_limit=150, gap_tolerance=1e-4)
    assert r.solver_stats["tree/spatial_score_picks"] > 0
    assert r.bound <= NOUS1_OPT + 1e-6
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(NOUS1_OPT, rel=1e-4)
