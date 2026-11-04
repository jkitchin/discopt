"""#2142 item 3: SCIP's branching-point rule for the native spatial kernel.

``DISCOPT_SPATIAL_BRANCH_CLAMP`` passes ``spatial_midpull_reldomtrig=0.5`` and
``spatial_clamp=0.2`` (SCIP's ``branching/midpullreldomtrig`` and ``branching/clamp``)
to the kernel on top of #1970's 0.75 midpull. Only the split point changes: every
child pair still covers its parent box, so both arms certify the same optimum. The
profile counters ``SpatialSplits`` / ``SpatialSplitMoved`` prove the rule ran and
moved a decision. ``performance-plan.md`` section 107.
"""

from __future__ import annotations

import discopt._rust as R
import discopt.modeling as dm
import numpy as np
import pytest


def _chain_model() -> dm.Model:
    m = dm.Model("chain")
    x, y, z, w, v = (m.continuous(n, lb=0, ub=2) for n in "xyzwv")
    m.subject_to(x + y + z + w + v <= 4)
    m.minimize(-x * y - y * z - z * w - w * v + 0.3 * x * v)
    return m


@pytest.mark.smoke
def test_flag_parsing(monkeypatch):
    from discopt.solver import _spatial_branch_clamp_kwargs

    for off in ("0", "", "off", "false"):
        monkeypatch.setenv("DISCOPT_SPATIAL_BRANCH_CLAMP", off)
        assert _spatial_branch_clamp_kwargs() == {}
    monkeypatch.delenv("DISCOPT_SPATIAL_BRANCH_CLAMP", raising=False)
    assert _spatial_branch_clamp_kwargs() == {}
    monkeypatch.setenv("DISCOPT_SPATIAL_BRANCH_CLAMP", "1")
    assert _spatial_branch_clamp_kwargs() == {
        "spatial_clamp": 0.2,
        "spatial_midpull_reldomtrig": 0.5,
    }


@pytest.mark.smoke
def test_kernel_refuses_out_of_range_knobs():
    """The binding validates the two knobs (a clamp past 0.5 would cross the
    midpoint and a trigger past 1 has no meaning)."""
    z = np.zeros(0)
    zi = np.zeros(0, dtype=np.int64)
    base = dict(
        n_cols=1,
        n_orig=1,
        c=np.array([1.0]),
        integrality=np.zeros(1, dtype=np.int64),
        global_lo=np.array([0.0]),
        global_hi=np.array([1.0]),
        fixed_row_ptr=np.zeros(1, dtype=np.int64),
        fixed_cols=zi,
        fixed_coeffs=z,
        fixed_rhs=z,
        term_kind=zi,
        term_i=zi,
        term_j=zi,
        term_out=zi,
        term_p=zi,
        term_coeff=z,
        term_cst=z,
        blf_w=zi,
        blf_a_ptr=np.zeros(1, dtype=np.int64),
        blf_a_cols=zi,
        blf_a_coeffs=z,
        blf_a_const=z,
        blf_b_ptr=np.zeros(1, dtype=np.int64),
        blf_b_cols=zi,
        blf_b_coeffs=z,
        blf_b_const=z,
        obbt_candidates=zi,
    )
    with pytest.raises(ValueError, match="spatial_clamp"):
        R.solve_spatial_tree_py(**base, spatial_clamp=0.6)
    with pytest.raises(ValueError, match="spatial_midpull_reldomtrig"):
        R.solve_spatial_tree_py(**base, spatial_midpull_reldomtrig=1.5)
    out = R.solve_spatial_tree_py(**base, spatial_clamp=0.2, spatial_midpull_reldomtrig=0.5)
    assert out["incumbent"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.smoke
def test_both_arms_certify_the_same_optimum_and_the_rule_moves_splits(monkeypatch):
    monkeypatch.setenv("DISCOPT_PROFILE", "1")
    res, ctr = {}, {}
    for flag in ("0", "1"):
        monkeypatch.setenv("DISCOPT_SPATIAL_BRANCH_CLAMP", flag)
        R.profile_reset_py()
        r = _chain_model().solve(time_limit=60, deterministic=True)
        ctr[flag] = dict(R.profile_counters_py())
        assert r.algorithm_route.startswith("native-spatial"), r.algorithm_route
        assert r.status == "optimal" and r.gap_certified
        assert r.objective == pytest.approx(-4.0, abs=1e-3)
        assert r.bound <= r.objective + 1e-9
        res[flag] = r
    assert ctr["0"]["SpatialSplits"] > 0, "the kernel never split spatially"
    assert ctr["0"]["SpatialSplitMoved"] == 0, "the OFF arm ran the new rule"
    assert ctr["1"]["SpatialSplits"] > 0
    assert ctr["1"]["SpatialSplitMoved"] > 0, "the rule never moved a split point"
