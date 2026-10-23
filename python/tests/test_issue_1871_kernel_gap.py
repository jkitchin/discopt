"""Issue #1871: the native kernel's stopping rule and root box on ``st_m2``.

``st_m2`` (QP, 30 vars, optimum -856648.8187 from minlplib.solu) holds its optimal
incumbent on the native kernel but never certifies. Two separate causes:

* The kernel's stop and prune test conjoins an absolute ``gap_tol = gap_tolerance``
  (1e-4) with the documented criterion. At an objective of about 1e6 that arm sits
  near the LP bound's own precision: with the cutoff box below, the kernel reaches
  a relative gap of 1.5e-9 and still runs to its limit.
  ``DISCOPT_KERNEL_DOCUMENTED_GAP`` lifts the arm when no ``abs_gap_tolerance`` was
  named.
* The kernel's root box is fixed before its seed incumbent exists.
  ``DISCOPT_KERNEL_CUTOFF_OBBT`` re-runs root OBBT with that incumbent as a cutoff,
  as the Python tree's Phase C does: 32,789 nodes -> 1.
"""

from __future__ import annotations

import os

import discopt.modeling as dm
import discopt.solver as SV
import numpy as np
import pytest
from discopt.tightening import fbbt_box

pytestmark = pytest.mark.unit

GAP = "DISCOPT_KERNEL_DOCUMENTED_GAP"
CUT = "DISCOPT_KERNEL_CUTOFF_OBBT"
ST_M2 = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "st_m2.nl")
OPT = -856648.8186999999  # minlplib.solu


def test_documented_gap_applies_only_without_an_absolute_request(monkeypatch):
    monkeypatch.setenv(GAP, "1")
    assert SV._kernel_documented_gap_applies(None)
    # #1243: a named absolute tolerance keeps the strict ``min`` mapping.
    assert not SV._kernel_documented_gap_applies(1e-9)
    assert not SV._kernel_documented_gap_applies(0.0)
    monkeypatch.setenv(GAP, "0")
    assert not SV._kernel_documented_gap_applies(None)


def test_cutoff_box_keeps_the_optimum_and_shrinks_the_box():
    """The cutoff box must contain the known optimal point and be strictly smaller."""
    m = dm.from_nl(ST_M2)
    b = fbbt_box(m)
    lb, ub = np.asarray(b.lb, float), np.asarray(b.ub, float)
    r = m.solve(time_limit=30)  # any route; we need a verified optimal point
    x = np.concatenate([np.ravel(np.asarray(r.x[v.name], float)) for v in m._variables])
    assert abs(r.objective - OPT) <= 1e-4 * abs(OPT)
    box = SV._kernel_cutoff_obbt_box(dm.from_nl(ST_M2), lb, ub, float(r.objective), None)
    assert box is not None
    clb, cub = box
    assert np.all(clb >= lb - 1e-9) and np.all(cub <= ub + 1e-9)
    assert np.sum(cub - clb) < np.sum(np.minimum(ub, 1e12) - lb)
    assert np.all(x >= clb - 1e-6) and np.all(x <= cub + 1e-6), "optimum cut from the box"


def test_both_flags_certify_st_m2_on_the_kernel(monkeypatch):
    monkeypatch.setenv(GAP, "1")
    monkeypatch.setenv(CUT, "1")
    SV.KERNEL_CUTOFF_OBBT_STATS.clear()
    r = dm.from_nl(ST_M2).solve(time_limit=60)
    assert (r.algorithm_route or "").startswith("native-spatial"), r.algorithm_route
    assert SV.KERNEL_CUTOFF_OBBT_STATS["adopted"] > 0, "the cutoff box was never used"
    assert r.status == "optimal" and r.gap_certified, (r.status, r.objective, r.bound)
    assert r.bound <= OPT + 1e-6 * abs(OPT)  # a bound never passes the optimum
    assert abs(r.objective - OPT) <= 1e-4 * abs(OPT)


def test_flag_defaults(monkeypatch):
    """Both graduated under CLAUDE.md section 5 (panels in the gates' docstrings)."""
    monkeypatch.delenv(GAP, raising=False)
    monkeypatch.delenv(CUT, raising=False)
    assert SV._kernel_documented_gap_enabled()
    assert SV._kernel_cutoff_obbt_enabled()
    monkeypatch.setenv(GAP, "0")
    monkeypatch.setenv(CUT, "0")
    assert not SV._kernel_documented_gap_enabled()
    assert not SV._kernel_cutoff_obbt_enabled()
