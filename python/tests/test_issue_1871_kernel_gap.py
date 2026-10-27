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

# The three solve-backed tests below take ~8 s each here and ~3x that on CI, over
# the fast lane's 15 s budget, so they run in the slow/correctness lanes; the fast
# lane keeps the gate logic, the defaults and the unseeded-OFF check.

GAP = "DISCOPT_KERNEL_DOCUMENTED_GAP"
CUT = "DISCOPT_KERNEL_CUTOFF_OBBT"
ST_M2 = os.path.join(os.path.dirname(__file__), "data", "issue_1871", "st_m2.nl")
OPT = -856648.8186999999  # minlplib.solu


def test_documented_gap_applies_only_without_an_absolute_request(monkeypatch):
    monkeypatch.setenv(GAP, "1")
    assert SV._kernel_documented_gap_applies(None)
    # #2000: a named absolute tolerance no looser than the default tightens the
    # documented criterion (``a OR rel``), so it is honoured as that criterion...
    assert SV._kernel_documented_gap_applies(1e-9)
    assert SV._kernel_documented_gap_applies(0.0)
    assert SV._kernel_documented_gap_applies(SV._DEFAULT_ABS_GAP_TOL)
    # ...while a looser one keeps #1243's strict ``min`` mapping.
    assert not SV._kernel_documented_gap_applies(1e-3)
    monkeypatch.setenv(GAP, "0")
    assert not SV._kernel_documented_gap_applies(None)
    assert not SV._kernel_documented_gap_applies(1e-9)


@pytest.mark.slow
@pytest.mark.correctness
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


@pytest.mark.slow
@pytest.mark.correctness
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


NOUS1 = os.path.join(os.path.dirname(__file__), "data", "issue_1871", "nous1.nl")
NOUS1_OPT = 1.567072  # minlplib.solu


def _nous1_seed():
    m = dm.from_nl(NOUS1)
    b = fbbt_box(m)
    lb, ub = np.asarray(b.lb, float), np.asarray(b.ub, float)
    return SV._native_kernel_seed(m, lb, ub, 1.0, 0.0, lb.size, None)


@pytest.mark.slow
@pytest.mark.correctness
def test_seed_ils_gives_nous1_a_verified_seed(monkeypatch):
    """The kernel seed's enumeration finds nothing on nous1; the local search does."""
    monkeypatch.setenv("DISCOPT_KERNEL_SEED_ILS", "1")
    SV.KERNEL_SEED_ILS_STATS.clear()
    value, point = _nous1_seed()
    assert SV.KERNEL_SEED_ILS_STATS["fired"] == 1, "the local search never ran"
    assert value is not None and point is not None
    assert value >= NOUS1_OPT - 1e-6  # a verified feasible point never beats the optimum
    ok, obj = SV._native_kernel_verify_point(dm.from_nl(NOUS1), point)
    assert ok and abs(obj - value) <= 1e-9 * (1 + abs(value))


def test_seed_ils_off_leaves_nous1_unseeded(monkeypatch):
    monkeypatch.setenv("DISCOPT_KERNEL_SEED_ILS", "0")
    assert _nous1_seed() == (None, None)


def test_seed_ils_defaults_on(monkeypatch):
    """Graduated under CLAUDE.md section 5 (panel in the gate's docstring)."""
    monkeypatch.delenv("DISCOPT_KERNEL_SEED_ILS", raising=False)
    assert SV._kernel_seed_ils_enabled()
    monkeypatch.setenv("DISCOPT_KERNEL_SEED_ILS", "0")
    assert not SV._kernel_seed_ils_enabled()
