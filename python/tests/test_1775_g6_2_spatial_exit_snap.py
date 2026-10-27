"""#1775 G6.2: the spatial exit polish gets a real budget when it is a repair.

At exit the spatial B&B snaps near-integral discrete columns to integers, judges
that integral point with the #1659 exit gate (1e-6, term-scaled), and re-solves
the continuous part with the integers fixed (the terminal polish). On a big-M row
a binary 1e-5 shy of 1 moves the row by ``M * 1e-5`` when snapped, so the integral
point can fail the gate while the fractional one passes on slack the user's model
does not have (#1561: ``verify_point`` judges the integral realisation). Then the
polish is the only repair -- and a clock-limited exit gave it ``0.1`` s. Measured
on GDPlib ``mod_hens`` (big-M, 300 s): 0.1 s in 4/4 runs, TIME_LIMIT in 2, and a
reported point 4.5e-5 outside row 431. The repair now gets #1870's exit slice,
and the polish can never trade a point the gate passes for one it refuses.

An earlier draft reported the fractional point instead; that passed the lowered
model's gate but failed ``verify_point`` on mod_hens (indicator row, 50.5x its
allowance) and was dropped.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solver as SV
import numpy as np
import pytest
from discopt._tape_nlp_evaluator import make_evaluator

pytestmark = pytest.mark.smoke

M = 4.5


def _bigm_row_model():
    m = dm.Model("snap")
    x = m.continuous("x", lb=0, ub=10)
    y = m.binary("y")
    m.subject_to(x >= M * y)  # a big-M style link: snapping y moves this row by M*dy
    m.minimize(x)
    return m


def _arb():
    m = _bigm_row_model()
    ev = make_evaluator(m)
    cl, cu = SV._infer_constraint_bounds(m, ev)
    return SV._RowArbiter(ev, cl, cu, None)


def _fractional():
    y = 1.0 - 1e-5  # within the 1e-5 integrality tolerance
    return np.array([M * y, y])  # exactly on the row as it stands


def test_the_loose_gate_admits_a_snap_the_exit_gate_refuses():
    """The defect's shape: two opinions of the same integral point disagree."""
    arb = _arb()
    rounded, feasible = SV._round_incumbent_integers(
        _fractional(), [1], [1], arb.evaluator, arb.cl, arb.cu
    )
    assert rounded[1] == 1.0 and feasible  # the 1e-4 gate keeps the snap
    assert SV._passes_exit_gate(arb, _fractional())
    assert not SV._passes_exit_gate(arb, rounded)  # 4.5e-5 outside: a repair is due


def test_repair_floor():
    assert SV._exit_polish_floor(300.0, False) == 0.1
    assert SV._exit_polish_floor(300.0, True) == 0.5
    assert SV._exit_polish_floor(4.0, True) == pytest.approx(0.2)  # 5% of a short budget
    assert SV._exit_polish_floor(1.0, True) == 0.1  # never below the refinement floor


def test_the_polish_never_trades_a_clean_point_for_a_dirty_one():
    arb = _arb()
    clean = np.array([M, 1.0])
    dirty = np.array([M * (1.0 - 1e-5), 1.0])  # 4.5e-5 outside the row
    assert not SV._polish_keeps_exit_gate(arb, dirty, clean)
    assert SV._polish_keeps_exit_gate(arb, clean, clean)
    assert SV._polish_keeps_exit_gate(arb, clean, dirty)  # the repair itself
    # An incumbent that already fails the gate is not held to it (#1199's
    # never-degrade bar decides that case).
    assert SV._polish_keeps_exit_gate(arb, dirty, dirty)


def test_a_one_row_numpy_bound_array_counts_as_rows():
    arb = _arb()
    arb = arb._replace(cl=np.asarray(arb.cl, dtype=float), cu=np.asarray(arb.cu, dtype=float))
    assert len(arb.cl) == 1 and SV._arb_has_rows(arb)
    assert not SV._passes_exit_gate(arb, np.array([M * (1.0 - 1e-5), 1.0]))


def _spatial_gdp():
    m = dm.Model("g62_exit")
    x = m.continuous("x", lb=0, ub=4)
    y = m.continuous("y", lb=0, ub=4)
    m.either_or([[x + y <= 2], [x + y >= 5, x <= 3.5]], name="d")
    m.subject_to(x * y >= 1)
    m.minimize(-x * y + 0.3 * x**2 - y)
    return m


@pytest.mark.parametrize("fails", [False, True])
def test_the_exit_asks_for_the_repair_floor_when_the_integral_point_fails(fails, monkeypatch):
    """Wiring: the polish budget is chosen from the INTEGRAL point's verdict."""
    # #2015: the native kernel now takes this model; the test is about the Python tree.
    monkeypatch.setenv("DISCOPT_NATIVE_SPATIAL_KERNEL", "0")
    seen: list[bool] = []
    real_floor = SV._exit_polish_floor

    def floor(tl, repair):
        seen.append(repair)
        return real_floor(tl, repair)

    monkeypatch.setattr(SV, "_exit_polish_floor", floor)
    real_gate = SV._passes_exit_gate
    monkeypatch.setattr(SV, "_passes_exit_gate", lambda a, x: False if fails else real_gate(a, x))
    _spatial_gdp().solve(time_limit=30, gdp_method="big-m")
    assert seen == [fails], seen
