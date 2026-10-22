"""Issue #1869: an incumbent just outside the gap, re-solved to full NLP accuracy.

``prolog`` (optimum 0) is the probe. Its objective cancels ~1e5-sized bilinear terms,
and the node NLPs' scaled tolerance leaves the Python tree's incumbent at 1.51e-6
under POUNCE 0.13.0 (8.2e-7 under 0.12.0). The verified bound is -1e-12, so the
absolute gap stays just above the 1e-6 tolerance and the tree runs to its limit.
``DISCOPT_ENDGAME_POLISH`` re-solves that incumbent once at ``tol=1e-12,
constr_viol_tol=1e-14``. The re-solved point is verified, adopted only on strict
improvement, and closes the gap.
"""

from __future__ import annotations

import os

import discopt.modeling as dm
import discopt.solver as SV
import pytest

pytestmark = pytest.mark.unit

FLAG = "DISCOPT_ENDGAME_POLISH"
PROLOG = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl", "prolog.nl")


def test_window_admits_only_a_nearly_closed_gap():
    # abs tol 1e-6, rel 1e-4: a 1.5e-6 gap at zero is in the window, a 1.0 gap is not.
    assert SV._endgame_polish_in_window(1.5e-6, -1e-12, 1e-4, 1e-6)
    assert not SV._endgame_polish_in_window(1.0, -1e-12, 1e-4, 1e-6)
    # relative arm: 5% at 100 is inside 1e3 * 1e-4 = 10%, 50% is not.
    assert SV._endgame_polish_in_window(105.0, 100.0, 1e-4, 1e-6)
    assert not SV._endgame_polish_in_window(200.0, 100.0, 1e-4, 1e-6)
    # no bound, no window
    assert not SV._endgame_polish_in_window(1.0, float("-inf"), 1e-4, 1e-6)


def test_flag_on_certifies_prolog_through_the_polish(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    SV.ENDGAME_POLISH_STATS.clear()
    r = dm.from_nl(PROLOG).solve(time_limit=60)
    assert SV.ENDGAME_POLISH_STATS["fired"] > 0, "the polish never ran"
    assert r.status == "optimal" and r.gap_certified, (r.status, r.objective, r.bound)
    assert abs(r.objective) <= 1e-6 and r.bound <= 1e-9  # optimum 0 (minlplib.solu)


def test_flag_off_never_polishes(monkeypatch):
    monkeypatch.setenv(FLAG, "0")
    SV.ENDGAME_POLISH_STATS.clear()
    dm.from_nl(PROLOG).solve(time_limit=5)
    assert SV.ENDGAME_POLISH_STATS["fired"] == 0


def test_flag_defaults_on(monkeypatch):
    """Graduated under CLAUDE.md section 5 (panel in the gate's docstring)."""
    monkeypatch.delenv(FLAG, raising=False)
    assert SV._endgame_polish_enabled()
    monkeypatch.setenv(FLAG, "0")
    assert not SV._endgame_polish_enabled()
