"""#2015: confirm a #1656 lift/model disagreement on the lift's exact value before
the native kernel declines (``DISCOPT_KERNEL_DECLINE_EXACT``).

The pricing callback declined whenever ``c'x`` at a McCormick-tight point and the
posed objective disagreed past ``1e-4 (1 + |v|)``. The acceptance test allows each
lifted term ``mccormick_tol`` of slack, and an objective coefficient multiplies it:
on MINLPLib ``rbrock`` (``100 (x1 - x0**2)**2 + (1 - x0)**2``) a point beside the
optimum has ``c'x = -3.2e-4`` against 6.4e-5, so an exact lift was declined and the
Python tree ended uncertified. Under the flag the disagreement is re-checked on the
spec's own node over a tight box around the point; ``prob10``, whose lift misses
its ``sin``, must still decline.
"""

from __future__ import annotations

import pathlib
import warnings

import pytest
from discopt.modeling.core import from_nl

DATA = pathlib.Path(__file__).parent / "data" / "minlplib"


def _solve(name, monkeypatch, flag, tl):
    monkeypatch.setenv("DISCOPT_KERNEL_DECLINE_EXACT", flag)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return from_nl(str(DATA / f"{name}.nl")).solve(time_limit=tl)


def test_an_exact_lift_is_kept_on_the_kernel(monkeypatch):
    r = _solve("rbrock", monkeypatch, "1", 30)
    assert r.algorithm_route.startswith("native-spatial"), r.algorithm_route
    assert r.status == "optimal" and r.gap_certified, (r.status, r.bound)
    assert r.objective == pytest.approx(0.0, abs=1e-6)
    assert r.bound <= r.objective + 1e-6
    assert r.solver_stats["tree/decline_exact_kept"] > 0


def test_default_on_with_an_opt_out(monkeypatch):
    from discopt.solver import _kernel_decline_exact_enabled

    monkeypatch.delenv("DISCOPT_KERNEL_DECLINE_EXACT", raising=False)
    assert _kernel_decline_exact_enabled()
    monkeypatch.setenv("DISCOPT_KERNEL_DECLINE_EXACT", "0")
    assert not _kernel_decline_exact_enabled()


def test_flag_off_keeps_the_1656_decline(monkeypatch):
    r = _solve("rbrock", monkeypatch, "0", 5)
    assert not r.algorithm_route.startswith("native-spatial"), r.algorithm_route


def test_a_lift_that_is_not_the_model_still_declines(monkeypatch):
    r = _solve("prob10", monkeypatch, "1", 30)
    assert r.algorithm_route.startswith("spatial-bb"), r.algorithm_route
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(3.44550379, rel=1e-6)
