"""xexp atom (issue #632 adjacent-atom family), gated DISCOPT_XEXP_ATOM.

``t*exp(t)`` is convex on ``t >= -2`` (``f'' = exp(t)(t+2) >= 0``), but the
factorable path shatters it into a loose bilinear ``t * exp(t)`` (t against the
convex-relaxed exp) -> root gap ~2.3 on an interior-min box. The gated atom emits
the exact 1-D convex envelope (tangent under + secant over) for a shared affine
``t``. Tests pin: (1) SOUND — the relaxation LB never exceeds the true box minimum;
(2) NEVER LOOSER, strictly TIGHTER where the convexity bites; (3) the ``lo < -2``
inflection guard leaves it byte-identical (atom abstains); (4) byte-identical off,
and a genuine bilinear ``x*exp(y)`` is untouched when on.
"""

from __future__ import annotations

import math
import os

import discopt.modeling as dm
import pytest
from discopt import Model
from discopt._jax.milp_relaxation import build_milp_relaxation
from discopt._jax.term_classifier import classify_nonlinear_terms

# Flag-off bound for x*exp(x) on [-1,1]. #640 Bucket 1: the recovered separable
# objective floor now recognizes ``x*exp(x) >= -1/e`` and closes this single-term
# bound to the exact global minimum (-1/e), independent of the xexp atom flag — so
# the flag-off baseline is the tight -1/e, not the old shattered-bilinear -1.1752.
_OFF_REF = -math.exp(-1.0)


@pytest.fixture
def _flag_on():
    prev = os.environ.get("DISCOPT_XEXP_ATOM")
    os.environ["DISCOPT_XEXP_ATOM"] = "1"
    yield
    if prev is None:
        os.environ.pop("DISCOPT_XEXP_ATOM", None)
    else:
        os.environ["DISCOPT_XEXP_ATOM"] = prev


def _bound(build) -> float:
    m = build()
    terms = classify_nonlinear_terms(m)
    milp, _ = build_milp_relaxation(m, terms, None, incumbent=None)
    r = milp.solve()
    return float(r.objective if r.objective is not None else r.bound)


def _xexp1(lo: float, hi: float):
    def build():
        m = Model("xe")
        x = m.continuous("x", lb=lo, ub=hi)
        m.minimize(x * dm.exp(x))
        return m

    return build


def _true_min(lo: float, hi: float) -> float:
    """Global min of t*exp(t) is at t=-1 (=-e^-1); on the box it's -1 if spanned else a corner."""
    cands = [lo, hi] + ([-1.0] if lo <= -1.0 <= hi else [])
    return min(t * math.exp(t) for t in cands)


# Boxes fully in the convex region (lo>=-2) where the curvature bites (interior min).
_CURVATURE_BOXES = [(-2.0, 1.0), (-2.0, 0.0), (-1.8, 0.5), (-1.5, 2.0)]
_ALL_BOXES = _CURVATURE_BOXES + [(0.0, 2.0)]  # + a corner-min box


@pytest.mark.parametrize("lo,hi", _ALL_BOXES)
def test_xexp_sound_and_never_looser(_flag_on, lo, hi):
    """ON is a VALID lower bound (<= true min) and NEVER looser than OFF."""
    on = _bound(_xexp1(lo, hi))
    os.environ.pop("DISCOPT_XEXP_ATOM", None)
    off = _bound(_xexp1(lo, hi))
    os.environ["DISCOPT_XEXP_ATOM"] = "1"
    tm = _true_min(lo, hi)
    assert on <= tm + 1e-5, f"UNSOUND: ON {on} > true {tm} on [{lo},{hi}]"
    assert on >= off - 1e-9, f"LOOSER than off on [{lo},{hi}]: ON={on} OFF={off}"


@pytest.mark.parametrize("lo,hi", _CURVATURE_BOXES)
def test_xexp_reaches_true_min_via_separable_floor(monkeypatch, lo, hi):
    """The single-term ``min t*exp(t)`` reaches its exact minimum with or without
    the xexp atom.

    #640 Bucket 1: the recovered separable objective floor is a GENERAL mechanism
    that recognizes ``t*exp(t) >= -1/e`` and closes the single-term bound exactly —
    subsuming the xexp atom's convex envelope for a separable objective. So where
    this test previously showed the atom strictly tighter than the shattered
    bilinear, ON and OFF now BOTH hit the true interior minimum. (The atom still
    tightens a NON-separable objective that couples ``t*exp(t)`` with other terms;
    that is out of scope for these single-term fixtures.) Assert the sound,
    tight contract the engine now delivers.
    """
    monkeypatch.setenv("DISCOPT_XEXP_ATOM", "1")
    on = _bound(_xexp1(lo, hi))
    monkeypatch.delenv("DISCOPT_XEXP_ATOM", raising=False)
    off = _bound(_xexp1(lo, hi))
    tm = _true_min(lo, hi)
    assert on == pytest.approx(tm, abs=1e-4), f"ON not tight on [{lo},{hi}]: {on} vs {tm}"
    assert off == pytest.approx(tm, abs=1e-4), f"OFF not tight on [{lo},{hi}]: {off} vs {tm}"
    assert on <= tm + 1e-5 and off <= tm + 1e-5  # both sound lower bounds


def test_xexp_inflection_guard_abstains(_flag_on):
    """A box spanning below -2 is not fully convex: the atom abstains -> ON == OFF."""
    on = _bound(_xexp1(-4.0, 1.0))
    os.environ.pop("DISCOPT_XEXP_ATOM", None)
    off = _bound(_xexp1(-4.0, 1.0))
    os.environ["DISCOPT_XEXP_ATOM"] = "1"
    assert on == off, f"atom fired on lo<-2 (non-convex box): {on} vs {off}"


def test_affine_arg_xexp_tighter(_flag_on):
    """``(a*x+b)*exp(a*x+b)`` fires the same atom via the shared affine form."""

    def build():
        m = Model("aff_xe")
        x = m.continuous("x", lb=-0.4, ub=1.0)
        t = 2.0 * x - 1.0  # t in [-1.8, 1] (clear of the -2 inflection), interior min at t=-1
        m.minimize(t * dm.exp(t))
        return m

    on = _bound(build)
    os.environ.pop("DISCOPT_XEXP_ATOM", None)
    off = _bound(build)
    os.environ["DISCOPT_XEXP_ATOM"] = "1"
    assert on <= -math.exp(-1.0) + 1e-5, f"UNSOUND: ON={on}"
    assert on > off + 0.1, f"not tighter: ON={on} OFF={off}"


def test_xexp_off_is_unchanged():
    os.environ.pop("DISCOPT_XEXP_ATOM", None)
    off1 = _bound(_xexp1(-1.0, 1.0))
    off2 = _bound(_xexp1(-1.0, 1.0))
    assert off1 == off2
    assert abs(off1 - _OFF_REF) < 1e-9, f"flag-off behavior changed: {off1} vs {_OFF_REF}"


def test_bilinear_xexpy_unaffected_when_on(_flag_on):
    """A genuine bilinear ``x*exp(y)`` (distinct vars) is NOT the xexp atom: ON == OFF."""

    def build():
        m = Model("bxy")
        x = m.continuous("x", lb=0.5, ub=2.0)
        y = m.continuous("y", lb=-1.0, ub=1.0)
        m.minimize(x * dm.exp(y))
        return m

    on = _bound(build)
    os.environ.pop("DISCOPT_XEXP_ATOM", None)
    off = _bound(build)
    os.environ["DISCOPT_XEXP_ATOM"] = "1"
    assert on == off, f"xexp flag changed a genuine bilinear x*exp(y): {on} vs {off}"


def test_xexp_full_solve_still_proves(_flag_on):
    m = Model("xe_solve")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    m.minimize(x * dm.exp(x))
    r = m.solve(time_limit=20)
    assert r.status == "optimal"
    assert r.objective is not None
    assert abs(r.objective - (-math.exp(-1.0))) < 1e-3
