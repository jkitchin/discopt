"""Relative-entropy atom (issue #632 adjacent-atom family), gated DISCOPT_RELENT_ATOM.

``x*log(x/y)`` is JOINTLY convex on x,y>0 (the perspective of ``x log x``), but the
factorable path relaxes it as ``x * log(x/y)`` (a bilinear against the concave log)
-> loose. The gated atom emits joint tangent-plane underestimators. Tests pin
soundness (LB <= true), tightness (exact where x=y forces D=0), byte-identical off,
and that a bare bilinear x*y is untouched.
"""

from __future__ import annotations

import math
import os

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt._jax.milp_relaxation import build_milp_relaxation
from discopt._jax.term_classifier import classify_nonlinear_terms

_OFF_REF_XEQY = -2.1698520434920034


@pytest.fixture
def _flag_on():
    prev = os.environ.get("DISCOPT_RELENT_ATOM")
    os.environ["DISCOPT_RELENT_ATOM"] = "1"
    yield
    if prev is None:
        os.environ.pop("DISCOPT_RELENT_ATOM", None)
    else:
        os.environ["DISCOPT_RELENT_ATOM"] = prev


def _bound(build):
    m = build()
    terms = classify_nonlinear_terms(m)
    milp, _ = build_milp_relaxation(m, terms, None, incumbent=None)
    r = milp.solve()
    return r.objective if r.objective is not None else r.bound


def _relent_on_line(xl, yl, s):
    """min x*log(x/y) s.t. x+y=s over a box; the true min is sampled on the line."""

    def build():
        m = Model("re")
        x = m.continuous("x", lb=xl, ub=xl + 1.0)
        y = m.continuous("y", lb=yl, ub=yl + 1.0)
        m.subject_to(x + y == s)
        m.minimize(x * dm.log(x / y))
        return m

    return build


def _true_min_on_line(xl, yl, s):
    lo = max(xl, s - (yl + 1.0))
    hi = min(xl + 1.0, s - yl)
    xs = np.linspace(lo, hi, 400)
    return min(xx * math.log(xx / (s - xx)) for xx in xs if 0 < s - xx and xx > 0)


@pytest.mark.parametrize(
    "xl,yl,s",
    [(0.5, 0.5, 2.0), (0.3, 0.8, 2.0), (1.0, 1.5, 3.0), (0.4, 0.4, 1.6)],
)
def test_relent_sound(_flag_on, xl, yl, s):
    """ON is a valid lower bound: never above the true min sampled on the line."""
    on = _bound(_relent_on_line(xl, yl, s))
    tm = _true_min_on_line(xl, yl, s)
    assert isinstance(on, (int, float))
    assert on <= tm + 1e-3, f"UNSOUND: ON {on} > true {tm} on ({xl},{yl},{s})"


def test_relent_exact_when_x_equals_y(_flag_on):
    """x=y forces D = x log 1 = 0; the joint tangent set is exact and tighter than off."""

    def build():
        m = Model("re_eq")
        x = m.continuous("x", lb=0.5, ub=2.0)
        y = m.continuous("y", lb=0.5, ub=2.0)
        m.subject_to(x == y)
        m.minimize(x * dm.log(x / y))
        return m

    on = _bound(build)
    os.environ.pop("DISCOPT_RELENT_ATOM", None)
    off = _bound(build)
    os.environ["DISCOPT_RELENT_ATOM"] = "1"
    assert abs(on - 0.0) < 1e-3, f"not exact: ON={on}"
    assert on > off + 1.0, f"not tighter: ON={on} OFF={off}"


def test_relent_off_is_unchanged():
    os.environ.pop("DISCOPT_RELENT_ATOM", None)

    def build():
        m = Model("re_eq")
        x = m.continuous("x", lb=0.5, ub=2.0)
        y = m.continuous("y", lb=0.5, ub=2.0)
        m.subject_to(x == y)
        m.minimize(x * dm.log(x / y))
        return m

    off1 = _bound(build)
    off2 = _bound(build)
    assert off1 == off2
    assert abs(off1 - _OFF_REF_XEQY) < 1e-9


def test_bilinear_unaffected_when_on(_flag_on):
    """A bare bilinear x*y is NOT relative entropy: ON == OFF exactly."""

    def build():
        m = Model("bil")
        x = m.continuous("x", lb=0.5, ub=2.0)
        y = m.continuous("y", lb=0.5, ub=2.0)
        m.minimize(x * y)
        return m

    on = _bound(build)
    os.environ.pop("DISCOPT_RELENT_ATOM", None)
    off = _bound(build)
    os.environ["DISCOPT_RELENT_ATOM"] = "1"
    assert on == off, f"relent flag changed a bilinear: {on} vs {off}"
