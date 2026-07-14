"""Euclidean-norm atom (issue #632 adjacent-atom family), gated DISCOPT_NORM_ATOM.

``sqrt(sum_i t_i^2)`` is convex, but the factorable path relaxes the outer ``sqrt``
as a CONCAVE atom over the square-sum -> wrong curvature (the underestimator, hence
the min bound, collapses; often the relaxation is UNBOUNDED). The gated atom emits
the convex OA: ``||t|| >= a . t`` for unit directions ``a`` (Cauchy-Schwarz) + axis
facets. Tests pin soundness (LB <= true), the unbounded->finite fix, byte-identical
off, and that a non-norm sqrt is untouched.
"""

from __future__ import annotations

import math
import os

import discopt.modeling as dm
import pytest
from discopt import Model
from discopt._jax.milp_relaxation import build_milp_relaxation
from discopt._jax.term_classifier import classify_nonlinear_terms


@pytest.fixture
def _flag_on():
    prev = os.environ.get("DISCOPT_NORM_ATOM")
    os.environ["DISCOPT_NORM_ATOM"] = "1"
    yield
    if prev is None:
        os.environ.pop("DISCOPT_NORM_ATOM", None)
    else:
        os.environ["DISCOPT_NORM_ATOM"] = prev


def _bound(build):
    m = build()
    terms = classify_nonlinear_terms(m)
    milp, _ = build_milp_relaxation(m, terms, None, incumbent=None)
    r = milp.solve()
    return r.status, (r.objective if r.objective is not None else r.bound)


def _norm_on_simplex(n: int, s: float):
    """min ||x|| s.t. sum x_i = s -> interior min |s|/sqrt(n) at x_i = s/n."""

    def build():
        m = Model("nrm")
        x = m.continuous("x", shape=n, lb=-2.0, ub=4.0)
        m.subject_to(dm.sum(x[i] for i in range(n)) == s)
        m.minimize(dm.sqrt(dm.sum(x[i] * x[i] for i in range(n))))
        return m

    return build


@pytest.mark.parametrize("n,s", [(2, 2.0), (2, 1.0), (3, 3.0), (2, -2.0), (4, 4.0)])
def test_norm_sound_and_finite(_flag_on, n, s):
    """ON is a valid LB (<= true), finite (fixes the loose-sqrt unboundedness), and
    exact on these symmetric-interior cases."""
    st, on = _bound(_norm_on_simplex(n, s))
    true_min = abs(s) / math.sqrt(n)
    assert isinstance(on, (int, float)), f"ON not finite: status={st}"
    assert on <= true_min + 1e-4, f"UNSOUND: ON {on} > true {true_min}"
    assert abs(on - true_min) < 1e-3, f"not exact on symmetric case: ON={on} true={true_min}"


def test_norm_unbounded_off_becomes_finite_on(_flag_on):
    """The loose concave-sqrt relaxation is unbounded here; the norm OA makes it finite."""
    st_on, on = _bound(_norm_on_simplex(2, 2.0))
    os.environ.pop("DISCOPT_NORM_ATOM", None)
    st_off, _off = _bound(_norm_on_simplex(2, 2.0))
    os.environ["DISCOPT_NORM_ATOM"] = "1"
    assert st_off == "unbounded"
    assert st_on == "optimal" and isinstance(on, (int, float))


def test_plain_sqrt_unaffected_when_on(_flag_on):
    """sqrt of an affine arg (not a sum of squares) is NOT the norm atom: ON == OFF."""

    def build():
        m = Model("sq")
        x = m.continuous("x", lb=0.0, ub=4.0)
        m.maximize(dm.sqrt(x + 0.1))
        return m

    _, on = _bound(build)
    os.environ.pop("DISCOPT_NORM_ATOM", None)
    _, off = _bound(build)
    os.environ["DISCOPT_NORM_ATOM"] = "1"
    assert on == off, f"norm flag changed a plain sqrt: {on} vs {off}"


def test_norm_full_solve_still_proves(_flag_on):
    m = Model("nrm_solve")
    x = m.continuous("x", lb=-2.0, ub=4.0)
    y = m.continuous("y", lb=-2.0, ub=4.0)
    m.subject_to(x + y == 2.0)
    m.minimize(dm.sqrt(x * x + y * y))
    r = m.solve(time_limit=20)
    assert r.status == "optimal"
    assert r.objective is not None
    assert abs(r.objective - math.sqrt(2.0)) < 1e-3
