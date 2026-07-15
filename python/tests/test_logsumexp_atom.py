"""Log-sum-exp atom (issue #632 adjacent-atom family), gated DISCOPT_LOGSUMEXP_ATOM.

``log(sum_i exp(t_i))`` is CONVEX in the exp arguments, but the factorable path
relaxes the outer ``log`` as a CONCAVE atom over the exp-sum — the wrong curvature,
so the underestimator collapses to a loose floor (root gap ~2.7). The gated atom
emits the exact convex OA: softmax-gradient tangent underestimators. Tests pin:
(1) SOUND — the relaxation LB never exceeds the true box minimum (a cut feasible
point would make it falsely tight); (2) TIGHTER — dramatically so where the
curvature bites (exact on the symmetric cases); (3) byte-identical when off and on
a non-LSE ``log``.
"""

from __future__ import annotations

import math
import os

import discopt.modeling as dm
import pytest
from discopt import Model
from discopt._jax.milp_relaxation import build_milp_relaxation
from discopt._jax.term_classifier import classify_nonlinear_terms

# Frozen pre-atom (flag-off) bound for log(exp x + exp y) on [-1,1]^2.
_OFF_REF = -0.30685281944005516


@pytest.fixture
def _flag_on():
    prev = os.environ.get("DISCOPT_LOGSUMEXP_ATOM")
    os.environ["DISCOPT_LOGSUMEXP_ATOM"] = "1"
    yield
    if prev is None:
        os.environ.pop("DISCOPT_LOGSUMEXP_ATOM", None)
    else:
        os.environ["DISCOPT_LOGSUMEXP_ATOM"] = prev


def _root_lb(build) -> float:
    m = build()
    terms = classify_nonlinear_terms(m)
    milp, _ = build_milp_relaxation(m, terms, None, incumbent=None)
    return float(milp.solve().objective)


def _lse2(lo: float, hi: float):
    def build():
        m = Model("lse2")
        x = m.continuous("x", lb=lo, ub=hi)
        y = m.continuous("y", lb=lo, ub=hi)
        m.minimize(dm.log(dm.exp(x) + dm.exp(y)))
        return m

    return build


@pytest.mark.parametrize("lo,hi", [(-2, 0), (-1, 1), (0, 2), (-3, 3), (1, 4), (-0.5, 0.5)])
def test_lse_sound_and_never_looser(_flag_on, lo, hi):
    """ON is a VALID lower bound (<= true min = corner value) and never looser than OFF."""
    on = _root_lb(_lse2(lo, hi))
    os.environ.pop("DISCOPT_LOGSUMEXP_ATOM", None)
    off = _root_lb(_lse2(lo, hi))
    os.environ["DISCOPT_LOGSUMEXP_ATOM"] = "1"
    # log(exp x + exp y) is increasing in both -> min at the lower corner x=y=lo.
    true_min = math.log(2.0 * math.exp(lo))
    assert on <= true_min + 1e-5, f"UNSOUND: ON {on} > true {true_min} on [{lo},{hi}]"
    assert on >= off - 1e-9, f"LOOSER than off on [{lo},{hi}]: ON={on} OFF={off}"


@pytest.mark.parametrize("n,k", [(2, 2.0), (3, 3.0), (4, 0.0)])
def test_lse_exact_on_symmetric_interior(_flag_on, n, k):
    """With sum_i x_i = k the min is interior (x_i = k/n); the convex OA is exact there."""

    def build():
        m = Model("lsen")
        x = m.continuous("x", shape=n, lb=-3.0, ub=5.0)
        m.subject_to(dm.sum(x[i] for i in range(n)) == k)
        m.minimize(dm.log(dm.sum(dm.exp(x[i]) for i in range(n))))
        return m

    on = _root_lb(build)
    os.environ.pop("DISCOPT_LOGSUMEXP_ATOM", None)
    off = _root_lb(build)
    os.environ["DISCOPT_LOGSUMEXP_ATOM"] = "1"
    true_min = math.log(n) + k / n  # log(n * exp(k/n))
    assert on <= true_min + 1e-5  # sound
    assert abs(on - true_min) < 1e-3, f"not exact: ON={on} true={true_min}"
    assert on > off + 1.0, f"not dramatically tighter: ON={on} OFF={off}"


def test_softplus_recognized(_flag_on):
    """softplus log(1+exp x) = LSE(0, x): the additive constant folds in as exp(log 1)."""

    def build():
        m = Model("sp")
        x = m.continuous("x", lb=-3.0, ub=3.0)
        m.minimize(dm.log(1 + dm.exp(x)) - 0.5 * x)  # interior min at x=0 -> log 2
        return m

    on = _root_lb(build)
    os.environ.pop("DISCOPT_LOGSUMEXP_ATOM", None)
    off = _root_lb(build)
    os.environ["DISCOPT_LOGSUMEXP_ATOM"] = "1"
    assert on <= math.log(2.0) + 1e-5  # sound
    assert on > off + 0.5, f"softplus not tightened: ON={on} OFF={off}"


def test_off_is_unchanged():
    os.environ.pop("DISCOPT_LOGSUMEXP_ATOM", None)
    off1 = _root_lb(_lse2(-1, 1))
    off2 = _root_lb(_lse2(-1, 1))
    assert off1 == off2
    assert abs(off1 - _OFF_REF) < 1e-9, f"flag-off behavior changed: {off1} vs {_OFF_REF}"


def test_plain_log_unaffected_when_on(_flag_on):
    """A bare log(x) is NOT log(sum exp): ON == OFF exactly (atom does not fire)."""

    def build():
        m = Model("logx")
        x = m.continuous("x", lb=0.5, ub=3.0)
        m.maximize(dm.log(x))
        return m

    on = _root_lb(build)
    os.environ.pop("DISCOPT_LOGSUMEXP_ATOM", None)
    off = _root_lb(build)
    os.environ["DISCOPT_LOGSUMEXP_ATOM"] = "1"
    assert on == off, f"logsumexp flag changed a plain log: {on} vs {off}"


def test_full_solve_still_proves(_flag_on):
    m = Model("lse_solve")
    x = m.continuous("x", lb=-2.0, ub=4.0)
    y = m.continuous("y", lb=-2.0, ub=4.0)
    m.subject_to(x + y == 2.0)
    m.minimize(dm.log(dm.exp(x) + dm.exp(y)))
    r = m.solve(time_limit=20)
    assert r.status == "optimal"
    assert r.objective is not None
    assert abs(r.objective - (math.log(2.0) + 1.0)) < 1e-3
