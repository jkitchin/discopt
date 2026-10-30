"""#2054: OA's cut classifier retries an abstained row on a cover of the box.

The whole-box interval Hessian of ``1/(exp(a*t) - 1)`` (a PFR reactor row) and of
the two-CSTR rational row encloses negative curvature on ``t in [0.25, 4]``
although both are convex there, so ``classify_oa_cut_convexity`` gave them no OA
cuts and LOA's master treated the reactors as lossless. Bisecting the box until
every piece is certified proves both (PSD on every piece of a cover is PSD on the
box). ``DISCOPT_OA_CONVEXITY_SUBDIVISION``, default ON since its §5 panel; ``=0`` opts out.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.convexity import Curvature
from discopt._relax.convexity.certificate import (
    _oa_convexity_subdivision_enabled,
    certify_convex,
    certify_convex_subdivided,
)
from discopt._relax.convexity.rules import classify_oa_cut_convexity

KAP, TH = 0.25, (0.25, 4.0)
FLAG = "DISCOPT_OA_CONVEXITY_SUBDIVISION"


def u_cstr2(th, ops):
    """Two CSTRs in series: A left unconverted per B made."""
    h = th / 2
    return (1 + KAP * h) ** 2 / (h * (2 + (1 + KAP) * h))


def u_pfr(th, ops):
    """PFR: (1 - kappa) / (exp((1 - kappa) theta) - 1)."""
    return (1 - KAP) / (ops.exp((1 - KAP) * th) - 1)


def _row_model(f, lb=TH[0], ub=TH[1]):
    m = dm.Model("row")
    th = m.continuous("theta", lb=lb, ub=ub)
    nu = m.continuous("NU", lb=0, ub=1e3)
    m.subject_to(50 * f(th, dm) - nu <= 0)
    m.minimize(nu)
    return m


def _numerically_convex(f, lb=TH[0], ub=TH[1]) -> bool:
    g = np.linspace(lb, ub, 2001)
    y = f(g, np)
    return bool((y[2:] - 2 * y[1:-1] + y[:-2] >= -1e-12).all())


def test_flag_defaults_on_with_an_opt_out(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert _oa_convexity_subdivision_enabled()
    monkeypatch.setenv(FLAG, "0")
    assert not _oa_convexity_subdivision_enabled()


@pytest.mark.parametrize("f", [u_cstr2, u_pfr], ids=["CSTR2", "PFR"])
def test_reactor_rows_are_certified_only_with_the_flag(f, monkeypatch):
    assert _numerically_convex(f)
    m = _row_model(f)
    # The premise: the whole-box certificate abstains.
    assert certify_convex(m._constraints[0].body, m) is None

    monkeypatch.setenv(FLAG, "0")
    off = classify_oa_cut_convexity(m)
    assert off.constraint_mask == [False] and off.subdivided == 0

    monkeypatch.setenv(FLAG, "1")
    on = classify_oa_cut_convexity(m)
    assert on.constraint_mask == [True] and on.subdivided == 1


@pytest.mark.parametrize("f", [u_cstr2, u_pfr], ids=["CSTR2", "PFR"])
def test_negated_reactor_row_is_concave_and_gets_no_cut(f, monkeypatch):
    m = dm.Model("neg")
    th = m.continuous("theta", lb=TH[0], ub=TH[1])
    nu = m.continuous("NU", lb=-1e3, ub=1e3)
    m.subject_to(-50 * f(th, dm) - nu <= 0)
    body = m._constraints[0].body
    assert certify_convex_subdivided(body, m) == Curvature.CONCAVE
    monkeypatch.setenv(FLAG, "1")
    assert classify_oa_cut_convexity(m).constraint_mask == [False]
    # ... and the ``>=`` form of the same concave body is a convex set.
    m2 = dm.Model("neg_ge")
    th2 = m2.continuous("theta", lb=TH[0], ub=TH[1])
    nu2 = m2.continuous("NU", lb=-1e3, ub=1e3)
    m2.subject_to(-50 * f(th2, dm) - nu2 >= 0)
    assert classify_oa_cut_convexity(m2).constraint_mask == [True]


@pytest.mark.parametrize(
    "f, lb, ub",
    [
        # Convex on t > 0 only: the cover meets the concave half.
        (lambda t, ops: t**3, -1.0, 1.0),
        # 1/g with g = exp(exp(t)) positive, increasing and convex -- yet
        # exp(-exp(t)) is concave for t < 0. "Reciprocal of a positive increasing
        # convex function" is not a convexity rule.
        (lambda t, ops: 1 / ops.exp(ops.exp(t)), -2.0, 2.0),
        (lambda t, ops: ops.sin(t), 0.5, 5.5),
    ],
    ids=["cube", "exp_neg_exp", "sin"],
)
def test_nonconvex_rows_are_never_certified(f, lb, ub, monkeypatch):
    assert not _numerically_convex(f, lb, ub)
    m = _row_model(f, lb, ub)
    assert certify_convex_subdivided(m._constraints[0].body, m) != Curvature.CONVEX
    monkeypatch.setenv(FLAG, "1")
    assert classify_oa_cut_convexity(m).constraint_mask == [False]


def test_abstains_on_unbounded_split_variable():
    m = dm.Model("free")
    th = m.continuous("theta", lb=0.25)  # no upper bound
    nu = m.continuous("NU", lb=0, ub=1e3)
    m.subject_to(50 * u_pfr(th, dm) - nu <= 0)
    assert certify_convex_subdivided(m._constraints[0].body, m) is None


def test_abstains_past_the_piece_budget():
    m = _row_model(u_cstr2)
    assert certify_convex_subdivided(m._constraints[0].body, m, max_pieces=4) is None


def test_abstains_on_a_wide_curvature_support():
    m = dm.Model("wide")
    xs = [m.continuous(f"x{i}", lb=0.25, ub=4.0) for i in range(2)]
    m.subject_to(sum(u_pfr(x, dm) for x in xs) <= 10)
    body = m._constraints[0].body
    assert certify_convex_subdivided(body, m, max_dims=1) is None
    # The cover is a product: ~16 pieces per variable, so 2-D needs a larger budget.
    assert certify_convex_subdivided(body, m, max_dims=2, max_pieces=1024) == Curvature.CONVEX


def test_a_past_deadline_abstains():
    m = _row_model(u_pfr)
    assert certify_convex_subdivided(m._constraints[0].body, m, deadline=0.0) is None
