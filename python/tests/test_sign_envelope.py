"""``sign`` gets an envelope in the default relaxation engine (#1873 follow-up).

``canonical_expr`` used to refuse ``sign`` ("discontinuous — no sound continuous
envelope"), so every model containing it was opaque to the relaxation and solved
with ``bound=None``. Its only envelope, ``relax_sign``, lived in the JAX relaxation
compiler retired in #1873. Measured before this change on four ``sign`` models
(``min sign(x) - x*y``, ``min x + 2 sign(x - 1/2)``, an integer ``sign`` sum, and a
``sign(x) >= 1/2`` constraint): three returned ``bound=None``, the fourth ran to its
20 s limit uncertified. After: all four certify (0.05-0.7 s).

A discontinuous function still has a convex hull. Over ``[lo, hi]`` the graph is
three pieces (``-1`` for ``t < 0``, ``0`` at ``0``, ``1`` for ``t > 0``) and its
closed hull is a polygon: the aux column's enclosure ``[sign(lo), sign(hi)]`` gives
the horizontal sides, ``_build_sign`` the slanted ones. This is BOUND-CHANGING
(CLAUDE.md §5) for the models it touches. It is not flag-gated: the models had no
bound at all before, so there is no "old bound" a corpus panel could compare
against. The in-repo ``.nl`` corpus cannot contain one: the ``.nl`` format has no
``sign`` opcode, so a corpus solve never reaches ``_build_sign``.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.uniform_relax import build_uniform_relaxation

pytestmark = [pytest.mark.smoke]

#: Boxes of the ARGUMENT ``t``: straddling 0, starting at 0, ending at 0, a point
#: box at 0, both sign-definite sides, a lopsided straddle, a tiny straddle.
BOXES = [
    (-2.0, 3.0),
    (0.0, 4.0),
    (-5.0, 0.0),
    (0.0, 0.0),
    (0.5, 2.0),
    (-3.0, -0.25),
    (-1e-3, 100.0),
    (-1e-6, 1e-6),
]

#: Affine arguments ``a*x + b`` so the rows carry a constant and a non-unit slope.
AFFINE = [(1.0, 0.0), (2.0, -1.0), (-0.5, 0.25)]


def _relaxation(a, b, tlo, thi):
    """``min sign(a*x + b)`` with ``x`` boxed so that ``t = a*x + b`` spans
    ``[tlo, thi]``; columns are ``[x, w]``."""
    xs = sorted(((tlo - b) / a, (thi - b) / a))
    m = dm.Model("sign_atom")
    x = m.continuous("x", lb=xs[0], ub=xs[1])
    m.minimize(dm.sign(a * x + b))
    rel = build_uniform_relaxation(m)
    kinds = [k for k, _ in rel.coverage.values()]
    assert kinds == ["univariate_call"], kinds
    return rel, xs


def _rows(rel):
    a = rel.model._A_ub
    if a is None:
        return np.zeros((0, rel.n_orig + rel.n_aux)), np.zeros(0)
    a = np.asarray(a.todense()) if hasattr(a, "todense") else np.asarray(a, dtype=float)
    return a, np.asarray(rel.model._b_ub, dtype=float)


def _graph_points(xlo, xhi, a, b):
    """``x`` samples including the exact jump point and its float neighbours."""
    xs = list(np.linspace(xlo, xhi, 301))
    x0 = -b / a
    if xlo <= x0 <= xhi:
        xs += [x0, np.nextafter(x0, -np.inf), np.nextafter(x0, np.inf)]
    return [x for x in xs if xlo <= x <= xhi]


def test_no_emitted_row_and_no_aux_bound_cuts_the_graph():
    """The whole risk of the change: every lifted graph point ``(x, sign(t))``
    must satisfy every row AND the aux column's bounds."""
    checked = 0
    rows_seen = 0
    for (a, b), (tlo, thi) in itertools.product(AFFINE, BOXES):
        rel, (xlo, xhi) = _relaxation(a, b, tlo, thi)
        A, rhs = _rows(rel)
        rows_seen += A.shape[0]
        wlo, whi = rel.model._bounds[rel.n_orig]
        for x in _graph_points(xlo, xhi, a, b):
            w = float(np.sign(a * x + b))
            assert wlo - 1e-12 <= w <= whi + 1e-12, (a, b, tlo, thi, x, w, wlo, whi)
            pt = np.zeros(A.shape[1])
            pt[0], pt[rel.n_orig] = x, w
            if A.shape[0]:
                worst = float(np.max(A @ pt - rhs))
                assert worst <= 1e-9 * max(1.0, abs(x)), (a, b, tlo, thi, x, w, worst)
            checked += 1
    assert rows_seen > 0, "no row was ever emitted -- the probe tested only the floor"
    assert checked > 5000, f"only {checked} graph points checked"


def _lp_range_of_w_at(rel, x):
    """``[min w, max w]`` over the relaxation with ``x`` pinned (2 columns, so a
    1-D feasibility interval in ``w``: intersect the rows by hand)."""
    A, rhs = _rows(rel)
    wlo, whi = rel.model._bounds[rel.n_orig]
    lo, hi = float(wlo), float(whi)
    for row, r in zip(A, rhs):
        cw, cx = row[rel.n_orig], row[0]
        if cw > 0:
            hi = min(hi, (r - cx * x) / cw)
        elif cw < 0:
            lo = max(lo, (r - cx * x) / cw)
    return lo, hi


@pytest.mark.parametrize("tlo,thi", [(-2.0, 3.0), (0.0, 4.0), (-5.0, 0.0)])
def test_the_rows_are_the_hull_not_just_sound(tlo, thi):
    """At the box ends the relaxation must pin ``w`` to the function value -- the
    slanted sides pass through ``(lo, sign(lo))`` and ``(hi, sign(hi))`` -- and at
    ``t = 0`` it must still admit the closure's whole vertical segment (the hull
    contains ``(0, -1)`` and ``(0, 1)`` as limits; a relaxation excluding them would
    cut graph points arbitrarily close to them)."""
    rel, (xlo, xhi) = _relaxation(1.0, 0.0, tlo, thi)
    for x in (xlo, xhi):
        lo, hi = _lp_range_of_w_at(rel, x)
        if x != 0.0:
            assert lo == pytest.approx(np.sign(x), abs=1e-12)
            assert hi == pytest.approx(np.sign(x), abs=1e-12)
    lo0, hi0 = _lp_range_of_w_at(rel, 0.0)
    assert lo0 == pytest.approx(-1.0 if tlo < 0 else 0.0, abs=1e-12)
    assert hi0 == pytest.approx(1.0 if thi > 0 else 0.0, abs=1e-12)


def test_the_enclosure_is_sign_of_the_endpoints():
    from discopt._relax.convexity.interval_eval import evaluate_interval

    for lo, hi in BOXES:
        m = dm.Model("enc")
        x = m.continuous("x", lb=lo, ub=hi)
        e = dm.sign(x)
        enc = evaluate_interval(e, m)
        assert (float(enc.lo), float(enc.hi)) == (np.sign(lo), np.sign(hi)), (lo, hi)


def _brute(fn, grids):
    return min(fn(*p) for p in itertools.product(*grids))


def _ints(lo, hi):
    return range(lo, hi + 1)


#: name -> (model builder, the true optimum, how it was derived)
CASES = {
    # min sign(x) - x*y on [-2,3]x[-1,1]: at x=-2,y=-1 -> -1 - 2 = -3.
    "bilinear": -3.0,
    # min x + 2 sign(x - 1/2) on [0, 1]: x = 0 -> -2.
    "shifted": -2.0,
    # integer sum, by enumeration (below).
    "integer": _brute(
        lambda a, b, c: np.sign(a - 1) - 2 * np.sign(b - 2) + 0.5 * np.sign(c) + 0.1 * a * b,
        [_ints(0, 3)] * 3,
    ),
    # sign(x) >= 1/2 forces x >= 1; then min (y-x)^2 + y is x - 1/4 -> 0.75.
    "constraint": 0.75,
    # sign(x) == 0 forces x == 0: (0 - 1)^2.
    "equality_at_jump": 1.0,
    # -sign(x - y) + 0.1 (x-y)^2 on [0,1]^2: x > y gives -1 + small -> inf -1.
    "two_var_arg": -1.0,
    # 3 sign(x^2 - 1) + x on [-2,2]: x in (-1,1) gives -3 + x -> inf -4 (not attained).
    "nonlinear_arg": -4.0,
}


def _build(name):
    m = dm.Model(name)
    if name == "bilinear":
        x = m.continuous("x", lb=-2, ub=3)
        y = m.continuous("y", lb=-1, ub=1)
        m.minimize(dm.sign(x) - x * y)
    elif name == "shifted":
        x = m.continuous("x", lb=0, ub=1)
        m.minimize(x + 2 * dm.sign(x - 0.5))
    elif name == "integer":
        z = m.integer("z", shape=(3,), lb=0, ub=3)
        m.minimize(
            dm.sign(z[0] - 1) - 2 * dm.sign(z[1] - 2) + 0.5 * dm.sign(z[2]) + 0.1 * z[0] * z[1]
        )
    elif name == "constraint":
        x = m.integer("x", lb=-3, ub=3)
        y = m.continuous("y", lb=-3, ub=3)
        m.subject_to(dm.sign(x) >= 0.5)
        m.minimize((y - x) ** 2 + y)
    elif name == "equality_at_jump":
        x = m.continuous("x", lb=-2, ub=2)
        m.subject_to(dm.sign(x) == 0)
        m.minimize((x - 1) ** 2)
    elif name == "two_var_arg":
        x = m.continuous("x", lb=0, ub=1)
        y = m.continuous("y", lb=0, ub=1)
        m.minimize(-dm.sign(x - y) + 0.1 * (x - y) ** 2)
    elif name == "nonlinear_arg":
        x = m.continuous("x", lb=-2, ub=2)
        m.minimize(3 * dm.sign(x**2 - 1) + x)
    return m


@pytest.mark.parametrize("name", sorted(CASES))
def test_sign_models_certify_and_the_bound_never_crosses_the_optimum(name):
    """Before: ``bound=None`` (or a time-limit run). After: certified, with the
    certificate invariant ``bound <= true optimum <= objective`` (min sense).
    ``nonlinear_arg`` has an infimum that is NOT attained, the hardest case for a
    discontinuous term: its bound must still not exceed the infimum."""
    true_opt = float(CASES[name])
    r = _build(name).solve(time_limit=60)
    assert r.bound is not None, f"{name}: no dual bound"
    assert r.bound <= true_opt + 1e-6 * max(1.0, abs(true_opt)), (name, r.bound, true_opt)
    assert r.objective is not None
    assert r.objective >= true_opt - 1e-6 * max(1.0, abs(true_opt)), (name, r.objective)
    assert r.gap_certified, (name, r.status, r.bound, r.objective)
    assert r.objective == pytest.approx(true_opt, abs=1e-4, rel=1e-4)
