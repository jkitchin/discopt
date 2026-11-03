"""Huge-but-finite bounds on a MULTI-ROW LP must certify, not error (issue #2125).

#937 pinned "the simplex certifies the whole band [1e3, 9.999e19]" on one LP shape:
a 2-column fixture whose every column sits at the huge bound. On a real multi-row LP
the band was not certified. The issue measured it on 40 mbigm M-value LPs from GDPlib
``batch_processing`` (200x260 / 40x260, 20 columns with a default-box side): OPTIMAL
40/40 up to 1e16, ERROR 36/40 at 3e16 and 40/40 from 5e16, while HiGHS solved all.

Measured here on the seeded LP below (60 <= rows, 12 = rows, 80 columns, 8 of them
with a huge side). On main, 72 of 192 sweep cases (3 modes x 8 magnitudes x 8 seeds)
agreed with HiGHS; the rest were ERROR. Three mechanisms, each with its own fix:

* **Huge bound ACTIVE, slack rounding (from ~1e11).** The column sits at ``u`` and
  its ``<=`` rows' slacks are basic at ``~|a*u|``, whose exact vertex value is not
  an ``f64``. The #1335 row floor takes the row's *minimum* grain, set by its
  moderate columns, so it forgave nothing. Fix: ``singleton_absorbs`` in the row
  audit.
* **Huge bound INACTIVE (from ~1e15, the issue's own class).** A cold solve parks
  the column at ``l = -u``; phase 1 cancels ``~u``-sized residuals and leaves
  ``~eps*u`` absolute error in the basics it touched. Fix: the relax-and-verify cold
  solve ``solve_huge_box_relaxed``, plus a dual-simplex cleanup of the basis.
* **Huge bound ACTIVE, drift (from ~1e16).** With basic slacks at ``~2e20`` the
  moderate basics carry ``~eps*1e20`` error; refined and unrefined ``x_B`` of one
  basis differed by ~3e3. Fix: ``solve_huge_active_fixed`` fixes the column at its
  bound, removes the rows it makes slack, and certifies the result for the original
  LP from duals.

With all three, that 192-case sweep certifies 192/192, and a wider run (3200 cases, a second
generator with random shapes, densities and box widths) certified 3199/3200 with every
returned point independently feasible and at HiGHS's objective. The one refusal is
pinned below: its exact optimal vertex is not representable in ``f64``.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("discopt._rust")
linprog = pytest.importorskip("scipy.optimize").linprog

from discopt import _rust  # noqa: E402
from discopt.solvers import SolveStatus  # noqa: E402
from discopt.solvers.lp_simplex import solve_lp as simplex_lp  # noqa: E402

#: The band of the issue, plus the decades below it where the active-bound
#: mechanism starts. 9.999e19 is the default ``Model.continuous`` box.
_BAND = [1e10, 1e12, 1e15, 1e16, 3e16, 1e17, 1e19, 9.999e19]

#: How the 8 huge columns get their huge side.
_MODES = ["lo", "hi", "both", "mix"]

#: Seeds swept. On main this subset already fails 75 of its 128 cases (measured);
#: it is kept small for the fast lane's per-test time budget.
_SEEDS = range(4)


def _seeded_lp(seed: int, u: float, mode: str):
    """A bounded, feasible LP whose moderate part does not depend on ``u``.

    ``A_ub x <= b_ub`` and ``A_eq x = b_eq`` are built around an interior point
    ``x0`` so the LP is feasible; every column has a box ``[-5, 5]`` except 8, whose
    lower side (``lo``), upper side (``hi``), both sides (``both``), or one side at
    random (``mix``) is pushed out to ``u``.
    """
    rng = np.random.default_rng(seed)
    m, me, n = 60, 12, 80
    rvs = lambda k: rng.uniform(-3.0, 3.0, k)  # noqa: E731
    A = sp.random(m, n, density=0.08, random_state=rng, data_rvs=rvs).tocsr()
    Ae = sp.random(me, n, density=0.08, random_state=rng, data_rvs=rvs).tocsr()
    x0 = rng.uniform(0.0, 1.0, n)
    b = A @ x0 + rng.uniform(0.1, 1.0, m)
    be = Ae @ x0
    c = rng.uniform(-1.0, 1.0, n)
    bounds = [(-5.0, 5.0)] * n
    for j in rng.choice(n, 8, replace=False):
        lower_side = rng.random() < 0.5
        if mode == "lo":
            bounds[j] = (-u, 5.0)
        elif mode == "hi":
            bounds[j] = (-5.0, u)
        elif mode == "both":
            bounds[j] = (-u, u)
        else:
            bounds[j] = (-u, 5.0) if lower_side else (-5.0, u)
    return dict(c=c, A_ub=A, b_ub=b, A_eq=Ae, b_eq=be, bounds=bounds)


def _highs(lp):
    return linprog(
        lp["c"],
        A_ub=lp["A_ub"],
        b_ub=lp["b_ub"],
        A_eq=lp["A_eq"],
        b_eq=lp["b_eq"],
        bounds=lp["bounds"],
        method="highs",
    )


def _independently_feasible(lp, x) -> bool:
    """Box and row feasibility of ``x``, checked here rather than by the engine.

    Rows get the engine's ``1e-6 * (1 + |b|)`` plus four ulps of the row's own
    magnitude ``sum |a_ij x_j|`` (the float evaluation of a row holding ``1e19``
    terms cannot be trusted below that).
    """
    x = np.asarray(x, dtype=np.float64)
    box = np.asarray(lp["bounds"], dtype=np.float64)
    tol = 1e-6 * (1.0 + np.abs(box).max(axis=1))
    if np.any(x < box[:, 0] - tol) or np.any(x > box[:, 1] + tol):
        return False
    eps4 = 4.0 * np.finfo(float).eps
    r_ub = lp["A_ub"] @ x - lp["b_ub"]
    m_ub = abs(lp["A_ub"]) @ np.abs(x)
    r_eq = lp["A_eq"] @ x - lp["b_eq"]
    m_eq = abs(lp["A_eq"]) @ np.abs(x)
    return bool(
        np.all(r_ub <= 1e-6 * (1 + np.abs(lp["b_ub"])) + eps4 * m_ub)
        and np.all(np.abs(r_eq) <= 1e-6 * (1 + np.abs(lp["b_eq"])) + eps4 * m_eq)
    )


@pytest.mark.smoke
def test_multirow_band_certifies_and_agrees_with_highs(monkeypatch):
    """Every case certifies, at HiGHS's objective, at an independently feasible point.

    On main 75 of these 128 cases failed (ERROR, or not at HiGHS's objective). The
    counters prove the new routes ran (CLAUDE.md section 6): the relaxation decided
    cases, and the active-bound fix certified the ones where the huge bound is
    active at the optimum.
    """
    monkeypatch.setenv("DISCOPT_PROFILE", "1")
    _rust.profile_reset_py()
    compared = 0
    for mode in _MODES:
        for u in _BAND:
            for seed in _SEEDS:
                lp = _seeded_lp(seed, u, mode)
                h = _highs(lp)
                assert h.status == 0, f"fixture must be bounded and feasible ({mode} {u:g} {seed})"
                r = simplex_lp(**lp)
                tag = f"mode={mode} u={u:.4g} seed={seed}"
                assert r.status == SolveStatus.OPTIMAL, f"{tag}: got {r.status.value}"
                assert r.objective == pytest.approx(h.fun, rel=1e-6, abs=1e-6), tag
                assert _independently_feasible(lp, r.x), f"{tag}: certified an infeasible point"
                compared += 1
    assert compared == len(_MODES) * len(_BAND) * len(_SEEDS), "sweep did not run every case"
    counters = dict(_rust.profile_counters_py())
    assert counters["HugeBoxRelaxDecided"] > 0, counters
    assert counters["HugeBoxFixActiveDecided"] > 0, counters
    assert counters["HugeBoxRelaxArmed"] == (
        counters["HugeBoxRelaxDecided"] + counters["HugeBoxRelaxFellBack"]
    ), counters


@pytest.mark.smoke
def test_active_huge_bound_certifies_up_to_1e15():
    """The slack-rounding mechanism, isolated: the column ends at ``x = u``.

    Seed 5 of the ``[-5, u]`` fixture puts column 76 at ``x = u`` (it appears only
    in three ``<=`` rows, with negative coefficients, and has a negative cost). Its
    row slacks are basic at ``~2u``. On main this was ERROR from ``u = 1e11``: the
    row audit refused a residual no ``f64`` slack could avoid.
    """
    checked = 0
    for u in [1e11, 1e12, 1e13, 1e14, 1e15]:
        lp = _seeded_lp(5, u, "hi")
        h = _highs(lp)
        assert h.status == 0 and abs(h.x[76] - u) <= 1e-9 * u, "fixture: x76 must sit at u"
        r = simplex_lp(**lp)
        assert r.status == SolveStatus.OPTIMAL, f"u={u:g}: got {r.status.value}"
        assert r.objective == pytest.approx(h.fun, rel=1e-9), f"u={u:g}"
        checked += 1
    assert checked == 5


@pytest.mark.smoke
def test_unrepresentable_vertex_is_refused_not_misreported():
    """The one refusal in the wider run, and why refusing is the correct answer.

    ``both``/``u = 1e19``/seed 15: column 37 sits at ``u = 1e19`` and column 11 is
    *interior* at ``~-8.8e18``, fixed by a BINDING ``<=`` row (row 19) it shares
    with column 37. ``ulp(8.8e18) = 2048``, so no ``f64`` value of ``x_11`` meets
    that row to anywhere near ``1e-6``. HiGHS reports ``optimal``, but its own point
    violates row 19 by about 371 in exact arithmetic (its float residual reads
    ``-22``). The simplex refuses (ERROR) rather than certify a point that breaks a
    row. What it must never do is return a wrong verdict, or an "optimal" point
    that violates a row.
    """
    lp = _seeded_lp(15, 1e19, "both")
    h = _highs(lp)
    assert h.status == 0
    row = lp["A_ub"].getrow(19)
    exact = sum(
        Fraction(float(a)) * Fraction(float(h.x[j])) for j, a in zip(row.indices, row.data)
    ) - Fraction(float(lp["b_ub"][19]))
    assert exact > 100, f"fixture: HiGHS's point should violate row 19, got {float(exact)}"
    r = simplex_lp(**lp)
    assert r.status in (SolveStatus.OPTIMAL, SolveStatus.ERROR), r.status.value
    if r.status == SolveStatus.OPTIMAL:
        assert _independently_feasible(lp, r.x), "an optimal point must satisfy its rows"
        assert r.objective == pytest.approx(h.fun, rel=1e-9)
