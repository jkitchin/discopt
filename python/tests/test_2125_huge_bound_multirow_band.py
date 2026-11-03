"""Huge-but-finite bounds on a MULTI-ROW LP must certify, not error (issue #2125).

#937 pinned "the simplex certifies the whole band [1e3, 9.999e19]" on one LP shape:
a 2-column fixture whose every column sits at the huge bound. On a real multi-row LP
the band was not certified. The issue measured it on 40 mbigm M-value LPs from GDPlib
``batch_processing`` (200x260 / 40x260, 20 columns with a default-box side): OPTIMAL
40/40 up to 1e16, ERROR 36/40 at 3e16 and 40/40 from 5e16, while HiGHS solved all.

Measured here on the seeded LP below (60 <= rows, 12 = rows, 80 columns, 8 of them
with a huge side), before the fix, two independent mechanisms:

* **Huge bound ACTIVE at the optimum (from ~1e11).** The column sits at ``u`` and
  its ``<=`` rows' slacks are basic at ``~|a*u|``, whose exact vertex value is not
  an ``f64``: the residual it leaves passes ``1e-6*(1 + |b|)`` once ``|s| > ~1e10``.
  The #1335 row floor takes the row's *minimum* grain, set by its moderate columns,
  so it forgave nothing: ``Rows`` audit failure, ERROR on 4/40 at 1e11.
* **Huge bound INACTIVE (from ~1e15, the issue's own class).** A cold solve parks
  the column at ``l = -u``; phase 1 then cancels ``~u``-sized residuals and leaves
  ``~eps*u`` absolute error in the basics it touched (``Phase1DriftDominates``):
  ERROR on 0/20 at 1e9, 15/20 at 1e17, 20/20 at 9.999e19.

The fixes are a singleton-absorption term in the row audit and a relax-and-verify
cold solve over the huge sides (``solve_huge_box_relaxed``). Each case is compared
against HiGHS, and every comparison is counted (CLAUDE.md section 6).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("discopt._rust")
linprog = pytest.importorskip("scipy.optimize").linprog

from discopt import _rust  # noqa: E402
from discopt.solvers import SolveStatus  # noqa: E402
from discopt.solvers.lp_simplex import solve_lp as simplex_lp  # noqa: E402

#: The band of the issue, plus the decade below it where the active-bound
#: mechanism starts. 9.999e19 is the default ``Model.continuous`` box.
_BAND = [1e10, 1e12, 1e15, 1e16, 3e16, 1e17, 1e19, 9.999e19]

#: How the 8 huge columns get their huge side: lower only, both, or a mix.
_MODES = ["lo", "both", "mix"]

_SEEDS = range(8)


def _seeded_lp(seed: int, u: float, mode: str):
    """A bounded, feasible LP whose moderate optimum does not depend on ``u``.

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


@pytest.mark.smoke
def test_multirow_band_agrees_with_highs(monkeypatch):
    """Every case HiGHS solves certifies at HiGHS's objective, except a known class.

    The exception is a huge bound that is ACTIVE at the optimum with ``u >= 1e16``:
    there the vertex needs ``~u``-sized values in the basis and the simplex
    returns ERROR on some instances (#2125's remaining work, below). Even there it
    must never certify a wrong objective, and it must never call the LP anything
    but optimal.
    """
    monkeypatch.setenv("DISCOPT_PROFILE", "1")
    _rust.profile_reset_py()
    compared = 0
    certified = 0
    active_errors = 0
    for mode in _MODES:
        for u in _BAND:
            for seed in _SEEDS:
                lp = _seeded_lp(seed, u, mode)
                h = _highs(lp)
                assert h.status == 0, f"fixture must be bounded and feasible ({mode} {u:g} {seed})"
                r = simplex_lp(**lp)
                tag = f"mode={mode} u={u:.4g} seed={seed}"
                compared += 1
                # Never a wrong certificate, at any magnitude.
                assert r.status in (SolveStatus.OPTIMAL, SolveStatus.ERROR), (
                    f"{tag}: a bounded feasible LP came back {r.status.value}"
                )
                if r.status == SolveStatus.OPTIMAL:
                    assert r.objective == pytest.approx(h.fun, rel=1e-6, abs=1e-6), tag
                    certified += 1
                    continue
                active = float(np.max(np.abs(h.x))) > 1e6
                assert active and u >= 1e16, (
                    f"{tag}: ERROR on a case the fix covers (huge bound "
                    f"{'active' if active else 'inactive'} at the optimum)"
                )
                active_errors += 1
    assert compared == len(_MODES) * len(_BAND) * len(_SEEDS), "sweep did not run every case"
    # The remaining class is a minority; this fails if a regression widens it.
    assert active_errors <= compared // 20, f"{active_errors}/{compared} ERROR"
    assert certified >= compared - active_errors
    counters = dict(_rust.profile_counters_py())
    # The relax-and-verify path fired and decided cases (the probe is not a no-op).
    assert counters["HugeBoxRelaxDecided"] > 0, counters
    assert counters["HugeBoxRelaxArmed"] == (
        counters["HugeBoxRelaxDecided"] + counters["HugeBoxRelaxFellBack"]
    ), counters


@pytest.mark.smoke
def test_active_huge_bound_certifies_up_to_1e15():
    """The ACTIVE-bound mechanism, isolated: the column ends at ``x = u``.

    Seed 5 of the ``[-5, u]`` fixture puts column 76 at ``x = u`` (it appears only
    in three ``<=`` rows, with negative coefficients, and has a negative cost). Its
    row slacks are basic at ``~2u``. Pre-fix this was ERROR from ``u = 1e11`` —
    the row audit refused a residual no ``f64`` slack could avoid.
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
