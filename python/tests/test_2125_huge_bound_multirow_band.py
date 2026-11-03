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

With all three, that 192-case sweep certifies 192/192. A wider run (3200 cases, a second
generator with random shapes, densities and box widths), with every returned point
checked in EXACT rational arithmetic: 3191 certified at an exactly feasible point at
HiGHS's objective; 1 certified at a point whose binding row is off by 3.6e-5 absolute
(4e-17 relative to its ~1e12 terms: product rounding, which the audit's compensated sum
does not see); 8 refused, all one instance (``both``, seed 15, from ``u = 1e15``),
whose vertex has a huge INTERIOR value fixed by a binding row. HiGHS's own point
violates that row in 5 of the 8; in 3 its rounding happens to land on an exactly
feasible point that this engine does not find. That refusal is pinned below.

(An earlier version of this docstring said "3199/3200, every point independently
feasible". The check behind it allowed ``4*eps*sum|a_ij x_j|`` per row, ~1e4 absolute
at 1e19, and passed points an unaudited warm path had returned. Retracted, CLAUDE.md
section 11.)
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
    """Box and row feasibility of ``x`` in EXACT rational arithmetic, not the engine's.

    A ``<=`` row must hold to ``1e-6 * (1 + |b|)`` exactly (its slack is implicit, so
    rounding is no excuse). An equality row gets the same plus the engine's floor,
    twice the finest grain ``|a_ij| * 2 eps |x_j|`` among its columns. An earlier
    version allowed ``4 eps * sum |a_ij x_j|`` per row, about ``1e4`` absolute at
    ``1e19``, and passed points that violated their rows.
    """
    x = np.asarray(x, dtype=np.float64)
    box = np.asarray(lp["bounds"], dtype=np.float64)
    if np.any(x < box[:, 0] - 1e-6 * (1.0 + np.abs(box[:, 0]))) or np.any(
        x > box[:, 1] + 1e-6 * (1.0 + np.abs(box[:, 1]))
    ):
        return False
    xf = [Fraction(float(v)) for v in x]
    eps2 = 2.0 * np.finfo(float).eps
    for key_a, key_b, equality in (("A_ub", "b_ub", False), ("A_eq", "b_eq", True)):
        A = lp[key_a].tocsr()
        for i in range(A.shape[0]):
            cols = A.indices[A.indptr[i] : A.indptr[i + 1]]
            vals = A.data[A.indptr[i] : A.indptr[i + 1]]
            bi = Fraction(float(lp[key_b][i]))
            r = sum((Fraction(float(a)) * xf[j] for j, a in zip(cols, vals)), Fraction(0)) - bi
            tol = Fraction(1, 10**6) * (1 + abs(bi))
            if equality:
                grain = min((abs(a) * eps2 * abs(x[j]) for j, a in zip(cols, vals)), default=0.0)
                if abs(r) > tol + Fraction(2.0 * grain):
                    return False
            elif r > tol:
                return False
    return True


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


_OPT_OUT_PROBE = r"""
import sys
sys.path.insert(0, {tests!r})
from discopt import _rust
from test_2125_huge_bound_multirow_band import _seeded_lp
from discopt.solvers.lp_simplex import solve_lp
_rust.profile_reset_py()
statuses = [solve_lp(**_seeded_lp(s, 1e17, "lo")).status.value for s in range(4)]
c = dict(_rust.profile_counters_py())
print(statuses.count("error"), c["HugeBoxRelaxArmed"], c["AuditRowsSingletonAbsorbed"])
"""


@pytest.mark.smoke
@pytest.mark.parametrize("value", ["0", "1"])
def test_opt_out_restores_the_legacy_path(value):
    """``DISCOPT_LP_HUGE_BOUND_RELAX=0`` is the pre-#2125 engine; ``=1`` is the default.

    Read once per process, so each arm runs in its own interpreter. On the ``lo``
    fixture at ``u = 1e17``, seeds 0-3, main returns ERROR on 4/4, and so does the
    ``=0`` arm, with no #2125 route armed. The default arm certifies all four.
    (Measured: ``=0`` is bit-identical to main on 257 LPs, the 256-case sweep plus
    #1287's LP.)
    """
    import os
    import pathlib
    import subprocess
    import sys

    tests = str(pathlib.Path(__file__).parent)
    env = dict(os.environ, DISCOPT_PROFILE="1", DISCOPT_LP_HUGE_BOUND_RELAX=value)
    out = subprocess.run(
        [sys.executable, "-c", _OPT_OUT_PROBE.format(tests=tests)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    errors, armed, absorbed = (int(v) for v in out[-3:])
    if value == "0":
        assert (errors, armed, absorbed) == (4, 0, 0), out
    else:
        assert errors == 0 and armed > 0, out


def test_opt_out_refuses_an_unrecognized_value():
    """A typo must not silently pick an arm."""
    import os
    import subprocess
    import sys

    probe = (
        "from discopt.solvers.lp_simplex import solve_lp\n"
        "solve_lp([1.0], A_ub=[[1.0]], b_ub=[1.0], bounds=[(0.0, 1e17)])\n"
    )
    env = dict(os.environ, DISCOPT_LP_HUGE_BOUND_RELAX="of")
    r = subprocess.run([sys.executable, "-c", probe], env=env, capture_output=True, text=True)
    assert r.returncode != 0 and "DISCOPT_LP_HUGE_BOUND_RELAX" in r.stderr, r.stderr[-500:]
