"""Issue #945: the NLP path must respect its declared box, and gap closure must
not be calibrated on incumbents that do not.

Two coupled properties, split out of #940/#943:

**(a)** ``nlp_pounce.solve_nlp`` and ``solver.py``'s POUNCE batch path did not
seed ``bound_relax_factor``, so they returned points up to ``1e-8*(1 + |bound|)``
outside their declared variable bounds — Ipopt's default deliberately relaxes
every bound, including the slack bounds standing in for inequality rows. #943
fixed the matrix-form ``lp_pounce``/``qp_pounce`` backends; this closes the last
entry point.

**(b)** With the incumbent honest, ``incumbent - bound`` stops being ``<= 0``, and
the ``1e-9`` absolute floor in OA's and GDPopt-LOA's gap tests turns out to be
beatable only by an incumbent that is *not* feasible. The floor is now discopt's
own ``1e-6`` absolute tolerance, and a materially inverted bound is reported as
"nothing proved" instead of being clamped to ``gap = 0``.

These tests pin BEHAVIOUR — the returned point, the certificate — not option
names, so a re-spelling of the option that is a no-op on the installed build
still fails them (the #940 lesson).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solvers._gap import GAP_ABS_TOL, optimality_gap

# ── (a) every POUNCE entry point stays inside the declared box ───────────────


@pytest.mark.smoke
def test_nlp_backend_seeds_the_shared_pounce_baseline():
    """``solve_nlp`` must inherit the shared baseline, not re-spell it.

    A second copy of these values is how one entry point silently keeps Ipopt's
    defaults while the rest move — which is exactly the state #945 fixed.
    """
    import inspect

    from discopt.solvers import nlp_pounce

    src = inspect.getsource(nlp_pounce.solve_nlp)
    assert "pounce_option_defaults()" in src, (
        "nlp_pounce.solve_nlp no longer seeds from solvers.pounce_option_defaults"
    )


@pytest.mark.smoke
def test_nlp_path_point_stays_inside_its_declared_box():
    """``dm.sum(y)`` over a bare indexed container routes to the general NLP path.

    Before #945 this returned a point ~7.5e-9 BELOW ``lb=1`` and an objective
    below the true optimum of 3.0 — super-optimal, bought by leaving the box.
    """
    m = dm.Model()
    s = m.set("S", [10, 20, 30])
    y = m.continuous("y", lb=1.0, ub=5.0, over=s)
    m.minimize(dm.sum(y))
    res = m.solve()

    assert res.status == "optimal"
    x = np.asarray(res.value(y), dtype=np.float64).ravel()
    assert x.size == 3
    below = float(np.max(1.0 - x))
    assert below <= 1e-12, f"returned point sits {below:.3e} below its declared lb=1"
    assert res.objective >= 3.0 - 1e-12, "objective is below the true optimum of 3.0"


@pytest.mark.smoke
def test_bound_relaxation_damage_is_not_bounded_by_the_relaxation_itself():
    """A squared row turns a 1e-8 bound relaxation into a 1e-4 error in ``x``.

    This is why ``bound_relax_factor`` cannot be dismissed as "1e-8, therefore
    negligible": relaxing ``(x-3)^2 <= 0`` to ``<= 1e-8`` admits every ``x``
    within ``1e-4`` of 3. On the MindtPy constraint-qualification fixture that
    produced a certified ``optimal`` at 2.9999000025 against an exact optimum of
    3.0. The row is reproduced here directly so the mechanism is pinned even if
    the fixture moves.
    """
    m = dm.Model("brf_squared_row")
    x = m.continuous("x", lb=1.0, ub=10.0)
    m.subject_to((x - 3.0) ** 2 <= 0.0)
    m.minimize(x)
    res = m.solve()

    assert res.status in ("optimal", "feasible")
    # x is pinned to exactly 3 by the row; anything meaningfully below it is the
    # relaxed box, not a better solution.
    assert float(np.asarray(res.x["x"])) >= 3.0 - 1e-6
    assert res.objective >= 3.0 - 1e-6


# ── (b) gap closure is an honest dual test ───────────────────────────────────


def test_absolute_criterion_matches_discopts_absolute_tolerance():
    """The absolute floor is 1e-6, not 1e-9.

    ``1e-9`` is tighter than discopt's own absolute tolerance and tighter than the
    IPM can deliver, so on a near-zero optimum it was only ever beaten by an
    incumbent outside its box.
    """
    assert GAP_ABS_TOL == 1e-6
    # The honest GDPopt incumbent that used to be reported as a 100% gap.
    assert optimality_gap(0.0, 2.45888227638e-09, denom_floor=1e-10) == 0.0
    # Just above the criterion it is a real gap again, not silently absorbed.
    assert optimality_gap(0.0, 1e-5, denom_floor=1.0) == pytest.approx(1e-5)


def test_inverted_bound_is_not_a_closed_gap():
    """``max(0.0, ub - lb)`` must not turn a broken certificate into ``gap = 0``.

    The numbers are the measured pre-#945 pair from the MindtPy CQ fixture: a
    dual bound of 2.99995 reported alongside an incumbent of 2.99990, i.e. the
    bound sitting 5e-5 ABOVE the incumbent it was supposed to bound.
    """
    assert optimality_gap(2.99995000083, 2.9999000025) == 1.0
    # ... while a rounding-scale inversion is still absorbed, as before.
    assert optimality_gap(4.0, 3.9999999999) == 0.0
    # Scale-aware: at |obj| ~ 1e6 a 1e-9 inversion is a single ulp.
    assert optimality_gap(1e6 + 1e-9, 1e6) == 0.0


def test_missing_bounds_report_nothing_proved():
    """The ±1e19 sentinels mean "no bound yet", never a closed gap."""
    assert optimality_gap(-1e20, 5.0) == 1.0
    assert optimality_gap(0.0, 1e20) == 1.0


def test_oa_and_loa_share_one_gap_definition():
    """Two near-copies that disagreed are now one definition (#945)."""
    from discopt.solvers.gdpopt_loa import _compute_gap as loa_gap
    from discopt.solvers.oa import _compute_gap as oa_gap

    for lb, ub in [(2.0, 4.0), (3.0, 3.0), (-1e20, 5.0), (0.0, 1e20)]:
        assert oa_gap(lb, ub) == loa_gap(lb, ub), (lb, ub)
    # Both close an honest near-zero gap that only one of them used to close.
    assert oa_gap(0.0, 2.4589e-9) == 0.0
    assert loa_gap(0.0, 2.4589e-9) == 0.0


@pytest.mark.smoke
def test_loa_certifies_a_near_zero_optimum_with_an_honest_incumbent():
    """``min x`` over ``(x<=3) or (x>=7)``, ``x in [0,10]``: optimum exactly 0.

    Pre-#945 this was certified ``optimal`` at ``-7.5e-09`` — below its own
    ``lb=0`` and below its own dual bound. With the incumbent inside the box the
    objective is ``+2.5e-09``, and LOA must still certify it: the absolute
    criterion, not the degenerate relative one, is what closes this gap.
    """
    m = dm.Model("loa_near_zero")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 3], [x >= 7]], name="choice")
    m.minimize(x)
    r = m.solve(time_limit=30, gdp_method="loa")

    assert r.status == "optimal"
    assert r.objective == pytest.approx(0.0, abs=1e-5)
    # The incumbent is inside its box: not below lb=0, and not below the optimum.
    assert float(np.asarray(r.x["x"])) >= 0.0
    assert r.objective >= 0.0
    # Certificate invariant: the dual bound never sits above the incumbent.
    assert r.bound is not None and r.bound <= r.objective + 1e-9
    assert r.gap == pytest.approx(0.0, abs=1e-9)
