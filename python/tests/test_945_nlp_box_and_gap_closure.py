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
def test_incumbent_options_are_requested_by_point_consumers_only():
    """Pin WHERE ``bound_relax_factor`` is requested, because the two sides differ.

    The split is not "LP versus NLP", it is what the caller consumes. A call site
    whose returned POINT becomes a solution needs it inside the declared box; a
    call site whose MULTIPLIERS are the product must not pin it, because a
    degenerate feasible set has no finite multiplier without Ipopt's relaxation.
    Applied backend-wide it costs the Benders dual LP its convergence (#940) and
    GBD its certification (#946).

    Asserted in both directions, so a future edit cannot quietly move it either
    way — the companion to
    ``test_940...::test_option_requests_are_wired_where_the_guard_checks``.
    """
    import inspect

    from discopt import solver as S
    from discopt.solvers import gdpopt_loa, nlp_pounce, oa, pounce_incumbent_options

    assert pounce_incumbent_options()["bound_relax_factor"] == 0.0
    # A fresh dict each call: a caller mutating it must not poison the next solve.
    pounce_incumbent_options()["bound_relax_factor"] = 999.0
    assert pounce_incumbent_options()["bound_relax_factor"] == 0.0

    for fn in (S._solve_continuous, oa._solve_nlp_attempt, gdpopt_loa._solve_nlp_subproblem):
        assert "pounce_incumbent_options()" in inspect.getsource(fn), (
            f"{fn.__name__} returns a point that becomes a solution, so it must "
            "request the incumbent options"
        )

    # The backend itself must stay neutral: it serves dual consumers too.
    backend_src = inspect.getsource(nlp_pounce.solve_nlp)
    assert "pounce_incumbent_options()" not in backend_src, (
        "bound_relax_factor must NOT be a backend-wide default on the NLP path — "
        "it reaches Benders/GBD recourse, whose product is the multipliers (#946)"
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
    within ``1e-4`` of 3, because the row squares the excursion. It is the
    mechanism behind the MindtPy constraint-qualification fixture certifying
    ``optimal`` at 2.9999000025 against an exact optimum of 3.0, reproduced here
    directly so it stays pinned even if that fixture moves.

    Measured on this model, both arms interleaved:

        pre-#945   status='optimal'          x = 2.9999000024987788  (1.0e-04 off)
        post-#945  status='iteration_limit'  x = 2.9999999998864917  (1.1e-10 off)

    The status is part of the finding rather than incidental. ``(x-3)^2 <= 0`` has
    a single feasible point and therefore empty interior — Slater fails, so this
    is exactly the degeneracy #849's KKT-residual guard exists to refuse to
    certify. Pre-#945 that guard could not see it: against the RELAXED box the
    residuals looked clean, so the solve came back ``optimal``. With the box
    honest the guard fires and withholds the certificate. So the assertion is that
    this is NOT certified, not that it is.
    """
    m = dm.Model("brf_squared_row")
    x = m.continuous("x", lb=1.0, ub=10.0)
    m.subject_to((x - 3.0) ** 2 <= 0.0)
    m.minimize(x)
    res = m.solve()

    # x is pinned to exactly 3 by the row; anything meaningfully below it is the
    # relaxed box, not a better solution. Fails at 1e-4 on the pre-#945 tree.
    assert float(np.asarray(res.x["x"])) >= 3.0 - 1e-6
    assert res.objective >= 3.0 - 1e-6
    # A degenerate feasible set must not come back certified (#849).
    assert res.status != "optimal"


@pytest.mark.smoke
def test_default_solve_path_does_not_certify_a_super_optimal_incumbent():
    """The same fixture through the DEFAULT ``m.solve()`` path, not ``mip-nlp``.

    Seeding OA and GDPopt-LOA left the entry point users actually hit still doing
    it: on the MINLP variant (exact optimum 3.0) ``m.solve()`` returned
    ``objective = 2.9999000090835057`` with ``status='optimal'`` and ``gap = 0.0``
    — 1.0e-04 below an optimum that a single point pins exactly.

    Attributed to two producers, both fixed:

    * ``primal_heuristics.feasibility_pump`` projected onto the RELAXED box and
      handed back ``x = 2.9999`` as a candidate incumbent, and
    * ``_solve_nlp_bb``'s terminal re-solve then adopted its own primal, which
      came from the relaxed box too — so even with the pump honest (``x = 3``
      exactly) the answer was overwritten back to 2.9999.

    No feasibility screen catches this: the bad point's worst row violation is
    1e-8, inside every tolerance discopt has. Only not producing it works.

        pre    obj = 2.9999000090835057   super-optimal by 9.999e-05
        post   obj = 2.9999999983777133   super-optimal by 1.622e-09

    Both assertions below fail on the pre tree; the second is the one that
    matters, since a certificate on a point that cannot exist is the CLAUDE.md §1
    failure.
    """
    m = dm.Model("mindtpy_cq_default_path")
    x = m.continuous("x", lb=1.0, ub=10.0)
    y = m.binary("y")
    m.subject_to((x - 3.0) ** 2 <= 50.0 * (1 - y))
    m.subject_to(x * dm.log(x) + 5.0 <= 50.0 * y)
    m.minimize(x)
    res = m.solve(time_limit=60)

    assert res.objective is not None
    assert res.objective >= 3.0 - 1e-6, (
        f"objective {res.objective!r} is below the exact optimum 3.0; y=0 leaves "
        "x*log(x)+5 <= 0 with no root on [1,10], so y=1 and (x-3)^2 <= 0 pins x=3"
    )
    if res.status == "optimal":
        assert res.bound is not None
        assert res.bound <= res.objective + 1e-5


@pytest.mark.smoke
def test_every_primal_heuristic_requests_the_incumbent_options():
    """The whole module is a point producer, so the seed is a module-wide rule.

    Six NLP option sites in ``_jax/primal_heuristics.py`` each built their own
    ``dict(nlp_options)``; one of them (``feasibility_pump``) supplied the
    super-optimal incumbent above. They are routed through one helper so a
    seventh heuristic cannot silently keep Ipopt's default — asserted here rather
    than left to review, because that is the failure mode #940 already had once.

    ``pounce_option_defaults()`` must NOT arrive with it: its ``constr_viol_tol``
    is separable and costs a 31%-worse incumbent on nvs05 on its own.
    """
    import inspect

    from discopt._jax import primal_heuristics as PH

    src = inspect.getsource(PH)
    assert "pounce_incumbent_options()" in inspect.getsource(PH._heuristic_nlp_options)
    assert "pounce_option_defaults()" not in src, (
        "constr_viol_tol is measurably harmful on this path and is not wanted here"
    )
    assert PH._heuristic_nlp_options()["bound_relax_factor"] == 0.0
    # An explicit caller request still wins over the seed.
    assert PH._heuristic_nlp_options({"bound_relax_factor": 1e-9})["bound_relax_factor"] == 1e-9

    # Every option site routes through the helper; none rebuilds its own dict.
    n_sites = src.count("_heuristic_nlp_options(")
    assert n_sites >= 7, f"expected the helper plus 6 call sites, found {n_sites}"
    # Everything except the helper itself, which is the one place allowed to set
    # the cap. Anchored on the helper's own source rather than on "the source
    # after the second ``def``": that positional slice silently assumed the
    # helper was the module's first function, so adding any function above it
    # (#950's ``_now`` clock seam) moved the cut into the helper's body and the
    # assertion failed on the helper's *own* line. Removing the helper's source
    # says what the rule means and also covers functions defined above it.
    helper_src = inspect.getsource(PH._heuristic_nlp_options)
    assert 'setdefault("max_iter", _HEURISTIC_NLP_MAX_ITER)' in helper_src
    assert 'setdefault("max_iter", _HEURISTIC_NLP_MAX_ITER)' not in src.replace(helper_src, ""), (
        "a heuristic still builds its NLP options inline instead of via the helper"
    )


@pytest.mark.smoke
def test_nlp_bb_terminal_resolve_separates_its_point_from_its_multipliers():
    """One call site, two products, two option sets (#945/#946).

    ``_solve_nlp_bb``'s terminal re-solve both refines the reported primal and
    recovers the duals at it. Those want opposite options, so it takes two solves:
    the refine one requests the incumbent options, the recover one must not (a
    degenerate feasible set has no finite multiplier without Ipopt's relaxation).
    Pinned in both directions so a future edit cannot collapse them back into one.
    """
    import inspect

    from discopt import solver as S

    src = inspect.getsource(S._solve_nlp_bb)
    assert "refine_opts.update(pounce_incumbent_options())" in src
    # The recover solve stays relaxed, and its own point is not adopted.
    assert "recover_opts.update(pounce_incumbent_options())" not in src
    assert "recover_opts.update(" not in src
    assert "sol_flat = np.asarray(nlp_recovered.x" not in src
    assert "obj_val = float(nlp_recovered.objective)" not in src


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
