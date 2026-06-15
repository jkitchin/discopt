"""Soundness regression lock for OBBT bound tightening (issue #145).

OBBT tightens a variable's bound to the optimum of ``min``/``max x_i`` over the
McCormick LP relaxation polytope. That is sound **only when the subproblem LP is
solved to its true optimum**: the returned objective becomes a hard variable
bound, so an inexact optimum cuts off feasible points.

The POUNCE interior-point backend returns the analytic center of the optimal
face; its reported objective normally matches the simplex optimum, but on an
ill-conditioned LP (here a ``1e6`` linking coefficient) it can be grossly wrong
while still reporting ``OPTIMAL``. Before the fix, the *default* solve
(``nlp_solver="ipm"`` silently routes to POUNCE) ran OBBT through that IPM and
over-tightened ``x0`` to ``[17, 17]`` — excluding the optimal ``x0 = 16`` — and
then certified a false optimum of ~2538 on a problem whose true optimum is ~1.64.

The fix routes OBBT through an *exact* LP oracle (HiGHS simplex) regardless of
``nlp_solver``; when no exact oracle is available it skips tightening rather than
trust the IPM. This test pins the soundness invariant on the minimal repro: a
large-coefficient trilinear equality with integer branching. The dual bound must
never exceed the true optimum, and the solver must never certify a false one.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import discopt.modeling as dm
import pytest

# True optimum of the box below: at (x0,x1,x2,x3)=(16,19,43,49), y=323/2107 and
# the objective x4+x5 ~= 1.642. The global minimum over the box is <= that, so a
# valid lower (dual) bound can never exceed it. The pre-fix bug reported ~2538.
_TRUE_OPT = 1.642
_FALSE_OPT_FLOOR = 5.0  # cleanly separates true (~1.64) from the false (~2538)


def _build_stiff_ratio_model():
    """Lifted gear4-style slice: trilinear equality + 1e6 linking coefficient.

    ``y * x2 * x3 == x0 * x1`` lifts the bilinear/bilinear ratio, and the linking
    row ``1e6*y + x4 - x5 == 144279.32...`` is what makes the relaxation LPs
    ill-conditioned. Integer branching on x0..x3 closes the combinatorial gap.
    """
    m = dm.Model("obbt_soundness")
    x0 = m.integer("x0", lb=16, ub=18)
    x1 = m.integer("x1", lb=18, ub=20)
    x2 = m.integer("x2", lb=42, ub=44)
    x3 = m.integer("x3", lb=48, ub=50)
    x4 = m.continuous("x4", lb=0.0, ub=2e5)
    x5 = m.continuous("x5", lb=0.0, ub=2e5)
    y = m.continuous("y", lb=0.04, ub=25.0)
    m.minimize(x4 + x5)
    m.subject_to(y * x2 * x3 - x0 * x1 == 0)
    m.subject_to(1e6 * y + x4 - x5 == 144279.32477276)
    return m


@pytest.mark.correctness
@pytest.mark.parametrize("obbt_at_root", [True, False])
@pytest.mark.parametrize("presolve", [True, False])
def test_stiff_ratio_obbt_is_sound(presolve, obbt_at_root):
    """OBBT must never over-tighten and certify a false optimum (#145).

    Across every presolve/OBBT toggle the dual bound must stay <= the true
    optimum, and the solver must never report ``gap_certified`` at a false
    incumbent. An honest non-certified result (``bound=None``) is acceptable —
    only an *unsound* certification is a failure.
    """
    from discopt.solver import solve_model

    r = solve_model(
        _build_stiff_ratio_model(),
        time_limit=20,
        gap_tolerance=1e-4,
        presolve=presolve,
        obbt_at_root=obbt_at_root,
    )

    # Core soundness invariant: a dual bound is a valid lower bound on the global
    # optimum, so it can never exceed the true optimum (~1.642).
    if r.bound is not None:
        assert r.bound <= _TRUE_OPT + 1e-2, (
            f"unsound dual bound {r.bound} > true optimum {_TRUE_OPT} "
            f"(presolve={presolve}, obbt_at_root={obbt_at_root})"
        )

    # A feasible incumbent must reflect the true optimum, not a box that OBBT
    # wrongly carved away.
    if r.objective is not None:
        assert r.objective < _FALSE_OPT_FLOOR, (
            f"incumbent {r.objective} is the pre-fix false optimum "
            f"(presolve={presolve}, obbt_at_root={obbt_at_root})"
        )

    # Never certify a false optimum.
    if r.gap_certified:
        assert r.objective is not None and r.objective <= _TRUE_OPT + 1e-2, (
            f"certified false optimum obj={r.objective} "
            f"(presolve={presolve}, obbt_at_root={obbt_at_root})"
        )


@pytest.mark.correctness
def test_root_obbt_keeps_optimum_even_when_pounce_requested():
    """``obbt_tighten_root`` must not exclude a feasible point (#145, unit-level).

    Fast guard on the OBBT layer itself: passing ``prefer_pounce=True`` must NOT
    route the bound-tightening LPs through the IPM (whose inexact optimum carved
    ``x0`` to ``[17,17]`` before the fix). The optimal integer assignment
    ``(16,19,43,49)`` must survive every OBBT sweep.
    """
    import time

    import numpy as np
    from discopt._jax.obbt import obbt_tighten_root

    m = _build_stiff_ratio_model()
    lb = np.array([v.lb for v in m._variables], dtype=float)
    ub = np.array([v.ub for v in m._variables], dtype=float)
    opt = {
        "x0": 16.0,
        "x1": 19.0,
        "x2": 43.0,
        "x3": 49.0,
        "x4": 0.0,
        "x5": 1.6434,
        "y": 323.0 / 2107.0,
    }

    res = obbt_tighten_root(
        m, lb, ub, rounds=3, deadline=time.perf_counter() + 15, prefer_pounce=True
    )
    assert not res.infeasible
    tlb = np.maximum(lb, res.lb)
    tub = np.minimum(ub, res.ub)
    for i, v in enumerate(m._variables):
        if v.name in opt:
            val = opt[v.name]
            assert tlb[i] - 1e-6 <= val <= tub[i] + 1e-6, (
                f"OBBT excluded feasible {v.name}={val}: tightened to [{tlb[i]}, {tub[i]}]"
            )
