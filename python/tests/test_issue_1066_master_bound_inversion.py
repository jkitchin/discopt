"""#1066: neither the single-tree driver nor the #1059 merge may publish a dual
bound that has crossed the incumbent it is reported against.

Measured on ``squfl020-150`` (MINLPLib, ``=best= 557.84865``) at the issue's own
defaults (``gap_tolerance=1e-4``, ``time_limit=60``). ``solve_lp_nlp_bb``
returned ``objective=557.848649973387`` with ``bound=558.0440365723572`` -- a
*lower* bound 0.0954 **above** its own incumbent -- and the #1059 merge then
republished it.

Where the 558.044 comes from is settled, and it is not a discopt cut. At the
restart that produced it the master held 69,330 rows; evaluating the instance's
true optimum against the matrix **HiGHS itself stored** (``h.getLp()``) gave
0 violated rows, worst residual 8.35e-15, cost 557.84865. Handing that same
model back to HiGHS from an MPS file, with no discopt in the loop, reproduces
the contradiction in isolation:

=========================================  ==============================
what is fixed at the known optimum          HiGHS returns
=========================================  ==============================
all 6,021 columns                           ``kOptimal`` obj **557.84865**,
                                            ``num_primal_infeasibilities=0``
only the 20 integer columns (same values)   ``kOptimal`` obj **558.04404**
=========================================  ==============================

Fixing *more* variables cannot lower an optimum, so the second answer is
invalid. Ruled out as causes, each by its own arm: retained solver state
(``clearSolver()`` -- byte-identical trace), growing the instance with
``addRow`` after ``run()`` (a fresh ``Highs`` per restart -- byte-identical
558.0440365723572), presolve (same MPS, ``presolve=off`` -- identical
558.0440365723573), and tolerances (``mip_feasibility_tolerance=1e-9`` moved it
the wrong way, to 558.0653).

So the master's dual bound cannot be trusted unconditionally, and discopt's
obligation is the one CLAUDE.md §1 states: never publish a bound known to be
invalid. ``solve_oa`` has refused this since the ``fac2`` incident (#1059) via
``_certified_bound_inverted``; the single-tree path and the merge's *winner*
had no such guard. These tests pin both.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solvers.milp_highs as milp_highs
import discopt.solvers.oa as oa
import numpy as np
import pytest
from discopt.modeling.core import SolveResult
from discopt.solver import _merge_route_and_fallback

SQUFL_OBJ = 557.848649973387
# Two inverted bounds were observed on this instance across runs; both are used
# below because they exercise different halves of the defect.
SQUFL_BAD_BOUND = 558.0440365723572  # inversion 0.1954, naive gap 3.50e-4
SQUFL_ROUTE_BOUND = 557.9460019817818  # inversion 0.0974, naive gap 1.75e-4


def _sr(objective, bound, *, gap_certified=False, status="feasible", node_count=0):
    return SolveResult(
        status=status,
        objective=objective,
        bound=bound,
        gap_certified=gap_certified,
        gap=None,
        node_count=node_count,
        x={},
    )


@pytest.mark.unit
class TestMergeGuardsTheWinnersBound:
    """``_merge_route_and_fallback`` checked only the *loser's* bound."""

    def test_the_squfl020_150_winner_bound_is_suppressed(self):
        route = _sr(SQUFL_OBJ, SQUFL_BAD_BOUND)
        fallback = _sr(1197.972555976441, 382.43)
        merged = _merge_route_and_fallback(route, fallback, False)
        assert merged.objective == pytest.approx(SQUFL_OBJ)
        assert merged.bound is None, "a bound above the incumbent must not be published"
        assert merged.gap is None
        assert merged.gap_certified is False

    def test_the_inversion_is_not_laundered_into_a_small_gap(self):
        """The old ``abs(bnd - obj) / max(|obj|, 1e-10)`` read this pair as 1.75e-4.

        That is *below* the issue's own 1e-4-scale tolerance band -- an
        inversion of 0.0954 presented as "nearly converged", precisely
        backwards. Whatever the merge reports, it must never be a small gap.
        """
        route = _sr(SQUFL_OBJ, SQUFL_ROUTE_BOUND)
        merged = _merge_route_and_fallback(route, _sr(1197.97, None), False)
        naive = abs(SQUFL_ROUTE_BOUND - SQUFL_OBJ) / max(abs(SQUFL_OBJ), 1e-10)
        assert naive < 2e-4, "premise check: the naive formula really is this flattering"
        assert merged.bound is None
        assert merged.gap is None or merged.gap > naive

    def test_maximize_direction_is_inverted(self):
        """For a maximize model the bound is an UPPER bound; below is the crossing."""
        merged = _merge_route_and_fallback(_sr(-70.0, -211.0), _sr(-500.0, None), True)
        assert merged.objective == pytest.approx(-70.0)
        assert merged.bound is None

    def test_a_sound_winner_bound_still_reports_its_gap(self):
        """Anti-vacuity (CLAUDE.md §6): the guard must not suppress everything."""
        merged = _merge_route_and_fallback(_sr(100.0, 90.0), _sr(500.0, 10.0), False)
        assert merged.bound == pytest.approx(90.0)
        assert merged.gap == pytest.approx(0.1, rel=1e-9)

    def test_rounding_at_scale_is_not_treated_as_a_crossing(self):
        merged = _merge_route_and_fallback(
            _sr(331837498.1769339, 331837498.1769349), _sr(4e8, None), False
        )
        assert merged.bound is not None, "1 ulp at |obj|~3e8 is noise, not a broken certificate"


@pytest.mark.unit
def test_lp_nlp_bb_refuses_a_master_bound_above_its_incumbent(monkeypatch):
    """Drive the real driver with a master whose bound has been corrupted.

    The corruption is applied to the *real* master result, so everything else
    about the solve is genuine -- only the number under test is wrong. This is
    the ``squfl020-150`` shape reproduced deterministically: without the guard
    the driver publishes ``bound > objective``.
    """
    real = milp_highs.solve_milp_with_lazy_cuts
    inflated: list[float] = []

    def corrupt(*args, **kwargs):
        res = real(*args, **kwargs)
        if res.bound is not None and np.isfinite(res.bound):
            res.bound = float(res.bound) + 50.0
            inflated.append(res.bound)
        return res

    monkeypatch.setattr(milp_highs, "solve_milp_with_lazy_cuts", corrupt)

    m = dm.Model("convex_minlp")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.binary("y")
    m.subject_to(x >= 2 * y)
    m.subject_to(x + y >= 1.5)
    m.minimize((x - 3) ** 2 + y)
    result = oa.solve_lp_nlp_bb(m, time_limit=30.0, gap_tolerance=1e-4, milp_solver="highs")

    # CLAUDE.md §6: if the seam was never reached the rest asserts nothing.
    assert inflated, "the master was never solved through the patched entry point"

    assert result.objective is not None
    if result.bound is not None:
        assert result.bound <= result.objective + 1e-6, (
            f"published an inverted certificate: bound={result.bound!r} "
            f"objective={result.objective!r}"
        )
    trace = getattr(result, "mip_nlp_trace", None) or {}
    assert trace.get("inverted_master_bound") is not None, (
        "the guard fired but did not record the bound it suppressed"
    )
    assert result.gap_certified is False
