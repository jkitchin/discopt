"""Regression: AMP recovers a feasible incumbent when the MILP relaxation
backend produces no usable result.

When the LP/MILP backend cannot solve AMP's relaxation (e.g. HiGHS is not
installed and the POUNCE matrix-MILP times out / errors on a degenerate
equality-encoded relaxation), the relaxation solve returns neither a dual
bound nor a solution point. AMP's main NLP step then has no binary assignment
to fix and cannot find an integer-feasible incumbent, so AMP previously
returned no solution at all (``objective=None``).

The recovery path enumerates the small integer domain (each assignment solved
as a binaries-fixed NLP) once, so a feasible incumbent is still recovered.
The bound stays uncertified.
"""

from __future__ import annotations

import discopt._jax.milp_relaxation as MR
import discopt.modeling as dm
import pytest
from discopt._jax.milp_relaxation import MilpRelaxationResult


@pytest.fixture
def _milp_backend_yields_nothing(monkeypatch):
    """Force every AMP MILP relaxation solve to return an unusable result:
    no bound (``-inf``) and no solution point, as a failing/absent MILP
    backend does."""

    def _unusable(self, *args, **kwargs):
        return MilpRelaxationResult(
            status="time_limit", objective=None, bound=float("-inf"), x=None
        )

    monkeypatch.setattr(MR.MilpRelaxationModel, "solve", _unusable)


def test_amp_recovers_incumbent_when_milp_backend_fails(_milp_backend_yields_nothing):
    # Small nonconvex MINLP: one binary, one continuous, with a solvable
    # binaries-fixed NLP. Global optimum is y=1, x=4 -> -(4**2)+5 = -11.
    m = dm.Model("amp_milp_fail_recovery")
    y = m.binary("y")
    x = m.continuous("x", lb=0.0, ub=4.0)
    m.minimize(-(x * x) + 5 * y)
    m.subject_to(x <= 2 + 2 * y)

    r = m.solve(solver="amp", nlp_solver="pounce", time_limit=15, gap_tolerance=1e-3)

    # Without the recovery path this returned objective=None (no incumbent).
    assert r.objective is not None, "AMP should recover a feasible incumbent"
    assert r.status == "feasible"
    # Recovery does not certify a bound.
    assert r.gap_certified is False
    # The recovered incumbent should be the (integer-feasible) global optimum.
    assert abs(r.objective - (-11.0)) <= 1e-3
