"""#2000: the native spatial kernel certified against an un-repaired incumbent.

Three coupled defects, measured on cstr nt=5 (a GDP benchmark; both instances are
in ``data/gdp_cstr``):

1. The kernel's incumbent pricer (#1656) evaluated the objective AT a
   McCormick-tight LP point whose nonlinear equalities carry 2.6e-6..1.1e-5 of
   residual. The tree fathomed against that value and exited ``optimal``; the
   published-incumbent repair (#1537 E) then moved the point to float noise, the
   objective rose (3.00885387 -> 3.00887207) and the certificate was withdrawn,
   with 99.5% of the budget unspent. The pricer now repairs first and prices the
   repaired point, and the kernel publishes that point. A certified exit whose
   point fails verification at the allowance edge is repaired before it is
   discarded.
2. An explicit ``abs_gap_tolerance`` below the default (GAMS ``optca=0`` reads as
   1e-9) clamped the kernel's absolute ``gap_tol`` to it, which switched the
   relative criterion off: honest pricing then branched to the time limit at a
   relative gap of 6.6e-8 against ``optcr=1e-6``.
3. A decertified result kept ``termination="gap"``, which GAMS reported as a
   normal completion.
"""

from __future__ import annotations

import os

import discopt.modeling as dm
import discopt.solver as SV
import numpy as np
import pytest
from discopt.gams import link
from discopt.modeling import SolveResult
from discopt.modeling.gams_parser import parse_gams_file
from discopt.validation.feasibility import verify_point

DATA = os.path.join(os.path.dirname(__file__), "data", "gdp_cstr")


def _hyperbola():
    m = dm.Model("hyperbola")
    x = m.continuous("x", lb=0.5, ub=4.0)
    y = m.continuous("y", lb=0.5, ub=4.0)
    m.subject_to(x * y == 2.0)
    m.minimize(x + y)
    return m


@pytest.mark.unit
def test_pricer_prices_and_keeps_the_repaired_point():
    m = _hyperbola()
    x0 = np.array([1.4, 2.0 / 1.4 - 4e-7])  # inside verify_point's allowance
    assert verify_point(m, x0).ok
    price = SV._native_kernel_incumbent_value(m, None, 1.0, 0.0, 2)
    v = price(x0, float(x0.sum()))
    assert isinstance(v, float)
    rep = price.repaired[x0.tobytes()]
    assert abs(rep[0] * rep[1] - 2.0) <= 4 * np.finfo(float).eps * 2.0
    assert v == pytest.approx(float(rep.sum()), abs=0, rel=1e-15)
    assert v > float(x0.sum())  # the residual had bought objective


@pytest.mark.unit
def test_noise_repair_keeps_a_certificate_only_when_the_gap_still_closes():
    m = _hyperbola()
    x0 = np.array([np.sqrt(2.0), np.sqrt(2.0) - 4e-7])
    opt = 2.0 * np.sqrt(2.0)
    out = SV._native_kernel_noise_repair(m, x0, None, opt - 1e-9, 1e-6, None)
    assert out is not None
    x_rep, obj = out
    assert verify_point(m, x_rep).ok and obj == pytest.approx(opt, abs=1e-6)
    # a bound the repaired objective does not close against: no certificate
    assert SV._native_kernel_noise_repair(m, x0, None, opt - 1e-3, 1e-6, None) is None


@pytest.mark.unit
def test_gams_reports_a_decertified_gap_stop_as_terminated():
    r = SolveResult(
        status="feasible",
        termination="gap",
        objective=3.00887207,
        bound=3.00885387,
        bound_valid=True,
        gap_certified=False,
        x={"x": 1.0},
    )
    assert link.status_to_gams(r, True) == (link.MODELSTAT_INTEGER, link.SOLVESTAT_TERMINATED)


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.parametrize(
    "name, objective",
    [("cstr_nt5_hull_0008.gms", 3.008872066), ("cstr_nt5_bigm_0006.gms", 2.791883794)],
)
def test_cstr_certifies_on_the_repaired_incumbent(name, objective):
    m = parse_gams_file(os.path.join(DATA, name))
    r = m.solve(time_limit=120, gap_tolerance=1e-6, abs_gap_tolerance=1e-9, threads=1)
    stats = r.solver_stats or {}
    assert "certificate/repair_decertified" not in stats
    assert r.status == "optimal" and r.gap_certified, (r.status, r.objective, r.bound)
    assert r.termination == "gap"
    assert r.objective == pytest.approx(objective, abs=2e-9)
    assert r.bound <= r.objective
    assert (r.objective - r.bound) <= 1e-6 * abs(r.objective)
    x = np.concatenate([np.ravel(np.asarray(r.x[v.name], float)) for v in m._variables])
    assert verify_point(m, x).ok
