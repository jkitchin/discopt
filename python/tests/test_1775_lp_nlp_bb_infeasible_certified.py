"""A proven ``infeasible`` from LP/NLP-BB carries its certificate (#1775).

``solve_lp_nlp_bb`` reports ``infeasible`` only when its master -- a relaxation
of the feasible set, every row certified convex, no heuristic no-good -- is
proven infeasible with no incumbent (#1673 B4). It returned that proof with
``gap_certified=False``, which the SolveResult contract reads as a heuristic
verdict. A caller that honours the contract therefore could not use it: a GDP
subproblem routed to the convex-MINLP route (``mip-nlp/lp_nlp_bb``) stayed
unresolved, and LOA could not certify. Measured on the ``gdp_parity`` panel:
``constrained_layout_clay0203``'s two infeasible configurations came back this
way, so LOA-global lost the instance once uncertified verdicts stopped
excluding configurations.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest
from discopt.gdp.algorithms.loa import LOA
from discopt.solvers.oa import solve_lp_nlp_bb

pytestmark = pytest.mark.smoke


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return fn(*a, **k)


def _disk():
    """A convex MIQCP no presolve refutes: the disk's largest x + y is
    sqrt(1.01) < 1.45 for either value of z, but interval reasoning sees 2."""
    m = dm.Model("disk")
    x = m.continuous("x", lb=0, ub=3)
    y = m.continuous("y", lb=0, ub=3)
    z = m.binary("z")
    m.subject_to(x**2 + y**2 <= 1 + 0.01 * z)
    m.subject_to(x + y >= 1.5 - 0.05 * z)
    m.minimize(x + 2 * y + z)
    return m


def test_lp_nlp_bb_certifies_its_proof():
    r = _quiet(solve_lp_nlp_bb, _disk(), time_limit=30)
    assert r.status == "infeasible" and r.gap_certified


def test_the_default_route_carries_the_certificate():
    r = _quiet(_disk().solve, time_limit=30)
    assert r.algorithm_route.startswith("mip-nlp/lp_nlp_bb")
    assert r.status == "infeasible" and r.gap_certified


def test_equality_relaxation_is_not_a_certificate():
    r = _quiet(solve_lp_nlp_bb, _disk(), time_limit=30, equality_relaxation=True)
    assert not r.gap_certified


def test_a_feasible_model_is_never_certified_infeasible():
    """The #1673 B4 model (a false convexity certificate once emptied its
    master): feasible, and never a certified ``infeasible``."""
    m = dm.Model("b4")
    x0 = m.continuous("x0", lb=0.1, ub=10)
    x2 = m.continuous("x2", lb=-10, ub=10)
    b = m.binary("b")
    m.subject_to(x0**-3.5 - x2**2 <= 0)
    m.subject_to(x0 + b <= 10)
    m.minimize((x0 + 4) ** 2 + (x2 - 2) ** 2)
    r = _quiet(solve_lp_nlp_bb, m, time_limit=30)
    assert r.status != "infeasible" and r.objective is not None


def test_loa_excludes_a_configuration_on_the_proof():
    """The GDP whose first disjunct's reduced subproblem is the disk: LOA-global
    ended ``unknown`` with that configuration unresolved; with the certificate
    it is excluded and the GDP is certified."""
    m = dm.Model("gdp_disk")
    x = m.continuous("x", lb=0, ub=3)
    y = m.continuous("y", lb=0, ub=3)
    w = m.binary("w")
    m.either_or([[x**2 + y**2 <= 1 + 0.01 * w, x + y >= 1.5 - 0.05 * w], [x + y <= 0.5]], name="d")
    m.minimize(-(x + y) + w)
    r = _quiet(LOA(m, time_limit=60).run)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(-0.5, abs=1e-6)
    assert r.solver_stats["n_unresolved"] == 0
