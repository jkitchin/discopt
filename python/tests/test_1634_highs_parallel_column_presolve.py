"""#1634: the HiGHS MILP route certified false optima on small badly scaled MILPs.

Five integer columns in ``[0, 2..4]``, two ``<=`` rows with coefficients up to 1e6:
the route certified ``optimal`` at a bound ABOVE the enumerated optimum, which was
strictly feasible (min slack 14 to 6e5). Every existing guard passed (#1295 ratio
2.04e-6 above the 2**-20 cap, #1410 dual violation 7e-17, #1509 refutation margin 0).

Root cause (seed 181, replayed on the route's standard form): HiGHS's MIP *presolve*
reduced the model to empty and returned the false value as "Presolve: Optimal" -- 0
nodes, 0 LP iterations. Bisecting ``presolve_reduction_limit`` puts the bad step at
reduction 5, and switching off one rule -- ``presolve_rule_off`` bit 13, "Parallel rows
and columns" -- fixes all three witnesses. In ``detectParallelRowsAndCols`` the
route's own slack ``s`` (cost 0, coefficient 1) and the integer ``x2`` (cost 0.0064,
coefficient -4.9e5) are parallel once one row remains; their cost difference per unit
of ``s`` is ``0.0064 / -4.9e5 = -1.3e-8``, under the ABSOLUTE
``dual_feasibility_tolerance = 1e-7``, so HiGHS treats them as cost-tied and fixes
``x2`` at its upper bound 3 -- the direction that costs 3 * 0.0064 = 0.0192, exactly the
gap between the certified 0.02082 and the true 0.00801.

The fix switches that rule off on the MILP route
(:data:`lp_milp_highs.MILP_PRESOLVE_RULE_OFF`).
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solver import _highs_std_form
from discopt.solvers import lp_milp_highs
from discopt.solvers.lp_milp_highs import MILP_PRESOLVE_RULE_OFF


@pytest.fixture(autouse=True)
def _highs_milp_route(monkeypatch):
    """These tests pin the HiGHS MILP route's own mechanics. Since 2026-11-09 the
    pure-MILP default is the Rust engine on every install (PR #2196), so they name
    HiGHS explicitly, the optional ``discopt[highs]`` extra."""
    pytest.importorskip("highspy")
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "highs")


highspy = pytest.importorskip("highspy")

# (c, A, b, ub, enumerated optimum) -- verbatim from the issue.
WITNESSES = {
    "seed181_cost1e-3": (
        [0.016422113453256008, 1.5785686489149076, 0.006402947197508945,
         0.23211550436289133, 0.0016118362136789844],
        [[-109159.8468922766, 60334.201191650696, -490025.3894761287, 10017.078735174448,
          -0.0027928104506268724],
         [-0.007588297250901298, 0.016053747317747457, -0.962476149861379, 0.279609118685822,
          -148.0240160211314]],
        [-57818.64349215431, -76.71966452895884],
        [3, 3, 3, 4, 3],
        0.008014783411187928,
    ),
    "seed151_cost1e-3": (
        [0.06707721406672994, -0.0020261531072412343, 0.0022560042559867507,
         -1.1169781841142587e-06, 2.1432802325608193],
        [[6.570629227753773, -0.04805795211072338, -719177.8353211597, 1629.260033941908,
          -136621.90600097223],
         [-0.0015374870886662839, -938651.444161964, -0.6740692727285625, 0.07740310835170049,
          -0.001432997557144032]],
        [-95795.1305167389, -860157.4572911527],
        [3, 4, 2, 4, 2],
        -0.005853076085714643,
    ),
    "seed265_cost1": (
        [132.46554019288587, -460.75131459165647, 0.018375734518718913,
         -0.001904722311427082, 1.2227744304875126],
        [[0.03707558493291472, 21.3188360162604, -0.013770211812094183, -4.081557980408068,
          -2.3385289660920208],
         [-223.8992695192557, 937354.6838057677, -673764.4860213215, -598.848579228129,
          0.04411308944572607]],
        [65.85440029434662, 1868337.8189215076],
        [3, 3, 3, 3, 3],
        -1382.2229064728663,
    ),
}  # fmt: skip


def _model(c, A, b, ub) -> dm.Model:  # noqa: N803
    m = dm.Model("i1634")
    xs = [m.integer(f"x{j}", lb=0, ub=int(ub[j])) for j in range(len(c))]
    m.minimize(sum(float(c[j]) * xs[j] for j in range(len(c))))
    for i in range(len(b)):
        m.subject_to(sum(float(A[i][j]) * xs[j] for j in range(len(c))) <= float(b[i]))
    return m


def _enumerate(c, A, b, ub) -> float | None:  # noqa: N803
    P = np.array(list(itertools.product(*[range(int(u) + 1) for u in ub])), float)  # noqa: N806
    feas = np.all(P @ np.asarray(A, float).T <= np.asarray(b, float), axis=1)
    return float((P[feas] @ np.asarray(c, float)).min()) if feas.any() else None


def _generate(seed: int, objscale: float, n: int = 5, m: int = 2):
    """The issue's generator, verbatim."""
    rng = np.random.default_rng(seed)

    def sgn(s):
        return rng.choice([-1.0, 1.0], size=s)

    c = sgn(n) * 10 ** rng.uniform(-3, 4, n) * objscale
    A = sgn((m, n)) * 10 ** rng.uniform(-3, 6, (m, n))  # noqa: N806
    ub = rng.integers(2, 5, n)
    x0 = np.array([rng.integers(0, u + 1) for u in ub], float)
    b = A @ x0 + np.abs(A) @ (ub * rng.uniform(0, 0.3))
    return c, A, b, ub


def _false_certificate(r, truth: float | None) -> bool:
    if not r.gap_certified:
        return False
    if truth is None:
        return r.status != "infeasible"
    if r.status == "infeasible":
        return True
    return r.bound is not None and r.bound > truth + 1e-9 * (1.0 + abs(truth))


def test_rule_bit_is_parallel_rows_and_cols():
    # HConst.h PresolveRuleType: ... kPresolveRuleAggregator = 12,
    # kPresolveRuleParallelRowsAndCols = 13. The route must switch exactly that off.
    assert MILP_PRESOLVE_RULE_OFF & (1 << 13)


@pytest.mark.parametrize("name", sorted(WITNESSES))
def test_witness_certifies_true_optimum(name):
    c, A, b, ub, truth = WITNESSES[name]  # noqa: N806
    assert _enumerate(c, A, b, ub) == pytest.approx(truth, rel=1e-12, abs=1e-15)
    r = _model(c, A, b, ub).solve()
    assert not _false_certificate(r, truth), (r.status, r.bound, truth)
    # The fix keeps the certificate: HiGHS without the rule finds and proves the optimum.
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(truth, rel=1e-9, abs=1e-12)
    assert r.bound <= truth + 1e-9 * (1.0 + abs(truth))


@pytest.mark.parametrize("objscale", [1.0, 1e-3])
def test_generator_panel_has_no_false_certificate(objscale):
    """The issue's generator over 300 seeds per cost scale, truth by enumeration.
    Seeds 151, 181 (1e-3) and 265 (1) are in range; before the fix this panel had
    false certificates at both scales."""
    checked = 0
    false = []
    for seed in range(0, 300):
        c, A, b, ub = _generate(seed, objscale)  # noqa: N806
        truth = _enumerate(c, A, b, ub)
        r = _model(c, A, b, ub).solve()
        checked += 1
        if _false_certificate(r, truth):
            false.append((seed, r.status, r.bound, truth))
    assert checked == 300
    assert false == []


# ── The presolve cross-check: rule 13 off is necessary, not sufficient ──


def test_six_column_seed_965_is_not_certified_false():
    """Generator panel (n=6, m=3, cost scale 1e-3) seed 965: with rule 13 off, an
    always-on presolve reduction still lifts the bound 3e-6 above the enumerated
    optimum, and the presolve-free solve's kOptimal incumbent fails the route's row
    re-verification. The cross-check therefore has no verdict and must withdraw the
    certificate rather than let it stand on the absence of evidence."""
    c, A, b, ub = _generate(965, 1e-3, n=6, m=3)  # noqa: N806
    truth = _enumerate(c, A, b, ub)
    assert truth == pytest.approx(-13.599131078378665, rel=1e-12)
    r = _model(c, A, b, ub).solve()
    assert not _false_certificate(r, truth), (r.status, r.bound, truth)
    assert r.bound is None or r.bound <= truth + 1e-9 * (1.0 + abs(truth))
    assert r.solver_stats.get("milp/presolve_cross_check_ran") == 1.0


@pytest.mark.parametrize("name", sorted(WITNESSES))
def test_cross_check_alone_refutes_the_witness(name, monkeypatch):
    """With the rule-13 switch removed, the presolve-free cross-solve on its own must
    catch the witness: its verified point sits far below the false bound."""
    # #1733: the route now folds its slack columns into row bounds, where HiGHS gets
    # this model right; the falsifier under test only fires on the slack form.
    monkeypatch.setenv("DISCOPT_MILP_ROW_FORM", "0")
    monkeypatch.setattr(lp_milp_highs, "MILP_PRESOLVE_RULE_OFF", 0)
    c, A, b, ub, truth = WITNESSES[name]  # noqa: N806
    # #2221: highspy 1.16 no longer prunes these witnesses with rule 13 on (see
    # :data:`FALSE_PRIMARY_115`), so the primary solve is now right
    # and the cross-check had nothing to refute. The fault is injected at the seam: the
    # certified primary handed to the judge is replaced by what highspy 1.15.1 returned.
    injected = _inject_false_primary(monkeypatch, name)
    r = _model(c, A, b, ub).solve()
    assert injected, "the presolve cross-check never judged a primary: no fault injected"
    assert injected[0] > truth + 1e-9 * (1.0 + abs(truth)), "the injected claim is not false"
    assert not _false_certificate(r, truth), (r.status, r.bound, truth)
    assert r.solver_stats.get("milp/presolve_certificate_refuted") == 1.0
    assert r.objective == pytest.approx(truth, rel=1e-9, abs=1e-12)


#: The structural point the presolve-on primary certified ``optimal`` (with
#: ``bound == objective``) on each witness, rule 13 on, ``DISCOPT_MILP_ROW_FORM=0``.
#: Measured on highspy 1.15.1 (#2221); 1.16.0 certifies the enumerated optimum instead.
#:
#: The mechanism replay that used to stand beside this (the route's own standard form in
#: HiGHS with the route's options: ``mip_dual_bound`` above the enumerated optimum with
#: rule 13 on, at or below it with :data:`MILP_PRESOLVE_RULE_OFF`) reproduced on
#: highspy 1.11.0-1.15.1 and skipped on 1.16.0, where all three witnesses come out right
#: with the rule on. With the floor at ``highspy>=1.16`` (#2332) it could only skip, so
#: it was deleted (PR #2334 review); this injection keeps the refutation path covered.
FALSE_PRIMARY_115 = {
    "seed151_cost1e-3": [0.0, 4.0, 2.0, 0.0, 0.0],  # -0.0035926 vs -0.0058531
    "seed181_cost1e-3": [0.0, 0.0, 3.0, 0.0, 1.0],  # 0.0208207 vs 0.0080148
    "seed265_cost1": [0.0, 3.0, 3.0, 3.0, 0.0],  # -1382.2045 vs -1382.2229
}


def _inject_false_primary(monkeypatch, name):
    """Hand ``_judge_presolve_cross`` the false certificate highspy 1.15.1 produced:
    a feasible, suboptimal point (logicals completed from it) certified ``optimal``
    at its own objective. The cross-solve itself still runs for real. Returns the
    injected claims."""
    real = lp_milp_highs._judge_presolve_cross
    claims: list[float] = []

    def judge(sf, out, cross, kw):
        xs = np.asarray(FALSE_PRIMARY_115[name], dtype=np.float64)
        x = lp_milp_highs.complete_seed(sf, xs, xs.size)
        assert x is not None, "the 1.15.1 point has no logical completion"
        assert lp_milp_highs._verified_mip_point(sf, x) is not None, "1.15.1 point infeasible"
        obj = float(sf.c @ x) + sf.obj_const
        out.status, out.x, out.objective, out.bound = "optimal", x, obj, obj
        out.gap_certified = True
        claims.append(obj)
        return real(sf, out, cross, kw)

    monkeypatch.setattr(lp_milp_highs, "_judge_presolve_cross", judge)
    return claims


@pytest.mark.parametrize("status", ["optimal", "infeasible"])
def test_no_budget_withdraws_the_certificate(status):
    """#1309 applied to the #1634 check: no time left to run it means no certificate."""
    c, A, b, ub, _ = WITNESSES[sorted(WITNESSES)[0]]  # noqa: N806
    _, _, sf = _highs_std_form(_model(c, A, b, ub))
    x = np.zeros(sf.n) if status == "optimal" else None
    out = lp_milp_highs.HighsOutcome(
        status, x=x, objective=0.0 if x is not None else None,
        bound=0.0 if x is not None else None, gap_certified=True, root_bound=-1.0,
    )  # fmt: skip
    out.wall_time = 5.0  # the primary solve spent the whole budget
    kw = {"time_limit": 5.0, "gap_tolerance": 1e-4}
    got = lp_milp_highs._cross_check_presolve(sf, out, kw)
    assert not got.gap_certified
    assert got.status == ("feasible" if status == "optimal" else "error")
    assert got.stats["milp/presolve_cross_check_skipped"] == 1.0
    assert got.labels["milp/certificate"] == "declined"
