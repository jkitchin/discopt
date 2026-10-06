"""#1634 reaches the OA / GDP master: HiGHS presolve prunes the master optimum.

The convex-MINLP route solves its OA master with ``milp_highs`` (always, since #1630).
A master whose model carries its own slack columns (``A x + s = b``, ``s >= 0``) has
exactly the structure #1634 found in the pure-MILP route: a zero-cost continuous
singleton parallel to an integer column, which HiGHS's parallel-column presolve rule
fixes on an absolute cost-tie test. The master bound then sits above the true master
optimum, and OA certifies it. Measured on the #1634 generator written that way (1200
instances, end to end through ``Model.solve``): 35 false certificates on ``main``.

Truth here is exhaustive enumeration over the integer box.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

pytest.importorskip("highspy")

from discopt import Model  # noqa: E402
from discopt.solvers import MILPResult, SolveStatus  # noqa: E402
from discopt.solvers import milp_highs as MH  # noqa: E402

# #1634 seed 181 (cost scale 1e-3): truth 0.008014783411187928 at x = [0, 0, 1, 0, 1].
C181 = [
    0.016422113453256008,
    1.5785686489149076,
    0.006402947197508945,
    0.23211550436289133,
    0.0016118362136789844,
]
A181 = [
    [
        -109159.8468922766,
        60334.201191650696,
        -490025.3894761287,
        10017.078735174448,
        -0.0027928104506268724,
    ],
    [
        -0.007588297250901298,
        0.016053747317747457,
        -0.962476149861379,
        0.279609118685822,
        -148.0240160211314,
    ],
]
B181 = [-57818.64349215431, -76.71966452895884]
UB181 = [3, 3, 3, 4, 3]
TRUTH181 = 0.008014783411187928
YARD = 1e-6 + 1e-6 * abs(TRUTH181)


def _gen(seed: int, n: int, objscale: float):
    """The #1634 generator."""
    rng = np.random.default_rng(seed)
    m = 2
    c = rng.choice([-1.0, 1.0], size=n) * 10 ** rng.uniform(-3, 4, n) * objscale
    A = rng.choice([-1.0, 1.0], size=(m, n)) * 10 ** rng.uniform(-3, 6, (m, n))
    ub = rng.integers(2, 5, n)
    x0 = np.array([rng.integers(0, u + 1) for u in ub], float)
    b = A @ x0 + np.abs(A) @ (ub * rng.uniform(0, 0.3))
    return c, A, b, ub


def _truth(c, A, b, ub) -> float:
    P = np.array(list(itertools.product(*[range(int(u) + 1) for u in ub])), float)
    feas = np.all(P @ np.asarray(A).T <= np.asarray(b), axis=1)
    assert feas.any()
    return float((P[feas] @ np.asarray(c)).min())


def _slack_master(c, A, b, ub):
    """The MILP in the form an OA master gets it: ``A x + s = b``, ``s >= 0``."""
    c, A = np.asarray(c, float), np.asarray(A, float)
    m, n = A.shape
    return dict(
        c=np.r_[c, np.zeros(m)],
        A_eq=np.c_[A, np.eye(m)],
        b_eq=np.asarray(b, float),
        bounds=[(0.0, float(u)) for u in ub] + [(0.0, None)] * m,
        integrality=np.r_[np.ones(n), np.zeros(m)],
        time_limit=60.0,
        gap_tolerance=1e-4,
    )


def test_truth_is_the_issue_value():
    assert _truth(C181, A181, B181, UB181) == pytest.approx(TRUTH181, abs=1e-15)


@pytest.mark.parametrize("route_method", ["oa", "lp_nlp_bb"])
def test_convex_minlp_route_does_not_certify_the_pruned_master(monkeypatch, route_method):
    """End to end: on ``main`` this certifies 0.02082 (``gap_certified=True``).
    Run under both route targets (#1658 moved the default to ``lp_nlp_bb``)."""
    monkeypatch.setenv("DISCOPT_CONVEX_ROUTE_METHOD", route_method)
    m = Model("oa1634")
    xs = [m.integer(f"x{j}", lb=0, ub=UB181[j]) for j in range(5)]
    z = m.continuous("z", lb=0.0, ub=1.0)
    m.minimize(sum(C181[j] * xs[j] for j in range(5)) + 1e-3 * (z - 0.3) ** 2)
    for i in range(2):
        s = m.continuous(f"s{i}", lb=0.0, ub=1e9)
        m.subject_to(sum(A181[i][j] * xs[j] for j in range(5)) + s == B181[i])
    r = m.solve(time_limit=60)
    route = str(r.algorithm_route)
    assert route.startswith(f"mip-nlp/{route_method}:") and "master=highs" in route, route
    assert r.bound is not None
    assert r.bound <= TRUTH181 + YARD, (r.bound, TRUTH181)
    if r.gap_certified:
        assert r.objective == pytest.approx(TRUTH181, abs=YARD)


def test_master_bound_is_valid_and_cross_checked():
    r = MH.solve_milp(**_slack_master(C181, A181, B181, UB181))
    cc = (r.callback_stats or {}).get("presolve_cross_check")
    assert cc is not None and cc["ran"], cc
    assert r.bound is not None and r.bound <= TRUTH181 + YARD, r.bound
    assert r.status == SolveStatus.OPTIMAL
    assert r.objective == pytest.approx(TRUTH181, abs=YARD)


def test_rule_13_off_alone_fixes_the_witness(monkeypatch):
    """With the cross-check disabled, the primary solve alone must already be right."""
    monkeypatch.setattr(MH, "_cross_check_presolve", lambda primary, *a, **k: primary)
    r = MH.solve_milp(**_slack_master(C181, A181, B181, UB181))
    assert r.bound is not None and r.bound <= TRUTH181 + YARD, r.bound


def test_rule_13_bit_is_the_parallel_rows_and_cols_rule():
    from discopt.solvers.lp_milp_highs import MILP_PRESOLVE_RULE_OFF

    assert MH.MILP_PRESOLVE_RULE_OFF is MILP_PRESOLVE_RULE_OFF
    assert MILP_PRESOLVE_RULE_OFF & (1 << 13)


def test_cross_check_catches_what_rule_13_does_not():
    """6-column seed 115 at cost scale 1: with rule 13 off HiGHS's presolve still
    certified -2.10546 against an enumerated -2.24028 (measured)."""
    c, A, b, ub = _gen(115, 6, 1.0)
    truth = _truth(c, A, b, ub)
    r = MH.solve_milp(**_slack_master(c, A, b, ub))
    assert r.bound is not None
    assert r.bound <= truth + 1e-6 + 1e-6 * abs(truth), (r.bound, truth)


def test_lazy_master_bound_is_valid():
    """The LP/NLP-BB master (``solve_milp_with_lazy_cuts``) with a separator that
    accepts every point is the plain master; its bound must be valid too."""
    kw = _slack_master(C181, A181, B181, UB181)
    calls = []

    def accept(x):
        calls.append(1)
        return None

    r = MH.solve_milp_with_lazy_cuts(lazy_callback=accept, **kw)
    assert calls, "the separator never ran"
    assert r.callback_stats["presolve_cross_check"]["ran"]
    assert r.bound is not None and r.bound <= TRUTH181 + YARD, r.bound


def _kw(**over):
    kw = _slack_master(C181, A181, B181, UB181)
    kw.pop("time_limit")
    kw["mip_start"] = None
    kw["A_ub"] = kw["b_ub"] = None
    kw.update(over)
    return kw


def test_false_infeasible_is_refuted_by_a_verified_point():
    kw = _kw()
    fake = MILPResult(status=SolveStatus.INFEASIBLE)
    r = MH._cross_check_presolve(fake, kw, None, 0.0)
    assert r.status == SolveStatus.ITERATION_LIMIT
    assert r.bound is None
    assert r.x is not None and MH._verified_objective(kw, r.x) is not None
    assert "refuted" in r.callback_stats["presolve_cross_check"]


def test_false_bound_is_lowered_below_the_verified_point():
    kw = _kw()
    x_bad = np.r_[[0, 0, 3, 0, 1], np.zeros(2)].astype(float)
    x_bad[5:] = np.asarray(B181) - np.asarray(A181) @ x_bad[:5]
    obj_bad = MH._verified_objective(kw, x_bad)
    assert obj_bad == pytest.approx(0.02082067780620582)
    fake = MILPResult(status=SolveStatus.OPTIMAL, x=x_bad, objective=obj_bad, bound=obj_bad)
    r = MH._cross_check_presolve(fake, kw, None, 0.0)
    assert r.bound <= TRUTH181 + YARD
    assert r.objective == pytest.approx(TRUTH181, abs=YARD)


def test_claim_is_withdrawn_without_budget():
    kw = _kw()
    fake = MILPResult(status=SolveStatus.OPTIMAL, x=None, objective=1.0, bound=1.0)
    r = MH._cross_check_presolve(fake, kw, 1.0, t0=-1e9)  # budget long spent
    assert r.bound is None
    assert r.status == SolveStatus.ITERATION_LIMIT
    assert "withdrawn" in r.callback_stats["presolve_cross_check"]


def _early_exit_model():
    """Convex MINLP whose LP/NLP-BB separator keeps cutting after the gap closes, so
    the driver's check-in (``callback_terminate``) stops the HiGHS master early
    (the #1066 ``_separator_outlives_the_certificate`` fixture)."""
    m = Model("oa1634_early_exit")
    xs = [m.continuous(f"x{i}", lb=0.0, ub=3.0) for i in range(8)]
    ys = [m.binary(f"y{i}") for i in range(8)]
    for xi, yi in zip(xs, ys):
        m.subject_to(xi <= 3.0 * yi)
    m.subject_to(sum(xs) >= 4.0)
    m.minimize(sum((xi - 0.5) ** 2 for xi in xs) + 0.001 * sum(ys))
    return m


#: The early-exit tests' gap tolerance. At the default 1e-4, whether the check-in ever
#: sees the gap closed before HiGHS finishes on its own depends on which OpenBLAS kernel
#: the runner's CPU selects: measured with ``OPENBLAS_CORETYPE``, the SkylakeX and
#: Prescott kernels exit early (413 / 534 incumbent offers), Haswell, Zen and Sandybridge
#: run to ``optimal`` (470) and never exit early -- so on an AVX2 CI runner the
#: unconfirmed-bound test failed and its siblings passed without testing anything. At
#: 1e-2 every one of those five kernels exits early (350-357 offers, against ~470 for a
#: run to completion) and certifies the same 0.008.
_EARLY_EXIT_GAP = 1e-2


def _lp_nlp_bb(model):
    """The single-tree driver on the HiGHS lazy master, called directly so a solve
    that does not certify is not replaced by the route's fallback, with a caller
    ``termination_hook`` that never stops (and must be seen to run)."""
    from discopt.solvers.oa import solve_lp_nlp_bb

    seen: list[dict] = []
    r = solve_lp_nlp_bb(
        model,
        time_limit=60.0,
        gap_tolerance=_EARLY_EXIT_GAP,
        milp_solver="highs",
        termination_hook=lambda ctx: bool(seen.append(dict(ctx))),
    )
    assert seen, "the termination hook never ran"
    return r


def test_lp_nlp_bb_early_exit_cannot_restore_an_unconfirmed_bound(monkeypatch):
    """The driver's early exit records the PRIMARY tree's check-in bound and used to
    publish ``max(master bound, check-in bound)``: a bound the presolve-free
    cross-solve lowered came straight back and certified on its own.

    The fault is injected at the cross-check (it reports the final master bound 1.0
    lower than it is), so the test does not depend on finding a rare instance where
    HiGHS's presolve-on tree is wrong with rule 13 off. A correct driver publishes
    the lowered bound and does not certify on it.
    """
    real = MH._cross_check_lazy_master
    fired = []

    def lowered(h, highspy, status, bound, time_left, objective=None):
        status, bound, diag = real(h, highspy, status, bound, time_left, objective=objective)
        if bound is not None:
            fired.append(bound)
            bound -= 1.0
            diag["bound_lowered"] = 1.0
            status = SolveStatus.ITERATION_LIMIT
        return status, bound, diag

    monkeypatch.setattr(MH, "_cross_check_lazy_master", lowered)
    r = _lp_nlp_bb(_early_exit_model())
    stats = r.mip_nlp_trace["summary"]["callback_stats"]
    assert fired, "the lazy-master cross-check never ran"
    assert stats["presolve_cross_check"]["ran"]
    # The early exit must actually have fired, or this tests nothing (CLAUDE.md §6).
    assert stats["early_exit_unconfirmed"] is True, stats
    assert stats["converged_early"] is False
    assert r.bound is not None and r.bound <= fired[-1] - 1.0 + 1e-9, (r.bound, fired)
    assert not r.gap_certified
    assert r.mip_nlp_trace["termination_reason"] == "early_exit_unconfirmed"


def test_lp_nlp_bb_early_exit_confirmed_by_a_bound_a_hair_lower(monkeypatch):
    """The presolve-free cross-solve lands a hair below the check-in reading on
    ordinary B&B noise (it did on the Linux CI runner). A bound that still passes
    the gap test confirms the stop; only it -- never the higher check-in value --
    is reported."""
    real = MH._cross_check_lazy_master
    fired = []

    def nudged(h, highspy, status, bound, time_left, objective=None):
        status, bound, diag = real(h, highspy, status, bound, time_left, objective=objective)
        if bound is not None:
            bound -= 1e-9
            fired.append(bound)
        return status, bound, diag

    monkeypatch.setattr(MH, "_cross_check_lazy_master", nudged)
    r = _lp_nlp_bb(_early_exit_model())
    stats = r.mip_nlp_trace["summary"]["callback_stats"]
    assert fired, "the lazy-master cross-check never ran"
    assert stats["converged_early"] is True, stats  # the early exit fired (CLAUDE.md §6)
    assert stats["early_exit_unconfirmed"] is False, stats
    assert r.gap_certified
    assert r.bound is not None and r.bound <= fired[-1] + 1e-12, (r.bound, fired)
    assert r.objective == pytest.approx(0.008, abs=1e-6)


def test_lp_nlp_bb_early_exit_still_certifies_when_confirmed():
    """Without a fault the cross-check confirms the check-in bound and the early exit
    certifies as before."""
    r = _lp_nlp_bb(_early_exit_model())
    stats = r.mip_nlp_trace["summary"]["callback_stats"]
    assert stats["presolve_cross_check"]["ran"]
    assert stats["converged_early"] is True, stats  # the early exit fired (CLAUDE.md §6)
    assert stats["early_exit_unconfirmed"] is False
    assert r.gap_certified
    assert r.objective == pytest.approx(0.008, abs=1e-6)


def test_lazy_cross_check_keeps_optimal_inside_the_gap():
    """A cross-solve bound a hair below the primary's, still inside ``mip_rel_gap`` of
    the incumbent, is B&B noise, not a refutation: ``OPTIMAL`` must survive. One that
    opens the gap must not."""
    assert MH._gap_closed(1.0, 1.0 - 5e-5, 1e-4)
    assert not MH._gap_closed(1.0, 0.99, 1e-4)
    assert not MH._gap_closed(None, 0.0, 1e-4)
