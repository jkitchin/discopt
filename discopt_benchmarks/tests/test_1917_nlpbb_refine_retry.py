"""#1917: NLP-BB retries an incumbent refine that hit its flat 5 s cap.

The refine in ``_solve_nlp_bb`` is capped at ``min(5.0, time left)``. On the
generated convex GDP ``cgdp.cover_n8k2_s2`` (hull, conic ON) its first call takes
1.7 s unloaded; at load1 60-96 it passed 5 s and returned TIME_LIMIT, the
unrefined incumbent violated row 49 by 5.2e-6, the #1561 exit gate refused it,
and all 3 panel rows reported ``feasible`` / uncertified (obj 47.0, bound
46.99999997039397, 1 node) where the gated rows certified. The tests force the
capped refines' TIME_LIMIT instead of loading the machine: forcing both 5 s-capped
attempts (the scaled refine and the unscaled retry) on the pre-fix tree reproduces
that row exactly; forcing only the first does not, as the unscaled retry rescues it.
"""

from __future__ import annotations

import warnings

import pytest

from benchmarks import convex_gdp_generator as cg

pytest.importorskip("pyomo.environ")

NAME = "cgdp.cover_n8k2_s2"
OPT = 47.0


def _is_refine(opts) -> bool:
    # The incumbent refine, the unscaled refine retry and the #1917 budget retry all
    # carry pounce_incumbent_options(); the first two use the flat 5 s cap, the #1917
    # retry the remaining budget (> 5 s).
    return opts.get("bound_relax_factor") == 0.0 and float(opts.get("max_wall_time", 0)) >= 5.0


def _forced(monkeypatch, mode):
    """Force refine-class calls to TIME_LIMIT and return a log of them.

    ``"capped"`` models the load: every call capped at the flat 5 s ends on
    TIME_LIMIT (the scaled refine *and* the unscaled retry, as at load1 60-96).
    ``"all"`` also forces the #1917 remaining-budget retry. ``"none"`` only logs.
    """
    from discopt import solver

    orig = solver._solve_node_nlp_kkt
    log = []

    def wrap(evaluator, x0, lb, ub, cb, opts):
        if _is_refine(opts):
            wall = float(opts["max_wall_time"])
            forced = mode == "all" or (mode == "capped" and wall <= 5.0)
            log.append((wall, forced))
            if forced:
                opts = dict(opts, max_wall_time=1e-9)
        return orig(evaluator, x0, lb, ub, cb, opts)

    monkeypatch.setattr(solver, "_solve_node_nlp_kkt", wrap)
    return log


def _solve(monkeypatch):
    from discopt import solver
    from discopt.pyomo.gdp import convert_pyomo_gdp

    monkeypatch.setenv("DISCOPT_HULL_CONIC_QUADRATIC", "1")
    monkeypatch.setattr(
        solver, "NLPBB_REFINE_RETRY_STATS", {"retries": 0, "recovered": 0}, raising=False
    )
    m = convert_pyomo_gdp(cg.builder(NAME)()).model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = m.solve(gdp_method="hull", time_limit=120)
    return r, solver.NLPBB_REFINE_RETRY_STATS


def test_refine_timeout_is_retried_and_the_certificate_survives(monkeypatch):
    log = _forced(monkeypatch, "capped")
    r, stats = _solve(monkeypatch)
    assert log and log[0] == (5.0, True), log  # the probe fired on the capped refine
    assert r.status == "optimal" and r.gap_certified, (r.status, r.gap_certified)
    assert stats == {"retries": 1, "recovered": 1}
    assert any(w > 5.0 and not f for w, f in log), log  # retry on the remaining budget
    assert r.objective == pytest.approx(OPT, abs=1e-6)
    assert r.bound <= r.objective


def test_both_attempts_timing_out_still_reports_uncertified(monkeypatch):
    log = _forced(monkeypatch, "all")
    r, stats = _solve(monkeypatch)
    assert any(w > 5.0 for w, _ in log) and all(f for _, f in log), log
    assert stats["retries"] == 1 and stats["recovered"] == 0
    # The pre-fix verdict: a feasible incumbent, no certificate claimed.
    assert r.status == "feasible" and not r.gap_certified
    assert r.objective == pytest.approx(OPT, abs=1e-6)
    assert r.bound is None or r.bound <= r.objective


def test_unforced_solve_never_retries(monkeypatch):
    log = _forced(monkeypatch, "none")
    r, stats = _solve(monkeypatch)
    assert log, "the refine never ran: the instrument measured nothing"
    assert stats == {"retries": 0, "recovered": 0}
    assert r.status == "optimal" and r.gap_certified
