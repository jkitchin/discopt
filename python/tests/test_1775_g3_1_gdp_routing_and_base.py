"""#1775 G3.1: the native GDP axis, ``gdp_options`` and ``algorithms/base.py``.

Before: the native ``gdp_method`` axis was ``{"loa"}``; ``"gloa"``, ``"lbb"`` and
``"ric"`` were refused as *unknown* names, ``Model.solve`` had no ``gdp_options``
argument, and there was no shared loop for the native algorithms.

After (NC; routing approved by the owner, 2026-10-21):

* the native axis is ``{loa, gloa, lbb, ric, enumerate, ldsda}``; each name whose
  algorithm has not landed is refused as *reserved*, and solve-routing §4 rule 1
  (a native method with ``solver="mip-nlp"`` raises) covers every one of them;
* ``gdp_options`` takes Pyomo GDPopt's option names; an unknown key, a key the
  method does not read, and options with a reformulation method all raise;
* ``GDPAlgorithm`` owns the deadline, the min-convention LB/UB and the §5.4 status
  rules: ``optimal`` only with a valid LB, nothing unresolved and a closed gap.
"""

from __future__ import annotations

import math
import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.gdp.algorithms import Deadline, GDPAlgorithm
from discopt.solver import IMPLEMENTED_NATIVE_GDP_METHODS, NATIVE_GDP_METHODS

pytestmark = pytest.mark.smoke

RESERVED = sorted(NATIVE_GDP_METHODS - IMPLEMENTED_NATIVE_GDP_METHODS)


def _gdp(sense="min"):
    """min (x - 3)^2 over [x <= 2] v [x >= 5], x in [0, 10]: optimum 1 at x = 2."""
    m = dm.Model("g31")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 2], [x >= 5]], name="d")
    if sense == "min":
        m.minimize((x - 3) ** 2)
    else:
        m.maximize(-((x - 3) ** 2))
    return m


# ── routing ──────────────────────────────────────────────────────────────────


def test_native_axis_is_the_spec_set():
    assert NATIVE_GDP_METHODS == {"loa", "gloa", "lbb", "ric", "enumerate", "ldsda"}
    assert RESERVED == ["enumerate", "gloa", "lbb", "ldsda", "ric"]


@pytest.mark.parametrize("method", RESERVED)
def test_reserved_native_method_is_refused_as_reserved(method):
    with pytest.raises(ValueError, match="reserved on the native GDP axis"):
        _gdp().solve(gdp_method=method, time_limit=10)


@pytest.mark.parametrize("method", sorted(NATIVE_GDP_METHODS))
def test_rule_1_covers_every_native_method(method):
    with pytest.raises(ValueError, match="conflicts with solver='mip-nlp'"):
        _gdp().solve(gdp_method=method, solver="mip-nlp", time_limit=10)


@pytest.mark.parametrize(
    "method, options, match",
    [
        ("loa", {"no_such_option": 1}, "Unknown gdp_options key"),
        ("loa", {"init_algorithm": "set_covering"}, "does not read gdp_options"),
        ("big-m", {"iterlim": 3}, "which reads none"),
        ("hull", {"iterlim": 3}, "which reads none"),
        ("loa", {"iterlim": 0}, "positive integer"),
        ("loa", {"iterlim": 2.5}, "positive integer"),
        ("loa", {"iterlim": True}, "positive integer"),
    ],
)
def test_gdp_options_refusals(method, options, match):
    with pytest.raises(ValueError, match=match):
        _gdp().solve(gdp_method=method, gdp_options=options, time_limit=10)


def test_gdp_options_must_be_a_mapping():
    with pytest.raises(TypeError, match="gdp_options must be a mapping"):
        _gdp().solve(gdp_method="loa", gdp_options=[("iterlim", 3)], time_limit=10)


def test_gdp_options_are_not_dropped_by_the_convex_kernel(monkeypatch):
    """Model.solve's convex-kernel shortcut returns without reaching solve_model;
    it must not take a solve that carries gdp_options (review of #1861). The
    kernel is stubbed to certify, which is what it does on its own class
    (clay0303hfsg, syn05hfsg) -- the small models here it declines."""
    import discopt.solvers._convex_kernel as ck
    from discopt.modeling.core import SolveResult

    calls = []

    def certify(model, **kw):
        calls.append(1)
        return SolveResult(status="optimal", objective=0.0, bound=0.0, x={"x": np.array(1.0)})

    monkeypatch.setattr(ck, "try_convex_solve", certify)
    m = dm.Model("convex")
    x = m.continuous("x", lb=-5, ub=5)
    m.minimize((x - 1) ** 2)
    assert m.solve(time_limit=10).status == "optimal" and calls == [1]  # the stub is live
    refusals = (({"iterlmi": 5}, "Unknown gdp_options key"), ({"iterlim": 5}, "reads none"))
    for options, match in refusals:
        with pytest.raises(ValueError, match=match):
            m.solve(gdp_options=options, time_limit=10)
    assert calls == [1]


def test_empty_gdp_options_with_a_reformulation_is_fine():
    r = _gdp().solve(gdp_method="hull", gdp_options={}, time_limit=30)
    assert r.status == "optimal" and r.objective == pytest.approx(1.0, abs=1e-5)


def test_loa_reads_iterlim(monkeypatch):
    import discopt.solvers.gdpopt_loa as loa_mod

    seen = {}
    real = loa_mod.solve_gdpopt_loa

    def spy(model, *args, **kwargs):
        seen["max_iterations"] = kwargs["max_iterations"]
        return real(model, *args, **kwargs)

    monkeypatch.setattr(loa_mod, "solve_gdpopt_loa", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = _gdp().solve(gdp_method="loa", gdp_options={"iterlim": 7}, time_limit=30)
    assert seen == {"max_iterations": 7}
    assert r.status == "optimal" and r.objective == pytest.approx(1.0, abs=1e-5)


# ── Deadline ─────────────────────────────────────────────────────────────────


class _Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def test_deadline_counts_down_and_caps_subsolves():
    c = _Clock()
    d = Deadline(10.0, clock=c)
    assert d.remaining() == 10.0 and d.sub_limit(3.0) == 3.0
    c.t += 8.5
    assert d.remaining() == pytest.approx(1.5) and d.sub_limit(3.0) == pytest.approx(1.5)
    c.t += 5
    assert d.expired() and d.remaining() == 0.0 and d.sub_limit() == 0.0
    assert Deadline(None, clock=c).remaining() == math.inf
    with pytest.raises(ValueError):
        Deadline(-1.0)


# ── GDPAlgorithm status rules (spec §5.4) ────────────────────────────────────


class _Scripted(GDPAlgorithm):
    """Plays a list of steps, one per iteration: ("x", point) offers an
    incumbent, ("lb", v) a bound, ("unresolved", cfg), ("solved", cfg),
    ("invalid", why), ("infeasible",), ("tick", seconds)."""

    route = "gdp:test"

    def __init__(self, model, steps, **kw):
        self.clock = _Clock()
        super().__init__(model, clock=self.clock, **kw)
        self.steps = list(steps)
        self.accepted = []

    def iterate(self):
        kind, *arg = self.steps.pop(0)
        if kind == "x":
            self.accepted.append(self.offer_incumbent(np.array([arg[0]])))
        elif kind == "lb":
            self.offer_bound(arg[0])
        elif kind == "unresolved":
            self.mark_unresolved(arg[0])
        elif kind == "solved":
            self.mark_solved(arg[0])
        elif kind == "invalid":
            self.invalidate_bound(arg[0])
        elif kind == "infeasible":
            self.prove_infeasible()
        elif kind == "tick":
            self.clock.t += arg[0]
        self.n_subproblems += 1
        return bool(self.steps)


def _run(steps, sense="min", **kw):
    alg = _Scripted(_gdp(sense), steps, **kw)
    return alg, alg.run()


def test_closed_gap_is_optimal_and_stops_early():
    alg, r = _run([("x", 2.0), ("lb", 1.0), ("x", 5.0)])
    assert r.status == "optimal" and r.objective == pytest.approx(1.0)
    assert r.bound == pytest.approx(1.0) and r.bound_valid and r.gap_certified
    assert alg.stop_reason == "converged" and len(alg.steps) == 1
    assert r.algorithm_route == "gdp:test"
    assert r.solver_stats["iterations"] == 2 and r.solver_stats["n_subproblems"] == 2
    assert [t[:3] for t in r.solver_stats["trace"]] == [(1, -math.inf, 1.0), (2, 1.0, 1.0)]
    assert float(r.x["x"]) == 2.0


def test_maximize_returns_model_units():
    _, r = _run([("x", 2.0), ("lb", 1.0)], sense="max")
    assert r.status == "optimal"
    assert r.objective == pytest.approx(-1.0) and r.bound == pytest.approx(-1.0)


def test_iteration_limit_with_an_open_gap_is_feasible_not_optimal():
    alg, r = _run([("x", 5.0), ("lb", 0.5), ("x", 2.0)], iteration_limit=2)
    assert alg.stop_reason == "iteration_limit"
    assert r.status == "feasible" and r.objective == pytest.approx(4.0)
    assert r.bound == pytest.approx(0.5) and r.gap == pytest.approx(3.5 / 4.0)
    # A valid bound on an open gap is not a certificate (review of #1861).
    assert r.bound_valid and not r.gap_certified


def test_time_limit_with_an_open_gap_is_not_certified():
    alg, r = _run([("x", 5.0), ("lb", 0.5), ("tick", 10.0), ("x", 2.0)], time_limit=5.0)
    assert alg.stop_reason == "time_limit" and r.status == "feasible"
    assert r.bound_valid and not r.gap_certified


def test_bound_within_tolerance_above_ub_is_published_clamped():
    """offer_bound admits ub + tol; the reported bound never exceeds the objective."""
    _, r = _run([("x", 2.0), ("lb", 1.0 + 5e-5)])
    assert r.status == "optimal" and r.gap_certified
    assert r.bound == r.objective == pytest.approx(1.0) and r.gap == 0.0


def test_reported_gap_agrees_with_status_near_zero():
    """|ub| < 1: a gap reported under gap_tolerance must mean converged."""
    m = dm.Model("small")
    x = m.continuous("x", lb=0, ub=1)
    m.either_or([[x <= 0.2], [x >= 0.5]], name="d")
    m.minimize(x * 0.01)
    alg = _Scripted(m, [("x", 0.2), ("lb", 0.0019)], gap_tolerance=1e-3)
    r = alg.run()
    # ub = 0.002, lb = 0.0019: diff 1e-4 > 1e-3 * 0.002, so not converged ...
    assert r.status == "feasible"
    # ... and the reported gap says so (it was 1e-4 under a max(1, |ub|) scale).
    assert r.gap > alg.gap_tolerance


def test_an_unresolved_configuration_blocks_optimal():
    """Gap closed on paper, but a configuration was never resolved (C-35)."""
    _, r = _run([("unresolved", (1,)), ("x", 2.0), ("lb", 1.0)])
    assert r.status == "feasible"
    assert r.bound is None and not r.bound_valid and not r.gap_certified
    assert r.solver_stats["n_unresolved"] == 1


def test_resolving_a_configuration_clears_it():
    _, r = _run([("unresolved", (1,)), ("solved", (1,)), ("x", 2.0), ("lb", 1.0)])
    assert r.status == "optimal" and r.solver_stats["n_unresolved"] == 0


def test_an_invalid_lb_never_certifies():
    """LOA with a non-convex OA'd row: the master value is not a bound."""
    _, r = _run([("invalid", "row 3 not certified convex"), ("x", 2.0), ("lb", 1.0)])
    assert r.status == "feasible" and r.bound is None and not r.gap_certified
    assert r.solver_stats["lb_invalid_reason"] == "row 3 not certified convex"


def test_an_unverified_incumbent_is_rejected():
    """x = 3 satisfies the bounds but neither disjunct: DeclaredLogic refuses it."""
    alg, r = _run([("x", 3.0), ("x", 11.0)])
    assert alg.accepted == [False, False]
    assert r.status == "unknown" and r.x is None and r.objective is None


def test_infeasible_needs_a_proof_with_nothing_unresolved():
    _, r = _run([("infeasible",)])
    assert r.status == "infeasible" and r.bound is None
    _, r = _run([("unresolved", (0,)), ("infeasible",)])
    assert r.status == "unknown"


def test_time_limit_without_incumbent():
    alg, r = _run([("tick", 4.0), ("tick", 4.0), ("tick", 4.0), ("x", 2.0)], time_limit=5.0)
    assert alg.stop_reason == "time_limit" and r.status == "time_limit"


def test_iteration_limit_without_incumbent():
    _, r = _run([("lb", 0.0), ("lb", 0.0), ("lb", 0.0)], iteration_limit=1)
    assert r.status == "iteration_limit" and r.bound == pytest.approx(0.0)


def test_exhausted_without_proof_is_unknown():
    alg, r = _run([("lb", 0.0)])
    assert alg.stop_reason == "exhausted" and r.status == "unknown"


def test_a_bound_crossing_the_incumbent_raises():
    with pytest.raises(RuntimeError, match="crosses the verified incumbent"):
        _run([("x", 2.0), ("lb", 1.5)])
    with pytest.raises(RuntimeError, match="lies below the bound"):
        _run([("lb", 3.0), ("x", 2.0)])


def test_infeasibility_proof_with_an_incumbent_raises():
    alg = _Scripted(_gdp(), [("x", 2.0)])
    alg.run()
    alg.prove_infeasible()
    with pytest.raises(RuntimeError, match="infeasibility was proved"):
        alg.result()
