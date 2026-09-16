"""Primal-dual warm start and ``SolveResult.kkt`` (#1247).

The plugin behind #1249 traces a phase diagram by re-solving a restricted
equilibrium NLP at condition after neighbouring condition, and states its
certificate in terms of that solve's KKT residual. Before this change
``Model.solve`` passed POUNCE a *point* and nothing else, and threw away the
terminal residuals POUNCE reports.

Measured on the restricted-equilibrium NLP used below (6 species, two elemental
balances, ``x ln x`` free energy), after a 1% change in the element amounts:

    condition   cold   initial_solution=   warm_start=
    +1%            7        5                  2
    +2%            7        5                  2
    +5%            7        5                  3
    -1%            7        5                  2

with the same optimum to 1e-7 in every arm. The dual information is what buys
the second halving; the point alone does not.
"""

from __future__ import annotations

import discopt.modeling as dm
import discopt.solvers.nlp_pounce as nlp_pounce
import numpy as np
import pytest

pytestmark = [pytest.mark.smoke, pytest.mark.requires_pounce]

_KKT_KEYS = {
    "primal_infeasibility",
    "dual_infeasibility",
    "complementarity",
    "kkt_error",
    "primal_infeasibility_unscaled",
    "dual_infeasibility_unscaled",
    "complementarity_unscaled",
    "kkt_error_unscaled",
    "barrier_parameter",
}


def _restricted_equilibrium():
    """Convex Gibbs-shaped NLP: ``min sum_i (y_i ln y_i + g_i y_i)`` s.t. ``A y = b``.

    The single-NLP route takes this (the objective is convex, the constraints
    linear), which is the route the plugin's restricted equilibrium takes.
    """
    n = 6
    g_vals = np.random.default_rng(0).normal(0.0, 1.0, size=n)
    a_rows = np.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0, 2.0, 1.0]])
    m = dm.Model("restricted_equilibrium")
    y = [m.continuous(f"y{k}", lb=1e-8, ub=10.0) for k in range(n)]
    g = [m.parameter(f"g{k}", float(g_vals[k])) for k in range(n)]
    b0 = m.parameter("b0", 1.0)
    b1 = m.parameter("b1", 2.0)
    m.minimize(sum(dm.xlogx(y[k]) + g[k] * y[k] for k in range(n)))
    m.subject_to(sum(float(a_rows[0, k]) * y[k] for k in range(n)) == b0)
    m.subject_to(sum(float(a_rows[1, k]) * y[k] for k in range(n)) == b1)
    return m, (b0, b1)


def _iteration_recorder(monkeypatch):
    """Record POUNCE iteration counts, in call order, for the solves that follow."""
    seen: list[int] = []
    original = nlp_pounce.solve_nlp

    def recording(*args, **kwargs):
        result = original(*args, **kwargs)
        seen.append(int(result.iterations))
        return result

    # ``_solve_continuous`` imports the symbol inside the function body, so
    # patching the module attribute is what the solve actually picks up.
    monkeypatch.setattr(nlp_pounce, "solve_nlp", recording)
    return seen


# --------------------------------------------------------------------------- #
# SolveResult.kkt
# --------------------------------------------------------------------------- #
def test_kkt_reported_on_the_single_nlp_route():
    m, _b = _restricted_equilibrium()
    r = m.solve(time_limit=30)
    assert r.status == "optimal", r.status
    assert r.kkt is not None, "the single-NLP route must report its terminal KKT residuals"
    assert set(r.kkt) == _KKT_KEYS, sorted(r.kkt)
    assert r.kkt["kkt_error"] < 1e-6, r.kkt
    assert r.kkt["primal_infeasibility"] < 1e-6, r.kkt
    assert r.kkt["barrier_parameter"] > 0.0, r.kkt


def test_kkt_error_is_pounces_final_kkt_error(monkeypatch):
    """Acceptance criterion 2: the reported number is POUNCE's, not a re-derivation."""
    captured: list[dict] = []
    original = nlp_pounce._kkt_from_info

    def capturing(info):
        captured.append(dict(info))
        return original(info)

    monkeypatch.setattr(nlp_pounce, "_kkt_from_info", capturing)

    m, _b = _restricted_equilibrium()
    r = m.solve(time_limit=30)
    assert captured, "the probe never saw a POUNCE info dict"
    info = captured[-1]
    assert r.kkt["kkt_error"] == float(info["final_kkt_error"])
    assert r.kkt["primal_infeasibility"] == float(info["final_constr_viol"])
    assert r.kkt["dual_infeasibility"] == float(info["final_dual_inf"])
    assert r.kkt["complementarity"] == float(info["final_compl"])
    assert r.kkt["kkt_error_unscaled"] == float(info["final_unscaled_kkt_error"])
    assert r.kkt["barrier_parameter"] == float(info["mu"])


def test_kkt_is_absent_on_a_branch_and_bound_route():
    """No single NLP, no residuals to report — the field fails closed rather than
    carrying a number from some inner subproblem."""
    m = dm.Model("nonconvex_bilinear")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    m.minimize(x * y - x - y)
    m.subject_to(x + y <= 1.5)
    r = m.solve(time_limit=30)
    assert r.status == "optimal", r.status
    assert r.kkt is None


# --------------------------------------------------------------------------- #
# The warm start itself
# --------------------------------------------------------------------------- #
def test_warm_start_beats_a_primal_only_start_after_a_small_change(monkeypatch):
    """Acceptance criterion 1: fewer POUNCE iterations than ``initial_solution``,
    with the same optimum."""
    m, (b0, b1) = _restricted_equilibrium()
    base = m.solve(time_limit=30)
    assert base.status == "optimal"

    b0.value = 1.01  # a 1% move in the element amounts
    b1.value = 2.02

    iters = _iteration_recorder(monkeypatch)

    del iters[:]
    cold = m.solve(time_limit=30)
    n_cold = iters[-1]

    del iters[:]
    primal = m.solve(time_limit=30, initial_solution={v: base.x[v.name] for v in m._variables})
    n_primal = iters[-1]

    del iters[:]
    warm = m.solve(time_limit=30, warm_start=base)
    n_warm = iters[-1]

    assert cold.status == primal.status == warm.status == "optimal"
    assert warm.objective == pytest.approx(cold.objective, abs=1e-7)
    assert primal.objective == pytest.approx(cold.objective, abs=1e-7)
    # The dual information has to buy something over the point alone, which is
    # the whole claim of the issue.
    assert n_warm < n_primal <= n_cold, (n_cold, n_primal, n_warm)


def test_warm_start_forwards_the_duals_and_the_barrier_parameter(monkeypatch):
    """The state POUNCE receives is the previous solve's, not a re-derivation."""
    seen: list[object] = []
    original = nlp_pounce.solve_nlp

    def recording(*args, **kwargs):
        seen.append(kwargs.get("warm_start"))
        return original(*args, **kwargs)

    monkeypatch.setattr(nlp_pounce, "solve_nlp", recording)

    m, (b0, _b1) = _restricted_equilibrium()
    base = m.solve(time_limit=30)
    b0.value = 1.01
    del seen[:]
    m.solve(time_limit=30, warm_start=base)

    states = [s for s in seen if s is not None]
    assert states, "no warm start reached the NLP backend"
    ws = states[-1]
    assert ws.lagrange is not None and len(ws.lagrange) == 2
    assert ws.zl is not None and len(ws.zl) == len(m._variables)
    assert ws.zu is not None and len(ws.zu) == len(m._variables)
    assert ws.mu == pytest.approx(base.kkt["barrier_parameter"])
    np.testing.assert_allclose(
        np.asarray(ws.x), np.array([float(base.x[v.name]) for v in m._variables]), atol=0
    )


def test_warm_start_on_a_global_route_uses_the_point():
    """A nonconvex model has no primal-dual seam, but the point is still a start —
    it must not be dropped, and it must not change the certified answer."""
    m = dm.Model("nonconvex_bilinear_ws")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    m.minimize(x * y - x - y)
    m.subject_to(x + y <= 1.5)
    first = m.solve(time_limit=30)
    again = m.solve(time_limit=30, warm_start=first)
    assert again.status == "optimal"
    assert again.objective == pytest.approx(first.objective, abs=1e-6)
    assert again.bound is not None and again.bound <= again.objective + 1e-6


# --------------------------------------------------------------------------- #
# Refusals: a warm start that silently does nothing is the failure to avoid
# --------------------------------------------------------------------------- #
def test_warm_start_with_initial_solution_raises():
    m, _b = _restricted_equilibrium()
    base = m.solve(time_limit=30)
    with pytest.raises(ValueError, match="two different starting points"):
        m.solve(
            time_limit=30,
            warm_start=base,
            initial_solution={v: base.x[v.name] for v in m._variables},
        )


def test_warm_start_of_the_wrong_type_raises():
    m, _b = _restricted_equilibrium()
    with pytest.raises(TypeError, match="must be a SolveResult"):
        m.solve(time_limit=30, warm_start={"y0": 1.0})


def test_warm_start_from_another_model_raises():
    m, _b = _restricted_equilibrium()
    other = dm.Model("other")
    z = other.continuous("z", lb=0, ub=1)
    other.minimize((z - 0.3) ** 2)
    foreign = other.solve(time_limit=30)
    with pytest.raises(ValueError, match="no value for variable"):
        m.solve(time_limit=30, warm_start=foreign)


def test_warm_start_from_a_result_without_a_point_raises():
    m, _b = _restricted_equilibrium()
    empty = dm.SolveResult(status="infeasible")
    with pytest.raises(ValueError, match="no solution vector"):
        m.solve(time_limit=30, warm_start=empty)


def test_streaming_refuses_a_warm_start():
    """The streaming driver has no seam for a start point or its duals, so taking
    one would make it silently inert."""
    m, _b = _restricted_equilibrium()
    base = m.solve(time_limit=30)
    with pytest.raises(ValueError, match="stream=True"):
        m.solve(time_limit=30, stream=True, warm_start=base)


def test_an_unknown_pounce_option_is_named_not_swallowed():
    """#1247's last item. pounce validates option NAMES at solve time, so a typo
    used to surface as a raw Rust ``OPTION_INVALID`` from inside the solver."""
    m, _b = _restricted_equilibrium()
    with pytest.raises(ValueError, match="POUNCE rejected a solver option"):
        m.solve(time_limit=30, ipopt_options={"this_option_does_not_exist": 3})
