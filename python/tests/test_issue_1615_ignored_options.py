"""Regression tests for #1615: options accepted and silently ignored.

Each option must either take effect or be refused/declared loudly. Covered here:

* D-03 -- an explicit ``tuning=`` never routed off the native spatial kernel,
  because ``_scoped_tuning`` popped it out of ``kwargs`` before the
  ``_native_kernel_bypass_reasons`` test that looked for it.
* D-01/D-02/D-08/D-13 -- search levers the native kernel does not read
  (``rlt``, ``strategy``, ``presolve``,
  ``in_tree_presolve_stride``, ``obbt_at_root``) now route to the Python engine.
* B-08 -- an explicit ``mu_init`` is honoured under ``warm_start=``.
* C-10a -- ``gdp_method="hull"`` lowers ``add_disjunction`` blocks by hull, with
  the Disjunct indicators as selectors; plain ``if_then`` rows (and a block the
  hull cannot take) are declared with a warning.
* C-10b -- the convex MINLP kernel (big-M only) is not attempted when the caller
  asked for another GDP reformulation.
* B-01b -- ``initial_solution`` / ``warm_start`` reach POUNCE's qp-ipm as its
  ``warm_start``; the lp-ipm arm (still a cold solve) warns.
* D-12 -- a looser relative ``gap_tolerance`` is honoured by the native kernel.
* B-02a -- ``skip_convex_check`` is documented as what it does (one local NLP).
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import discopt.solver as solver_mod
import pytest
from discopt import SolverTuning


def _haverly():
    m = dm.Model("haverly")
    fA, fB, fC = (m.continuous(n, lb=0, ub=300) for n in ["fA", "fB", "fC"])
    x, y = m.continuous("x", lb=0, ub=100), m.continuous("y", lb=0, ub=200)
    cX, cY = m.continuous("cX", lb=0, ub=100), m.continuous("cY", lb=0, ub=200)
    p = m.continuous("p", lb=1, ub=3)
    m.subject_to(fA + fB == x + y)
    m.subject_to(p * (x + y) == 3 * fA + fB)
    m.subject_to(fC == cX + cY)
    m.subject_to(x + cX <= 100)
    m.subject_to(y + cY <= 200)
    m.subject_to(p * x + 2 * cX <= 2.5 * (x + cX))
    m.subject_to(p * y + 2 * cY <= 1.5 * (y + cY))
    m.maximize(9 * (x + cX) + 15 * (y + cY) - 6 * fA - 16 * fB - 10 * fC)
    return m


@pytest.fixture
def kernel_calls(monkeypatch):
    """Record native-kernel hand-offs; decline them so the Python engine solves."""
    calls: list[int] = []

    def _fake(*args, **kwargs):
        calls.append(1)
        return None

    monkeypatch.setenv("DISCOPT_NATIVE_SPATIAL_KERNEL", "1")
    monkeypatch.setattr(solver_mod, "_try_native_spatial_kernel", _fake)
    return calls


def test_default_solve_reaches_native_kernel(kernel_calls):
    """Control arm: proves the probe fires, so the routed-off tests mean something."""
    r = _haverly().solve(time_limit=60)
    assert kernel_calls, "default solve never consulted the native kernel"
    assert r.objective == pytest.approx(400.0, abs=1e-3)


def test_explicit_tuning_routes_off_native_kernel(kernel_calls):
    r = _haverly().solve(time_limit=60, tuning=SolverTuning(root_fixpoint=False))
    assert kernel_calls == []
    assert r.objective == pytest.approx(400.0, abs=1e-3)


@pytest.mark.parametrize(
    "kw",
    [
        {"rlt": True},
        {"rlt": False},
        {"strategy": "depth_first"},
        {"presolve": False},
        {"in_tree_presolve_stride": 0},
        {"obbt_at_root": False},
    ],
)
def test_kernel_ignored_levers_route_to_python_engine(kernel_calls, kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _haverly().solve(time_limit=60, **kw)
    assert kernel_calls == [], f"{kw} still handed the solve to the native kernel"
    assert r.objective == pytest.approx(400.0, abs=1e-3)


def test_kernel_lever_defaults_are_not_flagged():
    assert (
        solver_mod._native_kernel_ignored_levers(
            strategy="best_first",
            rlt="auto",
            presolve=True,
            in_tree_presolve_stride=1,
            kwargs={},
        )
        == []
    )


def test_mu_init_honoured_under_warm_start():
    m = dm.Model("w")
    p = m.parameter("p", 1.0)
    x = m.continuous("x", shape=(2,), lb=-5, ub=5)
    m.minimize((x[0] - p) ** 2 + (x[1] - 2 * p) ** 2 + dm.exp(0.1 * x[0]))
    m.subject_to(x[0] + x[1] <= 3 * p)
    r0 = m.solve(solver="pounce")
    p.value = 1.1
    base = m.solve(solver="pounce", warm_start=r0)
    r = m.solve(solver="pounce", warm_start=r0, pounce_options={"mu_init": 0.1})
    mu0_base = base.solve_report["iterations"][0]["mu"]
    mu0 = r.solve_report["iterations"][0]["mu"]
    assert mu0_base < 1e-3  # the warm-start-derived value
    assert mu0 == pytest.approx(0.1)
    assert r.objective == pytest.approx(base.objective, abs=1e-6)


def _twobox(api):
    m = dm.Model("twobox")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    A = [x >= 4, x <= 5, y >= 6, y <= 7]
    B = [x >= 7, x <= 8, y >= 2, y <= 3]
    if api == "either_or":
        m.either_or([A, B])
    else:
        dA, dB = m.make_disjunct("A"), m.make_disjunct("B")
        for c in A:
            dA.subject_to(c)
        for c in B:
            dB.subject_to(c)
        m.add_disjunction([dA, dB])
    m.minimize(x + y)
    return m


def _gdp_warnings(fn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = fn()
    return r, [str(w.message) for w in rec if "gdp_method" in str(w.message)]


def test_hull_on_add_disjunction_is_honoured():
    # C-10a: the Disjunct blocks take the hull -- the same root bound as either_or
    # (9.00), not big-M's 6.30 -- and nothing warns.
    r, msgs = _gdp_warnings(lambda: _twobox("make_disjunct").solve(gdp_method="hull"))
    assert msgs == []
    assert r.status == "optimal"
    assert r.objective == pytest.approx(9.0, abs=1e-6)
    assert r.root_bound == pytest.approx(9.0, abs=1e-4)


def test_bigm_on_add_disjunction_is_unchanged():
    # Control: big-M on the same blocks keeps its weaker root bound.
    r = _twobox("make_disjunct").solve(gdp_method="big-m")
    assert r.objective == pytest.approx(9.0, abs=1e-6)
    assert r.root_bound == pytest.approx(6.3, abs=1e-4)


def test_hull_on_add_disjunction_uses_the_user_indicators():
    # The hull's selectors ARE the Disjunct indicators, so a constraint on an
    # indicator still binds: forbidding box B forces box A (min x+y = 10).
    from discopt._relax.gdp_reformulate import reformulate_gdp

    m = dm.Model("twobox_fix")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    dA, dB = m.make_disjunct("A"), m.make_disjunct("B")
    for c in [x >= 4, x <= 5, y >= 6, y <= 7]:
        dA.subject_to(c)
    for c in [x >= 7, x <= 8, y >= 2, y <= 3]:
        dB.subject_to(c)
    m.add_disjunction([dA, dB])
    m.subject_to(dB.indicator.variable <= 0)
    m.minimize(x + y)
    lowered = reformulate_gdp(m, method="hull")
    assert not any(v.name.startswith("_gdp_aux_hull") for v in lowered._variables)
    assert any(v.name.startswith("_hull_") for v in lowered._variables)
    r = m.solve(gdp_method="hull")
    assert r.objective == pytest.approx(10.0, abs=1e-6)


def test_hull_on_nonlinear_add_disjunction_matches_either_or_hull():
    # Perspective rows on the user's indicators: the same lowering, hence the same
    # solve, as the either_or hull of the same disjunction. (Compared against the
    # either_or hull rather than the true optimum: both currently accept a point
    # 5.9e-5 infeasible on the exp row -- pre-existing, tracked in #1659.)
    def build(api):
        m = dm.Model("nl_disj")
        x = m.continuous("x", lb=0, ub=4)
        y = m.continuous("y", lb=0, ub=20)
        A = [y >= dm.exp(x) - 1, x <= 1]
        B = [y >= x**2 + 3, x >= 2]
        if api == "either_or":
            m.either_or([A, B])
        else:
            d1, d2 = m.make_disjunct("lo"), m.make_disjunct("hi")
            for c in A:
                d1.subject_to(c)
            for c in B:
                d2.subject_to(c)
            m.add_disjunction([d1, d2])
        m.minimize(y - 2 * x)
        return m

    re = build("either_or").solve(gdp_method="hull")
    rh, msgs = _gdp_warnings(lambda: build("make_disjunct").solve(gdp_method="hull"))
    assert msgs == []
    assert rh.status == re.status == "optimal"
    assert rh.objective == pytest.approx(re.objective, abs=1e-8)


def test_hull_on_plain_if_then_still_warns():
    m = dm.Model("ifthen")
    x = m.continuous("x", lb=0, ub=10)
    z = m.binary("z")
    m.if_then(z, [x >= 5])
    m.subject_to(z >= 1)
    m.minimize(x)
    with pytest.warns(UserWarning, match=r"gdp_method='hull' does not apply to the 1 indicator"):
        r = m.solve(gdp_method="hull")
    assert r.objective == pytest.approx(5.0, abs=1e-6)


def test_hull_on_add_disjunction_with_empty_disjunct_warns():
    # An empty disjunct leaves no row carrying its indicator, so the hull has no
    # selector for it; the block stays on big-M and says so.
    m = dm.Model("empty_disj")
    x = m.continuous("x", lb=0, ub=10)
    dA, dB = m.make_disjunct("A"), m.make_disjunct("B")
    dA.subject_to(x >= 5)
    m.add_disjunction([dA, dB])
    m.minimize(-x)
    with pytest.warns(UserWarning, match=r"does not apply to the 1 indicator"):
        r = m.solve(gdp_method="hull")
    assert r.objective == pytest.approx(-10.0, abs=1e-6)


def test_add_disjunction_tag_survives_copy_and_serialization():
    from discopt import serialize
    from discopt.modeling.core import _IndicatorConstraint

    m = _twobox("make_disjunct")

    def tags(model):
        return [c.disjunction for c in model._constraints if isinstance(c, _IndicatorConstraint)]

    assert all(t is not None for t in tags(m)) and len(tags(m)) == 8
    assert tags(m.clone()) == tags(m)
    assert tags(serialize.loads(serialize.dumps(m))) == tags(m)


def test_hull_on_either_or_is_silent_and_takes_effect():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = _twobox("either_or").solve(gdp_method="hull")
    assert not [w for w in rec if "gdp_method" in str(w.message)]
    assert r.root_bound == pytest.approx(9.0, abs=1e-4)


def test_convex_kernel_not_attempted_for_non_bigm_gdp(monkeypatch):
    import discopt.solvers._convex_kernel as ck

    calls: list[str] = []
    real = ck.try_convex_solve

    def _spy(*args, **kwargs):
        calls.append("called")
        return real(*args, **kwargs)

    monkeypatch.setattr(ck, "try_convex_solve", _spy)
    _twobox("either_or").solve()
    assert calls, "control: the default solve never consulted the convex kernel"
    calls.clear()
    _twobox("either_or").solve(gdp_method="hull")
    assert calls == []


def _small_qp():
    m = dm.Model("qp")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    m.minimize((x - 1) ** 2 + (y - 2) ** 2)
    m.subject_to(x + y <= 2)
    return m, x, y


def _mpc_qp(n=30):
    # A linear-MPC chain QP in the shape of the issue's I.14b repro: a start from
    # the previous horizon cuts the qp-ipm iteration count.
    m = dm.Model("mpc")
    x0 = m.parameter("x0", 3.0)
    x = [m.continuous(f"x[{k}]", lb=-50, ub=50) for k in range(n + 1)]
    u = [m.continuous(f"u[{k}]", lb=-2, ub=2) for k in range(n)]
    m.subject_to(x[0] == x0)
    for k in range(n):
        m.subject_to(x[k + 1] == 0.95 * x[k] + 0.1 * u[k])
    m.minimize(sum(x[k] ** 2 for k in range(1, n + 1)) + 0.1 * sum(v**2 for v in u))
    return m, x0, x + u


def _pounce_starts(m, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = m.solve(solver="pounce", **kw)
    assert r.algorithm_route == "pounce:qp-ipm"
    return r, [str(w.message) for w in rec if "ignores" in str(w.message)]


def test_pounce_qp_initial_solution_is_used(monkeypatch):
    # B-01b: the start reaches POUNCE's qp-ipm as ``warm_start`` -- no warning,
    # same answer, fewer iterations.
    import pounce.qp as pq

    seen: list = []
    real = pq.solve_qp

    def _spy(*a, **kw):
        seen.append(kw.get("warm_start"))
        return real(*a, **kw)

    monkeypatch.setattr(pq, "solve_qp", _spy)
    m, x0, allv = _mpc_qp()
    r0, _ = _pounce_starts(m)
    x0.value = 3.1
    cold, msgs_cold = _pounce_starts(m)
    start = {v: r0.value(v) for v in allv}
    warm, msgs_warm = _pounce_starts(m, initial_solution=start)
    ws, msgs_ws = _pounce_starts(m, warm_start=r0)
    assert msgs_cold == msgs_warm == msgs_ws == []
    assert seen[:2] == [None, None]
    assert seen[2] is not None and seen[3] is not None
    assert warm.status == cold.status == ws.status == "optimal"
    assert warm.objective == pytest.approx(cold.objective, rel=1e-8)
    assert ws.objective == pytest.approx(cold.objective, rel=1e-8)
    it = [int(r.solver_stats["pounce/iterations"]) for r in (cold, warm, ws)]
    assert it[1] < it[0] and it[2] < it[0], it


def test_pounce_lp_initial_solution_still_warns():
    # lp-ipm remains a cold solve: the start is dropped, and declared.
    m = dm.Model("lp")
    x = m.continuous("x", lb=0, ub=10)
    m.maximize(3 * x)
    m.subject_to(x <= 4)
    with pytest.warns(UserWarning, match=r"ignores initial_solution .*lp-ipm"):
        r = m.solve(solver="pounce", initial_solution={x: 1.0})
    assert r.algorithm_route == "pounce:lp-ipm"


def test_pounce_qp_without_start_does_not_warn():
    m, _, _ = _small_qp()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m.solve(solver="pounce")
    assert not [w for w in rec if "initial_solution" in str(w.message)]


def test_skip_convex_check_docstring_states_local_solve():
    doc = dm.Model.solve.__doc__
    i = doc.index("skip_convex_check : bool")
    section = doc[i : i + 900]
    assert "local" in section and "no spatial branch-and-bound" in section


def _kernel_gap_solve(monkeypatch, **kw):
    monkeypatch.setenv("DISCOPT_NATIVE_SPATIAL_KERNEL", "1")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = _haverly().solve(time_limit=60, **kw)
    assert r.algorithm_route.startswith("native-spatial"), r.algorithm_route
    return r, [w for w in rec if "gap_tolerance" in str(w.message)]


def test_kernel_loose_relative_gap_is_honoured(monkeypatch):
    """D-12: a gap_tolerance looser than the default stops the kernel on the RELATIVE gap.

    It used to be applied absolutely, so 0.2 explored the same 233 nodes as 1e-4.
    """
    # Needs a tree. Since #1944 the default lineq RLT rows close Haverly's root.
    monkeypatch.setenv("DISCOPT_RLT_LINEQ", "0")
    tight, _ = _kernel_gap_solve(monkeypatch)
    loose, warned = _kernel_gap_solve(monkeypatch, gap_tolerance=0.2)
    assert warned == []
    assert loose.node_count < tight.node_count, (loose.node_count, tight.node_count)
    # Haverly maximizes: the bound sits above the incumbent, by at most 20%.
    assert loose.status == "optimal" and loose.gap_certified
    assert loose.bound is not None and loose.objective is not None
    assert loose.bound >= 400.0 - 1e-6  # still a valid bound on the optimum 400
    gap = (loose.bound - loose.objective) / max(abs(loose.bound), abs(loose.objective))
    assert 0.0 <= gap <= 0.2
    assert loose.solver_stats["gap_criterion"] == "relative"


def test_kernel_loose_gap_never_closes_without_a_bound(monkeypatch):
    """The lifted absolute arm must stay finite: ``inf`` closed at node 0 with no bound."""
    r, _ = _kernel_gap_solve(monkeypatch, gap_tolerance=0.5)
    assert r.bound is not None and r.node_count > 0


def test_kernel_default_gap_is_silent(monkeypatch):
    # #1871: with no ``abs_gap_tolerance`` named, the kernel stops on the documented
    # criterion (absolute 1e-6 OR relative 1e-4). Which arm the final gap lands in
    # is host-dependent (relative here, absolute on CI), so the label may be either;
    # what must hold is the documented criterion on the published pair.
    r, warned = _kernel_gap_solve(monkeypatch)
    assert warned == []
    assert r.status == "optimal" and r.gap_certified
    assert r.solver_stats["gap_criterion"] in ("absolute", "relative")
    gap = abs(r.bound - r.objective)
    assert gap <= 1e-6 or gap <= 1e-4 * max(abs(r.bound), abs(r.objective))


def test_kernel_absolute_arm_opt_out_is_silent(monkeypatch):
    # ``DISCOPT_KERNEL_DOCUMENTED_GAP=0`` restores the pre-#1871 absolute arm. The
    # #1871 cutoff box is pinned off too: it changes the search, and the label
    # records where the final gap lands (1.8e-8 with both off, 5.3e-5 with the box).
    monkeypatch.setenv("DISCOPT_KERNEL_DOCUMENTED_GAP", "0")
    monkeypatch.setenv("DISCOPT_KERNEL_CUTOFF_OBBT", "0")
    r, warned = _kernel_gap_solve(monkeypatch)
    assert warned == []
    assert r.status == "optimal" and r.solver_stats["gap_criterion"] == "absolute"


# --- B-01a: sensitivity/gradient off the JAX path ---------------------------


def _param_qp():
    # Bounds and a one-sided row that are active at the optimum, so the KKT system
    # exercises the at-bound and inactive-row eliminations as well as equalities.
    m = dm.Model("pqp")
    a = m.parameter("a", 1.5)
    b = m.parameter("b", 0.7)
    x = [m.continuous(f"x{i}", lb=-1.0, ub=2.0) for i in range(4)]
    m.subject_to(x[0] + x[1] + x[2] == a)
    m.subject_to(x[0] - x[3] <= b)
    m.subject_to(x[1] + x[3] >= -5.0)
    m.minimize(
        (x[0] - 3.0) ** 2 + (x[1] + a) ** 2 + 0.5 * x[2] ** 2 + (x[3] - b * a) ** 2 + x[0] * x[2]
    )
    return m, a, b


def test_gradient_uses_the_tape_evaluator(monkeypatch):
    # ``SolveResult.gradient()``'s envelope re-solve used to build a fresh JAX
    # ``NLPEvaluator`` (~9 s on the issue's 100-step MPC against a 0.2 s solve).
    import discopt._relax.nlp_evaluator as ne

    m, a, b = _param_qp()
    r = m.solve()
    expected = (float(r.gradient(a)), float(r.gradient(b)))

    def _refuse(*args, **kwargs):
        raise AssertionError("gradient() built a JAX NLPEvaluator")

    monkeypatch.setattr(ne, "NLPEvaluator", _refuse)
    r2 = m.solve()
    got = (float(r2.gradient(a)), float(r2.gradient(b)))
    assert got == pytest.approx(expected, rel=1e-6, abs=1e-8)
    assert all(abs(g) > 1e-3 for g in got), got


def test_order1_sensitivity_skips_the_jax_kkt_jacobian(monkeypatch):
    # Order-1 exact sensitivity assembles the KKT matrix from the tape evaluator;
    # only the p-derivative of the residual goes through JAX (forward mode). It
    # must agree with differentiating ``phi`` through ``custom_root``.
    import jax
    import jax.numpy as jnp
    import numpy as np
    from discopt.modeling.argmin import _build_layer
    from discopt.solvers.sipopt import pounce_sensitivity

    m, a, b = _param_qp()
    phi = _build_layer(m, [a, b], verify_minimizer=False, require_min=False, full=True)
    ref = np.asarray(jax.jacobian(phi)(jnp.asarray([1.5, 0.7])))
    n = phi.n_variables
    _, _, active, at_bound = phi.identify(np.array([1.5, 0.7]))
    # The instance must exercise every elimination, or the comparison is partial.
    assert at_bound.any() and active.any() and not active.all()

    calls = {"n": 0}
    real = jax.jacobian

    def _counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(jax, "jacobian", _counted)
    s = pounce_sensitivity(m, [a, b])
    assert calls["n"] == 0
    np.testing.assert_allclose(s.dx_dp, ref[:n], rtol=1e-9, atol=1e-11)
    np.testing.assert_allclose(s.dlambda_dp, ref[n:], rtol=1e-9, atol=1e-11)
    assert np.abs(s.dx_dp).max() > 1e-3


def test_pounce_cli_only_qp_option_names_the_cli_route():
    # A-17/A-24: a ``qp_*`` name pounce.qp.solve_qp does not take stays refused,
    # and the refusal says how to reach a CLI option. (``qp_crossover``,
    # ``qp_presolve``, ``qp_reg``, ``qp_hsde`` and ``qp_equilibrate`` are taken by
    # pounce-solver >= 0.13 and forwarded since #1985.)
    m = dm.Model("l")
    x = m.continuous("x", lb=0, ub=10)
    m.maximize(3 * x)
    m.subject_to(x <= 4)
    with pytest.raises(ValueError, match=r"\['qp_no_such_knob'\].*command line.*m\.to_nl"):
        m.solve(solver="pounce", pounce_options={"qp_no_such_knob": "yes"})
