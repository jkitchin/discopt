"""End-to-end tests for ``Model.solve(solver="direct")``.

Each benchmark here is present for a stated reason rather than for coverage; the
reason is in the test's docstring. Standard optima come from
``python/tests/support/direct_testfuncs.py``, which is also what the entry
experiment and the docs notebook use, so a number cannot drift between them.

Every objective is wrapped in ``dm.custom``. That is deliberate: an opaque body
is the class this backend exists for, and it is the path that today degrades to a
single local NLP (continuous) or raises outright (with integers).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import sys
from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from support import direct_testfuncs as tfs  # noqa: E402

_TIME_LIMIT = 120.0


def _solve(tf, **kwargs):
    model, _ = tfs.build_model(tf)
    kwargs.setdefault("time_limit", _TIME_LIMIT)
    return model.solve(solver="direct", **kwargs)


# ── the contract: no certificate, ever ───────────────────────────────────────


@pytest.mark.smoke
@pytest.mark.parametrize("name", ["branin", "sphere_2", "six_hump_camel"])
def test_never_claims_a_certificate(name):
    """DIRECT has no dual information, so it must report none.

    ``bound``/``gap`` are ``None``, ``gap_certified`` is ``False``, and the status
    is never ``"optimal"`` — even when the run lands exactly on the known optimum,
    which several of these do. Reporting the incumbent as a bound would be a false
    certificate (CLAUDE.md §1).
    """
    result = _solve(tfs.get(name), max_evals=400)
    assert result.status != "optimal", result.status
    assert result.bound is None
    assert result.gap is None
    assert result.gap_certified is False


@pytest.mark.smoke
def test_exhausted_budget_is_a_limit_not_infeasible():
    """A run that finds nothing reports a limit status, never ``"infeasible"``.

    DIRECT cannot prove infeasibility. Claiming it would be exactly the kind of
    false certificate the contract exists to prevent, so this pins the negative.

    The infeasible row is deliberately *opaque* -- wrapped in a ``custom`` call
    the bound-tightening pass cannot read. It used to be the plain algebraic
    ``x[0] + x[1] >= 10.0``, which since the declared-box check moved above the
    solver-family dispatch is proved infeasible before DIRECT ever starts. That
    proof is sound and belongs to presolve, not to DIRECT (see
    ``test_a_sound_presolve_proof_short_circuits_before_direct_runs``), but it
    means the algebraic form no longer exercises DIRECT's own exit -- the thing
    this test is about. Hiding the row behind a black box restores that.
    """
    model = dm.Model("no_feasible_point")
    x = model.continuous("x", shape=2, lb=0.0, ub=1.0)
    model.minimize(dm.custom(lambda v: v[0] + v[1], name="lin")(x))
    # A constraint no point in the box satisfies, opaque to bound tightening.
    model.subject_to(dm.custom(lambda v: v[0] + v[1], name="sumc")(x) >= 10.0)
    result = model.solve(solver="direct", max_evals=200, time_limit=_TIME_LIMIT)
    assert result.status != "infeasible", result.status
    assert result.status in ("iteration_limit", "time_limit"), result.status
    assert result.objective is None
    assert result.gap_certified is False


@pytest.mark.smoke
def test_a_sound_presolve_proof_short_circuits_before_direct_runs():
    """The other side: a *provable* infeasibility must not be spent on 200 evals.

    ``x in [0,1]^2`` with ``x[0] + x[1] >= 10`` is infeasible by inspection, and
    bound tightening says so
    (``separable_quadratic_upper_bound ... minimum separable quadratic activity
    exceeds the upper bound``). Since that check runs before the solver-family
    dispatch, ``solver="direct"`` now returns the proof instead of searching a
    box with nothing in it. This is not DIRECT claiming infeasibility -- the
    certificate is algebraic and holds whatever the black-box objective does --
    so ``gap_certified`` is legitimately ``True``.
    """
    model = dm.Model("provably_infeasible")
    x = model.continuous("x", shape=2, lb=0.0, ub=1.0)
    model.minimize(dm.custom(lambda v: v[0] + v[1], name="lin")(x))
    model.subject_to(x[0] + x[1] >= 10.0)
    result = model.solve(solver="direct", max_evals=200, time_limit=_TIME_LIMIT)
    assert result.status == "infeasible"
    assert result.gap_certified is True


@pytest.mark.smoke
def test_is_deterministic_across_runs():
    """DIRECT has no RNG; identical inputs must give byte-identical results."""
    tf = tfs.get("rastrigin_2")
    first = _solve(tf, max_evals=300, local_refine=False)
    second = _solve(tf, max_evals=300, local_refine=False)
    assert first.objective == second.objective
    np.testing.assert_array_equal(first.x["x"], second.x["x"])


# ── preconditions refuse loudly ──────────────────────────────────────────────


@pytest.mark.smoke
def test_non_finite_box_raises():
    """DIRECT is defined only on a finite box; a big-M substitute would be silent."""
    model = dm.Model("unbounded")
    x = model.continuous("x", lb=0.0, ub=np.inf)
    model.minimize(dm.custom(lambda v: v**2, name="sq")(x))
    with pytest.raises(ValueError, match="finite box"):
        model.solve(solver="direct", max_evals=50)


@pytest.mark.smoke
def test_missing_objective_raises():
    """Refused at both levels, with the modeling layer getting there first.

    ``Model.solve`` rejects an objectiveless model before dispatch, so that is
    what a user sees. ``solve_direct`` keeps its own guard for callers who reach
    it directly — DIRECT is a minimization method, not a feasibility search.
    """
    from discopt.solvers.direct import solve_direct

    model = dm.Model("no_objective")
    model.continuous("x", lb=0.0, ub=1.0)
    with pytest.raises(ValueError, match="No objective set"):
        model.solve(solver="direct", max_evals=50)
    with pytest.raises(ValueError, match="requires an objective"):
        solve_direct(model, max_evals=50)


@pytest.mark.smoke
def test_unknown_option_values_raise():
    tf = tfs.get("sphere_2")
    model, _ = tfs.build_model(tf)
    with pytest.raises(ValueError, match="direct_variant"):
        model.solve(solver="direct", direct_variant="sideways", max_evals=50)
    with pytest.raises(ValueError, match="local_refine_method"):
        model.solve(solver="direct", local_refine_method="magic", max_evals=50)


@pytest.mark.smoke
def test_unknown_selector_names_the_valid_ones():
    tf = tfs.get("sphere_2")
    model, _ = tfs.build_model(tf)
    with pytest.raises(ValueError, match="'direct'"):
        model.solve(solver="dirrect")


# ── the motivating case ──────────────────────────────────────────────────────


@pytest.mark.slow
def test_opaque_custom_body_beats_the_local_only_path():
    """The reason this backend exists.

    A non-MCBox ``dm.custom`` objective has no algebraic relaxation, so the
    default path is a single local NLP with no global search. On a multimodal
    objective that leaves real value on the table: measured in the entry
    experiment, arm A stalled at 15.06 on 2-D Ackley whose optimum is 0.
    """
    tf = tfs.get("ackley_2")
    default = tfs.build_model(tf)[0].solve(time_limit=_TIME_LIMIT)
    direct = _solve(tf, max_evals=1500)
    assert direct.objective <= default.objective + 1e-9, (
        f"DIRECT ({direct.objective}) must not lose to the local-only path ({default.objective})"
    )
    assert tf.relative_error(direct.objective) <= 1e-2


@pytest.mark.slow
def test_opaque_custom_body_with_integers_does_not_raise():
    """Today's default path raises here; ``solver="direct"`` must return an answer.

    Global branch-and-bound has no valid node relaxation for an opaque body, so
    the solver refuses when integers are also present (sound-or-refuse). DIRECT
    needs no relaxation, so the selector turns a refusal into a usable result.
    """
    import jax.numpy as jnp

    # The body must use a raw jnp intrinsic ON AN ARGUMENT to be genuinely
    # outside the MCBox scope. An arithmetic-only body (``(p-3)**2 + (q-2.5)**2``)
    # traces fine through MCBox and is solved GLOBALLY WITH A CERTIFICATE by the
    # reduced-space engine, integers included — for that model the default path is
    # strictly better than DIRECT, and this test would be asserting a fiction.
    def opaque(p, q):
        return jnp.cos(p * 1.7) + (q - 2.5) ** 2 + 0.05 * p

    def build():
        m = dm.Model("opaque_minlp")
        a = m.integer("a", lb=0, ub=6)
        b = m.continuous("b", lb=0.0, ub=6.0)
        m.minimize(dm.custom(opaque, name="opaque")(a, b))
        return m

    with pytest.raises(ValueError, match="OUTSIDE the sound reduced-space"):
        build().solve(time_limit=_TIME_LIMIT)  # the default path refuses

    result = build().solve(solver="direct", max_evals=1200, time_limit=_TIME_LIMIT)
    assert result.objective is not None
    assert result.gap_certified is False
    a_val = float(np.asarray(result.x["a"]).reshape(-1)[0])
    assert abs(a_val - round(a_val)) < 1e-9, f"integer variable is not integral: {a_val}"
    # Brute force over the 7 integer values, minimizing the continuous part exactly
    # (b = 2.5 whatever a is), gives the true optimum of this small MINLP.
    best = min(float(np.cos(k * 1.7) + 0.05 * k) for k in range(7))
    assert result.objective == pytest.approx(best, abs=1e-4)


# ── convergence on the standard functions ────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize(
    ("name", "budget", "tol"),
    [
        # Basic convergence, and the survey's own example of a *simple* problem
        # DIRECT is unimpressive on.
        ("sphere_2", 800, 1e-2),
        # Three global minima.
        ("branin", 800, 1e-3),
        # Two global minima.
        ("six_hump_camel", 800, 1e-3),
        # Sharp scaling.
        ("goldstein_price", 1200, 1e-2),
        # Moderate dimension.
        ("hartman_3", 1200, 1e-2),
    ],
)
def test_reaches_the_published_optimum(name, budget, tol):
    tf = tfs.get(name)
    result = _solve(tf, max_evals=budget)
    assert result.objective is not None
    assert tf.relative_error(result.objective) <= tol, (
        f"{name}: got {result.objective}, published optimum {tf.fstar}"
    )


@pytest.mark.slow
def test_goldstein_price_is_insensitive_to_affine_rescaling():
    """An additive/multiplicative rescaling of f must not change where we land.

    Selection compares ``f - K d`` across rectangles, so a scale-sensitive
    implementation quietly changes which rectangles are chosen. The optimizer's
    *location* is what must be invariant; the value scales with the transform.
    """
    tf = tfs.get("goldstein_price")
    plain = _solve(tf, max_evals=900, local_refine=False)

    scaled_model = dm.Model("gp_scaled")
    y = scaled_model.continuous("x", shape=tf.n, lb=tf.lb, ub=tf.ub)
    scaled_model.minimize(dm.custom(lambda v: 100.0 * tf.jnp_body(v) + 7.0, name="gp_scaled")(y))
    scaled = scaled_model.solve(
        solver="direct", max_evals=900, local_refine=False, time_limit=_TIME_LIMIT
    )
    np.testing.assert_allclose(plain.x["x"], scaled.x["x"], atol=1e-9)


# ── the measured design decisions ────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["classic", "gl"])
def test_both_variants_solve_a_multimodal_problem(variant):
    """Both selection rules are usable end to end and honour the contract.

    Which one is *faster* is a question about evaluations-to-accuracy, not about
    the objective at a fixed budget — at a generous budget both converge and the
    comparison degenerates into noise. That measurement lives in
    ``test_direct_units.py::test_variant_tradeoff_is_why_gl_is_not_the_default``,
    where the evaluation history is available.
    """
    tf = tfs.get("shubert")
    result = _solve(tf, max_evals=1200, direct_variant=variant)
    assert result.objective is not None
    assert result.gap_certified is False
    assert result.status != "optimal"


@pytest.mark.slow
def test_one_side_trisection_and_tie_breaking_cut_evaluations():
    """The survey's two endorsed revisions, measured on its own drag function.

    Fig. 15 minimizes ``1 + x1 + ... + x5`` and reports evaluations to 1%
    accuracy: 14,492 with the original rules, 470 breaking ties, 192 also
    trisecting one side. Reproduced here through the public API at a coarser
    budget: the ordering must hold.
    """
    tf = tfs.get("linear_5")
    original = _solve(tf, max_evals=4000, divide="all", break_ties=False, local_refine=False)
    revised = _solve(tf, max_evals=4000, divide="one", break_ties=True, local_refine=False)
    assert revised.objective < original.objective, (
        f"revised rules {revised.objective} should beat the original "
        f"{original.objective} at an equal budget"
    )


@pytest.mark.slow
def test_local_refinement_improves_the_answer():
    """The survey's most-endorsed change: DIRECT finds the basin, the local solve refines.

    Shubert is the case the survey uses to make the point (995 evaluations with a
    local optimizer versus 2967 without).
    """
    tf = tfs.get("shubert")
    without = _solve(tf, max_evals=1200, local_refine=False)
    with_refine = _solve(tf, max_evals=1200, local_refine=True)
    assert with_refine.objective <= without.objective + 1e-9


@pytest.mark.slow
def test_refinement_never_loses_to_the_supplied_start():
    """Best-of-both-starts: passing a good point can only help.

    Regression for the one genuine regression the entry experiment found — on
    griewank_3 DIRECT's incumbent sat in a worse basin than the default start, and
    refining from DIRECT's point alone lost to the local-only path.
    """
    tf = tfs.get("rastrigin_2")
    model, x = tfs.build_model(tf)
    optimum = np.zeros(tf.n)
    result = model.solve(
        solver="direct",
        max_evals=300,
        time_limit=_TIME_LIMIT,
        initial_solution={x: optimum},
    )
    assert result.objective == pytest.approx(0.0, abs=1e-6), (
        "starting at the global optimum must not be lost by the search"
    )


# ── constraints and stats ────────────────────────────────────────────────────


@pytest.mark.slow
def test_constrained_model_returns_a_feasible_point():
    """DIRECT-GLce: phase A finds feasibility, phase B optimizes within it.

    The unconstrained minimum of this objective is the origin, which the
    constraint excludes, so a solver that ignored the constraint would report 0.
    """
    model = dm.Model("glce")
    x = model.continuous("x", shape=2, lb=-5.0, ub=5.0)
    model.minimize(dm.custom(lambda v: v[0] ** 2 + v[1] ** 2, name="quad")(x))
    model.subject_to(x[0] + x[1] >= 2.0)
    result = model.solve(solver="direct", max_evals=1500, time_limit=_TIME_LIMIT)
    assert result.objective == pytest.approx(2.0, rel=1e-3), result.objective
    xs = np.asarray(result.x["x"]).reshape(-1)
    assert xs[0] + xs[1] >= 2.0 - 1e-4, f"returned point is infeasible: {xs}"


@pytest.mark.smoke
def test_solver_stats_report_the_evaluation_budget():
    """The evaluation count is the cost model, so it must be visible."""
    result = _solve(tfs.get("sphere_2"), max_evals=250)
    stats = result.solver_stats or {}
    assert stats.get("direct/evals", 0) > 0
    assert stats["direct/evals"] <= 250 + 1
    for key in ("direct/rectangles", "direct/iterations", "direct/local_solves"):
        assert key in stats, f"missing {key}: {sorted(stats)}"


@pytest.mark.smoke
def test_generic_options_it_cannot_honour_are_warned_about():
    """Silently accepting ``gap_tolerance`` would leave a wrong mental model."""
    tf = tfs.get("sphere_2")
    model, _ = tfs.build_model(tf)
    with pytest.warns(UserWarning, match="no dual bound"):
        model.solve(solver="direct", max_evals=100, gap_tolerance=1e-9, time_limit=_TIME_LIMIT)


# ── the default path is untouched ────────────────────────────────────────────


@pytest.mark.smoke
def test_default_path_still_certifies_an_algebraic_model():
    """The new selector must be unreachable from the default path.

    A model that can be written algebraically still goes through spatial B&B and
    still gets a certificate; adding a backend must not perturb that.
    """
    model = dm.Model("algebraic")
    x = model.continuous("x", lb=-2.0, ub=2.0)
    model.minimize((x - 0.5) ** 2)
    result = model.solve(time_limit=60.0)
    assert result.objective == pytest.approx(0.0, abs=1e-6)
    assert result.gap_certified is True, "the certified default path regressed"


@pytest.mark.slow
def test_n_jobs_changes_the_wall_clock_and_nothing_else():
    """Parallel evaluation through the public API returns the identical answer.

    Within an iteration DIRECT's sample points are independent, so they are
    evaluated together; ``n_jobs`` only decides how many threads do it. The
    result must be bit-identical, not merely close — verified here through
    ``Model.solve`` as well as at the engine level in ``test_direct_units.py``.
    """
    tf = tfs.get("hartman_3")
    runs = {}
    for n_jobs in (1, 4):
        result = _solve(tf, max_evals=900, n_jobs=n_jobs)
        runs[n_jobs] = (result.objective, np.asarray(result.x["x"]).tolist())
        stats = result.solver_stats or {}
        assert stats.get("direct/batches", 0) > 0
    assert runs[1] == runs[4], runs


@pytest.mark.smoke
def test_invalid_n_jobs_raises():
    tf = tfs.get("sphere_2")
    model, _ = tfs.build_model(tf)
    with pytest.raises(ValueError, match="n_jobs"):
        model.solve(solver="direct", n_jobs=0, max_evals=50)
