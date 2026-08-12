"""End-to-end tests for :func:`discopt.solvers.surrogate.solve_surrogate`.

Each benchmark here is present for a stated reason rather than for coverage; the
reason is in the test's docstring. Standard optima come from
``python/tests/support/direct_testfuncs.py`` — the same definitions the DIRECT
suite and the docs notebook use, so a number cannot drift between them, and so
the head-to-head evaluation-count comparison below is against the identical
function on the identical box.

Every objective is wrapped in ``dm.custom``. That is deliberate: an opaque body
is the class this backend exists for, and it is the path that today degrades to a
single local NLP (continuous) or raises outright (with integers).

``solve_surrogate`` is called directly rather than through ``Model.solve``: the
selector dispatch is not wired yet, by design of the phase split.
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
from discopt.solvers.surrogate import solve_surrogate

sys.path.insert(0, str(Path(__file__).parent))
from support import direct_testfuncs as tfs  # noqa: E402

_TIME_LIMIT = 600.0

#: The acquisition optimizer used wherever a test is about the *search* rather
#: than about the acquisition optimizer itself. Multistart is the fast path and
#: maximizes exactly the same acquisition function, so the search behaviour it
#: exercises is the same; the algebraic/B&B path has its own dedicated tests
#: below, where its cost is the point rather than an obstacle.
_FAST = {"acquisition_optimizer": "multistart"}


def _solve(tf, **kwargs):
    model, _ = tfs.build_model(tf)
    kwargs.setdefault("time_limit", _TIME_LIMIT)
    return solve_surrogate(model, **kwargs)


# ── the contract: no certificate, ever ───────────────────────────────────────


@pytest.mark.smoke
@pytest.mark.parametrize("name", ["branin", "sphere_2", "six_hump_camel"])
def test_never_claims_a_certificate(name):
    """A surrogate is a model of the objective, not a bound on it.

    An interpolant says nothing rigorous about the function between its data
    points, so ``bound``/``gap`` are ``None``, ``gap_certified`` is ``False``, and
    the status is never ``"optimal"`` — even when the run lands on the known
    optimum, which several of these do. Reporting the incumbent as a bound would
    be a false certificate (CLAUDE.md §1).
    """
    result = _solve(tfs.get(name), max_evals=25, **_FAST)
    assert result.status != "optimal", result.status
    assert result.bound is None
    assert result.gap is None
    assert result.gap_certified is False
    assert result.objective is not None


@pytest.mark.smoke
def test_exhausted_budget_is_a_limit_not_infeasible():
    """A run that finds nothing feasible reports a limit, never ``"infeasible"``.

    A sampling method cannot prove infeasibility. Claiming it would be exactly the
    kind of false certificate the contract exists to prevent, so this pins the
    negative.
    """
    model = dm.Model("no_feasible_point")
    x = model.continuous("x", shape=2, lb=0.0, ub=1.0)
    model.minimize(dm.custom(lambda v: v[0] + v[1], name="lin")(x))
    model.subject_to(x[0] + x[1] >= 10.0)  # no point in the box satisfies this
    result = solve_surrogate(model, max_evals=20, time_limit=_TIME_LIMIT, **_FAST)
    assert result.status != "infeasible", result.status
    assert result.status in ("iteration_limit", "time_limit"), result.status
    assert result.objective is None
    assert result.bound is None
    assert result.gap_certified is False


@pytest.mark.smoke
def test_is_deterministic_across_runs():
    """Same seed, same answer — the designs and multistart pools are all seeded.

    A surrogate method is full of randomness that is easy to leave unseeded (the
    initial design, the multistart pool, the maximin candidate pool). This is the
    only cheap way to notice that one of them escaped.
    """
    tf = tfs.get("rastrigin_2")
    first = _solve(tf, max_evals=22, seed=7, **_FAST)
    second = _solve(tf, max_evals=22, seed=7, **_FAST)
    assert first.objective == second.objective
    np.testing.assert_array_equal(first.x["x"], second.x["x"])


@pytest.mark.smoke
def test_a_different_seed_gives_a_different_search():
    """The counterpart: the seed must actually reach the randomness.

    Without this, a bug that hard-coded one RNG state would pass the determinism
    test above perfectly.
    """
    tf = tfs.get("rastrigin_2")
    a = _solve(tf, max_evals=22, seed=1, **_FAST)
    b = _solve(tf, max_evals=22, seed=2, **_FAST)
    assert not np.array_equal(a.x["x"], b.x["x"])


# ── preconditions refuse loudly ──────────────────────────────────────────────


@pytest.mark.smoke
def test_non_finite_box_raises():
    """The surrogate is fitted on normalized distances; an infinite side is NaN.

    Substituting a big-M box would be a silent approximation, and it is the caller
    who knows the real range (CLAUDE.md §3).
    """
    model = dm.Model("unbounded")
    x = model.continuous("x", lb=0.0, ub=np.inf)
    model.minimize(dm.custom(lambda v: v**2, name="sq")(x))
    with pytest.raises(ValueError, match="finite box"):
        solve_surrogate(model, max_evals=10)


@pytest.mark.smoke
def test_missing_objective_raises():
    """A surrogate method is a minimization method, not a feasibility search."""
    model = dm.Model("no_objective")
    model.continuous("x", lb=0.0, ub=1.0)
    with pytest.raises(ValueError, match="requires an objective"):
        solve_surrogate(model, max_evals=10)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"surrogate": "gaussian"}, "surrogate"),
        ({"rbf_kernel": "gaussian"}, "rbf_kernel"),
        ({"acquisition": "ucb"}, "acquisition"),
        ({"acquisition_optimizer": "magic"}, "acquisition_optimizer"),
        ({"max_evals": 0}, "max_evals"),
        ({"min_distance": 0.0}, "min_distance"),
        ({"acquisition": "ei"}, "requires surrogate='kriging'"),
        ({"acquisition": "cors", "surrogate": "kriging"}, "requires surrogate='rbf'"),
    ],
)
def test_unknown_or_incoherent_option_values_raise(kwargs, match):
    """Every option is validated, and an incoherent *combination* is rejected too.

    ``acquisition="ei"`` with an RBF is the one worth spelling out: an RBF
    interpolant has no predictive variance, so there is no distribution to take an
    expectation over. Quietly falling back to CORS would leave the caller with a
    wrong mental model of what ran.
    """
    model, _ = tfs.build_model(tfs.get("sphere_2"))
    with pytest.raises(ValueError, match=match):
        solve_surrogate(model, **kwargs)


# ── the acquisition: certified vs not, measured rather than claimed ──────────


@pytest.mark.slow
def test_the_acquisition_subproblem_certifies_while_the_outer_result_does_not():
    """The distinction the whole design rests on, asserted on both halves at once.

    The acquisition subproblem is an ordinary algebraic model — a sum of
    ``λ_i φ(q_i)`` over squared distances, plus one reversed convex quadratic per
    design point — and discopt's own spatial branch-and-bound solves it to
    **certified** global optimality. That is the thing Jones et al. built a
    bespoke branch-and-bound for in 1998 and that modern BO libraries approximate
    with multistart gradient ascent.

    It is emphatically *not* a certificate for the answer: the outer result still
    reports ``bound=None`` and ``gap_certified=False``, because certifying where
    to sample next says nothing about the black box between samples.
    """
    from discopt._relax.primal_heuristics import _generate_starts
    from discopt.solvers.surrogate import RBFSurrogate, _SurrogateSearch, build_cors_model

    tf = tfs.get("branin")
    model, _ = tfs.build_model(tf)

    # Half 1: the subproblem itself, built from a genuinely fitted surrogate.
    rng = np.random.default_rng(0)
    X = _generate_starts(tf.lb, tf.ub, 8, rng)
    y = np.array([float(tf.np_body(x)) for x in X])
    search = _SurrogateSearch(tf.lb, tf.ub)
    rbf = RBFSurrogate(kernel="linear").fit(search.normalize(X), y)
    acq = build_cors_model(model, tf.lb, tf.ub, rbf, delta=0.1)
    sub = acq.solve(time_limit=120.0)
    assert sub.gap_certified is True, (sub.status, sub.objective, sub.bound)
    assert sub.status == "optimal", sub.status
    point = np.asarray(sub.x["x"]).reshape(-1)
    gaps = np.linalg.norm(search.normalize(X) - search.normalize(point[None, :]), axis=1)
    assert gaps.min() >= 0.1 - 1e-6, (
        f"the certified point violates its own exclusion radius: {gaps.min()}"
    )

    # Half 2: the outer solve, whose acquisition ran on that same certified path.
    result = _solve(
        tf,
        max_evals=10,
        n_initial=6,
        rbf_kernel="linear",
        acquisition_optimizer="auto",
        acquisition_time_limit=120.0,
    )
    stats = result.solver_stats or {}
    assert stats.get("surrogate/acq_certified", 0) >= 1, stats
    assert stats.get("surrogate/acq_multistart", 0) == 0, stats
    assert result.bound is None
    assert result.gap is None
    assert result.gap_certified is False
    assert result.status != "optimal"


@pytest.mark.smoke
def test_certified_optimizer_refuses_rather_than_degrading_quietly():
    """``acquisition_optimizer="certified"`` must raise, not fall back.

    The failure it is pointed at is real and specific: with a correlation power
    other than 2 the EI subproblem cannot be written algebraically at all
    (``|z|^p`` has no discopt intrinsic), and approximating it would mean
    certifying a *different* model from the one that was fitted. Under
    ``"auto"`` that is a logged fallback; under ``"certified"`` it is an error.
    """
    tf = tfs.get("sphere_2")
    model, _ = tfs.build_model(tf)
    with pytest.raises(Exception, match="power|intrinsic"):
        solve_surrogate(
            model,
            max_evals=10,
            n_initial=6,
            surrogate="kriging",
            kriging_power=1.5,
            acquisition_optimizer="certified",
            acquisition_time_limit=10.0,
            time_limit=_TIME_LIMIT,
        )


@pytest.mark.slow
def test_a_non_expressible_acquisition_falls_back_and_says_so(caplog):
    """Under ``"auto"`` the same case degrades — but audibly, and it is counted.

    A fallback that leaves no trace is how "the certified path ran" becomes an
    unfalsifiable claim (CLAUDE.md §6). The counter and the log line are the
    evidence.
    """
    tf = tfs.get("sphere_2")
    model, _ = tfs.build_model(tf)
    with caplog.at_level("INFO", logger="discopt.solvers.surrogate"):
        result = solve_surrogate(
            model,
            max_evals=9,
            n_initial=6,
            surrogate="kriging",
            kriging_power=1.5,
            acquisition_optimizer="auto",
            time_limit=_TIME_LIMIT,
        )
    stats = result.solver_stats or {}
    assert stats.get("surrogate/acq_failures", 0) >= 1, stats
    assert stats.get("surrogate/acq_multistart", 0) >= 1, stats
    assert stats.get("surrogate/acq_certified", 0) == 0, stats
    assert any("not expressible" in rec.message for rec in caplog.records), caplog.text


@pytest.mark.smoke
def test_solver_stats_report_which_acquisition_optimizer_ran():
    """The counters are how the design's central claim stays a measurement."""
    result = _solve(tfs.get("sphere_2"), max_evals=18, **_FAST)
    stats = result.solver_stats or {}
    for key in (
        "surrogate/evals",
        "surrogate/fits",
        "surrogate/acq_certified",
        "surrogate/acq_bb_uncertified",
        "surrogate/acq_multistart",
        "surrogate/acq_failures",
        "surrogate/initial_design",
    ):
        assert key in stats, f"missing {key}: {sorted(stats)}"
    assert stats["surrogate/evals"] <= 18
    assert stats["surrogate/acq_multistart"] >= 1
    assert stats["surrogate/acq_certified"] == 0, "multistart was requested"


# ── sample efficiency: the reason this backend exists ────────────────────────


def _direct_evals_to_tolerance(tf, tol: float, budget: int = 4000) -> int | None:
    """Evaluations DIRECT needs before its incumbent first reaches ``tol``.

    Imported read-only from the DIRECT backend so the comparison is against the
    real algorithm on the real box, not against a remembered number.
    """
    from discopt.solvers.direct import _DirectSearch

    search = _DirectSearch(tf.lb, tf.ub)
    history: list[tuple[int, float]] = []
    search.run(
        lambda x: (float(tf.np_body(x)), 0.0),
        budget,
        on_iteration=lambda s: history.append((s.stats.evals, s.best_feasible_value)),
    )
    assert history, "the DIRECT probe recorded no iterations — it measured nothing"
    return next((e for e, v in history if v is not None and tf.relative_error(v) <= tol), None)


def _surrogate_evals_to_tolerance(tf, tol: float, budget: int, **kwargs) -> int | None:
    """Evaluations this backend needs before its incumbent first reaches ``tol``.

    Read off the public ``on_evaluation`` hook in a single run, rather than by
    re-running a ladder of budgets. The hook firing is asserted: a probe whose
    callback never ran would otherwise report "never reached the tolerance" and
    read as a genuine measurement (CLAUDE.md §6).
    """
    trace: list[tuple[int, float | None]] = []
    model, _ = tfs.build_model(tf)
    kwargs.setdefault("acquisition_optimizer", "multistart")
    solve_surrogate(
        model,
        max_evals=budget,
        time_limit=_TIME_LIMIT,
        on_evaluation=lambda n, v: trace.append((n, v)),
        **kwargs,
    )
    assert trace, "the on_evaluation hook never fired — this probe measured nothing"
    return next((n for n, v in trace if v is not None and tf.relative_error(v) <= tol), None)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("name", "tol", "budget"),
    [
        ("branin", 1e-2, 45),  # three global minima
        ("six_hump_camel", 1e-2, 45),  # two global minima
        ("hartman_3", 1e-2, 60),  # moderate dimension
        ("goldstein_price", 1e-2, 60),  # sharply scaled
    ],
)
def test_reaches_the_published_optimum_within_a_small_budget(name, tol, budget):
    """Convergence on the standard functions, at budgets a costly objective allows.

    The budgets are the point: ``solve_direct``'s own suite uses 800-1200
    evaluations for the same functions and tolerances. These are 45-60.
    """
    tf = tfs.get(name)
    result = _solve(tf, max_evals=budget, **_FAST)
    assert result.objective is not None
    assert tf.relative_error(result.objective) <= tol, (
        f"{name}: got {result.objective}, published optimum {tf.fstar}, budget {budget} evaluations"
    )


@pytest.mark.slow
@pytest.mark.parametrize("name", ["branin", "six_hump_camel", "goldstein_price"])
def test_uses_fewer_evaluations_than_direct_for_the_same_accuracy(name):
    """The claim this backend is for, measured in evaluations rather than asserted.

    DIRECT spends its budget on geometry and is happy to take hundreds of samples;
    this backend spends real computation between samples so that each one counts.
    On a genuinely expensive objective that difference is the whole value
    proposition, so it is measured head to head — same function, same box, same
    accuracy target, both engines driven from the same ``direct_testfuncs``
    definition.

    **Measured, evaluations to 1e-2 relative error** (median of seeds 0-2 for the
    surrogate; DIRECT is deterministic), with the published CORS cycle and a
    60-evaluation budget:

    ================  =========  ======  ======
    function          surrogate  DIRECT  factor
    ================  =========  ======  ======
    branin            38         69      1.8x
    six_hump_camel    32         137     4.3x
    ================  =========  ======  ======

    Two honest caveats, because "far fewer" is not uniformly true:

    * the margin on smooth 2-D functions is a factor of ~2, not the order of
      magnitude a surrogate wins by in the literature's harder settings — DIRECT
      is a strong baseline exactly here, and saying otherwise would be inventing a
      result;
    * DIRECT's engine is used *without* its local-refinement hybrid, since that
      spends uncounted evaluations through a different path; that is the same
      reason ``local_refine`` defaults off in this backend.

    The threshold below (1.5x, on the median over three seeds) is deliberately
    well under what is observed: seed-to-seed spread on branin is 30-42
    evaluations, and this must not be a flaky test. The table, not the threshold,
    is the record.
    """
    tf = tfs.get(name)
    tol = 1e-2
    direct_evals = _direct_evals_to_tolerance(tf, tol)
    assert direct_evals is not None, f"DIRECT never reached {tol} on {name}"

    hits = [_surrogate_evals_to_tolerance(tf, tol, budget=60, seed=s) for s in (0, 1, 2)]
    reached = [h for h in hits if h is not None]
    assert len(reached) >= 2, (
        f"{name}: the surrogate backend reached {tol} on only {len(reached)}/3 seeds "
        f"within 60 evaluations ({hits})"
    )
    median = float(np.median(reached))
    assert median * 1.5 <= direct_evals, (
        f"{name}: surrogate median {median} evaluations vs DIRECT {direct_evals} "
        f"(seeds {hits}) — the sample-efficiency advantage has regressed"
    )


# ── the motivating cases ─────────────────────────────────────────────────────


@pytest.mark.slow
def test_opaque_custom_body_beats_the_local_only_path():
    """The reason this backend exists at all.

    A non-MCBox ``dm.custom`` objective has no algebraic relaxation, so the
    default path is a single local NLP with no global search. On a multimodal
    objective that leaves real value on the table — and here it is recovered in a
    few dozen evaluations rather than a few thousand.
    """
    tf = tfs.get("ackley_2")
    default = tfs.build_model(tf)[0].solve(time_limit=120.0)
    got = _solve(tf, max_evals=60, **_FAST)
    assert got.objective <= default.objective + 1e-9, (
        f"surrogate ({got.objective}) must not lose to the local-only path ({default.objective})"
    )


@pytest.mark.slow
def test_opaque_custom_body_with_integers_returns_an_integral_answer():
    """Today's default path raises here; this backend must return an answer.

    Global branch-and-bound has no valid node relaxation for an opaque body, so
    the solver refuses when integers are also present (sound-or-refuse). A
    surrogate needs no relaxation — and the RBF handles integers natively, which
    is the first reason it is the default family here rather than a GP.
    """
    import jax.numpy as jnp

    # A raw jnp intrinsic ON AN ARGUMENT is what puts the body genuinely outside
    # the MCBox scope; an arithmetic-only body traces fine and is solved globally
    # WITH a certificate by the reduced-space engine, which would make this test
    # assert a fiction.
    def opaque(p, q):
        return jnp.cos(p * 1.7) + (q - 2.5) ** 2 + 0.05 * p

    def build():
        m = dm.Model("opaque_minlp")
        a = m.integer("a", lb=0, ub=6)
        b = m.continuous("b", lb=0.0, ub=6.0)
        m.minimize(dm.custom(opaque, name="opaque")(a, b))
        return m

    with pytest.raises(ValueError, match="OUTSIDE the sound reduced-space"):
        build().solve(time_limit=120.0)  # the default path refuses

    result = solve_surrogate(build(), max_evals=45, time_limit=_TIME_LIMIT, **_FAST)
    assert result.objective is not None
    assert result.gap_certified is False
    a_val = float(np.asarray(result.x["a"]).reshape(-1)[0])
    assert abs(a_val - round(a_val)) < 1e-9, f"integer variable is not integral: {a_val}"
    # Brute force over the 7 integer values, minimizing the continuous part exactly
    # (b = 2.5 whatever a is), gives the true optimum of this small MINLP.
    best = min(float(np.cos(k * 1.7) + 0.05 * k) for k in range(7))
    assert result.objective == pytest.approx(best, abs=1e-3)


@pytest.mark.slow
def test_constrained_model_returns_a_feasible_point():
    """Phase A finds feasibility, phase B optimizes within it.

    The unconstrained minimum of this objective is the origin, which the
    constraint excludes, so a run that ignored the constraint would report 0. The
    surrogate is fitted to the GLce merit, which is what carries the constraint
    into a method that only ever sees one scalar.
    """
    model = dm.Model("constrained")
    x = model.continuous("x", shape=2, lb=-5.0, ub=5.0)
    model.minimize(dm.custom(lambda v: v[0] ** 2 + v[1] ** 2, name="quad")(x))
    model.subject_to(x[0] + x[1] >= 2.0)
    result = solve_surrogate(model, max_evals=80, time_limit=_TIME_LIMIT, **_FAST)
    assert result.objective is not None
    xs = np.asarray(result.x["x"]).reshape(-1)
    assert xs[0] + xs[1] >= 2.0 - 1e-4, f"returned point is infeasible: {xs}"
    assert result.objective == pytest.approx(2.0, rel=5e-2), result.objective


# ── the second surrogate family ──────────────────────────────────────────────


@pytest.mark.slow
def test_kriging_with_expected_improvement_runs_end_to_end():
    """The EGO path, on the low-dimensional smooth problem it is meant for.

    Kriging + EI is the alternative rather than the default (see the module
    docstring), but it has to work: the whole reason to carry a second family is
    the smooth, very expensive, low-dimensional case where a GP's uncertainty
    model earns its fitting cost.
    """
    tf = tfs.get("branin")
    result = _solve(tf, max_evals=35, surrogate="kriging", **_FAST)
    assert result.objective is not None
    assert result.gap_certified is False
    assert tf.relative_error(result.objective) <= 5e-2, result.objective


@pytest.mark.slow
def test_a_noisy_objective_is_not_forced_through_its_own_measurement_error():
    """The nugget, end to end: EGO's most dated assumption, fixed.

    With deterministic noise added to a smooth function, an interpolating kriging
    model fits the noise and reports zero uncertainty at points whose value it
    does not actually know. A run with a nugget must still find the underlying
    optimum. The noise is a fixed function of ``x`` rather than an RNG draw, so
    the objective stays deterministic and the test cannot flake on resampling.
    """
    tf = tfs.get("sphere_2")
    model = dm.Model("noisy_sphere")
    x = model.continuous("x", shape=2, lb=tf.lb, ub=tf.ub)

    def noisy(v):
        import jax.numpy as jnp

        return jnp.sum(v**2) + 0.5 * jnp.sin(97.0 * v[0]) * jnp.cos(89.0 * v[1])

    model.minimize(dm.custom(noisy, name="noisy")(x))
    result = solve_surrogate(
        model,
        max_evals=45,
        surrogate="kriging",
        nugget=1e-2,
        time_limit=_TIME_LIMIT,
        **_FAST,
    )
    assert result.objective is not None
    # The noise floor is 0.5 in amplitude, so the underlying optimum of 0 can only
    # be located to about that; the assertion is that the search is in the right
    # basin, not that it beat the noise.
    xs = np.asarray(result.x["x"]).reshape(-1)
    assert float(np.linalg.norm(xs)) < 1.5, xs


# ── the RBF kernel choice, measured ──────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("kernel", ["cubic", "thin_plate", "linear"])
def test_every_kernel_converges_on_a_standard_function(kernel):
    """All three kernels have to be usable, not just the default.

    ``linear`` matters disproportionately: it is the kernel whose acquisition
    subproblem discopt certifies reliably (module docstring), so it must not be
    a second-class citizen in the search itself.
    """
    tf = tfs.get("branin")
    result = _solve(tf, max_evals=50, rbf_kernel=kernel, **_FAST)
    assert result.objective is not None
    assert tf.relative_error(result.objective) <= 5e-2, (kernel, result.objective)
