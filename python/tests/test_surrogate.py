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

#: Seeds the convergence panel runs. The panel asserts a *population* statistic
#: rather than one trajectory — see
#: ``test_reaches_the_published_optimum_within_a_small_budget`` for why a
#: single-seed pass/fail was the wrong shape for a chaotic deterministic search.
_PANEL_SEEDS = 5


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
        ({"acquisition_optimizer": "magic"}, "acquisition_optimizer"),
        ({"max_evals": 0}, "max_evals"),
        ({"n_initial": 0}, "n_initial"),
        ({"min_distance": 0.0}, "min_distance"),
    ],
)
def test_unknown_option_values_raise(kwargs, match):
    """Every option is validated before any evaluation is spent.

    ``min_distance=0`` is the one worth spelling out: it is not a harmless
    setting, it removes the guarantee that the acquisition cannot re-propose an
    already-sampled point, and a method that re-proposes one stalls silently.
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


# ── the budget is a stopping point, not a parameter of the search ────────────


@pytest.mark.smoke
@pytest.mark.parametrize("n_vars", [1, 2, 3, 6, 15])
def test_the_default_design_size_depends_on_the_dimension_alone(n_vars):
    """The signature is the invariant: ``max_evals`` is not an argument.

    Pinned as a *type* statement rather than only as a behavioural one because
    the regression it guards was a one-line formula that read perfectly
    reasonably (``min(10n, max_evals // 2)``) and quietly made the budget a
    parameter of the search. Also pinned: the result is at least the ``n+1`` an
    RBF with a linear tail needs, so the default can never produce a design the
    default surrogate cannot be fitted from.
    """
    import inspect

    from discopt.solvers.surrogate import _default_design_size

    assert list(inspect.signature(_default_design_size).parameters) == ["n_vars"]
    size = _default_design_size(n_vars)
    assert size >= n_vars + 1, (n_vars, size)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["branin", "hartman_3", "hartman_6"])
def test_a_larger_budget_continues_the_same_search(name):
    """A bigger ``max_evals`` extends the search; it does not start a different one.

    This is the property every "first reached the tolerance at evaluation ``k``,
    so budget ``B > k`` has headroom" statement in this file rests on, and it was
    silently false (issue #1036): the initial design was sized
    ``max(n+2, min(10n, max_evals // 2))``, so raising the budget raised the
    design and re-rolled the whole trajectory. Measured before the fix, on these
    same three functions and the same five budgets, **17 of 30 budget pairs
    diverged at evaluation 1**; ``branin`` was the only one that nested, and only
    because ``10n = 20`` happened to be under ``max_evals // 2`` for every budget
    tried — which is precisely the kind of accident that makes a panel look
    healthy while its rationale is unsound.

    The incumbent traces are compared elementwise rather than only the final
    objectives: a monotone-improvement check would pass on two entirely different
    searches that happen to end in the same basin, and pass most loudly on the
    default seed, which is where the original claim came from.

    ``hartman_6`` is in the list because at ``n = 6`` the old rule's two branches
    (``10n = 60`` and ``max_evals // 2``) crossed *inside* the budget range, so it
    is the case where the coupling bit hardest — 10 of its 10 pairs diverged.
    """
    tf = tfs.get(name)
    budgets = [40, 46, 60, 80, 100]
    traces: dict[int, list[float | None]] = {}
    for budget in budgets:
        trace: list[float | None] = []
        model, _ = tfs.build_model(tf)
        solve_surrogate(
            model,
            max_evals=budget,
            time_limit=_TIME_LIMIT,
            seed=0,
            on_evaluation=lambda _n, v: trace.append(v),
            **_FAST,
        )
        assert trace, f"{name}: the on_evaluation hook never fired at budget {budget}"
        traces[budget] = trace

    compared = 0
    for i, small in enumerate(budgets):
        for large in budgets[i + 1 :]:
            assert len(traces[large]) >= len(traces[small]), (
                f"{name}: budget {large} spent fewer evaluations than budget {small}"
            )
            prefix = traces[large][: len(traces[small])]
            compared += 1
            assert prefix == traces[small], (
                f"{name}: the budget-{large} run is not the budget-{small} run continued — "
                f"they diverge at evaluation "
                f"{next(k for k, (a, b) in enumerate(zip(traces[small], prefix), 1) if a != b)}"
            )
    assert compared == len(budgets) * (len(budgets) - 1) // 2, (
        f"{name}: compared {compared} budget pairs — this test measured nothing"
    )


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
    ("name", "tol", "budget", "quorum"),
    [
        # budget = the k-of-k crossing measured below, with ~50% headroom.
        ("branin", 1e-2, 65, 4),  # three global minima; 8/8 by 42
        ("six_hump_camel", 1e-2, 55, 4),  # two global minima; 8/8 by 34
        ("ackley_2", 1e-2, 100, 4),  # heavily multimodal; 8/8 by 64
        ("hartman_3", 1e-2, 90, 4),  # moderate dimension; 8/8 by 60
        ("goldstein_price", 1e-2, 225, 4),  # sharply scaled: see below; 8/8 by 150
    ],
)
def test_reaches_the_published_optimum_within_a_small_budget(name, tol, budget, quorum):
    """Convergence on the standard functions, at budgets a costly objective allows.

    The budgets are the point: ``solve_direct``'s own suite uses 800-1200
    evaluations for the same functions and tolerances. These are 55-225.

    **How they are derived, and why the previous derivation was invalid**
    (issue #1036). The old docstring argued "the incumbent first reached the
    tolerance at evaluation ``k`` on the default seed, so a budget of ``B > k``
    has headroom". That argument needs the run at ``B`` to *contain* the run at
    ``k``, and it did not: the initial design was sized
    ``max(n+2, min(10n, max_evals // 2))``, so changing the budget changed the
    design and started a different search. Measured on the ``on_evaluation``
    traces, 17 of 30 budget pairs over ``{40, 46, 60, 80, 100}`` diverged at
    evaluation 1. The design is now a function of the dimension alone
    (:func:`~discopt.solvers.surrogate._default_design_size`), the same probe
    finds 0 of 30, and ``test_a_larger_budget_continues_the_same_search`` below
    keeps it that way — so the headroom argument is now available, and used.

    Each budget is the evaluation at which the **last of 8 seeds** first reached
    the tolerance, with ~50% headroom on top:

    ================  ===========================================  =====
    function          per-seed first reach (seeds 0-7)             8 / 8
    ================  ===========================================  =====
    branin            30, 36, 41, 42, 16, 36, 23, 22               42
    six_hump_camel    23, 34, 17, 29, 18, 17, 22, 12               34
    ackley_2          42, 64, 47, 46, 46, 48, 52, 53               64
    hartman_3         60, 50, 19, 17, 24, 25, 20, 41               60
    goldstein_price   120, 143, 131, 150, 82, 144, 132, 84         150
    ================  ===========================================  =====

    **The assertion is a statement about the method, not about one trajectory.**
    It runs ``k`` seeds and requires a quorum of them to reach the tolerance,
    plus the median relative error to meet it. A single-seed pass/fail was the
    other half of what made this test fragile: the search is deterministic per
    seed but *chaotic*, so a different BLAS can move one seed into a neighbouring
    basin — which is exactly how the original failure was reported (rel. err.
    1.1202e-2 against 1e-2) on a machine where a re-derivation of the same panel
    passes. A convergence regression worth catching moves the population; a
    floating-point difference moves one seed.

    ``goldstein_price`` is here for coverage of a sharply scaled objective, and
    what it covers is a **limitation**, stated rather than hidden: its values span
    3 to ~10⁶ on the box, so an interpolant fitted to raw values resolves the 10⁶
    region and is nearly flat where the optimum is. It needs by far the largest
    budget — 150 evaluations for 8 of 8 seeds against 34-64 for the rest — and it
    is the one function in the panel where DIRECT beats this backend. This test
    pins only that the default configuration converges on it; the remedy is a
    monotone objective transformation before fitting, a named follow-up in the
    module docstring that is deliberately not implemented.
    """
    tf = tfs.get(name)
    seeds = list(range(_PANEL_SEEDS))
    errors = []
    for seed in seeds:
        result = _solve(tf, max_evals=budget, seed=seed, **_FAST)
        assert result.objective is not None, f"{name}: seed {seed} returned no incumbent"
        errors.append(tf.relative_error(result.objective))
    assert len(errors) == len(seeds), "the panel loop ran no seeds — this measured nothing"

    reached = sum(1 for e in errors if e <= tol)
    detail = ", ".join(f"seed {s}: {e:.4e}" for s, e in zip(seeds, errors))
    assert reached >= quorum, (
        f"{name}: only {reached}/{len(seeds)} seeds reached {tol} within {budget} "
        f"evaluations (need {quorum}); published optimum {tf.fstar}; {detail}"
    )
    assert float(np.median(errors)) <= tol, (
        f"{name}: median relative error {np.median(errors):.4e} exceeds {tol} "
        f"over {len(seeds)} seeds at {budget} evaluations; {detail}"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("name", "factor"),
    [
        ("six_hump_camel", 3.0),  # measured 6.0x
        ("branin", 1.4),  # measured 1.9x
        ("ackley_2", 1.2),  # measured 1.4x
    ],
)
def test_uses_fewer_evaluations_than_direct_for_the_same_accuracy(name, factor):
    """The claim this backend is for, measured in evaluations rather than asserted.

    DIRECT spends its budget on geometry and is happy to take hundreds of samples;
    this backend spends real computation between samples so that each one counts.
    On a genuinely expensive objective that difference is the whole value
    proposition, so it is measured head to head — same function, same box, same
    accuracy target, both engines driven from the same ``direct_testfuncs``
    definition.

    **Measured, evaluations to 1e-2 relative error** (median of seeds 0-2 for the
    surrogate; DIRECT is deterministic), 60-evaluation budget:

    ================  ==========  ======  ==============
    function          surrogate   DIRECT  factor
    ================  ==========  ======  ==============
    six_hump_camel    23          137     6.0x
    branin            36          69      1.9x
    hartman_3         50          79      1.6x
    ackley_2          44.5 (2/3)  67      1.5x
    goldstein_price   never 0/3   75      **loss**
    ================  ==========  ======  ==============

    Re-measured after issue #1036 resized the initial design; the DIRECT column,
    being deterministic, did not move. Read it as the 3-seed slice it is:
    ``six_hump_camel`` (32 → 23) and ``branin`` (38 → 36) improved,
    ``goldstein_price`` got worse, and the other two moved in both directions —
    ``hartman_3``'s median rose 46 → 50 because all three seeds now reach the
    tolerance where two did, and ``ackley_2``'s fell 48 → 44.5 because one seed
    stopped reaching it inside 60 evaluations. The verdict on that change is the
    8-function, 12-seed panel in
    ``docs/dev/surrogate-initial-design-2026-08-29.md``, not this table. The
    asserted factors below were not re-tuned to the new numbers — they were
    already well under the old ones and are further under these.

    Three honest points, because "far fewer" is not uniformly true and pretending
    otherwise would be the kind of published-then-retracted claim CLAUDE.md §11 is
    about:

    * the margin on smooth low-dimensional functions is a factor of ~2, not the
      order of magnitude a surrogate wins by in harder settings. DIRECT is a
      strong baseline exactly here;
    * **goldstein_price is a real loss** and is therefore not in the parametrize
      list — see the module docstring for why (objective dynamic range of ~10⁶)
      and for the named remedy that is not implemented;
    * DIRECT's engine is used *without* its local-refinement hybrid, since that
      spends uncounted evaluations through a different path. That is the same
      reason ``local_refine`` defaults off in this backend.

    The asserted factors sit well under the measured ones — seed-to-seed spread on
    branin alone is 16-42 evaluations over seeds 0-7 — because this must not be
    flaky. The table, not the threshold, is the record.
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
    assert median * factor <= direct_evals, (
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
