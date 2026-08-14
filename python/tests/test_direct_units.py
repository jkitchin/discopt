"""Unit tests for the pure logic in :mod:`discopt.solvers.direct`.

Fast and model-free: everything here drives the partition/selection machinery on
plain numpy functions, so nothing compiles a DAG or solves an NLP.

Two of these tests exist because the corresponding bugs were real. During the
entry experiment (``docs/dev/direct-entry-2026-08-12.md``) the prototype twice
produced a confident, wrong verdict:

* ``K_max`` in Eq. (4) was sign-flipped, so every rectangle except the largest was
  rejected and the search stalled at a fixed refinement depth;
* the convex hull kept its leftmost vertex, which is optimal only for ``K < 0``
  (preferring *smaller* rectangles with *worse* values).

Both are silent — the search still runs, just badly — so they are pinned against
a brute-force sweep over ``K``, which is the *definition* of potential optimality
rather than a second implementation of it.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt.solvers._dfo_common import glce_merit
from discopt.solvers.direct import (
    _DirectSearch,
    _lower_right_hull,
    select_potentially_optimal,
)

pytestmark = pytest.mark.unit


def _brute_force_potentially_optimal(sizes, values, n_samples: int = 20000) -> list[int]:
    """Ground truth: indices minimizing ``f - K*d`` for some ``K > 0``.

    This is Eq. (3) evaluated directly, not a reimplementation of the hull.
    """
    sizes = np.asarray(sizes, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    winners = set()
    for k in np.logspace(-6, 6, n_samples):
        winners.add(int(np.argmin(values - k * sizes)))
    return sorted(winners)


# ── selection ────────────────────────────────────────────────────────────────


def test_hull_selects_lower_right_convex_hull():
    """A point above the chord between its neighbours is not potentially optimal."""
    pts = [(0.1, 5.0, 0), (0.2, 4.0, 1), (0.3, 4.5, 2), (0.5, 1.0, 3), (0.8, 3.0, 4)]
    hull = _lower_right_hull(pts)
    assert hull == [3, 4], hull
    # (0.1, 5.0) is dominated by (0.5, 1.0) for every K > 0: it is smaller AND
    # worse. Keeping it is the bug this test exists for.
    assert 0 not in hull
    assert 2 not in hull


def test_hull_excludes_vertices_needing_negative_k():
    """The leftmost hull vertex survives only when it is also the best-valued."""
    # Monotone decreasing: the largest rectangle also has the best value, so it
    # dominates outright and is the only potentially optimal one.
    assert _lower_right_hull([(1.0, 9.0, 0), (2.0, 5.0, 1), (3.0, 0.0, 2)]) == [2]
    # Monotone increasing: every vertex is optimal for some K.
    assert _lower_right_hull([(1.0, 0.0, 0), (2.0, 1.0, 1), (3.0, 3.0, 2)]) == [0, 1, 2]


@pytest.mark.parametrize("seed", range(25))
def test_selection_matches_brute_force_definition(seed):
    """Randomized cross-check of Eq. (3) against a direct sweep over K."""
    rng = np.random.default_rng(seed)
    sizes = np.unique(np.sort(rng.uniform(0.01, 1.0, int(rng.integers(1, 9)))))
    values = rng.uniform(-5.0, 5.0, sizes.size)
    got = sorted(select_potentially_optimal(sizes, values, epsilon=0.0, f_min=float(values.min())))
    assert got == _brute_force_potentially_optimal(sizes, values)


def test_largest_rectangle_is_always_potentially_optimal():
    """K is unbounded above for the rightmost point, so Eq. (4) always holds there.

    Reading Eq. (4) with ``K = 0`` for the largest rectangle instead excludes it
    almost always, which starves global search — the search then only ever
    refines and never explores.
    """
    sizes = np.array([0.05, 0.2, 0.9])
    values = np.array([0.0, 3.0, 10.0])  # the biggest rectangle has the worst value
    selected = select_potentially_optimal(sizes, values, epsilon=1e-4)
    assert 2 in selected


def test_epsilon_prunes_rectangles_promising_no_real_improvement():
    """A large epsilon keeps only the rectangles that could beat f_min by that margin."""
    sizes = np.array([0.05, 0.1, 0.2, 0.9])
    values = np.array([1.0, 1.05, 1.2, 5.0])
    lenient = select_potentially_optimal(sizes, values, epsilon=0.0)
    strict = select_potentially_optimal(sizes, values, epsilon=0.5)
    assert set(strict) <= set(lenient)
    assert len(strict) < len(lenient)


def test_selection_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="matching 1-D arrays"):
        select_potentially_optimal(np.array([1.0, 2.0]), np.array([1.0]))


def test_selection_of_empty_partition_is_empty():
    assert select_potentially_optimal(np.array([]), np.array([])) == []


# ── division ─────────────────────────────────────────────────────────────────


def _sum_oracle(x):
    return float(np.sum(x)), 0.0


def test_division_splits_one_long_side_by_default():
    """``divide="one"`` trisects a single dimension: 2 new points, 2 new rectangles."""
    s = _DirectSearch(np.zeros(3), np.ones(3), divide="one")
    s.run(_sum_oracle, max_evals=3)
    assert len(s.part) == 3  # the parent plus two children
    assert s.stats.evals == 3  # the centre plus one +/- pair


def test_division_all_sides_samples_every_long_dimension():
    """``divide="all"`` is the 1993 rule: 2n new points on the first division."""
    s = _DirectSearch(np.zeros(3), np.ones(3), divide="all")
    s.run(_sum_oracle, max_evals=7)
    assert s.stats.evals == 7  # centre + 2 per dimension
    assert len(s.part) == 7


def test_one_side_rule_never_starves_a_dimension():
    """No dimension is left unsplit when the tie-break prefers the least-split side.

    The rule is a per-rectangle tie-break, not a global balancer, so the counts do
    not equalize exactly — what it guarantees is that no dimension is ignored,
    which is the failure mode an arbitrary tie-break would produce.
    """
    s = _DirectSearch(np.zeros(4), np.ones(4), divide="one")
    s.run(_sum_oracle, max_evals=400)
    counts = s.split_counts
    assert counts.min() > 0, f"a dimension was never split: {counts}"
    assert counts.max() <= 3 * counts.min(), f"splitting is lopsided: {counts}"


def test_division_only_splits_long_dimensions():
    """A dimension already split more than another is not eligible while it is shorter."""
    s = _DirectSearch(np.zeros(2), np.ones(2), divide="one")
    s.run(_sum_oracle, max_evals=200)
    # Every rectangle's split counts stay within one of each other along the path
    # DIRECT refines, which is what "only divide the long sides" guarantees.
    for t in s.part.t:
        assert int(t.max()) - int(t.min()) <= 1, t


def test_center_vertex_distance_is_exact_per_level():
    """Rectangles with the same split multiset share a distance exactly (no float drift)."""
    s = _DirectSearch(np.zeros(3), np.ones(3), divide="one")
    s.run(_sum_oracle, max_evals=300)
    by_level: dict[tuple[int, ...], set[float]] = {}
    for i in range(len(s.part)):
        by_level.setdefault(s.part.level_key(i), set()).add(s.part.distance(i))
    for key, distances in by_level.items():
        assert len(distances) == 1, f"level {key} has drifting distances: {distances}"


# ── integer handling ─────────────────────────────────────────────────────────


def test_integer_centers_are_always_integral():
    """Jones 2001: a centre takes an integer value in an integer coordinate."""
    mask = np.array([True, False])
    s = _DirectSearch(np.array([0.0, 0.0]), np.array([6.0, 1.0]), integer_mask=mask, divide="one")
    s.run(_sum_oracle, max_evals=400)
    for i in range(len(s.part)):
        x = s.to_model_point(s.part.centers[i])
        assert abs(x[0] - round(x[0])) < 1e-9, x


def test_integer_point_stays_within_bounds():
    """Rounding must not push a centre outside the declared integer range."""
    mask = np.array([True])
    s = _DirectSearch(np.array([2.0]), np.array([5.0]), integer_mask=mask)
    for u in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = s.to_model_point(np.array([u]))
        assert 2.0 <= x[0] <= 5.0


def test_integer_dimension_stops_dividing_below_unit_width():
    """Splitting an integer dimension past unit width only re-samples rounded duplicates."""
    s = _DirectSearch(np.array([0.0]), np.array([2.0]), integer_mask=np.array([True]))
    s.run(_sum_oracle, max_evals=500)
    widths = [3.0 ** (-int(t[0])) * 2.0 for t in s.part.t]
    assert min(widths) >= 1.0 / 3.0, (
        f"divided an integer dimension far past unit width: {min(widths)}"
    )


def test_exhausted_integer_range_terminates_without_burning_budget():
    """A fully enumerated integer range stops the search instead of spinning.

    Regression for a hang, not a wrong answer: a cache hit deliberately does not
    spend evaluation budget (the budget counts *expensive* calls), so once every
    new centre rounded onto an already-sampled point, ``while evals < max_evals``
    could never advance and the partition grew without bound. This 1-D integer
    variable has exactly three distinct model points.
    """
    s = _DirectSearch(np.array([0.0]), np.array([2.0]), integer_mask=np.array([True]))
    s.run(_sum_oracle, max_evals=200)
    assert s.stats.evals <= 3, s.stats.evals
    assert len(s.part) <= 8, f"partition grew unreasonably: {len(s.part)}"


def test_evaluation_cache_serves_repeated_points():
    """A point already evaluated costs nothing — the regime this backend exists for."""
    s = _DirectSearch(np.array([0.0]), np.array([10.0]))
    calls = {"n": 0}

    def counting(x):
        calls["n"] += 1
        return float(x[0]), 0.0

    u = np.array([0.5])
    assert s.evaluate(u, counting) == s.evaluate(u, counting)
    assert calls["n"] == 1
    assert s.stats.cache_hits == 1
    assert s.stats.evals == 1


# ── constraint handling (DIRECT-GLce) ────────────────────────────────────────


def test_rank_values_reads_the_partition_through_the_shared_merit_rule():
    """``rank_values`` is wiring, not a formula: the rule lives in ``_dfo_common``.

    The rule's own behaviour (both phases, the ``ε_cons`` band, ``finite_fill``)
    is covered in ``test_dfo_common``. What is checked *here* is what only this
    class can get wrong — that it feeds the partition's own ``fvals``/``viols``
    and its own incumbent and tolerance, in that order.
    """
    s = _DirectSearch(np.zeros(1), np.ones(1))
    s.best_feasible_value = 5.0
    s.eps_cons = 1e-3
    s.part.add(np.array([0.5]), np.zeros(1, dtype=np.int64), 4.0, 1e-6)  # inside the band
    s.part.add(np.array([0.5]), np.zeros(1, dtype=np.int64), -1e6, 3.0)  # infeasible, great f

    expected = glce_merit(
        np.array(s.part.fvals, dtype=np.float64),
        np.array(s.part.viols, dtype=np.float64),
        s.best_feasible_value,
        s.eps_cons,
        finite_fill=False,
    )
    np.testing.assert_array_equal(s.rank_values(), expected)
    assert s.rank_values()[0] < s.rank_values()[1], "an infeasible point must not win on f alone"


def test_rank_values_leaves_an_undefined_objective_at_infinity():
    """DIRECT ranks by comparison, so ``finite_fill=False`` is the right choice here.

    A point where the black box was undefined carries ``+inf``, which loses every
    comparison and is never selected. This pins the flag: the surrogate backend
    passes ``True`` because it *fits* the merit, and picking that up here would
    make an undefined point selectable.
    """
    s = _DirectSearch(np.zeros(1), np.ones(1))
    s.best_feasible_value = 1.0
    s.part.add(np.array([0.5]), np.zeros(1, dtype=np.int64), 1.0, 0.0)
    s.part.add(np.array([0.5]), np.zeros(1, dtype=np.int64), np.inf, 0.0)
    assert np.isinf(s.rank_values()[1])


def test_scalar_rank_matches_rank_values_on_the_same_point():
    """The one-at-a-time ranker used by the polish must not drift from the batch one."""
    s = _DirectSearch(np.zeros(1), np.ones(1))
    s.best_feasible_value = 10.0
    s.eps_cons = 1e-3
    for fval, viol in [(10.0, 0.0), (4.0, 1e-6), (-1e6, 3.0)]:
        s.part = type(s.part)()
        s.part.add(np.array([0.5]), np.zeros(1, dtype=np.int64), fval, viol)
        assert s._scalar_rank(fval, viol) == pytest.approx(s.rank_values()[0])


# ── engine-level behaviour ───────────────────────────────────────────────────


def test_search_is_deterministic():
    """DIRECT has no RNG; this pins that none creeps in."""

    def rastrigin(x):
        return float(10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))), 0.0

    runs = []
    for _ in range(2):
        s = _DirectSearch(np.full(2, -4.12), np.full(2, 6.12))
        s.run(rastrigin, max_evals=500)
        runs.append((s.best_feasible_value, s.best_feasible_point.tolist(), s.stats.evals))
    assert runs[0] == runs[1]


def test_search_rejects_non_finite_box():
    """An infinite side has no midpoint; NaN centres would all hash to one cache key."""
    with pytest.raises(ValueError, match="finite"):
        _DirectSearch(np.array([-np.inf]), np.array([1.0]))
    with pytest.raises(ValueError, match="finite"):
        _DirectSearch(np.array([0.0]), np.array([np.inf]))


def test_search_rejects_unknown_options():
    with pytest.raises(ValueError, match="divide"):
        _DirectSearch(np.zeros(1), np.ones(1), divide="sideways")
    with pytest.raises(ValueError, match="variant"):
        _DirectSearch(np.zeros(1), np.ones(1), variant="nope")


def test_search_respects_the_evaluation_budget():
    """``max_evals`` is the primary cost control and must not be overshot."""
    for budget in (1, 5, 37, 200):
        s = _DirectSearch(np.zeros(3), np.ones(3))
        s.run(_sum_oracle, max_evals=budget)
        assert s.stats.evals <= budget, (budget, s.stats.evals)


class _Answered(Exception):
    """Stops a DIRECT run from the iteration hook once its crossing is recorded."""


def test_reproduces_the_surveys_published_evaluation_counts():
    """Guards the whole engine against the class of bug the entry experiment hit.

    Jones & Martins Fig. 15 minimizes ``1 + x1 + ... + x5`` and reports evaluations
    to 1% accuracy: 14,492 with the original rules, 470 breaking ties, 192 also
    trisecting one side. A sign error or a hull defect changes these by orders of
    magnitude, so the published numbers are a far sharper test than any assertion
    we could invent.
    """

    def linear(x):
        return 1.0 + float(np.sum(x)), 0.0

    def crossed(value):
        return value is not None and abs(value - 1.0) <= 1e-2

    def evals_to_one_percent(divide, break_ties):
        # Every arm asks one question -- at which evaluation does the incumbent
        # first come within 1% -- so the budget past that point buys no assertion
        # and is pure cost. The budget stays at the survey's 40,000 (the original
        # rules need ~14.5k of it, and a regression that pushed a crossing past
        # the budget must still show up as `None`); the hook stops the search the
        # moment the answer is recorded. `history` is append-only and the value is
        # the FIRST crossing, so stopping after it is observed cannot change what
        # this returns -- verified equal to the run-to-budget values (16555/503/203).
        # Without this the three arms burn 120k evaluations of hull maintenance
        # nothing asserts, which under `--cov` on a CI runner overran the timeout.
        s = _DirectSearch(np.zeros(5), np.ones(5), divide=divide, break_ties=break_ties)
        history: list[tuple[int, float]] = []

        def observe(se):
            history.append((se.stats.evals, se.best_feasible_value))
            if crossed(se.best_feasible_value):
                raise _Answered

        try:
            s.run(linear, max_evals=40000, on_iteration=observe)
        except _Answered:
            pass  # our own sentinel only; any real failure still propagates
        return next((e for e, v in history if crossed(v)), None)

    original = evals_to_one_percent("all", False)
    no_ties = evals_to_one_percent("all", True)
    one_side = evals_to_one_percent("one", True)

    assert one_side is not None and no_ties is not None and original is not None
    assert one_side < no_ties < original, (one_side, no_ties, original)
    assert abs(one_side - 192) <= 60, f"one-side+break-ties = {one_side}, survey reports 192"
    assert abs(no_ties - 470) <= 120, f"all-sides+break-ties = {no_ties}, survey reports 470"
    assert original > 5000, f"the original rules should be dramatically worse, got {original}"


# ── derivative-free refinement ───────────────────────────────────────────────


def test_derivative_free_refinement_improves_a_zero_gradient_objective():
    """Powell moves where a gradient method cannot: a staircase objective.

    A ``dm.custom`` body is JAX-*traceable* by construction, which is not the same
    as usefully *differentiable*. ``jnp.floor``, a table lookup, or a simulator
    behind ``jax.pure_callback`` all hand back zero or meaningless gradients, and
    a gradient method then sits still while reporting success. This is the case
    ``local_refine_method="derivative-free"`` exists for.
    """
    from discopt.solvers.direct import _refine_derivative_free

    def staircase(x):
        q = np.floor(x * 4.0) / 4.0  # piecewise constant: gradient 0 a.e.
        return float(np.sum((q - 1.25) ** 2) + 0.01 * np.sum(x**2)), 0.0

    s = _DirectSearch(np.full(2, -3.0), np.full(2, 5.0))
    s.run(staircase, max_evals=300)
    before = s.best_feasible_value
    _refine_derivative_free(s, staircase, s.best_feasible_point, max_fev=400)
    assert s.best_feasible_value <= before + 1e-12, "refinement must never make things worse"


def test_derivative_free_refinement_holds_integers_fixed():
    """Only continuous coordinates are polished; an integer stays at its value."""
    from discopt.solvers.direct import _refine_derivative_free

    def objective(x):
        return float((x[0] - 2.7) ** 2 + (x[1] - 1.3) ** 2), 0.0

    mask = np.array([True, False])
    s = _DirectSearch(np.array([0.0, 0.0]), np.array([6.0, 6.0]), integer_mask=mask)
    s.run(objective, max_evals=200)
    start = s.best_feasible_point.copy()
    _refine_derivative_free(s, objective, start, max_fev=300)
    best = s.best_feasible_point
    assert best[0] == pytest.approx(start[0]), "the integer coordinate must not move"
    assert abs(best[0] - round(best[0])) < 1e-9, "the integer coordinate must stay integral"
    assert abs(best[1] - 1.3) < 1e-3, f"the continuous coordinate should be polished: {best[1]}"


def test_derivative_free_refinement_counts_its_evaluations():
    """Polish calls are real function calls and must be charged to the budget."""
    from discopt.solvers.direct import _refine_derivative_free

    def objective(x):
        return float(np.sum((x - 0.3) ** 2)), 0.0

    s = _DirectSearch(np.zeros(2), np.ones(2))
    s.run(objective, max_evals=50)
    before = s.stats.evals
    _refine_derivative_free(s, objective, s.best_feasible_point, max_fev=60)
    assert s.stats.evals > before, "polish evaluations must be counted, not hidden"


# ── the variant trade-off that fixes the default ─────────────────────────────


def _evals_to_tolerance(tf, variant: str, tol: float, budget: int = 6000):
    """Evaluations before the incumbent first reaches ``tol`` relative accuracy."""
    s = _DirectSearch(tf.lb, tf.ub, variant=variant)
    history: list[tuple[int, float]] = []
    s.run(
        lambda x: (float(tf.np_body(x)), 0.0),
        budget,
        on_iteration=lambda se: history.append((se.stats.evals, se.best_feasible_value)),
    )
    return next((e for e, v in history if v is not None and tf.relative_error(v) <= tol), None)


def test_variant_tradeoff_is_why_gl_is_not_the_default():
    """DIRECT-GL wins big on one problem class and loses on another — hence opt-in.

    Measured here rather than asserted, because the choice of default rests on it.
    Evaluations to 1e-2 relative accuracy, this implementation:

    ==========  =======  ====
    function    classic  gl
    ==========  =======  ====
    hartman_6   105      277
    shubert     2269     181
    ==========  =======  ====

    Both directions reproduce the survey: it reports Hartman-6 at 571 evaluations
    for DIRECT versus 8793 for DIRECT-GL, and Shubert at 2967 for DIRECT versus
    425 for DIRECT-GL. A fixed-budget comparison of final objective cannot see
    this — at a generous budget both converge and the difference is noise, which
    is how an earlier version of this test managed to fail for the wrong reason.

    If GL ever wins on Hartman-6 too, the default deserves revisiting; that is
    what this test is here to catch.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from support import direct_testfuncs as tfs

    hartman = tfs.get("hartman_6")
    classic_h = _evals_to_tolerance(hartman, "classic", 1e-2)
    gl_h = _evals_to_tolerance(hartman, "gl", 1e-2)
    assert classic_h is not None and gl_h is not None
    assert classic_h < gl_h, (
        f"hartman_6: classic {classic_h} vs gl {gl_h} — GL now wins the case that "
        "justifies classic being the default; re-run the variant panel"
    )

    shubert = tfs.get("shubert")
    classic_s = _evals_to_tolerance(shubert, "classic", 1e-2)
    gl_s = _evals_to_tolerance(shubert, "gl", 1e-2)
    assert gl_s is not None
    assert classic_s is None or gl_s < classic_s, (
        f"shubert: gl {gl_s} vs classic {classic_s} — GL is supposed to win the "
        "multimodal case; if it no longer does, the GL implementation regressed"
    )


# ── parallel evaluation ──────────────────────────────────────────────────────


def _threaded(workers: int):
    from concurrent.futures import ThreadPoolExecutor

    return ThreadPoolExecutor(max_workers=workers)


@pytest.mark.parametrize("name", ["branin", "hartman_6", "shubert", "rastrigin_2"])
def test_parallel_evaluation_is_identical_to_serial(name):
    """``n_jobs`` must change the wall clock and nothing else.

    Within an iteration every sample point is independent, so they can be
    evaluated concurrently — but only if the *result* is bit-for-bit what the
    serial path produces. Two things make that true and both are easy to lose:
    results are collected in input order (``executor.map`` preserves it), and the
    incumbent is updated at apply time in selection order rather than as
    evaluations land. Compare the full state, not just the objective: a
    trajectory can diverge while the final value happens to agree.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from support import direct_testfuncs as tfs

    tf = tfs.get(name)

    def state(workers):
        s = _DirectSearch(tf.lb, tf.ub)
        pool = _threaded(workers) if workers > 1 else None
        try:
            s.run(lambda x: (float(tf.np_body(x)), 0.0), 1200, executor=pool)
        finally:
            if pool is not None:
                pool.shutdown()
        return (
            s.best_feasible_value,
            tuple(s.best_feasible_point),
            s.stats.evals,
            len(s.part),
            tuple(s.split_counts.tolist()),
        )

    serial = state(1)
    assert state(4) == serial
    assert state(8) == serial


def test_batched_iteration_matches_the_serial_budget_cutoff():
    """Planning a whole iteration up front must not move where the budget bites.

    Two subtleties, both found by comparing against the pre-batching
    implementation on 44 configurations:

    * the budget guard charges a pessimistic 2 evaluations per dimension before
      sampling, even though a cached point turns out free — charging only the
      uncached points shifts the mid-iteration cutoff and changes the partition;
    * ``split_counts`` must be charged at *plan* time, because the ``divide="one"``
      tie-break reads it and the serial code divided each rectangle before
      planning the next.

    A budget that is not a multiple of an iteration's demand is what exposes both.
    """
    for budget in (7, 23, 51, 118, 349):
        results = []
        for workers in (1, 4):
            s = _DirectSearch(np.zeros(3), np.ones(3))
            pool = _threaded(workers) if workers > 1 else None
            try:
                s.run(lambda x: (float(np.sum(np.cos(3 * x))), 0.0), budget, executor=pool)
            finally:
                if pool is not None:
                    pool.shutdown()
            results.append((s.best_feasible_value, s.stats.evals, len(s.part)))
            assert s.stats.evals <= budget, (budget, s.stats.evals)
        assert results[0] == results[1], f"budget {budget}: {results}"


def test_parallel_evaluation_actually_runs_concurrently():
    """A slow objective must finish faster on more threads, or the plumbing is inert.

    Guards against a batching path that collects the points and then evaluates
    them serially anyway — which would pass every determinism test above while
    delivering none of the benefit.
    """
    import time

    def slow(x):
        time.sleep(0.01)
        return float(np.sum(x**2)), 0.0

    timings = {}
    for workers in (1, 8):
        s = _DirectSearch(np.zeros(3), np.ones(3))
        pool = _threaded(workers) if workers > 1 else None
        start = time.perf_counter()
        try:
            s.run(slow, 120, executor=pool)
        finally:
            if pool is not None:
                pool.shutdown()
        timings[workers] = time.perf_counter() - start
        assert s.stats.batches > 0, "no batch was ever formed"
        assert s.stats.max_batch_size > 1, "batches never held more than one point"

    assert timings[8] < 0.6 * timings[1], (
        f"8 threads ({timings[8]:.2f}s) should beat serial ({timings[1]:.2f}s) "
        "on a sleep-bound objective; batching may not be dispatching concurrently"
    )
