"""#1066: the linear-row FBBT sweep was O(m*n^2) and is now O(m*n).

For each linear row the pre-change code recomputed, inside the ``i`` loop, a
sum over every ``k != i`` -- a sum that differs between consecutive ``i`` only
in which single term is excluded.  Profiling a default 60 s solve of
``portfol_classical050_1`` (150 vars, 103 rows) put **19.8 s of 48.8 s in
``_tighten_node_bounds_with_status``, 9.9 s of it self time in those loops**,
with 56M ``builtins.abs`` calls -- 40% of the budget on a sum recomputed 150
times over.

The replacement keeps the running total and subtracts the excluded term, so
each ``i`` still observes the bounds tightened by ``i - 1``.  That matters:
the loop is Gauss-Seidel, not Jacobi -- the inner sum reads ``lb``/``ub``
live, so a fully vectorised rewrite would be a *different* (weaker) algorithm.
These tests pin the equivalence against a verbatim transcription of the
original loops, so a future vectorisation cannot silently weaken it.
"""

import numpy as np
from discopt.solver import _fbbt_linear_row_sweep


def _reference(J, g_j, cu_j, cl_j, mid, lb, ub):
    """Verbatim transcription of the pre-#1066 O(n^2) inner loops."""
    n = len(lb)
    changed = False
    if cu_j < 1e19:
        for i in range(n):
            if abs(J[i]) < 1e-12 or lb[i] == ub[i]:
                continue
            residual = cu_j - g_j
            for k in range(n):
                if k == i or abs(J[k]) < 1e-12:
                    continue
                residual -= J[k] * ((lb[k] if J[k] > 0 else ub[k]) - mid[k])
            if J[i] > 1e-12:
                new_ub = mid[i] + residual / J[i]
                if new_ub < ub[i] - 1e-10:
                    ub[i] = max(lb[i], new_ub)
                    changed = True
            elif J[i] < -1e-12:
                new_lb = mid[i] + residual / J[i]
                if new_lb > lb[i] + 1e-10:
                    lb[i] = min(ub[i], new_lb)
                    changed = True
    if cl_j > -1e19:
        for i in range(n):
            if abs(J[i]) < 1e-12 or lb[i] == ub[i]:
                continue
            residual = cl_j - g_j
            for k in range(n):
                if k == i or abs(J[k]) < 1e-12:
                    continue
                residual -= J[k] * ((ub[k] if J[k] > 0 else lb[k]) - mid[k])
            if J[i] > 1e-12:
                new_lb = mid[i] + residual / J[i]
                if new_lb > lb[i] + 1e-10:
                    lb[i] = min(ub[i], new_lb)
                    changed = True
            elif J[i] < -1e-12:
                new_ub = mid[i] + residual / J[i]
                if new_ub < ub[i] - 1e-10:
                    ub[i] = max(lb[i], new_ub)
                    changed = True
    return lb, ub, changed


def _apply(J, g_j, cu_j, cl_j, mid, lb, ub):
    changed = False
    if cu_j < 1e19 and _fbbt_linear_row_sweep(J, g_j, cu_j, mid, lb, ub, True):
        changed = True
    if cl_j > -1e19 and _fbbt_linear_row_sweep(J, g_j, cl_j, mid, lb, ub, False):
        changed = True
    return lb, ub, changed


def test_matches_the_quadratic_reference_including_unbounded_variables():
    """The sum is now incremental; the *decisions* must be bit-for-bit the same.

    Free variables are the case the incremental form could get wrong: a term
    drawing on an infinite bound is infinite, and ``sum - term_i`` would be
    ``inf - inf`` = NaN where the original produced a finite residual.  Infinite
    terms are counted instead, so the ``i`` that owns the only infinite term
    still tightens.
    """
    rng = np.random.default_rng(1066)
    checks = tightened = with_inf = 0
    for _ in range(600):
        n = int(rng.integers(2, 12))
        J = rng.normal(size=n) * rng.choice([1.0, 1.0, 1.0, 0.0], size=n)
        lb = -rng.uniform(0, 10, size=n)
        ub = rng.uniform(0, 10, size=n)
        for idx in rng.choice(n, size=int(rng.integers(0, n // 2 + 1)), replace=False):
            which = int(rng.integers(0, 3))
            if which != 1:
                lb[idx] = -np.inf
            if which != 0:
                ub[idx] = np.inf
        if rng.random() < 0.15:
            k = int(rng.integers(0, n))
            lb[k] = ub[k] = float(rng.normal())
        if not (np.isfinite(lb).all() and np.isfinite(ub).all()):
            with_inf += 1
        lo, hi = np.clip(lb, -1e6, 1e6), np.clip(ub, -1e6, 1e6)
        mid = lo + 0.5 * (hi - lo)
        g_j = float(rng.normal())
        cu_j = float(rng.normal()) if rng.random() < 0.8 else 1e20
        cl_j = float(rng.normal()) - 5.0 if rng.random() < 0.5 else -1e20

        r_lb, r_ub, r_ch = _reference(J, g_j, cu_j, cl_j, mid, lb.copy(), ub.copy())
        n_lb, n_ub, n_ch = _apply(J, g_j, cu_j, cl_j, mid, lb.copy(), ub.copy())

        checks += 1
        assert r_ch == n_ch
        if np.any(r_lb != lb) or np.any(r_ub != ub):
            tightened += 1
        np.testing.assert_allclose(r_lb, n_lb, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(r_ub, n_ub, rtol=1e-9, atol=1e-9)

    # CLAUDE.md §6: the probe must prove it fired, and that it fired on the
    # cases it exists to cover -- rows that actually move a bound, and rows
    # with infinite terms.
    assert checks == 600
    assert tightened > 100, f"vacuous: only {tightened} rows tightened"
    assert with_inf > 100, f"vacuous: only {with_inf} rows had an infinite bound"


def test_hoisting_the_sum_cannot_miss_a_bound_moved_earlier_in_the_sweep():
    """The pre-change loop read ``lb``/``ub`` live inside the ``k`` sum, so it
    could in principle have seen a bound tightened by an earlier ``i``.  It
    never could: in each sense the update writes the bound *opposite* the one
    the terms read (``J_i > 0`` writes ``ub_i``, while a ``J_k > 0`` term draws
    on ``lb_k``), so no update inside a sweep changes any term of that sweep.

    This pins that invariant directly -- rebuilding the terms from the *post*
    sweep bounds must reproduce them exactly -- on a row where the sweep really
    does move bounds.  If someone later changes which bound an update writes,
    the hoisted sum silently goes stale and this test is what catches it.
    """
    rng = np.random.default_rng(20260829)
    moved = checks = 0
    for _ in range(200):
        n = int(rng.integers(3, 10))
        J = rng.normal(size=n)
        lb = -rng.uniform(1, 8, size=n)
        ub = rng.uniform(1, 8, size=n)
        mid = lb + 0.5 * (ub - lb)
        g_j, rhs = float(rng.normal()), float(rng.normal())
        for is_upper in (True, False):
            before = np.where((J > 0) == is_upper, lb, ub) * 1.0
            terms_before = J * (before - mid)
            pre = (lb.copy(), ub.copy())
            changed = _fbbt_linear_row_sweep(J, g_j, rhs, mid, lb, ub, is_upper)
            after = np.where((J > 0) == is_upper, lb, ub) * 1.0
            terms_after = J * (after - mid)
            checks += 1
            np.testing.assert_array_equal(terms_before, terms_after)
            if changed and (np.any(lb != pre[0]) or np.any(ub != pre[1])):
                moved += 1
    assert checks == 400
    assert moved > 20, f"vacuous: the sweep moved a bound only {moved} times"


def test_an_all_zero_row_tightens_nothing():
    lb, ub = np.array([-1.0, -1.0]), np.array([1.0, 1.0])
    assert not _fbbt_linear_row_sweep(np.zeros(2), 0.0, 0.0, np.zeros(2), lb, ub, True)
    np.testing.assert_array_equal(lb, [-1.0, -1.0])
    np.testing.assert_array_equal(ub, [1.0, 1.0])


def test_a_fixed_variable_is_never_moved():
    """``lb == ub`` is skipped, so a branching decision cannot be undone."""
    J = np.array([1.0, 1.0])
    lb, ub = np.array([2.0, -5.0]), np.array([2.0, 5.0])
    mid = np.array([2.0, 0.0])
    _fbbt_linear_row_sweep(J, 0.0, 0.5, mid, lb, ub, True)
    assert lb[0] == 2.0 and ub[0] == 2.0


def test_the_unbounded_sentinel_is_not_summed_into_the_running_total():
    """The sentinel is ``1e20``, not ``inf`` -- ``isfinite`` does not see it.

    CLAUDE.md is explicit: "``INF`` in the Rust LP layer is the sentinel
    ``1e20``, not ``f64::INFINITY``... never [test] on a product... [it]
    destroys smaller terms in a running sum by cancellation."  The first
    #1066 sweep classified terms with ``np.isfinite``, which accepts a
    ``9.999e19`` bound as an ordinary float.  Its term then enters the running
    total, and the ulp of ``1e20`` is 16384 -- so every small term added to it
    is annihilated, and ``sum - term_i`` returns ``0`` where the true partial
    sum was the small remainder.  The original never formed that sum: it
    subtracted the ``k != i`` terms one at a time from ``rhs - g_j``.

    Hand-computed witness.  Row ``x + y <= 10`` linearised at ``mid = (0, 3)``,
    ``y in [0, 5]``, ``x`` unbounded via the sentinel.  With ``y >= 0`` the row
    implies ``x <= 10``, and that is what the reference returns.  Summing the
    sentinel term loses ``J_y * (lb_y - mid_y) = -3`` and yields ``x <= 7`` --
    an over-tightening that cuts the feasible point ``(10, 0)`` out of the box.
    """
    J = np.array([1.0, 1.0])
    mid = np.array([0.0, 3.0])
    lb = np.array([-9.999e19, 0.0])
    ub = np.array([9.999e19, 5.0])
    g_j, cu_j, cl_j = 3.0, 10.0, -1e20

    r_lb, r_ub, _ = _reference(J, g_j, cu_j, cl_j, mid, lb.copy(), ub.copy())
    n_lb, n_ub, _ = _apply(J, g_j, cu_j, cl_j, mid, lb.copy(), ub.copy())

    assert r_ub[0] == 10.0, f"reference itself is wrong: {r_ub[0]!r}"
    np.testing.assert_array_equal(n_lb, r_lb)
    np.testing.assert_array_equal(n_ub, r_ub)

    # The soundness statement the arithmetic exists to protect: (10, 0)
    # satisfies the row and the declared box, so no sweep may exclude it.
    for x, y in ((10.0, 0.0), (7.5, 2.0), (5.0, 5.0)):
        assert n_lb[0] <= x <= n_ub[0], f"({x}, {y}) cut out of x's box"
        assert n_lb[1] <= y <= n_ub[1], f"({x}, {y}) cut out of y's box"


def test_matches_the_reference_on_rows_carrying_the_sentinel_bound():
    """Randomized differential against the pre-#1066 loop, with ``9.999e19``.

    Bit-equality with the old loop is *unattainable* once a sentinel term is in
    the row, and not because of this rewrite: the old loop subtracted the
    ``k != i`` terms sequentially, so a big term subtracted early annihilates
    every small term subtracted after it, and its result depends on the column
    order.  That is how it "tightened" a ``-9.999e19`` bound to ``-2.03e19`` --
    an artifact, not information.  Treating the sentinel as infinite (what the
    ``cu[j] < 1e19`` guard at the call site already does) declines that move.

    So the assertions here are the ones that carry meaning:

    * **Soundness** -- the sweep is never *tighter* than the reference, so it
      can never cut a point the old loop kept.  This is the direction that
      would be a #1 violation.
    * **Ordinary-scale bounds are untouched** -- wherever the two differ, the
      bound is of sentinel magnitude, i.e. unbounded to every consumer.  This
      is what makes the change node-count neutral on real instances, which the
      bound-neutral panel then confirms end-to-end.
    * **No sentinel in the row: agreement to 1e-12** -- the common case keeps
      the old arithmetic up to the re-association rounding that forming the
      total once (rather than subtracting term by term) necessarily costs.
      That rounding is inherent to the O(n) rewrite, so bound-neutrality is
      settled end-to-end by the panel, not by bit-equality here.
    """
    rng = np.random.default_rng(11066)
    INF = 9.999e19
    checks = tightened = with_sentinel = clean_rows = 0
    sentinel_diffs = ulp_diffs = 0
    for _ in range(600):
        n = int(rng.integers(2, 12))
        J = rng.normal(size=n) * rng.choice([1.0, 1.0, 1.0, 0.0], size=n)
        lb = -rng.uniform(0, 10, size=n)
        ub = rng.uniform(0, 10, size=n)
        n_sent = int(rng.integers(0, n // 2 + 1))
        for idx in rng.choice(n, size=n_sent, replace=False):
            which = int(rng.integers(0, 3))
            if which != 1:
                lb[idx] = -INF
            if which != 0:
                ub[idx] = INF
        has_sentinel = bool(np.any(np.abs(lb) >= 1e19) or np.any(np.abs(ub) >= 1e19))
        with_sentinel += has_sentinel
        clean_rows += not has_sentinel
        lo, hi = np.clip(lb, -1e6, 1e6), np.clip(ub, -1e6, 1e6)
        mid = lo + 0.5 * (hi - lo)
        g_j = float(rng.normal())
        cu_j = float(rng.normal()) if rng.random() < 0.8 else 1e20
        cl_j = float(rng.normal()) - 5.0 if rng.random() < 0.5 else -1e20

        r_lb, r_ub, _ = _reference(J, g_j, cu_j, cl_j, mid, lb.copy(), ub.copy())
        n_lb, n_ub, _ = _apply(J, g_j, cu_j, cl_j, mid, lb.copy(), ub.copy())

        checks += 1
        if np.any(r_lb != lb) or np.any(r_ub != ub):
            tightened += 1

        if not has_sentinel:
            # No sentinel in the row: the only admissible difference is the
            # re-association rounding, never a changed decision.
            np.testing.assert_allclose(r_lb, n_lb, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(r_ub, n_ub, rtol=1e-12, atol=1e-12)
            assert np.all(np.abs(n_lb) < 1e19) == np.all(np.abs(r_lb) < 1e19)
            continue

        # Never tighter than the reference.
        tol = 1e-9 * (1.0 + np.abs(r_lb))
        assert np.all(n_lb <= r_lb + tol), (n_lb, r_lb)
        tol = 1e-9 * (1.0 + np.abs(r_ub))
        assert np.all(n_ub >= r_ub - tol), (n_ub, r_ub)

        # Every difference is either at sentinel magnitude -- i.e. unbounded to
        # every consumer -- or a re-association rounding of a few ulp, which is
        # what forming the total once instead of subtracting term by term costs.
        for a, b in list(zip(r_lb, n_lb)) + list(zip(r_ub, n_ub)):
            if a == b:
                continue
            if abs(a) >= 1e19 or abs(b) >= 1e19:
                sentinel_diffs += 1
                continue
            assert abs(a - b) <= 1e-9 * (1.0 + abs(a)), (a, b)
            ulp_diffs += 1

    # CLAUDE.md 6: prove the probe fired, on both populations it separates.
    assert checks == 600
    assert tightened > 100, f"vacuous: only {tightened} rows tightened"
    assert with_sentinel > 100, f"vacuous: only {with_sentinel} rows had a sentinel"
    assert clean_rows > 50, f"vacuous: only {clean_rows} sentinel-free rows"
    assert sentinel_diffs > 0, "vacuous: no sentinel-magnitude divergence was seen"
    # ``ulp_diffs`` is not asserted non-zero: it is allowed, not required.
    print(f"sentinel_diffs={sentinel_diffs} ulp_diffs={ulp_diffs}")
