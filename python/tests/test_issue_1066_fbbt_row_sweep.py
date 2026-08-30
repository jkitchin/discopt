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
