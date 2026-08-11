"""#863: the algebraic extractor re-summed variable sizes on every reference.

``problem_classifier._compute_var_offset`` used to compute a variable's flat offset
by re-summing ``model._variables[: var._index]`` from scratch — O(n_vars) per
*variable reference*, so O(references x n_vars) over a whole model. That is the
quadratic #654 removed everywhere else by memoizing an exclusive prefix-sum table on
the model (``Model._flat_var_offset``); the classifier kept its own rescan.

Measured on ``watercontamination0202`` (106,711 variables / 107,209 constraints,
mean ``var._index`` of 59,909 over the references actually walked), sampling rows
with a *stride* across the whole constraint range:

======================  ============  =========================
variant                 per row       extrapolated over 107,209
======================  ============  =========================
rescan (before)         4.108 ms      440 s
memoized (after)        0.012 ms      1.3 s
======================  ============  =========================

440 s is exactly the ">400 s and still running" that made that instance overrun its
``time_limit`` by >8x without ever reaching a solver. Sampling only the *first* rows
hides this completely — they reference the lowest-index variables, where the rescan
is cheapest (measured there: 1.7x, not 340x).

The rescan was copied verbatim into **eight** other places, so removing it only from
the classifier fixed the instance, not the class (CLAUDE.md §2). With extraction
fixed, stack sampling a ``solve(time_limit=30)`` on the same instance found the next
>265 s sitting in ``convexity/linear_context._compute_var_offset`` — the same loop,
reached from ``solver.py:_classify_model_convexity``. A/B on that path, with the
instrument's call count and affine-hit count identical in both arms:

======================  ============  =========================
variant                 per row       extrapolated over 107,209
======================  ============  =========================
rescan (before)         4.085 ms      438 s
memoized (after)        0.011 ms      1.1 s
======================  ============  =========================

Every copy now delegates to ``Model._flat_var_offset``: ``convexity/linear_context``,
``sparsity``, ``obbt`` (x4, nested closures) and ``solvers/amp`` (x3).
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt._relax.convexity.linear_context as linear_context  # noqa: E402
import discopt.modeling as dm  # noqa: E402
import discopt.solvers.amp as amp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._relax.problem_classifier import (  # noqa: E402
    _compute_var_offset,
    dense_A,
    extract_lp_data,
)
from discopt._relax.sparsity import _var_offset  # noqa: E402

# Every module-level copy of the offset helper. `obbt` has four more as nested
# closures, which are the same one-line delegation but not reachable from here.
OFFSET_HELPERS = {
    "problem_classifier": _compute_var_offset,
    "convexity.linear_context": linear_context._compute_var_offset,
    "sparsity": _var_offset,
    "solvers.amp": amp._compute_var_offset,
}

REFS_PER_ROW = 10


def _wide_model(n: int):
    """``n`` scalar variables and ``n // REFS_PER_ROW`` rows, each referencing
    ``REFS_PER_ROW`` variables spread across the whole index range.

    The total number of variable references grows like ``n``, so a correct
    extractor is O(n). The rescan makes it O(n^2), because the mean referenced
    ``var._index`` also grows like ``n``.
    """
    m = dm.Model(f"wide{n}")
    xs = [m.continuous(f"x{i}", lb=0.0, ub=10.0) for i in range(n)]
    m.minimize(xs[0] + xs[-1])
    n_rows = n // REFS_PER_ROW
    for r in range(n_rows):
        # Stride so the referenced indices span [0, n): a row's cost under the
        # rescan is the sum of the referenced indices.
        idx = [(r + k * n_rows) % n for k in range(REFS_PER_ROW)]
        m.subject_to(sum(xs[j] for j in idx) <= float(REFS_PER_ROW))
    return m


def test_offset_matches_the_rescan_it_replaced():
    """Correctness first: the memoized lookup must return exactly what re-summing
    ``model._variables[: var._index]`` returned, including for multi-element
    variables and for the very first/last variable."""
    m = dm.Model("mixed")
    a = m.continuous("a", shape=(3,), lb=0.0, ub=1.0)
    b = m.continuous("b", lb=0.0, ub=1.0)
    c = m.continuous("c", shape=(4, 2), lb=0.0, ub=1.0)
    d = m.binary("d")
    m.minimize(b)
    m.subject_to(b <= 1)

    for var in (a, b, c, d):
        rescan = sum(v.size for v in m._variables[: var._index])
        assert _compute_var_offset(var, m) == rescan

    # and the offsets are the expected exclusive prefix sums of the sizes
    assert [_compute_var_offset(v, m) for v in (a, b, c, d)] == [0, 3, 4, 12]


def test_offset_stays_correct_when_variables_are_appended_after_a_lookup():
    """The prefix-sum table is cached. A lookup performed *before* more variables
    are declared must not poison later lookups — a stale table would silently
    misplace every coefficient, which is a wrong-answer bug, not a slow one."""
    m = dm.Model("growing")
    first = m.continuous("first", shape=(5,), lb=0.0, ub=1.0)
    assert _compute_var_offset(first, m) == 0  # populates the cache at n_vars == 1

    later = m.continuous("later", shape=(2,), lb=0.0, ub=1.0)
    last = m.continuous("last", lb=0.0, ub=1.0)
    assert _compute_var_offset(first, m) == 0
    assert _compute_var_offset(later, m) == 5
    assert _compute_var_offset(last, m) == 7

    m.minimize(last)
    m.subject_to(sum(first) + sum(later) + last <= 3)
    a = dense_A(extract_lp_data(m).A_eq)
    # one row, all eight structural columns present with coefficient 1
    assert a.shape[0] == 1
    assert np.array_equal(a[0, :8], np.ones(8))


def test_extraction_is_correct_on_the_wide_model():
    """The scaling test below is only meaningful if the fast path extracts the same
    matrix. Every row must carry exactly REFS_PER_ROW unit coefficients."""
    n = 200
    a = dense_A(extract_lp_data(_wide_model(n)).A_eq)
    structural = a[:, :n]
    assert structural.shape == (n // REFS_PER_ROW, n)
    assert np.array_equal(np.count_nonzero(structural, axis=1), np.full(a.shape[0], REFS_PER_ROW))
    assert set(np.unique(structural[structural != 0.0]).tolist()) == {1.0}


@pytest.mark.slow
def test_constraint_extraction_does_not_scale_quadratically_with_variable_count():
    """4x the variables is 4x the variable references, so extraction must cost ~4x.

    Under the rescan it cost ~16x, because each of the 4x references also had to
    scan a 4x longer prefix. Measured on this model shape (n = 2000 -> 8000):

        rescan     0.058 s -> 0.826 s   14.2x
        memoized   0.010 s -> 0.049 s    4.9x

    The assertion is a ratio rather than an absolute time so it does not encode this
    machine's speed. 9x separates the two regimes with margin either way.
    """
    _wide_model(200)  # warm imports / any first-call caches

    t0 = time.perf_counter()
    extract_lp_data(_wide_model(2000))
    t_small = time.perf_counter() - t0

    t0 = time.perf_counter()
    extract_lp_data(_wide_model(8000))
    t_large = time.perf_counter() - t0

    ratio = t_large / max(t_small, 1e-6)
    assert ratio < 9.0, (
        f"constraint extraction cost grew {ratio:.1f}x for 4x the variables "
        f"({t_small:.3f}s -> {t_large:.3f}s) — the per-reference offset rescan "
        "looks like it is back (measured 14.2x with it, 4.9x without)"
    )


@pytest.mark.parametrize("where", sorted(OFFSET_HELPERS))
def test_every_offset_helper_matches_the_rescan_it_replaced(where):
    """The rescan was duplicated in eight places; all of them now delegate to the
    memoized table. Each must return exactly what re-summing
    ``model._variables[: var._index]`` returned — a misplaced offset silently writes
    a coefficient into the wrong column, which is a wrong-answer bug."""
    fn = OFFSET_HELPERS[where]
    m = dm.Model("mixed")
    a = m.continuous("a", shape=(3,), lb=0.0, ub=1.0)
    b = m.continuous("b", lb=0.0, ub=1.0)
    c = m.continuous("c", shape=(4, 2), lb=0.0, ub=1.0)
    d = m.binary("d")

    for var in (a, b, c, d):
        rescan = sum(v.size for v in m._variables[: var._index])
        assert fn(var, m) == rescan, f"{where}: {fn(var, m)} != {rescan}"
    assert [fn(v, m) for v in (a, b, c, d)] == [0, 3, 4, 12]


@pytest.mark.slow
def test_convexity_affine_extraction_does_not_scale_quadratically():
    """The same scaling guarantee for the convexity classifier's affine walk, which
    is where the next >265 s of a 30 s-limit solve went once extraction was fixed.

    Measured on this model shape (n = 2000 -> 8000):

        rescan     0.048 s -> 0.787 s   16.3x
        memoized   0.001 s -> 0.006 s    4.1x
    """

    def _affine_pass(n):
        model = _wide_model(n)
        n_vars = sum(v.size for v in model._variables)
        t0 = time.perf_counter()
        hits = 0
        for con in model._constraints:
            if linear_context.extract_affine(con.body, model, n_vars) is not None:
                hits += 1
        assert hits == len(model._constraints), "the walk stopped being affine"
        return time.perf_counter() - t0

    _affine_pass(200)  # warm
    t_small = _affine_pass(2000)
    t_large = _affine_pass(8000)

    ratio = t_large / max(t_small, 1e-6)
    assert ratio < 9.0, (
        f"convexity affine extraction grew {ratio:.1f}x for 4x the variables "
        f"({t_small:.3f}s -> {t_large:.3f}s) — the per-reference offset rescan looks "
        "like it is back in linear_context"
    )
