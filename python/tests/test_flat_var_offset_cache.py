"""Regression tests for the memoized flat-variable-offset table (issue #654).

``Model._flat_var_offset`` is the single source of truth for a variable's flat
start index in the stacked ``x`` vector. Before #654 every offset resolution
summed ``model._variables[: var._index]`` from scratch — O(n) per call — and the
relaxation / AD / term-classifier builds resolve one offset per variable leaf per
term, so the build was O(n·terms). That quadratic was the dominant *uninterruptible*
pre-B&B root-setup cost that made ``solve(time_limit=T)`` overrun its budget by
2–13× on large factorable models (super3t, sonet22v4; sub-site #507).

The fix memoizes an exclusive prefix-sum table (rebuilt only when the append-only
variable list grows), making each lookup O(1) and the whole build O(n + terms).
It must be a *pure speedup*: the offset returned is byte-for-byte the summed-slice
value, so no relaxation, cut, or bound can change (CLAUDE.md §5, bound-neutral).

These tests are corpus-free and deterministic:

* offset equivalence vs. the naive summation across mixed scalar/vector/matrix
  variables (the bound-neutrality guarantee),
* the classifier produces identical terms whether or not the cache is warm,
* the cache is built once and reused (the O(1)-lookup contract), and
* the cache self-invalidates when the model grows (no stale offsets).
"""

from __future__ import annotations

import sys
import time

import discopt.modeling as dm
import pytest
from discopt._jax.term_classifier import _compute_var_offset, classify_nonlinear_terms


def _naive_offset(var, model) -> int:
    """The pre-#654 summed-slice offset, kept as an independent oracle."""
    offset = 0
    for v in model._variables[: var._index]:
        offset += v.size
    return offset


def _mixed_model() -> dm.Model:
    m = dm.Model()
    m.continuous("a", lb=0.0, ub=1.0)  # scalar   -> offset 0
    m.continuous("b", shape=5, lb=0.0, ub=1.0)  # vector   -> offset 1
    m.continuous("c", shape=(3, 4), lb=0.0, ub=1.0)  # matrix   -> offset 6
    m.continuous("d", lb=0.0, ub=1.0)  # scalar   -> offset 18
    return m


def test_offset_matches_naive_summation():
    """Every variable's cached offset equals the from-scratch summation."""
    m = _mixed_model()
    for v in m._variables:
        assert m._flat_var_offset(v) == _naive_offset(v, m)
        assert _compute_var_offset(v, m) == _naive_offset(v, m)
    # Spot-check the exact expected prefix sums for this layout.
    assert [m._flat_var_offset(v) for v in m._variables] == [0, 1, 6, 18]


def test_cache_built_once_and_reused():
    """The prefix table is materialized once and reused for every lookup."""
    m = _mixed_model()
    assert m._flat_var_offsets_cache is None  # lazy: not built until first use
    m._flat_var_offset(m._variables[0])
    table = m._flat_var_offsets_cache
    assert table is not None
    assert len(table) == len(m._variables) + 1  # exclusive prefix sum + total
    # Repeated lookups must not rebuild (the O(1) contract): same object.
    for v in m._variables:
        m._flat_var_offset(v)
    assert m._flat_var_offsets_cache is table


def test_cache_invalidates_on_growth():
    """Appending a variable rebuilds the table so offsets never go stale."""
    m = _mixed_model()
    m._flat_var_offset(m._variables[0])
    stale = m._flat_var_offsets_cache
    d2 = m.continuous("e", shape=2, lb=0.0, ub=1.0)  # grows the model
    # First lookup after growth must rebuild and return the correct new offset.
    assert m._flat_var_offset(d2) == _naive_offset(d2, m) == 19
    assert m._flat_var_offsets_cache is not stale
    assert len(m._flat_var_offsets_cache) == len(m._variables) + 1


def test_classification_bound_neutral_across_cache_state():
    """Classifier output is identical whether the offset cache is cold or warm."""

    # Bilinear chain over many scalar vars — exactly the pattern (products of
    # distinct scalars) whose per-term offset resolution drove the quadratic.
    def build():
        m = dm.Model()
        xs = [m.continuous(f"x{i}", lb=0.0, ub=1.0) for i in range(40)]
        obj = 0
        for i in range(len(xs) - 1):
            obj = obj + xs[i] * xs[i + 1]
        m.minimize(obj)
        return m

    cold = build()
    cold_terms = classify_nonlinear_terms(cold)  # builds the cache internally

    warm = build()
    # Warm the cache before classifying to exercise the reuse path.
    for v in warm._variables:
        warm._flat_var_offset(v)
    warm_terms = classify_nonlinear_terms(warm)

    assert sorted(cold_terms.bilinear) == sorted(warm_terms.bilinear)
    assert len(cold_terms.bilinear) == 39


def _build_bilinear_chain(n: int) -> dm.Model:
    m = dm.Model()
    xs = [m.continuous(f"x{i}", lb=0.0, ub=1.0) for i in range(n)]
    obj = 0
    for i in range(n - 1):
        obj = obj + xs[i] * xs[i + 1]
    m.minimize(obj)
    return m


def _best_classify_wall(n: int, reps: int = 3) -> float:
    best = float("inf")
    for _ in range(reps):
        m = _build_bilinear_chain(n)
        t0 = time.perf_counter()
        classify_nonlinear_terms(m)
        best = min(best, time.perf_counter() - t0)
    return best


def test_classifier_build_scales_linearly_not_quadratically():
    """Guard against reverting the memoized offset table (the #654 root cause).

    With the O(n·terms) summed-slice offset the classifier build was quadratic:
    quadrupling the variable/term count blew the wall up ~16×. With the O(1)
    cached lookup the build is linear, so a 4× problem costs ~4× wall. Assert the
    4× → <9× envelope: comfortably above the ~4× linear cost (plus overhead /
    CI noise) yet far below the ~16× a quadratic regression would produce.
    """
    # The additive chain builds a deep expression tree; keep the recursive
    # classifier walk from hitting the default limit at the larger size.
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 300_000))
    try:
        small = _best_classify_wall(600)
        large = _best_classify_wall(2400)  # 4× the variables and bilinear terms
    finally:
        sys.setrecursionlimit(old_limit)

    # Skip rather than flake if the machine is so fast the small run is in the
    # timer-noise floor (the ratio would be dominated by fixed overhead).
    if small < 1e-3:
        pytest.skip(f"classify baseline too fast to time reliably ({small * 1e3:.2f}ms)")
    ratio = large / small
    assert ratio < 9.0, (
        f"classifier build scaling looks quadratic (4× size -> {ratio:.1f}× wall); "
        "the memoized flat-offset table (issue #654) may have regressed"
    )
