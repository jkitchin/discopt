"""EP4a bound-neutral gate: the multilinear facet cache is an exact memoization.

``separate_multilinear_envelope`` re-solves two tiny facet LPs per call; on
multilinear-heavy instances the same atom+box+query-point recurs across nodes and
OBBT probes. EP4a memoizes the result in a module-level dict keyed by the exact
float64 bytes of *every* input. This suite pins the byte-neutrality contract:

* a cache hit returns the byte-identical cut list a fresh (uncached) derivation
  produces, on ≥20 random boxes per arity (2 and 3);
* the key includes ``x_star`` — the supporting facet is selected at the query
  point, so a box-only key would be UNSOUND (a different point at the same box
  can have a different active facet); this suite exhibits such a box;
* a hit does not re-solve the facet LPs;
* the cap clears the dict wholesale (bounded memory).

Regime: bound-neutral (CLAUDE.md §5). Same facets ⇒ same cuts ⇒ same nodes.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt._jax import multilinear_separation as ms
from discopt._jax.multilinear_separation import (
    _separate_multilinear_envelope_uncached,
    separate_multilinear_envelope,
)

pytestmark = pytest.mark.filterwarnings("ignore")


def _cuts_byte_equal(a_cuts, b_cuts) -> None:
    """Assert two cut lists are byte-identical (sense, slope bytes, intercept)."""
    assert len(a_cuts) == len(b_cuts)
    for ca, cb in zip(a_cuts, b_cuts, strict=True):
        assert ca.sense == cb.sense
        assert ca.a.dtype == cb.a.dtype
        assert ca.a.tobytes() == cb.a.tobytes()  # exact float64 equality
        assert ca.b == cb.b or (np.isnan(ca.b) and np.isnan(cb.b))


@pytest.fixture(autouse=True)
def _clear_cache():
    ms._FACET_CACHE.clear()
    yield
    ms._FACET_CACHE.clear()


@pytest.mark.parametrize("arity", [2, 3])
def test_cache_hit_equals_fresh_derivation(arity):
    """Cache-hit facets == freshly derived facets on ≥20 random boxes per arity."""
    rng = np.random.default_rng(20250713 + arity)
    n_boxes = 25
    for _ in range(n_boxes):
        lb = rng.uniform(-3.0, 1.0, size=arity)
        ub = lb + rng.uniform(0.1, 4.0, size=arity)
        x_star = lb + rng.uniform(0.0, 1.0, size=arity) * (ub - lb)
        # Sweep w_star across/inside/outside the product range so both the
        # violated (cut emitted) and satisfied (empty) branches are exercised.
        fv_lo = float(np.min([np.prod(v) for v in _box_vertices(lb, ub)]))
        fv_hi = float(np.max([np.prod(v) for v in _box_vertices(lb, ub)]))
        for w_star in (fv_lo - 1.0, 0.5 * (fv_lo + fv_hi), fv_hi + 1.0):
            ms._FACET_CACHE.clear()
            fresh = _separate_multilinear_envelope_uncached(lb, ub, x_star, w_star)
            miss = separate_multilinear_envelope(lb, ub, x_star, w_star)  # populates
            hit = separate_multilinear_envelope(lb, ub, x_star, w_star)  # from cache
            _cuts_byte_equal(fresh, miss)
            _cuts_byte_equal(fresh, hit)


def _box_vertices(lb, ub):
    from itertools import product

    return [
        np.array(v) for v in product(*[(float(lb[d]), float(ub[d])) for d in range(lb.shape[0])])
    ]


def test_hit_does_not_resolve_lps(monkeypatch):
    """A cache hit skips the facet LPs entirely (no ``_solve_envelope`` call)."""
    lb = np.array([-1.0, 0.5, -2.0])
    ub = np.array([2.0, 3.0, 1.0])
    x_star = np.array([0.3, 1.4, -0.2])
    w_star = 0.0

    separate_multilinear_envelope(lb, ub, x_star, w_star)  # cold: populates cache

    def _boom(*a, **k):  # pragma: no cover - must never run on a hit
        raise AssertionError("cache hit must not re-solve the facet LP")

    monkeypatch.setattr(ms, "_solve_envelope", _boom)
    # Identical inputs -> served from cache, _solve_envelope untouched.
    hit = separate_multilinear_envelope(lb, ub, x_star, w_star)
    assert isinstance(hit, list)


def test_key_includes_x_star_not_box_only():
    """The facet is selected at ``x_star``; two points at the SAME box must key
    separately. Exhibit a box where the emitted slope differs by query point, so a
    box-only key would return a stale (wrong) facet."""
    lb = np.array([-1.0, -1.0])
    ub = np.array([1.0, 1.0])
    w_star = -5.0  # far below the convex envelope everywhere -> under cut emitted

    # Two query points in opposite corners of the bilinear box select different
    # active McCormick facets (slope (+1,+1) vs (-1,-1)).
    p1 = np.array([0.9, 0.9])
    p2 = np.array([-0.9, -0.9])
    c1 = separate_multilinear_envelope(lb, ub, p1, w_star)
    c2 = separate_multilinear_envelope(lb, ub, p2, w_star)
    assert c1 and c2
    # Distinct keys, distinct cached entries (no box-only collision).
    assert len(ms._FACET_CACHE) == 2
    # And the facets genuinely differ (this is why box-only keying is unsound).
    assert c1[0].a.tobytes() != c2[0].a.tobytes()


def test_cache_cap_clears(monkeypatch):
    """At the cap the dict clears wholesale (bounded memory, no unbounded growth)."""
    monkeypatch.setattr(ms, "_FACET_CACHE_CAP", 4)
    ms._FACET_CACHE.clear()
    lb = np.array([-1.0, 0.0])
    ub = np.array([1.0, 2.0])
    rng = np.random.default_rng(7)
    sizes = []
    for _ in range(12):
        x_star = lb + rng.uniform(0.0, 1.0, size=2) * (ub - lb)
        separate_multilinear_envelope(lb, ub, x_star, 0.0)
        sizes.append(len(ms._FACET_CACHE))
    assert max(sizes) <= 4  # never exceeds the cap
    assert min(sizes) <= 1  # cleared at least once
