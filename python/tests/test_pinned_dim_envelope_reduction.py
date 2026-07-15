"""Pinned-dim reduction of the vertex-hull separation LPs (#632 perf follow-up).

Branching / FBBT can set ``lb[d] == ub[d]`` exactly (a variable is *pinned*). The
exact multilinear (:mod:`discopt._jax.multilinear_separation`) and edge-concave
(:mod:`discopt._jax.edge_concave`) separators enumerate the ``2^n`` box vertices
of the term. A pinned dim's coordinate is constant on the box, so it emits
duplicate vertex columns and a redundant equality row → the vertex-hull LP is
degenerate → the in-house Rust simplex cycles to its pivot cap and falls back to
the POUNCE IPM (measured on nvs09: ~0.1–2.8 s/LP, ~41 % of solve wall).

The fix drops the pinned dims from the vertex enumeration, solving a
non-degenerate reduced LP, and returns slope 0 on the pinned dims. This suite
pins the contract that makes it safe:

* **slope is exactly 0 on every pinned dim** — a pinned var is fixed at its
  constant in the relaxation LP too, so it cannot carry an active gradient;
* **the separating value at ``x*`` is unchanged** — that value is the (unique)
  LP optimum, so it is invariant to the degenerate LP's facet freedom; it drives
  the emit decision, so cuts are emitted in exactly the same situations;
* **the cut is sound** — a valid under/over-estimator at every *full* box vertex
  (hence everywhere), which is what the intercept recompute guarantees.

The exact free-dim *slope* is intentionally NOT asserted equal to the full-LP
slope: the hull LP is degenerate at a pinned box, so multiple supporting facets
are optimal and the reduced (non-degenerate) LP may select a different — equally
valid — one. Both separators document that the cut is "rigorously valid for ANY
slope" because the intercept is recomputed to the vertices; this suite asserts
that documented property, plus solve-level node/objective neutrality is covered
by the nvs09 (multilinear) / nvs06 (edge-concave) probes.
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# multilinear_separation
# ---------------------------------------------------------------------------
def _ml_full_solve(lb, ub, x_star, maximize):
    """Pre-fix full (pinned-inclusive) vertex-hull solve, for reference."""
    from discopt._jax.multilinear_separation import _solve_envelope

    n = lb.shape[0]
    verts = np.array(list(product(*[(float(lb[d]), float(ub[d])) for d in range(n)])))
    fv = np.prod(verts, axis=1)
    return _solve_envelope(verts, fv, np.clip(x_star, lb, ub), maximize=maximize)


@pytest.mark.parametrize("seed", range(12))
def test_multilinear_pinned_reduction_contract(seed):
    from discopt._jax import multilinear_separation as ms

    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(3, 8))
    npin = int(rng.integers(1, n - 1))  # >=1 pinned, >=2 free
    lb = rng.uniform(-2.0, 1.0, n)
    ub = lb + rng.uniform(0.2, 3.0, n)
    for d in rng.choice(n, size=npin, replace=False):
        ub[d] = lb[d]  # exact pin
    pinned = lb == ub
    x_star = np.clip(rng.uniform(-2.0, 3.0, n), lb, ub)

    verts_full = np.array(list(product(*[(float(lb[d]), float(ub[d])) for d in range(n)])))
    fv_full = np.prod(verts_full, axis=1)

    for maximize, sense in ((False, "under"), (True, "over")):
        ref = _ml_full_solve(lb, ub, x_star, maximize)
        # Force a violated w_star so a cut is emitted (below/above the envelope).
        env_ref = ref[0]
        w_star = env_ref - 1.0 if not maximize else env_ref + 1.0
        cuts = ms._separate_multilinear_envelope_uncached(lb, ub, x_star, w_star)
        emitted = [c for c in cuts if c.sense == sense]
        assert emitted, f"expected a {sense} cut (seed={seed})"
        cut = emitted[0]
        # 1. slope is exactly 0 on every pinned dim.
        assert np.all(cut.a[pinned] == 0.0)
        # 2. separating value at x* matches the full-LP optimum (unique).
        sep_val = float(cut.a @ x_star + cut.b)
        assert sep_val == pytest.approx(env_ref, abs=1e-9)
        # 3. soundness: valid under/over-estimator at every full box vertex.
        lhs = verts_full @ cut.a + cut.b
        if sense == "under":
            assert np.all(lhs <= fv_full + 1e-9)
        else:
            assert np.all(lhs >= fv_full - 1e-9)


# ---------------------------------------------------------------------------
# edge_concave
# ---------------------------------------------------------------------------
def _ec_full_solve(block, lb, ub, x_star, maximize):
    from discopt._jax import edge_concave as ec
    from discopt._jax.edge_concave import _quad_values
    from discopt.solvers import SolveStatus

    solve_lp = ec._separation_lp_solver()
    n = len(block.var_idxs)
    idx = {v: k for k, v in enumerate(block.var_idxs)}
    verts = np.array(list(product(*[(float(lb[d]), float(ub[d])) for d in range(n)])))
    vals = _quad_values(block, verts, idx)
    m = verts.shape[0]
    xs = np.clip(np.asarray(x_star, float), lb[:n], ub[:n])
    a_eq = np.vstack([verts.T, np.ones(m)])
    res = solve_lp(
        -vals if maximize else vals, A_eq=a_eq, b_eq=np.append(xs, 1.0), bounds=[(0.0, np.inf)] * m
    )
    if res.status != SolveStatus.OPTIMAL or res.dual_values is None:
        return None
    duals = np.asarray(res.dual_values, float)
    A = -duals[:n] if maximize else duals[:n]
    resid = vals - verts @ A
    B = float(np.max(resid)) if maximize else float(np.min(resid))
    return float(A @ xs + B), verts, vals


@pytest.mark.parametrize("seed", range(12))
def test_edge_concave_pinned_reduction_contract(seed):
    from discopt._jax.edge_concave import EdgeConcaveQuadratic, separate_edge_concave_quadratic

    rng = np.random.default_rng(2000 + seed)
    n = int(rng.integers(3, 6))
    var_idxs = tuple(range(n))
    sq = {i: float(rng.uniform(0.1, 2.0)) for i in range(n)}  # convex -> under
    bilin = {}
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                bilin[(i, j)] = float(rng.uniform(-1.0, 1.0))
    lin = {i: float(rng.uniform(-1.0, 1.0)) for i in range(n)}
    block = EdgeConcaveQuadratic(var_idxs, sq, bilin, lin, float(rng.uniform(-1, 1)), "under")

    lb = rng.uniform(-2.0, 0.0, n)
    ub = lb + rng.uniform(0.3, 3.0, n)
    npin = int(rng.integers(1, n - 1))
    for d in rng.choice(n, size=npin, replace=False):
        ub[d] = lb[d]
    pinned = lb == ub
    x_star = np.clip(rng.uniform(-2.0, 3.0, n), lb, ub)

    ref = _ec_full_solve(block, lb, ub, x_star, maximize=False)
    assert ref is not None
    env_ref, verts_full, vals_full = ref
    q_star = env_ref - 1.0  # violate the underestimator so a cut is emitted
    out = separate_edge_concave_quadratic(block, lb, ub, x_star, q_star)
    assert out is not None, f"expected an under cut (seed={seed})"
    A, B = out
    # 1. slope exactly 0 on pinned dims.
    assert np.all(A[pinned] == 0.0)
    # 2. separating value at x* matches the full-LP optimum.
    xs = np.clip(x_star, lb[:n], ub[:n])
    assert float(A @ xs + B) == pytest.approx(env_ref, abs=1e-9)
    # 3. soundness: valid underestimator at every full box vertex.
    assert np.all(verts_full @ A + B <= vals_full + 1e-9)
