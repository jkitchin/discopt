"""Tests for the uniform factorable relaxation engine (issue #632).

Covers, for the ``build_uniform_relaxation`` engine that runs ALONGSIDE the
federation:

* per-atom-kind SOUNDNESS on controlled atoms — the engine's root LP bound never
  overshoots the true box optimum (a bound above the true optimum is unsound);
* AVM composition — inner-atom bounds flow into the outer envelope, recovering a
  finite bound on a composite the separable path drops;
* corpus COVERAGE — 0 fallbacks across the 62 vendored ``.nl`` instances;
* feasible-point soundness — the lifted true point is never cut.

Every bound is computed with discopt's in-house Rust simplex (``backend="simplex"``).
"""

from __future__ import annotations

import math
from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt._jax.uniform_relax import (
    ENVELOPE_LIBRARY,
    build_uniform_relaxation,
    relaxation_report,
)

_NL_DIR = Path(__file__).resolve().parent / "data" / "minlplib_nl"


def _root_bound(model):
    rel = build_uniform_relaxation(model)
    res = rel.model.solve(backend="simplex")
    return (float(res.bound) if res.bound is not None else None), res.status, rel


# --------------------------------------------------------------------------- #
# Per-atom-kind soundness: root bound <= true box minimum
# --------------------------------------------------------------------------- #
@pytest.mark.unit
@pytest.mark.parametrize(
    "build_fn, true_min",
    [
        # convex exp
        (lambda m, x: dm.exp(x), math.exp(-1.0)),  # [-1,2]
        # convex even power
        (lambda m, x: x**2, 0.0),
        # concave sqrt (min at endpoint)
        (lambda m, x: dm.sqrt(x + 2.0), math.sqrt(1.0)),  # x in [-1,2] -> arg[1,4]
        # concave log
        (lambda m, x: dm.log(x + 3.0), math.log(2.0)),  # arg[2,5]
        # convex reciprocal on positive box (shift base positive)
        (lambda m, x: 1.0 / (x + 3.0), 1.0 / 5.0),  # arg[2,5]
        # abs (convex, nonsmooth)
        (lambda m, x: dm.abs(x), 0.0),
        # composite univariate: log(x+3)**2  (AVM: inner ln bound flows into square).
        # arg=x+3 in [2,5] -> ln in [ln2,ln5]>0 -> (ln)^2 min = ln(2)^2.
        (lambda m, x: dm.log(x + 3.0) ** 2, math.log(2.0) ** 2),
    ],
)
def test_univariate_atom_bound_is_sound(build_fn, true_min):
    m = Model()
    x = m.continuous("x", lb=-1.0, ub=2.0)
    m.minimize(build_fn(m, x))
    bound, status, _ = _root_bound(m)
    assert status == "optimal"
    assert bound is not None
    # SOUND: the relaxation lower bound must not exceed the true minimum.
    assert bound <= true_min + 1e-6, f"overshoot: bound {bound} > true {true_min}"


@pytest.mark.unit
def test_bilinear_mccormick_is_tight_and_sound():
    m = Model()
    x = m.continuous("x", lb=1.0, ub=2.0)
    y = m.continuous("y", lb=1.0, ub=3.0)
    m.minimize(x * y)
    bound, status, _ = _root_bound(m)
    assert status == "optimal"
    # McCormick is exact for a single bilinear term on a box: min x*y = 1.
    assert bound <= 1.0 + 1e-6
    assert bound == pytest.approx(1.0, abs=1e-6)


@pytest.mark.unit
def test_multilinear_product_is_sound():
    m = Model()
    xs = [m.continuous(f"x{i}", lb=1.0, ub=2.0) for i in range(3)]
    m.minimize(xs[0] * xs[1] * xs[2])
    bound, status, _ = _root_bound(m)
    assert status == "optimal"
    assert bound <= 1.0 + 1e-6  # true min = 1


@pytest.mark.unit
def test_ratio_atom_is_sound():
    m = Model()
    x = m.continuous("x", lb=1.0, ub=4.0)
    y = m.continuous("y", lb=2.0, ub=5.0)
    m.minimize(x / y)
    bound, status, _ = _root_bound(m)
    assert status == "optimal"
    assert bound <= (1.0 / 5.0) + 1e-6  # true min x/y = 1/5


@pytest.mark.unit
def test_maximize_sense_is_negated_soundly():
    # max log(x) over [1,4]; engine minimizes -log(x), bound is LB on -max.
    m = Model()
    x = m.continuous("x", lb=1.0, ub=4.0)
    m.maximize(dm.log(x))
    bound, status, rel = _root_bound(m)
    assert status == "optimal"
    assert rel.obj_sense_sign == -1.0
    # -bound is an UPPER bound on max log(x)=log 4; so bound <= -log4.
    assert bound <= -math.log(4.0) + 1e-6


# --------------------------------------------------------------------------- #
# AVM composition: inner bounds flow up, composite gets a finite bound
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_avm_composition_recovers_finite_composite_bound():
    # nvs09-shape: (ln(x-2))**2 + (ln(10-x))**2 on [3,9] — the federation drops
    # the objective; the AVM composition must produce a FINITE sound bound.
    m = Model()
    x = m.continuous("x", lb=3.0, ub=9.0)
    m.minimize(dm.log(x - 2.0) ** 2 + dm.log(10.0 - x) ** 2)
    bound, status, rel = _root_bound(m)
    assert status == "optimal"
    assert bound is not None and math.isfinite(bound)
    # True objective is >= 0 (sum of squares); the sound relaxation floor is >= a
    # value below the true optimum but must be finite (not the fallback 0/None).
    rep = relaxation_report(m)
    assert rep["fallbacks"] == 0
    assert rep["n_atoms"] >= 4  # two ln atoms + two square atoms


@pytest.mark.unit
def test_composite_of_bilinear_inside_exp_is_sound():
    # exp(x*y): x*y is McCormick-relaxed into an aux; exp is 1-D over that aux box.
    m = Model()
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(dm.exp(x * y))
    bound, status, _ = _root_bound(m)
    assert status == "optimal"
    assert bound <= math.exp(0.0) + 1e-6  # true min exp(0) = 1


@pytest.mark.unit
def test_shared_subexpression_is_relaxed_once_cse():
    m = Model()
    x = m.continuous("x", lb=1.0, ub=3.0)
    g = dm.log(x)
    m.minimize(g * g + g)  # log(x) appears twice; one aux by CSE
    rel = build_uniform_relaxation(m)
    # 2 nonlinear atoms: the shared ln(x) and the (ln)^2 square (product/pow).
    assert rel.n_aux <= 3


# --------------------------------------------------------------------------- #
# Coverage: every atom kind has a builder; 0 fallbacks
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_envelope_library_covers_every_kind():
    for kind in ("univariate_call", "power", "product", "ratio", "multivar", "opaque"):
        assert kind in ENVELOPE_LIBRARY


@pytest.mark.slow
def test_corpus_zero_fallbacks():
    from discopt.modeling.core import from_nl

    files = sorted(_NL_DIR.glob("*.nl"))
    assert files, "no vendored .nl instances found"
    fallbacks = []
    for f in files:
        try:
            model = from_nl(str(f))
        except Exception:
            continue  # unbuildable model is a modeling-layer issue, not the engine
        rep = relaxation_report(model)
        if rep["fallbacks"] != 0:
            fallbacks.append(f.stem)
    assert fallbacks == [], f"engine fell back on: {fallbacks}"


# --------------------------------------------------------------------------- #
# Feasible-point soundness: the lifted true point is never cut
# --------------------------------------------------------------------------- #
def _sample_no_cut(model, n=400, seed=0, tol=1e-6):
    """Sample points in the box, set every aux to its TRUE value, assert no row cut.

    Uses the builder's ``track_aux_exprs`` map: each aux column carries the exact
    modeling Expression it represents (node value / relaxed power / McCormick
    partial product), so the lifted true point ``(x, w=f(x))`` is exact — and a
    sound outer relaxation must satisfy every row at that point.
    """
    from discopt._jax import uniform_relax as ur
    from discopt._jax.canonical_expr import canonicalize
    from discopt._jax.convexity.interval import Interval
    from discopt._jax.convexity.interval_eval import evaluate_interval
    from discopt._jax.model_utils import flat_variable_bounds

    flat_lb, flat_ub = flat_variable_bounds(model)
    dag = canonicalize(model)
    ctx = ur._Builder(model, flat_lb, flat_ub, track_aux_exprs=True)
    roots = ([dag.objective] if dag.objective is not None else []) + list(dag.constraints)
    for r in roots:
        ctx.rep(r)

    def eval_at(expr, xv):
        box = {}
        off = 0
        for v in model._variables:
            size = int(v.size)
            shape = tuple(getattr(v, "shape", ()) or ())
            pt = np.asarray(xv[off : off + size], dtype=np.float64).reshape(shape)
            box[v] = Interval(pt, pt)
            off += size
        return float(np.asarray(evaluate_interval(expr, model, box).lo))

    rng = np.random.default_rng(seed)
    n_cols = len(ctx.col_lb)
    fb = np.where(np.isfinite(flat_lb), flat_lb, -5.0)
    fu = np.where(np.isfinite(flat_ub), flat_ub, 5.0)
    for _ in range(n):
        xv = fb + rng.random(ctx.n_orig) * (fu - fb)
        z = np.zeros(n_cols)
        z[: ctx.n_orig] = xv
        for j in sorted(ctx.aux_expr):
            z[j] = eval_at(ctx.aux_expr[j], xv)
        for coeffs, rhs in ctx.rows:
            lhs = sum(c * z[jj] for jj, c in coeffs.items())
            assert lhs <= rhs + tol, f"feasible point cut: {lhs} > {rhs}"


@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_feasible_points_not_cut(seed):
    m = Model()
    x = m.continuous("x", lb=0.5, ub=2.5)
    y = m.continuous("y", lb=0.5, ub=2.0)
    m.minimize(dm.exp(x * y) + dm.log(x + y) ** 2 + x / y)
    _sample_no_cut(m, seed=seed)


@pytest.mark.unit
@pytest.mark.parametrize("seed", [0, 1])
def test_feasible_points_not_cut_multilinear_and_powers(seed):
    m = Model()
    x = m.continuous("x", lb=1.0, ub=3.0)
    y = m.continuous("y", lb=1.0, ub=2.5)
    z = m.continuous("z", lb=0.5, ub=2.0)
    m.minimize(x * y * z + x**3 + dm.sqrt(x + y) + (x * y) ** 0.5)
    _sample_no_cut(m, seed=seed)


# --------------------------------------------------------------------------- #
# Per-model analysis cache (issue #632 EP1) — bound-neutral byte-identity gate
# --------------------------------------------------------------------------- #
def _fp(rel):
    from discopt._jax.claim_audit import relaxation_fingerprint

    return relaxation_fingerprint(rel.model)


@pytest.mark.unit
def test_ep1_cache_hot_rebuild_is_byte_identical():
    """A second (cache-hot) build of the SAME (model, box) must be byte-identical
    to the first. Guards that reading box-independent analysis through the pinned
    per-model cache never perturbs the emitted relaxation."""
    from discopt._jax.uniform_relax import _ANALYSIS_ATTR

    m = Model()
    x = m.continuous("x", lb=0.5, ub=2.5)
    y = m.continuous("y", lb=0.5, ub=2.0)
    m.minimize(dm.exp(x * y) + dm.log(x + y) ** 2 + x / y)
    m.subject_to(x * y + dm.sqrt(x + y) <= 6.0)

    from discopt._jax.model_utils import flat_variable_bounds

    lb, ub = flat_variable_bounds(m)
    assert _ANALYSIS_ATTR not in m.__dict__  # cold: no cache yet
    r1 = build_uniform_relaxation(m, box=(lb.copy(), ub.copy()))
    assert _ANALYSIS_ATTR in m.__dict__  # cache pinned after first build
    r2 = build_uniform_relaxation(m, box=(lb.copy(), ub.copy()))
    assert _fp(r1) == _fp(r2), "cache-hot rebuild drifted from the cold build"

    # A shrunk child box (subset) rebuilt twice is also self-consistent (the
    # box-dependent enclosure/curvature caches key on the box).
    ub2 = ub.copy()
    ub2[0] = 0.5 * (lb[0] + ub[0])
    c1 = build_uniform_relaxation(m, box=(lb.copy(), ub2.copy()))
    c2 = build_uniform_relaxation(m, box=(lb.copy(), ub2.copy()))
    assert _fp(c1) == _fp(c2)


@pytest.mark.unit
def test_ep1_staleness_token_new_constraint_invalidates_cache():
    """Adding a constraint must invalidate the cache (staleness token changes) so
    the rebuilt relaxation reflects the new constraint's rows."""
    from discopt._jax.uniform_relax import _ANALYSIS_ATTR

    m = Model()
    x = m.continuous("x", lb=0.5, ub=2.5)
    y = m.continuous("y", lb=0.5, ub=2.0)
    m.minimize(dm.exp(x * y) + x / y)
    r1 = build_uniform_relaxation(m)
    tok1 = m.__dict__[_ANALYSIS_ATTR].token

    m.subject_to(x * y <= 3.0)  # mutate: one more constraint
    r2 = build_uniform_relaxation(m)
    tok2 = m.__dict__[_ANALYSIS_ATTR].token

    assert tok1 != tok2, "staleness token did not change after adding a constraint"
    assert _fp(r1) != _fp(r2), "cache was not invalidated: new constraint absent"
    # Row count grew (the new constraint added at least one row).
    assert r2.model._A_ub.shape[0] > r1.model._A_ub.shape[0]


@pytest.mark.unit
def test_ep1_staleness_token_new_objective_invalidates_cache():
    """Replacing the objective must invalidate the cache (id(_objective) changes)
    so a stale objective is never reused — a stale objective would be unsound."""
    from discopt._jax.uniform_relax import _ANALYSIS_ATTR

    m = Model()
    x = m.continuous("x", lb=0.5, ub=2.5)
    y = m.continuous("y", lb=0.5, ub=2.0)
    m.minimize(dm.exp(x * y))
    r1 = build_uniform_relaxation(m)
    tok1 = m.__dict__[_ANALYSIS_ATTR].token

    m.minimize(dm.log(x + y) ** 2 + x / y)  # new objective object
    r2 = build_uniform_relaxation(m)
    tok2 = m.__dict__[_ANALYSIS_ATTR].token

    assert tok1 != tok2, "staleness token did not change after a new objective"
    assert _fp(r1) != _fp(r2), "cache was not invalidated: stale objective reused"
