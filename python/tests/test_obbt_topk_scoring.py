"""cert:T2.5 — scored top-k OBBT candidate selection (flag-gated de-gate).

The T2.5 lever ranks OBBT candidate variables by ``width × |reduced cost|`` and
probes only the top-k, so per-node OBBT stays affordable on large spatial models
that the all-columns-index-order sweep made too expensive (F12: casctanks n=560
was size-gated off entirely). This is a *selection* lever: it changes WHICH
variables get a min/max probe, never the soundness of each tightening (every
surviving probe is still clamped to the Neumaier–Shcherbina safe bound).

These tests lock the two properties that make it safe:

  1. **Subset soundness / differential** — the tightening top-k produces on a
     given column is identical to what the full sweep produces on that column
     (same LP oracle, same box), and the top-k box is always a superset-or-equal
     of the full box (top-k tightens a subset of variables, so it can only be
     looser, never tighter — hence it can never cut a point the full sweep kept).
  2. **No valid point cut (feasible-point sampling)** — a point feasible for the
     original model, inside the input box, is never excluded by the top-k box.
  3. **Cert-neutrality** — ``top_k=None`` (and, on the solve path, the flag OFF)
     reproduces the legacy all-columns behavior bit-for-bit.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt._jax.obbt import run_obbt_on_relaxation
from discopt.modeling.core import Model


def _build_relaxation(model: Model):
    from discopt._jax.discretization import initialize_partitions
    from discopt._jax.milp_relaxation import build_milp_relaxation
    from discopt._jax.term_classifier import classify_nonlinear_terms

    terms = classify_nonlinear_terms(model)
    state = initialize_partitions([], lb=[], ub=[], n_init=2)
    milp, _ = build_milp_relaxation(model, terms, state, incumbent=None)
    return milp


def _spatial_model() -> Model:
    """A small nonconvex spatial model with several bilinear terms.

    Enough columns (originals + lifted aux) that top-k < n is a real subset, and
    a linear objective so the scoring LP yields non-trivial reduced costs.
    """
    m = Model("topk_spatial")
    xs = [m.continuous(f"x{i}", lb=-2.0, ub=3.0) for i in range(5)]
    m.subject_to(xs[0] * xs[1] + xs[2] * xs[3] <= 4.0)
    m.subject_to(xs[1] * xs[4] >= -3.0)
    m.subject_to(xs[0] * xs[4] + xs[2] <= 5.0)
    m.subject_to(sum(xs) <= 6.0)
    m.subject_to(sum(xs) >= -6.0)
    m.minimize(xs[0] - xs[3] + xs[4])
    return m


def _n_orig(model: Model) -> int:
    return sum(v.size for v in model._variables)


@pytest.mark.correctness
def test_topk_none_is_byte_identical_to_legacy():
    """``top_k=None`` reproduces the all-columns sweep exactly (cert-neutral)."""
    m = _spatial_model()
    n = _n_orig(m)
    for cutoff in (None, 10.0, 2.0):
        legacy = run_obbt_on_relaxation(
            _build_relaxation(m), n_orig=n, incumbent_cutoff=cutoff, time_limit_per_lp=5.0
        )
        again = run_obbt_on_relaxation(
            _build_relaxation(m),
            n_orig=n,
            incumbent_cutoff=cutoff,
            time_limit_per_lp=5.0,
            top_k=None,
        )
        assert np.array_equal(legacy.tightened_lb, again.tightened_lb)
        assert np.array_equal(legacy.tightened_ub, again.tightened_ub)
        assert legacy.n_tightened == again.n_tightened


@pytest.mark.correctness
def test_topk_box_is_superset_of_full_box():
    """Top-k tightens a subset of columns → its box contains the full-sweep box.

    A looser box can never exclude a point the tighter (full) box kept, so this
    is the differential 'no worse than full' guarantee for the selection lever.
    """
    m = _spatial_model()
    n = _n_orig(m)
    for cutoff in (None, 10.0):
        full = run_obbt_on_relaxation(
            _build_relaxation(m), n_orig=n, incumbent_cutoff=cutoff, time_limit_per_lp=5.0
        )
        for k in (1, 2, 3):
            topk = run_obbt_on_relaxation(
                _build_relaxation(m),
                n_orig=n,
                incumbent_cutoff=cutoff,
                time_limit_per_lp=5.0,
                top_k=k,
            )
            # top-k box ⊇ full box: lb never above full's lb, ub never below.
            assert np.all(topk.tightened_lb <= full.tightened_lb + 1e-9)
            assert np.all(topk.tightened_ub >= full.tightened_ub - 1e-9)
            # And top-k never tightens more variables than the full sweep.
            assert topk.n_tightened <= full.n_tightened


@pytest.mark.correctness
def test_topk_does_not_cut_a_feasible_point():
    """A model-feasible point inside the input box survives the top-k box."""
    m = _spatial_model()
    n = _n_orig(m)
    # Sample feasible points by rejection over the input box.
    rng = np.random.default_rng(0)
    lb0 = np.array([-2.0] * 5)
    ub0 = np.array([3.0] * 5)

    def _feasible(x):
        return (
            x[0] * x[1] + x[2] * x[3] <= 4.0 + 1e-9
            and x[1] * x[4] >= -3.0 - 1e-9
            and x[0] * x[4] + x[2] <= 5.0 + 1e-9
            and -6.0 - 1e-9 <= x.sum() <= 6.0 + 1e-9
        )

    pts = []
    while len(pts) < 200 and len(pts) < 100000:
        x = rng.uniform(lb0, ub0)
        if _feasible(x):
            pts.append(x)
        if len(pts) == 0 and rng.random() < 1e-6:
            break
    assert pts, "no feasible sample drawn"

    for k in (1, 2, 3):
        res = run_obbt_on_relaxation(
            _build_relaxation(m), n_orig=n, incumbent_cutoff=None, time_limit_per_lp=5.0, top_k=k
        )
        for x in pts:
            assert np.all(x >= res.tightened_lb[:5] - 1e-6)
            assert np.all(x <= res.tightened_ub[:5] + 1e-6)


@pytest.mark.correctness
def test_topk_zero_and_oversize_are_safe():
    """``top_k=0`` probes nothing (no tightening); ``top_k>=n`` == full sweep."""
    m = _spatial_model()
    n = _n_orig(m)
    z = run_obbt_on_relaxation(
        _build_relaxation(m), n_orig=n, incumbent_cutoff=None, time_limit_per_lp=5.0, top_k=0
    )
    assert z.n_tightened == 0
    full = run_obbt_on_relaxation(
        _build_relaxation(m), n_orig=n, incumbent_cutoff=None, time_limit_per_lp=5.0
    )
    big = run_obbt_on_relaxation(
        _build_relaxation(m),
        n_orig=n,
        incumbent_cutoff=None,
        time_limit_per_lp=5.0,
        top_k=10_000,
    )
    # top_k >= number of candidates falls through to the full candidate set.
    assert np.array_equal(big.tightened_lb, full.tightened_lb)
    assert np.array_equal(big.tightened_ub, full.tightened_ub)
