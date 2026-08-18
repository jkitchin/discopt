"""Regression test for #1061: root presolve computed bounds and threw them away.

The Rust orchestrator accumulates every pass's tightening in ``ctx.bounds`` and
deliberately does not write it into the model repr it returns -- the NOTE closing
``presolve::orchestrator::run`` explains why (mutating the repr's declared bounds
can flip an inactive bound to active and change LP duals). It hands the box back
to the caller in ``PresolveResult::bounds`` instead.

``propagate_bounds_to_model`` is the caller whose entire job is to deliver that
box to the Python ``Model``, and it read only the repr. So the box was computed
on every solve and discarded on every solve.

Measured on the MINLPLib ``syn``/``rsyn`` class, where the effect is total: root
presolve reports ``n_tightened=0`` and terminates ``NoProgress`` while leaving
83/169/122/99 continuous variables with ``ub=+inf`` -- and ``stats['fbbt']['ub']``,
from that same run, holds a finite upper bound for every one of them (87 of 130
entries strictly tighter on ``syn40m``, 91 of 161 on ``rsyn0805m``).

The tests below reproduce the mechanism on a two-variable big-M row, which is the
structure that makes the bound derivable (``x - 40*y <= 0`` with ``y`` binary
gives ``x <= 40`` by interval arithmetic alone), without depending on any
instance.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.presolve_pipeline import propagate_bounds_to_model, run_root_presolve
from discopt._rust import model_to_repr

INF = 1e19


def _bigm_model(ub=np.inf):
    """``x <= 40*y`` with ``y`` binary: ``ub(x) = 40`` follows from the row alone."""
    m = dm.Model("bigm_1061")
    x = m.continuous("x", 1, lb=0.0, ub=ub)
    y = m.binary("y", 1)
    m.subject_to(x[0] - 40.0 * y[0] <= 0, name="link")
    m.subject_to(x[0] >= 0.5, name="demand")
    m.minimize(-x[0] + 2.0 * y[0])
    return m


def _flat_ub(model):
    return np.concatenate(
        [np.asarray(v.ub, dtype=np.float64).reshape(-1) for v in model._variables]
    )


@pytest.mark.smoke
def test_presolve_derives_a_finite_bound_that_the_repr_never_carries():
    """The premise: the box exists in stats and is absent from the repr."""
    model = _bigm_model()
    # §6: without a genuinely infinite declared bound this test proves nothing.
    assert np.any(_flat_ub(model) >= INF), "model has no unbounded variable to tighten"

    repr_ = model_to_repr(model, getattr(model, "_builder", None))
    new_repr, stats = run_root_presolve(repr_, eliminate=True, fbbt=True, time_limit_ms=10_000)

    repr_ub = np.asarray(
        [float(v) for bi in range(new_repr.n_var_blocks) for v in new_repr.var_ub(bi)]
    )
    stats_ub = np.asarray(stats["bounds_hi"], dtype=np.float64)

    assert np.any(repr_ub >= INF), (
        "the repr already carries a finite bound — the Rust orchestrator's "
        "documented behaviour changed and this test no longer tests anything"
    )
    assert np.all(stats_ub < INF), (
        f"presolve did not derive a finite box for the big-M row: {stats_ub}"
    )


@pytest.mark.smoke
def test_propagation_delivers_the_box_only_when_handed_the_stats():
    """The defect, and the fix, in one comparison."""
    _, stats = run_root_presolve(
        model_to_repr(_bigm_model(), None), eliminate=True, fbbt=True, time_limit_ms=10_000
    )

    # Historical behaviour: repr only. The bound stays infinite.
    without = _bigm_model()
    repr_w, stats_w = run_root_presolve(
        model_to_repr(without, getattr(without, "_builder", None)),
        eliminate=True,
        fbbt=True,
        time_limit_ms=10_000,
    )
    propagate_bounds_to_model(without, repr_w)
    assert np.any(_flat_ub(without) >= INF), (
        "the repr-only path tightened the bound — the defect this test guards "
        "against cannot be reproduced, so the fix below is unverifiable"
    )

    # With the stats: the derived bound lands on the model.
    with_stats = _bigm_model()
    repr_s, stats_s = run_root_presolve(
        model_to_repr(with_stats, getattr(with_stats, "_builder", None)),
        eliminate=True,
        fbbt=True,
        time_limit_ms=10_000,
    )
    n = propagate_bounds_to_model(with_stats, repr_s, stats_s)
    ub = _flat_ub(with_stats)
    assert np.all(ub < INF), f"bound still unbounded after propagation: {ub}"
    assert n > 0, "propagation reported no tightening despite changing a bound"
    # The row implies exactly x <= 40; anything looser means the box was not the
    # one presolve derived, anything tighter means it cut the feasible region.
    assert ub[0] == pytest.approx(40.0), f"expected ub(x)=40 from the big-M row, got {ub[0]}"


@pytest.mark.smoke
def test_propagation_never_loosens_a_declared_bound():
    """Intersection only: a model that already declares a tighter box keeps it."""
    model = _bigm_model(ub=5.0)
    repr_, stats = run_root_presolve(
        model_to_repr(model, getattr(model, "_builder", None)),
        eliminate=True,
        fbbt=True,
        time_limit_ms=10_000,
    )
    propagate_bounds_to_model(model, repr_, stats)
    assert _flat_ub(model)[0] <= 5.0 + 1e-12, "propagation loosened a declared bound"


@pytest.mark.smoke
def test_optimality_derived_bounds_are_refused_not_silently_skipped():
    """Cutoff-derived bounds must never become declared bounds (CLAUDE.md §3)."""
    model = _bigm_model()
    repr_, stats = run_root_presolve(
        model_to_repr(model, getattr(model, "_builder", None)),
        eliminate=True,
        fbbt=True,
        time_limit_ms=10_000,
    )
    stats["bounds_optimality_derived"] = True
    with pytest.raises(ValueError, match="optimality-derived"):
        propagate_bounds_to_model(model, repr_, stats)

    # And the default pass list must not be flagged that way, or the fix would
    # be permanently inert.
    _, clean = run_root_presolve(
        model_to_repr(_bigm_model(), None), eliminate=True, fbbt=True, time_limit_ms=10_000
    )
    assert clean["bounds_optimality_derived"] is False


@pytest.mark.smoke
def test_a_misaligned_box_is_dropped_rather_than_misapplied():
    """A wrong-length array would stamp one block's interval onto another."""
    model = _bigm_model()
    repr_, stats = run_root_presolve(
        model_to_repr(model, getattr(model, "_builder", None)),
        eliminate=True,
        fbbt=True,
        time_limit_ms=10_000,
    )
    stats = dict(stats)
    stats["bounds_lo"] = np.asarray(stats["bounds_lo"], dtype=np.float64)[:-1]
    stats["bounds_hi"] = np.asarray(stats["bounds_hi"], dtype=np.float64)[:-1]
    propagate_bounds_to_model(model, repr_, stats)
    # Falls back to the repr-only box, which leaves the bound infinite.
    assert np.any(_flat_ub(model) >= INF), "a misaligned box was applied anyway"
