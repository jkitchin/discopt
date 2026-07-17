"""Regression tests for #689 (and the #280 primal gap that it blocked).

``reformulate_integer_bilinear`` binary-expands + big-M-linearizes integer-factor
products into an exact pure MILP, appending aux columns after the originals. A
caller's ``initial_solution`` is built over the ORIGINAL variables, so after the
lift it no longer matches the lifted width and ``_solve_milp_simplex``'s
``size == n_orig`` guard dropped it *silently* — discarding even a known-optimal
incumbent (#689).

``extend_initial_point`` reconstructs every lifted column from its exact
definition (the expansion bits of ``x = lo + sum 2^k e_k``, and each big-M product
``v = e * other``), so the seed survives the lift. That in turn is what lets the
MILP path improve an assignment-structured incumbent with a one-hot swap, which
is only well defined on the pre-lift model (#280).

These tests are synthetic and corpus-free so they run in CI.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.integer_product_reform import (
    extend_initial_point,
    reformulate_integer_bilinear,
)
from discopt._jax.nlp_evaluator import cached_evaluator
from discopt._jax.primal_heuristics import _check_constraint_feasibility


def _partition_model(N, K, per, edges):
    """min within-partition edge weight; each node in one partition, balanced.

    The same assignment-structured shape as the ``graphpart_*`` family, built to
    match how the ``.nl`` reader actually encodes it: **scalar** decision columns
    declared ``INTEGER`` over ``[0, 1]`` (not ``BINARY``). That distinction is
    load-bearing here — ``_int_factor_range`` excludes declared binaries, so a
    ``m.binary(...)`` model is never lifted and would not exercise this path at
    all, while the real family is.
    """
    m = dm.Model("gp")
    x = [[m.integer(f"x_{i}_{k}", lb=0, ub=1) for k in range(K)] for i in range(N)]
    for i in range(N):
        m.subject_to(dm.sum(x[i][k] for k in range(K)) == 1, name=f"assign_{i}")
    for k in range(K):
        m.subject_to(dm.sum(x[i][k] for i in range(N)) == per, name=f"bal_{k}")
    m.minimize(dm.sum(w * dm.sum(x[i][k] * x[j][k] for k in range(K)) for (i, j, w) in edges))
    return m, x


def _edges(N, seed=0):
    rng = np.random.default_rng(seed)
    return [(i, j, float(rng.integers(1, 9))) for i in range(N) for j in range(i + 1, N)]


def _flat_assignment(N, K, assign):
    xf = np.zeros(N * K, dtype=np.float64)
    for i, k in enumerate(assign):
        xf[i * K + k] = 1.0
    return xf


def _lifted():
    m, _ = _partition_model(6, 3, 2, _edges(6, seed=1))
    rm = reformulate_integer_bilinear(m)
    return m, rm


def test_lift_records_reconstruction_metadata():
    """The reform must record what it needs to rebuild the lifted columns."""
    m, rm = _lifted()
    assert len(rm._variables) > len(m._variables), "expected the reform to lift"
    assert rm._ipx_n_orig_flat == sum(int(v.size) for v in m._variables)
    assert rm._ipx_aux_spec, "no aux spec recorded — the seed cannot survive the lift"


def test_extended_point_is_exactly_reconstructed():
    """The reconstructed point must satisfy every row of the LIFTED model and
    reproduce the original objective — the guarantee the seed rests on.

    This is the #689 defect: without the extension the point is the wrong width
    and gets dropped; with a *wrong* extension the aux rows would be violated.
    """
    m, rm = _lifted()
    x0 = _flat_assignment(6, 3, [0, 0, 1, 1, 2, 2])

    xe = extend_initial_point(rm, x0)
    assert xe is not None
    assert xe.size == sum(int(v.size) for v in rm._variables)
    assert np.allclose(xe[: x0.size], x0), "the originals must be preserved verbatim"

    ev_lift, ev_orig = cached_evaluator(rm), cached_evaluator(m)
    assert _check_constraint_feasibility(ev_lift, xe), "reconstructed aux violate the lifted rows"
    assert float(ev_lift.evaluate_objective(xe)) == pytest.approx(
        float(ev_orig.evaluate_objective(x0)), abs=1e-9
    )


def test_extension_declines_rather_than_guesses():
    """A point the reform cannot reconstruct exactly must yield ``None`` (no seed),
    never a partial or invented one."""
    _, rm = _lifted()
    n0 = rm._ipx_n_orig_flat
    assert extend_initial_point(rm, np.zeros(n0 + 1)) is None  # wrong width
    assert extend_initial_point(rm, np.full(n0, np.nan)) is None  # non-finite
    # Fractional where a bit expansion needs an integer.
    assert extend_initial_point(rm, np.full(n0, 0.5)) is None
    # A model that was never lifted carries no metadata.
    assert extend_initial_point(dm.Model("bare"), np.zeros(3)) is None


def test_seed_reaches_the_engine_at_lifted_width(monkeypatch):
    """The caller's ``initial_solution`` must arrive at the MILP driver, lifted.

    This is the #689 defect precisely. The seed is built over the ORIGINAL
    columns; the reform appends aux, so pre-fix the driver's ``size == n_orig``
    guard saw a short vector and set ``initial_incumbent=None`` — silently. A
    weaker end-to-end assertion (``objective <= optimum``) does not discriminate:
    the engine often reaches the optimum unaided on a small instance. So assert
    on what actually regressed — that a seed is handed to the driver, at the
    lifted width, and that it is the point we asked for.
    """
    import discopt._rust as rust

    seen: list = []
    real = rust.solve_milp_py

    def spy(*args, **kwargs):
        seen.append(kwargs.get("initial_incumbent"))
        return real(*args, **kwargs)

    monkeypatch.setattr(rust, "solve_milp_py", spy)

    m, _ = _partition_model(6, 3, 2, _edges(6, seed=1))
    x0 = _flat_assignment(6, 3, [0, 0, 1, 1, 2, 2])
    m.solve(time_limit=10, initial_solution={v: float(x0[i]) for i, v in enumerate(m._variables)})

    seeds = [s for s in seen if s is not None]
    assert seeds, "the initial_solution never reached the MILP driver (silently dropped)"
    seed = np.asarray(seeds[0], dtype=np.float64)
    assert seed.size > x0.size, "seed was not lifted to the reformulated width"
    assert np.allclose(seed[: x0.size], x0), "seed's original columns were not preserved"
