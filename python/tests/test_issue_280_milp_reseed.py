"""Regression tests for the #280 primal lever on the Rust MILP fast path.

On set-partition / assignment-structured models (``sum_k x[i,k] == 1``) a single
bit flip always breaks a one-hot row, so the engine's generic flip/dive
heuristics stall well above the optimum while the dual bound is already tight
(#280). The feasibility-preserving move is a *swap*, and it is only well defined
on the pre-lift model: permuting bits in the integer-bilinear-lifted space leaves
the expansion / big-M auxiliaries stale (genuinely infeasible).

``_one_hot_swap_reseed`` runs the assignment-aware swap over the ORIGINAL
variables and maps the improved point back through the lift's reconstruction spec
(#689 / #696), producing a lifted-width seed for a re-entry of the MILP engine on
an uncertified ``feasible``. It is purely primal: the seed is only ever handed to
the Rust driver as an ``initial_incumbent`` (re-validated + re-scored there) and
the re-entry adoption gate re-verifies the returned point, so a worse or invalid
seed can only cost search, never the certificate.

These tests are synthetic and corpus-free so they run in CI. They pin: the reform
stashing the pre-lift model, the swap-reseed strictly improving a stalled
assignment incumbent at the lifted width, and the no-op gating (maximize / not
lifted / no budget) that keeps the move from firing where it does not apply.
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
from discopt.solver import _one_hot_swap_reseed

# A weighted K4 where the balanced split {0,1}|{2,3} costs 2 but {0,2}|{1,3} costs
# 20 — one swap of nodes 1 and 2 turns the bad split optimal.
_EDGES = [(0, 2, 10.0), (1, 3, 10.0), (0, 1, 1.0), (2, 3, 1.0), (0, 3, 1.0), (1, 2, 1.0)]


def _partition_model(N, K, per, edges, *, maximize=False):
    """min (or max) within-partition edge weight; each node in one partition,
    balanced. Built the way the ``.nl`` reader encodes the real ``graphpart_*``
    family: **scalar** decision columns declared ``INTEGER`` over ``[0, 1]`` (not
    ``BINARY``) — ``_int_factor_range`` excludes declared binaries, so only this
    shape exercises the integer-bilinear lift (#689)."""
    m = dm.Model("gp")
    x = [[m.integer(f"x_{i}_{k}", lb=0, ub=1) for k in range(K)] for i in range(N)]
    for i in range(N):
        m.subject_to(dm.sum(x[i][k] for k in range(K)) == 1, name=f"assign_{i}")
    for k in range(K):
        m.subject_to(dm.sum(x[i][k] for i in range(N)) == per, name=f"bal_{k}")
    obj = dm.sum(w * dm.sum(x[i][k] * x[j][k] for k in range(K)) for (i, j, w) in edges)
    m.maximize(obj) if maximize else m.minimize(obj)
    return m, x


def _flat_assignment(N, K, assign):
    xf = np.zeros(N * K, dtype=np.float64)
    for i, k in enumerate(assign):
        xf[i * K + k] = 1.0
    return xf


@pytest.mark.smoke
def test_reform_stashes_source_model():
    """The lift must stash the pre-lift model so the swap can run over the
    originals (the metadata ``_one_hot_swap_reseed`` depends on)."""
    m, _ = _partition_model(4, 2, 2, _EDGES)
    rm = reformulate_integer_bilinear(m)
    assert len(rm._variables) > len(m._variables), "expected the reform to lift"
    assert getattr(rm, "_ipx_source_model", None) is m
    assert rm._ipx_n_orig_flat == sum(int(v.size) for v in m._variables)


@pytest.mark.smoke
def test_reseed_improves_stalled_assignment_incumbent():
    """From the bad split (cost 20) the swap-reseed returns a lifted-width seed at
    the optimum (cost 2), feasible on every lifted row."""
    m, _ = _partition_model(4, 2, 2, _EDGES)
    rm = reformulate_integer_bilinear(m)
    ev = cached_evaluator(rm)

    bad = extend_initial_point(rm, _flat_assignment(4, 2, [0, 1, 0, 1]))  # {0,2}|{1,3}
    assert bad is not None
    assert float(ev.evaluate_objective(bad)) == pytest.approx(20.0, abs=1e-9)

    seed = _one_hot_swap_reseed(rm, bad, budget=2.0)
    assert seed is not None, "swap-reseed should improve the stalled incumbent"
    # Lifted width — matches the MILP driver's structural column count (n_orig),
    # which is the guard the re-entry applies before handing it back.
    assert seed.size == sum(int(v.size) for v in rm._variables)
    assert float(ev.evaluate_objective(seed)) < 20.0 - 1e-9
    assert float(ev.evaluate_objective(seed)) == pytest.approx(2.0, abs=1e-6)
    # The reconstructed aux must satisfy the lifted rows (a wrong extension would
    # hand the engine a genuinely infeasible seed).
    assert _check_constraint_feasibility(ev, seed)


@pytest.mark.smoke
def test_reseed_gates_off_when_inapplicable():
    """Self-gates to ``None`` where the move does not apply — never a guess."""
    m, _ = _partition_model(4, 2, 2, _EDGES)
    rm = reformulate_integer_bilinear(m)
    bad = extend_initial_point(rm, _flat_assignment(4, 2, [0, 1, 0, 1]))

    # No time to run the search.
    assert _one_hot_swap_reseed(rm, bad, budget=0.1) is None

    # A model that was never lifted carries no ``_ipx_source_model``.
    bare = dm.Model("bare")
    xb = bare.integer("x", lb=0, ub=1)
    bare.minimize(xb)
    assert _one_hot_swap_reseed(bare, np.zeros(1), budget=2.0) is None

    # Maximize: the swap only searches the improving (decreasing) direction, so it
    # would drive the wrong way — skip rather than hand back a worse seed.
    mm, _ = _partition_model(4, 2, 2, _EDGES, maximize=True)
    rmm = reformulate_integer_bilinear(mm)
    bad_max = extend_initial_point(rmm, _flat_assignment(4, 2, [0, 0, 1, 1]))
    assert bad_max is not None
    assert _one_hot_swap_reseed(rmm, bad_max, budget=2.0) is None
