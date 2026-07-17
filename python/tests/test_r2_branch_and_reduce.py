"""cert:T2.3 — root branch-and-reduce fixpoint (GRADUATED ``root_fixpoint``).

These tests pin the R2 soundness invariants (cert-gap-plan §14 T2.3):

  * **T2.4a marginal neutrality** — requesting node-LP marginals never changes the
    LP ``lower_bound``/``x`` (additive side-channel only).
  * **run_root_fixpoint tighten-only + oracle-sound** — every reduced box is a subset
    of the input box and its dual bound never crosses the known optimum.
  * **round-2 improvement** — on a synthetic bilinear model a second fixpoint round
    (re-deriving the envelope on the round-1-tightened box) tightens strictly more
    than a single round, so the loop (not a one-shot pass) is load-bearing.

(The T2.4 per-node ``reduce_node`` tests were removed with the
``DISCOPT_NODE_REDUCE`` flag in #581 — deprecated as graduated-gate net-negative.)
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import time

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.root_reduce import run_root_fixpoint


def _bilinear_model():
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.subject_to(x + y <= 6)
    m.minimize(x * y - 2 * x)
    return m


# --------------------------------------------------------------------------- #
# T2.4a — marginal neutrality                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_marginals_do_not_change_bound():
    m = _bilinear_model()
    r = MccormickLPRelaxer(m)
    lb = np.array([0.0, 0.0])
    ub = np.array([4.0, 4.0])
    r0 = r.solve_at_node(lb, ub, want_marginals=False)
    r1 = r.solve_at_node(lb, ub, want_marginals=True)
    assert r0.status == r1.status == "optimal"
    # Byte-identical bound + point whether or not marginals are requested.
    assert r0.lower_bound == r1.lower_bound
    np.testing.assert_array_equal(np.asarray(r0.x), np.asarray(r1.x))
    # Marginals populated on the incremental fast path.
    if r._inc is not None:
        assert r1.dual is not None
        assert r1.safe_bound is not None
        assert r1.reduced_costs is not None
        assert r1.reduced_costs.shape[0] == 2
        # safe_bound is the reported (NS-safe) LP bound.
        assert r1.safe_bound == r1.lower_bound


# (The reduce_node feasible-point-retention + DBBT tests were removed with the
# ``DISCOPT_NODE_REDUCE`` flag / ``discopt._jax.node_reduce`` module in #581. The
# root branch-and-reduce fixpoint below — the GRADUATED ``root_fixpoint`` path —
# retains its own tighten-only + oracle-soundness coverage.)


# --------------------------------------------------------------------------- #
# run_root_fixpoint — tighten-only + oracle-sound                             #
# --------------------------------------------------------------------------- #
@pytest.mark.smoke
def test_root_fixpoint_tighten_only_and_sound():
    m = _bilinear_model()
    lb = np.array([0.0, 0.0])
    ub = np.array([4.0, 4.0])
    # Box optimum of x*y-2x over [0,4]^2 s.t. x+y<=6 is -8 (x=4,y=0).
    res = run_root_fixpoint(
        m,
        lb,
        ub,
        incumbent_cutoff=-8.0 + 1e-4,
        deadline=time.perf_counter() + 10,
        tol=1e-6,
        measure_bound=True,
    )
    assert not res.infeasible
    # Tighten-only: reduced box is a subset.
    assert np.all(res.lb >= lb - 1e-9)
    assert np.all(res.ub <= ub + 1e-9)
    # Oracle soundness: the root LP bound never crosses the optimum -8.
    if res.root_bound_after is not None and np.isfinite(res.root_bound_after):
        assert res.root_bound_after <= -8.0 + 1e-4


@pytest.mark.smoke
def test_root_fixpoint_no_cutoff_is_sound_noop_or_structural():
    """With no incumbent the cutoff stages degrade to their structural subset and
    the loop stays tighten-only (never loosens)."""
    m = _bilinear_model()
    lb = np.array([0.0, 0.0])
    ub = np.array([4.0, 4.0])
    res = run_root_fixpoint(
        m,
        lb,
        ub,
        incumbent_cutoff=None,
        deadline=time.perf_counter() + 10,
        measure_bound=False,
    )
    assert not res.infeasible
    assert np.all(res.lb >= lb - 1e-9)
    assert np.all(res.ub <= ub + 1e-9)


@pytest.mark.slow
def test_root_fixpoint_round_two_improves_bound():
    """The fixpoint LOOP (>=2 rounds) is load-bearing: a second round, re-deriving
    the McCormick envelope on the round-1-tightened box, tightens strictly more
    than a single round. Fails if the loop is capped to one round."""
    m = _bilinear_model()
    lb = np.array([0.0, 0.0])
    ub = np.array([4.0, 4.0])
    cutoff = -8.0 + 1e-4
    one = run_root_fixpoint(
        m,
        lb,
        ub,
        incumbent_cutoff=cutoff,
        deadline=time.perf_counter() + 10,
        max_rounds=1,
        measure_bound=False,
    )
    two = run_root_fixpoint(
        m,
        lb,
        ub,
        incumbent_cutoff=cutoff,
        deadline=time.perf_counter() + 10,
        max_rounds=2,
        measure_bound=False,
    )
    # Two rounds tighten at least as many half-bounds; the box is a subset of the
    # one-round box (never looser). Both remain sound (oracle not crossed).
    assert two.n_tightened >= one.n_tightened
    assert np.all(two.lb >= one.lb - 1e-9)
    assert np.all(two.ub <= one.ub + 1e-9)


# --------------------------------------------------------------------------- #
# C-41 — block<->flat cutoff-FBBT index-alignment guard                        #
# --------------------------------------------------------------------------- #
#
# The Rust ``fbbt_with_cutoff`` returns one interval per ``model.variables``
# BLOCK. Both ``node_reduce._fbbt_on_node`` and ``root_reduce._fbbt`` map that
# block-indexed result onto the FLAT scalar box via ``fbbt_lbs[bi] -> lb[flat]``.
# When a builder-mode / reformulated repr returns a block count that DIVERGES
# from ``len(model._variables)`` (C-40: 144 for a 145-column model), a positional
# read lands on the WRONG variable's bound and can write a crossed ``lb>ub`` box,
# wrongly fathoming the optimum-containing node. C-41 adds the same 1:1-alignment
# guard used at solver.py:7443 / solvers/_root_presolve.py:43: on a misaligned
# repr, forgo the *optional* tightening (a valid, looser box), never corrupt it.
#
# These tests inject a deliberately misaligned FBBT result (a short array whose
# positional values, if applied, would CROSS a variable's box) and assert the
# guard forgoes the tightening. Pre-fix (positional apply with only an OOB skip)
# these FAIL: the box is corrupted / the node is falsely fathomed.


class _MisalignedRepr:
    """A stand-in Rust repr whose ``fbbt_with_cutoff`` returns FEWER intervals
    than the model has blocks (the C-40 144-vs-145 divergence), with the last
    (misaligned) entry crafted to cross variable 1's box if applied positionally.
    """

    def __init__(self, n_blocks):
        # n_blocks-1 intervals => guaranteed length mismatch. Entry 0 is a benign
        # (loose) interval; the loop, unguarded, would still read fbbt[0] for the
        # first scalar block. Make every returned lower bound cross that block's
        # upper bound so an unguarded apply produces a crossed box (false fathom).
        self._n = max(n_blocks - 1, 1)

    def fbbt_with_cutoff(self, max_iter=10, tol=1e-8, incumbent_bound=None):
        import numpy as _np

        # Lower bounds far above the true box uppers -> any positional apply
        # crosses (lb>ub). A correct guard never applies these.
        lbs = _np.full(self._n, 1e9, dtype=float)
        ubs = _np.full(self._n, -1e9, dtype=float)
        return lbs, ubs


@pytest.mark.smoke
def test_root_reduce_misaligned_repr_forgoes_tightening():
    """C-41: root cutoff-FBBT with a misaligned (short) repr must forgo the
    tightening, not corrupt the root box or report a false infeasible."""
    import discopt._jax.root_reduce as rr
    import discopt._rust as _rust

    m = _bilinear_model()
    lb = np.array([0.0, 0.0])
    ub = np.array([4.0, 4.0])
    n_blocks = len(m._variables)
    orig = _rust.model_to_repr

    def _fake(model, builder):  # noqa: ARG001 - signature parity
        return _MisalignedRepr(n_blocks)

    _rust.model_to_repr = _fake
    try:
        new_lb, new_ub, n_tight, infeasible = rr._stage_fbbt_with_cutoff(
            m, lb.copy(), ub.copy(), -8.0 + 1e-4, max_iter=5, tol=1e-8
        )
    finally:
        _rust.model_to_repr = orig

    # Misaligned repr -> tightening forgone: box unchanged, no false infeasible.
    assert not infeasible
    assert n_tight == 0
    assert np.allclose(new_lb, lb)
    assert np.allclose(new_ub, ub)
    assert np.all(new_lb <= new_ub + 1e-9)
