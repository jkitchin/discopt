"""Tests for the structure-gated auto cut policy (cuts="auto").

The policy (see the Wave-2 A/B sweep) picks at most one QCQP cut family by
structure: RLT when the model has linear constraints, PSD on pure box-QP, and
neither above the size gate. It is purely a performance choice — every cut family
is sound — so it must always preserve the optimum.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling.core as dm
import numpy as np
import pytest
from discopt._relax.mccormick_lp import MccormickLPRelaxer
from discopt.solver import _AUTO_CUTS_MAX_VARS, _apply_auto_cut_policy


def _qcqp(n: int, seed: int, constrained: bool) -> dm.Model:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Q = (A + A.T) / 2
    m = dm.Model("q")
    x = m.continuous("x", shape=(n,), lb=0, ub=1)
    expr = None
    for i in range(n):
        for j in range(n):
            term = float(Q[i, j]) * x[i] * x[j]
            expr = term if expr is None else expr + term
    m.minimize(expr)
    if constrained:
        m.subject_to(dm.sum([x[i] for i in range(n)]) <= 0.6 * n)
        m.subject_to(x[0] + x[1] <= 1.2)
    return m


# ───────────────────────── policy unit tests (fast) ─────────────────────────


def test_policy_picks_psd_on_box_qp():
    m = _qcqp(5, 0, constrained=False)
    r = MccormickLPRelaxer(m)
    _apply_auto_cut_policy(m, r)
    assert r._psd_cuts is True and r._rlt_cuts is False


def test_policy_picks_rlt_on_constrained_qcqp():
    m = _qcqp(5, 0, constrained=True)
    r = MccormickLPRelaxer(m)
    _apply_auto_cut_policy(m, r)
    assert r._rlt_cuts is True and r._psd_cuts is False


def test_policy_declines_above_size_gate():
    # A cheap-to-build diagonal QCQP with > gate variables (sum of squares + a
    # linear constraint): quadratic + constrained, but oversize -> no cuts.
    n = _AUTO_CUTS_MAX_VARS + 2
    m = dm.Model("big")
    x = m.continuous("x", shape=(n,), lb=0, ub=1)
    m.minimize(dm.sum([x[i] * x[i] for i in range(n)]) - dm.sum([x[i] for i in range(n)]))
    m.subject_to(dm.sum([x[i] for i in range(n)]) <= 0.5 * n)
    r = MccormickLPRelaxer(m)
    _apply_auto_cut_policy(m, r)
    assert r._psd_cuts is False and r._rlt_cuts is False


# ───────────────────────── end-to-end (slow) ─────────────────────────


@pytest.mark.slow
def test_auto_is_the_default():
    """A default solve (no cuts kwarg) applies the auto policy; cuts='manual' opts out."""
    default = _qcqp(6, 3, constrained=True).solve(time_limit=120)
    manual = _qcqp(6, 3, constrained=True).solve(cuts="manual", time_limit=120)
    auto = _qcqp(6, 3, constrained=True).solve(cuts="auto", time_limit=120)
    assert abs(float(default.objective) - float(manual.objective)) < 1e-3
    # Default behaves like auto, not like the cut-free manual baseline.
    assert default.node_count == auto.node_count
    assert default.node_count < manual.node_count


@pytest.mark.slow
@pytest.mark.parametrize(("n", "seed"), [(6, 0), (10, 0), (10, 2)])
@pytest.mark.parametrize("constrained", [False, True])
def test_auto_matches_best_family_and_preserves_optimum(n, seed, constrained):
    """Auto reproduces the family its structural rule selects, exactly.

    #1039: this test is named for that claim but never asserted it. Both halves
    compared ``auto`` against the *cut-free* baseline through an invented
    ``< base/2`` ratio -- neither PSD nor RLT was ever solved, so "matches best
    family" went untested. Worse, the box-QP half failed that ratio (111 vs a
    demanded < 90.5 against a 181-node baseline), which meant the constrained
    half below it never executed at all.

    Measured over 8 draws (``scratchpad/issue1039/probe_auto_family.py``), auto
    reproduces its selected family's node count **exactly**, 8/8:

        n=6  s0 box     base=181  auto=111  psd=111  rlt=181
        n=6  s0 constr  base= 13  auto= 11  psd= 13  rlt= 11
        n=6  s3 constr  base=143  auto= 45  psd=121  rlt= 45
        n=10 s0 box     base= 83  auto= 23  psd= 23  rlt= 83
        n=10 s0 constr  base= 83  auto= 77  psd= 29  rlt= 77
        n=10 s2 box     base= 39  auto= 19  psd= 19  rlt= 39
        n=10 s2 constr  base=125  auto= 63  psd= 51  rlt= 63

    So the assertion is now equality against the selected family -- the policy's
    actual promise, deterministic, and strictly stronger than any ratio.

    Note what the RLT rows show and this test deliberately does NOT assert: on
    the two ``n=10`` constrained draws the structurally-selected family (RLT, 77
    / 63) is beaten by the one the policy declined (PSD, 29 / 51). That is the
    documented structural rule behaving as specified, not a defect -- both
    families are sound and the optimum is preserved either way -- but "best
    family" is a claim about the *rule*, not a measured optimality guarantee.
    """
    base = _qcqp(n, seed, constrained).solve(cuts="manual", time_limit=120)
    auto = _qcqp(n, seed, constrained).solve(cuts="auto", time_limit=120)
    selected = _qcqp(n, seed, constrained).solve(
        **({"rlt_cuts": True} if constrained else {"psd_cuts": True}), time_limit=120
    )

    # Cut families are purely a performance choice: the optimum is preserved.
    assert abs(float(base.objective) - float(auto.objective)) < 1e-3
    assert abs(float(base.objective) - float(selected.objective)) < 1e-3

    # The policy reproduces the family its structural rule picks.
    family = "RLT" if constrained else "PSD"
    assert auto.node_count == selected.node_count, (
        f"n={n} seed={seed} constrained={constrained}: auto {auto.node_count} != "
        f"{family} {selected.node_count} -- auto did not select {family}"
    )
