"""Regression tests for the sparse-bilinear RLT auto-gate widening (issue #727).

Issue #727 (SOTA wall-time gap on medium MINLPs) attributes the pooling /
bilinear-flow-network cluster to a *weak McCormick root bound*. The measured
mechanism: build-time level-1 RLT (with per-node RLT cuts) certifies these
instances at the ROOT in seconds, but the default RLT auto policy gates on a raw
*variable count* (``_AUTO_RLT_LEVEL1_MAX_VARS`` / ``_AUTO_CUTS_MAX_VARS``), which
excludes medium pooling instances even though their RLT relaxation is small and
cheap. The variable count is a poor cost proxy: a sparse-bilinear pooling network
has product terms growing ~linearly with the variable count, so its RLT relaxation
stays small, whereas a dense QCQP grows products quadratically and is correctly
excluded.

The ``rlt_sparse_auto`` flag (``DISCOPT_RLT_SPARSE_AUTO``) widens the auto gate for
exactly the sparse-bilinear structural class. It is **default-off** (bound-changing;
CLAUDE.md §5 keeps it off until the corpus-wide differential graduation panel), so
the default dispatch is byte-identical.

These tests exercise:
  * the flag defaults off and the gate is a no-op then (byte-identical dispatch);
  * the structural admit/reject logic (sparse pooling admitted; dense QCQP and
    oversize models rejected);
  * end-to-end: a medium pooling model (48 continuous vars, above the 40/50 raw
    caps) that the default policy leaves uncertified is certified at the root with
    the flag on — with the correctness invariant intact (the reported bound is a
    valid dual bound and the incumbent equals the true optimum).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solver_tuning import SolverTuning, reset_current, set_current


def _kblock_haverly(k: int) -> dm.Model:
    """``k`` independent Haverly-I pooling blocks in one model.

    Each block is the canonical Haverly-I pooling problem (true optimum 400 as a
    maximize); ``k`` blocks give ``6*k`` continuous variables, ``4*k`` bilinear
    product terms (sparse: products grow linearly with variables), and a known
    optimum of ``400*k``. This is a controlled proxy for the #727 pooling cluster:
    the bilinear structure and weak McCormick root bound are the same, and the
    optimum is known exactly.
    """
    m = dm.Model(f"kpool{k}")
    quality = np.array([3.0, 1.0, 2.0])
    rev = np.array([9.0, 15.0])
    sc = np.array([6.0, 16.0, 10.0])
    obj = 0
    for b in range(k):
        y = m.continuous(f"y{b}", shape=(2,), lb=0, ub=100)
        x = m.continuous(f"x{b}", shape=(2,), lb=0, ub=100)
        z = m.continuous(f"z{b}", lb=0, ub=100)
        p = m.continuous(f"p{b}", lb=0.0, ub=3.0)
        obj = obj + rev[0] * x[0] + rev[1] * (x[1] + z) - sc[0] * y[0] - sc[1] * y[1] - sc[2] * z
        m.subject_to(y[0] + y[1] == x[0] + x[1], name=f"mb{b}")
        m.subject_to(p * (y[0] + y[1]) == quality[0] * y[0] + quality[1] * y[1], name=f"qb{b}")
        m.subject_to(p * x[0] <= 2.5 * x[0], name=f"s0_{b}")
        m.subject_to(p * x[1] + quality[2] * z <= 1.5 * (x[1] + z), name=f"s1_{b}")
        m.subject_to(x[0] <= 100, name=f"d0_{b}")
        m.subject_to(x[1] + z <= 200, name=f"d1_{b}")
    m.maximize(obj)
    return m


def _dense_qp(n: int) -> dm.Model:
    """A dense box-QP: every pairwise product present (products ~ n^2/2)."""
    m = dm.Model(f"dqp{n}")
    x = m.continuous("x", shape=(n,), lb=-1, ub=1)
    obj = 0
    for i in range(n):
        for j in range(i, n):
            obj = obj + ((i * 7 + j * 3) % 5 - 2) * x[i] * x[j]
    m.minimize(obj)
    m.subject_to(dm.sum(x) <= 1)
    return m


def test_flag_defaults_off():
    """The sparse-bilinear widening is opt-in; the default dispatch is unchanged."""
    tun = SolverTuning()
    assert tun.rlt_sparse_auto is False
    assert tun.rlt_sparse_max_vars == 200
    assert tun.rlt_sparse_max_terms == 300


def test_admit_gate_noop_when_flag_off():
    """With the flag off, ``_rlt_sparse_admit`` never widens (returns False)."""
    from discopt.solver import _rlt_sparse_admit

    m = _kblock_haverly(12)  # 72 vars, sparse bilinear — would be admitted if on
    n_vars = sum(v.size for v in m._variables)
    tok = set_current(SolverTuning(rlt_sparse_auto=False))
    try:
        assert _rlt_sparse_admit(m, n_vars) is False
    finally:
        reset_current(tok)


def test_admit_gate_accepts_sparse_pooling_when_on():
    """A medium sparse-bilinear pooling model (above the raw caps) is admitted."""
    from discopt.solver import _AUTO_RLT_LEVEL1_MAX_VARS, _rlt_sparse_admit

    m = _kblock_haverly(12)
    n_vars = sum(v.size for v in m._variables)
    assert n_vars > _AUTO_RLT_LEVEL1_MAX_VARS  # excluded by the raw cap
    tok = set_current(SolverTuning(rlt_sparse_auto=True))
    try:
        assert _rlt_sparse_admit(m, n_vars) is True
    finally:
        reset_current(tok)


def test_admit_gate_rejects_dense_qp_when_on():
    """A dense QCQP (products ~ n^2) exceeds the lifted-column budget → excluded."""
    from discopt.solver import _rlt_sparse_admit

    m = _dense_qp(30)  # ~465 bilinear terms, above the 300 default budget
    n_vars = sum(v.size for v in m._variables)
    tok = set_current(SolverTuning(rlt_sparse_auto=True))
    try:
        assert _rlt_sparse_admit(m, n_vars) is False
    finally:
        reset_current(tok)


def test_admit_gate_rejects_oversize_vars_when_on():
    """Beyond the variable-count ceiling the widening declines even if sparse."""
    from discopt.solver import _rlt_sparse_admit

    m = _kblock_haverly(12)  # 72 vars
    n_vars = sum(v.size for v in m._variables)
    tok = set_current(SolverTuning(rlt_sparse_auto=True, rlt_sparse_max_vars=50))
    try:
        assert _rlt_sparse_admit(m, n_vars) is False
    finally:
        reset_current(tok)


@pytest.mark.slow
def test_medium_pooling_certifies_at_root_with_flag():
    """End-to-end: a medium pooling model the default policy leaves uncertified is
    certified at the root with the sparse-bilinear widening — and the certified
    result is correct (valid dual bound, incumbent = true optimum)."""
    k = 8  # 48 continuous vars: above _AUTO_CUTS_MAX_VARS (40)
    opt = 400.0 * k

    m = _kblock_haverly(k)
    res_on = m.solve(time_limit=20, tuning=SolverTuning(rlt_sparse_auto=True))

    # Certified globally optimal at (essentially) the root.
    assert res_on.status == "optimal"
    assert res_on.gap_certified is True
    assert res_on.objective == pytest.approx(opt, abs=1e-3)
    # Correctness invariant: for a MAXIMIZE, the dual bound is a valid UPPER bound —
    # it must never fall below the true optimum (a false, too-tight bound), and the
    # certified gap pins it to the optimum.
    assert res_on.bound >= opt - 1e-3
    assert res_on.bound == pytest.approx(opt, abs=1e-2)


@pytest.mark.slow
def test_flag_off_default_unchanged_on_small_pooling():
    """The flag off leaves the already-fast small-pooling path exactly as-is: the
    canonical single-block Haverly still certifies at the root."""
    m = _kblock_haverly(1)
    res = m.solve(time_limit=20)  # default tuning (flag off)
    assert res.status == "optimal"
    assert res.gap_certified is True
    assert res.objective == pytest.approx(400.0, abs=1e-3)
