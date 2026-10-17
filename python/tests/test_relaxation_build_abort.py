"""Relaxation builds honor the solve's deadline (QPLIB_3361 / QPLIB_3709).

Both instances are dense nonconvex QPs (3361: 1024 vars, 163,680 products;
3709: 600 vars, 165,300 products) that ran 103-119 s on a 60 s limit and past
590 s on a 30 s one. Two causes, both covered here:

* every relaxation build was one uninterruptible walk of the objective
  (``build_deadline`` only truncates *constraint* rows), so a build started near
  the deadline outlived it. ``abort_deadline`` makes the walk itself poll, and a
  caller with a sound way to go without the relaxation (a node reported
  bound-less, a declined incremental structure or native spec) passes it;
* under the default-on perspective-OA gate every bilinear ``x_i * x_j`` was
  lift-eligible, so each paid a DCP walk plus up to 64 interval-Hessian walks to
  reach the only possible verdict (its Hessian ``[[0,1],[1,0]]`` is indefinite on
  every box). That is bound-neutral to skip, and asserted so below.
"""

from __future__ import annotations

import time
import types

import discopt._relax.incremental_mccormick as IM
import discopt._relax.lp_spatial_bb as LSB
import discopt._relax.milp_relaxation as MR
import discopt._relax.uniform_relax as U
import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.spatial_producer import build_spatial_kernel_spec
from discopt._relax.term_classifier import classify_nonlinear_terms


def _dense_qp(n: int = 24, seed: int = 0) -> dm.Model:
    """A small dense nonconvex box QP: the QPLIB_3361/3709 class in miniature."""
    rng = np.random.default_rng(seed)
    Q = rng.integers(-5, 6, size=(n, n)).astype(float)
    m = dm.Model("dense_qp")
    # Scalar variables, as the QPLIB reader builds them (the native spec's scope).
    x = [m.continuous(f"x{i}", lb=0.0, ub=1.0) for i in range(n)]
    m.minimize(sum(Q[i, j] * x[i] * x[j] for i in range(n) for j in range(i + 1, n)))
    m.subject_to(sum(x) <= n / 2)
    return m


def _arrays(rel):
    milp = rel.model
    return [np.asarray(milp._A_ub.toarray()), np.asarray(milp._b_ub), np.asarray(milp._c)]


def _frozen_clock(monkeypatch, module) -> None:
    """Freeze ``module``'s own clock at 0 so its entry check passes while the
    relaxation builder, on the real clock, sees an already-spent deadline of 1.0."""
    monkeypatch.setattr(module, "time", types.SimpleNamespace(perf_counter=lambda: 0.0))
    assert time.perf_counter() > 1.0


def test_past_abort_deadline_raises_inside_canonicalization_and_caches_nothing():
    m = _dense_qp()
    with pytest.raises(U.RelaxationBuildAborted, match="canonicalization"):
        U.build_uniform_relaxation(m, abort_deadline=time.perf_counter() - 1.0)
    assert U._ANALYSIS_ATTR not in m.__dict__  # the partial DAG was discarded
    U.build_uniform_relaxation(m)  # and a later unbudgeted build still succeeds


def test_past_abort_deadline_raises_inside_the_walk():
    m = _dense_qp()
    U._get_analysis_cache(m)  # warm: the abort must come from the relaxation walk
    with pytest.raises(U.RelaxationBuildAborted, match="node relaxations"):
        U.build_uniform_relaxation(m, abort_deadline=time.perf_counter() - 1.0)


def test_unspent_abort_deadline_is_byte_identical():
    m = _dense_qp()
    base = _arrays(U.build_uniform_relaxation(m))
    timed = _arrays(U.build_uniform_relaxation(m, abort_deadline=time.perf_counter() + 3600.0))
    assert len(base) == len(timed) == 3
    for a, b in zip(base, timed):
        assert np.array_equal(a, b)


def test_cold_node_bound_declines_as_unresolved(monkeypatch):
    m = _dense_qp()
    terms = classify_nonlinear_terms(m)
    lb, ub = np.zeros(24), np.ones(24)
    assert LSB._relax_bound(m, terms, lb, ub) is not None  # in-budget: a bound
    _frozen_clock(monkeypatch, LSB)
    assert LSB._relax_bound(m, terms, lb, ub, deadline=1.0) is None


def test_incremental_structure_declines(monkeypatch):
    m = _dense_qp()
    terms = classify_nonlinear_terms(m)
    _frozen_clock(monkeypatch, IM)
    inc = IM.IncrementalMcCormickLP(m, terms, deadline=1.0)
    assert inc.decline_reason is not None
    assert inc.decline_reason.startswith("RelaxationBuildAborted")


def test_native_spec_declines_past_deadline():
    m = _dense_qp()
    box = (np.zeros(24), np.ones(24))
    assert build_spatial_kernel_spec(m, bounds=box) is not None  # in scope
    assert build_spatial_kernel_spec(m, bounds=box, deadline=time.perf_counter() - 1.0) is None


def test_bilinear_products_skip_the_curvature_certificate(monkeypatch):
    assert U.perspective_oa_enabled()  # the default that made every product eligible
    calls = []
    orig = MR._multivar_box_curvature

    def counting(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(MR, "_multivar_box_curvature", counting)
    m = _dense_qp()
    U.build_uniform_relaxation(m)
    # Only the objective sum may reach the certificate; none of its 276 products.
    assert len(calls) <= 1


def test_bilinear_skip_is_bound_neutral(monkeypatch):
    m = _dense_qp()
    skipped = _arrays(U.build_uniform_relaxation(m))
    orig = U._Builder._lift_eligible

    def legacy(self, node):
        # The pre-fix gate: every product lift-eligible under perspective OA.
        if node.kind == "prod" and U._perspective_oa_active():
            return len(U.var_support(node)) >= 2
        return orig(self, node)

    monkeypatch.setattr(U._Builder, "_lift_eligible", legacy)
    m2 = _dense_qp()  # a fresh model: no shared analysis cache with ``m``
    legacy_arrays = _arrays(U.build_uniform_relaxation(m2))
    assert len(skipped) == len(legacy_arrays) == 3
    for a, b in zip(skipped, legacy_arrays):
        assert np.array_equal(a, b)
