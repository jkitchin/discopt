"""Regression tests for root-cut-pool inheritance (THRU-4, ``DISCOPT_CUT_INHERIT``).

The lever: THRU-3 measured the two per-node point separators — the
univariate-square tangent loop and the PSD (moment) loop — as the dominant
per-node cost on dense integer QPs (73% + 12% of the nvs24 solve wall), each
re-deriving cuts via up to 8 full MILP re-solves at every node. Under
``cut_inherit`` the root separates the full chain once, the rows are pooled, and
nodes inherit the pool instead of re-separating.

Soundness invariants pinned here:

* **Root-pool validity on children** — every pooled row is satisfied by every
  feasible lifted point of the ROOT box, hence of every descendant sub-box
  (a child's feasible set is a subset of the root's). Square tangents
  (``s >= 2 x0 x - x0**2``) and PSD eigencuts (``v^T M v >= 0``) are valid at
  any lifted feasible point independent of the box.
* **Box-dependent rows are NOT inheritable and are excluded by design** — the
  static child-box relaxation carries rows (e.g. the secant overestimator of
  ``x**2`` on a sub-box) that CUT feasible points outside that sub-box; the
  implementation therefore pools rows captured at the ROOT box only, never from
  node-level solves. A synthetic case demonstrates the hazard and asserts the
  flag-ON solve still certifies the true optimum that naive child-row
  inheritance would cut off.
* **Force-off neutrality** — with ``cut_inherit=False`` (the shipped DEFAULT) no
  pool-skip is performed.
* **Structure gate (CUT-INHERIT-GRAD, OPT-IN)** — with ``cut_inherit=None`` (env
  ``DISCOPT_CUT_INHERIT=gated``) the gate auto-engages iff a non-empty root pool is
  separated (the pool-fires predicate): ON on a pool-firing dense QP, byte-identical
  to force-off on a model that separates no square/PSD pool. The gate is validated
  broadly beneficial where it fires but stays OPT-IN — the default-ON flip is
  blocked by a flag-path false-optimal on the MINLP cold-path class (nvs22), see
  ``docs/dev/cut-inherit-grad-2026-07-08.md``. Env precedence: unset/``0`` ⇒
  force-off (default), ``gated``/``auto`` ⇒ structure-gated, ``1`` ⇒ force-on.
"""

from __future__ import annotations

import itertools
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling.core as dm
import numpy as np
import pytest
from discopt.solver_tuning import SolverTuning

# Dense indefinite Q over a small signed integer box — a miniature of the nvs24
# class (dense integer QP, no linear constraints) where the root square/PSD
# separation loop verifiably fires (17 cuts captured at the root box).
_Q = np.array([[1.0, -2.0], [-2.0, 1.0]])
_C = np.array([0.5, -0.5])
_N = 2
_LB, _UB = -3, 3


def _build_dense_int_qp() -> dm.Model:
    m = dm.Model("mini_nvs")
    x = m.integer("x", shape=(_N,), lb=_LB, ub=_UB)
    expr = None
    for i in range(_N):
        for j in range(_N):
            if _Q[i, j] != 0.0:
                term = _Q[i, j] * x[i] * x[j]
                expr = term if expr is None else expr + term
    for i in range(_N):
        if _C[i] != 0.0:
            expr = expr + _C[i] * x[i]
    m.minimize(expr)
    return m


def _brute_force_opt() -> tuple[float, np.ndarray]:
    best, best_x = np.inf, None
    for pt in itertools.product(range(_LB, _UB + 1), repeat=_N):
        v = np.asarray(pt, dtype=float)
        val = float(v @ _Q @ v + _C @ v)
        if val < best:
            best, best_x = val, v
    return best, best_x


def _lifted_point(x: np.ndarray, varmap: dict, n_total: int) -> np.ndarray | None:
    """Exact lifted vector z for an original point x: z carries x, the lifted
    squares ``X_ii = x_i**2`` and products ``X_ij = x_i x_j``. Returns None if any
    lifted column is not covered by a family this helper reproduces (test guard:
    the model must stay simple enough that the exact lift is stateable)."""
    z = np.zeros(n_total)
    covered: set[int] = set()
    for col in (varmap.get("original") or {}).values():
        z[int(col)] = float(x[int(col)])
        covered.add(int(col))
    for fam in ("monomial", "univariate_square"):
        for (i, p), col in (varmap.get(fam) or {}).items():
            z[int(col)] = float(x[int(i)]) ** int(p)
            covered.add(int(col))
    for fam in ("bilinear", "trilinear", "multilinear"):
        for key, col in (varmap.get(fam) or {}).items():
            z[int(col)] = float(np.prod([x[int(i)] for i in key]))
            covered.add(int(col))
    if covered != set(range(n_total)):
        return None
    return z


def _root_pool_and_varmap():
    """Capture the root cut pool on the dense integer QP, plus the lifted layout."""
    from discopt._jax.mccormick_lp import MccormickLPRelaxer
    from discopt._jax.milp_relaxation import build_milp_relaxation

    model = _build_dense_int_qp()
    relaxer = MccormickLPRelaxer(model, psd_cuts=True)
    lb = np.full(_N, float(_LB))
    ub = np.full(_N, float(_UB))
    chunks: list = []
    res = relaxer.solve_at_node(lb, ub, time_limit=30.0, out_cuts=chunks)
    assert res.status == "optimal", f"root solve failed: {res.status}"
    assert chunks, "root separation captured no cut pool"
    A_pool, b_pool = chunks[0]
    A_pool = np.asarray(
        A_pool.todense() if hasattr(A_pool, "todense") else A_pool, dtype=np.float64
    )
    b_pool = np.asarray(b_pool, dtype=np.float64)
    assert A_pool.shape[0] >= 1, "empty root cut pool"
    milp, varmap = build_milp_relaxation(
        model, relaxer._terms, relaxer._disc, bound_override=(lb, ub)
    )
    return A_pool, b_pool, varmap, len(milp._c)


@pytest.mark.smoke
def test_root_pool_cuts_valid_on_every_child_feasible_point():
    """Feasible-point sampling: every pooled root cut must hold at EVERY feasible
    lifted point of the root box — hence at every point of every child sub-box."""
    A_pool, b_pool, varmap, n_total = _root_pool_and_varmap()
    assert A_pool.shape[1] == n_total
    checked = 0
    for pt in itertools.product(range(_LB, _UB + 1), repeat=_N):
        z = _lifted_point(np.asarray(pt, dtype=float), varmap, n_total)
        assert z is not None, "test model triggered an unmodelled lifted family"
        viol = A_pool @ z - b_pool
        assert viol.max() <= 1e-6, (
            f"root pool cut violated at feasible point {pt}: max violation {viol.max():.3e}"
        )
        checked += 1
    assert checked == (_UB - _LB + 1) ** _N


@pytest.mark.smoke
def test_box_dependent_child_rows_would_be_invalid_and_are_excluded():
    """The hazard THRU-4 must exclude: rows of a CHILD-box relaxation (e.g. the
    secant of ``x**2`` on a sub-box) cut feasible points outside that sub-box, so
    they must never be inherited. Demonstrate the hazard, then assert the shipped
    root pool contains no such row and the flag-ON solve still certifies the true
    optimum that naive child-row inheritance would have cut off."""
    from discopt._jax.mccormick_lp import MccormickLPRelaxer
    from discopt._jax.milp_relaxation import build_milp_relaxation

    model = _build_dense_int_qp()
    relaxer = MccormickLPRelaxer(model, psd_cuts=True)

    # Child box pinning x to the upper corner region [2, 3]^_N.
    child_lb = np.full(_N, 2.0)
    child_ub = np.full(_N, float(_UB))
    milp_child, varmap_child = build_milp_relaxation(
        model, relaxer._terms, relaxer._disc, bound_override=(child_lb, child_ub)
    )
    A_child = milp_child._A_ub
    A_child = np.asarray(
        A_child.todense() if hasattr(A_child, "todense") else A_child, dtype=np.float64
    )
    b_child = np.asarray(milp_child._b_ub, dtype=np.float64)
    n_total = len(milp_child._c)

    # A root-feasible point OUTSIDE the child box (the origin). At least one
    # child-box row must cut it (e.g. a McCormick under/overestimator built from
    # the child bounds), proving child rows are not globally inheritable.
    z0 = _lifted_point(np.zeros(_N), varmap_child, n_total)
    assert z0 is not None
    child_viol = (A_child @ z0 - b_child).max()
    assert child_viol > 1e-6, (
        "expected the child-box relaxation to cut a root-feasible point; "
        "the box-dependence hazard this test pins vanished"
    )

    # The shipped pool (captured at the ROOT box) must NOT cut that point.
    A_pool, b_pool, varmap_root, n_root = _root_pool_and_varmap()
    z0_root = _lifted_point(np.zeros(_N), varmap_root, n_root)
    assert z0_root is not None
    assert (A_pool @ z0_root - b_pool).max() <= 1e-6

    # End-to-end guard: flag-ON must reproduce the brute-force optimum with a
    # valid bound (bound <= optimum for min). Naive child-row inheritance would
    # have fathomed the region holding the optimum.
    opt, _ = _brute_force_opt()
    res = model.solve(time_limit=60, tuning=SolverTuning(cut_inherit=True))
    assert res.objective is not None
    assert res.objective == pytest.approx(opt, abs=1e-5)
    if res.bound is not None:
        assert res.bound <= opt + 1e-6, f"dual bound {res.bound} crossed the optimum {opt}"
    # Fire-proof: the pool populated and was inherited/skipped at node solves.
    stats = res.solver_stats or {}
    assert stats.get("pool/size", 0) >= 1, f"root pool did not populate: {stats}"
    assert (
        stats.get("pool/inherited_nodes", 0) >= 1 or stats.get("pool/skipped_separations", 0) >= 1
    ), f"pool was never inherited at a node: {stats}"


@pytest.mark.smoke
def test_cut_inherit_force_off_no_skip():
    """Force-off (``cut_inherit=False``): the square/PSD separators run as before —
    no pool-skip, regardless of structure."""
    model = _build_dense_int_qp()
    res = model.solve(time_limit=60, tuning=SolverTuning(cut_inherit=False))
    opt, _ = _brute_force_opt()
    assert res.objective == pytest.approx(opt, abs=1e-5)
    stats = res.solver_stats or {}
    assert stats.get("pool/skipped_separations", 0) == 0
    assert stats.get("pool/gate_decision", 0.0) == 0.0
    assert stats.get("pool/gate_mode", 0.0) == 0.0  # forced off


@pytest.mark.smoke
def test_cut_inherit_structure_gated_fires_on_dense_qp():
    """CUT-INHERIT-GRAD: the STRUCTURE-GATED opt-in (``cut_inherit=None``, env
    ``DISCOPT_CUT_INHERIT=gated``) must auto-engage on a model that separates a
    non-empty root pool — the pool-fires predicate. On the dense integer QP the
    pool populates, so the gate decision is ON and the square/PSD separators are
    skipped at nodes, while the certificate is preserved (bound <= optimum)."""
    model = _build_dense_int_qp()
    res = model.solve(time_limit=60, tuning=SolverTuning(cut_inherit=None))
    opt, _ = _brute_force_opt()
    assert res.objective == pytest.approx(opt, abs=1e-5)
    if res.bound is not None:
        assert res.bound <= opt + 1e-6, f"dual bound {res.bound} crossed the optimum {opt}"
    stats = res.solver_stats or {}
    assert stats.get("pool/size", 0) >= 1, f"root pool did not populate: {stats}"
    assert stats.get("pool/gate_mode", 0.0) == -1.0, "expected structure-gated mode"
    assert stats.get("pool/gate_decision", 0.0) == 1.0, (
        f"structure gate did not engage on a pool-firing model: {stats}"
    )
    assert stats.get("pool/skipped_separations", 0) >= 1, (
        f"gated-on solve did not skip any per-node separation: {stats}"
    )


def _build_linear_int() -> dm.Model:
    m = dm.Model("linear_int")
    x = m.integer("x", shape=(3,), lb=0, ub=5)
    m.minimize(x[0] + 2 * x[1] + 3 * x[2])
    m.subject_to(x[0] + x[1] + x[2] >= 4)
    return m


@pytest.mark.smoke
def test_cut_inherit_structure_gated_inert_when_pool_empty():
    """CUT-INHERIT-GRAD off-class: on a model with no liftable square/PSD structure
    the root pool stays empty, so the structure gate does NOT engage and the solve
    is byte-identical to force-off (same node_count + objective). A pure linear
    integer model separates no square/PSD pool."""
    res_gated = _build_linear_int().solve(time_limit=30, tuning=SolverTuning(cut_inherit=None))
    res_off = _build_linear_int().solve(time_limit=30, tuning=SolverTuning(cut_inherit=False))
    # Gate must not fire (empty/absent pool) and the two paths must be identical.
    sg = res_gated.solver_stats or {}
    assert sg.get("pool/gate_decision", 0.0) == 0.0, f"gate wrongly fired off-class: {sg}"
    assert res_gated.node_count == res_off.node_count
    assert res_gated.objective == pytest.approx(res_off.objective, abs=1e-9)


@pytest.mark.smoke
def test_cut_inherit_env_tristate_precedence(monkeypatch):
    """The env override precedence (CUT-INHERIT-GRAD, opt-in default): unset ⇒
    force-off (``False``, the shipped default — the gated flip is blocked by the
    nvs22 flag-path false-optimal), ``=0`` ⇒ force-off, ``=gated``/``=auto`` ⇒
    structure-gated opt-in (``None``), ``=1`` (any other non-``0``) ⇒ force-on."""
    from discopt.solver_tuning import SolverTuning as ST

    monkeypatch.delenv("DISCOPT_CUT_INHERIT", raising=False)
    assert ST().cut_inherit is False  # default force-off (opt-in flag)
    monkeypatch.setenv("DISCOPT_CUT_INHERIT", "0")
    assert ST().cut_inherit is False  # explicit force-off
    monkeypatch.setenv("DISCOPT_CUT_INHERIT", "gated")
    assert ST().cut_inherit is None  # structure-gated opt-in
    monkeypatch.setenv("DISCOPT_CUT_INHERIT", "auto")
    assert ST().cut_inherit is None  # auto alias
    monkeypatch.setenv("DISCOPT_CUT_INHERIT", "1")
    assert ST().cut_inherit is True  # force-on
    monkeypatch.setenv("DISCOPT_CUT_INHERIT", "yes")
    assert ST().cut_inherit is True  # any other non-"0" ⇒ on
