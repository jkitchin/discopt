"""End-to-end + soundness tests for the ``DISCOPT_RELAX_SPACE=reduced`` per-node
bounding mode (MAiNGO-parity plan §2 P2.3).

CORRECTNESS-CRITICAL: the reduced-space Kelley bound certifies the node dual
bound (and hence global optimality). These tests pin:
  * a small QP certifies its known optimum under ``reduced`` mode;
  * ``reduced`` produces a VALID dual lower bound (<= true optimum) and a feasible
    incumbent (>= true optimum) — never a false optimal;
  * ``hybrid`` refuses loudly (reserved for P2.5);
  * the default (``lifted``/``auto``) is byte-identical to unset on node_count +
    certified objective (a pure add; the default path must not move at all).
"""

from __future__ import annotations

import os

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import discopt.modeling as dm  # noqa: E402
from discopt import SolverTuning  # noqa: E402

pytestmark = pytest.mark.unit


def _small_qp():
    """min x0*x1 - x0 + 2*x1 s.t. x0 + x1 <= 2, box [-1,3]x[-2,2].

    A nonconvex bilinear QP with a unique global optimum reached by spatial
    branching; small enough that the (slower) reduced-space Kelley bound still
    solves it in a few seconds.
    """
    m = dm.Model()
    x = m.continuous("x", 2, lb=[-1.0, -2.0], ub=[3.0, 2.0])
    m.minimize(x[0] * x[1] - x[0] + 2.0 * x[1])
    m.subject_to(x[0] + x[1] <= 2.0)
    return m


def _oracle_qp():
    """Brute-force global optimum of _small_qp over a fine grid (feasible only)."""
    xs = np.linspace(-1.0, 3.0, 801)
    ys = np.linspace(-2.0, 2.0, 801)
    X, Y = np.meshgrid(xs, ys)
    F = X * Y - X + 2.0 * Y
    feas = (X + Y) <= 2.0 + 1e-9
    return float(F[feas].min())


@pytest.mark.smoke
def test_reduced_mode_certifies_small_qp():
    opt = _oracle_qp()
    m = _small_qp()
    res = m.solve(time_limit=60, tuning=SolverTuning(relax_space="reduced"))
    assert res.status == "optimal", f"expected certified optimal, got {res.status}"
    # incumbent matches the oracle
    assert res.objective == pytest.approx(opt, abs=1e-3, rel=1e-4)
    # dual bound is a VALID lower bound (never above the true optimum)
    assert res.bound <= opt + 1e-4 * (abs(opt) + 1.0)
    # certificate invariant: bound <= incumbent (min sense)
    assert res.bound <= res.objective + 1e-6


@pytest.mark.smoke
def test_reduced_mode_bound_never_above_optimum_time_limited():
    """Even when the reduced solve is cut off by a tight time limit, the reported
    dual bound must remain a valid lower bound and the incumbent feasible."""
    opt = _oracle_qp()
    m = _small_qp()
    res = m.solve(time_limit=5, tuning=SolverTuning(relax_space="reduced"))
    if np.isfinite(res.bound):
        assert res.bound <= opt + 1e-4 * (abs(opt) + 1.0), "dual bound above true optimum"
    if res.objective is not None and np.isfinite(res.objective):
        assert res.objective >= opt - 1e-4 * (abs(opt) + 1.0), "incumbent below true optimum"


def test_hybrid_refuses_loudly():
    m = _small_qp()
    with pytest.raises(NotImplementedError, match="hybrid"):
        m.solve(time_limit=5, tuning=SolverTuning(relax_space="hybrid"))


@pytest.mark.parametrize("mode", ["lifted", "auto"])
def test_default_paths_byte_identical(mode):
    """``lifted`` and ``auto`` reproduce the unset-flag path exactly (node_count +
    certified objective). P2.3 is a pure add; the default path must not move."""
    m1 = _small_qp()
    base = m1.solve(time_limit=60)  # unset -> "lifted"
    m2 = _small_qp()
    got = m2.solve(time_limit=60, tuning=SolverTuning(relax_space=mode))
    assert got.node_count == base.node_count
    assert got.objective == pytest.approx(base.objective, abs=0.0, rel=0.0)
    assert got.status == base.status


_NVS22 = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/nvs22.nl")


@pytest.mark.xfail(
    reason="P2.3 NO-GO: reduced_mccormick_lp_bound returns false-`infeasible` on a "
    "node box that contains the feasible optimum of nvs22 (equality-constrained "
    "division/sqrt intermediates + unbounded leaves the builder fails to refuse). "
    "This fathoms the node holding the optimum -> false optimal (11.44 vs 6.06). "
    "Reduced mode must NOT be wired into the solver until this UPSTREAM evaluator "
    "soundness/scope bug is fixed (see docs/dev/maingo-parity-plan.md §7 P2.3).",
    strict=True,
)
@pytest.mark.skipif(not os.path.exists(_NVS22), reason="MINLPLib corpus (nvs22.nl) not available")
def test_reduced_bound_no_false_infeasible_nvs22():
    """Regression pin for the P2.3 blocker: a valid outer relaxation must NEVER
    report the relaxed feasible set empty on a box that contains a feasible point.
    XFAIL(strict) until the reduced-space evaluator's infeasibility test / scope
    guard is fixed; flip to a passing soundness test once it is."""
    from discopt._jax.mccormick_subgradient import reduced_mccormick_lp_bound

    m = dm.from_nl(_NVS22)
    # x* is the (lifted-)certified global optimum; a box of half-width ~1 around it.
    xstar = np.array([5, 1, 1, 2, 2121.6408, 10782.6705, 3.1623, 0.3162], dtype=float)
    lb, ub = xstar.copy(), xstar.copy()
    for i in range(4):
        lb[i], ub[i] = max(1.0, xstar[i] - 1.0), xstar[i] + 1.0
    for i in range(4, 8):
        span = abs(xstar[i]) * 0.1 + 1.0
        lb[i], ub[i] = xstar[i] - span, xstar[i] + span
    assert np.all(lb <= xstar) and np.all(xstar <= ub)  # x* is inside the box
    rb = reduced_mccormick_lp_bound(m, lb, ub)
    # A sound relaxation of a non-empty set is never "infeasible".
    assert rb.status != "infeasible", (
        f"FALSE-INFEASIBLE on a box containing feasible x* (bound={rb.bound})"
    )


def test_reduced_falls_back_on_unsupported_model():
    """A model outside the sound reduced-space (MCBox) scope must fall back to the
    lifted path for the whole solve — never error to the user."""
    # A ``/`` by a variable whose denominator interval spans 0 is out of MCBox
    # scope at the root; the solve must still complete via the lifted fallback.
    m = dm.Model()
    x = m.continuous("x", 2, lb=[-1.0, -1.0], ub=[2.0, 2.0])
    # sin is not in the sound reduced-space op set -> unsupported at build.
    m.minimize(dm.sin(x[0]) + x[1] * x[1])
    m.subject_to(x[0] + x[1] <= 1.0)
    res = m.solve(time_limit=30, tuning=SolverTuning(relax_space="reduced"))
    # Fallback path is the lifted solver; it must return a sound result, not raise.
    assert res.status in ("optimal", "feasible", "infeasible")
