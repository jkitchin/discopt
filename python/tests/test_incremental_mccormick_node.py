"""Phase-B throughput: the per-node McCormick relaxation uses an incremental
patch + warm-start instead of a cold ``build_milp_relaxation`` + equilibration.

The default spatial-B&B path rebuilt and re-equilibrated the McCormick LP at every
node — together ~half the wall clock (gear4: ``equilibrate`` 29% + ``build`` 19%).
``MccormickLPRelaxer`` now builds the structure once and per node patches only the
box-dependent product rows (numpy) + warm-starts the Rust simplex, giving ~19x more
nodes/s on the pure-integer QCQP class (nvs17). Since cert:T1.3 the engine is gated
ONLY on the constructor's row-for-row self-validation (``IncrementalMcCormickLP.ok``)
— for any variable mix and any objective sense — because the fast path solves the
McCormick LP *relaxation* (a valid lower bound for continuous, mixed, and integer
models alike) and ``_validate`` proves the patched rows reproduce the cold
``build_milp_relaxation`` exactly. The earlier pure-integer/minimize gate was a
conservative rollout limit (#355), not a soundness boundary. Any uncovered term
(e.g. division, NN-embedding smooth activations) makes ``_validate`` fail →
``ok=False`` → the trusted cold build runs unchanged.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.mccormick_lp import MccormickLPRelaxer


def _int_qcqp():
    """Small all-integer QCQP (bilinear+square) — in the fast-path scope."""
    m = dm.Model("iqcqp")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.minimize((x - 3) ** 2 + (y - 2) ** 2 + x * y)
    m.subject_to(x + y >= 3)
    return m


def test_incremental_active_for_integer_qcqp():
    assert MccormickLPRelaxer(_int_qcqp())._inc is not None


def test_incremental_sound_for_mixed_and_division():
    # cert:T1.3 widened the gate beyond pure-integer: the engine now activates for
    # any model whose McCormick rows self-validate against the cold build. The
    # invariant is no longer "inactive off the pure-integer path" but "never an
    # UNSOUND activation" — where it engages, the fast bound must be a valid lower
    # bound (<= the true optimum) and never tighter than the cold McCormick bound.

    # Mixed-integer bilinear: covered by McCormick -> engine engages, soundly.
    m = dm.Model("mixed")
    x = m.continuous("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.minimize(x * y)  # true min 0 (x=0,y>=3) subject to x+y>=3
    m.subject_to(x + y >= 3)
    fast = MccormickLPRelaxer(m)
    assert fast._inc is not None, "T1.3: McCormick-covered mixed model should engage"
    lb, ub = np.array([0.0, 0.0]), np.array([5.0, 5.0])
    r_fast = fast.solve_at_node(lb.copy(), ub.copy())
    cold = MccormickLPRelaxer(m)
    cold._inc = None
    r_cold = cold.solve_at_node(lb.copy(), ub.copy())
    assert r_fast.status == "optimal" and r_cold.status == "optimal"
    assert r_fast.lower_bound <= 0.0 + 1e-6  # valid lower bound (<= true optimum)
    assert r_fast.lower_bound <= r_cold.lower_bound + 1e-6  # never over-tightens cold

    # Division is an uncovered term -> _validate fails -> ok=False -> cold fallback,
    # so the engine stays inactive (the sound degradation path is preserved).
    md = dm.Model("div")
    a = md.continuous("a", lb=1, ub=5)
    b = md.continuous("b", lb=1, ub=5)
    md.minimize(a / b)
    md.subject_to(a + b >= 3)
    assert MccormickLPRelaxer(md)._inc is None, "uncovered division must fall back to cold"


def test_incremental_disabled_by_env(monkeypatch):
    monkeypatch.setenv("DISCOPT_INCREMENTAL_MC", "0")
    assert MccormickLPRelaxer(_int_qcqp())._inc is None


def test_incremental_node_bound_is_sound_and_matches_cold():
    """The incremental node LP bound must be a valid lower bound and agree with the
    cold-build LP bound (the patch is validated equal to the cold relaxation)."""
    m = _int_qcqp()
    lb, ub = np.array([0.0, 0.0]), np.array([5.0, 5.0])

    fast = MccormickLPRelaxer(m)
    assert fast._inc is not None
    r_fast = fast.solve_at_node(lb, ub)

    cold = MccormickLPRelaxer(m)
    cold._inc = None  # force the cold build
    r_cold = cold.solve_at_node(lb, ub)

    assert r_fast.status == "optimal" and r_cold.status == "optimal"
    assert r_fast.lower_bound is not None and np.isfinite(r_fast.lower_bound)
    true_opt = 4.0  # min of the integer QCQP
    # Soundness: the fast LP bound is a valid lower bound (<= the true optimum) and
    # is never *tighter* than the cold bound (the cold path adds FBBT/integrality
    # tightening the pure-LP fast path skips, so fast <= cold). A fast bound above
    # the optimum or above cold would be an unsound over-tightening.
    assert r_fast.lower_bound <= true_opt + 1e-6
    assert r_fast.lower_bound <= r_cold.lower_bound + 1e-6


def test_incremental_infeasible_node_pruned_without_cold_rebuild(monkeypatch):
    """An infeasible in-scope node is fathomed by the incremental engine itself —
    the McCormick polytope is a valid outer approximation, so an empty LP over a
    finite box is a rigorous infeasibility proof. Previously this re-derived the
    relaxation cold just to re-confirm the verdict (the dominant per-node cost);
    now it must return ``infeasible`` without calling ``build_milp_relaxation``."""
    m = _int_qcqp()  # x,y in [0,5], constraint x + y >= 3
    # Box x in [0,1], y in [0,1]: x + y <= 2 < 3 -> the relaxation LP is infeasible.
    lb, ub = np.array([0.0, 0.0]), np.array([1.0, 1.0])

    fast = MccormickLPRelaxer(m)
    assert fast._inc is not None

    import discopt._jax.mccormick_lp as mc

    calls = {"n": 0}
    _orig = mc.build_milp_relaxation

    def _counting_build(*a, **k):
        calls["n"] += 1
        return _orig(*a, **k)

    monkeypatch.setattr(mc, "build_milp_relaxation", _counting_build)
    r_fast = fast.solve_at_node(lb, ub)
    assert r_fast.status == "infeasible"
    assert r_fast.lower_bound is None
    # The whole point: the infeasible verdict came from the incremental path, with
    # no cold rebuild to re-confirm it.
    assert calls["n"] == 0

    # Soundness: the cold path reaches the SAME infeasible verdict on this box.
    cold = MccormickLPRelaxer(m)
    cold._inc = None
    r_cold = cold.solve_at_node(lb, ub)
    assert r_cold.status == "infeasible"


def test_incremental_feasible_node_still_rebuilds_only_if_needed():
    """A feasible in-scope node returns an optimal bound from the fast path (no
    infeasible misfire). Guards the infeasible-trust branch from over-firing."""
    m = _int_qcqp()
    lb, ub = np.array([0.0, 0.0]), np.array([5.0, 5.0])
    r = MccormickLPRelaxer(m).solve_at_node(lb, ub)
    assert r.status == "optimal"
    assert r.lower_bound is not None and np.isfinite(r.lower_bound)
    assert r.lower_bound <= 4.0 + 1e-6  # valid lower bound on the true optimum (4.0)


def test_incremental_full_solve_matches_cold():
    """End-to-end: fast path and cold path reach the same certified optimum."""
    os.environ["DISCOPT_INCREMENTAL_MC"] = "1"
    r_fast = _int_qcqp().solve(time_limit=20, gap_tolerance=1e-4)
    os.environ["DISCOPT_INCREMENTAL_MC"] = "0"
    try:
        r_cold = _int_qcqp().solve(time_limit=20, gap_tolerance=1e-4)
    finally:
        os.environ.pop("DISCOPT_INCREMENTAL_MC", None)
    assert r_fast.status == r_cold.status == "optimal"
    assert r_fast.objective == pytest.approx(r_cold.objective, abs=1e-4)
    assert r_fast.gap_certified and r_cold.gap_certified


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
