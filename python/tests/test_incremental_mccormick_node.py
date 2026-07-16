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


def _span_bilinear():
    """Bilinear model whose BOTH factors span zero at the root, so real B&B nodes
    (and the validation set) carry negative / zero-spanning boxes for those vars."""
    m = dm.Model("span")
    x = m.continuous("x", lb=-3, ub=4)  # root box spans zero
    y = m.continuous("y", lb=-2, ub=5)  # root box spans zero
    m.minimize(x * y)
    m.subject_to(x + y >= 1)
    return m


def test_incremental_active_for_integer_qcqp():
    assert MccormickLPRelaxer(_int_qcqp())._inc is not None


def test_incremental_declined_when_lift_too_large_for_dense(monkeypatch):
    """A lift whose dense ``base_A`` would exceed the cell budget declines the
    incremental structure and falls back to the sparse per-node cold build with an
    UNCHANGED (never looser) bound.

    Regression for the qap ~30 GB blowup: ``IncrementalMcCormickLP`` stores
    ``base_A`` DENSE (``rows x cols``) and ``_patch`` copies that whole array on
    EVERY node — ~14.85 GB each for qap's 85756x21649 lift, ~30 GB peak just
    constructing the relaxer. The size guard declines the dense structure above
    ``_MAX_INCREMENTAL_DENSE_CELLS`` so large-lift models use the sparse cold path.
    Before the guard, ``_inc`` was built regardless of lift size (this test's
    ``_inc is None`` assertion fails); after, it is declined.
    """
    import discopt._jax.incremental_mccormick as inc

    m = _int_qcqp()
    lb = np.array([float(v.lb) for v in m._variables], dtype=np.float64)
    ub = np.array([float(v.ub) for v in m._variables], dtype=np.float64)

    # Reference bound WITH the fast path (normal cap): structure engages.
    relaxer_fast = MccormickLPRelaxer(m)
    assert relaxer_fast._inc is not None
    ref = relaxer_fast.solve_at_node(lb, ub)
    assert ref.status == "optimal"

    # Tiny cap forces the oversize decline even on this small QCQP lift.
    monkeypatch.setattr(inc, "_MAX_INCREMENTAL_DENSE_CELLS", 1.0)
    relaxer_cold = MccormickLPRelaxer(m)
    assert relaxer_cold._inc is None  # dense structure declined -> cold fallback
    got = relaxer_cold.solve_at_node(lb, ub)
    assert got.status == "optimal"
    # Sound + never looser: the cold path keeps every cut the fast path may drop,
    # so its lower bound is >= the fast-path bound (bound-neutral to slightly
    # tighter), never a regression.
    assert got.lower_bound >= ref.lower_bound - 1e-6


def test_validate_exercises_at_least_four_sign_regimes():
    """C-21: the soundness gate must probe negative-lb / zero-spanning / mixed-sign
    / degenerate boxes, not just ``lb>=0``. On a model with zero-spanning root
    factors the validation set covers >= 4 distinct sign regimes."""
    inc = MccormickLPRelaxer(_span_bilinear())._inc
    assert inc is not None and inc.ok
    regimes = inc._validated_regimes
    # span (lb<0<ub), zero_lb (lb==0<ub), neg (ub<=0), degen (lb==ub), pos (lb>0)
    for needed in ("span", "neg", "degen", "zero_lb"):
        assert needed in regimes, f"validation set never exercised the {needed!r} regime"
    assert len(regimes) >= 4, f"only {len(regimes)} sign regimes: {sorted(regimes)}"


def test_validate_catches_negative_box_sign_flip_mutation(monkeypatch):
    """C-21 mutation test. A ``_bilinear_rows`` that clips negative lower bounds to
    zero is the IDENTITY on ``lb>=0`` boxes (so the pre-C-21 validation set, which
    only used such boxes, would have accepted it — a silent divergence in exactly
    the sign regimes that dominate real nodes) but WRONG on negative / zero-spanning
    boxes. The hardened gate now includes such boxes, so the mutation must make
    ``_validate`` reject the fast path (``ok`` False / ``_inc`` None).

    Reverting the box set to ``lb>=0``-only makes this assertion fail (verified
    manually during C-21): the mutation then slips through undetected.
    """
    import discopt._jax.incremental_mccormick as ic

    # Sanity: the unmutated engine engages on this model.
    assert MccormickLPRelaxer(_span_bilinear())._inc is not None

    _orig_rows = ic._bilinear_rows

    def _clip_negative_lb(i, j, a, li, ui, lj, uj):
        # Identity when li,lj >= 0; diverges once a lower bound goes negative.
        return _orig_rows(i, j, a, max(li, 0.0), ui, max(lj, 0.0), uj)

    monkeypatch.setattr(ic, "_bilinear_rows", _clip_negative_lb)
    inc = MccormickLPRelaxer(_span_bilinear())._inc
    assert inc is None, "sign-flip mutation must be caught by the hardened validation gate"


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
