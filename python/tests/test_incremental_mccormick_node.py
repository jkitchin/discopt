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
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.mccormick_lp import MccormickLPRelaxer


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


def test_incremental_structure_is_sparse_and_patch_matches_dense():
    """The incremental structure holds ``base_A`` SPARSE and ``_patch`` returns a
    sparse CSR whose dense form equals a from-scratch dense patch — the fixed-pattern
    ``.data`` rewrite is bit-identical to the old dense ``A[k]=0; A[k,col]=coef``.

    Regression for the sparse-incremental rewrite: before it ``base_A`` was a dense
    ``rows x cols`` array (``.todense()``, ~14.85 GB per copy on qap). Here we assert
    the representation is sparse AND the patched values are unchanged.
    """
    import numpy as np
    import scipy.sparse as sp
    from discopt._relax.incremental_mccormick import (
        IncrementalMcCormickLP,
        _affine_square_rows,
        _bilinear_rows,
        _monomial_rows,
    )
    from discopt._relax.term_classifier import classify_nonlinear_terms

    m = _int_qcqp()  # (x-3)^2 + (y-2)^2 + x*y : bilinear + affine squares
    inc = IncrementalMcCormickLP(m, classify_nonlinear_terms(m))
    assert inc.ok
    # base_A is sparse, not a dense ndarray.
    assert sp.issparse(inc.base_A)

    lb = np.array([1.0, 0.0])
    ub = np.array([5.0, 4.0])
    A_sp, b_sp, bd_sp = inc._patch(lb, ub)
    assert sp.issparse(A_sp)  # patched matrix is sparse

    # Independent dense reference of the same patch (zero the product rows, set the
    # McCormick/monomial/affine-square coefficients) — must match the sparse patch.
    A_ref = inc.base_A.toarray().copy()
    for (i, j, a), rows in inc.bilin_rows.items():
        for k, (ci, cj, cw, rhs) in zip(rows, _bilinear_rows(i, j, a, lb[i], ub[i], lb[j], ub[j])):
            A_ref[k] = 0.0
            A_ref[k, i] += ci
            A_ref[k, j] += cj
            A_ref[k, a] = cw
    for (i, a, p), rows in inc.mono_rows.items():
        for k, (ci, cs, rhs) in zip(rows, _monomial_rows(lb[i], ub[i], p)):
            A_ref[k] = 0.0
            A_ref[k, i] = ci
            A_ref[k, a] = cs
    for (j, a, coeff, const), rows in inc.affsq_rows.items():
        for k, (cx, cw, rhs) in zip(rows, _affine_square_rows(coeff, const, lb[j], ub[j])):
            A_ref[k] = 0.0
            A_ref[k, j] = cx
            A_ref[k, a] = cw
    assert np.allclose(A_sp.toarray(), A_ref, atol=1e-12)


def test_incremental_declined_when_lift_exceeds_nnz_budget(monkeypatch):
    """A lift whose nonzero count exceeds the budget declines the incremental
    structure and falls back to the sparse per-node cold build with an UNCHANGED
    (never looser) bound.

    The structure now holds ``base_A`` SPARSE and ``_patch`` copies only its
    ``.data`` (~nnz floats) per node, so the guard is by NONZEROS
    (``_MAX_INCREMENTAL_NNZ``), not the old dense ``rows*cols`` cells. Above it the
    fast path is declined (``_inc is None``) and ``solve_at_node`` uses the cold
    build; declining only forgoes the speedup, never soundness.
    """
    import discopt._relax.incremental_mccormick as inc

    m = _int_qcqp()
    lb = np.array([float(v.lb) for v in m._variables], dtype=np.float64)
    ub = np.array([float(v.ub) for v in m._variables], dtype=np.float64)

    # Reference bound WITH the fast path (normal budget): structure engages.
    relaxer_fast = MccormickLPRelaxer(m)
    assert relaxer_fast._inc is not None
    ref = relaxer_fast.solve_at_node(lb, ub)
    assert ref.status == "optimal"

    # Tiny nnz budget forces the decline even on this small QCQP lift.
    monkeypatch.setattr(inc, "_MAX_INCREMENTAL_NNZ", 1)
    relaxer_cold = MccormickLPRelaxer(m)
    assert relaxer_cold._inc is None  # structure declined -> cold fallback
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
    import discopt._relax.incremental_mccormick as ic

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

    import discopt._relax.mccormick_lp as mc

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


# --------------------------------------------------------------------------- #
# Objective CONSTANT: the incremental path solves ``min c·x`` but the relaxation's
# objective is ``c·x + obj_offset``. Dropping the offset made the fast-path node
# bound differ from the cold build's by that constant — merely weak for a positive
# constant, but a dual bound ABOVE the true node optimum for a negative one, which
# is the false-fathom class. Found while widening the LP-spatial engine (#860).
# --------------------------------------------------------------------------- #


def _const_qcqp(const):
    """Bilinear integer model whose objective carries an additive constant."""
    m = dm.Model("cq")
    x = m.integer("x", lb=1, ub=4)
    y = m.integer("y", lb=1, ub=4)
    m.subject_to(x + y >= 6)
    m.minimize(x * y + const)
    return m


@pytest.mark.smoke
@pytest.mark.parametrize("const", [0.0, -100.0, 100.0])
def test_incremental_node_bound_includes_objective_constant(const):
    """Fail-before/pass-after: the fast path's node bound must equal the cold
    build's, constant and all. Before the fix the fast path returned 8.0 for every
    ``const`` (the offset-free ``min c·x``), so at ``const=-100`` it reported a lower
    bound of +8.0 on a node whose true McCormick optimum is -92.0."""
    lb, ub = np.array([1.0, 1.0]), np.array([4.0, 4.0])
    os.environ["DISCOPT_INCREMENTAL_MC"] = "1"
    try:
        relaxer = MccormickLPRelaxer(_const_qcqp(const))
        assert relaxer._inc is not None, "test needs the incremental fast path engaged"
        r_fast = relaxer.solve_at_node(lb, ub, want_marginals=True)
        os.environ["DISCOPT_INCREMENTAL_MC"] = "0"
        r_cold = MccormickLPRelaxer(_const_qcqp(const)).solve_at_node(lb, ub, want_marginals=True)
    finally:
        os.environ.pop("DISCOPT_INCREMENTAL_MC", None)
    assert r_fast.lower_bound == pytest.approx(r_cold.lower_bound, abs=1e-6)
    # The node's true McCormick optimum shifts exactly with the constant.
    assert r_fast.lower_bound == pytest.approx(8.0 + const, abs=1e-6)
    # ... and the certificate's safe bound rides the same origin as the bound beside
    # it, so a consumer reading either one gets the same answer.
    if r_fast.safe_bound is not None:
        assert r_fast.safe_bound == pytest.approx(r_cold.safe_bound, abs=1e-6)
        assert r_fast.safe_bound <= r_fast.lower_bound + 1e-6


@pytest.mark.smoke
def test_incremental_solve_bound_matches_cold_builder_with_constant():
    """Same invariant one level down, at ``IncrementalMcCormickLP.solve`` itself:
    its bound is the relaxation objective (``c·x + obj_offset``), the scale
    ``MilpRelaxationModel.solve`` reports."""
    from discopt._relax.discretization import DiscretizationState
    from discopt._relax.incremental_mccormick import IncrementalMcCormickLP
    from discopt._relax.milp_relaxation import build_milp_relaxation
    from discopt._relax.term_classifier import classify_nonlinear_terms

    m = _const_qcqp(-100.0)
    lb, ub = np.array([1.0, 1.0]), np.array([4.0, 4.0])
    terms = classify_nonlinear_terms(m)
    inc = IncrementalMcCormickLP(m, terms)
    assert inc.ok
    assert inc.obj_offset == pytest.approx(-100.0)
    b_inc, _x, _basis = inc.solve(lb, ub)
    relax, _info = build_milp_relaxation(m, terms, DiscretizationState(), bound_override=(lb, ub))
    relax._integrality = None
    b_cold = relax.solve().bound
    assert b_inc == pytest.approx(b_cold, abs=1e-6)


def _spy_warm_lp(monkeypatch):
    """Record the ``time_limit`` every incremental-path LP is issued with.

    ``incremental_mccormick.solve_assembled_full`` imports ``solve_lp_warm_std``
    from ``discopt.solvers.milp_simplex`` *inside* the function body, so patching
    the module attribute intercepts it. The real solver still runs — this only
    reads the argument in flight.
    """
    from discopt.solvers import milp_simplex

    real = milp_simplex.solve_lp_warm_std
    seen: list = []

    def _spy(*a, **kw):
        seen.append(kw.get("time_limit", "ABSENT"))
        return real(*a, **kw)

    monkeypatch.setattr(milp_simplex, "solve_lp_warm_std", _spy)
    return seen


@pytest.mark.smoke
def test_incremental_fast_path_receives_the_callers_time_limit(monkeypatch):
    """#1009: ``solve_at_node(time_limit=T)`` must reach the fast path's LP.

    Before the fix ``_try_incremental_node`` was called before the node-wide
    deadline was even anchored and ``solve_assembled_full`` had no parameter to
    carry one, so every LP on this path was issued unbounded — one such LP ran
    416 s against a caller's 20 s budget. The LP-level ``time_limit`` is the only
    thing that reaches ``SimplexOptions::deadline``.
    """
    monkeypatch.delenv("DISCOPT_LP_WARM_DEADLINE", raising=False)
    m = _int_qcqp()
    relaxer = MccormickLPRelaxer(m)
    assert relaxer._inc is not None, "fast path must be active or this proves nothing"

    seen = _spy_warm_lp(monkeypatch)
    lb, ub = np.array([0.0, 0.0]), np.array([5.0, 5.0])
    relaxer.solve_at_node(lb, ub, time_limit=7.0)

    assert seen, "probe never fired: no warm LP was issued on the fast path"
    checked = 0
    for tl in seen:
        assert tl != "ABSENT", "solve_lp_warm_std called without a time_limit kwarg"
        assert tl is not None, "the caller's budget was dropped before the LP"
        # A slice of the caller's duration, never more than it, never <= 0 (the
        # backend rejects a nonpositive budget — hence the floor).
        assert 0.0 < tl <= 7.0
        checked += 1
    assert checked == len(seen) and checked > 0, "probe fired on every LP"


@pytest.mark.smoke
def test_no_time_limit_leaves_the_fast_path_unbounded_as_before(monkeypatch):
    """Bound-neutrality: with no caller budget the LP is issued exactly as it was
    — unbounded. This is the arm that must not move."""
    monkeypatch.delenv("DISCOPT_LP_WARM_DEADLINE", raising=False)
    relaxer = MccormickLPRelaxer(_int_qcqp())
    assert relaxer._inc is not None

    seen = _spy_warm_lp(monkeypatch)
    relaxer.solve_at_node(np.array([0.0, 0.0]), np.array([5.0, 5.0]))

    assert seen, "probe never fired"
    for tl in seen:
        assert tl is None, f"an unbudgeted node LP acquired a deadline: {tl}"


@pytest.mark.smoke
def test_warm_deadline_opt_out_restores_the_unbounded_fast_path(monkeypatch):
    """``DISCOPT_LP_WARM_DEADLINE=0`` is the documented escape hatch for exactly
    this guarantee; it must cover this path too, or the opt-out is a lie."""
    monkeypatch.setenv("DISCOPT_LP_WARM_DEADLINE", "0")
    relaxer = MccormickLPRelaxer(_int_qcqp())
    assert relaxer._inc is not None

    seen = _spy_warm_lp(monkeypatch)
    relaxer.solve_at_node(np.array([0.0, 0.0]), np.array([5.0, 5.0]), time_limit=7.0)

    assert seen, "probe never fired"
    for tl in seen:
        assert tl is None, f"opt-out ignored: LP still budgeted at {tl}"


@pytest.mark.smoke
def test_an_exhausted_budget_still_hands_the_lp_a_positive_floor(monkeypatch):
    """A node that starts with its budget already spent must not hand the backend
    a zero/negative duration (it would reject it, turning a bounded solve into an
    error). ``_SOLVE_DEADLINE_FLOOR_S`` is the guard."""
    from discopt._relax.mccormick_lp import _SOLVE_DEADLINE_FLOOR_S

    monkeypatch.delenv("DISCOPT_LP_WARM_DEADLINE", raising=False)
    relaxer = MccormickLPRelaxer(_int_qcqp())
    assert relaxer._inc is not None

    seen = _spy_warm_lp(monkeypatch)
    # An absolute round grant that expired in the past — the tightest possible case.
    relaxer.solve_at_node(
        np.array([0.0, 0.0]),
        np.array([5.0, 5.0]),
        time_limit=7.0,
        round_deadline=time.perf_counter() - 100.0,
    )

    assert seen, "probe never fired"
    for tl in seen:
        # Every LP issued under an expired grant collapses to the floor, not to the
        # caller's full 7 s. (The cold fallback re-slices the same floor against its
        # own elapsed time, so this is an upper bound, not an equality.)
        assert 0.0 < tl <= _SOLVE_DEADLINE_FLOOR_S + 1e-9, f"floor not applied: {tl}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
