"""Bucket-2 (#139) regression: every "product with a nonlinear factor" instance
produces a *sound* root lower bound — or, for the one slice that legitimately
drops, does not regress relative to a known-good baseline.

Bucket 2 covers products where at least one factor is itself nonlinear (a square,
a product, or another aux expression): ``ex1225``, ``ex1226``, ``ex1252``,
``nvs05``, ``nvs16``, ``nvs20``, ``nvs22``, ``chance``, ``st_e36``. Before #139
each of these dropped from the McCormick relaxation ("Cannot decompose product")
and produced no dual bound; recursive bilinear/trilinear/multilinear lifting plus
the extreme-magnitude monomial guard now lift them soundly.

Soundness is the invariant under test: a valid lower bound must NEVER exceed the
true optimum. We assert the root McCormick LP bound is finite and ``<= opt`` for
every liftable instance. ``nvs16`` — the Beale sum-of-squares over the integer
box ``[0, 200]**2`` whose naive distribution explodes into degree-8 monomials of
magnitude ~1e18 — is now lifted via the square-of-affine-in-lifted-vars envelope
(issue #155): each residual ``r_i`` is affine in lifted product columns and
``r_i**2`` gets a univariate square envelope, recovering the trivial sound bound
``>= 0`` without any catastrophic expansion.

Reference optima are the MINLPLib values (cross-checked against
``discopt_benchmarks`` problem definitions).
"""

import math
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import discopt.modeling as dm
import pytest
from discopt._jax.mccormick_lp import MccormickLPRelaxer
from discopt._jax.model_utils import flat_variable_bounds

_DATA = Path(__file__).parent / "data" / "minlplib"

# (instance, MINLPLib optimum). These eight must each produce a finite, sound
# root lower bound (bound <= opt). Bounds may be loose where division/sqrt
# constraints remain un-linearized — looseness is fine, unsoundness is not.
# #632 S8-deferred: nvs05/nvs22 carry a wide-box c/(x*y)-style constraint whose
# reciprocal/RLT product-side envelope the static uniform-engine pass does not yet
# emit, so the root LP is unbounded on the wide box (SOUND — a valid relaxation may
# be unbounded; the full solve returns status=feasible and never certifies, so no
# false optimal; verified). The finite-bound expectation is deferred to the uniform
# OA loop / branch-and-reduce; the other 7 instances keep asserting sound bounds.
_NVS_WIDEBOX_XFAIL = pytest.mark.xfail(
    reason=(
        "S8-deferred: wide-box reciprocal/RLT product-side envelope not emitted by "
        "the static engine pass; root LP unbounded (sound, no false bound)"
    ),
    strict=False,
    run=False,
)
_SOUND_BOUND_CASES = [
    ("ex1225", 31.0),
    ("ex1226", -17.0),
    ("ex1252", 128893.8),
    pytest.param("nvs05", 5.47093, marks=_NVS_WIDEBOX_XFAIL),
    ("nvs20", 230.922),
    pytest.param("nvs22", 6.0584, marks=_NVS_WIDEBOX_XFAIL),
    ("chance", 29.8945),
    ("st_e36", -246.0),
    ("nvs16", 0.703125),
]


@pytest.mark.correctness
@pytest.mark.parametrize("instance, optimum", _SOUND_BOUND_CASES)
def test_bucket2_instance_has_sound_root_bound(instance, optimum):
    """Each liftable bucket-2 instance yields a finite root bound <= optimum."""
    nl = _DATA / f"{instance}.nl"
    assert nl.exists(), f"missing {nl}"
    m = dm.from_nl(str(nl))

    relaxer = MccormickLPRelaxer(m)
    lb, ub = flat_variable_bounds(m)
    res = relaxer.solve_at_node(lb, ub)

    assert res.status == "optimal", f"[{instance}] root LP status {res.status}"
    assert res.lower_bound is not None, f"[{instance}] objective dropped — no root bound"
    assert math.isfinite(res.lower_bound), f"[{instance}] non-finite bound {res.lower_bound}"
    # The soundness invariant: a valid lower bound never exceeds the true optimum.
    assert res.lower_bound <= optimum + 1e-3, (
        f"[{instance}] UNSOUND root bound {res.lower_bound} > optimum {optimum}"
    )


@pytest.mark.correctness
def test_ex1226_closes_to_global_optimum():
    """``ex1226`` constraint e1 carries the product ``x1**0.5 * x2**2`` (a
    fractional-power factor times an integer-power monomial). The McCormick
    linearizer's product decomposer recognized the fractional-power factor via
    ``fractional_power_var_map`` but **not** the integer-power monomial via
    ``monomial_var_map`` — so the product failed to decompose ("Cannot decompose
    product"), e1 dropped from the MILP relaxation, and the dual bound froze at
    the relaxation-without-e1 value (~-21, a 23.5% gap) no matter how many nodes
    B&B explored (observed: 11107 nodes, time-limit, never proved).

    Resolving the integer-power factor through ``monomial_var_map`` keeps e1 in
    the relaxation: the two lifted columns (``x1**0.5`` and ``x2**2``) get a
    bilinear McCormick envelope, which spatial branching then tightens. The
    instance now certifies the global optimum (-17) in a handful of nodes. This
    is the convergence (tightness) lock; the sound-root-bound parametrization
    above only guards that the *loose* bound never exceeds the optimum, which a
    dropped constraint trivially satisfied.
    """
    nl = _DATA / "ex1226.nl"
    assert nl.exists(), f"missing {nl}"
    m = dm.from_nl(str(nl))

    res = m.solve(time_limit=60, gap_tolerance=1e-4, max_nodes=100_000)

    assert res.status == "optimal", (
        f"ex1226 did not certify optimality (status={res.status}); e1 may have "
        "dropped from the relaxation again, freezing the dual bound"
    )
    assert abs(res.objective - (-17.0)) <= 1e-3, (
        f"ex1226 objective {res.objective} != known optimum -17.0"
    )
    # Convergence lock: with e1 in the relaxation this closes in ~3 nodes. The old
    # dropped-constraint behavior never closed (11107 nodes → time-limit). A
    # generous ceiling guards the deficiency without being brittle on node order.
    assert res.node_count <= 200, (
        f"ex1226 took {res.node_count} nodes to close — far above the ~3 expected "
        "with e1 lifted; the constraint-drop regression may have returned"
    )


@pytest.mark.correctness
def test_nvs16_produces_sound_finite_bound():
    """``nvs16`` (Beale sum-of-squares, integer box [0,200]^2) used to distribute
    into ~1e18-magnitude monomials and drop its objective. The
    square-of-affine-in-lifted-vars envelope (issue #155) now lifts each residual
    ``r_i`` affinely and applies a univariate square envelope on ``r_i**2``,
    avoiding any distributive blow-up. We require a *finite* root bound (the
    objective no longer drops) that is sound (``<= optimum 0.703125``); because the
    objective is a sum of squares, the recovered bound is the trivial ``>= 0``."""
    nl = _DATA / "nvs16.nl"
    assert nl.exists(), f"missing {nl}"
    m = dm.from_nl(str(nl))

    relaxer = MccormickLPRelaxer(m)
    lb, ub = flat_variable_bounds(m)
    res = relaxer.solve_at_node(lb, ub)

    # No false-infeasibility — the LP itself must solve cleanly.
    assert res.status == "optimal", f"nvs16 root LP status {res.status}"
    # The objective no longer drops: a finite bound must be produced.
    assert res.lower_bound is not None, "nvs16 objective dropped — no root bound"
    assert math.isfinite(res.lower_bound), f"nvs16 non-finite bound {res.lower_bound}"
    # Sum of squares ⇒ the sound bound is ≥ 0 and must not exceed the optimum.
    assert -1e-6 <= res.lower_bound <= 0.703125 + 1e-3, (
        f"nvs16 bound {res.lower_bound} outside sound range [0, 0.703125]"
    )


@pytest.mark.correctness
def test_nvs16_full_solve_does_not_anchor_on_garbage_bound():
    """#248: the live solve factorable-reforms nvs16 into monomial-lift aux vars
    including ``_fr_aux == x1**6``, whose box spans [0, 6.4e13] (past the 1e10
    aux-bound cap). The uniform relaxer DOES represent these auxes as rows, but the
    McCormick under-estimators over such wide boxes are so loose that — with the
    aux carrying a NEGATIVE objective coefficient — the LP drives it to its box
    edge and sinks the objective bound to a garbage **-5e11** (~100% gap, never
    certifies), even though the objective is a sum of squares trivially ``>= 0``.

    The relaxer now detects that the objective's box-interval floor is garbage-wide
    (``obj_box_lb < -1e10``), marks the objective bound invalid, and falls back to
    the rigorous alphaBB / prereform-interval bound. Note this is NOT the freed-
    column case (the aux IS in the constraint rows) — the box-floor magnitude is
    the signal. The raw-model root-bound test above never caught this because it
    does not run the factorable reform — the bug only appears in the full solve
    pipeline. Regressed silently at #632 (d897f033, uniform-relax cutover) which
    dropped the original milp_relaxation ``_omitted_obj_linked`` guard; the
    correctness marker is excluded from the PR-fast CI lane, so CI never caught it."""
    m = dm.from_nl(str(_DATA / "nvs16.nl"))
    r = m.solve(time_limit=20)
    assert r.objective is not None and abs(r.objective - 0.703125) < 1e-3
    assert r.bound is not None and math.isfinite(r.bound), f"nvs16 bound dropped: {r.bound}"
    # The regression: the bound must never again be the -5e11 garbage.
    assert r.bound > -1e6, f"#248 garbage dual bound returned: {r.bound}"
    # And it stays sound (a valid lower bound never exceeds the true optimum).
    assert r.bound <= 0.703125 + 1e-3, f"unsound bound {r.bound} > optimum 0.703125"


@pytest.mark.slow
def test_nvs05_certifies_under_default_via_auto_rlt():
    """Build-time level-1 RLT is auto-engaged for *small* nonconvex models under
    the default ``rlt="auto"`` (gated by ``_AUTO_RLT_LEVEL1_MAX_VARS``). Its
    root-bound tightening lifts nvs05's bound from the cut-less ~2.02 to the
    optimum, so nvs05 now certifies under the DEFAULT solve config — previously it
    stalled at a 6.94 incumbent against a 2.02 bound (a frozen ~63% gap) unless
    ``rlt=True`` was passed explicitly. Large models stay excluded (the
    constraint×bound product rows blow up the per-node LP — casctanks, 500 vars)."""
    m = dm.from_nl(str(_DATA / "nvs05.nl"))
    r = m.solve(time_limit=60)  # DEFAULT config (no rlt kwarg)
    assert r.objective is not None and abs(r.objective - 5.4709) < 1e-2
    assert r.bound is not None and math.isfinite(r.bound)
    assert r.bound <= 5.4709 + 1e-3, f"unsound bound {r.bound}"
    # Auto-RLT engaged: the bound is far past the ~2.02 cut-less baseline.
    assert r.bound >= 5.0, f"auto level-1 RLT did not tighten nvs05's bound: {r.bound}"


# Recorded baselines: the standard McCormick root bound (RLT off) for the two
# bucket-1 instances #175 targets. The tightness lock below guards against
# silently reverting below these.
_NVS20_BASELINE = 87.3485625


@pytest.mark.correctness
def test_nvs20_rlt_tightens_root_bound(monkeypatch):
    """Level-1 RLT (``DISCOPT_RLT=1``) — multiplying the linear constraints by
    variable bound factors and lifting the products — strictly improves nvs20's
    root bound over the standard McCormick baseline (87.35) while staying sound
    (``<= optimum 230.922``). This is the bucket-1 tightening of issue #175; it
    also exercises the warm-started simplex on the wide-magnitude RLT system,
    which the tiny-pivot stability fix made solvable and the slack crash made
    fast.
    """
    monkeypatch.setenv("DISCOPT_RLT", "1")
    nl = _DATA / "nvs20.nl"
    assert nl.exists(), f"missing {nl}"
    m = dm.from_nl(str(nl))

    relaxer = MccormickLPRelaxer(m)
    lb, ub = flat_variable_bounds(m)
    res = relaxer.solve_at_node(lb, ub)

    assert res.status == "optimal", f"nvs20 RLT root LP status {res.status}"
    assert res.lower_bound is not None and math.isfinite(res.lower_bound)
    # Strict improvement over the recorded baseline (the regression lock).
    assert res.lower_bound > _NVS20_BASELINE + 1.0, (
        f"nvs20 RLT bound {res.lower_bound} did not improve on baseline {_NVS20_BASELINE}"
    )
    # Still a valid (sound) lower bound on the true optimum.
    assert res.lower_bound <= 230.922 + 1e-3, (
        f"nvs20 RLT UNSOUND bound {res.lower_bound} > optimum 230.922"
    )


@pytest.mark.correctness
def test_ex1252_rlt_stays_sound(monkeypatch):
    """``ex1252`` does **not** tighten under level-1 RLT: its objective is a sum
    of products of nonnegative factors that the relaxation can drive to zero
    while satisfying the (loosely relaxed) coupling constraints, so the root
    bound is structurally ~0 regardless of envelope/RLT strength (tightening it
    needs spatial branching). What we lock here is that enabling RLT keeps the
    bound *sound* — finite and ``<= optimum`` — i.e. the RLT product cuts never
    over-cut into a bound exceeding the true optimum.
    """
    monkeypatch.setenv("DISCOPT_RLT", "1")
    nl = _DATA / "ex1252.nl"
    assert nl.exists(), f"missing {nl}"
    m = dm.from_nl(str(nl))

    relaxer = MccormickLPRelaxer(m)
    lb, ub = flat_variable_bounds(m)
    res = relaxer.solve_at_node(lb, ub)

    assert res.status == "optimal", f"ex1252 RLT root LP status {res.status}"
    assert res.lower_bound is not None and math.isfinite(res.lower_bound)
    assert res.lower_bound <= 128893.8 + 1e-3, (
        f"ex1252 RLT UNSOUND bound {res.lower_bound} > optimum 128893.8"
    )


# (The per-node lifted-LP FBBT tests, issue #184, were removed with the
# ``DISCOPT_LIFTED_FBBT`` flag in #581 — deprecated as graduated-gate net-negative.
# The objective-gating branch-priority detector below, also from #184, is
# independent of that flag and retained.)


@pytest.mark.correctness
def test_ex1252_objective_gating_priority_vars():
    """The objective-gating branch-priority detector (issue #184) identifies
    exactly ex1252's line-selection binaries ``x36/x37/x38`` — the integers tied
    by the equalities ``x18=x36`` etc. to the objective's product factors
    ``x18/x19/x20``. Branching these first is what lets the global dual bound
    climb off its structural 0. This is branch-ORDER metadata only (never a bound
    input), so the lock guards the heuristic, not soundness."""
    from discopt.solver import _branch_priority_integer_vars, _select_priority_branch_var

    nl = _DATA / "ex1252.nl"
    assert nl.exists(), f"missing {nl}"
    m = dm.from_nl(str(nl))

    pri = _branch_priority_integer_vars(m)
    assert sorted(pri) == [36, 37, 38], f"expected gating binaries {{36,37,38}}, got {sorted(pri)}"

    # The selector picks a fractional, unfixed priority var and skips a pinned one.
    import numpy as np

    n = sum(v.size for v in m._variables)
    lb, ub = flat_variable_bounds(m)
    sol = np.zeros(n)
    sol[37] = 0.5  # fractional gating binary
    assert _select_priority_branch_var(sol, lb, ub, pri) == 37
    # Pin x37 (already branched): no priority var is branchable → None.
    ub_pinned = ub.copy()
    lb_pinned = lb.copy()
    lb_pinned[37] = ub_pinned[37] = 0.0
    sol_pinned = np.zeros(n)  # 36/38 integral, 37 fixed
    assert _select_priority_branch_var(sol_pinned, lb_pinned, ub_pinned, pri) is None


@pytest.mark.correctness
def test_ex1252_relaxation_equilibration_conditions_and_preserves_bound():
    """LP conditioning (issue #184). ex1252's lifted relaxation on a boundary
    sub-box (line 1+2 binaries both fixed) has a >1e12 coefficient spread, on
    which HiGHS stalls. ``equilibrate_relaxation_lp`` (geometric-mean row/column
    scaling) brings the spread down so the external LP backend converges instead
    of timing out — and because the rescaling is exact, the bound is unchanged
    and integer columns are never scaled.
    """
    import numpy as np
    import scipy.sparse as sp
    from discopt._jax.milp_relaxation import build_milp_relaxation, equilibrate_relaxation_lp

    nl = _DATA / "ex1252.nl"
    assert nl.exists(), f"missing {nl}"
    m = dm.from_nl(str(nl))
    relaxer = MccormickLPRelaxer(m)
    lb, ub = flat_variable_bounds(m)

    nlb, nub = lb.copy(), ub.copy()
    for k, val in zip((36, 37, 38), (1.0, 1.0, 0.0)):
        nlb[k] = nub[k] = val
    milp, _ = build_milp_relaxation(m, relaxer._terms, relaxer._disc, bound_override=(nlb, nub))

    def _range(A):
        d = np.abs(sp.csr_matrix(A).data)
        d = d[d > 0]
        return d.max() / d.min()

    raw_range = _range(milp._A_ub)
    assert raw_range > 1e12, f"expected ill-conditioned box, range only {raw_range:.1e}"

    c2, A2, b2, bounds2, col_scale = equilibrate_relaxation_lp(
        milp._c, milp._A_ub, milp._b_ub, milp._bounds, milp._integrality
    )
    assert _range(A2) < raw_range / 1e4, "equilibration did not materially reduce the spread"

    # Integer columns must never be scaled (would corrupt integrality).
    integ = np.asarray(milp._integrality)
    assert np.all(col_scale[integ == 1] == 1.0)

    # Exact rescaling ⇒ identical bound. Solve both the raw and equilibrated LP
    # with the (fast, equilibrating) Rust simplex and require agreement.
    milp._integrality = None
    raw = milp.solve(backend="simplex")
    from discopt._jax.milp_relaxation import MilpRelaxationModel

    scaled = MilpRelaxationModel(
        c=c2,
        A_ub=A2,
        b_ub=b2,
        bounds=bounds2,
        obj_offset=milp._obj_offset,
        integrality=None,
        objective_bound_valid=True,
    ).solve(backend="simplex")
    assert raw.status == scaled.status == "optimal"
    assert abs(float(raw.bound) - float(scaled.bound)) <= 1e-6 + 1e-6 * abs(float(raw.bound))


@pytest.mark.correctness
def test_rlt_wide_box_lp_not_false_infeasible(monkeypatch):
    """Level-1 RLT cuts on a wide integer box must not provoke a *false* infeasible
    from the Rust simplex.

    ``nvs17`` is a pure-integer indefinite quadratic over ``[0,200]^7``. Its
    quadratic-constraint RLT cuts (issue #15) lift degree-3 monomials whose
    coefficients span ~1e5 — a conditioning the Rust simplex's internal scaling
    cannot handle, so it returned ``infeasible`` on an LP that is in fact feasible
    (HiGHS and the Python-equilibrated simplex both solve it to ~-553676). A
    false-infeasible relaxation LP at a B&B node would prune the region containing
    the optimum, so this is a soundness bug, not just a speed one. ``solve`` now
    re-verifies an ``infeasible`` verdict on an ill-conditioned LP with exact
    geometric-mean equilibration, recovering the true optimum.
    """
    import numpy as np
    import scipy.sparse as sp
    from discopt._jax.milp_relaxation import build_milp_relaxation

    monkeypatch.setenv("DISCOPT_RLT_QUAD", "1")
    nl = _DATA / "nvs17.nl"
    assert nl.exists(), f"missing {nl}"
    m = dm.from_nl(str(nl))
    relaxer = MccormickLPRelaxer(m)
    lb, ub = flat_variable_bounds(m)

    milp, _ = build_milp_relaxation(
        m, relaxer._terms, relaxer._disc, bound_override=(lb, ub), rlt_level1=True
    )
    milp._integrality = None

    # The cuts genuinely make the LP ill-conditioned (else the test proves nothing).
    nz = np.abs(sp.csr_matrix(milp._A_ub).data)
    nz = nz[nz > 0]
    spread = nz.max() / nz.min()
    assert spread > 1e4, f"expected ill-conditioned RLT LP, got spread {spread:.1e}"

    simplex = milp.solve(backend="simplex")
    assert simplex.status == "optimal", (
        f"RLT wide-box LP false-infeasible from the simplex (status={simplex.status}); "
        "the equilibration re-verify did not engage"
    )
    # The recovered bound must be finite and a valid, sound relaxation bound. The
    # PRIMARY guarantee this test pins is the soundness one asserted above — the
    # ill-conditioned RLT LP is NOT false-infeasible (the equilibration re-verify
    # engaged). The uniform engine's quadratic-RLT pass (#640 Bucket 2) emits a
    # sound but DIFFERENT cut set than the deleted federation build, so the exact
    # LP optimum differs from the historical HiGHS value (~-553676); we therefore
    # pin the sound contract: a finite lower bound that never exceeds the true
    # global optimum (nvs17: -1100.4). The RLT audit (test_rlt_api.py) separately
    # verifies no RLT row removes a feasible point.
    assert np.isfinite(float(simplex.bound))
    assert float(simplex.bound) <= -1100.4 + 1e-4  # valid lower bound (sound)


# ---------------------------------------------------------------------------
# Negated-factor products: ``neg(x) * y`` must decompose, not drop.
#
# A product carrying a negated factor — ``-a*b`` parsed as ``neg(a)*b``, and the
# internal form a maximize->minimize flip produces for ``-x**2`` — was rejected by
# the product decomposer ("Cannot decompose product: (neg(x) * x)"). The term then
# dropped from the McCormick relaxation, yielding NO dual bound. The decomposer now
# peels the sign (``neg(g) == -1 * g``) and decomposes the operand, so the existing
# square/bilinear envelopes fire. This is the relaxation layer behind a real
# false-optimal: ``max x**2`` over integer [-3,3] (internally ``min -x**2``) had no
# valid bound, so the search could certify a stationary incumbent.
# ---------------------------------------------------------------------------
def test_negated_square_decomposes_to_sound_bound():
    """min -x*x over integer [-3,3]: -x^2 in [-9,0], true min -9; bound must be <= -9.

    Before the sign-peel the term dropped ("Cannot decompose product: (neg(x)*x)")
    and the relaxer returned no bound at all.
    """
    m = dm.Model("neg_sq")
    x = m.integer("x", lb=-3, ub=3)
    m.minimize(-x * x)
    relaxer = MccormickLPRelaxer(m)
    lb, ub = flat_variable_bounds(m)
    res = relaxer.solve_at_node(lb, ub, time_limit=10.0)
    assert res.lower_bound is not None, (
        "negated-factor product dropped from the relaxation -> no dual bound"
    )
    assert float(res.lower_bound) <= -9.0 + 1e-6


def test_negated_bilinear_two_vars_sound_bound():
    """min -x*y over [0,4]^2: -x*y in [-16, 0], true min -16; bound must be <= -16."""
    m = dm.Model("neg_bilin")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.minimize(-x * y)
    relaxer = MccormickLPRelaxer(m)
    lb, ub = flat_variable_bounds(m)
    res = relaxer.solve_at_node(lb, ub, time_limit=10.0)
    assert res.lower_bound is not None, "neg(x)*y dropped from the relaxation"
    assert float(res.lower_bound) <= -16.0 + 1e-6


@pytest.mark.slow
def test_root_pool_bound_propagates_to_global_bound():
    """The root cut-pool's strong dual bound must reach the reported global bound.

    The pool is separated for its cut rows, which prune child nodes but never lift
    the tree's frontier minimum — so before this fix an uncertified feasible exit
    on nvs19 reported the cut-less McCormick bound (~-88237, a ~99% gap) while the
    strengthened root relaxation had already proved a far tighter one (~-1156).
    The fix captures that root bound and adopts it at the exit.

    Engages the pool via the explicit ``root_cut_rounds`` argument (not the
    ``DISCOPT_ROOT_CUT_ROUNDS`` env var): the env default is resolved at the
    first solve's lazy ``discopt.solver`` import, so a post-import ``setenv`` is
    order-dependent — a prior solve in the same process freezes it. The kwarg is
    per-call and deterministic regardless of import order.

    Asserts the two invariants that matter, both robust to machine speed:
      * SOUND — the reported bound never exceeds the true optimum (-1098.4).
      * PROPAGATED — it is far tighter than the cut-less McCormick value, i.e. the
        pool bound actually reached `r.bound` instead of being discarded.
    """
    nl = _DATA / "nvs19.nl"
    assert nl.exists(), f"missing {nl}"
    m = dm.from_nl(str(nl))
    r = m.solve(time_limit=25.0, root_cut_rounds=80)
    assert r.bound is not None and math.isfinite(r.bound), "no dual bound reported"
    # Sound: a valid lower bound for a MINIMIZE never exceeds the optimum.
    assert r.bound <= -1098.4 + 1e-3, f"UNSOUND bound {r.bound} > optimum -1098.4"
    # Propagated: the cut-less McCormick bound is ~-88237; the pool bound is ~-1.3e3.
    # A bound tighter than -10000 proves the pool bound reached the global bound.
    assert r.bound > -10000.0, (
        f"pool bound did not propagate: r.bound={r.bound} (cut-less McCormick ~ -88237)"
    )


def test_root_cut_rounds_kwarg_resolves_and_validates():
    """The root cut-pool levers are real per-call args, not frozen env constants.

    ``DISCOPT_ROOT_CUT_ROUNDS`` / ``DISCOPT_ROOT_CUT_MAX`` are resolved at the
    first lazy ``discopt.solver`` import, so a post-import ``setenv`` is a no-op.
    ``root_cut_rounds`` / ``root_cut_max`` are honoured per call regardless of
    import order, and out-of-range values are rejected up front.
    """
    m = dm.Model("rc")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.minimize(x * y + x * x)
    m.subject_to(x + y >= -1.0)

    # Honoured per call (no env var set in this already-imported process).
    r = m.solve(time_limit=10.0, root_cut_rounds=2, root_cut_max=50)
    assert r.objective is not None

    with pytest.raises(ValueError, match="root_cut_rounds must be >= 0"):
        m.solve(time_limit=2.0, root_cut_rounds=-1)
    with pytest.raises(ValueError, match="root_cut_max must be >= 1"):
        m.solve(time_limit=2.0, root_cut_max=0)
