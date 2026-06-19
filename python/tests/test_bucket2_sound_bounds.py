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
_SOUND_BOUND_CASES = [
    ("ex1225", 31.0),
    ("ex1226", -17.0),
    ("ex1252", 128893.8),
    ("nvs05", 5.47093),
    ("nvs20", 230.922),
    ("nvs22", 6.0584),
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


# Per-node lifted-LP FBBT (issue #184). ``ex1252``'s root bound is structurally 0
# and no envelope/RLT cut can move it (the objective products are nonnegative and
# zeroable on the box boundary). The bound only lifts once branching fixes the
# line-selection binaries ``x36/x37/x38`` (≡ ``x18/x19/x20``): then propagating
# the relaxation's *own* McCormick rows recovers the bilinear-implied factor
# bounds (``x18=1`` ⟹ ``x9·x3=400`` ⟹ ``x3>=1``; ``x21=1`` ⟹ ``x0>=1``) that
# purely linear FBBT misses, and rebuilding on the tightened box gives an
# envelope that no longer underestimates the product to 0.
_EX1252_LINE_BINARIES = (36, 37, 38)  # x18=x36, x19=x37, x20=x38 (constraints 40-42)


@pytest.mark.correctness
def test_ex1252_lifted_fbbt_tightens_branched_node(monkeypatch):
    """At the branched node ``x36=1, x37=0, x38=0`` (line 1 selected), per-node
    lifted-LP FBBT (``DISCOPT_LIFTED_FBBT=1``) drives the node bound off the
    structural 0 — to ~18987 — while staying sound (``<= optimum 128893.8``). The
    root bound is unchanged (binaries still relaxed there, so FBBT derives
    nothing): the tightening is genuinely per-node, which is what lets the global
    B&B certify optimality. This is the regression lock for #184.
    """
    monkeypatch.setenv("DISCOPT_LIFTED_FBBT", "1")
    nl = _DATA / "ex1252.nl"
    assert nl.exists(), f"missing {nl}"
    m = dm.from_nl(str(nl))

    relaxer = MccormickLPRelaxer(m)
    lb, ub = flat_variable_bounds(m)

    # Root: binaries relaxed, so lifted FBBT cannot tighten — bound stays ~0.
    root = relaxer.solve_at_node(lb, ub)
    assert root.status == "optimal", f"ex1252 root LP status {root.status}"
    assert root.lower_bound is not None and root.lower_bound <= 1.0, (
        f"ex1252 root bound unexpectedly {root.lower_bound} (expected ~0)"
    )

    # Branched node: fix line-selection binaries (x36=1, x37=x38=0).
    nlb, nub = lb.copy(), ub.copy()
    for k in _EX1252_LINE_BINARIES:
        val = 1.0 if k == 36 else 0.0
        nlb[k] = nub[k] = val
    node = relaxer.solve_at_node(nlb, nub)

    assert node.status == "optimal", f"ex1252 branched LP status {node.status}"
    assert node.lower_bound is not None and math.isfinite(node.lower_bound), (
        "ex1252 branched node dropped its objective bound"
    )
    # Tightness lock: the bound must climb well off 0 (validated ~18987).
    assert node.lower_bound > 1000.0, (
        f"ex1252 lifted-FBBT bound {node.lower_bound} did not lift off the "
        "structural 0 — per-node tightening regressed"
    )
    # Soundness: a valid lower bound never exceeds the true optimum.
    assert node.lower_bound <= 128893.8 + 1e-3, (
        f"ex1252 lifted-FBBT UNSOUND bound {node.lower_bound} > optimum 128893.8"
    )


@pytest.mark.correctness
@pytest.mark.parametrize("instance, optimum", _SOUND_BOUND_CASES)
def test_bucket2_lifted_fbbt_stays_sound(monkeypatch, instance, optimum):
    """Enabling per-node lifted-LP FBBT keeps every bucket-2 root bound sound
    (finite and ``<= optimum``). Lifted FBBT only ever tightens with valid
    relaxation rows, so no instance may gain an unsound (over-tight) bound."""
    monkeypatch.setenv("DISCOPT_LIFTED_FBBT", "1")
    nl = _DATA / f"{instance}.nl"
    assert nl.exists(), f"missing {nl}"
    m = dm.from_nl(str(nl))

    relaxer = MccormickLPRelaxer(m)
    lb, ub = flat_variable_bounds(m)
    res = relaxer.solve_at_node(lb, ub)

    assert res.status == "optimal", f"[{instance}] lifted-FBBT root LP status {res.status}"
    assert res.lower_bound is not None and math.isfinite(res.lower_bound), (
        f"[{instance}] lifted-FBBT dropped the root bound"
    )
    assert res.lower_bound <= optimum + 1e-3, (
        f"[{instance}] lifted-FBBT UNSOUND root bound {res.lower_bound} > optimum {optimum}"
    )


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
    # The recovered bound must match the HiGHS reference (a valid, sound relaxation
    # bound — far below the -1100 optimum, but finite and correct).
    highs = milp.solve(backend="highs")
    assert highs.status == "optimal"
    assert abs(float(simplex.bound) - float(highs.bound)) <= 1e-3 + 1e-6 * abs(float(highs.bound))
