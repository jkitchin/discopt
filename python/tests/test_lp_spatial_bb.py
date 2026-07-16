"""LP-node spatial branch-and-bound engine (discopt#280: SCIP-grade integer
products). Pins correctness (matches brute force / valid dual bound) and the
scope gate. See docs/dev/scip-gap-nvs-diagnosis.md."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._jax.lp_spatial_bb import _separate_node_cuts, solve_lp_spatial_bb  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib")


def _assemble_node_lp(ux=6, uy=5, uz=4):
    """Build a small all-integer bilinear node LP that drives the incremental
    McCormick path (``inc.ok``) and yields at least one GMI cut. Returns
    ``(inc, A, b, bounds, x, ncol)`` for the root box."""
    from discopt._jax.incremental_mccormick import IncrementalMcCormickLP
    from discopt._jax.term_classifier import classify_nonlinear_terms

    m = dm.Model("c10")
    a = m.integer("a", lb=0, ub=ux)
    b = m.integer("b", lb=0, ub=uy)
    c = m.integer("c", lb=0, ub=uz)
    m.minimize(a + b + c)
    m.subject_to(a * b + c >= 10)
    terms = classify_nonlinear_terms(m)
    inc = IncrementalMcCormickLP(m, terms)
    assert inc.ok, "incremental McCormick path required for the GMI cut branch"
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([float(ux), float(uy), float(uz)])
    A, b_, bounds = inc.assemble(lb, ub, [])
    # ``inc.assemble`` now returns a sparse CSR ``A``; this test's dense GMI reference
    # (``_raw_gmi_cuts``: ``np.hstack([A, np.eye])`` etc.) needs a dense array, so
    # densify here (small node LP) to reproduce the pre-sparse-incremental contract.
    import scipy.sparse as sp

    if sp.issparse(A):
        A = A.toarray()
    bd, x, _ = inc.solve_assembled(A, b_, bounds)
    assert x is not None
    return inc, A, b_, bounds, x, inc.ncol


def _raw_gmi_cuts(inc, A, b, bounds, x, ncol):
    """Reproduce the *unmargined* GMI cuts exactly as ``_separate_node_cuts``'s
    GMI branch derives them (crossover vertex -> Rust GMI -> negate to <=), so a
    test can assert the emitted cut carries the safety margin the raw cut lacks."""
    from discopt._jax.crossover import crossover_to_vertex
    from discopt._jax.problem_classifier import LPData
    from discopt.solver import _separate_gomory_cuts

    m_rows = A.shape[0]
    A_eq = np.hstack([A, np.eye(m_rows)])
    cc = np.concatenate([np.asarray(inc.c, dtype=np.float64)[:ncol], np.zeros(m_rows)])
    xl = np.concatenate([bounds[:, 0], np.zeros(m_rows)])
    xu = np.concatenate([bounds[:, 1], np.full(m_rows, 1e20)])
    xv = crossover_to_vertex(np.concatenate([x, b - A @ x]), A_eq, b.copy(), cc, xl, xu)
    gc = _separate_gomory_cuts(LPData(cc, A_eq, b.copy(), xl, xu, 0.0), xv, ncol, list(range(ncol)))
    if gc is None:
        return []
    return [(-np.asarray(gc[0][i])[:ncol], -float(gc[1][i])) for i in range(len(gc[1]))]


# --------------------------------------------------------------------------- #
# C-10: LP-spatial GMI cuts must carry the machine-precision rhs safety margin #
# every other GMI consumer applies, else a cut whose boundary passes through a #
# feasible integer point can shave it (invalid cut -> false bound).            #
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_c10_lp_spatial_gmi_cut_carries_safety_margin():
    """Fail-before/pass-after (C-10): each GMI cut ``_separate_node_cuts`` emits
    must relax its ``<=`` rhs outward by exactly ``1e-7*(1+||row||_1)`` versus the
    raw (unmargined) GMI cut. Pre-fix the emitted rhs equals the raw rhs (no
    margin) and this fails; post-fix it is relaxed by the margin."""
    inc, A, b, bounds, x, ncol = _assemble_node_lp()
    raw = _raw_gmi_cuts(inc, A, b, bounds, x, ncol)
    assert raw, "expected at least one raw GMI cut on this node"
    emitted = _separate_node_cuts(A, b, bounds, x, ncol, inc.c)
    # the GMI cuts are appended first, in order, so emitted[:len(raw)] are the GMI rows
    assert len(emitted) >= len(raw)
    for (row_raw, rhs_raw), (row_emit, rhs_emit) in zip(raw, emitted[: len(raw)]):
        assert np.allclose(row_emit, row_raw)  # coefficients unchanged
        margin = 1e-7 * (1.0 + float(np.abs(np.asarray(row_raw)).sum()))
        # rhs is relaxed OUTWARD (>=) by the margin, never left raw and never tightened
        assert rhs_emit == pytest.approx(rhs_raw + margin, abs=1e-15)
        assert rhs_emit > rhs_raw


@pytest.mark.smoke
def test_c10_no_feasible_integer_point_is_cut():
    """Property (C-10 class): no integer-feasible point of the model is violated by
    any emitted node cut (GMI, MIR, ...). Encodes the invariant a valid cut must
    satisfy — the margin guarantees float error can't shave a boundary point."""
    ux, uy, uz = 6, 5, 4
    inc, A, b, bounds, x, ncol = _assemble_node_lp(ux, uy, uz)
    cuts = _separate_node_cuts(A, b, bounds, x, ncol, inc.c)
    assert cuts, "expected node cuts to exercise the property"
    # aux column layout for this model: [a, b, c, w=a*b]
    for ai in range(ux + 1):
        for bi in range(uy + 1):
            for ci in range(uz + 1):
                if ai * bi + ci < 10:
                    continue  # infeasible for the original model
                pt = np.array([ai, bi, ci, ai * bi], dtype=float)[:ncol]
                for co, rhs in cuts:
                    co = np.asarray(co, dtype=float)
                    assert float(co @ pt) <= rhs + 1e-9, (
                        f"cut {co}·x<= {rhs} shaves feasible point {(ai, bi, ci)}"
                    )


def _tiny(ux, uy, rhs, coef):
    m = dm.Model("t")
    a = m.integer("a", lb=0, ub=ux)
    b = m.integer("b", lb=0, ub=uy)
    m.minimize(a + coef * b)
    m.subject_to(a * b >= rhs)
    return m


def _brute(ux, uy, rhs, coef):
    return min(
        (a + coef * b for a in range(ux + 1) for b in range(uy + 1) if a * b >= rhs),
        default=None,
    )


@pytest.mark.smoke
def test_no_false_optimum_on_dense_integer_quadratic():
    """Soundness invariant (fail-before/pass-after for the #636 univariate-square
    regression): on a dense all-integer quadratic the engine must never report an
    objective *below* the true optimum, never a dual bound *above* it, and never
    declare ``optimal`` at a suboptimal point.

    The engine's incumbent objective used to be the McCormick relaxation value at a
    collapsed box; once a bilinear product is lifted outside ``info`` (univariate-
    square post-#636) that value is loose, and ``_worst_product_var`` cannot see the
    product to branch it, so the loose bound was accepted as a certified optimum
    (nvs17: -1836.2 vs true -1100.4). This small dense model drives the same
    fallback path (``inc.ok`` is False), so the exact-verify + unresolved-bound
    invariant is exercised without the slow nvs17 solve."""
    n, ub = 3, 5
    m = dm.Model("dq")
    xs = [m.integer(f"x{i}", lb=0, ub=ub) for i in range(n)]
    obj = 0
    for i in range(n):
        obj = obj - xs[i] * xs[i]
        for j in range(i + 1, n):
            obj = obj - xs[i] * xs[j]
    m.minimize(obj)
    m.subject_to(sum(xs) <= 2 * n)

    brute = min(
        -(a * a + b * b + c * c + a * b + a * c + b * c)
        for a in range(ub + 1)
        for b in range(ub + 1)
        for c in range(ub + 1)
        if a + b + c <= 2 * n
    )
    r = solve_lp_spatial_bb(m, time_limit=20, gap_tolerance=1e-6)
    assert r is not None
    if r.objective is not None:
        assert r.objective >= brute - 1e-6  # incumbent can never beat the true optimum
    if r.bound is not None:
        assert r.bound <= brute + 1e-6  # dual bound is a valid lower bound
    if r.status == "optimal":  # a claimed proof must be a real one
        assert r.objective == pytest.approx(brute, abs=1e-6)


# --------------------------------------------------------------------------- #
# scope gate (pure logic, fast)
# --------------------------------------------------------------------------- #


def test_out_of_scope_continuous_returns_none():
    """A model with a continuous variable is out of scope -> None (caller falls
    back). The collapsed-box exactness argument needs every var integer."""
    m = dm.Model("c")
    x = m.continuous("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.minimize(x + y)
    m.subject_to(x * y >= 4)
    assert solve_lp_spatial_bb(m, time_limit=5) is None


def test_out_of_scope_maximize_returns_none():
    """The McCormick relaxation bound is only a valid *lower* bound for minimize."""
    m = dm.Model("mx")
    a = m.integer("a", lb=0, ub=5)
    b = m.integer("b", lb=0, ub=5)
    m.maximize(a + b)
    m.subject_to(a * b <= 6)
    assert solve_lp_spatial_bb(m, time_limit=5) is None


# --------------------------------------------------------------------------- #
# correctness: matches brute force exactly, and proves optimality
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.requires_pounce
@pytest.mark.parametrize(
    "ux,uy,rhs,coef",
    [(5, 4, 7, 1), (6, 6, 20, 2), (8, 4, 15, 1), (7, 3, 11, 3)],
)
def test_matches_brute_force(ux, uy, rhs, coef):
    r = solve_lp_spatial_bb(_tiny(ux, uy, rhs, coef), time_limit=20, gap_tolerance=1e-6)
    assert r is not None and r.status == "optimal"
    assert r.objective == pytest.approx(_brute(ux, uy, rhs, coef), abs=1e-6)


@pytest.mark.slow
@pytest.mark.requires_pounce
def test_nvs17_dual_bound_is_valid_and_tight():
    """On nvs17 the LP-node engine must (a) never report a dual bound *above* the
    true optimum (soundness) and (b) get far closer than the default path's frozen
    root value (-65842). Full closure needs cuts (a later step)."""
    path = os.path.join(_DATA, "nvs17.nl")
    if not os.path.exists(path):
        pytest.skip("nvs17 unavailable")
    r = solve_lp_spatial_bb(dm.from_nl(path), time_limit=45, gap_tolerance=1e-4)
    assert r is not None
    assert r.bound is not None and r.bound <= -1100.4 + 1e-4  # valid lower bound
    assert r.bound >= -2000.0  # vastly tighter than the -65842 frozen root
    if r.objective is not None:
        assert r.objective >= -1100.4 - 1e-4  # incumbent can't beat the true optimum
