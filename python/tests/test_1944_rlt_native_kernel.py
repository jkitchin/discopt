"""#1944: the RLT families SCIP's separator uses, made available to the native kernel.

Measured on SCIP 10 at ``limits/nodes = 1``, every RLT cut SCIP applies at the
root of nous1/nous2 is a *linear equality x variable* row over existing products,
and every one at tln4 is a *linear inequality x bound factor* row over existing
products. Three default-OFF pieces put those rows in front of the native spatial
kernel, and each one gets a test family here:

1. ``DISCOPT_DISTRIBUTE_AFFINE_PRODUCTS``: a product of two affine forms over
   disjoint variables becomes a sum of bilinear products. They share columns with
   the same products written elsewhere, and the result is never looser. Without
   it, nous1's objective ``(c + a'x) * x0`` is one product, and the lineq rows
   have no ``x2*x0`` column to use.
2. The kernel producer no longer declines a model whose ``DISCOPT_RLT_LINEQ`` rows
   look like envelope rows. ``x36 == 0`` times ``x_j`` is the one-column row
   ``X_{36,j} == 0``, and the producer's support-based claim counted it as a
   fifth McCormick row.
3. ``DISCOPT_RLT_INEQ``: bound-factor rows for linear inequalities, built at the
   kernel's root box. They are valid in every sub-box of it, and they must come
   from the real-box build, never from the producer's probe build.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.spatial_producer import build_spatial_kernel_spec, solve_with_native_kernel
from discopt._relax.uniform_relax import build_uniform_relaxation
from scipy.optimize import linprog

pytestmark = [pytest.mark.relaxation]


def _lp_bound(rel) -> float:
    """Root LP value of a uniform relaxation, in model units (HiGHS, for runtime)."""
    M = rel.model
    bnds = [
        (float(lo) if np.isfinite(lo) else None, float(hi) if np.isfinite(hi) else None)
        for lo, hi in np.asarray(M._bounds, dtype=float)
    ]
    res = linprog(
        np.asarray(M._c, dtype=float).ravel(),
        A_ub=sp.csr_matrix(M._A_ub),
        b_ub=np.asarray(M._b_ub, dtype=float).ravel(),
        bounds=bnds,
        method="highs",
    )
    assert res.status == 0, res.message
    return rel.obj_sense_sign * (float(res.fun) + rel.obj_offset)


def _product_cols(rel) -> dict[int, tuple[int, int]]:
    cols: dict[int, tuple[int, int]] = {}
    for (a, b), col in rel.bilinear_map.items():
        cols[int(col)] = (int(a), int(b))
    for (i, p), col in rel.monomial_map.items():
        if p == 2:
            cols[int(col)] = (int(i), int(i))
    return cols


def _lifted(rel, x: np.ndarray) -> np.ndarray:
    z = np.zeros(len(rel.model._bounds))
    z[: rel.n_orig] = x
    for col, (a, b) in _product_cols(rel).items():
        z[col] = x[a] * x[b]
    return z


# --------------------------------------------------------------------------- #
# 1. distribution of affine x affine products
# --------------------------------------------------------------------------- #
def _shared_product_model(seed: int):
    """``min (c + a'x) * y`` with ``w_k == x_k * y`` constraints on some of the x.

    This is the nous1 objective's shape in general form. The distributed pairs
    ``x_k * y`` are the same products the constraints define.
    """
    rng = np.random.default_rng(seed)
    m = dm.Model()
    x = [m.continuous(f"x{k}", lb=0.0, ub=1.0) for k in range(3)]
    y = m.continuous("y", lb=0.0, ub=4.0)
    w = [m.continuous(f"w{k}", lb=0.0, ub=4.0) for k in range(2)]
    c = float(rng.uniform(-1.0, 1.0))
    a = rng.uniform(-1.0, 1.0, size=3)
    m.subject_to(x[0] + x[1] + x[2] == 1.0)
    for k in range(2):
        m.subject_to(w[k] == x[k] * y)
    r = float(rng.uniform(0.5, 2.0))
    m.subject_to(w[0] + w[1] >= r)
    m.minimize((c + a[0] * x[0] + a[1] * x[1] + a[2] * x[2]) * y)
    return m, c, a, r


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_distribution_tightens_and_never_crosses(monkeypatch, seed):
    """ON >= OFF, and ON <= the best feasible objective found by sampling."""
    m, c, a, r = _shared_product_model(seed)
    lb = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    ub = np.array([1, 1, 1, 4, 4, 4], dtype=float)

    monkeypatch.setenv("DISCOPT_DISTRIBUTE_AFFINE_PRODUCTS", "0")
    off = _lp_bound(build_uniform_relaxation(m, box=(lb, ub)))
    monkeypatch.setenv("DISCOPT_DISTRIBUTE_AFFINE_PRODUCTS", "1")
    on = _lp_bound(build_uniform_relaxation(m, box=(lb, ub)))

    # Upper bound on the optimum from feasible samples: x on the simplex (its
    # vertices included), y in [0, 4], and (x0 + x1) * y >= r.
    rng = np.random.default_rng(100 + seed)
    best = np.inf
    checked = 0
    for xs in [*np.eye(3), *(rng.dirichlet(np.ones(3)) for _ in range(2000))]:
        for yv in np.linspace(0.0, 4.0, 41):
            if (xs[0] + xs[1]) * yv >= r:
                checked += 1
                best = min(best, (c + a @ xs) * yv)
    assert checked > 0
    assert on >= off - 1e-7, f"distribution loosened the bound: {on} < {off}"
    assert on <= best + 1e-6, f"FALSE BOUND: {on} > feasible objective {best}"


def _eval_cnode(node, x: np.ndarray) -> float:
    """Value of a canonical node built from var / const / sum / prod / pow only."""
    if node.kind == "var":
        return float(x[node.payload])  # the flat variable index
    if node.kind == "const":
        return float(node.payload)
    if node.kind == "sum":
        coeffs, const = node.payload
        return const + sum(c * _eval_cnode(ch, x) for c, ch in zip(coeffs, node.children))
    if node.kind == "prod":
        (exps,) = node.payload
        return float(np.prod([_eval_cnode(ch, x) ** e for ch, e in zip(node.children, exps)]))
    if node.kind == "pow":
        return _eval_cnode(node.children[0], x) ** node.payload
    raise AssertionError(f"unexpected node kind {node.kind!r}")


def test_distribution_is_value_identical(monkeypatch):
    """The distributed canonical form evaluates to the original product everywhere."""
    from discopt._relax.canonical_expr import canonicalize

    m, c, a, _r = _shared_product_model(1)
    monkeypatch.setenv("DISCOPT_DISTRIBUTE_AFFINE_PRODUCTS", "1")
    dag = canonicalize(m)
    node = dag.objective
    assert node.kind == "sum", "the objective was not distributed"
    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(200):
        x = rng.uniform(-3.0, 3.0, size=6)
        want = (c + a @ x[:3]) * x[3]
        assert abs(_eval_cnode(node, x) - want) <= 1e-10 * (1.0 + abs(want))
        checked += 1
    assert checked == 200


def test_distributed_pairs_share_the_constraint_columns(monkeypatch):
    """``x_k * y`` from the distributed objective IS the column of ``w_k == x_k * y``."""
    m, _c, _a, _r = _shared_product_model(0)
    lb = np.zeros(6)
    ub = np.array([1, 1, 1, 4, 4, 4], dtype=float)
    monkeypatch.setenv("DISCOPT_DISTRIBUTE_AFFINE_PRODUCTS", "1")
    rel = build_uniform_relaxation(m, box=(lb, ub))
    pairs = sorted(rel.bilinear_map)
    # One column per pair x_k*y (k = 0, 1, 2): nothing duplicated, no affine-form product.
    assert pairs == [(0, 3), (1, 3), (2, 3)], pairs
    assert not any(len(A) > 1 or len(B) > 1 for _w, A, _ca, B, _cb in rel.bilinear_linform_specs)


def test_distribution_refuses_overlapping_supports(monkeypatch):
    """``(x + y) * (x + z)`` would create the square ``x*x``; it stays one product."""
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    z = m.continuous("z", lb=0.0, ub=1.0)
    m.minimize((x + y) * (x + z))
    monkeypatch.setenv("DISCOPT_DISTRIBUTE_AFFINE_PRODUCTS", "1")
    rel = build_uniform_relaxation(m, box=(np.zeros(3), np.ones(3)))
    assert not rel.bilinear_map
    assert any(len(A) == 2 and len(B) == 2 for _w, A, _ca, B, _cb in rel.bilinear_linform_specs)


def _three_by_three(shared: bool):
    m = dm.Model()
    x = [m.continuous(f"x{k}", lb=0.0, ub=1.0) for k in range(3)]
    y = [m.continuous(f"y{k}", lb=0.0, ub=1.0) for k in range(3)]
    if shared:
        m.subject_to(x[0] * y[0] <= 0.5)  # the pair x0*y0 exists without distribution
    m.minimize((x[0] + x[1] + x[2]) * (y[0] - y[1] + y[2]))
    return m


def test_distribution_respects_the_pair_cap(monkeypatch):
    monkeypatch.setenv("DISCOPT_DISTRIBUTE_AFFINE_PRODUCTS", "1")
    monkeypatch.setenv("DISCOPT_DISTRIBUTE_AFFINE_PRODUCTS_MAX", "8")  # 9 pairs needed
    rel = build_uniform_relaxation(_three_by_three(True), box=(np.zeros(6), np.ones(6)))
    assert sorted(rel.bilinear_map) == [(0, 3)]  # only the constraint's own product
    monkeypatch.setenv("DISCOPT_DISTRIBUTE_AFFINE_PRODUCTS_MAX", "9")
    rel = build_uniform_relaxation(_three_by_three(True), box=(np.zeros(6), np.ones(6)))
    assert len(rel.bilinear_map) == 9


def test_distribution_needs_a_pair_the_model_already_has(monkeypatch):
    """No shared pair, no distribution: the pairs would all be new columns.

    Measured on tanksize (#1944): 38 new pairs and none shared; the root moved
    0.8382 -> 0.8419 but the kernel tree doubled.
    """
    monkeypatch.setenv("DISCOPT_DISTRIBUTE_AFFINE_PRODUCTS", "1")
    rel = build_uniform_relaxation(_three_by_three(False), box=(np.zeros(6), np.ones(6)))
    assert not rel.bilinear_map
    assert any(len(A) == 3 and len(B) == 3 for _w, A, _ca, B, _cb in rel.bilinear_linform_specs)
    rel = build_uniform_relaxation(_three_by_three(True), box=(np.zeros(6), np.ones(6)))
    assert len(rel.bilinear_map) == 9


# --------------------------------------------------------------------------- #
# 2. lineq rows admitted by the native-kernel producer
# --------------------------------------------------------------------------- #
def _pinned_component_model():
    """A composition ``a + b + c == 1`` with ``c == 0`` and products with ``y``.

    ``(c == 0) * y`` linearizes to the one-column row ``X_cy == 0``: the nous1
    ``x36 == 0`` / ``x43 == 0`` shape.
    """
    m = dm.Model()
    a = m.continuous("a", lb=0.0, ub=1.0)
    b = m.continuous("b", lb=0.0, ub=1.0)
    c = m.continuous("c", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    m.subject_to(a + b + c == 1.0)
    m.subject_to(c == 0.0)
    m.subject_to(a * y + 2.0 * b * y + c * y >= 1.0)
    m.minimize(a * y - b * y + 3.0 * c * y + 0.5 * y)
    return m


def test_lineq_rows_reach_the_kernel(monkeypatch):
    monkeypatch.setenv("DISCOPT_RLT_LINEQ", "1")
    monkeypatch.setenv("DISCOPT_RLT_LINEQ_MAX", "0")
    bounds = (np.zeros(4), np.array([1.0, 1.0, 1.0, 2.0]))
    spec = build_spatial_kernel_spec(_pinned_component_model(), bounds=bounds)
    assert spec is not None, "the producer declined a model carrying lineq rows"

    # Prove the test exercises the path it names: without the row tags the same
    # model is declined (the pre-#1944 behaviour).
    import discopt._relax.uniform_relax as ur

    orig_build = ur.build_uniform_relaxation

    def untagged(*args, **kwargs):
        rel = orig_build(*args, **kwargs)
        rel.nonenvelope_rows = []
        return rel

    monkeypatch.setattr(ur, "build_uniform_relaxation", untagged)
    # The pre-#1944 behaviour is the #861 claim's decline; #2015's full-support claim
    # (default ON) leaves such rows to the fixed rows instead, so pin the claim.
    monkeypatch.setenv("DISCOPT_KERNEL_BLF_FULL_SUPPORT", "0")
    assert build_spatial_kernel_spec(_pinned_component_model(), bounds=bounds) is None


@pytest.mark.correctness
def test_lineq_kernel_solve_matches_the_flag_off_optimum(monkeypatch):
    bounds = (np.zeros(4), np.array([1.0, 1.0, 1.0, 2.0]))
    out = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("DISCOPT_RLT_LINEQ", flag)
        monkeypatch.setenv("DISCOPT_RLT_LINEQ_MAX", "0")
        res = solve_with_native_kernel(
            _pinned_component_model(), bounds=bounds, max_nodes=20000, gap_tol=1e-6
        )
        assert res is not None and res["status"] == "optimal", res
        out[flag] = res["incumbent"]
    assert abs(out["0"] - out["1"]) < 1e-5, out


# --------------------------------------------------------------------------- #
# 3. inequality x bound-factor rows at the root box
# --------------------------------------------------------------------------- #
def _pattern_model():
    """A small cutting-pattern model (the tln shape, generic data).

    ``n_k`` copies of pattern ``k``, ``p_kj`` pieces of width ``j`` in it; each
    pattern fits a roll (a linear knapsack row in the ``p``), and demand is
    bilinear ``sum_k n_k p_kj >= d_j``.
    """
    m = dm.Model()
    n = [m.integer(f"n{k}", lb=0, ub=4) for k in range(2)]
    p = [[m.integer(f"p{k}{j}", lb=0, ub=3) for j in range(2)] for k in range(2)]
    widths = (3.0, 4.0)
    for k in range(2):
        m.subject_to(widths[0] * p[k][0] + widths[1] * p[k][1] <= 10.0)
        m.subject_to(widths[0] * p[k][0] + widths[1] * p[k][1] >= 6.0)
    demand = (5.0, 4.0)
    for j in range(2):
        m.subject_to(n[0] * p[0][j] + n[1] * p[1][j] >= demand[j])
    m.minimize(n[0] + n[1])
    return m


_PAT_LB = np.zeros(6)
_PAT_UB = np.array([4, 4, 3, 3, 3, 3], dtype=float)


def _pattern_points(lb, ub):
    """Every integer point of the box satisfying the model's linear rows."""
    import itertools

    pts = []
    for v in itertools.product(*(range(int(lo), int(hi) + 1) for lo, hi in zip(lb, ub))):
        x = np.asarray(v, dtype=float)
        ok = True
        for k in range(2):
            s = 3.0 * x[2 + 2 * k] + 4.0 * x[3 + 2 * k]
            ok &= 6.0 <= s <= 10.0
        if ok:
            pts.append(x)
    return pts


def test_ineq_rows_cut_no_feasible_point_of_the_build_box():
    lb, ub = _PAT_LB, _PAT_UB
    rel = build_uniform_relaxation(_pattern_model(), box=(lb, ub), rlt_ineq=True)
    rows = rel.root_box_rows
    assert rows, "the pass emitted no rows"
    assert set(rows) <= set(rel.nonenvelope_rows)
    A = sp.csr_matrix(rel.model._A_ub)[rows]
    b = np.asarray(rel.model._b_ub, dtype=float).ravel()[rows]
    checked = 0
    for x in _pattern_points(lb, ub):
        assert np.max(A @ _lifted(rel, x) - b) <= 1e-9
        checked += 1
    assert checked > 100


def test_ineq_rows_tighten_and_never_cross():
    lb, ub = _PAT_LB, _PAT_UB
    off = _lp_bound(build_uniform_relaxation(_pattern_model(), box=(lb, ub)))
    on = _lp_bound(build_uniform_relaxation(_pattern_model(), box=(lb, ub), rlt_ineq=True))
    best = min(
        x[0] + x[1]
        for x in _pattern_points(lb, ub)
        if x[0] * x[2] + x[1] * x[4] >= 5.0 and x[0] * x[3] + x[1] * x[5] >= 4.0
    )
    assert on > off + 1e-6, f"no tightening: {on} vs {off}"
    assert on <= best + 1e-7, f"FALSE BOUND: {on} > optimum {best}"


def test_kernel_spec_carries_the_real_box_rows_not_the_probe_box_rows(monkeypatch):
    """Every fixed row of the spec holds at every lifted feasible point of the box.

    The producer reads structure off a probe box ``[1, 7+k]``, which does not
    contain this box. A bound-factor row taken from the probe build cuts points
    here, so this fails if the producer forwards the probe copies.
    """
    monkeypatch.setenv("DISCOPT_RLT_INEQ", "1")
    lb, ub = _PAT_LB, _PAT_UB
    model = _pattern_model()
    spec = build_spatial_kernel_spec(model, bounds=(lb, ub))
    assert spec is not None
    rel = build_uniform_relaxation(model, box=(lb, ub), rlt_ineq=True)
    prod = _product_cols(rel)
    known = set(range(rel.n_orig)) | set(prod)
    ptr, cols, vals, rhs = (
        spec["fixed_row_ptr"],
        spec["fixed_cols"],
        spec["fixed_coeffs"],
        spec["fixed_rhs"],
    )
    pts = [
        x
        for x in _pattern_points(lb, ub)
        if x[0] * x[2] + x[1] * x[4] >= 5.0 and x[0] * x[3] + x[1] * x[5] >= 4.0
    ]
    assert len(pts) > 20
    checked = 0
    for r in range(len(rhs)):
        cc = cols[ptr[r] : ptr[r + 1]]
        if not set(int(c) for c in cc) <= known:
            continue
        vv = vals[ptr[r] : ptr[r + 1]]
        for x in pts:
            z = _lifted(rel, x)
            assert float(vv @ z[cc]) <= rhs[r] + 1e-9, f"fixed row {r} cuts {x}"
            checked += 1
    assert checked > 1000


@pytest.mark.correctness
def test_ineq_kernel_solve_certifies_the_same_optimum(monkeypatch):
    lb, ub = _PAT_LB, _PAT_UB
    out = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("DISCOPT_RLT_INEQ", flag)
        res = solve_with_native_kernel(
            _pattern_model(), bounds=(lb, ub), max_nodes=50000, gap_tol=1e-6
        )
        assert res is not None and res["status"] == "optimal", res
        out[flag] = res
    assert abs(out["0"]["incumbent"] - out["1"]["incumbent"]) < 1e-6
    assert out["1"]["bound"] <= out["1"]["incumbent"] + 1e-6


@pytest.mark.correctness
@pytest.mark.parametrize("case", [1, 2, 3])
def test_default_relaxation_closes_haverly_at_the_root(monkeypatch, case):
    """Graduation regression (#1944): the default lineq rows close Haverly's root.

    Haverly's q-formulation with the composition rows ``sum_l q_l = 1`` times each
    flow is the pq-formulation, whose LP closes these instances. Measured before
    the graduation: 59 / 125 / 67 nodes. The optimum must be unchanged.
    """
    import os
    import sys

    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "..", "..", "discopt_benchmarks", "scripts")
    )
    from _pooling import pool18

    out = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("DISCOPT_RLT_LINEQ", flag)
        r = pool18.build_pool(pool18.haverly(case), "q")[0].solve(time_limit=60)
        assert r.status == "optimal" and r.gap_certified, (flag, r.status)
        assert r.algorithm_route.startswith("native-spatial"), r.algorithm_route
        out[flag] = r
    assert out["1"].node_count == 1, out["1"].node_count
    assert out["0"].node_count > 1, out["0"].node_count
    ref = out["0"].objective
    assert abs(out["1"].objective - ref) <= 1e-6 * (1.0 + abs(ref))
