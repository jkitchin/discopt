"""#2142: negative integer powers of an affine single-column base reach the native kernel.

``w = (c*x + k)**p`` with ``p < 0`` (the hull eps-perspective's ``1/yhat``) is relaxed
by ``_emit_1d``; the kernel regenerates the same 4 rows per box as an ``AffinePow``
term (kind 4, ``mccormick_patch::Univariate::Pow``; its row fixtures are Rust tests).
The producer compares its probe-box and real-box builds row for row, so both builds
hold a vacuous placeholder wherever a row is dropped (``keep_row_shape``), and a
placeholder on the PROBE box declines. ``DISCOPT_KERNEL_AFFINE_POW=0`` is the
debugging opt-out that restores the pre-#2142 producer.
"""

from __future__ import annotations

import collections

import discopt._relax.spatial_producer as sp
import discopt._relax.uniform_relax as ur
import discopt.modeling as dm
import numpy as np
import pytest
import scipy.optimize as sopt
import scipy.sparse as sps

# y = 1, x = (3 + sqrt 7)/2, z = 0.5/x: -x + 1.5 + 0.2 z (y = 0 gives -1.2326).
SMALL_OPT = -(3.0 + 7.0**0.5) / 2.0 + 1.5 + 0.2 * 0.5 / ((3.0 + 7.0**0.5) / 2.0)


def _small():
    """``x / (0.5 y + 0.5)``: the ratio's denominator is a reciprocal factor of an
    affine base in the binary ``y``, box ``t in [0.5, 1]``."""
    m = dm.Model("affpow_small")
    x = m.continuous("x", lb=0, ub=4)
    y = m.binary("y")
    z = m.continuous("z", lb=0, ub=4)
    m.subject_to(x / (0.5 * y + 0.5) + z <= 3)
    m.subject_to(z * x >= 0.5)
    m.minimize(-x + 1.5 * y + 0.2 * z)
    return m


def _spy_builds(monkeypatch):
    builds = []
    orig = ur.build_uniform_relaxation

    def spy(*a, **k):
        rel = orig(*a, **k)
        builds.append(rel)
        return rel

    monkeypatch.setattr(ur, "build_uniform_relaxation", spy)
    return builds


def test_keep_row_shape_holds_dropped_rows_and_leaves_the_lp_unchanged():
    m = dm.Model("big_product")
    x = m.continuous("x", lb=1, ub=1e10)
    y = m.continuous("y", lb=1, ub=1e10)
    m.subject_to(x * y >= 2)
    m.minimize(x + y)
    plain = ur.build_uniform_relaxation(m)
    shaped = ur.build_uniform_relaxation(m, keep_row_shape=True)
    assert plain.shape_placeholder_rows == []
    # (xH, yH) has cc = -1e20 >= _BIG: _emit_mccormick skips it.
    assert len(shaped.shape_placeholder_rows) >= 1
    A0 = sps.csr_matrix(plain.model._A_ub)
    A1 = sps.csr_matrix(shaped.model._A_ub)
    b0 = np.asarray(plain.model._b_ub).ravel()
    b1 = np.asarray(shaped.model._b_ub).ravel()
    assert A1.shape[0] == A0.shape[0] + len(shaped.shape_placeholder_rows)
    ph = shaped.shape_placeholder_rows
    assert all(A1.getrow(r).nnz == 0 and b1[r] == 0.0 for r in ph)
    keep = np.setdiff1d(np.arange(A1.shape[0]), ph)
    assert (A1[keep] != A0).nnz == 0
    np.testing.assert_array_equal(b1[keep], b0)


def test_small_model_gets_an_affine_pow_term(monkeypatch):
    builds = _spy_builds(monkeypatch)
    spec = sp.build_spatial_kernel_spec(_small())
    assert spec is not None
    kind4 = spec["term_kind"] == 4
    assert int(kind4.sum()) == 1
    assert spec["term_p"][kind4].tolist() == [-1]
    assert spec["term_coeff"][kind4].tolist() == [0.5]
    assert spec["term_cst"][kind4].tolist() == [0.5]
    assert len(builds) == 2 and all(not b.shape_placeholder_rows for b in builds)


def test_opt_out_restores_the_decline(monkeypatch):
    monkeypatch.setenv("DISCOPT_KERNEL_AFFINE_POW", "0")
    builds = _spy_builds(monkeypatch)
    assert sp.build_spatial_kernel_spec(_small()) is None
    assert builds and all(not b.shape_placeholder_rows for b in builds)


def test_kernel_root_lp_is_the_cold_build_lp():
    """Root LP of the kernel (propagation off) = the cold build's LP at the same box."""
    from discopt import _rust
    from discopt.solver_tuning import current

    m = _small()
    spec = sp.build_spatial_kernel_spec(m)
    assert spec is not None
    for k in [k for k in spec if k.startswith("meta_")]:
        spec.pop(k)
    res = _rust.solve_spatial_tree_py(
        **spec, max_nodes=1, run_propagation=False, run_obbt=False, time_limit_s=30.0
    )
    rel = ur.build_uniform_relaxation(
        m,
        rlt_lineq=bool(current().rlt_lineq),
        rlt_ineq=bool(current().rlt_ineq),
        skip_separable_floor=True,
        skip_convex_lift=True,
    )
    lp = rel.model
    bnds = np.asarray(lp._bounds, dtype=float)
    r = sopt.linprog(
        np.asarray(lp._c, dtype=float).ravel(),
        A_ub=sps.csr_matrix(lp._A_ub),
        b_ub=np.asarray(lp._b_ub, dtype=float).ravel(),
        bounds=list(zip(bnds[:, 0], bnds[:, 1])),
        method="highs",
    )
    assert r.status == 0
    assert res["root_bound"] == pytest.approx(r.fun, abs=1e-7)


def test_kernel_certifies_the_small_model():
    res = sp.solve_with_native_kernel(_small(), max_nodes=5000, time_limit_s=60.0)
    assert res is not None and res["status"] == "optimal"
    assert res["incumbent"] == pytest.approx(SMALL_OPT, abs=1e-6)
    assert res["bound"] <= SMALL_OPT + 1e-6
    assert res["bound"] == pytest.approx(SMALL_OPT, abs=1e-5)


def test_probe_box_placeholder_declines(monkeypatch):
    """A row the PROBE build drops cannot be claimed or compared: decline."""
    m = dm.Model("probe_drop")
    x = m.continuous("x", lb=0, ub=1)
    z = m.continuous("z", lb=0, ub=1)
    # On the probe box x in [1, 7], z in [1, 8]: x**40 ~ 1e33, so the product's
    # McCormick rows reach _BIG there and nowhere on the real box.
    m.subject_to(x**40 * z <= 0.5)
    m.minimize(-x - z)
    builds = _spy_builds(monkeypatch)
    assert sp.build_spatial_kernel_spec(m) is None
    assert len(builds) == 1 and builds[0].shape_placeholder_rows


def test_cstr_hull_spec_has_affine_pow_terms(monkeypatch):
    """GDPlib cstr under hull: 20 ``yc**-1`` atoms become kind-4 terms; the real-box
    build holds placeholders for the 44 McCormick rows of the ``yc**-3`` chains that
    reach ``_BIG``, the probe build none.

    Pinned to the eps-perspective lowering (``DISCOPT_HULL_EXACT_PERSPECTIVE=0``): the
    ``yc**-1`` atoms exist only there. The default homogenized hull is division-free and
    is pinned by ``test_1917_hull_plain_equality.test_cstr_hull_reaches_the_native_kernel``.
    """
    monkeypatch.setenv("DISCOPT_HULL_EXACT_PERSPECTIVE", "0")
    pytest.importorskip("pyomo")
    cstr = pytest.importorskip("gdplib.cstr")
    from discopt._relax.gdp_reformulate import reformulate_gdp
    from discopt.pyomo.gdp import convert_pyomo_gdp

    low = reformulate_gdp(convert_pyomo_gdp(cstr.build_model()).model, "hull")
    builds = _spy_builds(monkeypatch)
    spec = sp.build_spatial_kernel_spec(low)
    assert spec is not None, "the producer declined cstr under hull"
    kinds = collections.Counter(spec["term_kind"].tolist())
    assert kinds[4] == 20
    assert set(spec["term_p"][spec["term_kind"] == 4].tolist()) == {-1}
    probe, real = builds
    assert probe.shape_placeholder_rows == []
    assert len(real.shape_placeholder_rows) == 44
