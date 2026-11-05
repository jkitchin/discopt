"""#2168: the native spatial kernel's column-eliminating presolve.

``discopt._relax.spatial_presolve.presolve_spec`` rewrites a kernel spec into an
equivalent one with fewer free columns. These tests pin each reduction (issue items
a-d), check the expand/compress maps on planted feasible points and on the kernel's
own optima (item e), and, slow, the cstr entry (item f).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from discopt import _rust
from discopt._relax.spatial_presolve import presolve_spec

INF = 1e20


def make_spec(lo, hi, rows, *, c=None, integ=None, terms=(), blfs=(), n_orig=None):
    """A kernel spec from readable pieces.

    ``rows``: ``(dict col->coef, rhs)`` meaning ``<=``; ``("eq", dict, rhs)`` adds both
    directions. ``terms``: dicts with ``kind, i, out`` and optionally ``j, p, coeff,
    cst``. ``blfs``: ``((A, a0), (B, b0), w)``.
    """
    n = len(lo)
    flat = []
    for r in rows:
        if r[0] == "eq":
            flat.append((r[1], r[2]))
            flat.append(({k: -a for k, a in r[1].items()}, -r[2]))
        else:
            flat.append(r)
    ptr, cols, coefs, rhs = [0], [], [], []
    for row, b in flat:
        for k, a in row.items():
            cols.append(k)
            coefs.append(a)
        rhs.append(b)
        ptr.append(len(cols))
    spec = dict(
        n_cols=n,
        n_orig=n if n_orig is None else n_orig,
        c=np.asarray(c if c is not None else np.zeros(n), dtype=np.float64),
        integrality=np.asarray(integ if integ is not None else np.zeros(n), dtype=np.int64),
        global_lo=np.asarray(lo, dtype=np.float64),
        global_hi=np.asarray(hi, dtype=np.float64),
        fixed_row_ptr=np.asarray(ptr, dtype=np.int64),
        fixed_cols=np.asarray(cols, dtype=np.int64),
        fixed_coeffs=np.asarray(coefs, dtype=np.float64),
        fixed_rhs=np.asarray(rhs, dtype=np.float64),
        term_kind=np.asarray([t["kind"] for t in terms], dtype=np.int64),
        term_i=np.asarray([t["i"] for t in terms], dtype=np.int64),
        term_j=np.asarray([t.get("j", -1) for t in terms], dtype=np.int64),
        term_out=np.asarray([t["out"] for t in terms], dtype=np.int64),
        term_p=np.asarray([t.get("p", 0) for t in terms], dtype=np.int64),
        term_coeff=np.asarray([t.get("coeff", 0.0) for t in terms], dtype=np.float64),
        term_cst=np.asarray([t.get("cst", 0.0) for t in terms], dtype=np.float64),
        obbt_candidates=np.arange(n, dtype=np.int64),
    )
    for s, idx in (("a", 0), ("b", 1)):
        p_, c_, co_, cs_ = [0], [], [], []
        for blf in blfs:
            form, c0 = blf[idx]
            for k, a in form.items():
                c_.append(k)
                co_.append(a)
            cs_.append(c0)
            p_.append(len(c_))
        spec[f"blf_{s}_ptr"] = np.asarray(p_, dtype=np.int64)
        spec[f"blf_{s}_cols"] = np.asarray(c_, dtype=np.int64)
        spec[f"blf_{s}_coeffs"] = np.asarray(co_, dtype=np.float64)
        spec[f"blf_{s}_const"] = np.asarray(cs_, dtype=np.float64)
    spec["blf_w"] = np.asarray([b[2] for b in blfs], dtype=np.int64)
    return spec


def residual(spec, x):
    """Max violation of ``spec``'s rows, box, term and BLF definitions at ``x``."""
    x = np.asarray(x, dtype=np.float64)
    v = 0.0
    ptr = spec["fixed_row_ptr"]
    for r in range(len(spec["fixed_rhs"])):
        s = sum(
            spec["fixed_coeffs"][q] * x[spec["fixed_cols"][q]] for q in range(ptr[r], ptr[r + 1])
        )
        v = max(v, s - spec["fixed_rhs"][r])
    v = max(v, float(np.max(spec["global_lo"] - x[: spec["n_cols"]], initial=0.0)))
    v = max(v, float(np.max(x[: spec["n_cols"]] - spec["global_hi"], initial=0.0)))
    for t in range(len(spec["term_kind"])):
        k, i, o = int(spec["term_kind"][t]), int(spec["term_i"][t]), int(spec["term_out"][t])
        p, a, b = int(spec["term_p"][t]), spec["term_coeff"][t], spec["term_cst"][t]
        if k == 0:
            f = x[i] * x[int(spec["term_j"][t])]
        elif k == 1:
            f = x[i] ** p
        elif k == 2:
            f = (a * x[i] + b) ** 2
        elif k == 3:
            f = np.sqrt(a * x[i] + b)
        else:
            f = (a * x[i] + b) ** p
        v = max(v, abs(x[o] - f))
    for t in range(len(spec["blf_w"])):
        vals = []
        for s in ("a", "b"):
            p0, p1 = spec[f"blf_{s}_ptr"][t], spec[f"blf_{s}_ptr"][t + 1]
            vals.append(
                spec[f"blf_{s}_const"][t]
                + sum(
                    spec[f"blf_{s}_coeffs"][q] * x[spec[f"blf_{s}_cols"][q]] for q in range(p0, p1)
                )
            )
        v = max(v, abs(x[int(spec["blf_w"][t])] - vals[0] * vals[1]))
    return float(v)


def _check_maps(spec, x_star, tol=1e-9, **kw):
    """``x_star`` (feasible in ``spec``) compressed is feasible in the reduced spec, and
    expanding it gives ``x_star`` back. Returns the presolve output."""
    assert residual(spec, x_star) <= tol, "the planted point is not feasible"
    out = presolve_spec(spec, **kw)
    assert out is not None
    red, expand, info = out
    y = info["compress"](x_star)
    assert residual(red, y) <= 1e-7, residual(red, y)
    np.testing.assert_allclose(expand(y), x_star, atol=1e-9)
    return out


def _kernel(spec, **kw):
    return _rust.solve_spatial_tree_py(**spec, max_nodes=kw.pop("max_nodes", 20000), **kw)


def _same_optimum(spec, reduced, expand):
    """The kernel's optimum on ``spec`` equals the one on ``reduced``; the expanded
    reduced optimum is feasible in ``spec``."""
    r0, r1 = _kernel(spec), _kernel(reduced)
    assert r0["status"] == r1["status"] == "optimal", (r0["status"], r1["status"])
    assert r1["incumbent"] == pytest.approx(r0["incumbent"], abs=1e-5)
    x = expand(np.asarray(r1["incumbent_x"]))
    assert residual(spec, x) <= 1e-5
    assert float(spec["c"] @ x[: spec["n_cols"]]) == pytest.approx(r1["incumbent"], abs=1e-6)


# --- (a) a two-row identity aggregates away ------------------------------------------


def test_a_identity_pair_aggregates_and_expands():
    # x0 = x1 (two rows), x0 + x2 <= 3, min -x1 - x2 ; x2 in [0, 2]
    spec = make_spec(
        [0, 0, 0],
        [5, 5, 2],
        [("eq", {0: 1.0, 1: -1.0}, 0.0), ({0: 1.0, 2: 1.0}, 3.0)],
        c=[0.0, -1.0, -1.0],
    )
    red, expand, info = _check_maps(spec, np.array([1.0, 1.0, 2.0]))
    # x1 carries the objective, so x0 is the one eliminated
    assert info["eliminated"] == [0]
    assert red["global_lo"][0] == red["global_hi"][0] == 0.0
    assert 0 not in red["fixed_cols"].tolist()
    assert 0 not in red["obbt_candidates"].tolist()
    np.testing.assert_allclose(expand(np.array([0.0, 2.5, 0.5])), [2.5, 2.5, 0.5])
    _same_optimum(spec, red, expand)


# --- (b) negated binary; x_j = 2 x_k + 1 into a square and a BLF side --------------------


def test_b_negated_binary_and_affine_inputs():
    # cols: 0 y, 1 y' (binaries, y + y' = 1), 2 x, 3 z (x = 2 z + 1), 4 s = x^2,
    # 5 w = (x + y') * z
    lo = [0, 0, -INF, -2, 0, -INF]
    hi = [1, 1, INF, 2, 30, INF]
    spec = make_spec(
        lo,
        hi,
        [
            ("eq", {0: 1.0, 1: 1.0}, 1.0),
            ("eq", {2: 1.0, 3: -2.0}, 1.0),
            ({5: -1.0, 4: 0.1}, 2.0),
            ({3: 1.0, 1: 1.0}, 2.0),
        ],
        c=[0.3, 0.0, 0.0, 0.1, 1.0, -1.0],  # z carries the objective, so x goes
        integ=[1, 1, 0, 0, 0, 0],
        terms=[dict(kind=1, i=2, out=4, p=2)],
        blfs=[(({2: 1.0, 1: 1.0}, 0.0), ({3: 1.0}, 0.0), 5)],
        n_orig=4,
    )
    z = 0.5
    x_star = np.array([1.0, 0.0, 2 * z + 1, z, (2 * z + 1) ** 2, (2 * z + 1) * z])
    red, expand, info = _check_maps(spec, x_star)
    assert set(info["eliminated"]) == {1, 2}
    # the square of an affine input is an affine square of the representative
    (t,) = range(len(red["term_kind"]))
    assert int(red["term_kind"][t]) == 2 and int(red["term_i"][t]) == 3
    assert red["term_coeff"][t] == 2.0 and red["term_cst"][t] == 1.0
    # y' = 1 - y and x = 2 z + 1 inside the BLF side: 2 z + 1 + 1 - y = 2 z - y + 2
    a_cols = red["blf_a_cols"][red["blf_a_ptr"][0] : red["blf_a_ptr"][1]].tolist()
    a_coef = red["blf_a_coeffs"][red["blf_a_ptr"][0] : red["blf_a_ptr"][1]].tolist()
    assert dict(zip(a_cols, a_coef)) == {0: -1.0, 3: 2.0}
    assert red["blf_a_const"][0] == 2.0
    assert red["integrality"][0] == 1
    _same_optimum(spec, red, expand)


# --- (c) output / output pairs -----------------------------------------------------------


@pytest.mark.parametrize("rhs", [0.0, 1.0])
def test_c_output_pair_rescales_only_without_offset(rhs):
    # w1 = x0 x1, w2 = x2 x3, w1 - 2 w2 = rhs
    lo, hi = [0, 0, 0, 0, -INF, -INF], [2, 2, 2, 2, INF, INF]
    spec = make_spec(
        lo,
        hi,
        [("eq", {4: 1.0, 5: -2.0}, rhs), ({0: 1.0, 2: 1.0}, 3.0)],
        c=[0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
        terms=[dict(kind=0, i=0, j=1, out=4), dict(kind=0, i=2, j=3, out=5)],
        n_orig=4,
    )
    x2, x3 = 1.0, 1.5
    w2 = x2 * x3
    w1 = 2 * w2 + rhs
    x0 = 2.0
    x_star = np.array([x0, w1 / x0, x2, x3, w1, w2])
    out = presolve_spec(spec)
    if rhs != 0.0:
        # w1 = 2 w2 + 1: neither output can be eliminated, nothing else reduces
        assert out is None
        return
    red, expand, info = _check_maps(spec, x_star)
    assert info["eliminated"] == [4]  # w2 carries the objective and survives
    assert len(red["term_kind"]) == 1 and len(red["blf_w"]) == 1
    assert int(red["blf_w"][0]) == 5  # x0 x1 / 2 is now a product onto w2
    assert red["blf_a_coeffs"][0] == 0.5
    _same_optimum(spec, red, expand)


# --- (d) a product with a fixed factor is linearised --------------------------------------


def test_d_fixed_factor_linearises_product():
    # x0 fixed at 3 by its box, w = x0 * x1 (kind 0) and w = (x0 - 1) * x2 (BLF), w <= 4
    lo, hi = [3, -5, -5, -INF, -INF], [3, 5, 5, INF, INF]
    spec = make_spec(
        lo,
        hi,
        [({3: 1.0}, 4.0), ({4: 1.0, 1: 1.0}, 1.0)],
        c=[0.0, -1.0, 0.0, 0.0, -1.0],
        terms=[dict(kind=0, i=0, j=1, out=3)],
        blfs=[(({0: 1.0}, -1.0), ({2: 1.0}, 0.0), 4)],
        n_orig=3,
    )
    x_star = np.array([3.0, 1.0, -0.5, 3.0, -1.0])
    red, expand, info = _check_maps(spec, x_star)
    assert len(red["term_kind"]) == 0 and len(red["blf_w"]) == 0
    assert info["stats"]["product_linearised"] == 2
    assert info["stats"]["aggregated"] == 2  # w = 3 x1 and w' = 2 x2 eliminate w, w'
    _same_optimum(spec, red, expand)


def test_fixed_term_input_fixes_output():
    # x0 fixed at 2, s = x0^3 (kind 1), r = sqrt(2 x0 + 5) (kind 3); min x1 + s + r
    spec = make_spec(
        [2, 0, -INF, -INF],
        [2, 1, INF, INF],
        [({1: -1.0}, 0.0)],
        c=[0.0, 1.0, 1.0, 1.0],
        terms=[dict(kind=1, i=0, out=2, p=3), dict(kind=3, i=0, out=3, coeff=2.0, cst=5.0)],
        n_orig=2,
    )
    red, expand, info = _check_maps(spec, np.array([2.0, 0.0, 8.0, 3.0]))
    assert len(red["term_kind"]) == 0
    assert red["global_lo"][2] == red["global_hi"][2] == 8.0
    assert red["global_lo"][3] == red["global_hi"][3] == 3.0


def test_integer_aggregation_needs_unit_integral_map():
    # y integer, x continuous, y = 2 x: y may not be eliminated; x may (then the
    # representative y keeps its integrality)
    spec = make_spec(
        [0, 0],
        [10, 10],
        [("eq", {0: 1.0, 1: -2.0}, 0.0)],
        c=[1.0, 0.0],
        integ=[1, 0],
    )
    red, expand, info = _check_maps(spec, np.array([4.0, 2.0]))
    assert info["eliminated"] == [1]
    assert red["integrality"][0] == 1 and red["global_hi"][0] == 10.0
    # y integer, x continuous, y = 2 x + 0.5: y cannot go (non-integral offset) and x
    # cannot go either (it carries the objective with an offset)
    spec2 = make_spec(
        [0, 0], [10, 10], [("eq", {0: 1.0, 1: -2.0}, 0.5)], c=[0.0, 1.0], integ=[1, 0]
    )
    assert presolve_spec(spec2) is None


def test_inconsistent_spec_declines():
    # x0 = x1 + 5 with both boxes [0, 1]: aggregation empties the box
    spec = make_spec([0, 0], [1, 1], [("eq", {0: 1.0, 1: -1.0}, 5.0)])
    assert presolve_spec(spec) is None


def test_eliminated_columns_have_no_occurrence():
    spec = make_spec(
        [0, 0, 0, -INF],
        [4, 4, 4, INF],
        [("eq", {0: 1.0, 1: -1.0}, 0.0), ("eq", {1: 1.0, 2: -1.0}, 0.0)],
        c=[0.0, 0.0, 0.0, 1.0],
        terms=[dict(kind=0, i=0, j=2, out=3)],
        n_orig=3,
    )
    red, expand, info = _check_maps(spec, np.array([2.0, 2.0, 2.0, 4.0]))
    dead = set(info["eliminated"])
    assert len(dead) == 2
    used = set(red["fixed_cols"].tolist()) | set(red["term_i"].tolist())
    used |= set(red["term_out"].tolist()) | set(red["blf_w"].tolist())
    used |= set(red["blf_a_cols"].tolist()) | set(red["blf_b_cols"].tolist())
    assert not dead & used
    # x0 x2 with x0 = x2 = x1 is the square of the survivor
    assert len(red["blf_w"]) == 1
    _same_optimum(spec, red, expand)


# --- #2171 review: cascading fixings, exact pairs, the sqrt domain, hook points ---------


def test_cascading_fix_through_a_term_chain():
    """``x * x**2`` (kind 1 then kind 0 on its output) with ``x`` fixed: fixing ``x``
    fixes ``w1 = x**2``, whose substitution pops the product and linearises it while
    ``x`` is still being substituted. Recursing from inside that loop raised
    ``KeyError`` (#2171 review); the fixing is now queued."""
    from discopt._relax.spatial_presolve import _Problem

    spec = make_spec(
        [-3, 0, -INF],
        [3, 9, INF],
        [({2: 1.0}, 50.0)],
        c=[0.0, 0.0, 1.0],
        terms=[dict(kind=1, i=0, out=1, p=2), dict(kind=0, i=0, j=1, out=2)],
        n_orig=1,
    )
    prob = _Problem(spec)
    prob.fix(0, 2.0)
    assert prob.lo[1] == prob.hi[1] == 4.0
    assert not prob.terms
    assert not prob.col_rows.get(0) and not prob.col_rows.get(1)
    # through presolve_spec, with the box fixing x
    spec["global_lo"][0] = spec["global_hi"][0] = 2.0
    red, expand, info = _check_maps(spec, np.array([2.0, 4.0, 8.0]))
    assert red["global_lo"][2] == red["global_hi"][2] == 8.0
    _same_optimum(spec, red, expand)


def test_cascading_fix_through_a_blf_chain():
    """The BLF variant: ``w1 = x * x`` and ``w2 = (x + 1) * w1`` as BLFs, ``x`` fixed:
    both sides of the second become constant only after the first fixes ``w1``."""
    from discopt._relax.spatial_presolve import _Problem

    spec = make_spec(
        [-3, 0, -INF],
        [3, 9, INF],
        [({2: 1.0}, 50.0)],
        c=[0.0, 0.0, 1.0],
        blfs=[
            (({0: 1.0}, 0.0), ({0: 1.0}, 0.0), 1),
            (({0: 1.0}, 1.0), ({1: 1.0}, 0.0), 2),
        ],
        n_orig=1,
    )
    prob = _Problem(spec)
    prob.fix(0, 2.0)
    assert prob.lo[1] == prob.hi[1] == 4.0
    assert not prob.blfs
    spec["global_lo"][0] = spec["global_hi"][0] = 2.0
    red, expand, info = _check_maps(spec, np.array([2.0, 4.0, 12.0]))
    assert red["global_lo"][2] == red["global_hi"][2] == 12.0
    _same_optimum(spec, red, expand)


def test_cascade_to_an_inconsistent_value_declines():
    # the same chain with w2 <= 5: the cascade fixes w2 = 8 and must decline, not raise
    spec = make_spec(
        [2, 0, -INF],
        [2, 9, INF],
        [({2: 1.0}, 5.0)],
        c=[0.0, 0.0, 1.0],
        terms=[dict(kind=1, i=0, out=1, p=2), dict(kind=0, i=0, j=1, out=2)],
        n_orig=1,
    )
    assert presolve_spec(spec) is None


def test_near_parallel_rows_are_not_an_equality():
    # x0 - x1 <= 0 and -x0 + (1 + 1e-13) x1 <= 0: equal to 12 significant digits, but
    # not exact negatives, so no aggregation (the 1e-12 rounding used to pair them)
    spec = make_spec(
        [0, 0],
        [4, 4],
        [({0: 1.0, 1: -1.0}, 0.0), ({0: -1.0, 1: 1.0 + 1e-13}, 0.0)],
        c=[1.0, 0.0],
    )
    assert presolve_spec(spec) is None


def test_sqrt_argument_kept_in_its_domain():
    """``s = sqrt(x - 0.1)`` with ``x = 3 z - 0.2`` aggregated (``x`` eliminated) and
    ``x`` in ``[0.1, 2]``: ``z``'s lower end ``(0.1 + 0.2) / 3`` rounds outward, and
    the composed argument ``3 z + 0.2 - 0.1 ... `` may dip below 0 by rounding. Every
    sqrt argument on the reduced box must be >= 0."""
    for x_lo in (0.1, 0.3, 1.0 / 3.0, 0.7):
        spec = make_spec(
            [x_lo, -5, -INF],
            [2.0, 5, INF],
            [("eq", {0: 1.0, 1: -3.0}, -0.2)],
            c=[0.0, 1.0, 1.0],
            terms=[dict(kind=3, i=0, out=2, coeff=1.0, cst=-x_lo)],
            n_orig=2,
        )
        red, expand, info = presolve_spec(spec)
        assert info["eliminated"] == [0]
        (t,) = range(len(red["term_kind"]))
        assert int(red["term_kind"][t]) == 3 and int(red["term_i"][t]) == 1
        a, c = red["term_coeff"][t], red["term_cst"][t]
        lo, hi = red["global_lo"][1], red["global_hi"][1]
        assert min(a * lo + c, a * hi + c) >= 0.0, (x_lo, a * lo + c)


def test_hook_points_are_not_expanded():
    """``_kernel_presolve_main``'s result map: a kernel LP point is expanded; the point
    the primal hook returned (a model point, with the value the kernel adopted it
    at) is passed through unchanged."""
    import discopt.solver as S

    spec = make_spec(
        [0, 0, 0],
        [5, 5, 2],
        [("eq", {0: 1.0, 1: -1.0}, 0.0), ({0: 1.0, 2: 1.0}, 3.0)],
        c=[0.0, -1.0, -1.0],
    )
    model_point = [1.25, 1.25, 1.75]

    def hook(x, incumbent):
        return (-3.0, model_point)

    kw = {"incumbent_value": lambda x, lp: float(lp), "primal_hook": hook}
    reduced, expand_result = S._kernel_presolve_main(spec, kw)
    assert expand_result is not None and reduced is not spec
    lp_point = np.array([0.0, 2.5, 0.5, 9.0])  # one envelope column past n_cols
    assert kw["primal_hook"](lp_point, None) == (-3.0, model_point)
    # an LP incumbent: x0 restored from its representative, the tail untouched
    np.testing.assert_allclose(
        expand_result({"incumbent": -3.0 + 1.0, "incumbent_x": lp_point}), [2.5, 2.5, 0.5, 9.0]
    )
    # the hook's incumbent: returned as it is
    assert expand_result({"incumbent": -3.0, "incumbent_x": model_point}) is model_point


# --- (e) differential: planted points and kernel optima -----------------------------------


def _planted(rng):
    """A random spec with doubletons, products and squares through a planted point."""
    n_x = 6
    x = rng.uniform(-2.0, 2.0, n_x)
    cols = list(x)
    lo = list(x - rng.uniform(0.5, 2.0, n_x))
    hi = list(x + rng.uniform(0.5, 2.0, n_x))
    rows, terms, blfs = [], [], []
    integ = [0] * n_x
    # doubletons x_new = a x_k + b
    for _ in range(4):
        k = int(rng.integers(len(cols)))
        a = float(rng.choice([-2.0, -1.0, 0.5, 1.0, 3.0]))
        b = float(rng.choice([0.0, 0.0, 1.0, -0.5]))
        v = a * cols[k] + b
        cols.append(v)
        lo.append(v - 10.0)
        hi.append(v + 10.0)
        integ.append(0)
        rows.append(("eq", {len(cols) - 1: 1.0, k: -a}, b))
    # products of random columns, squares, an affine-form product
    for _ in range(3):
        i, j = (int(v) for v in rng.integers(len(cols), size=2))
        cols.append(cols[i] * cols[j])
        lo.append(-INF)
        hi.append(INF)
        integ.append(0)
        terms.append(dict(kind=0, i=i, j=j, out=len(cols) - 1))
    for _ in range(2):
        i = int(rng.integers(len(cols)))
        cols.append(cols[i] ** 2)
        lo.append(0.0)
        hi.append(INF)
        integ.append(0)
        terms.append(dict(kind=1, i=i, out=len(cols) - 1, p=2))
    i, j, k = (int(v) for v in rng.choice(n_x, size=3, replace=False))
    cols.append((cols[i] + 2 * cols[j] - 1) * (cols[k] + 0.5))
    lo.append(-INF)
    hi.append(INF)
    integ.append(0)
    blfs.append((({i: 1.0, j: 2.0}, -1.0), ({k: 1.0}, 0.5), len(cols) - 1))
    # an output/output identity and a few slack inequalities through the point
    a, b = len(cols) - 1, len(cols) - 3
    cols.append(cols[a])
    lo.append(-INF)
    hi.append(INF)
    integ.append(0)
    rows.append(("eq", {len(cols) - 1: 1.0, a: -1.0}, 0.0))
    # a support-3 equality defining a fresh column (stage 4's input)
    i, j = (int(v) for v in rng.choice(n_x, size=2, replace=False))
    cols.append(cols[i] - 2.0 * cols[j] + 0.25)
    lo.append(-INF)
    hi.append(INF)
    integ.append(0)
    rows.append(("eq", {len(cols) - 1: 1.0, i: -1.0, j: 2.0}, 0.25))
    xs = np.array(cols)
    for _ in range(3):
        sup = rng.choice(len(cols), size=3, replace=False)
        coef = {int(s): float(rng.normal()) for s in sup}
        rows.append((coef, float(sum(c * xs[s] for s, c in coef.items())) + 0.1))
    c = np.zeros(len(cols))
    c[rng.choice(n_x, size=2, replace=False)] = rng.normal(size=2)
    c[b] = 0.3
    spec = make_spec(lo, hi, rows, c=c, integ=integ, terms=terms, blfs=blfs, n_orig=n_x)
    return spec, xs


@pytest.mark.parametrize("seed", range(12))
def test_e_planted_point_round_trips(seed):
    spec, x_star = _planted(np.random.default_rng(seed))
    red, expand, info = _check_maps(
        spec,
        x_star,
        tol=1e-7,
    )
    assert info["n_eliminated"] >= 4
    # random points of the reduced box: expand maps reduced row values onto original ones
    rng = np.random.default_rng(1000 + seed)
    lo = np.maximum(red["global_lo"], -50.0)
    hi = np.minimum(red["global_hi"], 50.0)
    for _ in range(5):
        y = rng.uniform(lo, hi)
        y[info["eliminated"]] = 0.0
        x = expand(y)
        # the eliminated columns obey their defining equalities exactly
        y2 = info["compress"](x)
        np.testing.assert_allclose(expand(y2), x, atol=1e-12)
        # every original doubleton equality holds at the expanded point
        assert float(spec["c"] @ x) == pytest.approx(float(red["c"] @ y), abs=1e-9)


@pytest.mark.parametrize("seed", range(6))
def test_e_kernel_optimum_is_preserved(seed):
    spec, _ = _planted(np.random.default_rng(seed))
    # bound every column so the kernel can certify
    spec["global_lo"] = np.maximum(spec["global_lo"], -200.0)
    spec["global_hi"] = np.minimum(spec["global_hi"], 200.0)
    red, expand, _info = presolve_spec(
        spec,
    )
    # 2,000 nodes: every seed but 4 certifies in <= 237; seed 4 is uncertified in both
    # arms even at 50,000 (31 s, over the fast-lane budget), and the cross-bound checks
    # below need no certificate
    r0, r1 = _kernel(spec, max_nodes=2000), _kernel(red, max_nodes=2000)
    assert r0["incumbent"] is not None and r1["incumbent"] is not None
    # each arm's incumbent respects the other arm's rigorous bound (gap tolerance
    # 1e-4 relative), whichever of them certified
    tol = 2e-4 * (1.0 + abs(r0["incumbent"]))
    assert r1["incumbent"] >= r0["bound"] - tol
    assert r0["incumbent"] >= r1["bound"] - tol
    if r0["status"] == r1["status"] == "optimal":
        assert r1["incumbent"] == pytest.approx(r0["incumbent"], abs=tol)
    x = expand(np.asarray(r1["incumbent_x"]))
    assert residual(spec, x) <= 1e-5
    assert float(spec["c"] @ x[: spec["n_cols"]]) == pytest.approx(r1["incumbent"], abs=1e-6)


def test_e_cstr_homog_spec_reduces():
    """The captured #2142 cstr homog spec: SCIP fixes 51 columns; this presolve finds
    the same 51 fixings and aggregates >= 100 columns, without loosening the root."""
    path = (
        Path(__file__).resolve().parents[2]
        / "docs/dev/data/2142-exact-hull/scip/cstr_homog_spec.npz"
    )
    d = dict(np.load(path))
    spec = {k: (int(v) if v.ndim == 0 else v) for k, v in d.items() if not k.startswith("meta_")}
    red, expand, info = presolve_spec(spec)
    assert info["n_fixed"] == 51
    assert info["n_eliminated"] >= 100
    free = int(np.sum(red["global_hi"] > red["global_lo"]))
    assert free <= 125  # 234 free columns before; SCIP's presolved model has 110
    r0 = _rust.solve_spatial_tree_py(**spec, max_nodes=1)
    r1 = _rust.solve_spatial_tree_py(**red, max_nodes=1)
    assert r1["bound"] >= r0["bound"] - 1e-9


# --- solver integration ---------------------------------------------------------------------


def test_solver_flag_routes_main_call_through_presolve(monkeypatch):
    """With ``DISCOPT_KERNEL_PRESOLVE=1`` a kernel-routed solve with a doubleton
    reaches the presolve, and returns the OFF arm's certified optimum."""
    import discopt.modeling as dm
    import discopt.solver as S

    def build():
        m = dm.Model("presolve_route")
        x = m.continuous("x", lb=-3, ub=3)
        y = m.continuous("y", lb=-3, ub=3)
        z = m.continuous("z", lb=-3, ub=3)
        m.subject_to(z == 2 * y + 1)
        m.subject_to(x * y + z * z >= 1.5)
        m.subject_to(x + y <= 2)
        m.minimize(x * z - y)
        return m

    monkeypatch.setenv("DISCOPT_KERNEL_PRESOLVE", "0")
    off = build().solve(time_limit=60)
    monkeypatch.setenv("DISCOPT_KERNEL_PRESOLVE", "1")
    S.KERNEL_PRESOLVE_STATS.clear()
    on = build().solve(time_limit=60)
    assert S.KERNEL_PRESOLVE_STATS["calls"] >= 1, "the kernel was never reached"
    assert S.KERNEL_PRESOLVE_STATS["applied"] >= 1
    assert off.status == on.status == "optimal"
    assert on.objective == pytest.approx(off.objective, abs=1e-5)


# --- (f) slow: the cstr entry --------------------------------------------------------------

#: The entry run takes ~300 s (20,000 kernel nodes), five times the slow lane's 60 s
#: budget, and no smaller budget separates the arms: at 2,000 nodes the original and
#: presolved specs reach 0.3421 and 0.3412 (``docs/dev/data/2168-kernel-presolve/``).
#: So it is skipped at collection; ``DISCOPT_RUN_2168_ENTRY=1`` re-runs it.
_ENTRY = os.environ.get("DISCOPT_RUN_2168_ENTRY") == "1"


@pytest.mark.slow
@pytest.mark.skipif(not _ENTRY, reason="~300 s; set DISCOPT_RUN_2168_ENTRY=1 (measured 2.7936)")
def test_f_cstr_homog_presolved_beats_bigm_line_at_20k():
    """The captured cstr homog-hull spec, presolved, seeded with the verified incumbent
    value, 20,000 deterministic kernel nodes: the bound passes the big-M default's 20k
    line 2.7875 (the pre-registered kill line; measured 2.7936, original spec 1.9044)."""
    script = (
        Path(__file__).resolve().parents[2] / "docs/dev/data/2168-kernel-presolve/kernel_arm.py"
    )
    env = dict(os.environ, PYTHONHASHSEED="0")
    out = subprocess.run(
        [sys.executable, str(script), "pre", "20000", "seeded"],
        capture_output=True,
        text=True,
        env=env,
        timeout=3000,
    )
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    line = next(ln for ln in out.stdout.splitlines() if ln.startswith("RESULT"))
    bound = float(line.split("bound=")[1].split()[0])
    status = line.split("status=")[1].split()[0]
    assert status == "optimal" or bound > 2.7875, line
