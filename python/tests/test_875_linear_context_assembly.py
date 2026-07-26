"""#875 residual: ``build_linear_context`` assembled its matrix in O(rows x vars).

After PR #878 and the follow-up call-site fix took ``watercontamination0202`` from
579 s to 47 s against a 30 s ``time_limit``, the owner's profile put
``_classify_model_convexity`` -> ``build_linear_context`` at **15.8 s over 2 calls**,
second only to the relaxation build among what remains.

The obvious guess was wrong, and the measurement is why this file exists. The guess
was that the cost is ``extract_affine``'s per-row ``np.zeros(n_vars)`` -- the same
shape #878 fixed in ``_linearize_affine_expr``. Profiling the function at
n_vars=128,000 with 300 rows says otherwise:

    np.vstack                      3.015 s   99.4%
    all 302 np.zeros calls         0.002 s
    the affine walk                0.008 s

``calloc`` hands back lazily-mapped zero pages, so allocating a dense row per
constraint is nearly free; ``np.vstack`` then COPIES m x n float64 (307 MB here) and
faults in every one of those pages. Allocating the ``(m, n)`` array directly costs
0.0001 s for exactly the same reason.

So the fix is to scatter the sparse rows into one pre-allocated array -- O(nnz)
assembly instead of O(m * n) copying -- rather than to avoid the per-row allocation.
Measured on the probe below: 0.281 s -> 0.0041 s at n_vars=128,000, and flat in
n_vars (3.28x per 4x vars -> 1.14x).

The output contract is unchanged: dense ``A_ub`` / ``A_eq``, because the only
consumer (``LinearContext.affine_range``) hands them to the POUNCE LP. Verified
bit-identical (shapes, nnz, sums, sums of squares) across all 66 in-repo MINLPLib
instances.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import time  # noqa: E402

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._jax.convexity.linear_context import (  # noqa: E402
    build_linear_context,
    extract_affine,
    extract_affine_sparse,
)


def _sparse_row_model(n_vars: int, n_rows: int):
    """Rows with two leaves each, over ``n_vars`` variables — maximally sparse."""
    m = dm.Model(f"lc{n_vars}")
    x = m.continuous("x", shape=(n_vars,), lb=-10.0, ub=10.0)
    for k in range(n_rows):
        m.subject_to(x[k % n_vars] + 2.0 * x[(k + 1) % n_vars] <= 5.0)
    m.minimize(x[0])
    return m


# --------------------------------------------------------------------------
# the sparse core and its dense view agree
# --------------------------------------------------------------------------


def test_sparse_and_dense_extraction_agree():
    m = dm.Model("agree")
    x = m.continuous("x", shape=(5,), lb=-1.0, ub=1.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    m.minimize(x[0])
    m.subject_to(2.0 * x[0] + 3.0 * x[3] - 4.0 * y <= 1.0)
    m.subject_to(x[1] - x[1] + x[4] / 2.0 == 0.0)  # cancelling terms -> explicit zero
    m.subject_to(-(x[2] + 1.0) >= -3.0)

    n = 6
    for c in m._constraints:
        dense = extract_affine(c.body, m, n)
        sparse = extract_affine_sparse(c.body, m, n)
        assert (dense is None) == (sparse is None)
        if dense is None:
            continue
        coeffs, const = dense
        terms, sconst = sparse
        assert sconst == const
        rebuilt = np.zeros(n, dtype=np.float64)
        for i, v in terms.items():
            rebuilt[i] = v
        assert np.array_equal(rebuilt, coeffs)


def test_both_forms_refuse_a_nonlinear_body():
    m = dm.Model("refuse")
    x = m.continuous("x", shape=(2,), lb=0.1, ub=2.0)
    m.minimize(x[0])
    m.subject_to(x[0] * x[1] <= 1.0)
    body = m._constraints[0].body
    assert extract_affine(body, m, 2) is None
    assert extract_affine_sparse(body, m, 2) is None


# --------------------------------------------------------------------------
# the assembly
# --------------------------------------------------------------------------


def test_assembly_is_flat_in_the_variable_count():
    """The cost test, and the one that FAILS before the fix.

    Same rows, more variables: assembly must not get slower. Before, at 300 rows,
    0.0254 s / 0.0856 s / 0.2810 s for n_vars 8k / 32k / 128k (3.37x, 3.28x per 4x
    vars). After: 0.0029 / 0.0036 / 0.0041.

    Absolute threshold primary, ratio secondary and gated on the baseline clearing
    the timer floor — a ratio of two ~4 ms timings is noise over noise on a loaded
    runner (the #875 lesson). ``min`` over repetitions for the same reason.
    """
    n_rows = 300
    walls = {}
    for n_vars in (8_000, 128_000):
        m = _sparse_row_model(n_vars, n_rows)
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            ctx = build_linear_context(m)
            best = min(best, time.perf_counter() - t0)
        assert ctx is not None
        assert ctx.A_ub.shape == (n_rows, n_vars)
        walls[n_vars] = best

    assert walls[128_000] < 0.10, (
        f"assembly still copies O(rows x vars): {walls[128_000]:.3f}s for {n_rows} "
        f"two-leaf rows at n_vars=128,000 (vstack measured 0.281s, scatter 0.004s)"
    )
    if walls[8_000] > 5e-3:
        ratio = walls[128_000] / walls[8_000]
        assert ratio < 4.0, f"cost still scales with n_vars: {ratio:.1f}x — {walls}"


def test_assembled_matrices_are_exactly_right():
    """Scattering must place every coefficient where the dense build did, and leave
    every other entry at 0.0 — including the sign flip a ``>=`` row gets."""
    m = dm.Model("shape")
    x = m.continuous("x", shape=(4,), lb=-5.0, ub=5.0)
    m.minimize(x[0])
    m.subject_to(x[0] + 2.0 * x[1] <= 3.0)  # ub row 0
    m.subject_to(x[2] - x[3] >= 1.0)  # ub row 1, negated
    m.subject_to(3.0 * x[1] + x[3] == 2.0)  # eq row 0
    m.subject_to(dm.log(x[0] + 10.0) <= 4.0)  # nonlinear -> dropped

    ctx = build_linear_context(m)
    assert ctx is not None
    assert ctx.A_ub.shape == (2, 4)
    assert ctx.A_eq.shape == (1, 4)
    assert np.array_equal(ctx.A_ub[0], [1.0, 2.0, 0.0, 0.0])
    assert ctx.b_ub[0] == pytest.approx(3.0)
    assert np.array_equal(ctx.A_ub[1], [0.0, 0.0, -1.0, 1.0])
    assert ctx.b_ub[1] == pytest.approx(-1.0)
    assert np.array_equal(ctx.A_eq[0], [0.0, 3.0, 0.0, 1.0])
    assert ctx.b_eq[0] == pytest.approx(2.0)


def test_a_model_with_no_linear_rows_keeps_the_empty_shapes():
    """The empty arms fed ``affine_range``'s ``.size == 0`` short-circuit; their
    shapes and dtypes must not drift."""
    m = dm.Model("nolinear")
    x = m.continuous("x", shape=(3,), lb=0.1, ub=2.0)
    m.minimize(x[0])
    m.subject_to(x[0] * x[1] <= 1.0)

    ctx = build_linear_context(m)
    assert ctx is not None
    assert ctx.A_ub.shape == (0, 3) and ctx.A_eq.shape == (0, 3)
    assert ctx.b_ub.shape == (0,) and ctx.b_eq.shape == (0,)
    assert ctx.A_ub.dtype == np.float64 and ctx.b_ub.dtype == np.float64
    assert ctx.A_ub.size == 0 and ctx.A_eq.size == 0


def test_affine_range_still_uses_the_rows():
    """End of the chain: the assembled matrix must still tighten a range beyond the
    box, which is the whole reason LinearContext exists. ``x1 <= x0`` makes
    ``1 + x0 - x1`` provably >= 1 even though the box alone allows -1."""
    m = dm.Model("range")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=1.0)
    m.minimize(x[0])
    m.subject_to(x[1] - x[0] <= 0.0)

    ctx = build_linear_context(m)
    assert ctx is not None
    coeffs = np.array([1.0, -1.0])
    lo, hi = ctx.affine_range(coeffs, 1.0)
    assert lo >= 1.0 - 1e-6, f"the linear row was not used: lo={lo}"
    assert hi <= 2.0 + 1e-6
