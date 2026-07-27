"""#875 item 3: ``build_linear_context`` must not densify one row per constraint.

The convexity analyser's ``LinearContext`` built a dense length-``n_vars`` array for
every linear constraint and kept them all in a ``vstack``, making assembly
``O(rows x n_vars)`` in time *and* memory regardless of how few columns each body
actually touches. On ``watercontamination0202`` (106,711 vars / 107,209 rows) that
was **15.8 s** of a constant ~17 s root-setup overrun — the single largest remaining
item after PR #878 took the instance from 19.3x to 1.57x over ``time_limit``.

Entry experiment (``scratchpad/issue875_linear_context_probe.py``), holding the row
count FIXED at 400 two-term rows (800 nonzeros total) and varying only ``n_vars``, so
any growth can only come from the dense row:

    n_vars    before            after
     2,000    0.076 s /   6.4 MB    0.018 s / 6.4 KB
     8,000    0.220 s /  25.6 MB    0.012 s / 6.4 KB
    32,000    0.760 s / 102.4 MB    0.044 s / 6.4 KB
   128,000    2.406 s / 409.6 MB    0.160 s / 6.4 KB

Bound-neutral (CLAUDE.md §5 regime 1): over all 119 in-repo instances the sparse
matrices are **bit-identical** to the dense assembly they replace — 0 mismatches
(``scratchpad/issue875_linear_context_neutrality.py``).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from discopt._jax.convexity.linear_context import (  # noqa: E402
    LinearContext,
    build_linear_context,
    extract_affine,
    extract_affine_sparse,
)

pytestmark = pytest.mark.smoke


def _model(n_vars: int, n_rows: int = 40) -> dm.Model:
    """``n_rows`` two-term rows over ``n_vars`` variables (support independent of n)."""
    m = dm.Model(f"lc{n_vars}")
    xs = [m.continuous(f"x{i}", lb=0.0, ub=10.0) for i in range(n_vars)]
    for r in range(n_rows):
        m.subject_to(xs[(2 * r) % n_vars] + 2.0 * xs[(2 * r + 1) % n_vars] <= 5.0)
    m.minimize(xs[0])
    return m


def test_matrices_are_sparse():
    """Fail-before: both were dense ``ndarray`` from ``np.vstack``."""
    ctx = build_linear_context(_model(200))
    assert ctx is not None
    assert sp.issparse(ctx.A_ub), f"A_ub is {type(ctx.A_ub).__name__}, expected sparse"
    assert sp.issparse(ctx.A_eq), f"A_eq is {type(ctx.A_eq).__name__}, expected sparse"


def test_storage_does_not_grow_with_variable_count():
    """THE defect: at a fixed row count the matrix storage was ``rows * n_vars * 8``.

    Fail-before: 40 rows over 200 vars is 64,000 bytes and over 3,200 vars is
    1,024,000 — a 16x growth for the same 80 nonzeros. After, both are 640 bytes."""
    small = build_linear_context(_model(200))
    large = build_linear_context(_model(3_200))
    assert small is not None and large is not None
    assert small.A_ub.nnz == large.A_ub.nnz == 80
    assert small.A_ub.data.nbytes == large.A_ub.data.nbytes, (
        "matrix storage still scales with the variable count"
    )


def test_rows_are_bit_identical_to_the_dense_assembly():
    """Bound-neutral: same numbers, different storage. ``array_equal``, not
    ``allclose`` — a refactor that shifts a coefficient at all is wrong."""
    m = dm.Model("mix")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=-2.0, ub=3.0)
    z = m.continuous("z", lb=0.0, ub=1.0)
    m.subject_to(x + 2.0 * y <= 7.0)
    m.subject_to(3.0 * x - y >= 1.0)  # negated into the <= system
    m.subject_to(y + z == 2.0)
    m.minimize(x)
    ctx = build_linear_context(m)
    assert ctx is not None

    n = 3
    want_ub = np.vstack([np.array([1.0, 2.0, 0.0]), -np.array([3.0, -1.0, 0.0])])
    want_b_ub = np.array([7.0, -1.0])
    want_eq = np.array([[0.0, 1.0, 1.0]])
    want_b_eq = np.array([2.0])

    assert ctx.A_ub.shape == (2, n) and ctx.A_eq.shape == (1, n)
    assert np.array_equal(ctx.A_ub.toarray(), want_ub)
    assert np.array_equal(ctx.b_ub, want_b_ub)
    assert np.array_equal(ctx.A_eq.toarray(), want_eq)
    assert np.array_equal(ctx.b_eq, want_b_eq)


def test_extract_affine_dense_view_is_unchanged():
    """``extract_affine`` is public and still returns the dense vector — the sparse
    core is an addition, not a replacement."""
    m = dm.Model("d")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(x)
    dense = extract_affine(x + 3.0 * y - 2.0, m, 2)
    assert dense is not None
    coeffs, const = dense
    assert isinstance(coeffs, np.ndarray) and coeffs.shape == (2,)
    assert np.array_equal(coeffs, np.array([1.0, 3.0]))
    assert const == pytest.approx(-2.0)

    sparse = extract_affine_sparse(x + 3.0 * y - 2.0, m, 2)
    assert sparse == ({0: 1.0, 1: 3.0}, -2.0)


def test_sparse_and_dense_refuse_the_same_expressions():
    """Same recognition rules and same refusals — a nonlinear body yields ``None``
    from both, so the sparse core cannot silently admit a row the dense one dropped."""
    m = dm.Model("r")
    x = m.continuous("x", lb=1.0, ub=2.0)
    y = m.continuous("y", lb=1.0, ub=2.0)
    m.minimize(x)
    for expr in (x * y, x / y, x**2):
        assert extract_affine(expr, m, 2) is None
        assert extract_affine_sparse(expr, m, 2) is None


def test_out_of_range_columns_are_dropped_by_both():
    """The dense fill silently skipped ``idx >= n_vars``; the sparse core must too."""
    m = dm.Model("o")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(x)
    # n_vars=1 hides column 1 from both forms.
    assert extract_affine_sparse(x + y, m, 1) == ({0: 1.0}, 0.0)
    dense = extract_affine(x + y, m, 1)
    assert dense is not None and np.array_equal(dense[0], np.array([1.0]))


def test_all_zero_rows_still_count_as_linear_rows():
    """The ``.size`` trap. On a dense array ``A.size`` is ``rows * n_vars``, but on a
    sparse one it is the NONZERO count — so a row set whose coefficients all cancel
    would have read as "no linear rows" and skipped the LP the dense form ran.
    ``_n_rows`` uses the row count, which matches the dense behaviour exactly."""
    zero_rows = sp.csr_matrix(
        (np.zeros(0), np.zeros(0, dtype=np.int64), np.zeros(3, dtype=np.int64)), shape=(2, 4)
    )
    assert zero_rows.nnz == 0
    assert LinearContext._n_rows(zero_rows) == 2  # not 0
    empty = sp.csr_matrix((0, 4))
    assert LinearContext._n_rows(empty) == 0
    # Dense arrays keep working (hand-built contexts in other tests pass them).
    assert LinearContext._n_rows(np.zeros((5, 4))) == 5


def test_constraint_aware_sign_reasoning_still_works():
    """End-to-end: the context exists to prove an argument's sign from linear rows.
    ``log(1 + x - y)`` with ``y <= x`` is the module docstring's own example — the
    argument is ``>= 1 > 0``, so the model classifies. This is what would break if
    the sparse rows reached the LP wrong."""
    from discopt._jax.convexity.rules import classify_model

    m = dm.Model("ctx")
    x = m.continuous("x", lb=0.0, ub=5.0)
    y = m.continuous("y", lb=0.0, ub=5.0)
    m.subject_to(y <= x)
    m.minimize(-dm.log(1.0 + x - y))
    ctx = build_linear_context(m)
    assert ctx is not None and ctx.A_ub.shape[0] == 1
    lo, hi = ctx.affine_range(np.array([1.0, -1.0]), 1.0)
    assert lo >= 1.0 - 1e-6, f"linear rows should prove the log argument >= 1, got {lo}"
    assert hi <= 6.0 + 1e-6
    classify_model(m)  # must not raise on the sparse context


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
