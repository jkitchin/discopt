"""Sliced IndexExpressions must not crash nonlinear bound tightening.

Regression test for a crash found while running AMP on a DAE-collocation model:
``FlatVariableMetadata.scalar_flat_index`` assumed every ``IndexExpression`` on
a Variable carries one integer per axis and fed the raw subscript to
``np.ravel_multi_index``. The DAE transcriber (``discopt.dae.DAEBuilder``)
builds vectorized constraints from *sliced* expressions such as ``var[:, 1:]``
and ``var[1:, 0]``, so any equality-rule pass over such a model raised
``TypeError: only int indices permitted`` inside AMP presolve. A sliced
subscript addresses many scalars — the contract answer is ``None``, not a
crash.
"""

import discopt.modeling as dm
import numpy as np
from discopt._jax.nonlinear_bound_tightening import (
    build_flat_variable_metadata,
    tighten_nonlinear_bounds,
)


def _dae_like_model():
    """Compact model with the sliced-index + nonlinear equality shape DAEBuilder emits."""
    m = dm.Model("sliced")
    x = m.continuous("x", shape=(3, 2), lb=0.0, ub=1.5)
    a = m.continuous("a", lb=-10.0, ub=10.0)
    m.subject_to(x[1:, 0] == x[:-1, 1])  # continuity-style sliced equality
    m.subject_to(x[:, 1] == a * dm.exp(-x[:, 0]))  # nonlinear vector equality
    m.minimize(dm.sum((x - 0.5) ** 2))
    return m, x, a


def test_scalar_flat_index_returns_none_for_sliced_subscripts():
    m, x, a = _dae_like_model()
    meta = build_flat_variable_metadata(m)

    assert meta.scalar_flat_index(x[1:, 0]) is None
    assert meta.scalar_flat_index(x[:, 1]) is None
    assert meta.scalar_flat_index(x[0]) is None  # partial index covers a row
    # Scalar subscripts still resolve, including negative (Python-style) indices.
    assert meta.scalar_flat_index(x[1, 1]) == 3
    assert meta.scalar_flat_index(x[-1, -1]) == 5
    assert meta.scalar_flat_index(a) == 6


def test_tighten_nonlinear_bounds_survives_sliced_constraints():
    m, _, _ = _dae_like_model()
    n = sum(v.size for v in m._variables)
    lb = np.concatenate([np.zeros(6), [-10.0]])
    ub = np.concatenate([np.full(6, 1.5), [10.0]])

    out_lb, out_ub, _stats = tighten_nonlinear_bounds(m, lb, ub)

    assert out_lb.shape == (n,)
    # Tightening never widens the box.
    assert np.all(out_lb >= lb - 1e-12)
    assert np.all(out_ub <= ub + 1e-12)


def test_export_var_index_refuses_sliced_subscript_loudly():
    """`export/_extract._var_index` must raise ValueError (not TypeError) on a slice.

    This is a *refusal* path, not an analysis path: a single variable reference
    genuinely cannot be a sliced subscript, so the contract is a clear
    ValueError naming the variable and subscript — never a bare
    ``TypeError: only int indices permitted`` from np.ravel_multi_index.
    """
    import pytest
    from discopt.export._extract import _var_index, flatten_variables

    m, x, _ = _dae_like_model()
    flat_vars = flatten_variables(m)
    model_vars = list(m._variables)

    # Scalar element still resolves.
    assert _var_index(x[1, 1], flat_vars, model_vars) == 3
    # Sliced subscript is refused with an actionable message.
    with pytest.raises(ValueError, match=r"Non-scalar subscript.*'x'"):
        _var_index(x[:, 1], flat_vars, model_vars)
    with pytest.raises(ValueError, match=r"Non-scalar subscript.*'x'"):
        _var_index(x[1:, 0], flat_vars, model_vars)


def test_cut_result_to_dense_refuses_sliced_subscript_loudly():
    """`callbacks.cut_result_to_dense` must raise ValueError on a sliced cut key."""
    import pytest
    from discopt.callbacks import CutResult, cut_result_to_dense

    m, x, _ = _dae_like_model()

    # A scalar element key works.
    ok = CutResult(terms=[(x[1, 1], 1.0)], sense="<=", rhs=0.0)
    coeffs, rhs, sense = cut_result_to_dense(ok, m)
    assert coeffs[3] == 1.0

    # A sliced key is refused with an actionable message.
    bad = CutResult(terms=[(x[:, 1], 1.0)], sense="<=", rhs=0.0)
    with pytest.raises(ValueError, match=r"Non-scalar subscript.*'x'"):
        cut_result_to_dense(bad, m)


def test_convexity_struct_hash_survives_sliced_subscripts():
    """`_struct_hash` / `_hash_index` on tuple subscripts containing slices.

    Same trigger as above, one layer deeper: AMP's convexity classification
    hashes expression structure, and ``x[1:, 0]`` produces an index tuple whose
    surrogate must be hashable element-wise (a bare ``np.asarray`` of it builds
    an object array whose ``tolist()`` smuggles the slice back in).
    """
    from discopt._jax.convexity.rules import _hash_index, _struct_hash

    m, x, _ = _dae_like_model()
    key = _hash_index((slice(1, None), 0))
    hash(key)  # must not raise
    # Full structural hash over a sliced expression graph must not raise either.
    h = _struct_hash(x[1:, 0] - x[:-1, 1], {})
    assert isinstance(h, int)
