"""Regression tests for C-29 and C-30 (false certified answers on the DEFAULT
linear-body classify/extract path in ``discopt._jax.problem_classifier``).

Both bugs certified a *wrong* answer on ``m.solve()`` with no flags:

* **C-29** — a vector-valued constraint body (``a + b >= 1`` with ``a, b`` shape
  ``(2,)``) is one ``Constraint`` whose body is array-valued. The algebraic
  extractor summed every element into a single row, so ``a + b >= 1`` became
  ``Σa + Σb >= 1`` and a point infeasible by the model's own semantics was
  certified optimal (objective 1.0 for a problem whose true optimum is 2.0).

* **C-30** — ``maximize sum(const * var)`` raised a raw ``ValueError`` from
  ``_eval_const`` (``float(v.item())`` on a size>1 array), aborting the algebraic
  walk and falling over to the autodiff extractor, which — unlike every other
  extractor — never applied the maximize sense negation. The solver then
  *minimized* a maximize model and returned 0 instead of the true max.

The fix makes both algebraic walkers refuse to collapse an array node in scalar
position (``_NotLinearError``) so the body routes to the per-component autodiff
extractor, and makes that autodiff extractor apply the maximize negation. These
tests call the extractor directly (sub-second) plus one end-to-end solve to lock
the certificate.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt._jax import problem_classifier as pc


@pytest.mark.smoke
def test_c29_vector_body_not_collapsed_to_single_row():
    """A per-element constraint ``a + b >= 1`` (shape (2,)) must extract as two
    rows, not one summed row. Pre-fix the algebraic walk produced one collapsed
    row; the fix refuses (``_NotLinearError``) so the per-component autodiff
    extractor produces one row per element.
    """
    m = Model()
    a = m.continuous("a", shape=(2,), lb=0.0, ub=1.0)
    b = m.continuous("b", shape=(2,), lb=0.0, ub=1.0)
    m.minimize(dm.sum(a) + dm.sum(b))
    m.subject_to(a + b >= 1)

    n_orig = 4  # a[0], a[1], b[0], b[1]

    # The algebraic walker must NOT silently collapse the vector body to one row.
    with pytest.raises(pc._NotLinearError):
        pc._extract_linear_coefficients((a + b >= 1).body, m, n_orig)

    # The full dispatcher routes to the per-component extractor: two rows, one per
    # element (each with a distinct slack), never a single summed row.
    lp = pc.extract_lp_data(m)
    A = np.asarray(lp.A_eq)
    # 2 constraint rows (one per element), n_orig + 2 slack columns.
    assert A.shape == (2, n_orig + 2)
    # Row r couples exactly a[r] and b[r] (plus its own slack), never both elements.
    assert np.count_nonzero(A[0, :n_orig]) == 2
    assert np.count_nonzero(A[1, :n_orig]) == 2
    # a[0]/b[0] appear only in row 0; a[1]/b[1] only in row 1.
    assert A[0, 1] == 0.0 and A[0, 3] == 0.0  # a[1], b[1] absent from row 0
    assert A[1, 0] == 0.0 and A[1, 2] == 0.0  # a[0], b[0] absent from row 1


@pytest.mark.smoke
def test_c29_solve_certifies_feasible_point():
    """End-to-end: the certified optimum is 2.0 and the returned point is
    feasible per element (pre-fix: objective 1.0 at an infeasible point)."""
    m = Model()
    a = m.continuous("a", shape=(2,), lb=0.0, ub=1.0)
    b = m.continuous("b", shape=(2,), lb=0.0, ub=1.0)
    m.minimize(dm.sum(a) + dm.sum(b))
    m.subject_to(a + b >= 1)

    res = m.solve()
    assert res.objective == pytest.approx(2.0, abs=1e-4)
    xa = np.asarray(res.value(a))
    xb = np.asarray(res.value(b))
    # No per-element constraint is violated — the false-certified point had
    # a + b == 0.5 < 1 element-wise.
    assert np.all(xa + xb >= 1.0 - 1e-5)


@pytest.mark.smoke
def test_c29_milp_set_cover_vector_body():
    """MILP set-cover with a vector body: ``y + z >= 1`` (shape (3,) binaries)
    must certify 3.0 (pre-fix: 1.0)."""
    m = Model()
    y = m.binary("y", shape=(3,))
    z = m.binary("z", shape=(3,))
    m.minimize(dm.sum(y) + dm.sum(z))
    m.subject_to(y + z >= 1)
    res = m.solve()
    assert res.objective == pytest.approx(3.0, abs=1e-4)


@pytest.mark.smoke
def test_c30_eval_const_refuses_array_with_notlinear():
    """``_eval_const`` on a non-scalar array constant must raise ``_NotLinearError``
    (not a raw ``ValueError``), so the algebraic walk routes to a sense-aware path
    instead of the old sense-dropping fallback."""
    from discopt.modeling.core import Constant

    arr = Constant(np.array([1.0, 1.0]))
    with pytest.raises(pc._NotLinearError):
        pc._eval_const(arr)

    # A scalar / size-1 array constant still evaluates fine.
    assert pc._eval_const(Constant(np.array(3.0))) == 3.0
    assert pc._eval_const(Constant(np.array([2.5]))) == 2.5


@pytest.mark.smoke
def test_c30_maximize_sense_preserved_in_extracted_c():
    """``extract_lp_data`` on ``maximize sum(const * var)`` must return a negated
    ``c`` (solvers minimize). Pre-fix it returned the un-negated ``[1, 1, 0]``."""
    m = Model()
    x = m.continuous("x", shape=(2,), lb=0.0, ub=10.0)
    m.maximize(dm.sum(np.array([1.0, 1.0]) * x))
    m.subject_to(dm.sum(np.array([1.0, 1.0]) * x) <= 4)

    lp = pc.extract_lp_data(m)
    c = np.asarray(lp.c)
    # Original variable coefficients negated for the maximize sense.
    assert c[0] == pytest.approx(-1.0)
    assert c[1] == pytest.approx(-1.0)


@pytest.mark.smoke
def test_c30_autodiff_extractor_applies_maximize_sense():
    """The autodiff LP fallback itself (the sense-dropping culprit) must now
    negate ``c`` for a maximize model."""
    m = Model()
    x = m.continuous("x", shape=(2,), lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.maximize(x[0] + x[1] + y)
    m.subject_to(x[0] + x[1] + y <= 4)

    lp = pc._extract_lp_data_autodiff(m)
    c = np.asarray(lp.c)
    assert np.all(c[:3] == pytest.approx(-1.0))


@pytest.mark.smoke
def test_c30_solve_maximize_returns_true_max():
    """End-to-end: ``maximize sum(const * var) s.t. sum(const * var) <= 4``
    certifies 4.0 (pre-fix: ~0)."""
    m = Model()
    x = m.continuous("x", shape=(2,), lb=0.0, ub=10.0)
    m.maximize(dm.sum(np.array([1.0, 1.0]) * x))
    m.subject_to(dm.sum(np.array([1.0, 1.0]) * x) <= 4)
    res = m.solve()
    assert res.objective == pytest.approx(4.0, abs=1e-4)


@pytest.mark.smoke
def test_legit_sum_stays_on_fast_algebraic_path():
    """Bound-neutral guard: a genuine ``sum(array_var)`` (scalar body) must still
    extract on the fast algebraic path as ONE row — the fix must not over-refuse
    legitimate reductions."""
    m = Model()
    a = m.continuous("a", shape=(3,), lb=0.0, ub=5.0)
    m.minimize(dm.sum(a))
    m.subject_to(dm.sum(a) >= 6)

    # Algebraic extraction succeeds (does not raise) and yields a single row.
    lp = pc.extract_lp_data_algebraic(m)
    A = np.asarray(lp.A_eq)
    assert A.shape[0] == 1
    # All three components carry the same (uniform) coefficient in the summed row.
    assert np.count_nonzero(A[0, :3]) == 3
    assert m.solve().objective == pytest.approx(6.0, abs=1e-4)
