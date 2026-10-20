"""Regression lock for issue #1825: linear-row extraction summed array-valued constants.

``x >= np.array([5., 1.])`` is ONE ``Constraint`` object standing for two rows,
``x >= 5`` and ``x >= 1``. The one-row linear extractors -- the GDP layer's
``_extract_body_coeffs`` (mbigm's M LP, LOA/OA masters, GBD, decomposition) and
OBBT's ``_extract_linear_constraints`` / ``_extract_linear_objective_with_offset``
(``run_obbt``, ``bootstrap_finite_bounds``, the mbigm LP polytope, RLT-1 and Shor)
-- read a ``Constant``/``Parameter`` leaf as ``float(np.sum(value))``. That made the
two rows the single row ``x >= 6``, which excludes the optimum ``x = 5``:

* mbigm computed its M over the too-small region and certified 1.0 (truth 10) and
  9.0 (truth 5);
* LOA's master used the same row and certified 9.0 (truth 5);
* ``run_obbt`` tightened ``x >= 6`` and AMP certified 6.0 (truth 5);
* Benders' master used the same row and certified 6.0 (truth 5);
* the hull's disjunct-bound reader raised ``TypeError`` on the array (summing it
  instead, the tempting "fix", reads ``x >= [5, 1]`` as ``x >= 6``).

The fix (``_relax/scalarize.scalar_leaf_value``) declines the whole row when a
leaf has more than one element. Declining a row only enlarges the LP region, so
every M and bound read off it is looser, never invalid. Scalar rows are untouched.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import gdp_reformulate as gr
from discopt._relax import obbt as ob
from discopt.modeling.core import Constant


def _solve(m, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.solve(time_limit=30, **kw)


def _global_vector_row_model(use_parameter: bool = False):
    m = dm.Model("vg")
    x = m.continuous("x", lb=0, ub=10)
    rhs = np.array([5.0, 1.0])
    m.subject_to(x >= (m.parameter("p", value=rhs) if use_parameter else rhs))
    m.either_or([[x >= 9], [x <= 5.5]], name="v")
    m.minimize(x)
    return m


def _open_box_vector_row_model(scalar_companion: bool = False):
    m = dm.Model("vo")
    x = m.continuous("x", lb=-1e30, ub=10)
    y = m.continuous("y", lb=-1, ub=1)
    m.subject_to(x >= np.array([5.0, 1.0]))
    if scalar_companion:
        # A scalar row the extractor must keep, so its output is never ``None`` and
        # the assertions on it always execute.
        m.subject_to(x <= 9)
    m.minimize(x + y * y)
    return m


# ── the leaf rule ──


@pytest.mark.smoke
def test_scalar_leaf_value_refuses_arrays():
    from discopt._relax.scalarize import scalar_leaf_value

    assert scalar_leaf_value(Constant(np.array([5.0, 1.0]))) is None
    assert scalar_leaf_value(Constant(np.array(3.0))) == 3.0
    assert scalar_leaf_value(Constant(np.array([2.0]))) == 2.0


# ── _extract_body_coeffs (mbigm M, LOA/OA, GBD, decomposition) ──


@pytest.mark.smoke
def test_extract_body_coeffs_declines_vector_constant():
    m = dm.Model("d")
    x = m.continuous("x", lb=0, ub=10)
    assert gr._extract_body_coeffs((x >= np.array([5.0, 1.0])).body, m, 1) is None
    p = m.parameter("p", value=np.array([5.0, 1.0]))
    assert gr._extract_body_coeffs((x >= p).body, m, 1) is None
    # A scalar row is extracted exactly as before.
    c_vec, offset = gr._extract_body_coeffs((x >= 5.0).body, m, 1)
    assert c_vec.tolist() == [-1.0] and offset == 5.0


@pytest.mark.smoke
def test_mbigm_vector_row_in_disjunct():
    m = dm.Model("vb")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= np.array([5.0, 1.0])], [x >= 8]], name="v")
    m.maximize(x)
    r = _solve(m, gdp_method="mbigm")
    assert r.status == "optimal"
    assert r.objective == pytest.approx(10.0, abs=1e-6)  # was a certified 1.0


@pytest.mark.smoke
@pytest.mark.parametrize("use_parameter", [False, True])
def test_mbigm_vector_global_row(use_parameter):
    r = _solve(_global_vector_row_model(use_parameter), gdp_method="mbigm")
    assert r.status == "optimal"
    assert r.objective == pytest.approx(5.0, abs=1e-6)  # was a certified 9.0


@pytest.mark.smoke
def test_loa_vector_global_row():
    r = _solve(_global_vector_row_model(), gdp_method="loa")
    assert r.status == "optimal"
    assert r.objective == pytest.approx(5.0, abs=1e-6)  # was a certified 9.0


@pytest.mark.smoke
def test_hull_vector_row_in_disjunct_does_not_raise():
    # ``_extract_disjunct_bounds`` did ``float()`` on the array constant and raised
    # TypeError; the row is now skipped and the variable keeps its global bounds.
    m = dm.Model("vh")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= np.array([5.0, 1.0])], [x >= 8]], name="v")
    m.maximize(x)
    r = _solve(m, gdp_method="hull")
    assert r.status == "optimal"
    assert r.objective == pytest.approx(10.0, abs=1e-6)


@pytest.mark.smoke
@pytest.mark.parametrize("gdp_method", ["hull", "auto"])
def test_hull_vector_lower_bound_row_in_disjunct(gdp_method):
    # ``x >= [5, 1]`` is ``x >= 5``. Reading the array constant as one bound by
    # summing it gives the disjunct bound ``x >= 6`` and the hull certifies 6.0;
    # the row must be skipped (global bounds kept) instead.
    m = dm.Model("vhl")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x >= np.array([5.0, 1.0])], [x >= 7]], name="v")
    m.minimize(x)
    r = _solve(m, gdp_method=gdp_method)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(5.0, abs=1e-6)


# ── decomposition (Benders reads rows through ``_extract_body_coeffs``) ──


@pytest.mark.smoke
def test_benders_vector_row():
    m = dm.Model("bd")
    y = m.integer("y", lb=0, ub=2)
    x = m.continuous("x", lb=0, ub=20)
    m.subject_to(x + 2 * y >= np.array([5.0, 1.0]))
    m.minimize(3 * y + x)
    r = _solve(m, decomposition="benders")
    assert r.status == "optimal"
    assert r.objective == pytest.approx(5.0, abs=1e-6)  # was a certified 6.0
    assert r.bound <= 5.0 + 1e-6


# ── OBBT extraction (run_obbt, bootstrap_finite_bounds, RLT-1, Shor, mbigm LP) ──


@pytest.mark.smoke
def test_obbt_extraction_keeps_the_optimum_feasible():
    model = _open_box_vector_row_model(scalar_companion=True)
    A_ub, b_ub, _, _, _ = ob._extract_linear_constraints(model)
    assert A_ub is not None
    # Only the scalar companion row ``x <= 9``; the vector row is declined.
    assert A_ub.tolist() == [[1.0, 0.0]] and b_ub.tolist() == [9.0]
    assert np.all(A_ub @ np.array([5.0, 0.0]) <= b_ub + 1e-12)


@pytest.mark.smoke
def test_obbt_extraction_sum_of_broadcast_is_not_folded_per_leaf():
    # sum(x + [1, 2]) >= 5 is 2x + 3 >= 5, i.e. x >= 1. Folding each leaf as a
    # scalar gave x + 3 >= 5, i.e. x >= 2, which cuts x = 1.
    m = dm.Model("s")
    x = m.continuous("x", lb=-10, ub=10)
    m.subject_to(dm.sum(x + np.array([1.0, 2.0])) >= 5)
    m.subject_to(x <= 9)  # scalar companion, so A_ub is never None
    m.minimize(x)
    A_ub, b_ub, _, _, _ = ob._extract_linear_constraints(m)
    assert A_ub is not None
    assert np.all(A_ub @ np.array([1.0]) <= b_ub + 1e-12)


@pytest.mark.smoke
def test_obbt_extraction_full_reduction_of_bare_constant_kept():
    # sum([1, 2]) is exactly 3: still extracted, so scalar models are unchanged.
    m = dm.Model("s2")
    x = m.continuous("x", lb=-10, ub=10)
    m.subject_to(x + dm.sum(np.array([1.0, 2.0])) <= 5)
    m.minimize(x)
    A_ub, b_ub, _, _, _ = ob._extract_linear_constraints(m)
    assert A_ub.tolist() == [[1.0]] and b_ub.tolist() == [2.0]


@pytest.mark.smoke
def test_obbt_objective_extraction_declines_sum_of_broadcast():
    m = dm.Model("so")
    x = m.continuous("x", lb=0, ub=10)
    m.minimize(dm.sum(x + np.array([1.0, 2.0])))  # 2x + 3
    res = ob._extract_linear_objective_with_offset(m, 1)
    # Declining is sound; a per-leaf fold would give ``x + 3``.
    assert res is None or (res[0].tolist() == [2.0] and res[1] == 3.0)


@pytest.mark.smoke
def test_obbt_objective_extraction_folds_full_reduction_of_bare_constant():
    # sum([1, 2]) is exactly 3, so the objective is extracted, offset included.
    m = dm.Model("so2")
    x = m.continuous("x", lb=0, ub=10)
    m.minimize(2 * x + dm.sum(np.array([1.0, 2.0])))
    res = ob._extract_linear_objective_with_offset(m, 1)
    assert res is not None
    assert res[0].tolist() == [2.0] and res[1] == 3.0


@pytest.mark.smoke
def test_run_obbt_does_not_tighten_past_vector_row():
    res = ob.run_obbt(_open_box_vector_row_model())
    assert res.tightened_lb[0] <= 5.0  # was 6.0


@pytest.mark.smoke
def test_bootstrap_finite_bounds_does_not_tighten_past_vector_row():
    lb = np.array([-np.inf, -1.0])
    ub = np.array([10.0, 1.0])
    new_lb, _, _, _ = ob.bootstrap_finite_bounds(_open_box_vector_row_model(), lb, ub)
    assert new_lb[0] <= 5.0  # was 5.999993


@pytest.mark.smoke
def test_amp_vector_row():
    m = dm.Model("va")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=-1, ub=1)
    m.subject_to(x >= np.array([5.0, 1.0]))
    m.minimize(x + x * y * y)
    r = _solve(m, solver="amp")
    assert r.status == "optimal"
    assert r.objective == pytest.approx(5.0, abs=1e-6)  # was a certified 6.0
