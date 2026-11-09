"""The HiGHS MILP master must fit whole-model rows to HiGHS's coefficient window.

``passModel`` drops every ``|a_ij| <= small_matrix_value`` (1e-9) and answers
``kWarning``. The lazy path already fitted each cut with ``_prepare_cut_row``
(#1066), but ``solve_milp`` -- the master plain OA rebuilds every iteration --
passed rows as given and read the warning as ``HiGHS rejected the master model``.
With HiGHS as the convex route's default master that raise sent ``st_test1``,
``nvs12``, ``cvxnonsep_psig30`` and ``clay0303hfsg`` straight to the spatial
fallback without running OA at all (measured on the route panel, 2026-10-04).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("highspy")

from discopt.solvers import SolveStatus  # noqa: E402
from discopt.solvers.milp_highs import _fit_rows_to_window, solve_milp  # noqa: E402


@pytest.fixture(autouse=True)
def _pin_highs_oa_master(monkeypatch):
    """These tests cover the HiGHS OA master, an opt-out since highs-parity D3 flipped
    the default to the in-house master (2026-11-07). Pin it, as a test that sets the
    variable itself still overrides."""
    monkeypatch.setenv("DISCOPT_CONVEX_ROUTE_OA_MASTER", "highs")


DATA = pathlib.Path(__file__).parent / "data"


def test_a_row_with_a_tiny_entry_is_rescaled_not_rejected():
    """``x + 1e-11 y <= 3.5``: an exact power-of-two rescale lifts the entry."""
    r = solve_milp(
        c=np.array([-1.0, 0.0]),
        A_ub=np.array([[1.0, 1e-11]]),
        b_ub=np.array([3.5]),
        bounds=[(0, 10), (0, 5)],
        integrality=np.array([1, 0]),
    )
    assert r.status == SolveStatus.OPTIMAL
    assert r.objective == pytest.approx(-3.0)
    assert r.bound <= r.objective + 1e-9


def test_rows_without_a_tiny_entry_pass_bit_for_bit():
    a = sp.csr_matrix(np.array([[1.0, 2.0], [3.0, 0.0]]))
    rhs = np.array([1.0, 2.0])
    out_a, out_rhs = _fit_rows_to_window(
        a, rhs, np.zeros(2), np.ones(2), 1e-9, 1e15, equality=False
    )
    assert out_a is a and out_rhs is rhs


def test_a_dropped_term_loosens_the_row_on_its_valid_side():
    """Span 1e-14..1e11 exceeds the window: the tiny term is dropped and absorbed."""
    a = sp.csr_matrix(np.array([[1e11, -1e-14]]))
    out_a, out_rhs = _fit_rows_to_window(
        a, np.array([1.0]), np.zeros(2), np.array([10.0, 5.0]), 1e-9, 1e15, equality=False
    )
    row = out_a.toarray().ravel()
    assert row[1] == 0.0
    scale = row[0] / 1e11
    # Implied by the original: every x the original row admits, this one admits.
    assert out_rhs >= scale * (1.0 + 5e-14)


def test_an_unbounded_binding_column_is_refused():
    a = sp.csr_matrix(np.array([[1e11, -1e-14]]))
    with pytest.raises(ValueError, match="unbounded on the binding side"):
        _fit_rows_to_window(
            a,
            np.array([1.0]),
            np.zeros(2),
            np.array([10.0, np.inf]),
            1e-9,
            1e15,
            equality=False,
        )


def test_an_equality_row_that_would_lose_a_term_is_refused():
    a = sp.csr_matrix(np.array([[1e11, 1e-14]]))
    with pytest.raises(ValueError, match="equality row 0"):
        _fit_rows_to_window(a, np.array([1.0]), np.zeros(2), np.ones(2), 1e-9, 1e15, equality=True)


def test_an_equality_row_is_exactly_rescaled_when_it_fits():
    a = sp.csr_matrix(np.array([[1.0, 1e-11]]))
    out_a, out_rhs = _fit_rows_to_window(
        a, np.array([2.0]), np.zeros(2), np.ones(2), 1e-9, 1e15, equality=True
    )
    row = out_a.toarray().ravel()
    scale = row[0]
    assert np.frexp(scale)[0] == 0.5  # a power of two: exact
    assert row[1] == 1e-11 * scale and out_rhs[0] == 2.0 * scale


def test_st_test1_runs_oa_on_the_highs_master_instead_of_falling_back(monkeypatch):
    from discopt.modeling.core import from_nl

    monkeypatch.setenv("DISCOPT_CONVEX_ROUTE_OA_MASTER", "highs")
    path = next(DATA.glob("minlplib*/st_test1.nl"))
    r = from_nl(str(path)).solve(time_limit=20)
    route = r.algorithm_route or ""
    assert "master=highs" in route
    assert "fell back" not in route, route
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(0.0, abs=1e-6)


def test_a_row_with_an_oversize_entry_is_scaled_down_not_rejected():
    """``1e16 x + y <= 3.5e16``: HiGHS refuses ``|a| >= 1e15`` outright (kError)."""
    r = solve_milp(
        c=np.array([-1.0, 0.0]),
        A_ub=np.array([[1e16, 1.0]]),
        b_ub=np.array([3.5e16]),
        bounds=[(0, 10), (0, 5)],
        integrality=np.array([1, 0]),
    )
    assert r.status == SolveStatus.OPTIMAL
    assert r.objective == pytest.approx(-3.0)


def test_the_lazy_cut_row_is_scaled_down_by_a_power_of_two():
    from discopt.solvers.milp_highs import _prepare_cut_row

    idx, vals, rhs = _prepare_cut_row(
        np.array([1.6e22, 2.0]), 4.0e22, np.zeros(2), np.ones(2), 1e-9, 1e15
    )
    scale = vals[0] / 1.6e22
    assert np.frexp(scale)[0] == 0.5 and scale < 1.0
    assert np.abs(vals).max() < 1e15
    assert list(idx) == [0, 1] and vals[1] == 2.0 * scale and rhs == 4.0e22 * scale
