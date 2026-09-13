"""One LP standard form, shared by both marshalers.

Before this consolidation the two marshaling paths that reach the same Rust
driver (``solve_milp_csc_py``) laid the columns out *differently*:

* the ``Model.solve`` extractors in :mod:`discopt._relax.problem_classifier`
  gave a logical (slack) column only to inequality rows, so a model with any
  equality row produced a basis short of ``m`` columns -- which
  ``lp/gomory.rs`` (``SepGomoryShortBasis``), ``lp/simplex/dual.rs``
  (``DualPrepRejectShape``) and ``bnb/milp_driver.rs``'s ``BaseRows::build``
  (``SubstDropNoSlack``) all refuse to work with;
* ``discopt.solvers.milp_simplex._marshal_std_form`` split every equality into
  two opposing ``<=`` rows so both halves got a logical -- the right shape, at
  the cost of an always-tight opposing pair.

:mod:`discopt._relax.std_form` is now the single definition, and
``DISCOPT_LP_ROW_LOGICALS`` selects the layout for both. These tests pin (a) the
flag-OFF arm to the legacy behaviour, (b) the flag-ON layout contract the engine
indexes by, (c) that the two entry points agree, and (d) that the equality-row
classification survives the round trip -- the property the MILP feasibility gate
depends on.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.std_form import (
    INF,
    LOGICAL_COEF,
    logical_block,
    logical_is_fixed,
    row_logicals_enabled,
)
from discopt.solver import _decompose_eq_slack_form
from discopt.solvers.milp_simplex import _marshal_std_form

# min -x0 - x1  s.t.  x0 + x1 == 1,  x0 - x1 <= 0.5,  0 <= x <= 1.
# One equality and one inequality: the smallest model on which the two layouts
# differ at all.
_C = np.array([-1.0, -1.0])
_A_EQ = np.array([[1.0, 1.0]])
_B_EQ = np.array([1.0])
_A_UB = np.array([[1.0, -1.0]])
_B_UB = np.array([0.5])
_BOUNDS = [(0.0, 1.0), (0.0, 1.0)]


@pytest.fixture
def row_logicals(monkeypatch, request):
    """Set ``DISCOPT_LP_ROW_LOGICALS`` for one test.

    ``solver_tuning.current()`` re-reads the environment on every call outside a
    solve scope, so setting the variable is enough -- there is no cache to clear.
    The fixture asserts that, rather than assuming it.
    """
    from discopt import solver_tuning

    monkeypatch.setenv("DISCOPT_LP_ROW_LOGICALS", "1" if request.param else "0")
    assert solver_tuning.current().lp_row_logicals is bool(request.param)
    yield bool(request.param)


# --------------------------------------------------------------------------
# The module itself
# --------------------------------------------------------------------------


def test_logical_block_row_layout():
    """Every row gets a logical at ``n_struct + r``; an equality's is fixed."""
    block = logical_block(["le", "eq", "ge"], 4, eq_logicals=True)
    assert block.n_logical == 3
    assert block.col_of_row.tolist() == [4, 5, 6]
    assert block.coef.tolist() == [1.0, 1.0, -1.0]
    assert block.lb.tolist() == [0.0, 0.0, 0.0]
    assert block.ub.tolist() == [INF, 0.0, INF]
    rows, cols, vals = block.entries()
    assert rows.tolist() == [0, 1, 2]
    assert cols.tolist() == [4, 5, 6]
    assert vals.tolist() == [1.0, 1.0, -1.0]


def test_logical_block_legacy_layout_skips_equalities():
    """With ``eq_logicals=False`` an equality row gets no column at all."""
    block = logical_block(["le", "eq", "ge"], 4, eq_logicals=False)
    assert block.n_logical == 2
    assert block.col_of_row.tolist() == [4, -1, 5]
    rows, cols, vals = block.entries()
    assert rows.tolist() == [0, 2]
    assert cols.tolist() == [4, 5]
    assert vals.tolist() == [1.0, -1.0]
    assert block.ub.tolist() == [INF, INF]


def test_logical_block_refuses_an_unknown_sense():
    """A typo must raise, not silently produce a logical-free row."""
    with pytest.raises(ValueError, match="unknown constraint sense"):
        logical_block(["le", "<="], 2, eq_logicals=True)


def test_logical_coefficients_match_the_documented_table():
    assert LOGICAL_COEF == {"le": 1.0, "ge": -1.0, "eq": 1.0}


def test_logical_is_fixed_reads_the_box_not_the_pattern():
    assert logical_is_fixed(0.0)
    assert not logical_is_fixed(INF)
    assert not logical_is_fixed(np.inf)


@pytest.mark.parametrize("row_logicals", [False, True], indirect=True)
def test_row_logicals_enabled_follows_the_flag(row_logicals):
    assert row_logicals_enabled() is row_logicals
    # ``eq_logicals=None`` defers to the flag.
    block = logical_block(["eq"], 1)
    assert block.n_logical == (1 if row_logicals else 0)


# --------------------------------------------------------------------------
# The marshaler
# --------------------------------------------------------------------------


def _std_matrix(std):
    return sp.csc_matrix(
        (std.vals, std.row_idx, std.col_ptr), shape=(std.m, std.n + std.m)
    ).toarray()


def test_marshal_legacy_arm_splits_equalities_into_opposing_rows(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_ROW_LOGICALS", "0")
    std = _marshal_std_form(_C, _A_UB, _B_UB, _A_EQ, _B_EQ, _BOUNDS, None)

    # 1 inequality + 2 halves of the equality.
    assert std.m == 3
    assert std.n == 2
    A = _std_matrix(std)
    np.testing.assert_allclose(A[:, :2], [[1.0, -1.0], [1.0, 1.0], [-1.0, -1.0]])
    np.testing.assert_allclose(std.b, [0.5, 1.0, -1.0])
    # Every logical is free: the legacy arm has no fixed column.
    np.testing.assert_allclose(A[:, 2:], np.eye(3))
    np.testing.assert_allclose(std.ub[2:], [INF, INF, INF])


def test_marshal_row_logical_arm_keeps_one_row_per_equality(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_ROW_LOGICALS", "1")
    std = _marshal_std_form(_C, _A_UB, _B_UB, _A_EQ, _B_EQ, _BOUNDS, None)

    assert std.m == 2  # the equality is ONE row now
    assert std.n == 2
    A = _std_matrix(std)
    np.testing.assert_allclose(A[:, :2], [[1.0, -1.0], [1.0, 1.0]])
    np.testing.assert_allclose(std.b, [0.5, 1.0])
    # Still one logical per row -- the shape the engine indexes by -- but the
    # equality's is fixed at zero, so it adds a column and no freedom.
    np.testing.assert_allclose(A[:, 2:], np.eye(2))
    np.testing.assert_allclose(std.lb[2:], [0.0, 0.0])
    assert std.ub[2] == INF
    assert std.ub[3] == 0.0


@pytest.mark.parametrize("row_logicals", [False, True], indirect=True)
def test_marshal_gives_every_row_exactly_one_logical(row_logicals):
    """The invariant both call sites' ``n + m`` column count depends on."""
    std = _marshal_std_form(_C, _A_UB, _B_UB, _A_EQ, _B_EQ, _BOUNDS, None)
    A = _std_matrix(std)
    assert A.shape == (std.m, std.n + std.m)
    for r in range(std.m):
        col = A[:, std.n + r]
        assert np.flatnonzero(col).tolist() == [r]
        assert abs(abs(col[r]) - 1.0) < 1e-12
        assert std.lb[std.n + r] == 0.0
        assert std.ub[std.n + r] == 0.0 or std.ub[std.n + r] >= INF


# --------------------------------------------------------------------------
# The round trip the feasibility gate depends on
# --------------------------------------------------------------------------


def _row_logical_form():
    """``x0 + x1 == 1, x0 - x1 <= 0.5`` in the row-logical standard form."""
    A = np.array(
        [
            [1.0, 1.0, 1.0, 0.0],  # equality, logical fixed at [0, 0]
            [1.0, -1.0, 0.0, 1.0],  # inequality, logical free
        ]
    )
    b = np.array([1.0, 0.5])
    x_u = np.array([1.0, 1.0, 0.0, INF])
    return A, b, x_u


def test_decompose_without_col_ub_misreads_a_fixed_logical():
    """Sparsity alone cannot classify under the row-logical layout.

    This is the hazard the ``col_ub`` argument exists for, pinned so a future
    caller that forgets to pass it fails a test rather than a certificate: the
    equality silently becomes a ONE-SIDED inequality and the gate stops checking
    the other direction.
    """
    A, b, x_u = _row_logical_form()
    A_ub, b_ub, A_eq, b_eq = _decompose_eq_slack_form(A, b, 2, 2)
    assert A_eq is None  # the equality was lost
    assert b_ub is not None and np.size(b_ub) == 2


def test_decompose_with_col_ub_recovers_the_equality():
    A, b, x_u = _row_logical_form()
    A_ub, b_ub, A_eq, b_eq = _decompose_eq_slack_form(A, b, 2, 2, x_u)
    np.testing.assert_allclose(A_eq, [[1.0, 1.0]])
    np.testing.assert_allclose(b_eq, [1.0])
    np.testing.assert_allclose(A_ub, [[1.0, -1.0]])
    np.testing.assert_allclose(b_ub, [0.5])


def test_decompose_sparse_and_dense_agree_on_the_row_logical_layout():
    A, b, x_u = _row_logical_form()
    dense = _decompose_eq_slack_form(A, b, 2, 2, x_u)
    sparse = _decompose_eq_slack_form(sp.csr_matrix(A), b, 2, 2, x_u)
    for d, s in zip(dense, sparse):
        if d is None:
            assert s is None
            continue
        np.testing.assert_allclose(np.asarray(d), np.asarray(s.todense() if sp.issparse(s) else s))


def test_decompose_legacy_layout_is_unchanged_by_col_ub():
    """The legacy layout has no fixed logical, so passing ``col_ub`` is a no-op."""
    A = np.array([[1.0, 1.0, 0.0], [1.0, -1.0, 1.0]])  # eq has no logical at all
    b = np.array([1.0, 0.5])
    x_u = np.array([1.0, 1.0, INF])
    without = _decompose_eq_slack_form(A, b, 2, 1)
    with_ub = _decompose_eq_slack_form(A, b, 2, 1, x_u)
    for a, c in zip(without, with_ub):
        np.testing.assert_allclose(np.asarray(a), np.asarray(c))


# --------------------------------------------------------------------------
# Both entry points, same model
# --------------------------------------------------------------------------


@pytest.mark.parametrize("row_logicals", [False, True], indirect=True)
def test_both_entry_points_produce_the_same_standard_form(row_logicals):
    """A model marshaled by the extractor and by ``_marshal_std_form`` agrees.

    The extractor's output is projected back to native ``A_ub``/``A_eq`` form and
    re-marshaled; under either layout the two must land on the same shape, the
    same logical boxes and the same rows -- otherwise a measurement on one path
    says nothing about the other, which is what this consolidation fixes.
    """
    from discopt import Model
    from discopt._relax.problem_classifier import extract_lp_data

    m = Model()
    x = m.continuous("x", 2, lb=0.0, ub=1.0)
    m.subject_to(x[0] + x[1] == 1.0)
    m.subject_to(x[0] - x[1] <= 0.5)
    m.minimize(-x[0] - x[1])

    lp = extract_lp_data(m)
    A_full = lp.A_eq.toarray() if sp.issparse(lp.A_eq) else np.asarray(lp.A_eq)
    n_struct = 2
    n_rows, n_total = A_full.shape
    x_u = np.asarray(lp.x_u, dtype=np.float64)

    # Layout of the extractor's own output.
    assert n_total == n_struct + (n_rows if row_logicals else n_rows - 1)

    A_ub, b_ub, A_eq, b_eq = _decompose_eq_slack_form(
        A_full, np.asarray(lp.b_eq), n_struct, n_total - n_struct, x_u
    )
    assert b_eq is not None and np.size(b_eq) == 1, "the equality must survive the round trip"

    bounds = list(zip(lp.x_l[:n_struct].tolist(), lp.x_u[:n_struct].tolist()))
    std = _marshal_std_form(np.asarray(lp.c[:n_struct]), A_ub, b_ub, A_eq, b_eq, bounds, None)

    if row_logicals:
        # One row per constraint on BOTH paths, and the same fixed-logical count.
        assert std.m == n_rows
        assert std.n + std.m == n_total
        assert sum(logical_is_fixed(u) for u in std.ub[std.n :]) == 1
        assert sum(logical_is_fixed(u) for u in x_u[n_struct:]) == 1
    else:
        # The legacy layouts genuinely differ -- that is the bug this flag fixes.
        # Pin the difference so the OFF arm cannot drift silently.
        assert std.m == n_rows + 1  # the equality was split into two rows
        assert n_total == n_struct + n_rows - 1  # ... but the extractor dropped it
        assert sum(logical_is_fixed(u) for u in std.ub[std.n :]) == 0


# --- the deferring driver's work counters must survive the fallback -----------
#
# When the monolithic Rust MILP driver exhausts its budget without an incumbent
# it returns ``None`` so a sound engine takes over, and the SolveResult the
# caller sees is the *fallback's*. Its own node/iteration counts used to be
# dropped there -- so ``lp/iters`` was absent on exactly the budget-exhausting
# instances an A/B most needs a work metric for. ``_merge_engine_stats`` carries
# them across under a ``milp_driver/`` namespace.


def test_merge_engine_stats_adds_the_drivers_counters():
    from discopt.modeling.core import SolveResult
    from discopt.solver import _merge_engine_stats

    res = SolveResult(status="time_limit", node_count=5183)
    out = _merge_engine_stats(res, {"milp_driver/iters": 106819.0, "milp_driver/nodes": 100021.0})
    assert out.solver_stats == {"milp_driver/iters": 106819.0, "milp_driver/nodes": 100021.0}
    # Instrumentation only -- nothing about the reported solution moves.
    assert out.status == res.status
    assert out.node_count == res.node_count
    assert out.objective == res.objective
    assert out.bound == res.bound


def test_merge_engine_stats_never_overwrites_the_fallbacks_own_counters():
    from discopt.modeling.core import SolveResult
    from discopt.solver import _merge_engine_stats

    res = SolveResult(status="time_limit", solver_stats={"cuts/gomory": 7.0})
    out = _merge_engine_stats(res, {"cuts/gomory": 99.0, "milp_driver/iters": 12.0})
    assert out.solver_stats["cuts/gomory"] == 7.0, "the fallback's own count must win"
    assert out.solver_stats["milp_driver/iters"] == 12.0


@pytest.mark.parametrize("empty", [None, {}])
def test_merge_engine_stats_is_a_no_op_when_the_driver_never_ran(empty):
    from discopt.modeling.core import SolveResult
    from discopt.solver import _merge_engine_stats

    res = SolveResult(status="optimal", objective=1.0)
    assert _merge_engine_stats(res, empty) is res


def test_lp_iters_sums_both_driver_invocations_when_the_reentry_is_adopted(monkeypatch):
    """``lp/iters`` must accumulate wherever ``node_count`` accumulates.

    The #698 re-entry runs the Rust driver a SECOND time from an improved seed
    and folds ``_nodes2`` into ``nodes``. Its ``_iters2`` was returned and
    dropped, so a two-run search reported run 2's nodes against run 1's
    iterations alone -- a per-node cost no run ever had, on exactly the harder
    instances that reach the re-entry. This drives both invocations through a
    stubbed driver and pins the sum.
    """
    import discopt._rust as _rust
    from discopt import Model
    from discopt.solver import _solve_milp_simplex

    # The swap reseed calls into the model; it is orthogonal to the counter and
    # only adds a failure mode to this test.
    monkeypatch.setenv("DISCOPT_MILP_SWAP_RESEED", "0")

    # Binary, not continuous: ``_solve_milp_simplex`` is the pure-MILP driver,
    # and the integer index array is also how the stub below tells a search call
    # from the integer-relaxed root-bound probe.
    m = Model("reentry-iters")
    x = m.binary("x", 2)
    m.subject_to(x[0] + x[1] <= 1.0)
    m.minimize(-x[0] - x[1])

    calls = []

    def fake_driver(*args, **kwargs):
        # Positional 10 is ``n_orig``, positional 9 the integer index array, in
        # every call site; read them rather than assume the model was not lifted.
        int_idx, n_orig = np.asarray(args[9]), int(args[10])
        pt = np.zeros(n_orig, dtype=np.float64)
        pt[0] = 1.0
        if int_idx.size == 0:
            # A THIRD call site solves the integer-relaxed root LP purely to
            # report a root bound, and discards its nodes and iterations on
            # purpose -- it is not search work. Counting it here would assert
            # the opposite of what the driver should do.
            return "optimal", pt, -1.0, -1.0, 0, 7
        calls.append(kwargs.get("time_limit_s"))
        if len(calls) == 1:
            # Run 1 stops with an incumbent but no proof -- the state that opens
            # the re-entry.
            return "feasible", pt, -1.0, -2.0, 700, 1300
        # Run 2 certifies, so the adoption gate takes it.
        return "optimal", pt, -1.0, -1.0, 40, 90

    monkeypatch.setattr(_rust, "solve_milp_csc_py", fake_driver)

    res = _solve_milp_simplex(
        m, time_limit=300.0, gap_tolerance=1e-9, max_nodes=10_000, t_start=time.perf_counter()
    )
    assert res is not None, "the stubbed driver returned a feasible, certified point"
    assert len(calls) == 2, f"the re-entry did not fire; search calls: {calls}"
    assert res.node_count == 700 + 40
    assert res.solver_stats["lp/iters"] == 1300 + 90
    assert res.solver_stats["lp/driver_nodes"] == 700 + 40
