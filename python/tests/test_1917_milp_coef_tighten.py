"""#1917 G3.2 step 1: the Rust MILP big-M coefficient-tightening pass.

``DISCOPT_RUST_MILP_COEF_TIGHTEN`` (default OFF) rewrites integer coefficients of
``<=`` rows (HiGHS ``HPresolve::rowPresolve`` ``strengthenCoefs``; SCIP
``consdataTightenCoefs``). Its contract has two parts:

* the integer projection is exact (Rust unit tests in ``bnb/coef_tighten.rs``);
* the LP relaxation only shrinks.

This file checks the second part on the in-repo MILP fixtures. It takes the exact
standard form ``milp_simplex`` hands the Rust driver, runs the pass through
``milp_coef_tighten_csc_py``, and requires the HiGHS LP optimum of the tightened
rows to be at least that of the original rows, within 1e-9 relative.

The check uses an exact LP rather than the Rust root bound on purpose. The Rust
node bound is the Neumaier-Shcherbina safe bound over the node box, and on a
column left at the Python default box ``±9.999e19`` it multiplies dual rounding
noise by ~1e20. Measured on clay0203_master with the flag ON: the NS root is
−1.04e6, the vertex root −1.4e-12 and the LP optimum 0.0. That weakness belongs to
the NS bound, not to the pass (``docs/dev/g32-tableau-cmir-entry-gate.md``).

Caveat on what fails before the fix: the LP-containment test also passes on the
pass *without* its default-box and precision guards, because none of the rows
these fixtures tighten contains a default-box column. The guards are pinned by
the Rust tests ``default_box_sides_are_unbounded`` and
``cancelled_huge_activity_is_refused_on_precision``, both mutation-checked (they
fail with the guards reverted). ``test_default_box_rows_are_left_alone`` here is
the first of those rows sent through the Python binding; it was not
mutation-checked separately.
"""

from __future__ import annotations

import gzip
import tempfile
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

highspy = pytest.importorskip("highspy")

import discopt._rust as R  # noqa: E402
from discopt.solvers import milp_simplex  # noqa: E402

DATA = Path(__file__).parent / "data"
INF = 1e20
NAMES = ["c", "m", "n", "col_ptr", "row_idx", "vals", "b", "lb", "ub", "integer_cols", "n_struct"]


class _Captured(BaseException):
    pass


def _from_mps(path: Path) -> dict:
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    p = str(path)
    if p.endswith(".gz"):
        with tempfile.NamedTemporaryFile(suffix=".mps", delete=False) as tf:
            tf.write(gzip.open(p).read())
        p = tf.name
    h.readModel(p)
    lp = h.getLp()
    big = 1e30
    A = sp.csc_matrix(
        (lp.a_matrix_.value_, lp.a_matrix_.index_, lp.a_matrix_.start_),
        shape=(lp.num_row_, lp.num_col_),
    ).tocsr()
    rl, ru = np.array(lp.row_lower_), np.array(lp.row_upper_)
    eq = rl == ru
    up = (~eq) & (ru < big)
    lo = (~eq) & (rl > -big)
    cl, cu = np.array(lp.col_lower_), np.array(lp.col_upper_)
    return dict(
        c=np.array(lp.col_cost_),
        A_ub=sp.vstack([A[up], -A[lo]]).tocsr(),
        b_ub=np.r_[ru[up], -rl[lo]],
        A_eq=A[eq].tocsr(),
        b_eq=ru[eq],
        bounds=[(None if a <= -big else a, None if b >= big else b) for a, b in zip(cl, cu)],
        integrality=np.array(
            [1 if t == highspy.HighsVarType.kInteger else 0 for t in lp.integrality_]
        ),
    )


def _from_master(path: Path) -> dict:
    z = np.load(path)

    def mk(p):
        return sp.csr_matrix(
            (z[p + "_data"], z[p + "_indices"], z[p + "_indptr"]), shape=tuple(z[p + "_shape"])
        )

    return dict(
        c=z["c"],
        A_ub=mk("A_ub"),
        b_ub=z["b_ub"],
        A_eq=mk("A_eq"),
        b_eq=z["b_eq"],
        bounds=[
            (None if not np.isfinite(a) else a, None if not np.isfinite(b) else b)
            for a, b in zip(z["lb"], z["ub"])
        ],
        integrality=z["integrality"],
    )


# Fixtures where the pass fires on the box as ``milp_simplex`` hands it over. In the
# driver the pass runs after Rust FBBT, so it also fires on rsyn0830m_master0; on the
# raw box every rsyn0830m row has an unbounded side (460 skipped). i1183,
# highs_presolve_cycle_1671 and issue1229 do not fire on the raw box either, and a
# fixture where nothing changes would compare an LP with itself.
FIXTURES = [
    ("clay0203_master", _from_mps, DATA / "issue2069" / "clay0203_master.mps.gz"),
    ("fac2_master0", _from_master, DATA / "oa_masters" / "fac2_master0.npz"),
    ("tls2_master0", _from_master, DATA / "oa_masters" / "tls2_master0.npz"),
]


def _capture(monkeypatch, kw: dict) -> dict:
    """The standard form ``milp_simplex`` passes to the Rust MILP driver."""
    cap: dict = {}

    def grab(*a, **k):
        d = dict(zip(NAMES, a))
        d.update(k)
        cap.update({key: d[key] for key in NAMES})
        raise _Captured()

    monkeypatch.setattr(R, "solve_milp_csc_py", grab)
    with pytest.raises(_Captured):
        milp_simplex.solve_milp(time_limit=60.0, **kw)
    monkeypatch.undo()
    return cap


def _lp_value(d: dict, vals: np.ndarray, b: np.ndarray) -> float:
    n, m = d["n"], d["m"]
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = n, m
    lp.col_cost_ = np.asarray(d["c"], float)
    lb, ub = np.asarray(d["lb"], float), np.asarray(d["ub"], float)
    lp.col_lower_ = np.where(lb <= -INF, -highspy.kHighsInf, lb)
    lp.col_upper_ = np.where(ub >= INF, highspy.kHighsInf, ub)
    lp.row_lower_ = lp.row_upper_ = np.asarray(b, float)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = np.asarray(d["col_ptr"], np.int64)
    lp.a_matrix_.index_ = np.asarray(d["row_idx"], np.int64)
    lp.a_matrix_.value_ = np.asarray(vals, float)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.passModel(lp)
    h.run()
    status = h.modelStatusToString(h.getModelStatus())
    assert status == "Optimal", status
    return h.getInfo().objective_function_value


def _tighten(d: dict):
    return R.milp_coef_tighten_csc_py(
        int(d["m"]),
        int(d["n"]),
        int(d["n_struct"]),
        np.asarray(d["col_ptr"], np.int64),
        np.asarray(d["row_idx"], np.int64),
        np.asarray(d["vals"], float),
        np.asarray(d["b"], float),
        np.asarray(d["lb"], float),
        np.asarray(d["ub"], float),
        np.asarray(d["integer_cols"], np.int64),
    )


@pytest.mark.parametrize("name,loader,path", FIXTURES, ids=[f[0] for f in FIXTURES])
def test_tightened_lp_relaxation_is_contained_in_the_original(monkeypatch, name, loader, path):
    d = _capture(monkeypatch, loader(path))
    vals, b, rows, coefs, skip_unb, skip_prec = _tighten(d)
    # The CSC structure is unchanged; only values move.
    assert len(vals) == len(d["vals"])
    assert coefs > 0, f"{name}: the pass changed nothing, so the comparison is vacuous"
    before = _lp_value(d, np.asarray(d["vals"], float), np.asarray(d["b"], float))
    after = _lp_value(d, vals, b)
    assert after >= before - 1e-9 * max(1.0, abs(before)), (name, before, after)


def test_the_fixtures_exercise_the_pass(monkeypatch):
    """Anti-vacuity (CLAUDE.md §6): every fixture fires, and clay0203 has
    default-box columns that the guard has to keep out of the activity."""
    fired = 0
    for name, loader, path in FIXTURES:
        d = _capture(monkeypatch, loader(path))
        _, _, rows, coefs, _, _ = _tighten(d)
        fired += coefs > 0
        if name == "clay0203_master":
            ub = np.asarray(d["ub"], float)[: d["n_struct"]]
            lb = np.asarray(d["lb"], float)[: d["n_struct"]]
            assert ((np.abs(lb) >= 1e19) & (np.abs(lb) < INF)).any() or (
                (np.abs(ub) >= 1e19) & (np.abs(ub) < INF)
            ).any()
    assert fired == len(FIXTURES)


def test_default_box_rows_are_left_alone():
    """``x_a - x_b + 100 y <= 10`` with ``x_a`` in ``[0, 9.999e19]`` and ``x_b``
    fixed at ``9.999e19``: the float max activity is exactly 100. Without the
    default-box guard the pass rewrites ``100 y`` from that cancelled activity."""
    big = 9.999e19
    A = sp.csc_matrix(np.array([[1.0, -1.0, 100.0, 1.0]]))
    lb = np.array([0.0, big, 0.0, 0.0])
    ub = np.array([big, big, 1.0, INF])
    vals, b, rows, coefs, skip_unb, skip_prec = R.milp_coef_tighten_csc_py(
        1,
        4,
        3,
        A.indptr.astype(np.int64),
        A.indices.astype(np.int64),
        A.data,
        np.array([10.0]),
        lb,
        ub,
        np.array([2], np.int64),
    )
    assert (coefs, skip_unb) == (0, 1)
    np.testing.assert_array_equal(vals, A.data)
    assert b[0] == 10.0
