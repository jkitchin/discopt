"""``to_lp`` continues long rows under LP readers' line limits (#1739).

LP readers cap the physical line: CPLEX at 560 characters, lp_solve at 510
(255 in 5.0/5.1). ``to_lp`` wrote each row, the objective and the
``Generals``/``Binaries`` lists as one line, so a dense model produced lines of
100k+ characters. HiGHS reads those anyway, which is why nothing in the repo
noticed.

The oracle is the MPS export of the same model, whose lines are short by
construction: HiGHS must read the wrapped ``.lp`` to exactly the same columns,
bounds, integrality, matrix and objective (Hessian included).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.export import to_lp, to_mps
from discopt.export.lp import _MAX_LINE, _wrap_line

highspy = pytest.importorskip("highspy")

pytestmark = pytest.mark.unit


def _read(text: str, suffix: str):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f"m{suffix}"
        p.write_text(text)
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        assert h.readModel(str(p)) == highspy.HighsStatus.kOk
        lp = h.getLp()
        names = [h.getColName(i)[1] for i in range(lp.num_col_)]
        order = np.argsort(names)
        a = lp.a_matrix_
        dense = np.zeros((lp.num_row_, lp.num_col_))
        for j in range(lp.num_col_):
            for k in range(a.start_[j], a.start_[j + 1]):
                dense[a.index_[k], j] = a.value_[k]
        hess = h.getModel().hessian_
        q = np.zeros((lp.num_col_, lp.num_col_))
        if hess.dim_:
            for j in range(hess.dim_):
                for k in range(hess.start_[j], hess.start_[j + 1]):
                    q[hess.index_[k], j] = hess.value_[k]
        return {
            "names": sorted(names),
            "cost": np.asarray(lp.col_cost_)[order],
            "col_lo": np.asarray(lp.col_lower_)[order],
            "col_hi": np.asarray(lp.col_upper_)[order],
            "integrality": [int(v) for v in np.asarray(lp.integrality_)[order]]
            if len(lp.integrality_)
            else [0] * lp.num_col_,
            "row_lo": np.asarray(lp.row_lower_),
            "row_hi": np.asarray(lp.row_upper_),
            "a": dense[:, order],
            "q": q[np.ix_(order, order)],
            "offset": lp.offset_,
        }


def _dense_model(quadratic: bool):
    n, n_int = 1500, 400
    m = dm.Model("dense_lp")
    x = m.continuous("x", shape=(n,), lb=0.0, ub=4.0)
    k = m.integer("k", shape=(n_int,), lb=0, ub=9)
    b = m.binary("b", shape=(n_int,))
    w = np.linspace(-3.0, 5.0, n)
    m.subject_to(dm.sum(lambda i: w[i] * x[i], over=range(n)) <= 250.0, name="dense")
    m.subject_to(
        dm.sum(lambda i: (i % 5 + 1) * k[i] - 2.5 * b[i], over=range(n_int)) >= -7.0,
        name="mixed",
    )
    obj = dm.sum(lambda i: (1.0 + (i % 7)) * x[i], over=range(n)) + 3.0
    if quadratic:
        obj = obj + dm.sum(lambda i: 0.5 * x[i] * x[i] + x[i] * x[i + 1], over=range(300))
    m.minimize(obj)
    return m


@pytest.mark.parametrize("quadratic", [False, True])
def test_dense_lp_is_wrapped_and_reads_back_identically(quadratic):
    m = _dense_model(quadratic)
    text = to_lp(m)
    longest = max(len(line) for line in text.split("\n"))
    assert longest <= _MAX_LINE, longest
    assert len(text.split("\n")) > 100  # the rows really were continued

    got = _read(text, ".lp")
    want = _read(to_mps(m), ".mps")
    assert got["names"] == want["names"]
    assert got["integrality"] == want["integrality"]
    assert sum(got["integrality"]) == 800
    compared = 0
    for key in ("cost", "col_lo", "col_hi", "row_lo", "row_hi", "a", "q"):
        np.testing.assert_array_equal(got[key], want[key], err_msg=key)
        compared += np.asarray(want[key]).size
    assert got["offset"] == want["offset"] == 3.0
    if quadratic:
        assert np.count_nonzero(got["q"]) > 0
    assert compared > 0


def test_wrap_never_splits_a_term():
    line = "  c1: " + " ".join(f"+ {i + 1.5} x_{i}" for i in range(200)) + " <= 3"
    pieces = _wrap_line(line, width=60)
    assert len(pieces) > 1 and all(len(p) <= 60 for p in pieces)
    assert " ".join(p.strip() for p in pieces) == line.strip()
    # Every continuation starts at a term sign, so `coeff var` stays together.
    assert all(p.strip()[0] in "+-" for p in pieces[1:])


def test_short_lines_are_untouched():
    m = dm.Model("small")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.subject_to(x <= 0.5, name="c")
    m.minimize(x)
    text = to_lp(m)
    assert "  obj:  x\n" in text
    assert "  c:  x <= 0.5\n" in text
