"""#2225: the MIPcc27 entry decides infeasibility over the checker's box.

The Rust engine proves infeasibility exactly, so it can drop elements down to a set
that is infeasible only by a sliver -- on MIPcc27 dell, 1.7e-7 on one row -- which
the competition checker (HiGHS at 1e-5) finds feasible; the entry then had to write
the whole model. ``mipcc.checker_box`` widens every finite side by 1.25 x 1e-5; the
engine decides infeasibility over it and certifies over the model's own sides
(``witness_box``), so an answer is infeasible under the checker's rule.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
from discopt import _iis_milp, mipcc
from discopt._iis_milp import _elimination_certificate, _verify_integer_certificate

INF = 1e20


DELL = Path(__file__).parent / "data" / "iis_2225_dell_sliver_subsystem.npz"


def _dell():
    """MIPcc27 dell and the 929-element subsystem the Rust engine returned when it
    decided on the model's own sides (the run the checker-settings re-solve then
    rejected): ``(data, engine arrays pieces, active sets)``."""
    z = np.load(DELL)
    A = sp.csc_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])).tocsr()
    data = {k: z[k] for k in ("row_lo", "row_hi", "col_lb", "col_ub", "is_int")}
    data["A"] = A
    cons = set(np.flatnonzero(z["rows"]).tolist())
    ints = set(np.flatnonzero(z["held"]).tolist())
    bounds = {(int(j), "lower") for j in np.flatnonzero(z["lb_kept"] & (z["col_lb"] > -INF))}
    bounds |= {(int(j), "upper") for j in np.flatnonzero(z["ub_kept"] & (z["col_ub"] < INF))}
    return data, cons, bounds, ints


def _engine(data, sides, witness_box=None):
    rlo, rhi, lb, ub = sides
    m = rlo.size
    return _iis_milp._Engine(
        data["A"],
        rlo,
        rhi,
        lb,
        ub,
        data["is_int"].astype(bool),
        {i: np.array([i]) for i in range(m)},
        30.0,
        engine="rust",
        witness_box=witness_box,
    )


def test_dell_sliver_set_is_infeasible_on_the_model_sides():
    """The premise: on the model's own sides the set is infeasible (exactly: one row
    is short by 1.7e-7, 7.5e-5 in the units of its integer sum)."""
    data, cons, bounds, ints = _dell()
    own = (data["row_lo"], data["row_hi"], data["col_lb"], data["col_ub"])
    assert _engine(data, own).probe(cons, bounds, ints)[0] is True


@pytest.mark.requires_highs
def test_the_checker_settings_find_the_sliver_set_feasible():
    """...and the checker's own solve accepts a point in it, so as an answer it fails."""
    pytest.importorskip("highspy")
    data, cons, bounds, ints = _dell()

    res = SimpleNamespace(constraints=cons, integrality=ints, bounds=sorted(bounds))
    assert mipcc.checker_rejects(data, res, 60.0)


def test_over_the_checker_box_the_sliver_set_is_not_infeasible():
    """Deciding over the checker box (certifying over the model's sides), the engine
    no longer calls the set infeasible, so the filter keeps the element whose removal
    led there. Before #2225's checker box this set was the answer."""
    data, cons, bounds, ints = _dell()
    own = (data["row_lo"], data["row_hi"], data["col_lb"], data["col_ub"])
    eng = _engine(data, mipcc.checker_box(data), witness_box=own)
    assert eng.probe(cons, bounds, ints)[0] is not True


def test_checker_box_moves_every_finite_side_out():
    data = {
        "row_lo": np.array([-INF, 2.0]),
        "row_hi": np.array([1.0, INF]),
        "col_lb": np.array([0.0, -INF]),
        "col_ub": np.array([INF, 5.0]),
    }
    rlo, rhi, lb, ub = mipcc.checker_box(data)
    w = mipcc._WIDEN_SAFETY * mipcc.CHECK_TOL
    assert rlo[0] == -INF and rhi[1] == INF and ub[0] == INF and lb[1] == -INF
    assert rlo[1] == 2.0 - w and rhi[0] == 1.0 + w
    assert lb[0] == -w and ub[1] == 5.0 + w


def test_certificates_hold_over_widened_ranges():
    """A widened equality is a range; the integer certificate still holds exactly
    when no multiple of the gcd fits the combined range, and fails when one does."""
    # x0 + x1 + y = 1, x1 + x2 - y = 1, x0 + x2 = 1, each widened by w.
    A = sp.csr_matrix(np.array([[1, 1, 0, 1], [0, 1, 1, -1], [1, 0, 1, 0]], dtype=float))
    held = np.array([True, True, True, False])
    w = 1.25e-5
    lo, hi = np.ones(3) - w, np.ones(3) + w
    lam = _elimination_certificate(A, lo, hi, held)
    assert lam is not None and _verify_integer_certificate(A, lo, hi, held, lam)
    # Ranges wide enough to hold the even value 2 x1 = 2 or 0: no certificate.
    lo2, hi2 = np.ones(3) - 0.6, np.ones(3) + 0.6
    good = {0: Fraction(1), 1: Fraction(1), 2: Fraction(-1)}
    assert not _verify_integer_certificate(A, lo2, hi2, held, good)
