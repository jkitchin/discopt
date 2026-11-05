"""Per-rule skip lists in the nonlinear tightening scan (bound-neutral).

Every rule in ``nonlinear_bound_tightening`` used to walk every row of the model on
every call and re-decide, through a cache lookup or a pattern match, that most rows
are not its shape. That verdict depends only on the row's expression, never on the
box. On GDPlib ``modprodnet`` under hull (11.5k rows) those re-decisions were ~17 s
of a 26 s nonlinear pass. ``_rule_rows``/``_reject_row`` let a rule skip its
structural misses on later scans.

Pinned here (CLAUDE.md 5, bound-neutral; CLAUDE.md 6, prove the probe fired):

* the skipping scan gives **bit-identical** boxes, verdicts and stats to the full
  scan, on corpus models and random sub-boxes with a warm cache;
* the skip lists are real: rows are rejected, and a later scan visits fewer rows;
* a row list or a body changed in place drops the skip lists, so a row that now
  matches is scanned again.
"""

from __future__ import annotations

from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import nonlinear_bound_tightening as nbt
from discopt._relax.model_utils import flat_variable_bounds
from discopt.modeling.core import Model, from_nl

pytestmark = pytest.mark.unit

_CORPUS = Path(__file__).parent / "data" / "minlplib_nl"
# Small, fast corpus files covering squares, sqrt/exp/log terms, bilinear rows,
# reciprocals and integer quadratics.
_FILES = [
    "ex1224.nl",
    "ex1225.nl",
    "ex14_1_7.nl",
    "haverly.nl",
    "launch.nl",
    "prolog.nl",
    "nvs05.nl",
]


def _full_scan(monkeypatch):
    monkeypatch.setattr(nbt, "_rule_rows", lambda m, name, md, d: nbt._rows_until(m, d))
    monkeypatch.setattr(nbt, "_reject_row", lambda *a: None)


def _run(model, lb, ub):
    try:
        lo, hi, st = nbt.tighten_nonlinear_bounds(model, lb.copy(), ub.copy())
    except nbt.NonlinearBoundTighteningInfeasible as exc:
        return ("infeasible", str(exc))
    return ("ok", lo, hi, st.applied_rules, st.n_tightened)


def _boxes(model, rng, n=5):
    lb0, ub0 = flat_variable_bounds(model)
    lo = np.where(np.isfinite(lb0), lb0, -1e3)
    hi = np.where(np.isfinite(ub0), ub0, 1e3)
    out = [(lb0, ub0)]
    for _ in range(n):
        u = rng.random((2, lb0.size))
        a = lo + (hi - lo) * u.min(axis=0)
        b = lo + (hi - lo) * u.max(axis=0)
        out.append(
            (
                np.where(rng.random(lb0.size) < 0.5, lb0, a),
                np.where(rng.random(lb0.size) < 0.5, ub0, b),
            )
        )
    return out


def _same(a, b):
    assert a[0] == b[0]
    if a[0] == "infeasible":
        assert a[1] == b[1]
        return
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[2], b[2])
    assert a[3:] == b[3:]


def test_skipping_scan_is_bit_identical_to_the_full_scan(monkeypatch):
    rng = np.random.default_rng(0)
    warm = {f: from_nl(str(_CORPUS / f)) for f in _FILES}
    boxes = {f: _boxes(m, rng) for f, m in warm.items()}
    skipping = {f: [_run(warm[f], lb, ub) for lb, ub in boxes[f]] for f in _FILES}
    rejected = sum(len(e[0]) for m in warm.values() for e in m._nl_struct_cache["rowskip"].values())

    _full_scan(monkeypatch)
    compared = 0
    for f in _FILES:
        cold = from_nl(str(_CORPUS / f))
        for (lb, ub), got in zip(boxes[f], skipping[f]):
            _same(got, _run(cold, lb, ub))
            compared += 1
    assert compared == len(_FILES) * 6
    assert rejected > 0, "no row was ever rejected -- the differential compared a scan to itself"


def _sos_model():
    m = Model("rowskip")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    y = m.continuous("y", lb=-10.0, ub=10.0)
    m.subject_to(x**2 + y**2 <= 4.0)  # sum of squares: tightens x, y to [-2, 2]
    m.subject_to(x + 2.0 * y <= 30.0)  # linear: every pattern rule rejects it
    m.subject_to(dm.exp(x) - y <= 50.0)
    m.minimize(x + y)
    return m


def test_a_later_scan_visits_fewer_rows(monkeypatch):
    m = _sos_model()
    lb, ub = flat_variable_bounds(m)
    seen = {"n": 0}
    real = nbt._cached_flat_terms

    def counting(model, expr):
        seen["n"] += 1
        return real(model, expr)

    monkeypatch.setattr(nbt, "_cached_flat_terms", counting)
    rule = nbt.SumOfSquaresUpperBoundRule()
    md = nbt._cached_flat_metadata(m)
    lo1, hi1 = rule.tighten(m, lb, ub, md)
    first = seen["n"]
    seen["n"] = 0
    lo2, hi2 = rule.tighten(m, lb, ub, md)
    assert first == 3
    assert seen["n"] == 1, "rejected rows were scanned again"
    np.testing.assert_array_equal(lo1, lo2)
    np.testing.assert_array_equal(hi1, hi2)
    np.testing.assert_allclose(hi1, [2.0, 2.0], rtol=0, atol=1e-12)  # outward-rounded


def test_a_body_replaced_in_place_is_scanned_again():
    m = _sos_model()
    x, y = m._variables
    lb, ub = flat_variable_bounds(m)
    rule = nbt.SumOfSquaresUpperBoundRule()
    md = nbt._cached_flat_metadata(m)
    rule.tighten(m, lb, ub, md)
    linear = m._constraints[1]
    assert id(linear) in m._nl_struct_cache["rowskip"][rule.name][0]

    # Same object, new body -- what the #2090 exact pass does. It is now a
    # sum of squares bounding y to [-1, 1]; a stale skip list would miss it.
    linear.body = y**2 - 1.0
    _, hi = rule.tighten(m, lb, ub, md)
    assert hi[1] == pytest.approx(1.0, abs=1e-12)


def test_a_row_list_changed_in_place_is_scanned_again():
    m = _sos_model()
    x, y = m._variables
    lb, ub = flat_variable_bounds(m)
    rule = nbt.SumOfSquaresUpperBoundRule()
    md = nbt._cached_flat_metadata(m)
    rule.tighten(m, lb, ub, md)
    replacement = Model("donor")
    replacement.subject_to(x**2 <= 0.25)
    m._constraints[1] = replacement._constraints[0]  # same list, same length
    _, hi = rule.tighten(m, lb, ub, md)
    assert hi[0] == pytest.approx(0.5, abs=1e-12)


def test_a_foreign_metadata_scans_every_row():
    m = _sos_model()
    lb, ub = flat_variable_bounds(m)
    rule = nbt.SumOfSquaresUpperBoundRule()
    rule.tighten(m, lb, ub, nbt._cached_flat_metadata(m))
    own = nbt.build_flat_variable_metadata(m)
    rows = list(nbt._rule_rows(m, rule.name, own, None))
    assert len(rows) == len(m._constraints)
