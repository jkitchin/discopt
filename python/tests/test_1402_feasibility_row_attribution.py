"""``check_feasibility`` must check every constraint ROW, not every object (#1402).

#1404 closed the two mechanisms the issue names — a swallowed evaluator exception
and NaN failing every strict comparison. A third of the same class survived it,
because it raises nothing and produces no NaN: the verifier walked
``model._constraints`` with **one row index per** :class:`Constraint` **object**
while the evaluator emits **one row per flat element**.

* An array-valued body is ONE ``Constraint`` and MANY rows — ``x <= 1`` on a
  3-vector is one object and three rows — so rows 1..k-1 went unread, and every
  constraint behind the first vector one read the *wrong* row.
* The evaluator's row set is ``model._constraints`` **plus**
  ``model._builder_linear_constraints()`` (#840), so builder-resident rows were
  never examined at all.

Both are #908's diagnosis, which built ``evaluator.constraint_row_map()`` for
exactly this and migrated the two *in-solver* incumbent verifiers onto it;
``check_feasibility`` — the verifier the *benchmark* correctness gate calls at
``discopt_benchmarks/tests/test_correctness.py:348`` — was not migrated with them.

Every violated arm here is violated **by 4.0**, far outside any tolerance, so a
``True`` is unambiguous evidence the row was never read. The control arms are not
optional: without them every assertion here would pass against a function
hardwired to ``False``, which is the CLAUDE.md §6 failure this fix is about.
"""

from __future__ import annotations

import numpy as np
import pytest
from discopt import modeling as dm
from discopt._tape_nlp_evaluator import make_evaluator
from discopt.warm_start import check_feasibility


def _vector_model(sense: str = "<="):
    """``x <sense> 1`` on a 3-vector — ONE Constraint object, THREE evaluator rows."""
    m = dm.Model(f"f1402_vec_{'le' if sense == '<=' else 'ge' if sense == '>=' else 'eq'}")
    x = m.continuous("x", shape=(3,), lb=-10.0, ub=10.0)
    if sense == "<=":
        m.subject_to(x <= 1.0)
    elif sense == ">=":
        m.subject_to(x >= 1.0)
    else:
        m.subject_to(x == 1.0)
    m.minimize(dm.sum(x))
    return m


def _builder_model():
    """Two rows added through ``add_linear_constraints`` — the #840 builder store.

    ``model._constraints`` is EMPTY for this model; the rows live only in
    ``model._builder_linear_constraints()``, and the evaluator emits both.
    """
    m = dm.Model("f1402_builder")
    w = m.continuous("w", shape=(2,), lb=-10.0, ub=10.0)
    m.add_linear_constraints(np.eye(2), w, "<=", np.array([1.0, 1.0]))
    m.minimize(dm.sum(w))
    return m


# ── the defect: rows that were never read ────────────────────────────────────


def test_a_violated_row_of_a_vector_constraint_is_not_skipped():
    """Rows 1 and 2 are violated by 4.0; only row 0 used to be read."""
    ok, viols = check_feasibility(_vector_model(), np.array([0.5, 5.0, 5.0]))
    assert not ok, (
        "a point violating rows 1 and 2 of a size-3 vector constraint BY 4.0 was "
        "reported FEASIBLE: the walk read only row 0 (which is satisfied) because "
        "it advanced one index per Constraint object"
    )
    assert len(viols) == 2, f"expected both violated rows to be named, got {viols}"


def test_a_vector_constraint_does_not_desynchronise_the_rows_behind_it():
    """The row the verifier reads for a later constraint must be *its* row."""
    m = dm.Model("f1402_desync")
    y = m.continuous("y", shape=(3,), lb=-10.0, ub=10.0)
    z = m.continuous("z", lb=-10.0, ub=10.0)
    m.subject_to(y <= 9.0)  # rows 0-2, satisfied below
    m.subject_to(z <= 1.0)  # row 3, VIOLATED below
    m.minimize(z)

    ok, viols = check_feasibility(m, np.array([0.0, 0.0, 0.0, 5.0]))
    assert not ok, (
        "`z <= 1` violated by 4.0 was reported FEASIBLE: it is evaluator row 3, "
        "but the per-object walk read row 1 — a satisfied row belonging to the "
        "vector constraint ahead of it"
    )
    assert len(viols) == 1, f"only `z <= 1` is violated here, got {viols}"


def test_builder_resident_rows_are_checked():
    """#840 rows are not in ``model._constraints`` and were never examined."""
    m = _builder_model()
    assert not [c for c in m._constraints], (
        "the fixture must keep its rows in the BUILDER store — if they moved into "
        "model._constraints this test no longer exercises the #840 hole"
    )
    assert len(m._builder_linear_constraints()) == 2

    ok, viols = check_feasibility(m, np.array([5.0, 5.0]))
    assert not ok, (
        "two builder-resident rows, each violated by 4.0, were reported FEASIBLE: "
        "the walk over model._constraints never saw them"
    )
    assert len(viols) == 2, viols


@pytest.mark.parametrize(
    ("sense", "point"),
    [
        pytest.param("<=", np.array([0.0, 5.0, 0.0]), id="le-row1"),
        pytest.param(">=", np.array([2.0, -3.0, 2.0]), id="ge-row1"),
        pytest.param("==", np.array([1.0, 5.0, 1.0]), id="eq-row1"),
    ],
)
def test_every_sense_branch_reads_the_interior_row(sense, point):
    """All three sense branches shared the one broken index, so pin all three."""
    ok, viols = check_feasibility(_vector_model(sense), point)
    assert not ok, f"row 1 of a size-3 `{sense}` constraint, violated by 4.0, was reported FEASIBLE"
    assert any("[1]" in v for v in viols), (
        f"the violation must name WHICH row of the vector constraint failed: {viols}"
    )


# ── controls: fail-closed must not mean always-closed ────────────────────────


def test_control_a_feasible_vector_point_is_still_accepted():
    ok, viols = check_feasibility(_vector_model(), np.array([0.5, 0.5, 0.5]))
    assert ok, f"a feasible point was rejected, so the fix over-closed: {viols}"
    assert viols == []


def test_control_a_feasible_builder_point_is_still_accepted():
    ok, viols = check_feasibility(_builder_model(), np.array([0.5, 0.5]))
    assert ok, f"a feasible point was rejected, so the fix over-closed: {viols}"
    assert viols == []


def test_control_a_scalar_row_still_reports_exactly_as_before():
    """The shape the old walk DID handle keeps its bare, un-indexed name."""
    m = dm.Model("f1402_scalar")
    a = m.continuous("a", lb=-10.0, ub=10.0)
    m.subject_to(a <= 1.0)
    m.minimize(a)
    ok, viols = check_feasibility(m, np.array([5.0]))
    assert not ok
    assert viols == ["Constraint 'constraint_0': value 4 > 0 (sense <=)"], viols


# ── the guards, exercised directly ───────────────────────────────────────────


def test_a_row_the_map_does_not_cover_fails_closed(monkeypatch):
    """A future drift between the map and the row stream must refuse, not pass.

    Injected, because the map and the row stream are built from the same
    ``_source_constraints`` / ``_constraint_flat_sizes`` and so cannot drift
    today — which is exactly why the guard would otherwise never be executed and
    could rot unnoticed (CLAUDE.md §6).
    """
    m = _vector_model()
    ev = make_evaluator(m)
    full = ev.constraint_row_map()
    assert full and full[0][1] - full[0][0] == 3, full

    # Claim only row 0, as a per-object walk would.
    monkeypatch.setattr(type(ev), "constraint_row_map", lambda self: [(0, 1, full[0][2])])
    monkeypatch.setattr("discopt._tape_nlp_evaluator.make_evaluator", lambda _m: ev)

    ok, viols = check_feasibility(m, np.array([0.5, 0.5, 0.5]))
    assert not ok, "a map leaving 2 of 3 rows unclaimed was reported as a clean check"
    assert any("never checked" in v for v in viols), viols


def test_an_unrecognised_sense_fails_closed(monkeypatch):
    """A sense matching none of the three branches used to fall through silently."""
    m = _vector_model()
    ev = make_evaluator(m)
    con = ev.constraint_row_map()[0][2]
    monkeypatch.setattr(con, "sense", "≈")
    monkeypatch.setattr("discopt._tape_nlp_evaluator.make_evaluator", lambda _m: ev)

    ok, viols = check_feasibility(m, np.array([0.5, 0.5, 0.5]))
    assert not ok, "a row with an unrecognised sense was treated as satisfied"
    assert len(viols) == 3, f"every row of the constraint must refuse, got {viols}"
    assert all("unrecognised sense" in v for v in viols), viols


def test_a_short_row_stream_fails_closed(monkeypatch):
    """Fewer values than rows must refuse rather than check the prefix."""
    m = _vector_model()
    ev = make_evaluator(m)
    monkeypatch.setattr(type(ev), "evaluate_constraints", lambda self, x: np.zeros(1))
    monkeypatch.setattr("discopt._tape_nlp_evaluator.make_evaluator", lambda _m: ev)

    ok, viols = check_feasibility(m, np.array([0.5, 0.5, 0.5]))
    assert not ok, "a 1-value stream for a 3-row model was accepted as verified"
    assert any("NOT verified" in v for v in viols), viols


def test_the_verifier_grades_every_arm():
    """CLAUDE.md §6: count verdicts directly rather than trusting collection."""
    cases = [
        (_vector_model(), np.array([0.5, 5.0, 5.0]), False),
        (_vector_model(">="), np.array([2.0, -3.0, 2.0]), False),
        (_vector_model("=="), np.array([1.0, 5.0, 1.0]), False),
        (_builder_model(), np.array([5.0, 5.0]), False),
        (_vector_model(), np.array([0.5, 0.5, 0.5]), True),
        (_builder_model(), np.array([0.5, 0.5]), True),
    ]
    graded = 0
    for model, point, expected in cases:
        ok, _ = check_feasibility(model, point)
        assert ok is expected, f"{model.name} at {point!r}: expected ok={expected}, got {ok}"
        graded += 1
    assert graded == 6, f"graded {graded} verdicts, expected 6"
