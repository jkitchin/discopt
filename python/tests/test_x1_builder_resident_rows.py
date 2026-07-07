"""Regression tests for X-1 — the builder-resident-rows blind spot (issue #413).

Fast-API constraint rows (``Model.constraint(...)`` linear fast path,
``Model.add_linear_constraints``) live in ``model._builder_linear_blocks`` / the
Rust builder, NOT in ``model._constraints``. Consumers that read only
``_constraints`` therefore see a strict *subset* of the model — a silent-wrong
hazard (false optimum / false certificate / empty export). These tests pin the
whole failure *class* by exercising every routed consumer directly:

* **GP-1** — the auto-GP path recognised a GP-shaped model and solved the log-space
  reformulation WITHOUT the fast-path rows → wrong certified optimum.
* **VAL-1** — the validation examiner certified a point that violates a fast-path row.
* **EX-2** — MPS/LP/GAMS export of a fast-API model emitted an empty/zero model.
* **M12** — ``num_constraints`` / ``summary`` under-reported fast-path rows.

Each assertion below fails on pre-fix ``main`` and passes after routing the
consumers through ``discopt.export._common.iter_builder_linear_rows`` /
``Model._has_builder_only_rows``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp
from discopt import Model
from discopt.gp import classify_gp
from discopt.validation.examiner import examine


def _fast_linear_model(name: str = "x1"):
    """min x0+x1 s.t. x0+x1 >= 3 (0<=x<=10), the row via the fast-API builder."""
    m = Model(name)
    x = m.continuous("x", shape=(2,), lb=0.0, ub=10.0)
    m.minimize(x[0] + x[1])
    m.add_linear_constraints(sp.csr_matrix(np.array([[1.0, 1.0]])), x, ">=", np.array([3.0]))
    return m


# ── M12: introspection counts the fast-path rows ────────────────────────────


@pytest.mark.smoke
def test_m12_num_constraints_counts_builder_rows():
    m = _fast_linear_model()
    assert m.num_constraints == 1
    assert "Constraints: 1" in m.summary()
    assert m._has_builder_only_rows() is True


@pytest.mark.smoke
def test_m12_multiple_blocks_and_multi_row():
    m = Model("m12b")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=10.0)
    m.minimize(x[0])
    m.add_linear_constraints(sp.csr_matrix(np.eye(3)), x, "<=", np.array([5.0, 5.0, 5.0]))
    m.add_linear_constraints(sp.csr_matrix(np.array([[1.0, 1.0, 1.0]])), x, ">=", np.array([2.0]))
    assert m.num_constraints == 4  # 3 rows + 1 row


@pytest.mark.smoke
def test_has_builder_only_rows_false_for_expression_model():
    m = Model("expr")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    m.subject_to(x >= 3, name="c")
    assert m._has_builder_only_rows() is False
    assert m.num_constraints == 1


# ── EX-2: exporters emit the fast-path rows and round-trip ───────────────────


@pytest.mark.smoke
def test_ex2_mps_carries_builder_row():
    mps = _fast_linear_model("ex2mps").to_mps()
    # A ">=" row must appear (G) with the RHS.
    assert " G  " in mps
    assert "RHS" in mps
    # Both variable coefficients must be present in COLUMNS.
    assert mps.count("  x_0  ") >= 2  # OBJ + the constraint row
    assert mps.count("  x_1  ") >= 2


@pytest.mark.smoke
def test_ex2_lp_carries_builder_row():
    lp = _fast_linear_model("ex2lp").to_lp()
    subject_to = lp.split("Subject To")[1].split("Bounds")[0]
    assert ">= 3" in subject_to
    assert "x_0" in subject_to and "x_1" in subject_to


@pytest.mark.smoke
def test_ex2_gams_carries_builder_row():
    gms = _fast_linear_model("ex2gams").to_gams()
    assert "=g= 3" in gms
    # The row must reference both variable elements.
    assert "x('1')" in gms and "x('2')" in gms


@pytest.mark.smoke
def test_ex2_mixed_expression_and_builder_rows():
    m = Model("mix")
    s = m.set("s", ["a", "b"])
    x = m.continuous("x", over=s, lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.minimize(x["a"] + x["b"] + y)
    m.constraint(s, lambda i: x[i] >= 2, "floor")  # fast path -> builder rows
    m.subject_to(y >= 5, name="ycon")  # expression row
    assert m.num_constraints == 3
    mps = m.to_mps()
    # 2 builder rows (floor_0/floor_1) + 1 expression row (ycon).
    row_lines = [ln for ln in mps.splitlines() if ln[:3] in (" L ", " G ", " E ")]
    assert len(row_lines) == 3


@pytest.mark.smoke
def test_ex2_mps_round_trips_true_optimum():
    """The exported MPS must solve to the TRUE optimum, not the empty-model 0."""
    highspy = pytest.importorskip("highspy")
    import os
    import tempfile

    mps = _fast_linear_model("ex2rt").to_mps()
    with tempfile.NamedTemporaryFile("w", suffix=".mps", delete=False) as f:
        f.write(mps)
        path = f.name
    try:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.readModel(path)
        h.run()
        assert abs(h.getObjectiveValue() - 3.0) < 1e-6
    finally:
        os.unlink(path)


# ── EX-2 (objective): builder-resident objective is recovered, not `0` ───────


def _builder_objective_model(name: str = "objx"):
    """min x0 + 2 x1 s.t. x0 + x1 >= 3, objective set via `add_linear_objective`."""
    m = Model(name)
    x = m.continuous("x", shape=(2,), lb=0.0, ub=10.0)
    m.add_linear_objective(np.array([1.0, 2.0]), x, sense="minimize")
    m.add_linear_constraints(sp.csr_matrix(np.array([[1.0, 1.0]])), x, ">=", np.array([3.0]))
    return m


@pytest.mark.smoke
def test_ex2_lp_recovers_builder_objective():
    lp = _builder_objective_model("objlp").to_lp()
    obj_line = next(ln for ln in lp.splitlines() if ln.strip().startswith("obj:"))
    assert "x_0" in obj_line and "x_1" in obj_line
    assert obj_line.strip() != "obj: 0"


@pytest.mark.smoke
def test_ex2_mps_recovers_builder_objective():
    mps = _builder_objective_model("objmps").to_mps()
    assert "  x_0  OBJ  1" in mps
    assert "  x_1  OBJ  2" in mps


@pytest.mark.smoke
def test_ex2_gams_recovers_builder_objective():
    gms = _builder_objective_model("objgms").to_gams()
    obj_def = next(ln for ln in gms.splitlines() if ln.startswith("obj_eq.."))
    assert "=e= 0;" not in obj_def
    assert "x('1')" in obj_def and "x('2')" in obj_def


@pytest.mark.smoke
def test_ex2_mps_objective_round_trips():
    highspy = pytest.importorskip("highspy")
    import os
    import tempfile

    mps = _builder_objective_model("objrt").to_mps()
    with tempfile.NamedTemporaryFile("w", suffix=".mps", delete=False) as f:
        f.write(mps)
        path = f.name
    try:
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.readModel(path)
        h.run()
        # min x0 + 2 x1 s.t. x0+x1 >= 3, x >= 0 -> x0 = 3, obj = 3.0.
        assert abs(h.getObjectiveValue() - 3.0) < 1e-6
    finally:
        os.unlink(path)


# ── VAL-1: examiner rejects a point violating a fast-path row ────────────────


@pytest.mark.smoke
def test_val1_examiner_rejects_violating_builder_row():
    m = Model("val1")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=10.0)
    m.minimize(x[0] + x[1])
    # x[0] >= 3 lives only in the builder.
    m.add_linear_constraints(sp.csr_matrix(np.array([[1.0, 0.0]])), x, ">=", np.array([3.0]))
    # [0, 0] grossly violates x[0] >= 3.
    rep = examine(SimpleNamespace(x={"x": np.array([0.0, 0.0])}, objective=0.0), m)
    assert rep.passed is False
    builder_checks = [c for c in rep.checks if "builder" in c.name]
    assert builder_checks and builder_checks[0].passed is False


@pytest.mark.smoke
def test_val1_examiner_accepts_feasible_builder_row():
    m = Model("val1ok")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=10.0)
    m.minimize(x[0] + x[1])
    m.add_linear_constraints(sp.csr_matrix(np.array([[1.0, 0.0]])), x, ">=", np.array([3.0]))
    rep = examine(SimpleNamespace(x={"x": np.array([3.0, 0.0])}, objective=3.0), m)
    builder_checks = [c for c in rep.checks if "builder" in c.name]
    assert builder_checks and builder_checks[0].passed is True


@pytest.mark.smoke
def test_val1_no_builder_check_for_expression_model():
    """Expression-only models are untouched: no builder-row check is added."""
    m = Model("val1expr")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    m.subject_to(x >= 3, name="c")
    rep = examine(SimpleNamespace(x={"x": np.array([3.0])}, objective=3.0), m)
    assert not any("builder" in c.name for c in rep.checks)


# ── GP-1: auto-GP refuses (falls back to B&B) when builder rows exist ────────


@pytest.mark.smoke
def test_gp1_classify_gp_refuses_builder_rows_constraint_fastpath():
    m = Model("gp1")
    s = m.set("s", ["a"])
    xv = m.continuous("xv", over=s, lb=0.5, ub=10.0)
    m.minimize(1 / xv["a"])  # monomial -> GP-shaped
    m.constraint(s, lambda i: xv[i] <= 2, "cap")  # fast path -> builder rows
    assert classify_gp(m) is None


@pytest.mark.smoke
def test_gp1_classify_gp_refuses_add_linear_constraints():
    m = Model("gp1b")
    xv = m.continuous("xv", shape=(1,), lb=0.5, ub=10.0)
    m.minimize(1 / xv[0])
    m.add_linear_constraints(sp.csr_matrix(np.array([[1.0]])), xv, "<=", np.array([2.0]))
    assert classify_gp(m) is None


@pytest.mark.smoke
def test_gp1_solve_returns_true_optimum_with_fastpath_cap():
    """End-to-end: the fast-path cap binds; solve must honour it (0.5, not 0.1)."""
    m = Model("gp1solve")
    s = m.set("s", ["a"])
    xv = m.continuous("xv", over=s, lb=0.5, ub=10.0)
    m.minimize(1 / xv["a"])
    m.constraint(s, lambda i: xv[i] <= 2, "cap")
    res = m.solve()
    assert res.status == "optimal"
    assert abs(res.objective - 0.5) < 1e-4


@pytest.mark.smoke
def test_gp1_pure_gp_still_recognised():
    """No regression: a GP with no builder rows is still recognised as a GP."""
    m = Model("puregp")
    x = m.continuous("x", lb=0.5, ub=10.0)
    m.minimize(1 / x)
    m.subject_to(x <= 2, name="cap")  # expression path
    assert classify_gp(m) is not None
