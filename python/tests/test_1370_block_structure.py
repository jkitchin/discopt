"""Issue #1370 Part A: declared block structure, resolved into NLP index space.

The point of these tests is **stable identity**, not format. A label that is
merely in range after a column moves names a neighbouring variable's block, and
POUNCE cannot detect a permutation that is still internally consistent — so the
tests that matter here are the ones that would fail on an off-by-one, and the
ones that assert the validator actually traversed something (CLAUDE.md
measurement discipline §6: a probe that checks nothing reads exactly like a
probe that found nothing wrong).
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.block_structure import (
    BlockStructureError,
    block_structure_for_model,
    has_declaration,
    resolve_block_structure,
    validate_block_labels,
)
from discopt.modeling.core import Model
from discopt.solvers import SolveStatus

pounce = pytest.importorskip("pounce")

from discopt.solvers.nlp_pounce import solve_nlp_from_model  # noqa: E402


class _RecordingProblem:
    """Wrap a real ``pounce.Problem``, recording one method's arguments."""

    def __init__(self, inner, captured: dict):
        self._inner = inner
        self._captured = captured

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def set_block_structure(self, var_blocks, con_blocks):
        self._captured["var_blocks"] = list(var_blocks)
        self._captured["con_blocks"] = list(con_blocks)
        # 0.12.0 has no such method; when it gains one, still call through.
        inner = getattr(self._inner, "set_block_structure", None)
        if inner is not None:  # pragma: no cover - depends on the installed pounce
            inner(var_blocks, con_blocks)


def _install_recording_problem(monkeypatch, captured: dict, *, with_method: bool = True):
    original = pounce.Problem

    def make_problem(*args, **kwargs):
        prob = original(*args, **kwargs)
        if not with_method:
            captured["no_method"] = True
            return prob
        return _RecordingProblem(prob, captured)

    monkeypatch.setattr(pounce, "Problem", make_problem)


def two_block_model(*, quartic: bool = False) -> tuple[Model, dict]:
    """Two independent blocks joined by one shared variable.

    Block k: ``min (x_k - k)^2`` subject to ``x_k + s >= k``. ``s`` is the border.
    Deliberately built so the NLP index space is *not* the naive one:

    * ``x`` is ONE array variable of shape (2,) whose elements sit in different
      blocks — the case whole-variable annotation cannot express, and the reason
      element-wise labels exist;
    * one array-valued constraint contributes two rows from a single
      ``Constraint`` (so ``constraint_row_map`` is not the identity);
    * a fast-builder row is appended after ``model._constraints``.
    """
    m = Model("two_block")
    x = m.continuous("x", shape=(2,), lb=-10, ub=10)
    s = m.continuous("s", lb=-10, ub=10)
    power = 4 if quartic else 2
    m.minimize((x[0] - 1.0) ** power + (x[1] - 2.0) ** power + 0.5 * s**2)
    row = x + s >= np.array([1.0, 2.0])  # 2 rows, one per block
    m.subject_to(row, name="split")
    m.add_linear_constraints(np.array([[1.0]]), s, "<=", np.array([5.0]), name="border")
    m.set_block(x, [0, 1])
    m.set_block(s, -1)
    return m, {"x": x, "s": s, "row": row}


def _evaluator(model: Model):
    from discopt._tape_nlp_evaluator import make_evaluator

    return make_evaluator(model)


class TestResolution:
    def test_labels_land_on_the_right_columns_and_rows(self):
        m, _ = two_block_model()
        ev = _evaluator(m)
        bs = resolve_block_structure(m, ev)

        # Columns: x[0] -> block 0, x[1] -> block 1, s -> border.
        assert ev.n_variables == 3
        assert bs.var_blocks.tolist() == [0, 1, -1]
        # Rows: the array body's two rows split across the blocks; the builder
        # row touches only the border variable and therefore belongs to it.
        assert ev.n_constraints == 3
        assert bs.con_blocks.tolist() == [0, 1, -1]
        assert (bs.n_blocks, bs.border_dim, bs.linking_rows) == (2, 1, 1)

    def test_validation_actually_traversed_the_problem(self):
        """A validator that checks nothing must not read as a validator that passed."""
        m, _ = two_block_model()
        bs = resolve_block_structure(m, _evaluator(m))
        assert bs.checks.rows_checked == 3
        assert bs.checks.jacobian_entries_checked > 0
        assert bs.checks.hessian_entries_checked > 0

    def test_row_map_is_consumed_not_assumed(self):
        """One Constraint, many rows: labels must follow the row map, not the object count."""
        m = Model("array_rows")
        v = m.continuous("v", shape=(4,), lb=-5, ub=5)
        w = m.continuous("w", lb=-5, ub=5)
        m.minimize(dm.sum(v**2) + w**2)
        m.subject_to(v + w >= -1.0)  # ONE Constraint, FOUR rows
        m.set_block(v, [0, 0, 1, 1])
        m.set_block(w, -1)
        bs = resolve_block_structure(m, _evaluator(m))
        assert bs.var_blocks.tolist() == [0, 0, 1, 1, -1]
        assert bs.con_blocks.tolist() == [0, 0, 1, 1]

    def test_off_by_one_declaration_is_refused(self):
        """Shift the labels by one column: in range, internally consistent, wrong."""
        m, _ = two_block_model()
        ev = _evaluator(m)
        good = resolve_block_structure(m, ev)
        shifted = np.roll(good.var_blocks, 1)  # [-1, 0, 1]
        with pytest.raises(BlockStructureError):
            validate_block_labels((shifted, good.con_blocks), ev)

    def test_whole_variable_annotation_still_works(self):
        """set_block(var, k) — the decomposition form — is the same declaration."""
        m = Model("whole")
        a = m.continuous("a", lb=-5, ub=5)
        b = m.continuous("b", lb=-5, ub=5)
        c = m.continuous("c", lb=-5, ub=5)
        m.minimize(a**2 + b**2 + c**2)
        m.subject_to(a + c >= 1)
        m.subject_to(b + c >= 1)
        m.set_block(a, 0).set_block(b, 1).set_block(c, -1)
        assert m._decomp_blocks == {"a": 0, "b": 1, "c": -1}
        bs = resolve_block_structure(m, _evaluator(m))
        assert bs.var_blocks.tolist() == [0, 1, -1]
        assert bs.con_blocks.tolist() == [0, 1]

    def test_block_ids_are_densified(self):
        m = Model("sparse_ids")
        a = m.continuous("a", lb=-5, ub=5)
        b = m.continuous("b", lb=-5, ub=5)
        m.minimize(a**2 + b**2)
        m.subject_to(a >= 1)
        m.subject_to(b >= 1)
        m.set_block(a, 17).set_block(b, 4)
        bs = resolve_block_structure(m, _evaluator(m))
        assert bs.var_blocks.tolist() == [1, 0]  # ascending declared id -> dense
        assert bs.n_blocks == 2

    def test_no_declaration_is_none(self):
        m = Model("plain")
        x = m.continuous("x", lb=-5, ub=5)
        m.minimize(x**2)
        m.subject_to(x >= 1)
        assert not has_declaration(m)
        assert block_structure_for_model(m, _evaluator(m)) is None
        with pytest.raises(BlockStructureError):
            block_structure_for_model(m, _evaluator(m), required=True)


class TestRefusals:
    def test_constraint_declared_into_the_wrong_block(self):
        m, handles = two_block_model()
        m.set_constraint_block(handles["row"], [1, 1])  # row 0 touches block 0
        with pytest.raises(BlockStructureError, match="columns lie in block"):
            resolve_block_structure(m, _evaluator(m))

    def test_row_spanning_two_blocks_must_be_linking(self):
        m = Model("spanning")
        x = m.continuous("x", shape=(2,), lb=-5, ub=5)
        m.minimize(x[0] ** 2 + x[1] ** 2)
        spanning = x[0] + x[1] >= 1  # couples both blocks
        m.subject_to(spanning)
        m.set_block(x, [0, 1])
        # Derived: linking, and accepted.
        assert resolve_block_structure(m, _evaluator(m)).con_blocks.tolist() == [-1]
        # Declared into a block: refused.
        m.set_constraint_block(spanning, 0)
        with pytest.raises(BlockStructureError, match="spans more than one block"):
            resolve_block_structure(m, _evaluator(m))

    def test_objective_coupling_two_blocks_is_caught_by_the_hessian_check(self):
        """The Jacobian never sees this; the arrowhead is over the whole KKT matrix."""
        m = Model("obj_coupled")
        x = m.continuous("x", shape=(2,), lb=-5, ub=5)
        m.minimize(x[0] * x[1])  # cross term between the two blocks
        m.subject_to(x[0] >= 1)
        m.subject_to(x[1] >= 1)
        m.set_block(x, [0, 1])
        with pytest.raises(BlockStructureError, match="Lagrangian Hessian couples"):
            resolve_block_structure(m, _evaluator(m))
        # The same model is fine once the check is switched off — i.e. the
        # refusal above came from the Hessian check and not from something else.
        bs = resolve_block_structure(m, _evaluator(m), check_hessian=False)
        assert bs.checks.hessian_entries_checked == 0

    def test_single_block_is_refused(self):
        m = Model("one_block")
        x = m.continuous("x", shape=(2,), lb=-5, ub=5)
        m.minimize(x[0] ** 2 + x[1] ** 2)
        m.subject_to(x[0] >= 1)
        m.set_block(x, [0, 0])
        with pytest.raises(BlockStructureError, match="at least two blocks"):
            resolve_block_structure(m, _evaluator(m))

    def test_wrong_label_count_for_a_variable(self):
        m = Model("bad_count")
        x = m.continuous("x", shape=(3,), lb=-5, ub=5)
        m.minimize(dm.sum(x**2))
        with pytest.raises(ValueError, match="broadcastable"):
            m.set_block(x, [0, 1])

    def test_non_integer_labels_refused(self):
        m = Model("bad_dtype")
        x = m.continuous("x", shape=(2,), lb=-5, ub=5)
        m.minimize(dm.sum(x**2))
        with pytest.raises(TypeError, match="must be integers"):
            m.set_block(x, [0.5, 1.5])


class TestPassthrough:
    def test_labels_reach_pounce_verbatim(self, monkeypatch):
        captured: dict = {}
        _install_recording_problem(monkeypatch, captured)
        m, _ = two_block_model()

        result = solve_nlp_from_model(m, x0=np.zeros(3))

        assert result.status == SolveStatus.OPTIMAL
        assert captured["var_blocks"] == [0, 1, -1]
        assert captured["con_blocks"] == [0, 1, -1]

    def test_explicit_pair_is_validated_then_passed(self, monkeypatch):
        captured: dict = {}
        _install_recording_problem(monkeypatch, captured)
        m, _ = two_block_model()

        result = solve_nlp_from_model(m, x0=np.zeros(3), block_structure=([0, 1, -1], [0, 1, -1]))
        assert result.status == SolveStatus.OPTIMAL
        assert captured["var_blocks"] == [0, 1, -1]

        with pytest.raises(BlockStructureError):
            solve_nlp_from_model(m, x0=np.zeros(3), block_structure=([-1, 0, 1], [0, 1, -1]))

    def test_none_skips_the_feature(self, monkeypatch):
        captured: dict = {}
        _install_recording_problem(monkeypatch, captured)
        m, _ = two_block_model()
        result = solve_nlp_from_model(m, x0=np.zeros(3), block_structure=None)
        assert result.status == SolveStatus.OPTIMAL
        assert "var_blocks" not in captured

    def test_graceful_fallback_when_pounce_has_no_method(self, monkeypatch):
        """#394's pattern: an older pounce degrades to the full-space solve."""
        captured: dict = {}
        _install_recording_problem(monkeypatch, captured, with_method=False)
        m, _ = two_block_model()
        result = solve_nlp_from_model(m, x0=np.zeros(3))
        assert captured.get("no_method") is True
        assert result.status == SolveStatus.OPTIMAL

    def test_solution_is_unchanged_by_the_declaration(self):
        """Factorization-only: the certificate cannot move."""
        m, _ = two_block_model()
        declared = solve_nlp_from_model(m, x0=np.zeros(3))

        plain = Model("two_block_plain")
        x = plain.continuous("x", shape=(2,), lb=-10, ub=10)
        s = plain.continuous("s", lb=-10, ub=10)
        plain.minimize((x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2 + 0.5 * s**2)
        plain.subject_to(x + s >= np.array([1.0, 2.0]))
        plain.add_linear_constraints(np.array([[1.0]]), s, "<=", np.array([5.0]), name="border")
        baseline = solve_nlp_from_model(plain, x0=np.zeros(3))

        assert declared.status == baseline.status == SolveStatus.OPTIMAL
        assert np.allclose(declared.x, baseline.x, atol=1e-9)
        assert abs(declared.objective - baseline.objective) < 1e-12

    def test_length_mismatch_is_refused_not_dropped(self):
        """Labels built against a different problem are a bug, not a hint."""
        from discopt.solvers.nlp_pounce import solve_nlp

        m, _ = two_block_model()
        ev = _evaluator(m)
        with pytest.raises(ValueError, match="but this NLP has"):
            solve_nlp(ev, np.zeros(3), block_structure=([0, 1], [0, 1, -1]))

    def test_linear_solver_report_is_surfaced_when_present(self):
        """0.12.0 reports none; the field must be None, not an invented dict."""
        from discopt.solvers.nlp_pounce import _linear_solver_from_info

        assert _linear_solver_from_info({}) is None
        assert _linear_solver_from_info({"linear_solver": {"blocks": 3, "border_dim": 7}}) == {
            "blocks": 3,
            "border_dim": 7,
        }


class TestModelSolvePath:
    def test_model_solve_passes_the_declaration(self, monkeypatch):
        """The declaration reaches POUNCE through the ordinary solve entry point.

        The objective is quartic on purpose: a continuous *quadratic* model is
        classified at entry and routed to ``pounce.solve_qp``, which has no
        block-structure surface at all, so it would exercise nothing here.
        """
        captured: dict = {}
        _install_recording_problem(monkeypatch, captured)
        m, _ = two_block_model(quartic=True)

        result = m.solve(nlp_solver="pounce")

        assert result.status in ("optimal", SolveStatus.OPTIMAL)
        assert captured.get("var_blocks") == [0, 1, -1]


class TestSerialization:
    def test_declaration_round_trips(self):
        from discopt.modeling import dumps, loads

        m, handles = two_block_model()
        m.set_constraint_block(handles["row"], [0, 1])
        m.set_constraint_block("border_0", -1)

        back = loads(dumps(m))
        # `s` was declared whole-variable, so it stays in the decomposition
        # store; only `x` carries element-wise labels.
        assert {k: [int(i) for i in v] for k, v in back._block_labels_var.items()} == {"x": [0, 1]}
        assert back._decomp_blocks == {"s": -1}
        bs = resolve_block_structure(back, _evaluator(back))
        assert bs.var_blocks.tolist() == [0, 1, -1]
        assert bs.con_blocks.tolist() == [0, 1, -1]


class TestNlCompanionFile:
    """The `.nl` file route: labels in the `.nl`'s OWN column order.

    ``.nl`` puts nonlinear variables first (#210), so the export permutes
    columns. A companion file written in model order would name the wrong
    columns on any model whose permutation is non-trivial — which is why these
    tests use one.
    """

    def _permuting_model(self) -> Model:
        m = Model("permuting")
        # `lin` is linear-only and continuous, `nl` is nonlinear: the writer
        # moves `nl` ahead of `lin`, so model order != .nl order.
        lin = m.continuous("lin", shape=(2,), lb=0, ub=10)
        nl = m.continuous("nl", shape=(2,), lb=0.1, ub=10)
        shared = m.continuous("shared", lb=0, ub=10)
        m.minimize(dm.log(nl[0]) + dm.log(nl[1]) + lin[0] + lin[1] + shared)
        m.subject_to(nl[0] + lin[0] + shared >= 1, name="b0")
        m.subject_to(nl[1] + lin[1] + shared >= 1, name="b1")
        m.set_block(lin, [0, 1])
        m.set_block(nl, [0, 1])
        m.set_block(shared, -1)
        return m

    def test_permutation_is_non_trivial(self):
        """Guard the premise: if the writer stopped permuting, these tests are vacuous."""
        from discopt.export.nl import _NLWriter

        m = self._permuting_model()
        writer = _NLWriter(m)
        writer.write()
        assert [v.name for v, _ in writer._flat_vars][:2] == ["nl", "nl"]

    def test_labels_follow_the_nl_column_order(self, tmp_path):
        from discopt.export.nl import _NLWriter, to_nl

        m = self._permuting_model()
        nl_path = tmp_path / "m.nl"
        blocks_path = tmp_path / "m.blocks"
        to_nl(m, nl_path, block_structure_file=blocks_path)

        lines = blocks_path.read_text().split("\n")
        n, mm = (int(t) for t in lines[0].split())
        var_labels = [int(t) for t in lines[1].split()]
        con_labels = [int(t) for t in lines[2].split()]
        assert (n, mm) == (5, 2)
        assert len(var_labels) == n and len(con_labels) == mm

        # Check every label against the variable it actually names in the .nl.
        writer = _NLWriter(m)
        writer.write()
        expected = {"lin": [0, 1], "nl": [0, 1], "shared": [-1]}
        for idx, (var, elem) in enumerate(writer._flat_vars):
            assert var_labels[idx] == expected[var.name][elem], (
                f".nl column {idx} is {var.name}[{elem}]"
            )
        assert con_labels == [0, 1]

    def test_writers_agree_so_the_permutation_is_authoritative(self, tmp_path):
        """The companion file leans on the two .nl writers being byte-identical."""
        from discopt.export.nl import _NLWriter, _rust_nl_text

        m = self._permuting_model()
        rust = _rust_nl_text(m, None)
        python_text = _NLWriter(m).write()
        if rust is None:  # pragma: no cover - the Rust writer declined
            pytest.skip("the Rust .nl writer declined this model")
        assert rust == python_text

    def test_undeclared_model_is_refused(self, tmp_path):
        from discopt.export.nl import to_nl

        m = Model("undeclared")
        x = m.continuous("x", lb=0, ub=10)
        m.minimize(x**2)
        m.subject_to(x >= 1)
        with pytest.raises(BlockStructureError):
            to_nl(m, tmp_path / "m.nl", block_structure_file=tmp_path / "m.blocks")


class TestSharedOnlyRow:
    def test_row_over_only_shared_columns_cannot_be_claimed_by_a_block(self):
        """The dual has to go with the border, or the eliminated block is singular."""
        m, _ = two_block_model()
        # "border_0" touches only `s`, the shared variable.
        m.set_constraint_block("border_0", 0)
        with pytest.raises(BlockStructureError, match="touches no column of that block"):
            resolve_block_structure(m, _evaluator(m))

    def test_and_is_derived_onto_the_border_when_left_alone(self):
        m, _ = two_block_model()
        assert resolve_block_structure(m, _evaluator(m)).con_blocks.tolist()[-1] == -1


class TestLegacyAnnotationsAreNotPunished:
    """A Benders/Lagrangian annotation must not break an ordinary solve.

    ``set_block(var, k)`` predates this feature. A partition that is a perfectly
    good decomposition can be a poor *arrowhead* — its blocks meeting in the
    objective, say — and refusing to solve such a model would be #1370
    punishing it for an annotation aimed at `discopt.decomposition`.
    """

    def _coupled_legacy_model(self) -> Model:
        m = Model("legacy")
        a = m.continuous("a", lb=-5, ub=5)
        b = m.continuous("b", lb=-5, ub=5)
        m.minimize(a * b + a**2 + b**2)  # couples the two blocks in the objective
        m.subject_to(a >= 1)
        m.subject_to(b >= 1)
        m.set_block(a, 0).set_block(b, 1)
        return m

    def test_unsuitable_legacy_partition_is_declined_not_raised(self):
        m = self._coupled_legacy_model()
        assert has_declaration(m)
        assert block_structure_for_model(m, _evaluator(m)) is None

    def test_but_the_same_partition_declared_element_wise_raises(self):
        m = Model("explicit")
        v = m.continuous("v", shape=(2,), lb=-5, ub=5)
        m.minimize(v[0] * v[1] + v[0] ** 2 + v[1] ** 2)
        m.subject_to(v[0] >= 1)
        m.subject_to(v[1] >= 1)
        m.set_block(v, [0, 1])
        with pytest.raises(BlockStructureError):
            block_structure_for_model(m, _evaluator(m))

    def test_legacy_model_still_solves(self):
        m = self._coupled_legacy_model()
        result = m.solve(nlp_solver="pounce")
        assert result.status in ("optimal", SolveStatus.OPTIMAL)
