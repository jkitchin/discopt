"""Coverage tests for the AMPL .nl exporter and the discopt CLI (#87).

Exporter (``discopt.export.nl``): the load-bearing property is the ROUND
TRIP — a model written with ``to_nl`` and re-loaded with ``from_nl`` must
evaluate to exactly the same objective and constraint values at sampled
points (via ``NLPEvaluator``), for continuous/binary/integer variables,
linear and nonlinear operators (exp/log/trig/power/division), array
constraint bodies (matmul, axis sums), and builder-only objectives and
constraint blocks. Documented refusal paths (opaque ``dm.custom`` nodes,
unexported operators) must raise loudly, never emit wrong .nl text.

CLI (``discopt.cli``): subcommand paths that need no network or external
solver — about/test failure branches, convert error paths, mocked
gams-register/daemon plumbing, install-skills project scope, ``--tuning``
parsing, and the ``solve`` override/daemon-response branches.
"""

from __future__ import annotations

import io
import os
import sys
from unittest.mock import MagicMock, patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._jax.dag_compiler import compile_expression
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.export.nl import _NLWriter
from discopt.modeling.core import (
    BinaryOp,
    Constant,
    Constraint,
    Expression,
    FunctionCall,
    IndexExpression,
    SumExpression,
    SumOverExpression,
    UnaryOp,
)

pytestmark = pytest.mark.unit


# ──────────────────────────────────────────────────────────
# Round-trip helpers
# ──────────────────────────────────────────────────────────


def _var_size(v) -> int:
    return max(1, int(np.prod(v.shape)))


def _nl_perm(model, writer: _NLWriter) -> np.ndarray:
    """Map .nl flat index -> original-model flat index (writer reorders vars)."""
    offsets, off = {}, 0
    for v in model._variables:
        offsets[v.name] = off
        off += _var_size(v)
    return np.array([offsets[var.name] + elem for var, elem in writer._flat_vars])


def _flat_bounds(model) -> tuple[np.ndarray, np.ndarray]:
    lbs, ubs = [], []
    for v in model._variables:
        n = _var_size(v)
        lbs.append(np.broadcast_to(np.asarray(v.lb, dtype=float).ravel(), (n,)))
        ubs.append(np.broadcast_to(np.asarray(v.ub, dtype=float).ravel(), (n,)))
    return np.concatenate(lbs), np.concatenate(ubs)


def _write_and_reload(model, tmp_path):
    """Export via _NLWriter (so the var permutation is known) and reload."""
    writer = _NLWriter(model)
    path = tmp_path / f"{model.name}.nl"
    path.write_text(writer.write())
    reloaded = dm.from_nl(str(path))
    return writer, reloaded


def _sample_points(model, n_points=4, seed=0):
    lb, ub = _flat_bounds(model)
    rng = np.random.default_rng(seed)
    return [lb + rng.random(len(lb)) * (ub - lb) for _ in range(n_points)]


def _assert_roundtrip_values(model, tmp_path, n_points=4, seed=0):
    writer, reloaded = _write_and_reload(model, tmp_path)
    perm = _nl_perm(model, writer)
    ev1, ev2 = NLPEvaluator(model), NLPEvaluator(reloaded)
    rhs1 = np.array([c.rhs for c in model._constraints], dtype=float)
    rhs2 = np.array([c.rhs for c in reloaded._constraints], dtype=float)
    assert [c.sense for c in model._constraints] == [c.sense for c in reloaded._constraints]
    for x in _sample_points(model, n_points, seed):
        assert ev1.evaluate_objective(x) == pytest.approx(ev2.evaluate_objective(x[perm]), abs=1e-9)
        if model._constraints:
            c1 = ev1.evaluate_constraints(x) - rhs1
            c2 = ev2.evaluate_constraints(x[perm]) - rhs2
            np.testing.assert_allclose(c1, c2, atol=1e-9)
    return writer, reloaded, perm


# ──────────────────────────────────────────────────────────
# Exporter: round trips
# ──────────────────────────────────────────────────────────


class TestNLRoundTrip:
    @staticmethod
    def _minlp_model():
        m = dm.Model("rt_minlp")
        x = m.continuous("x", lb=0.5, ub=2.0)
        y = m.continuous("y", lb=0.1, ub=3.0)
        b = m.binary("b")
        k = m.integer("k", lb=1, ub=4)
        m.minimize(dm.exp(x) + dm.log(y) + x * y + 2.5 * b + k**2 + x / y + dm.tan(0.3 * x))
        m.subject_to(dm.sin(x) + dm.cos(y) <= 1.5)
        m.subject_to(x + 2 * y + b + k <= 10)
        m.subject_to(x**2 + y**2 == 3)
        m.subject_to(dm.sqrt(x) * k >= 0.5)
        m.subject_to(dm.log1p(y) + dm.sigmoid(x) + dm.softplus(y) + dm.log2(x) <= 8)
        return m

    def test_minlp_operators_roundtrip(self, tmp_path):
        """exp/log/trig/power/division/composite functions survive the round trip."""
        _assert_roundtrip_values(self._minlp_model(), tmp_path)

    def test_minlp_var_metadata_roundtrip(self, tmp_path):
        """Bounds and discrete types survive under the canonical .nl reorder."""
        m = self._minlp_model()
        writer, reloaded = _write_and_reload(m, tmp_path)
        perm = _nl_perm(m, writer)
        lb, ub = _flat_bounds(m)
        lb2, ub2 = _flat_bounds(reloaded)
        np.testing.assert_allclose(lb2, lb[perm])
        np.testing.assert_allclose(ub2, ub[perm])
        orig_types = [var.var_type for var, _ in writer._flat_vars]
        new_types = [v.var_type for v in reloaded._variables]
        assert new_types == orig_types

    def test_array_constraint_bodies_roundtrip(self, tmp_path):
        """Matmul / axis-sum / array function-call bodies expand to faithful rows.

        The writer scalarizes each array constraint body in row-major order;
        the compiled original array bodies are the oracle for the reloaded
        model's scalar rows.
        """
        m = dm.Model("rt_array")
        x = m.continuous("x", shape=(3,), lb=0.1, ub=2.0)
        big_x = m.continuous("X", shape=(2, 2), lb=0.1, ub=1.5)
        s = m.continuous("s", lb=0.2, ub=1.0)
        a_mat = np.array([[1.0, 2.0, 0.5], [0.0, 1.0, 3.0]])
        rect = np.array([[1.0, 0.5], [2.0, 1.0], [0.0, 1.5]])
        sq = np.array([[1.0, 2.0], [0.5, 1.0]])
        cv = np.array([1.0, -1.0, 2.0])
        m.minimize(x[0] + x[1] + x[2] + big_x[0, 0] + s)
        m.subject_to(Constant(a_mat) @ x <= 4.0)  # 2d @ 1d
        m.subject_to(x @ Constant(cv) >= -5.0)  # 1d @ 1d (scalar body)
        m.subject_to(x @ Constant(rect) <= 6.0)  # 1d @ 2d
        m.subject_to(big_x @ Constant(sq) <= 5.0)  # 2d @ 2d
        m.subject_to(dm.sum(x) <= 5.0)  # full reduction
        m.subject_to(dm.sum(big_x, axis=1) <= 3.0)  # axis reduction
        m.subject_to(dm.exp(x) + s <= 9.0)  # array FunctionCall + scalar var

        writer, reloaded = _write_and_reload(m, tmp_path)
        perm = _nl_perm(m, writer)
        ev2 = NLPEvaluator(reloaded)
        rhs2 = np.array([c.rhs for c in reloaded._constraints], dtype=float)
        for xv in _sample_points(m, n_points=2, seed=1):
            expected = np.concatenate(
                [np.atleast_1d(compile_expression(c.body, m)(xv)).ravel() for c in m._constraints]
            )
            got = ev2.evaluate_constraints(xv[perm]) - rhs2
            assert got.shape == expected.shape
            np.testing.assert_allclose(got, expected, atol=1e-9)

    def test_maximize_roundtrip(self, tmp_path):
        m = dm.Model("rt_max")
        x = m.continuous("x", lb=0.5, ub=2.0)
        m.maximize(dm.log(x) - x**2)
        assert "O0 1" in m.to_nl()
        _assert_roundtrip_values(m, tmp_path)

    def test_abs_and_neg_roundtrip(self, tmp_path):
        """__abs__ (o15) and an unfoldable neg (o16) both encode and round-trip."""
        m = dm.Model("rt_absneg")
        x = m.continuous("x", lb=-2.0, ub=2.0)
        m.minimize(abs(x - 1) + dm.exp(x) * (-x))
        nl = m.to_nl()
        assert "o15" in nl and "o16" in nl
        _assert_roundtrip_values(m, tmp_path)

    def test_objective_constant_offset_roundtrip(self, tmp_path):
        """A linear objective's constant lands in the O body as an n-node."""
        m = dm.Model("rt_const")
        x = m.continuous("x", lb=0.0, ub=1.0)
        y = m.continuous("y", lb=0.0, ub=1.0)
        m.minimize(x + 2 * y + 5)
        m.subject_to(x + y >= 0.5)
        assert "n5.0" in m.to_nl()
        _assert_roundtrip_values(m, tmp_path)

    def test_quadratic_builder_objective_roundtrip(self, tmp_path):
        """add_quadratic_objective (builder-only) exports 0.5 x'Sx faithfully.

        Explicitly-stored zeros in Q must be skipped, and a single-term
        quadratic must not be wrapped in a SUMLIST.
        """
        m = dm.Model("rt_quad")
        x = m.continuous("x", shape=(2,), lb=0.0, ub=1.0)
        q_mat = sp.csr_matrix(
            (np.array([0.0, 2.0]), (np.array([0, 1]), np.array([0, 1]))), shape=(2, 2)
        )
        m.add_quadratic_objective(q_mat, np.zeros(2), x)
        writer, reloaded = _write_and_reload(m, tmp_path)
        perm = _nl_perm(m, writer)
        ev2 = NLPEvaluator(reloaded)
        for xv in _sample_points(m, n_points=3, seed=2):
            # 0.5 * x' Q x with only Q[1,1]=2 stored -> x1^2.
            assert ev2.evaluate_objective(xv[perm]) == pytest.approx(xv[1] ** 2, abs=1e-9)

    def test_quadratic_builder_multi_term_with_constant_roundtrip(self, tmp_path):
        """Multi-term Q (diag + cross) plus a constant emits a SUMLIST body.

        The builder reads only triu(Q) and reflects it, so the exported
        objective is 0.5 x'(triu(Q)+striu(Q)')x + constant.
        """
        m = dm.Model("rt_quad2")
        x = m.continuous("x", shape=(2,), lb=0.0, ub=1.0)
        q_mat = np.array([[2.0, 1.0], [0.0, 4.0]])  # triu; reflected to symmetric
        m.add_quadratic_objective(q_mat, np.array([0.5, 0.0]), x, constant=3.0)
        writer, _ = _write_and_reload(m, tmp_path)
        assert "o54" in writer.write()  # SUMLIST body
        _writer, reloaded = _write_and_reload(m, tmp_path)
        perm = _nl_perm(m, writer)
        ev2 = NLPEvaluator(reloaded)
        s_mat = np.array([[2.0, 1.0], [1.0, 4.0]])
        for xv in _sample_points(m, n_points=3, seed=3):
            expected = 0.5 * xv @ s_mat @ xv + 0.5 * xv[0] + 3.0
            assert ev2.evaluate_objective(xv[perm]) == pytest.approx(expected, abs=1e-9)

    def test_builder_linear_blocks_ge_eq_roundtrip(self, tmp_path):
        """add_linear_constraints blocks with >= and == senses export r rows."""
        m = dm.Model("rt_blocks")
        x = m.continuous("x", shape=(2,), lb=0.0, ub=4.0)
        m.add_linear_objective(np.array([1.0, 2.0]), x)
        m.add_linear_constraints(np.array([[1.0, 1.0]]), x, ">=", np.array([1.0]))
        m.add_linear_constraints(np.array([[1.0, -1.0]]), x, "==", np.array([0.5]))
        nl = m.to_nl()
        r_section = nl[nl.index("r\n") : nl.index("b\n")]
        assert "2 1.0" in r_section and "4 0.5" in r_section
        _writer, reloaded = _write_and_reload(m, tmp_path)
        # The DSL normalizes body >= rhs to (rhs - body) <= 0 on reload.
        assert [c.sense for c in reloaded._constraints] == ["<=", "=="]

    def test_sum_over_terms_roundtrip(self, tmp_path):
        """Indexed dm.sum(..., over=...) bodies round-trip, bare and nested.

        The bare sum is decomposed term-by-term; the one nested inside exp()
        survives as a SUMLIST (o54) node.
        """
        m = dm.Model("rt_sumover")
        x = m.continuous("x", shape=(3,), lb=0.1, ub=1.0)
        m.minimize(x[0] + x[1] + x[2])
        m.subject_to(dm.sum(lambda i: x[i] ** 2, over=range(3)) <= 2.0)
        m.subject_to(dm.exp(dm.sum(lambda i: 0.5 * x[i], over=range(3))) <= 4.0)
        m.subject_to(dm.sum(x, axis=0) <= 2.5)  # axis reduction to a scalar
        assert "o54" in m.to_nl()
        _assert_roundtrip_values(m, tmp_path)


# ──────────────────────────────────────────────────────────
# Exporter: structural edge cases
# ──────────────────────────────────────────────────────────


class TestNLWriterEdgeCases:
    def test_ge_and_unknown_sense_constraint_rows(self):
        """A raw >= constraint shifts its lb by the body constant; an unknown
        sense degrades to a free row (type 3) rather than a wrong bound."""
        m = dm.Model("senses")
        x = m.continuous("x", lb=0.0, ub=5.0)
        m.minimize(x)
        m._constraints.append(Constraint(x + 2.0, ">=", 1.0))  # lb = 1 - 2 = -1
        m._constraints.append(Constraint(x + 1.0, "??", 0.0))  # free row
        nl = m.to_nl()
        r_section = nl[nl.index("r\n") : nl.index("b\n")]
        assert "2 -1.0" in r_section
        assert "\n3\n" in r_section

    def test_ub_only_variable_bound(self):
        m = dm.Model("ubonly")
        z = m.continuous("z", ub=5.0)
        m.minimize(z)
        nl = m.to_nl()
        assert "1 5.0" in nl[nl.index("b\n") :]

    def test_quadratic_builder_all_zero_Q_gives_constant_objective(self):
        """All-explicit-zero Q + zero c: n0 objective body and no G section."""
        m = dm.Model("quad0")
        x = m.continuous("x", shape=(2,), lb=0.0, ub=1.0)
        q_mat = sp.csr_matrix((np.array([0.0]), (np.array([0]), np.array([0]))), shape=(2, 2))
        m.add_quadratic_objective(q_mat, np.zeros(2), x)
        nl = m.to_nl()
        assert "O0 0\nn0\n" in nl
        assert "G0" not in nl

    def test_builder_linear_block_skips_zero_coeff_and_empty_row(self):
        """Explicit zeros in an add_linear_constraints block are dropped; an
        all-zero row gets no J block at all."""
        m = dm.Model("linrows")
        x = m.continuous("x", shape=(2,), lb=0.0, ub=1.0)
        m.add_linear_objective(np.array([1.0, 1.0]), x)
        a_mat = sp.csr_matrix(
            (np.array([1.0, 0.0]), (np.array([0, 1]), np.array([0, 1]))), shape=(2, 2)
        )
        m.add_linear_constraints(a_mat, x, "<=", np.array([1.0, 2.0]))
        nl = m.to_nl()
        assert "J0 1" in nl  # row 0: single surviving nonzero
        assert "J1" not in nl  # row 1: only an explicit zero -> empty


# ──────────────────────────────────────────────────────────
# Exporter: documented refusals
# ──────────────────────────────────────────────────────────


class TestNLExportRefusals:
    @staticmethod
    def _scalar_model():
        m = dm.Model("refuse")
        m.continuous("x", lb=0.1, ub=2.0)
        return m, m._variables[0]

    def test_custom_call_refused(self):
        m, x = self._scalar_model()
        weird = dm.custom(lambda a: a**2, name="weird")
        m.minimize(weird(x))
        with pytest.raises(ValueError, match="dm.custom"):
            m.to_nl()

    def test_matmul_objective_refused(self):
        m = dm.Model("mm_obj")
        x = m.continuous("x", shape=(2,), lb=0.0, ub=1.0)
        m.minimize(x @ Constant(np.array([1.0, 2.0])))
        with pytest.raises(ValueError, match="MatMul"):
            m.to_nl()

    @pytest.mark.parametrize("fn_name", ["erf", "sign", "min", "max", "frobnicate"])
    def test_unexportable_functions_refused(self, fn_name):
        m, x = self._scalar_model()
        m.minimize(FunctionCall(fn_name, x))
        with pytest.raises(ValueError):
            m.to_nl()

    def test_unknown_binary_op_refused(self):
        m, x = self._scalar_model()
        m.minimize(BinaryOp("%", x, Constant(2.0)))
        with pytest.raises(ValueError, match="Unknown binary operator"):
            m.to_nl()

    def test_unknown_unary_op_refused(self):
        m, x = self._scalar_model()
        m.minimize(UnaryOp("floor", x))
        with pytest.raises(ValueError, match="Unknown unary operator"):
            m.to_nl()

    def test_array_variable_unindexed_refused(self):
        m = dm.Model("arrvar")
        x = m.continuous("x", shape=(2,), lb=0.0, ub=1.0)
        m.minimize(x[0])
        writer = _NLWriter(m)
        writer.write()
        with pytest.raises(ValueError, match="array variable"):
            writer._write_expr(x, io.StringIO())

    def test_unresolvable_index_refused(self):
        m = dm.Model("badidx")
        x = m.continuous("x", shape=(3,), lb=0.0, ub=1.0)
        m.minimize(x[0])
        writer = _NLWriter(m)
        writer.write()
        with pytest.raises(ValueError, match="Cannot resolve"):
            writer._write_expr(IndexExpression(x, slice(0, 2)), io.StringIO())

    def test_unknown_expression_type_refused(self):
        class Alien(Expression):
            pass

        m, x = self._scalar_model()
        m.minimize(x)
        writer = _NLWriter(m)
        writer.write()
        with pytest.raises(ValueError, match="Cannot write expression type"):
            writer._write_expr(Alien(), io.StringIO())

    @pytest.mark.xfail(
        reason="array-structured objectives are not scalarized by the .nl writer "
        "(constraint bodies are): minimize(dm.sum(x)) on an array variable raises "
        "'Cannot write array variable' even though the model solves",
        strict=False,
    )
    def test_sum_of_array_variable_objective_exports(self):
        m = dm.Model("sum_obj")
        x = m.continuous("x", shape=(3,), lb=0.0, ub=1.0)
        m.minimize(dm.sum(x))
        nl = m.to_nl()
        assert nl is not None and nl.startswith("g3")


# ──────────────────────────────────────────────────────────
# Exporter: internal helper units
# ──────────────────────────────────────────────────────────


class TestNLWriterInternals:
    @pytest.fixture()
    def writer(self):
        m = dm.Model("internals")
        x = m.continuous("x", shape=(2,), lb=0.0, ub=1.0)
        m.minimize(x[0] + x[1])
        w = _NLWriter(m)
        w.write()
        return w, m, x

    def test_split_expr_array_variable_is_nonlinear(self, writer):
        w, _m, x = writer
        linear, nonlinear, const = w._split_expr(x)
        assert linear == {} and const == 0.0
        assert nonlinear is x

    def test_split_expr_unresolved_index_is_nonlinear(self, writer):
        w, _m, x = writer
        node = IndexExpression(x, slice(0, 1))
        linear, nonlinear, _ = w._split_expr(node)
        assert linear == {} and nonlinear is node

    def test_resolve_var_index_rejects_non_variable_base(self, writer):
        w, _m, x = writer
        assert w._resolve_var_index(IndexExpression(x[0] + 1.0, 0)) is None

    def test_resolve_var_index_rejects_non_int_index(self, writer):
        w, _m, x = writer
        assert w._resolve_var_index(IndexExpression(x, "a")) is None

    def test_sum_terms_empty_is_zero(self, writer):
        w, _m, _x = writer
        out = w._sum_terms([])
        assert isinstance(out, Constant) and float(out.value) == 0.0

    def test_matmul_scalar_rejects_unsupported_shapes(self, writer):
        w, _m, _x = writer
        scalar = np.empty((), dtype=object)
        scalar[()] = Constant(1.0)
        with pytest.raises(ValueError, match="Unsupported matmul"):
            w._matmul_scalar(scalar, scalar)

    def test_scalarize_sum_over_broadcasts_terms(self, writer):
        w, m, x = writer
        node = SumOverExpression([Constant(np.array([1.0, 2.0])), x])
        arr = w._scalarize(node)
        assert arr.shape == (2,)
        fn0 = compile_expression(arr[0], m)
        fn1 = compile_expression(arr[1], m)
        xv = np.array([0.25, 0.75])
        assert float(fn0(xv)) == pytest.approx(1.0 + 0.25)
        assert float(fn1(xv)) == pytest.approx(2.0 + 0.75)

    def test_scalarize_opaque_leaf_passthrough(self, writer):
        w, _m, _x = writer

        class Opaque(Expression):
            pass

        leaf = Opaque()
        arr = w._scalarize(leaf)
        assert arr.shape == () and arr[()] is leaf

    def test_collect_var_indices_through_unary_and_sum(self, writer):
        w, _m, x = writer
        got: set[int] = set()
        w._collect_var_indices(UnaryOp("abs", x[0]), got)
        w._collect_var_indices(SumExpression(x[1]), got)
        assert got == {0, 1}

    def test_write_O_section_no_objective_writes_nothing(self, writer):
        w, m, _x = writer
        m._objective = None
        buf = io.StringIO()
        w._write_O_section(buf)
        assert buf.getvalue() == ""

    def test_write_r_section_range_row(self, writer):
        w, _m, _x = writer
        w._con_bounds = [(0, 1.0, 2.0)]
        buf = io.StringIO()
        w._write_r_section(buf)
        assert buf.getvalue() == "r\n0 1.0 2.0\n"

    def test_write_expr_sum_expression_and_sumover(self, writer):
        w, _m, x = writer
        buf = io.StringIO()
        w._write_expr(SumExpression(x[0]), buf)
        assert buf.getvalue() == "v0\n"
        buf = io.StringIO()
        w._write_expr(SumOverExpression([]), buf)
        assert buf.getvalue() == "n0\n"
        buf = io.StringIO()
        w._write_expr(SumOverExpression([x[1]]), buf)
        assert buf.getvalue() == "v1\n"  # single term: no SUMLIST wrapper


# ──────────────────────────────────────────────────────────
# CLI: about / test subcommands (failure branches)
# ──────────────────────────────────────────────────────────


from discopt.cli import main as _cli_main  # noqa: E402  (after JAX env setup)


def _main(argv):
    with patch("sys.argv", ["discopt", *argv]):
        _cli_main()


class TestCliAboutAndTest:
    def test_about_degrades_without_rust_ext_and_dep_metadata(self, capsys):
        import importlib.metadata

        def _raise(_name):
            raise importlib.metadata.PackageNotFoundError(_name)

        with patch.dict(sys.modules, {"discopt._rust": None}):
            with patch("importlib.metadata.version", side_effect=_raise):
                _main(["about"])
        out = capsys.readouterr().out
        assert "Rust ext:     not available" in out
        assert "not installed" in out

    def test_smoke_test_reports_core_import_failures(self, capsys):
        with patch.dict(sys.modules, {"discopt": None}):
            with pytest.raises(SystemExit) as exc_info:
                _main(["test"])
        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "FAIL" in out and "checks passed" in out

    def test_smoke_test_reports_jax_failure(self, capsys):
        with patch.dict(sys.modules, {"jax": None}):
            with pytest.raises(SystemExit) as exc_info:
                _main(["test"])
        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "FAIL  JAX" in out

    def test_smoke_test_flags_wrong_objective(self, capsys):
        class _FakeVar:
            def __ge__(self, other):
                return "fake-constraint"

        fake_model = MagicMock()
        fake_model.continuous.return_value = _FakeVar()
        fake_model.solve.return_value.objective = 42.0
        with patch("discopt.Model", return_value=fake_model):
            with patch("discopt.modeling.Model", side_effect=RuntimeError("dag boom")):
                with pytest.raises(SystemExit) as exc_info:
                    _main(["test"])
        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "expected objective ~1.0" in out
        assert "FAIL  DAG compiler: dag boom" in out


# ──────────────────────────────────────────────────────────
# CLI: convert
# ──────────────────────────────────────────────────────────


class TestCliConvert:
    def test_nl_input_to_gms(self, tmp_path, capsys):
        m = dm.Model("conv")
        x = m.continuous("x", lb=0.0, ub=1.0)
        m.minimize(x)
        m.subject_to(x >= 0.5)
        nl_path = tmp_path / "conv.nl"
        m.to_nl(str(nl_path))
        out_path = tmp_path / "conv.gms"
        _main(["convert", str(nl_path), str(out_path)])
        assert "Converted" in capsys.readouterr().out
        assert out_path.exists()

    def test_missing_input_file_errors(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            _main(["convert", str(tmp_path / "nope.gms"), str(tmp_path / "out.nl")])
        assert exc_info.value.code == 1
        assert "Error" in capsys.readouterr().err

    def test_invalid_nl_content_errors(self, tmp_path, capsys):
        bad = tmp_path / "bad.nl"
        bad.write_text("this is not an nl file\n")
        with pytest.raises(SystemExit) as exc_info:
            _main(["convert", str(bad), str(tmp_path / "out.gms")])
        assert exc_info.value.code == 1
        assert "Error" in capsys.readouterr().err


# ──────────────────────────────────────────────────────────
# CLI: gams-register / gams-daemon / gams-verify (mocked seams)
# ──────────────────────────────────────────────────────────


class TestCliGams:
    def test_register_check_ok(self, capsys):
        with patch("discopt.gams.check_gamsapi", return_value=(True, "all good")):
            with pytest.raises(SystemExit) as exc_info:
                _main(["gams-register", "--check"])
        assert exc_info.value.code == 0
        assert "OK: all good" in capsys.readouterr().out

    def test_register_check_problem(self, capsys):
        with patch("discopt.gams.check_gamsapi", return_value=(False, "no gamsapi")):
            with pytest.raises(SystemExit) as exc_info:
                _main(["gams-register", "--check"])
        assert exc_info.value.code == 1
        assert "PROBLEM: no gamsapi" in capsys.readouterr().out

    @staticmethod
    def _registration(directory):
        return {
            "action": "created",
            "config": str(directory / "gamsconfig.yaml"),
            "script": str(directory / "discopt-gams"),
        }

    def test_register_writes_to_custom_directory(self, tmp_path, capsys):
        with patch("discopt.gams.check_gamsapi", return_value=(True, "ok")):
            with patch(
                "discopt.gams.write_registration", return_value=self._registration(tmp_path)
            ) as wr:
                _main(["gams-register", "--directory", str(tmp_path)])
        out = capsys.readouterr().out
        assert "Created" in out and "Merge" in out  # non-home dir: manual-merge hint
        wr.assert_called_once()

    def test_register_home_directory_message(self, tmp_path, capsys):
        from pathlib import Path

        home_gams = Path.home() / ".gams"
        with patch("discopt.gams.check_gamsapi", return_value=(True, "ok")):
            with patch(
                "discopt.gams.write_registration", return_value=self._registration(tmp_path)
            ):
                _main(["gams-register", "--directory", str(home_gams)])
        assert "GAMS reads" in capsys.readouterr().out

    def test_gams_daemon_dispatches_action(self):
        with patch("discopt.gams.daemon.main", return_value=0) as dmain:
            _main(["gams-daemon", "status"])
        dmain.assert_called_once_with(["status"])

    def test_gams_verify_exit_code_propagates(self):
        with patch("discopt.gams.verify", return_value=3) as ver:
            with pytest.raises(SystemExit) as exc_info:
                _main(["gams-verify", "--keep"])
        assert exc_info.value.code == 3
        assert ver.call_args.kwargs["keep"] is True


# ──────────────────────────────────────────────────────────
# CLI: install-skills / daemon
# ──────────────────────────────────────────────────────────


class TestCliInstallSkillsAndDaemon:
    def test_install_skills_project_scope(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        _main(["install-skills", "--project-scope"])
        out = capsys.readouterr().out
        dest = tmp_path / ".claude" / "commands"
        assert dest.is_dir() and any(dest.iterdir())
        assert "Installed" in out and str(tmp_path / ".claude") in out

    def test_install_skills_force_replaces_directory_collision(self, tmp_path, monkeypatch):
        from discopt.skills import iter_commands

        name = next(iter(iter_commands())).name
        monkeypatch.chdir(tmp_path)
        blocker = tmp_path / ".claude" / "commands" / name
        blocker.mkdir(parents=True)
        (blocker / "junk.txt").write_text("x")
        _main(["install-skills", "--project-scope", "--force"])
        assert blocker.is_file()  # the directory was rmtree'd and replaced

    def test_install_skills_user_scope_uses_home(self, tmp_path, capsys):
        import discopt.cli as cli_mod

        with patch.object(cli_mod.Path, "home", return_value=tmp_path):
            _main(["install-skills"])
        dest = tmp_path / ".claude" / "commands"
        assert dest.is_dir() and any(dest.iterdir())

    def test_daemon_subcommand_dispatches_action(self):
        with patch("discopt.daemon.main", return_value=0) as dmain:
            _main(["daemon", "status"])
        dmain.assert_called_once_with(["status"])


# ──────────────────────────────────────────────────────────
# CLI: --tuning parsing helpers
# ──────────────────────────────────────────────────────────


class TestTuningParsing:
    def test_coerce_bool_variants(self):
        from discopt.cli import _coerce_tuning_value

        for s in ("true", "1", "yes", "on", "True"):
            assert _coerce_tuning_value(s, bool) is True
        for s in ("false", "0", "no", "off", "OFF"):
            assert _coerce_tuning_value(s, bool) is False
        with pytest.raises(ValueError, match="expected a boolean"):
            _coerce_tuning_value("maybe", bool)

    def test_coerce_int_float_str(self):
        from discopt.cli import _coerce_tuning_value

        assert _coerce_tuning_value("7", int) == 7
        assert _coerce_tuning_value("0.5", float) == 0.5
        assert _coerce_tuning_value("plain", str) == "plain"
        # string type hints (dataclass `from __future__ import annotations`)
        assert _coerce_tuning_value("3", "int") == 3

    def test_parse_tuning_valid_pairs(self):
        from discopt.cli import _parse_tuning

        out = _parse_tuning(["rlt_quad=false", "rlt_quad_max=8"])
        assert out == {"rlt_quad": False, "rlt_quad_max": 8}
        assert _parse_tuning(None) == {}

    @pytest.mark.parametrize(
        "pair", ["rlt_quad", "no_such_field=1", "rlt_quad=maybe"], ids=["no-eq", "unknown", "bad"]
    )
    def test_parse_tuning_errors_exit_2(self, pair, capsys):
        from discopt.cli import _parse_tuning

        with pytest.raises(SystemExit) as exc_info:
            _parse_tuning([pair])
        assert exc_info.value.code == 2
        assert "Error" in capsys.readouterr().err


# ──────────────────────────────────────────────────────────
# CLI: solve (overrides + daemon response branches)
# ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tiny_nl(tmp_path_factory):
    m = dm.Model("tiny")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.minimize(x)
    m.subject_to(x >= 1.0)
    path = tmp_path_factory.mktemp("solve") / "tiny.nl"
    m.to_nl(str(path))
    return path


@pytest.fixture(scope="module")
def real_result(tiny_nl):
    """One real solve, reused by every daemon/override test below."""
    return dm.from_nl(str(tiny_nl)).solve(time_limit=30)


@pytest.mark.smoke
class TestCliSolve:
    def test_missing_file_errors(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            _main(["solve", str(tmp_path / "ghost.nl"), "--no-daemon"])
        assert exc_info.value.code == 1
        assert "no such file" in capsys.readouterr().err

    def test_all_flag_overrides_reach_solve(self, tiny_nl, real_result, capsys):
        fake_model = MagicMock()
        fake_model.solve.return_value = real_result
        argv = [
            "solve",
            str(tiny_nl),
            "--time-limit",
            "3",
            "--gap",
            "0.25",
            "--threads",
            "2",
            "--solver",
            "amp",
            "--partitions",
            "2",
            "--rlt",
            "off",
            "--no-nlp-bb",
            "--tuning",
            "rlt_quad=false",
            "--no-daemon",
            "--quiet",
        ]
        with patch("discopt.modeling.core.from_nl", return_value=fake_model):
            with pytest.raises(SystemExit) as exc_info:
                _main(argv)
        assert exc_info.value.code == 0
        kwargs = fake_model.solve.call_args.kwargs
        assert kwargs["time_limit"] == 3.0
        assert kwargs["gap_tolerance"] == 0.25
        assert kwargs["threads"] == 2
        assert kwargs["solver"] == "amp"
        assert kwargs["partitions"] == 2
        assert kwargs["rlt"] == "off"
        assert kwargs["nlp_bb"] is False
        assert kwargs["tuning"].rlt_quad is False

    def test_daemon_error_response_exits_1(self, tiny_nl, capsys):
        resp = {"ok": False, "error": "boom"}
        with patch("discopt.daemon.solve_via_daemon", return_value=resp):
            with pytest.raises(SystemExit) as exc_info:
                _main(["solve", str(tiny_nl)])
        assert exc_info.value.code == 1
        assert "Error (daemon): boom" in capsys.readouterr().err

    def test_daemon_success_response(self, tiny_nl, real_result, capsys):
        from discopt.result_io import serialize_result

        resp = {"ok": True, "result": serialize_result(real_result)}
        with patch("discopt.daemon.solve_via_daemon", return_value=resp):
            with pytest.raises(SystemExit) as exc_info:
                _main(["solve", str(tiny_nl)])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "status:" in out and "optimal" in out

    def test_daemon_unavailable_falls_back_in_process(self, tiny_nl, real_result, capsys):
        fake_model = MagicMock()
        fake_model.solve.return_value = real_result
        with patch("discopt.daemon.solve_via_daemon", return_value=None):
            with patch("discopt.modeling.core.from_nl", return_value=fake_model):
                with pytest.raises(SystemExit) as exc_info:
                    _main(["solve", str(tiny_nl)])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "daemon unavailable" in captured.err
        assert "status:" in captured.out


# ──────────────────────────────────────────────────────────
# CLI: plugin entry-point seam
# ──────────────────────────────────────────────────────────


def test_plugin_entry_point_scan_failure_returns_empty():
    from discopt.cli import _cli_plugin_entry_points

    with patch("importlib.metadata.entry_points", side_effect=RuntimeError("broken")):
        assert _cli_plugin_entry_points() == []
