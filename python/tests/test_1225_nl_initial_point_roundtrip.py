"""Regression tests for issue #1225 — the ``.nl`` ``x`` segment survives a read.

#1224 taught the *writer* to emit an ``x`` (initial primal guess) section. The
*reader* still threw the section away: the Rust parser filled a dense
``vec![0.0; n_vars]`` named ``_x0`` that nothing read, so every one of the 202
corpus instances carrying an ``x`` block lost it on round-trip.

The point is now carried alongside the model — sparse ``(column, value)`` pairs
on ``PyModelRepr``, not a field of ``ModelRepr`` — so no presolve transform can
inherit column indices it renumbered. ``from_nl`` attaches it to the ``Model``
and ``to_nl`` writes it back.

The oracle is the original AMPL-written file in ``data/minlplib_nl``: its own
``x`` segment is what the round-trip has to reproduce.
"""

from __future__ import annotations

import re
from pathlib import Path

import discopt.modeling as dm
import pytest
from discopt.export.nl import _NLWriter
from discopt.modeling.core import _attach_repr_initial_point

pytestmark = pytest.mark.unit

CORPUS = Path(__file__).parent / "data" / "minlplib_nl"
X_HEADER = re.compile(r"^x(\d+)\s*$")


def _file_x_entries(text: str) -> dict[int, float] | None:
    """A ``.nl`` text's own ``x`` segment as ``{column: value}``, else ``None``.

    ``None`` (no segment) and ``{}`` (an ``x0`` segment) are deliberately
    distinct — the tests below assert on the difference.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        match = X_HEADER.match(line)
        if match:
            count = int(match.group(1))
            entries = {}
            for row in lines[i + 1 : i + 1 + count]:
                column, value = row.split()[:2]
                entries[int(column)] = float(value)
            return entries
    return None


def _text_corpus_files() -> list[Path]:
    """Corpus files in text (``g``) format — the binary one has its own test."""
    return [
        p
        for p in sorted(CORPUS.glob("*.nl"))
        if p.read_text(encoding="latin-1").lstrip().startswith("g")
    ]


def _files_with_x() -> list[Path]:
    return [p for p in _text_corpus_files() if _file_x_entries(p.read_text(encoding="latin-1"))]


def _files_without_x() -> list[Path]:
    return [
        p for p in _text_corpus_files() if _file_x_entries(p.read_text(encoding="latin-1")) is None
    ]


class TestCorpusRoundTrip:
    """``to_nl(from_nl(f))`` reproduces the file's own ``x`` segment."""

    def test_corpus_covers_the_defect(self):
        """Guard the parametrization: an empty split would make it all a no-op."""
        with_x, without_x = _files_with_x(), _files_without_x()
        assert len(with_x) >= 40, f"only {len(with_x)} corpus files carry an x segment"
        assert without_x, "no corpus file lacks an x segment; the negative arm is vacuous"

    @pytest.mark.parametrize("path", _files_with_x(), ids=lambda p: p.name)
    def test_reader_attaches_the_files_own_entries(self, path: Path):
        """The attached point equals the file's ``x`` segment, value for value.

        ``from_nl`` names flat column *i* ``x{i}``, so the file's column keys map
        onto variable names exactly and the comparison needs no permutation.
        """
        original = _file_x_entries(path.read_text(encoding="latin-1"))
        model = dm.from_nl(str(path))
        assert model._initial_point == {(f"x{col}", 0): val for col, val in original.items()}

    @pytest.mark.parametrize("path", _files_with_x(), ids=lambda p: p.name)
    def test_writer_reemits_every_entry_on_the_right_column(self, path: Path):
        """Re-export puts each value on the column the writer assigns its variable.

        The exported column is *not* the source column: the writer reorders
        variables into the canonical ``[both | cons-only | objs-only | linear]``
        sequence. What must hold is that each value still travels with its own
        variable, which is what the writer's own ``_var_index`` permutation says.
        """
        model = dm.from_nl(str(path))
        writer = _NLWriter(model)
        exported = _file_x_entries(writer.write())
        expected = {writer._var_index[key]: val for key, val in model._initial_point.items()}
        assert exported == expected

    @pytest.mark.parametrize("path", _files_without_x(), ids=lambda p: p.name)
    def test_file_without_the_segment_attaches_nothing(self, path: Path):
        model = dm.from_nl(str(path))
        assert model._initial_point == {}
        assert _file_x_entries(model.to_nl()) is None

    def test_binary_nl_file_carries_its_segment_too(self):
        """The binary (``b3``) encoding round-trips through the transcoder."""
        binary = [p for p in sorted(CORPUS.glob("*.nl")) if p.read_bytes().lstrip()[:1] == b"b"]
        assert binary, "no binary-format .nl in the corpus; this arm is vacuous"
        for path in binary:
            model = dm.from_nl(str(path))
            assert model._initial_point, f"{path.name}: binary x segment was dropped"
            assert _file_x_entries(model.to_nl())


class TestAttachedPointSemantics:
    """How the attached point interacts with ``to_nl`` and the public setter."""

    def _model(self):
        m = dm.Model("attached")
        x = m.continuous("x", lb=0, ub=10)
        y = m.continuous("y", lb=0, ub=10)
        m.minimize(x * x + y)
        m.subject_to(x + y >= 1)
        return m, x, y

    def test_explicit_argument_wins_over_the_attached_point(self):
        m, x, y = self._model()
        m.set_initial_point({x: 1.0, y: 2.0})
        block = _file_x_entries(m.to_nl(initial_point={x: 9.0}))
        assert block == {0: 9.0}  # y's attached 2.0 is not written

    def test_empty_dict_suppresses_the_attached_point(self):
        m, x, y = self._model()
        m.set_initial_point({x: 1.0})
        assert _file_x_entries(m.to_nl()) == {0: 1.0}
        assert _file_x_entries(m.to_nl(initial_point={})) is None

    def test_setter_validates_and_property_is_element_keyed(self):
        m = dm.Model("setter")
        x = m.continuous("x", lb=0, ub=5)
        n = m.integer("n", lb=0, ub=4)
        z = m.continuous("z", shape=(2,), lb=0, ub=1)
        m.minimize(x + n + z[0])
        with pytest.warns(UserWarning):
            m.set_initial_point({x: 99.0, n: 2.4, z: [0.25, 0.75]})
        assert m.initial_point == {x: {0: 5.0}, n: {0: 2.0}, z: {0: 0.25, 1: 0.75}}

    def test_setter_with_empty_mapping_clears(self):
        m, x, _y = self._model()
        m.set_initial_point({x: 1.0})
        m.set_initial_point({})
        assert m.initial_point == {}
        assert _file_x_entries(m.to_nl()) is None

    def test_property_returns_a_copy(self):
        m, x, _y = self._model()
        m.set_initial_point({x: 1.0})
        m.initial_point.clear()
        assert m.initial_point == {x: {0: 1.0}}

    def test_a_files_values_are_attached_verbatim_not_clamped(self, tmp_path):
        """A source file's guess is copied as-is, so the round-trip reproduces it.

        Only :meth:`Model.set_initial_point` clamps, because only it takes user
        input. Here the file starts ``x0`` at 500 with an upper bound of 100 —
        AMPL-written files do carry such starts, and silently moving the value to
        the bound would make the export disagree with its source.
        """
        source = tmp_path / "out_of_bounds.nl"
        source.write_text(_MINIMAL_NL_WITH_X.replace("0 1.5\n", "0 500\n"))
        model = dm.from_nl(str(source))
        assert model._initial_point == {("x0", 0): 500.0}
        assert _file_x_entries(model.to_nl()) == {0: 500.0}


class TestTransformsDropThePoint:
    """A transform renumbers columns, so its output must start from empty."""

    def _repr_with_point(self):
        from discopt._rust import parse_nl_file

        path = CORPUS / "alan.nl"
        rep = parse_nl_file(str(path))
        assert rep.initial_point(), "fixture carries no initial point"
        return rep

    def test_substitute_output_has_no_point(self):
        reduced, _chain = self._repr_with_point().substitute(4)
        assert reduced.initial_point() == []

    def test_eliminate_variables_output_has_no_point(self):
        eliminated, _stats = self._repr_with_point().eliminate_variables()
        assert eliminated.initial_point() == []

    def test_reformulate_polynomial_output_has_no_point(self):
        reformed, _stats = self._repr_with_point().reformulate_polynomial()
        assert reformed.initial_point() == []

    def test_out_of_range_column_raises_rather_than_being_skipped(self):
        """A column past the model's variables means the two disagree — refuse.

        Unreachable from the parser (it drops such a column itself), so this
        pins the helper's own invariant: never attach a quietly incomplete point.
        """

        class _Rep:
            def initial_point(self):
                return [(0, 1.0), (7, 2.0)]

        m = dm.Model("range")
        m.continuous("a", lb=0, ub=1)
        with pytest.raises(ValueError, match="outside the model's"):
            _attach_repr_initial_point(m, _Rep())


class TestSolveIsUnaffected:
    """The point is an export hint: the solve path must not read it."""

    def test_attached_point_does_not_change_the_solve(self):
        def build():
            m = dm.Model("solve_hint")
            x = m.continuous("x", lb=-3, ub=3)
            y = m.continuous("y", lb=-3, ub=3)
            m.minimize((x - 1) ** 2 + (y + 1) ** 2)
            m.subject_to(x + y <= 2)
            return m, x, y

        plain, _x, _y = build()
        hinted, hx, hy = build()
        # A start pinned to the far corner of the box, away from the optimum.
        hinted.set_initial_point({hx: -3.0, hy: 3.0})

        base = plain.solve()
        with_hint = hinted.solve()
        assert with_hint.objective == pytest.approx(base.objective, rel=1e-12, abs=1e-12)
        assert with_hint.node_count == base.node_count


# A 1-variable, 1-constraint text .nl with an x segment, used by the
# verbatim-attachment test above. Bounds are [0, 100]; the x value is patched in.
_MINIMAL_NL_WITH_X = """g3 1 1 0	# problem minimal
 1 1 1 0 0	# vars, constraints, objectives
 0 0	# nonlinear constraints, objectives
 0 0	# network constraints
 0 0 0	# nonlinear vars in cons, objs, both
 0 0 0 1 0	# flags
 0 0 0 0 0	# nbv niv nlvbi nlvci nlvoi
 1 1	# Jacobian, gradient nonzeros
 0 0	# max name lengths
 0 0 0 0 0	# common expressions
C0
n0
O0 0
n0
r
1 10.0
b
0 0.0 100.0
x1
0 1.5
k0
J0 1
0 1.0
G0 1
0 1.0
"""
