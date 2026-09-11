"""Regression tests for issue #1222 — the .nl writer's header and ``x`` section.

Two defects, both invisible to a round-trip through discopt's *own* reader
(which mirrored the writer's mistake) and therefore both checked here against
an external oracle instead:

1. **Header line 4 ``nlvo``** was the raw count of variables appearing
   nonlinearly in objectives. ASL sizes its nonlinear-column prefix as
   ``max(nlvc, nlvo)`` and reads objective-only nonlinear columns from
   ``[nlvc, nlvo)``, so whenever a model had *both* cons-only and objs-only
   nonlinear variables the declared prefix was short and the reader
   mis-assigned columns — a silent wrong answer (Ipopt printed
   ``Optimal Solution Found.`` for a different problem). The oracle here is the
   corpus itself: every file in ``data/minlplib_nl`` was written by AMPL, so
   its own header line 4 is ground truth for the model it encodes.

2. **The ``x`` section** (initial primal guess) was never emitted — the module
   docstring listed it, no code wrote it.
"""

from __future__ import annotations

from pathlib import Path

import discopt.modeling as dm
import pytest

pytestmark = pytest.mark.unit

CORPUS = Path(__file__).parent / "data" / "minlplib_nl"

# The five in-repo instances whose AMPL header line 4 the pre-fix writer got
# wrong. Listed by name only to make the regression legible in a failure
# report; the parametrized corpus test below covers all 66 files, so the fix is
# not keyed to these names.
KNOWN_NLVO_DISAGREEMENTS = (
    "casctanks.nl",
    "contvar.nl",
    "heatexch_gen2.nl",
    "heatexch_gen3.nl",
    "st_e11.nl",
)


def _header_line(text: str, index: int) -> list[int]:
    """Numeric fields of header line *index* (comment stripped)."""
    return [int(tok) for tok in text.splitlines()[index].split("#")[0].split()]


def _nlvar_line(text: str) -> list[int]:
    """Header line 4: ``nlvc nlvo nlvb``, zero-padded to three fields."""
    fields = _header_line(text, 4)
    return fields + [0] * (3 - len(fields))


def _corpus_files() -> list[Path]:
    return sorted(CORPUS.glob("*.nl"))


class TestNlvoPrefixBound:
    """Header line 4 must carry ASL's prefix bound, not the raw objective count."""

    @pytest.mark.parametrize("path", _corpus_files(), ids=lambda p: p.name)
    def test_roundtrip_matches_ampl_header(self, path: Path):
        """``to_nl(from_nl(f))`` reproduces AMPL's own ``nlvc nlvo nlvb``.

        AMPL wrote these files, so the comparison is against an external
        oracle rather than against discopt's own reading of its own output.
        """
        original = path.read_text(encoding="latin-1")
        exported = dm.from_nl(str(path)).to_nl()
        assert _nlvar_line(exported) == _nlvar_line(original), (
            f"{path.name}: header line 4 disagrees with the AMPL original"
        )

    def test_corpus_is_present_and_covers_the_defect(self):
        """Guard the parametrization: an empty corpus would make it a no-op."""
        names = {p.name for p in _corpus_files()}
        assert len(names) >= 60, f"corpus unexpectedly small: {len(names)} files"
        missing = set(KNOWN_NLVO_DISAGREEMENTS) - names
        assert not missing, f"instances that exercised the defect are gone: {missing}"

    def test_objective_only_nonlinear_var_extends_the_prefix(self):
        """The ``fuel`` shape: cons-only *and* objs-only nonlinear variables.

        ``a`` is nonlinear in the constraint only, ``b`` in the objective only.
        The canonical column order is ``[both | cons-only | objs-only | linear]``
        = ``[a | b | ...]``, so the objective-only block ends at column 2 and
        ``nlvo`` must be 2 — the raw count (1) would truncate ASL's prefix to
        one column and hand the objective's nonlinear variable to the wrong
        column.
        """
        m = dm.Model("objs_only")
        a = m.continuous("a", lb=0, ub=10)
        b = m.continuous("b", lb=0, ub=10)
        m.minimize(b * b + a)
        m.subject_to(a * a + b <= 5)
        nlvc, nlvo, nlvb = _nlvar_line(m.to_nl())
        assert (nlvc, nlvb) == (1, 0)
        assert nlvo == 2

    def test_no_objective_nonlinearity_leaves_nlvo_zero(self):
        """A linear objective keeps ``nlvo = 0`` however large ``nlvc`` is.

        This is the case the unconditional ``nlvc + nlvo - nlvb`` correction
        proposed in #1222 gets wrong (it would declare ``nlvo = nlvc``); it is
        also the common case — that form breaks 29 of the 66 corpus
        instances.
        """
        m = dm.Model("linear_obj")
        a = m.continuous("a", lb=0, ub=10)
        b = m.continuous("b", lb=0, ub=10)
        m.minimize(a + b)
        m.subject_to(a * b <= 5)
        assert _nlvar_line(m.to_nl()) == [2, 0, 0]

    def test_objective_nonlinear_vars_subset_of_constraints(self):
        """No objs-only variable: ``nlvo`` stays at the both-count, not ``nlvc``."""
        m = dm.Model("subset")
        a = m.continuous("a", lb=0, ub=10)
        b = m.continuous("b", lb=0, ub=10)
        m.minimize(a * a + b)
        m.subject_to(a * b <= 5)
        nlvc, nlvo, nlvb = _nlvar_line(m.to_nl())
        assert (nlvc, nlvb) == (2, 1)
        assert nlvo == 1


class TestInitialPointSection:
    """The ``x`` section is emitted when — and only when — a guess is supplied."""

    @staticmethod
    def _x_block(text: str) -> list[str]:
        """The ``x`` section's lines, or ``[]`` when there is none."""
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.startswith("x") and line[1:].strip().isdigit():
                count = int(line[1:].strip())
                return lines[i : i + 1 + count]
        return []

    def _model(self):
        m = dm.Model("x_section")
        x = m.continuous("x", lb=0, ub=10)
        n = m.integer("n", lb=0, ub=5)
        z = m.continuous("z", shape=(3,), lb=0, ub=1)
        m.minimize(x * x + n + z[0])
        m.subject_to(x * n + z[1] <= 4)
        return m, x, n, z

    def test_no_section_without_an_initial_point(self):
        m, _x, _n, _z = self._model()
        assert self._x_block(m.to_nl()) == []
        assert self._x_block(m.to_nl(initial_point={})) == []

    def test_section_carries_only_the_supplied_variables(self):
        """A partial guess writes a partial section — no invented midpoints."""
        m, x, _n, z = self._model()
        block = self._x_block(m.to_nl(initial_point={x: 2.5, z: [0.1, 0.2, 0.3]}))
        assert block[0] == "x4"
        # x is nonlinear in both -> column 0; z is linear -> columns 2..4
        # (column 1 is the integer n, deliberately absent: no guess was given).
        assert [line.split()[0] for line in block[1:]] == ["0", "2", "3", "4"]
        assert [float(line.split()[1]) for line in block[1:]] == [2.5, 0.1, 0.2, 0.3]

    def test_columns_follow_the_canonical_reorder(self):
        """Entries use final ``.nl`` columns, not declaration order.

        ``lin`` is declared first but is linear, so the canonical reorder sends
        it behind the nonlinear ``nl``. A guess keyed to ``lin`` must land on
        the reordered column, not on column 0.
        """
        m = dm.Model("reorder")
        lin = m.continuous("lin", lb=0, ub=10)
        nl = m.continuous("nl", lb=0, ub=10)
        m.minimize(nl * nl + lin)
        block = self._x_block(m.to_nl(initial_point={lin: 4.0}))
        assert block == ["x1", "1 4.0"]

    def test_values_are_validated_against_bounds_and_integrality(self):
        """Out-of-bounds / fractional guesses are clamped and rounded, with a warning."""
        m, x, n, _z = self._model()
        with pytest.warns(UserWarning):
            block = self._x_block(m.to_nl(initial_point={x: 99.0, n: 2.4}))
        values = {line.split()[0]: float(line.split()[1]) for line in block[1:]}
        assert values["0"] == 10.0  # clamped to ub
        assert values["1"] == 2.0  # rounded to the nearest integer

    def test_rejects_a_variable_from_another_model(self):
        """A foreign Variable is refused loudly rather than silently dropped."""
        m, _x, _n, _z = self._model()
        other = dm.Model("other")
        stranger = other.continuous("stranger", lb=0, ub=1)
        with pytest.raises(ValueError, match="not part of this model"):
            m.to_nl(initial_point={stranger: 0.5})

    def test_rust_parser_reads_a_file_carrying_an_x_section(self, tmp_path):
        """The emitted section must not break discopt's own reader."""
        m, x, n, _z = self._model()
        path = tmp_path / "with_x.nl"
        m.to_nl(str(path), initial_point={x: 1.5, n: 3})
        assert "\nx2\n" in path.read_text()
        reparsed = dm.from_nl(str(path))
        # The reader splits blocks into scalars, so compare scalar counts.
        assert sum(v.size for v in reparsed._variables) == sum(v.size for v in m._variables)
        assert len(reparsed._constraints) == len(m._constraints)
