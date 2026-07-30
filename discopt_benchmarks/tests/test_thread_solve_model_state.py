"""The migration tool must refuse rather than half-land, and lower what it cannot rename.

``thread_solve_model_state.py`` rewrites locals of a 7,700-line function into
attributes of a state object. It is safe only because it *proves* the textual edit
equals an AST substitution and refuses otherwise. These tests pin the refusals and
the one non-obvious rewrite, because each was a real event:

* The first run refused on ``_mc_con_relax_fns: list[Callable] | None = None``.
  ``ast.AnnAssign.simple`` is 1 only for a bare ``Name`` target, so renaming the
  target changes the *shape*, not just the name, and the tool's own proof caught
  it. Renaming it anyway would have produced ``_mc.con_relax_fns: T = v``, which
  CPython accepts and **mypy rejects** — so "make the proof accept it" was the
  wrong fix. The tool lowers such a statement to a plain assignment instead, and
  the annotation moves to the dataclass field.
* A holder name already used inside the target function would make every rewritten
  site read that other object, with no ``NameError`` to catch it. That guard did
  not exist until this suite; the tool only checked the *migrated* names against
  module-level ones.

Every test asserts on an outcome the tool prints or writes, not on it "running".
"""

from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path

import pytest

_BENCH_ROOT = Path(__file__).resolve().parent.parent

if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from scripts.thread_solve_model_state import rewrite  # noqa: E402

pytestmark = pytest.mark.smoke


def _write(tmp_path: Path, body: str) -> Path:
    src = tmp_path / "sample.py"
    src.write_text(textwrap.dedent(body).lstrip())
    return src


def test_annotated_assignment_is_lowered_not_renamed(tmp_path: Path, capsys) -> None:
    """``x: T = v`` becomes ``h.f = v`` — a plain ``Assign``, annotation dropped."""
    src = _write(
        tmp_path,
        """
        from typing import Optional


        def solve_model():
            _mc_con_relax_fns: Optional[list] = None
            _mc_con_relax_fns = [1]
            return _mc_con_relax_fns
        """,
    )
    rewrite(src, "_mc", {"_mc_con_relax_fns": "con_relax_fns"}, apply=True)
    out = capsys.readouterr().out
    assert "AnnAssign lowered : 1" in out
    assert "annotations DROPPED" in out

    fn = ast.parse(src.read_text()).body[-1]
    assert isinstance(fn, ast.FunctionDef)
    first = fn.body[0]
    assert isinstance(first, ast.Assign), "the annotated assignment was not lowered"
    assert ast.unparse(first) == "_mc.con_relax_fns = None"
    assert not any(isinstance(n, ast.AnnAssign) for n in ast.walk(fn))
    # …and the rest of the migration still happened.
    assert ast.unparse(fn.body[1]) == "_mc.con_relax_fns = [1]"
    assert ast.unparse(fn.body[2]) == "return _mc.con_relax_fns"


def test_bare_annotation_is_refused_rather_than_deleted(tmp_path: Path) -> None:
    """``x: T`` has no value; lowering it would silently delete a statement."""
    src = _write(
        tmp_path,
        """
        def solve_model():
            _mc_mode: str
            _mc_mode = "lp"
            return _mc_mode
        """,
    )
    before = src.read_text()
    with pytest.raises(SystemExit) as ei:
        rewrite(src, "_mc", {"_mc_mode": "mode"}, apply=True)
    assert "bare annotation with no value" in str(ei.value)
    assert src.read_text() == before, "the tool wrote despite refusing"


def test_holder_name_already_in_use_is_refused(tmp_path: Path) -> None:
    """A holder name that collides silently rebinds a live object. Refuse loudly."""
    src = _write(
        tmp_path,
        """
        def solve_model():
            _mc = object()
            _mc_mode = "lp"
            return _mc, _mc_mode
        """,
    )
    before = src.read_text()
    with pytest.raises(SystemExit) as ei:
        rewrite(src, "_mc", {"_mc_mode": "mode"}, apply=True)
    assert "already referenced inside solve_model" in str(ei.value)
    assert src.read_text() == before


def test_sibling_definitions_must_stay_ast_identical(tmp_path: Path, capsys) -> None:
    """A same-named local in a sibling function must NOT be rewritten."""
    src = _write(
        tmp_path,
        """
        def other():
            _mc_mode = "nlp"
            return _mc_mode


        def solve_model():
            _mc_mode = "lp"
            return _mc_mode
        """,
    )
    rewrite(src, "_mc", {"_mc_mode": "mode"}, apply=True)
    out = capsys.readouterr().out
    assert "sibling defs proved AST-identical: 1" in out
    tree = ast.parse(src.read_text())
    other, solve = tree.body[0], tree.body[1]
    assert isinstance(other, ast.FunctionDef) and isinstance(solve, ast.FunctionDef)
    assert ast.unparse(other.body[0]) == "_mc_mode = 'nlp'"
    assert ast.unparse(solve.body[0]) == "_mc.mode = 'lp'"


def test_module_level_collision_is_refused(tmp_path: Path) -> None:
    """With a module-level twin, a *missed* site reads the global instead of raising."""
    src = _write(
        tmp_path,
        """
        _mc_mode = "global"


        def solve_model():
            _mc_mode = "lp"
            return _mc_mode
        """,
    )
    before = src.read_text()
    with pytest.raises(SystemExit) as ei:
        rewrite(src, "_mc", {"_mc_mode": "mode"}, apply=True)
    assert "also exist at module level" in str(ei.value)
    assert src.read_text() == before


def test_no_sites_is_refused_so_a_no_op_cannot_read_as_success(tmp_path: Path) -> None:
    """CLAUDE.md §6: a probe that traverses nothing must not print a pass."""
    src = _write(
        tmp_path,
        """
        def solve_model():
            return 1
        """,
    )
    with pytest.raises(SystemExit) as ei:
        rewrite(src, "_mc", {"_mc_mode": "mode"}, apply=True)
    assert "fired on nothing" in str(ei.value)
