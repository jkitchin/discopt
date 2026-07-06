"""Tests for the ``python -m discopt.debug`` CLI entry point.

Kept in a separate module from ``test_debug_bnb.py`` so the CLI surface can grow
without churning the core debugger tests.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from discopt.debug import __main__ as cli

_TINY_NL = Path(__file__).parent / "data" / "minlplib_nl" / "st_miqp3.nl"


# ── unit: argument parsing and helpers (no solve) ────────────────────────────


@pytest.mark.unit
def test_resolve_kind_maps_flags():
    parse = cli.build_parser().parse_args
    assert cli._resolve_kind(parse(["m.nl"])) is True  # REPL
    assert cli._resolve_kind(parse(["m.nl", "--json"])) == "json"
    assert cli._resolve_kind(parse(["m.nl", "--on-error"])) == "on-error"


@pytest.mark.unit
def test_json_and_on_error_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["m.nl", "--json", "--on-error"])


@pytest.mark.unit
def test_read_script_strips_comments_and_blanks(tmp_path):
    f = tmp_path / "cmds.pdbg"
    f.write_text("# a comment\n\ninfo\n// slash comment\ncontinue\n")
    assert cli._read_script(str(f)) == ["info", "continue"]
    assert cli._read_script(None) is None


@pytest.mark.unit
def test_missing_model_file_returns_error_code():
    err = io.StringIO()
    rc = cli.main(["/no/such/model.nl"], err=err)
    assert rc == 2
    assert "could not load" in err.getvalue() or "no such model" in err.getvalue()


@pytest.mark.unit
def test_script_with_json_is_rejected():
    err = io.StringIO()
    rc = cli.main(["m.nl", "--json", "--script", "x"], err=err)
    assert rc == 2
    assert "--script applies to the human REPL only" in err.getvalue()


# ── integration: a real scripted solve through the CLI ───────────────────────


@pytest.mark.smoke
@pytest.mark.skipif(not _TINY_NL.exists(), reason="minlplib_nl test corpus absent")
def test_cli_runs_scripted_repl_solve(tmp_path):
    script = tmp_path / "cmds.pdbg"
    script.write_text("info\nbreak if nodes>=1\ncontinue\nprint bound\ncontinue\n")
    err = io.StringIO()
    rc = cli.main([str(_TINY_NL), "--script", str(script), "--time-limit", "20"], err=err)
    assert rc == 0
    out = err.getvalue()
    assert "status=optimal" in out
    assert "paused at" in out  # the scripted debugger actually ran
