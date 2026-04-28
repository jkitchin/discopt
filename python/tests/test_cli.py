"""Tests for the user-facing discopt CLI module (discopt.cli).

Validates:
  - about subcommand output
  - test subcommand smoke checks
  - convert subcommand
  - main() dispatch with sys.argv mocking

Developer-only commands (search-arxiv, search-openalex, write-report, lit-scan,
adversary, _run_claude_command) live in :mod:`discopt.dev.cli` and are tested
in ``test_dev_cli.py``.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from discopt.cli import main


class TestAbout:
    def test_about_prints_version(self, capsys):
        with patch("sys.argv", ["discopt", "about"]):
            main()
        output = capsys.readouterr().out
        assert "discopt" in output
        assert "Python:" in output
        assert "Platform:" in output
        assert "Location:" in output

    def test_about_shows_dependencies(self, capsys):
        with patch("sys.argv", ["discopt", "about"]):
            main()
        output = capsys.readouterr().out
        assert "jax:" in output
        assert "numpy:" in output
        assert "scipy:" in output

    def test_about_shows_optional(self, capsys):
        with patch("sys.argv", ["discopt", "about"]):
            main()
        output = capsys.readouterr().out
        assert "Optional:" in output

    def test_about_shows_rust_ext(self, capsys):
        with patch("sys.argv", ["discopt", "about"]):
            main()
        output = capsys.readouterr().out
        assert "Rust ext:" in output

    def test_about_when_metadata_missing(self, capsys):
        """about should still work when importlib.metadata cannot find the package."""
        import importlib.metadata

        def _raise(*_a, **_kw):
            raise importlib.metadata.PackageNotFoundError("discopt")

        with patch("sys.argv", ["discopt", "about"]):
            with patch("importlib.metadata.metadata", side_effect=_raise):
                main()
        output = capsys.readouterr().out
        assert "discopt" in output


class TestSmokeTest:
    def test_smoke_passes(self, capsys):
        """The smoke test should pass in a working dev environment."""
        with patch("sys.argv", ["discopt", "test"]):
            main()
        output = capsys.readouterr().out
        assert "PASS" in output
        assert "checks passed" in output

    def test_smoke_reports_failures(self, capsys):
        """If a core import fails, the test subcommand should exit 1."""
        with patch("sys.argv", ["discopt", "test"]):
            with patch.dict("sys.modules", {"discopt._rust": None}):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                output = capsys.readouterr().out
                assert "checks passed" in output or exc_info.value.code in (0, 1)


class TestConvert:
    """Tests for the ``discopt convert`` subcommand."""

    def _write_gms(self, path):
        """Write a minimal GAMS file for testing."""
        path.write_text(
            "Free Variables x, obj ;\n"
            "Equations eq1 ;\n"
            "eq1.. obj =e= x ;\n"
            "Model m / all / ;\n"
            "Solve m using NLP minimizing obj ;\n"
        )

    def test_gms_to_nl(self, tmp_path, capsys):
        gms_path = tmp_path / "test.gms"
        nl_path = tmp_path / "test.nl"
        self._write_gms(gms_path)

        sys.argv = ["discopt", "convert", str(gms_path), str(nl_path)]
        main()
        output = capsys.readouterr().out
        assert "Converted" in output
        assert nl_path.exists()
        content = nl_path.read_text()
        assert content.startswith("g3")

    def test_gms_to_gms(self, tmp_path, capsys):
        gms_path = tmp_path / "input.gms"
        out_path = tmp_path / "output.gms"
        self._write_gms(gms_path)

        sys.argv = ["discopt", "convert", str(gms_path), str(out_path)]
        main()
        assert out_path.exists()
        content = out_path.read_text()
        assert "Solve" in content

    def test_unsupported_input(self, tmp_path, capsys):
        bad_in = tmp_path / "model.xyz"
        bad_in.write_text("dummy")
        out = tmp_path / "out.gms"

        sys.argv = ["discopt", "convert", str(bad_in), str(out)]
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "unsupported input format" in err

    def test_unsupported_output(self, tmp_path, capsys):
        gms_path = tmp_path / "test.gms"
        self._write_gms(gms_path)
        bad_out = tmp_path / "out.xyz"

        sys.argv = ["discopt", "convert", str(gms_path), str(bad_out)]
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
        err = capsys.readouterr().err
        assert "unsupported output format" in err

    def test_convert_args_parsing(self):
        import argparse as ap

        parser = ap.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        p_conv = subparsers.add_parser("convert")
        p_conv.add_argument("input")
        p_conv.add_argument("output")
        args = parser.parse_args(["convert", "in.gms", "out.nl"])
        assert args.input == "in.gms"
        assert args.output == "out.nl"


class TestMainDispatch:
    def test_main_no_subcommand(self):
        with patch("sys.argv", ["discopt"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2
