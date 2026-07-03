"""Tests for the generic ``discopt.cli`` entry-point plugin mechanism.

The seam is ``discopt.cli._cli_plugin_entry_points``: tests monkeypatch it
with fake entry points, so no real plugin package needs to be installed.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import discopt.cli as cli
import pytest

pytestmark = pytest.mark.smoke


class FakeEntryPoint:
    def __init__(self, name, module=None, error=None, value="fake.plugin.cli"):
        self.name = name
        self.value = value
        self._module = module
        self._error = error

    def load(self):
        if self._error is not None:
            raise self._error
        return self._module


def _plugin_module(name, run_result=0):
    """A module-like namespace conforming to the add_subparser/run protocol."""
    calls = {}

    def add_subparser(subparsers):
        p = subparsers.add_parser(name, help=f"{name} plugin")
        p.add_argument("--flag", default="unset")

    def run(args):
        calls["args"] = args
        return run_result

    return SimpleNamespace(add_subparser=add_subparser, run=run), calls


def _main_with(eps, argv):
    with patch.object(cli, "_cli_plugin_entry_points", lambda: eps):
        with patch("sys.argv", ["discopt", *argv]):
            cli.main()


class TestDispatch:
    def test_plugin_command_dispatches_with_parsed_args(self):
        mod, calls = _plugin_module("fake")
        with pytest.raises(SystemExit) as exc:
            _main_with([FakeEntryPoint("fake", mod)], ["fake", "--flag", "on"])
        assert exc.value.code == 0
        assert calls["args"].flag == "on"

    def test_nonzero_return_code_propagates(self):
        mod, _ = _plugin_module("fake", run_result=3)
        with pytest.raises(SystemExit) as exc:
            _main_with([FakeEntryPoint("fake", mod)], ["fake"])
        assert exc.value.code == 3

    def test_none_return_means_success(self):
        mod, _ = _plugin_module("fake", run_result=None)
        with pytest.raises(SystemExit) as exc:
            _main_with([FakeEntryPoint("fake", mod)], ["fake"])
        assert exc.value.code == 0


class TestHelp:
    def test_help_lists_plugin_subcommand(self, capsys):
        mod, _ = _plugin_module("fake")
        _main_with([FakeEntryPoint("fake", mod)], ["help"])
        assert "fake" in capsys.readouterr().out

    def test_broken_plugin_warns_but_help_prints(self, capsys):
        broken = FakeEntryPoint("broken", error=ImportError("missing dep"))
        _main_with([broken], ["help"])
        captured = capsys.readouterr()
        assert "failed to load" in captured.err
        assert "usage" in captured.out.lower()


class TestErrors:
    def test_broken_plugin_own_command_exits_1(self, capsys):
        broken = FakeEntryPoint("broken", error=ImportError("missing dep"))
        with pytest.raises(SystemExit) as exc:
            _main_with([broken], ["broken"])
        assert exc.value.code == 1
        assert "Error:" in capsys.readouterr().err

    def test_builtin_name_cannot_be_shadowed(self, capsys):
        mod, calls = _plugin_module("solve")
        # `help` path scans plugins; the reserved name is skipped with a warning.
        _main_with([FakeEntryPoint("solve", mod)], ["help"])
        captured = capsys.readouterr()
        assert "name already taken" in captured.err
        assert calls == {}

    def test_duplicate_plugin_names_first_wins(self, capsys):
        mod1, calls1 = _plugin_module("fake")
        mod2, calls2 = _plugin_module("fake")
        with pytest.raises(SystemExit):
            _main_with(
                [FakeEntryPoint("fake", mod1), FakeEntryPoint("fake", mod2)],
                ["fake"],
            )
        assert "args" in calls1
        assert calls2 == {}
        assert "name already taken" in capsys.readouterr().err


class TestLaziness:
    def test_builtin_command_never_scans_entry_points(self, capsys):
        def fail():
            pytest.fail("entry-point scan must be skipped for builtin commands")

        with patch.object(cli, "_cli_plugin_entry_points", fail):
            with patch("sys.argv", ["discopt", "about"]):
                cli.main()
        assert "discopt" in capsys.readouterr().out

    def test_other_plugins_not_loaded_for_plugin_command(self):
        mod, _ = _plugin_module("fake")
        never = FakeEntryPoint("other", error=AssertionError("must not be loaded"))
        with pytest.raises(SystemExit) as exc:
            _main_with([FakeEntryPoint("fake", mod), never], ["fake"])
        assert exc.value.code == 0
