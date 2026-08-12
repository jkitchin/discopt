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


# Issue #433 regression: the DOE extraction (#410) left stale `doe` references in
# pyproject.toml (extras, `all`, the `discopt.cli` entry point, a phantom maturin
# include). They caused a spurious plugin-load warning on every `discopt help`.
# Guard against them (and future extractions) reappearing.
def test_pyproject_has_no_stale_doe_references():
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():  # running from an installed wheel, not the repo
        pytest.skip("pyproject.toml not present (installed context)")
    text = pyproject.read_text()
    stale = [
        'doe = ["openpyxl',  # the `doe` extra
        "doe-gui = ",  # the `doe-gui` extra
        "doe,doe-gui",  # the `all` aggregate
        "discopt.doe.cli",  # the core `discopt.cli` entry point (moved to discopt-doe)
        "doe/gui/discopt-logo.png",  # the phantom maturin include (file removed by #410)
    ]
    found = [s for s in stale if s in text]
    assert not found, f"stale DOE references survive in pyproject.toml (issue #433): {found}"


# Issue #433, generalized. The pyproject guard above catches stale references in
# *one* file, but the DOE extraction (#389/#410) also left dangling
# `python/discopt/doe/...` paths and a `discopt[doe]` extra in the **shipped
# skill bundle** — the markdown `discopt install-skills` copies onto a user's
# machine, where a wrong path sends a reader (or an agent) hunting for a file
# that is not there. These two tests check the *class* rather than the DOE
# instance, so the next module extraction is caught automatically.
def _repo_root():
    import pathlib

    return pathlib.Path(__file__).resolve().parents[2]


def _shipped_skill_files():
    root = _repo_root()
    skills = root / "python" / "discopt" / "skills"
    if not skills.is_dir():  # running from an installed wheel, not the repo
        pytest.skip("skills bundle not present (installed context)")
    return sorted(skills.rglob("*.md"))


def test_shipped_skills_reference_only_existing_in_tree_paths():
    """Every ``python/discopt/...`` path named in a shipped skill must exist.

    Catches a module extracted to a plugin (DOE, #389) or a file renamed out from
    under the docs — ``solvers/ipopt_wrapper.py`` and ``solvers/lp_highs.py`` were
    both dead when this test was written.
    """
    import re

    root = _repo_root()
    # Paths as written in prose: a slash path under python/discopt with an
    # optional file extension. Trailing punctuation is stripped by the charset.
    pattern = re.compile(r"python/discopt/[A-Za-z0-9_/]+(?:\.[A-Za-z0-9]+)?")
    dangling: dict[str, list[str]] = {}
    checked = 0
    for path in _shipped_skill_files():
        for ref in sorted(set(pattern.findall(path.read_text()))):
            checked += 1
            target = root / ref
            # A bare directory reference may be written without its trailing
            # slash, and a module may be named without its ``.py``.
            if target.exists() or (root / f"{ref}.py").exists():
                continue
            dangling.setdefault(ref, []).append(str(path.relative_to(root)))
    assert checked > 0, "probe scanned no path references — the pattern or the bundle moved"
    assert not dangling, (
        "shipped skills name in-tree paths that do not exist "
        f"(extracted to a plugin, or renamed?): {dangling}"
    )


def test_shipped_skills_only_recommend_declared_extras():
    """``pip install "discopt[x]"`` in a shipped skill must name a real extra.

    The DOE extraction removed the ``doe`` / ``doe-gui`` extras but left
    ``pip install "discopt[...]", `[doe]``` advice in ``skills/commands/debug.md``,
    which fails for the user who follows it.
    """
    import re

    root = _repo_root()
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("pyproject.toml not present (installed context)")
    block = pyproject.read_text().split("[project.optional-dependencies]", 1)[1].split("\n[", 1)[0]
    declared = set(re.findall(r"^([a-z0-9-]+) = \[", block, re.M))
    assert declared, "failed to parse any extras out of pyproject.toml"

    undeclared: list[tuple[str, str]] = []
    checked = 0
    for path in _shipped_skill_files():
        rel = str(path.relative_to(root))
        for line in path.read_text().split("\n"):
            names = re.findall(r"discopt\[([a-z0-9,\-]+)\]", line)
            # debug.md's continuation form: `pip install "discopt[nn]"`, `[llm]`, ...
            if "pip install" in line:
                names += re.findall(r"`\[([a-z0-9,\-]+)\]`", line)
            for group in names:
                for name in group.split(","):
                    checked += 1
                    if name not in declared:
                        undeclared.append((rel, name))
    assert checked > 0, "probe found no extras references — the pattern or the bundle moved"
    assert not undeclared, (
        f"shipped skills recommend extras that pyproject.toml does not declare: {undeclared} "
        f"(declared: {sorted(declared)})"
    )
