"""Tests for the ``discopt install-skills`` CLI subcommand."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pytest
from discopt import skills
from discopt.cli import _cmd_install_skills

# ──────────────────────────────────────────────────────────
# Package-data discovery
# ──────────────────────────────────────────────────────────


class TestSkillsPackage:
    def test_commands_dir_exists(self):
        assert skills.commands_dir().is_dir()

    def test_agents_dir_exists(self):
        assert skills.agents_dir().is_dir()

    def test_iter_commands_nonempty(self):
        names = [p.name for p in skills.iter_commands()]
        assert len(names) >= 10
        assert "formulate.md" in names
        assert "doe.md" in names
        assert "adversary.md" in names

    def test_iter_agents_nonempty(self):
        names = [p.name for p in skills.iter_agents()]
        assert names == sorted(names)  # alphabetical
        assert "highs-expert.md" in names
        assert "scip-expert.md" in names

    def test_every_command_is_markdown(self):
        for p in skills.iter_commands():
            assert p.name.endswith(".md")

    def test_every_agent_is_markdown(self):
        for p in skills.iter_agents():
            assert p.name.endswith(".md")


# ──────────────────────────────────────────────────────────
# install-skills CLI
# ──────────────────────────────────────────────────────────


def _run_install(tmp_path, *, dev=False, force=False, project_scope=True, cwd=None):
    """Invoke ``_cmd_install_skills`` with argparse-style Namespace.

    When ``project_scope=True`` (default), the CLI treats ``cwd`` (or
    *tmp_path*) as the project root and writes into ``cwd/.claude/``.
    """
    import os

    args = argparse.Namespace(
        project_scope=project_scope,
        dev=dev,
        force=force,
    )
    old_cwd = Path.cwd()
    os.chdir(cwd or tmp_path)
    try:
        _cmd_install_skills(args)
    finally:
        os.chdir(old_cwd)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestInstallSkills:
    def test_project_scope_copy_mode(self, tmp_path):
        _run_install(tmp_path)

        dest_commands = tmp_path / ".claude" / "commands"
        dest_agents = tmp_path / ".claude" / "agents"
        assert dest_commands.is_dir()
        assert dest_agents.is_dir()

        pkg_commands = {p.name for p in skills.iter_commands()}
        got_commands = {p.name for p in dest_commands.iterdir()}
        assert got_commands == pkg_commands

        pkg_agents = {p.name for p in skills.iter_agents()}
        got_agents = {p.name for p in dest_agents.iterdir()}
        assert got_agents == pkg_agents

    def test_copy_mode_byte_identical(self, tmp_path):
        _run_install(tmp_path)
        dest = tmp_path / ".claude" / "commands"
        for src in skills.iter_commands():
            got = dest / src.name
            assert got.is_file()
            assert not got.is_symlink()
            assert _hash(Path(str(src))) == _hash(got)

    def test_dev_mode_creates_symlinks(self, tmp_path):
        _run_install(tmp_path, dev=True)
        dest = tmp_path / ".claude" / "commands"
        for src in skills.iter_commands():
            link = dest / src.name
            assert link.is_symlink(), f"{link} is not a symlink"
            # resolve() should point back to package data.
            assert "skills/commands" in str(link.resolve())

    def test_force_overwrites_existing(self, tmp_path):
        dest = tmp_path / ".claude" / "commands"
        dest.mkdir(parents=True)
        stub = dest / "formulate.md"
        stub.write_text("stub content")
        stub_hash = _hash(stub)

        _run_install(tmp_path, force=True)

        pkg_src = next(p for p in skills.iter_commands() if p.name == "formulate.md")
        assert _hash(stub) == _hash(Path(str(pkg_src)))
        assert _hash(stub) != stub_hash

    def test_skip_existing_without_force(self, tmp_path, capsys):
        dest = tmp_path / ".claude" / "commands"
        dest.mkdir(parents=True)
        stub = dest / "formulate.md"
        stub.write_text("stub content")
        stub_hash = _hash(stub)

        _run_install(tmp_path, force=False)

        # Stub preserved.
        assert _hash(stub) == stub_hash
        captured = capsys.readouterr()
        assert "skip  formulate.md" in captured.out
        assert "already existed" in captured.out

    def test_project_scope_resolves_to_cwd(self, tmp_path):
        # Install from tmp_path and check the target is relative to it.
        _run_install(tmp_path)
        assert (tmp_path / ".claude").is_dir()
        assert not (tmp_path / ".." / ".claude").is_dir()

    def test_dev_mode_then_copy_mode_overwrites(self, tmp_path):
        _run_install(tmp_path, dev=True)
        dest = tmp_path / ".claude" / "commands" / "formulate.md"
        assert dest.is_symlink()

        _run_install(tmp_path, force=True)  # plain copy
        assert dest.is_file()
        assert not dest.is_symlink()


# ──────────────────────────────────────────────────────────
# Argparse wiring
# ──────────────────────────────────────────────────────────


class TestCLIWiring:
    def test_install_skills_subcommand_registered(self):
        from discopt.cli import main

        # parse_args would call sys.exit on missing subcommand; bypass by
        # testing that the subparser is known via dispatch on --help.
        with pytest.raises(SystemExit):
            import sys

            old = sys.argv[:]
            sys.argv = ["discopt", "install-skills", "--help"]
            try:
                main()
            finally:
                sys.argv = old
