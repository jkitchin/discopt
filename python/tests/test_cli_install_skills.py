"""Tests for the ``discopt install-skills`` CLI subcommand."""

from __future__ import annotations

import argparse
import hashlib
import re
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
        assert len(names) >= 5
        assert "formulate.md" in names
        # doe.md moved to the discopt-doe plugin (#389).
        assert "doe.md" not in names
        # Dev-only commands (discoptbot, adversary) intentionally don't ship
        # in the bundled skills; they live in `.claude/commands/` only.
        assert "discoptbot.md" not in names
        assert "adversary.md" not in names

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

    def test_iter_skills_yields_the_overview_skill(self):
        names = [p.name for p in skills.iter_skills()]
        assert names == sorted(names)
        assert "discopt" in names
        for p in skills.iter_skills():
            assert (p / "SKILL.md").is_file()


# ──────────────────────────────────────────────────────────
# Agent Skills / knowhow layout (https://github.com/jkitchin/knowhow)
# ──────────────────────────────────────────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Agent Skills spec name rule, as enforced by ``knowhow validate``.
_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def _frontmatter(path: Path) -> dict[str, str]:
    text = path.read_text()
    assert text.startswith("---\n"), f"{path} has no YAML frontmatter"
    block = text[4 : text.index("\n---", 4)]
    out = {}
    for line in block.splitlines():
        key, sep, value = line.partition(":")
        if sep and not line.startswith(" "):
            out[key.strip()] = value.strip()
    return out


class TestKnowhowLayout:
    """``knowhow add jkitchin/discopt`` discovers ``commands/``, ``agents/`` and
    ``skills/`` at the repository root only; the root entries are symlinks to the
    package data so there is one source of truth."""

    @pytest.mark.parametrize(
        "name,target",
        [
            ("commands", skills.commands_dir),
            ("agents", skills.agents_dir),
            ("skills", skills.skills_dir),
        ],
    )
    def test_root_link_resolves_to_package_data(self, name, target):
        link = _REPO_ROOT / name
        if not link.exists():
            pytest.skip("not running from a source checkout")
        assert link.is_symlink(), f"{link} must be a symlink, not a copy"
        assert link.resolve() == Path(str(target())).resolve()

    def test_skill_frontmatter_meets_agent_skills_spec(self):
        checked = 0
        for d in skills.iter_skills():
            fm = _frontmatter(d / "SKILL.md")
            assert fm.get("name") == d.name, "skill name must equal its directory"
            assert _SKILL_NAME_RE.match(d.name)
            assert len(d.name) <= 64
            desc = fm.get("description", "")
            assert 0 < len(desc) <= 1024
            assert re.search(r"\b(use|when|trigger)", desc, re.I), "say when to use it"
            checked += 1
        assert checked >= 1

    def test_no_skill_name_collides_with_a_command(self):
        # A skill and a command of the same name shadow each other in Claude Code.
        commands = {Path(p.name).stem for p in skills.iter_commands()}
        assert not commands & {p.name for p in skills.iter_skills()}

    def test_agents_have_string_name_and_description(self):
        # knowhow only lists a subagent whose frontmatter has both as strings.
        checked = 0
        for p in skills.iter_agents():
            fm = _frontmatter(Path(str(p)))
            assert fm.get("name"), p.name
            assert fm.get("description"), p.name
            checked += 1
        assert checked >= 1


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

    def test_skills_installed_as_directories(self, tmp_path):
        _run_install(tmp_path)
        dest = tmp_path / ".claude" / "skills"
        for src in skills.iter_skills():
            got = dest / src.name / "SKILL.md"
            assert got.is_file()
            assert not (dest / src.name).is_symlink()
            assert _hash(got) == _hash(Path(str(src)) / "SKILL.md")

    def test_dev_mode_links_skill_directories(self, tmp_path):
        _run_install(tmp_path, dev=True)
        for src in skills.iter_skills():
            link = tmp_path / ".claude" / "skills" / src.name
            assert link.is_symlink()
            assert link.resolve() == Path(str(src)).resolve()

    def test_force_replaces_skill_directory(self, tmp_path):
        stale = tmp_path / ".claude" / "skills" / "discopt"
        stale.mkdir(parents=True)
        (stale / "SKILL.md").write_text("stale")
        (stale / "leftover.txt").write_text("x")

        _run_install(tmp_path, force=False)
        assert (stale / "SKILL.md").read_text() == "stale"

        _run_install(tmp_path, force=True)
        assert (stale / "SKILL.md").read_text() != "stale"
        assert not (stale / "leftover.txt").exists()

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
