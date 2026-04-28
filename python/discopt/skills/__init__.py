"""Claude Code skills and agents shipped with discopt.

The project's shareable slash commands (``.claude/commands/*.md``) and
agent personas (``.claude/agents/*.md``) live here so they ship as package
data. Use :func:`discopt.cli._cmd_install_skills` (exposed as
``discopt install-skills``) to copy or symlink them into
``~/.claude/`` or the current project's ``.claude/`` directory.

This module exposes small helpers for locating the shipped files via
:mod:`importlib.resources`; it does not import Claude Code or modify any
external state.

Examples
--------
>>> from discopt.skills import iter_commands
>>> [p.name for p in iter_commands()]          # doctest: +ELLIPSIS
['adversary.md', 'benchmark-report.md', ...]
"""

from __future__ import annotations

from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import Iterator

_PACKAGE = "discopt.skills"


def commands_dir() -> Traversable:
    """Return a :class:`Traversable` rooted at the shipped commands/ directory."""
    return files(_PACKAGE) / "commands"


def agents_dir() -> Traversable:
    """Return a :class:`Traversable` rooted at the shipped agents/ directory."""
    return files(_PACKAGE) / "agents"


def iter_commands() -> Iterator[Traversable]:
    """Yield every ``*.md`` file under :func:`commands_dir`, sorted by name."""
    return iter(
        sorted(
            (p for p in commands_dir().iterdir() if p.name.endswith(".md")), key=lambda p: p.name
        )
    )


def iter_agents() -> Iterator[Traversable]:
    """Yield every ``*.md`` file under :func:`agents_dir`, sorted by name."""
    return iter(
        sorted((p for p in agents_dir().iterdir() if p.name.endswith(".md")), key=lambda p: p.name)
    )


__all__ = ["agents_dir", "commands_dir", "iter_agents", "iter_commands"]
