"""Claude Code skills and agents shipped with discopt.

The project's shareable slash commands (``commands/*.md``), agent personas
(``agents/*.md``) and Agent Skills (``skills/<name>/SKILL.md``) live here so
they ship as package data. The repository root links ``commands/``,
``agents/`` and ``skills/`` to these directories, which is the layout
`knowhow <https://github.com/jkitchin/knowhow>`_ and ``npx skills`` discover,
so ``uvx knowhow add jkitchin/discopt --all`` installs the same files.

Use :func:`discopt.cli._cmd_install_skills` (exposed as
``discopt install-skills``) to copy or symlink them into
``~/.claude/`` or the current project's ``.claude/`` directory.

This module exposes small helpers for locating the shipped files via
:mod:`importlib.resources`; it does not import Claude Code or modify any
external state.

Examples
--------
>>> from discopt.skills import iter_commands
>>> [p.name for p in iter_commands()]          # doctest: +ELLIPSIS
['benchmark-report.md', 'convert.md', ...]
"""

from __future__ import annotations

import sys
from importlib.resources import files
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    # ``Traversable`` appears here only in annotations, and PEP 563 (the
    # ``__future__`` import above) makes those strings -- so this never executes
    # at runtime. It has to be guarded because ``importlib.resources.abc`` is
    # 3.11+, while ``requires-python`` promises 3.10: until #1055 this was an
    # unconditional import, and ``import discopt.skills`` (hence
    # ``discopt install-skills``) raised ModuleNotFoundError on 3.10. On 3.10 the
    # same class is reachable as ``importlib.abc.Traversable``.
    if sys.version_info >= (3, 11):
        from importlib.resources.abc import Traversable
    else:
        from importlib.abc import Traversable

_PACKAGE = "discopt.skills"


def commands_dir() -> Traversable:
    """Return a :class:`Traversable` rooted at the shipped commands/ directory."""
    return files(_PACKAGE) / "commands"


def agents_dir() -> Traversable:
    """Return a :class:`Traversable` rooted at the shipped agents/ directory."""
    return files(_PACKAGE) / "agents"


def skills_dir() -> Traversable:
    """Return a :class:`Traversable` rooted at the shipped skills/ directory."""
    return files(_PACKAGE) / "skills"


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


def iter_skills() -> Iterator[Traversable]:
    """Yield every skill directory (one containing ``SKILL.md``) under
    :func:`skills_dir`, sorted by name."""
    return iter(
        sorted(
            (p for p in skills_dir().iterdir() if p.is_dir() and (p / "SKILL.md").is_file()),
            key=lambda p: p.name,
        )
    )


__all__ = [
    "agents_dir",
    "commands_dir",
    "iter_agents",
    "iter_commands",
    "iter_skills",
    "skills_dir",
]
