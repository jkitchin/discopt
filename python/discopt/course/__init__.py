"""Packaged ``discopt`` course content.

The course tree (lesson notebooks, syllabus, rubrics, progress template,
and the ``/course:`` Claude-Code slash commands under
``_claude_assets/``) ships as package data so ``discopt tutor`` works
out-of-the-box for pip-installed users.

For mutable state (a student's own ``progress.yaml``) the tutor
materializes a writable copy of this tree into the user's working
directory via ``discopt tutor install`` — the packaged copy stays
read-only.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def package_root() -> Path:
    """Return the on-disk path to the packaged ``course/`` tree.

    Resolves through :mod:`importlib.resources` so it works whether
    ``discopt`` is installed in editable mode, as a wheel, or from sdist.
    """
    return Path(str(files("discopt.course")))
