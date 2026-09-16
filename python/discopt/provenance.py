"""FAIR provenance metadata for saved models: what wrote a document, and when.

Why
---
The native format (:mod:`discopt.serialize`) used to carry two provenance fields:
the schema identifier and ``discopt.__version__``. The version was *written but
never read* -- :func:`discopt.serialize.loads` validated only the schema major, so
a document written by one discopt and reloaded by another produced no signal at
all, and the reloaded :class:`~discopt.modeling.core.Model` could not say which
version wrote it. Neither was there a timestamp, so a ``.dopt`` on disk could not
be ordered against the solve log that produced it.

That is the gap this module fills. A saved model is a research artifact: the
point of saving one is to come back to it -- or to hand it to a reviewer -- and
reproduce the number. Reproducing it needs to know which solver built it, which
is a claim about the Rust core and the interpreter as much as about the Python
package version.

What is recorded
----------------
``created``
    UTC ISO-8601 timestamp, second resolution.
``software``
    The writing software's identity: ``name``, ``version``
    (``discopt.__version__``), ``rust_core`` (the version compiled into the
    ``discopt._rust`` extension -- the Expression IR, ``.nl`` parser and LP
    layer live there, so it can move independently of the Python version),
    ``repository`` and ``license``.
``git_commit``
    HEAD of the checkout the package was imported from, when it *is* a checkout
    (an editable / development install). See the caveat on :func:`_git_head`:
    this is HEAD, and says nothing about uncommitted changes.
``source_fingerprint``
    ``__version__`` plus the newest mtime across the installed package tree --
    the same key :mod:`discopt._daemon_core` uses to evict a stale daemon. This
    is what *does* move when an editable checkout is edited without committing,
    so it complements ``git_commit`` rather than duplicating it.
``platform``
    ``python`` (the interpreter version), ``implementation``, ``system`` and
    ``machine``. A result can depend on all four through floating-point details.
``author``
    Optional, and never inferred. See below.

On authorship
-------------
``author`` is recorded only when the caller passes it (``Model.save(...,
author=...)``) or sets ``DISCOPT_PROVENANCE_AUTHOR`` in the environment. It is
deliberately NOT read from the repository's ``CITATION.cff``: that file names the
author of *discopt*, and a saved model is the user's artifact, not discopt's.
Copying discopt's author onto a stranger's model would be a false attribution --
precisely the kind of metadata error FAIR provenance exists to prevent. The
``software`` block is where discopt's own identity belongs.

Nothing here is read back into the model's mathematics. Provenance is descriptive
metadata: :func:`discopt.serialize.loads` attaches it to ``Model.provenance`` and
warns on a version mismatch, and that is the whole of its effect.
"""

from __future__ import annotations

import os
import platform as _platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

#: Stable identity of the writing software. Mirrors ``pyproject.toml``'s
#: ``license`` and ``project.urls.Repository``.
_REPOSITORY = "https://github.com/jkitchin/discopt"
_LICENSE = "EPL-2.0"

__all__ = ["capture", "skew_warning"]


def _rust_core_version() -> Optional[str]:
    """Version compiled into the Rust extension, or ``None`` if it is absent.

    Only :class:`ImportError` is treated as "absent" -- a ``version()`` that
    raises anything else is a real fault in the extension and propagates rather
    than being recorded as a missing version.
    """
    try:
        from discopt import _rust
    except ImportError:
        return None
    return str(_rust.version())


def _git_head(root: Path) -> Optional[str]:
    """HEAD of the git checkout containing *root*, read from ``.git`` directly.

    No subprocess: ``git rev-parse`` on a large or network-mounted repository can
    take hundreds of milliseconds or hang, and this runs on every save.

    CAVEAT, and it is the reason the field is named ``git_commit`` rather than
    anything implying a clean tree: this is the commit HEAD points at. Detecting
    uncommitted changes needs a full index comparison, which is exactly the
    subprocess this avoids. ``source_fingerprint`` is the field that moves when an
    editable checkout is edited, so the pair is honest where either alone would
    not be.
    """
    for parent in [root, *root.parents]:
        git = parent / ".git"
        if not git.exists():
            continue
        # A worktree or submodule has `.git` as a file pointing at the real dir.
        if git.is_file():
            try:
                line = git.read_text(encoding="utf-8").strip()
            except OSError:
                return None
            if not line.startswith("gitdir:"):
                return None
            git = Path(line.split(":", 1)[1].strip())
            if not git.is_dir():
                return None
        try:
            head = (git / "HEAD").read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if not head.startswith("ref:"):
            # Detached HEAD: the file holds the commit itself.
            return head or None
        ref = head.split(":", 1)[1].strip()
        try:
            return (git / ref).read_text(encoding="utf-8").strip() or None
        except OSError:
            pass
        # A packed ref (`git gc` moves loose refs into `packed-refs`).
        try:
            packed = (git / "packed-refs").read_text(encoding="utf-8")
        except OSError:
            return None
        for entry in packed.splitlines():
            if entry.startswith(("#", "^")):
                continue
            parts = entry.split(None, 1)
            if len(parts) == 2 and parts[1].strip() == ref:
                return parts[0]
        return None
    return None


def capture(*, author: Optional[str] = None, previous: Optional[dict] = None) -> dict[str, Any]:
    """Build the provenance block written into a saved document.

    Parameters
    ----------
    author : str, optional
        Creator of the *model*. Falls back to ``DISCOPT_PROVENANCE_AUTHOR``, and
        is omitted entirely when neither is set -- never inferred from the
        repository or the host's user account.
    previous : dict, optional
        Provenance of the document this model was loaded from, recorded under
        ``derived_from``. ``created`` always describes the document being written
        *now* -- it would be a lie otherwise -- so without this a load-edit-save
        cycle would erase the record of what the model came from, which is the
        chain the block exists to preserve.

        Exactly ONE level is kept: *previous*'s own ``derived_from`` is dropped,
        so a model round-tripped a thousand times does not carry a thousand-deep
        block. That is a real loss of the deeper history and is stated here rather
        than left for someone to discover -- the full chain lives in the files
        themselves, each of which retains its own immediate predecessor.
    """
    import discopt

    root = Path(discopt.__file__).resolve().parent

    # `_FINGERPRINT`, not `_source_fingerprint()`: the constant is computed when
    # `_daemon_core` is first imported, and re-walking the package tree here costs
    # ~7 ms per save (measured) for a value that is also slightly *wrong* -- a file
    # edited after import changes what the function returns but not the code that
    # is actually running and writing this document.
    from discopt._daemon_core import _FINGERPRINT

    block: dict[str, Any] = {
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "software": {
            "name": "discopt",
            "version": getattr(discopt, "__version__", None),
            "rust_core": _rust_core_version(),
            "repository": _REPOSITORY,
            "license": _LICENSE,
        },
        "source_fingerprint": _FINGERPRINT,
        "platform": {
            "python": _platform.python_version(),
            "implementation": _platform.python_implementation(),
            "system": _platform.system(),
            "machine": _platform.machine(),
        },
    }

    commit = _git_head(root)
    if commit is not None:
        block["git_commit"] = commit

    who = author if author is not None else os.environ.get("DISCOPT_PROVENANCE_AUTHOR")
    if who:
        block["author"] = str(who)

    if isinstance(previous, dict):
        block["derived_from"] = {k: v for k, v in previous.items() if k != "derived_from"}

    return block


def skew_warning(saved: Optional[dict]) -> Optional[str]:
    """Message describing a version difference between *saved* and this discopt.

    Returns ``None`` when the versions agree, when there is no provenance block
    (a document written before this format carried one -- legitimately silent,
    not worth a warning on every load), or when the block records no version.

    Both the Python version and the Rust core version are compared: a solve's
    numbers can move with either, and the Rust core is the one that is easy to
    forget because it is not what ``pip show`` reports.
    """
    if not saved:
        return None
    software = saved.get("software")
    if not isinstance(software, dict):
        return None

    import discopt

    differences = []
    saved_py = software.get("version")
    current_py = getattr(discopt, "__version__", None)
    if saved_py and saved_py != current_py:
        differences.append(f"discopt {saved_py} -> {current_py}")

    saved_rust = software.get("rust_core")
    current_rust = _rust_core_version()
    if saved_rust and current_rust and saved_rust != current_rust:
        differences.append(f"Rust core {saved_rust} -> {current_rust}")

    if not differences:
        return None
    created = saved.get("created")
    when = f" on {created}" if created else ""
    return (
        f"this model was saved{when} by a different discopt ({'; '.join(differences)}). "
        "The model is read as written; this is a provenance notice, not a conversion. "
        "Re-solving under a different version can give different numbers -- see "
        "Model.provenance for the full record."
    )
