"""Where the MINLPLib benchmark corpus lives — resolved once, never hardcoded.

Measurement harnesses must not read the corpus out of a **synced folder**. The
canonical snapshot lives in Dropbox (deliberately: it is the backed-up copy), but
Dropbox is a shared resource under someone else's control — its indexer can wake at
any moment for reasons unrelated to a benchmark run, and a 926 MB tree of ``.nl``
files is exactly the kind of thing it wakes for. A timing panel that reads through it
is measuring the machine *and* the sync daemon.

That confound was observed directly (Dropbox at 121 % CPU during a timing run on
2026-07-27). Two hypotheses for what triggered it were tested and **both falsified**:

* *"our solvers write ``.sol`` files next to the ``.nl`` inputs"* — measured: 0 files
  modified anywhere in the tree in 24 h, exactly 1 ``.sol`` present and stale since
  Jul 21, and the BARON harness already stages into ``tempfile.mkdtemp`` with
  ``cwd=work``. We do not write there.
* *"reads trigger re-indexing"* — no evidence; reads do not generate FSEvents.

So the spike was most likely Dropbox's own background work *coinciding* with a run.
That is the strongest argument for the mirror rather than a weaker one: the trigger is
outside our control and unpredictable, so the fix is to remove the dependency, not to
explain it. The mirror also removes a real future hazard — the one stale ``.sol``
proves a solver *has* written into the corpus at least once, and any harness that
forgets to stage would silently reintroduce a sync cascade mid-panel.

Resolution order (first hit wins):

1. ``$DISCOPT_MINLP_BENCH`` — explicit override, for CI or an alternate snapshot.
2. ``~/projects/discopt-minlp-benchmark`` — the local mirror (not synced). Refresh
   with ``scripts/refresh_benchmark_mirror.sh``.
3. ``~/Dropbox/projects/discopt-minlp-benchmark`` — the canonical snapshot, used only
   when no mirror exists so nothing breaks on a fresh checkout.

``corpus_is_synced()`` reports whether the resolved root sits under a sync folder, so
a timing harness can warn (or refuse) rather than silently publish a contaminated
number.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV = "DISCOPT_MINLP_BENCH"
_MIRROR = Path.home() / "projects" / "discopt-minlp-benchmark"
_CANONICAL = Path.home() / "Dropbox" / "projects" / "discopt-minlp-benchmark"

# A root only counts if it actually carries the oracle; a half-synced or empty
# directory must not shadow a good one (a mirror that silently resolved to an empty
# tree would make every oracle check vacuous, which is the failure mode this repo
# has hit more than once).
_SENTINEL = "minlplib.solu"


def _usable(root: Path) -> bool:
    return (root / _SENTINEL).is_file()


def corpus_root() -> Path | None:
    """Resolved corpus root, or ``None`` when no usable snapshot is installed."""
    raw = os.environ.get(_ENV)
    if raw:
        p = Path(os.path.expanduser(raw))
        return p if _usable(p) else None
    for cand in (_MIRROR, _CANONICAL):
        if _usable(cand):
            return cand
    return None


def nl_dir() -> Path | None:
    """Directory holding the ``.nl`` instances, or ``None``."""
    root = corpus_root()
    if root is None:
        return None
    d = root / "minlplib" / "nl"
    return d if d.is_dir() else None


def nl_path(name: str) -> Path | None:
    """Path to a named instance (with or without the ``.nl`` suffix), or ``None``."""
    d = nl_dir()
    if d is None:
        return None
    p = d / (name if name.endswith(".nl") else f"{name}.nl")
    return p if p.is_file() else None


def solu_path() -> Path | None:
    """Path to ``minlplib.solu`` under the resolved root, or ``None``."""
    root = corpus_root()
    if root is None:
        return None
    p = root / _SENTINEL
    return p if p.is_file() else None


def corpus_is_synced() -> bool:
    """True when the resolved root sits inside a known sync folder.

    Timing harnesses should treat this as a contaminated-measurement warning: the
    numbers are then a function of the sync daemon's mood as much as the solver's.
    """
    root = corpus_root()
    if root is None:
        return False
    parts = {p.lower() for p in root.parts}
    return bool(parts & {"dropbox", "google drive", "onedrive", "icloud drive"})


def warn_if_synced(context: str = "measurement") -> bool:
    """Print a loud warning when measuring against a synced corpus. Returns True if synced."""
    if not corpus_is_synced():
        return False
    print(
        f"WARNING: {context} is reading the corpus from a SYNCED folder "
        f"({corpus_root()}). Wall-clock numbers may reflect the sync daemon, not the "
        f"solver. Create the local mirror with scripts/refresh_benchmark_mirror.sh.",
        flush=True,
    )
    return True


__all__ = [
    "corpus_root",
    "nl_dir",
    "nl_path",
    "solu_path",
    "corpus_is_synced",
    "warn_if_synced",
]
