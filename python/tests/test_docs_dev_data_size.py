"""Guard: ``docs/dev/data`` holds measurement summaries, not raw solver traces.

#2169 landed 174 MB of per-arm solver logs (single files of ~255k lines) under
``docs/dev/data``, an 18x jump over everything else in that directory; they were
gzipped and then removed (the results they support are recorded in each
directory's README / findings). Every clone and worktree checks this directory
out, so the evidence that belongs here is the ``.jsonl`` rows, summaries and the
scripts that regenerate them. A raw trace stays in the scratchpad, or is cut to
the lines a document actually cites (e.g. a ``*_tail.log``).

Both limits sit above every file tracked when this guard was added (largest data
file 1.1 MB, largest log 33 kB), so this fails only on a new oversized file.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

REPO = Path(__file__).resolve().parents[2]
DATA = "docs/dev/data"
MAX_FILE_BYTES = 2_000_000  # any one tracked file
MAX_LOG_BYTES = 100_000  # a raw .log / .log.gz trace


def _tracked_files() -> list[Path]:
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", DATA],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    paths = [REPO / p for p in out.split("\0") if p]
    # Prove the probe fired (CLAUDE.md §6): the directory is never empty.
    assert len(paths) > 100, f"expected the tracked {DATA} corpus, found {len(paths)} files"
    return [p for p in paths if p.is_file()]


def _is_log(p: Path) -> bool:
    return p.name.endswith(".log") or p.name.endswith(".log.gz")


def test_no_oversized_file_in_docs_dev_data():
    big = {
        str(p.relative_to(REPO)): p.stat().st_size
        for p in _tracked_files()
        if p.stat().st_size > MAX_FILE_BYTES
    }
    assert not big, f"files over {MAX_FILE_BYTES} bytes in {DATA}: {big}"


def test_no_raw_solver_log_in_docs_dev_data():
    files = _tracked_files()
    logs = [p for p in files if _is_log(p)]
    assert logs, "expected some small tracked logs (tails, panel logs) to check"
    big = {
        str(p.relative_to(REPO)): p.stat().st_size for p in logs if p.stat().st_size > MAX_LOG_BYTES
    }
    assert not big, (
        f"raw solver logs over {MAX_LOG_BYTES} bytes in {DATA}: {big}. Keep the trace out of "
        "the repo; commit the .jsonl/summary it supports, or a tail of the cited lines."
    )
