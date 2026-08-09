"""The CI ``changes`` gate decides whether the heavy test jobs run at all.

When it answers ``code=false`` every solver job below it reports ``skipped`` —
which renders as *not red*. So a wrong ``false`` is not a missing test, it is a
green-looking wall of nothing, the failure mode #953 was filed for. This suite
runs the gate's actual shell script (extracted from ``ci.yml``, not a copy) in a
throwaway git repo and pins its answer per event type.

The case that motivated it: ``github.event.before`` exists only on ``push``, so
on ``workflow_dispatch`` the script's generic branch fell through to
``HEAD~1..HEAD`` and gated the whole matrix on the tip commit's file list. A
manual re-run of #960 came back with rust / python-fast / claim-boundary /
correctness / AMP all skipped.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_CI = _REPO / ".github" / "workflows" / "ci.yml"

pytestmark = pytest.mark.skipif(
    not _CI.exists() or shutil.which("git") is None,
    reason="needs .github/workflows/ci.yml and git",
)

_ZERO_SHA = "0" * 40


def _gate_script() -> str:
    """The `changes` job's filter step, read out of the workflow itself."""
    wf = yaml.safe_load(_CI.read_text())
    steps = wf["jobs"]["changes"]["steps"]
    for step in steps:
        if step.get("id") == "filter":
            return step["run"]
    raise AssertionError("no `filter` step in the `changes` job — did it get renamed?")


def _render(script: str, *, event: str, before: str, sha: str, base_ref: str) -> str:
    """Substitute the ``${{ github.* }}`` expressions GitHub would expand."""
    subs = {
        "github.event_name": event,
        "github.event.before": before,
        "github.sha": sha,
        "github.base_ref": base_ref,
    }
    out = script
    for key, val in subs.items():
        out = re.sub(r"\$\{\{\s*" + re.escape(key) + r"\s*\}\}", val, out)
    leftover = re.findall(r"\$\{\{.*?\}\}", out)
    assert not leftover, f"unsubstituted workflow expressions: {leftover}"
    return out


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def _commit(repo: Path, files: list[str], msg: str) -> str:
    for rel in files:
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"{msg}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", msg)
    return _git(repo, "rev-parse", "HEAD")


def _make_repo(tmp_path: Path, tip_files: list[str]) -> tuple[Path, str, str]:
    """A two-commit repo; the second commit touches ``tip_files``."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "seed.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "seed")
    base = _git(repo, "rev-parse", "HEAD")
    return repo, base, _commit(repo, tip_files, "tip")


def _make_pr_repo(tmp_path: Path) -> tuple[Path, str, str]:
    """A PR-shaped repo: base, then a solver change, then a docs-only tip.

    The real change is two commits back, so a gate that diffs ``HEAD~1..HEAD``
    cannot see it — which is the whole point of diffing against the merge base.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "feature")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "seed.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "seed")
    base = _git(repo, "rev-parse", "HEAD")
    _commit(repo, ["python/discopt/solver.py"], "solver change")
    tip = _commit(repo, ["docs/notes.md"], "docs tip")
    return repo, base, tip


def _run_gate(repo: Path, script: str) -> str:
    """Execute the gate; return its ``code=`` output."""
    out_file = repo / "gh_output"
    out_file.write_text("")
    proc = subprocess.run(
        ["bash", "-c", script],
        cwd=repo,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "GITHUB_OUTPUT": str(out_file),
            "HOME": str(repo),
        },
    )
    written = out_file.read_text()
    assert proc.returncode == 0, f"gate script failed: {proc.stderr[-2000:]}"
    m = re.search(r"^code=(true|false)$", written, re.M)
    assert m, f"gate wrote no code= line.\nstdout:\n{proc.stdout}\nGITHUB_OUTPUT:\n{written}"
    return m.group(1)


# ----------------------------------------------------------------------
# The regression: a manual run must never gate on a guessed diff
# ----------------------------------------------------------------------


def test_workflow_dispatch_always_runs_everything(tmp_path):
    """The bug: docs-only tip commit + dispatch => the whole matrix skipped.

    ``github.event.before`` is empty on ``workflow_dispatch``; the tip commit here
    touches only allowlisted paths, which is exactly the #960 re-run shape
    (CHANGELOG.md + docs/dev/data/claim-baseline.jsonl).
    """
    repo, _base, sha = _make_repo(tmp_path, ["CHANGELOG.md", "docs/dev/data/claim-baseline.jsonl"])
    script = _render(_gate_script(), event="workflow_dispatch", before="", sha=sha, base_ref="main")
    assert _run_gate(repo, script) == "true"


def test_workflow_dispatch_runs_everything_even_on_a_code_tip(tmp_path):
    repo, _base, sha = _make_repo(tmp_path, ["python/discopt/solver.py"])
    script = _render(_gate_script(), event="workflow_dispatch", before="", sha=sha, base_ref="main")
    assert _run_gate(repo, script) == "true"


# ----------------------------------------------------------------------
# An unknown base is the documented "uncertainty" case: run
# ----------------------------------------------------------------------


@pytest.mark.parametrize("before", ["", _ZERO_SHA, "not-a-sha", "deadbeef" * 5])
def test_push_with_an_unusable_base_runs_everything(tmp_path, before):
    """First push of a branch, force-push, or a shallow clone: base unknown.

    The gate promises `code=true` on uncertainty; substituting `HEAD~1..HEAD` is
    a guess, not a resolution, and on a docs-only tip it silently skips.
    """
    repo, _base, sha = _make_repo(tmp_path, ["CHANGELOG.md"])
    script = _render(_gate_script(), event="push", before=before, sha=sha, base_ref="main")
    assert _run_gate(repo, script) == "true"


def test_pull_request_with_no_merge_base_runs_everything(tmp_path):
    """`pull_request` is the trigger that fires on nearly every change here.

    ``git merge-base origin/<base_ref> HEAD`` yields nothing when the ref is
    missing — the `git fetch` above it is `|| true`, so a fetch failure is
    silent, and a shallow fetch boundary hides a far-back merge base the same
    way. Falling back to `HEAD~1..HEAD` then gates the whole matrix on the tip
    commit: this repo's tip is docs-only while the solver change sits a commit
    behind it, so every solver job would be skipped on a PR that changes the
    solver — not red, just absent.
    """
    repo, _base, tip = _make_pr_repo(tmp_path)
    script = _render(_gate_script(), event="pull_request", before="", sha=tip, base_ref="main")
    assert _run_gate(repo, script) == "true"


def test_pull_request_with_a_merge_base_sees_past_the_tip_commit(tmp_path):
    """Control for the test above, and the behaviour that must be preserved.

    Same repo, same event, same rendered script — only `origin/main` is made
    resolvable. The gate must diff from the merge base, so it sees the solver
    change two commits back even though the tip is docs-only.
    """
    repo, base, tip = _make_pr_repo(tmp_path)
    _git(repo, "update-ref", "refs/remotes/origin/main", base)
    script = _render(_gate_script(), event="pull_request", before="", sha=tip, base_ref="main")
    assert _run_gate(repo, script) == "true"


def test_pull_request_docs_only_against_its_merge_base_skips(tmp_path):
    """A genuinely docs-only PR must still skip — the gate's reason for being."""
    repo, base, tip = _make_repo(tmp_path, ["docs/notes.md", "README.md"])
    _git(repo, "update-ref", "refs/remotes/origin/main", base)
    script = _render(_gate_script(), event="pull_request", before="", sha=tip, base_ref="main")
    assert _run_gate(repo, script) == "false"


# ----------------------------------------------------------------------
# The behaviour that must NOT change
# ----------------------------------------------------------------------


def test_push_docs_only_still_skips(tmp_path):
    """A real base with only allowlisted files: skipping is the whole point."""
    repo, base, sha = _make_repo(tmp_path, ["CHANGELOG.md", "docs/notes.md"])
    script = _render(_gate_script(), event="push", before=base, sha=sha, base_ref="main")
    assert _run_gate(repo, script) == "false"


def test_push_with_a_code_file_runs(tmp_path):
    repo, base, sha = _make_repo(tmp_path, ["docs/notes.md", "python/discopt/solver.py"])
    script = _render(_gate_script(), event="push", before=base, sha=sha, base_ref="main")
    assert _run_gate(repo, script) == "true"


def test_schedule_still_skips_the_pr_matrix(tmp_path):
    """The nightly lane does not gate on this output; the rest must not re-run."""
    repo, _base, sha = _make_repo(tmp_path, ["python/discopt/solver.py"])
    script = _render(_gate_script(), event="schedule", before="", sha=sha, base_ref="main")
    assert _run_gate(repo, script) == "false"
