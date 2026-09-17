"""``provenance._git_head`` must record the checkout's own commit (#1291).

Three defects: a relative ``gitdir:`` was resolved against the process's working
directory (a submodule recorded whatever repository the process ran in); a linked
worktree on a branch recorded nothing because ``commondir`` was ignored; and a
document holding a bare ``NaN`` token loaded, then failed the next save.
"""

import json
import subprocess

import discopt.modeling as dm
import discopt.serialize as ser
import pytest
from discopt.provenance import _git_head

SUB = "1" * 40
DECOY = "d" * 40


def _bare_repo(gitdir, sha):
    (gitdir / "refs" / "heads").mkdir(parents=True)
    (gitdir / "HEAD").write_text("ref: refs/heads/main\n")
    (gitdir / "refs" / "heads" / "main").write_text(sha + "\n")


def test_relative_gitdir_is_relative_to_the_dot_git_file(tmp_path, monkeypatch):
    _bare_repo(tmp_path / "super" / ".git" / "modules" / "sub", SUB)
    _bare_repo(tmp_path / "decoy" / ".git" / "modules" / "sub", DECOY)
    (tmp_path / "super" / "sub" / "pkg").mkdir(parents=True)
    (tmp_path / "super" / "sub" / ".git").write_text("gitdir: ../.git/modules/sub\n")
    (tmp_path / "decoy" / "cwd").mkdir(parents=True)
    root = tmp_path / "super" / "sub" / "pkg"

    monkeypatch.chdir(tmp_path / "decoy" / "cwd")
    assert _git_head(root) == SUB
    monkeypatch.chdir("/")
    assert _git_head(root) == SUB


def _worktree_layout(tmp_path, branch_store):
    main = tmp_path / "main" / ".git"
    main.mkdir(parents=True)
    (main / "HEAD").write_text("ref: refs/heads/main\n")
    if branch_store == "loose":
        (main / "refs" / "heads").mkdir(parents=True)
        (main / "refs" / "heads" / "topic").write_text("e" * 40 + "\n")
    else:
        (main / "packed-refs").write_text(f"{'e' * 40} refs/heads/topic\n")
    wt_git = main / "worktrees" / "wt"
    wt_git.mkdir(parents=True)
    (wt_git / "HEAD").write_text("ref: refs/heads/topic\n")
    (wt_git / "commondir").write_text("../..\n")
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / ".git").write_text(f"gitdir: {wt_git}\n")
    return wt


@pytest.mark.parametrize("branch_store", ["loose", "packed"])
def test_linked_worktree_on_a_branch_reads_the_common_dir(tmp_path, branch_store):
    assert _git_head(_worktree_layout(tmp_path, branch_store)) == "e" * 40


def test_non_object_id_is_not_recorded(tmp_path):
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("not-a-commit\n")
    assert _git_head(tmp_path) is None


def test_real_linked_worktree_matches_rev_parse(tmp_path):
    def git(*args, cwd):
        return subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        ).stdout.strip()

    repo = tmp_path / "repo"
    repo.mkdir()
    git("init", "-q", "-b", "main", cwd=repo)
    (repo / "f").write_text("x")
    git("add", "f", cwd=repo)
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "c", cwd=repo)
    git("worktree", "add", "-q", "-b", "topic", str(tmp_path / "wt"), cwd=repo)
    (tmp_path / "wt" / "g").write_text("y")
    git("add", "g", cwd=tmp_path / "wt")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "d", cwd=tmp_path / "wt")
    expected = git("rev-parse", "HEAD", cwd=tmp_path / "wt")
    assert expected != git("rev-parse", "HEAD", cwd=repo)
    assert _git_head(tmp_path / "wt") == expected
    git("pack-refs", "--all", cwd=repo)
    assert _git_head(tmp_path / "wt") == expected


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_bare_non_finite_token_is_refused_on_load(token):
    m = dm.Model("n")
    x = m.continuous("x", lb=0, ub=1)
    m.minimize(x)
    doc = json.loads(ser.dumps(m))
    doc["provenance"]["extra"] = "__TOKEN__"
    text = json.dumps(doc).replace('"__TOKEN__"', token)
    with pytest.raises(ser.SerializationError, match="bare"):
        ser.loads(text)
