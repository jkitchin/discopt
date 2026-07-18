#!/usr/bin/env python3
"""Fan out discopt release events to dependent repositories (issue #414).

Reads the registry at ``.github/dependents.yml`` and, for each dependent:

* sends a ``repository_dispatch`` event (so the dependent's CI re-runs against
  the just-released discopt), and/or
* opens a "review / update against discopt vX.Y.Z" tracking issue.

Driven by ``.github/workflows/notify-dependents.yml``. The network-touching code
is deliberately thin; the payload/manifest logic is pure and unit-tested in
``python/tests/test_notify_dependents.py``.

Environment variables:
    GH_TOKEN        GitHub token with dispatch + issue write on the dependents.
    RELEASE_TAG     Release tag (e.g. ``v1.2.3``); falls back to "unreleased".
    RELEASE_URL     URL of the release (optional, embedded in the issue body).
    SOURCE_REPO     ``owner/name`` of discopt (default ``jkitchin/discopt``).
    RUN_URL         URL of the triggering Actions run (optional).
    MANIFEST        Path to the registry (default ``.github/dependents.yml``).
    DRY_RUN         "true" to log intended actions without calling the API.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

API_ROOT = "https://api.github.com"
DEFAULT_EVENT_TYPE = "discopt-updated"
DEFAULT_SOURCE_REPO = "jkitchin/discopt"


@dataclass(frozen=True)
class Dependent:
    """One row of the dependents registry, with defaults applied."""

    repo: str
    private: bool = False
    dispatch: bool = True
    notify: bool = True
    event_type: str = DEFAULT_EVENT_TYPE
    note: str = ""


def parse_manifest(text: str) -> list[Dependent]:
    """Parse the dependents.yml contents into a list of ``Dependent``.

    Raises ``ValueError`` on a malformed manifest so misconfiguration fails
    loudly rather than silently skipping notifications.
    """
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict) or "dependents" not in data:
        raise ValueError("manifest must be a mapping with a 'dependents' key")
    raw = data["dependents"] or []
    if not isinstance(raw, list):
        raise ValueError("'dependents' must be a list")

    out: list[Dependent] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict) or "repo" not in entry:
            raise ValueError(f"dependent #{i} is missing required 'repo' field")
        repo = str(entry["repo"]).strip()
        if repo.count("/") != 1 or repo.startswith("/") or repo.endswith("/"):
            raise ValueError(f"dependent #{i} 'repo' must be 'owner/name', got {repo!r}")
        out.append(
            Dependent(
                repo=repo,
                private=bool(entry.get("private", False)),
                dispatch=bool(entry.get("dispatch", True)),
                notify=bool(entry.get("notify", True)),
                event_type=str(entry.get("event_type", DEFAULT_EVENT_TYPE)),
                note=str(entry.get("note", "")),
            )
        )
    return out


def dispatch_payload(
    source_repo: str, release_tag: str, release_url: str, run_url: str
) -> dict[str, Any]:
    """Build the ``client_payload`` sent with a repository_dispatch event."""
    return {
        "source": source_repo,
        "release_tag": release_tag,
        "release_url": release_url,
        "run_url": run_url,
        "reason": "discopt release published",
    }


def issue_title(release_tag: str) -> str:
    return f"Review / update against discopt {release_tag}"


def issue_body(source_repo: str, release_tag: str, release_url: str, run_url: str) -> str:
    lines = [
        f"discopt **{release_tag}** was released. Please review this package for",
        "compatibility and update pins / code as needed.",
        "",
        "This issue was opened automatically from the discopt release pipeline",
        f"([{source_repo}](https://github.com/{source_repo}), issue #414).",
        "",
        "**Checklist**",
        "- [ ] CI passes against the new discopt release",
        "- [ ] Version pin updated if required",
        "- [ ] Changelog / docs reviewed for breaking changes",
    ]
    if release_url:
        lines += ["", f"Release notes: {release_url}"]
    if run_url:
        lines += [f"Triggering run: {run_url}"]
    return "\n".join(lines)


class GitHub:
    """Minimal GitHub REST client over stdlib urllib."""

    def __init__(self, token: str) -> None:
        self._token = token

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        url = f"{API_ROOT}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self._token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        req.add_header("User-Agent", "discopt-notify-dependents")
        with urllib.request.urlopen(req) as resp:  # noqa: S310 (fixed api.github.com host)
            payload = resp.read()
        return json.loads(payload) if payload else None

    def send_dispatch(self, repo: str, event_type: str, client_payload: dict[str, Any]) -> None:
        self._request(
            "POST",
            f"/repos/{repo}/dispatches",
            {"event_type": event_type, "client_payload": client_payload},
        )

    def find_open_issue(self, repo: str, title: str) -> int | None:
        """Return the number of an existing open issue with ``title``, else None."""
        issues = self._request("GET", f"/repos/{repo}/issues?state=open&per_page=100") or []
        for issue in issues:
            # /issues also returns PRs; skip those and match the exact title.
            if "pull_request" not in issue and issue.get("title") == title:
                return int(issue["number"])
        return None

    def create_issue(self, repo: str, title: str, body: str) -> int:
        created = self._request("POST", f"/repos/{repo}/issues", {"title": title, "body": body})
        return int(created["number"])


def _summary(line: str) -> None:
    print(line)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def main() -> int:
    manifest_path = Path(os.environ.get("MANIFEST", ".github/dependents.yml"))
    release_tag = os.environ.get("RELEASE_TAG") or "unreleased"
    release_url = os.environ.get("RELEASE_URL", "")
    source_repo = os.environ.get("SOURCE_REPO") or DEFAULT_SOURCE_REPO
    run_url = os.environ.get("RUN_URL", "")
    dry_run = os.environ.get("DRY_RUN", "").lower() == "true"

    dependents = parse_manifest(manifest_path.read_text(encoding="utf-8"))
    active = [d for d in dependents if d.dispatch or d.notify]
    if not active:
        _summary("No dependents to notify.")
        return 0

    token = os.environ.get("GH_TOKEN", "")
    if not token and not dry_run:
        print(
            "::error::GH_TOKEN is not set. Configure the DEPENDENTS_DISPATCH_TOKEN "
            "secret (see docs/dev/dependents.md).",
            file=sys.stderr,
        )
        return 1

    gh = GitHub(token) if not dry_run else None
    payload = dispatch_payload(source_repo, release_tag, release_url, run_url)
    title = issue_title(release_tag)
    body = issue_body(source_repo, release_tag, release_url, run_url)

    failures = 0
    _summary(f"### Notify dependents — discopt {release_tag}\n")
    for dep in active:
        actions: list[str] = []
        if dep.dispatch:
            try:
                if dry_run:
                    actions.append(f"would dispatch `{dep.event_type}`")
                else:
                    assert gh is not None
                    gh.send_dispatch(dep.repo, dep.event_type, payload)
                    actions.append(f"dispatched `{dep.event_type}`")
            except (urllib.error.HTTPError, urllib.error.URLError) as exc:
                failures += 1
                actions.append(f"dispatch FAILED ({exc})")
                print(f"::warning title=dispatch failed::{dep.repo}: {exc}", file=sys.stderr)
        if dep.notify:
            try:
                if dry_run:
                    actions.append("would open review issue")
                else:
                    assert gh is not None
                    existing = gh.find_open_issue(dep.repo, title)
                    if existing is not None:
                        actions.append(f"review issue exists (#{existing})")
                    else:
                        num = gh.create_issue(dep.repo, title, body)
                        actions.append(f"opened review issue (#{num})")
            except (urllib.error.HTTPError, urllib.error.URLError) as exc:
                failures += 1
                actions.append(f"issue FAILED ({exc})")
                print(f"::warning title=issue failed::{dep.repo}: {exc}", file=sys.stderr)
        _summary(f"- **{dep.repo}**: {', '.join(actions)}")

    if failures:
        # Per-repo failures are warnings, not a hard job failure: one unreachable
        # private repo must not red the release. Surface the count for visibility.
        print(f"::warning::{failures} dependent notification(s) failed.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
