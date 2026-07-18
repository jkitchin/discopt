"""Unit tests for the dependents notification script (issue #414).

Covers the pure manifest/payload logic in ``.github/scripts/notify_dependents.py``
and asserts it stays in sync with the checked-in ``.github/dependents.yml``.
No network is touched.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "notify_dependents.py"
MANIFEST = REPO_ROOT / ".github" / "dependents.yml"


def _load_module():
    spec = importlib.util.spec_from_file_location("notify_dependents", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass introspection can resolve the module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


nd = _load_module()


def test_parse_manifest_applies_defaults():
    deps = nd.parse_manifest(
        """
        dependents:
          - repo: owner/only-repo
          - repo: owner/full
            private: true
            dispatch: false
            notify: false
            event_type: custom-event
        """
    )
    assert len(deps) == 2
    d0, d1 = deps
    assert d0.repo == "owner/only-repo"
    assert d0.private is False
    assert d0.dispatch is True and d0.notify is True
    assert d0.event_type == nd.DEFAULT_EVENT_TYPE
    assert d1.private is True
    assert d1.dispatch is False and d1.notify is False
    assert d1.event_type == "custom-event"


def test_parse_manifest_empty_list():
    assert nd.parse_manifest("dependents: []") == []


@pytest.mark.parametrize(
    "text",
    [
        "not_a_mapping",
        "other_key: 1",  # missing 'dependents'
        "dependents:\n  - private: true",  # missing 'repo'
        "dependents:\n  - repo: no-slash",  # malformed repo
        "dependents:\n  - repo: too/many/slashes",
    ],
)
def test_parse_manifest_rejects_malformed(text):
    with pytest.raises(ValueError):
        nd.parse_manifest(text)


def test_dispatch_payload_shape():
    payload = nd.dispatch_payload("jkitchin/discopt", "v1.2.3", "https://rel", "https://run")
    assert payload["source"] == "jkitchin/discopt"
    assert payload["release_tag"] == "v1.2.3"
    assert payload["release_url"] == "https://rel"
    assert payload["run_url"] == "https://run"


def test_issue_title_and_body():
    assert nd.issue_title("v1.2.3") == "Review / update against discopt v1.2.3"
    body = nd.issue_body("jkitchin/discopt", "v1.2.3", "https://rel", "https://run")
    assert "v1.2.3" in body
    assert "https://rel" in body
    assert "https://run" in body
    assert "#414" in body


def test_checked_in_manifest_parses():
    deps = nd.parse_manifest(MANIFEST.read_text(encoding="utf-8"))
    assert deps, "dependents.yml should list at least one dependent"
    repos = {d.repo for d in deps}
    # The repos named in issue #414 must be tracked.
    assert {"jkitchin/discopt-doe", "jkitchin/discopt-mkm", "jkitchin/discopt-aggregation"} <= repos
    for d in deps:
        assert d.repo.count("/") == 1
