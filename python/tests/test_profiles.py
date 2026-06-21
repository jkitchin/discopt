"""Tests for the ``discopt solve`` named-profile mechanism."""

from __future__ import annotations

import pytest
from discopt import profiles as P

pytestmark = pytest.mark.unit


def test_builtin_profiles_present():
    profs = P.load_profiles()
    for name in ("default", "fast", "exact", "feasible"):
        assert name in profs
    assert profs["fast"]["time_limit"] == 10.0
    assert profs["fast"]["tuning"]["node_bound_mode"] == "lp"


def test_resolve_precedence_cli_over_profile(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no user discopt.toml here
    # CLI override wins over the profile's value.
    opts = P.resolve_options("fast", {"time_limit": 99.0})
    assert opts["time_limit"] == 99.0
    assert opts["gap_tolerance"] == 1e-2  # untouched profile value survives


def test_resolve_tuning_merges_key_wise(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # fast sets node_bound_mode + node_nlp_stride; override only one field.
    opts = P.resolve_options("fast", {"tuning": {"node_nlp_stride": 2}})
    assert opts["tuning"]["node_nlp_stride"] == 2  # overridden
    assert opts["tuning"]["node_bound_mode"] == "lp"  # preserved from profile


def test_unknown_profile_raises(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(KeyError, match="unknown profile"):
        P.resolve_options("nope", {})


def test_user_toml_overrides_and_adds(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "discopt.toml").write_text(
        "[profiles.fast]\ntime_limit = 5\n\n[profiles.mine]\nsolver = 'amp'\n"
    )
    profs = P.load_profiles()
    assert profs["fast"]["time_limit"] == 5  # cwd file overrode the built-in
    assert profs["mine"]["solver"] == "amp"  # new user profile
    opts = P.resolve_options("mine", {})
    assert opts["solver"] == "amp"


def test_no_profile_just_overrides(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    opts = P.resolve_options(None, {"time_limit": 7.0})
    assert opts == {"time_limit": 7.0}
