"""Tests for LLM request-timeout resolution and timeout feedback.

Covers ``DISCOPT_LLM_TIMEOUT``: the precedence rules in
:func:`discopt.llm.provider.resolve_timeout`, tolerance of a malformed value,
and the requirement that a timeout failure tell the user how to raise the
limit rather than just reporting that it expired.
"""

from __future__ import annotations

import logging

import pytest
from discopt.llm import provider
from discopt.llm.provider import (
    DEFAULT_TIMEOUT,
    DEFAULT_TOOL_TIMEOUT,
    TIMEOUT_ENV_VAR,
    _failure_message,
    _is_timeout,
    resolve_timeout,
)


@pytest.fixture(autouse=True)
def _clear_timeout_env(monkeypatch):
    """Keep the ambient environment from deciding these tests."""
    monkeypatch.delenv(TIMEOUT_ENV_VAR, raising=False)


class TestResolveTimeout:
    def test_default_when_unset(self):
        assert resolve_timeout() == DEFAULT_TIMEOUT
        assert resolve_timeout(default=5.0) == 5.0

    def test_env_var_overrides_default(self, monkeypatch):
        monkeypatch.setenv(TIMEOUT_ENV_VAR, "300")
        assert resolve_timeout() == 300.0
        # The point of routing call sites through here: one variable raises
        # even the short per-site defaults.
        assert resolve_timeout(default=5.0) == 300.0

    def test_explicit_argument_beats_env_var(self, monkeypatch):
        monkeypatch.setenv(TIMEOUT_ENV_VAR, "300")
        assert resolve_timeout(2.0) == 2.0

    @pytest.mark.parametrize("bad", ["", "abc", "30s", "nan-ish"])
    def test_malformed_env_falls_back_with_warning(self, monkeypatch, caplog, bad):
        monkeypatch.setenv(TIMEOUT_ENV_VAR, bad)
        with caplog.at_level(logging.WARNING, logger=provider.__name__):
            assert resolve_timeout(default=7.0) == 7.0
        assert TIMEOUT_ENV_VAR in caplog.text

    @pytest.mark.parametrize("bad", ["0", "-1", "-12.5"])
    def test_nonpositive_env_falls_back_with_warning(self, monkeypatch, caplog, bad):
        monkeypatch.setenv(TIMEOUT_ENV_VAR, bad)
        with caplog.at_level(logging.WARNING, logger=provider.__name__):
            assert resolve_timeout(default=7.0) == 7.0
        assert TIMEOUT_ENV_VAR in caplog.text

    def test_float_env_value(self, monkeypatch):
        monkeypatch.setenv(TIMEOUT_ENV_VAR, "12.5")
        assert resolve_timeout() == 12.5


class TestIsTimeout:
    def test_builtin_timeout_error(self):
        assert _is_timeout(TimeoutError("nope"))

    def test_matches_provider_sdk_class_by_name(self):
        # litellm/openai raise APITimeoutError, which we must not import just
        # to name it.
        exc_type = type("APITimeoutError", (Exception,), {})
        assert _is_timeout(exc_type("slow"))

    def test_matches_wrapped_cause(self):
        try:
            try:
                raise TimeoutError("inner")
            except TimeoutError as inner:
                raise RuntimeError("outer") from inner
        except RuntimeError as outer:
            assert _is_timeout(outer)

    def test_rejects_unrelated_error(self):
        assert not _is_timeout(ValueError("bad key"))

    def test_terminates_on_cyclic_cause(self):
        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__cause__ = b
        b.__cause__ = a
        assert not _is_timeout(a)  # must return, not hang


class TestFailureMessage:
    def test_timeout_message_names_env_var_and_a_value(self):
        msg = _failure_message("LLM call", TimeoutError("expired"), 30.0)
        assert TIMEOUT_ENV_VAR in msg
        assert "30s" in msg
        assert f"export {TIMEOUT_ENV_VAR}=120" in msg

    def test_suggested_value_scales_with_current_timeout(self):
        msg = _failure_message("LLM call", TimeoutError("expired"), 100.0)
        assert f"export {TIMEOUT_ENV_VAR}=400" in msg

    def test_non_timeout_message_omits_guidance(self):
        msg = _failure_message("LLM call", ValueError("invalid api key"), 30.0)
        assert TIMEOUT_ENV_VAR not in msg
        assert "invalid api key" in msg


class TestProviderWiring:
    """The resolved timeout must actually reach litellm, and surface on failure."""

    @staticmethod
    def _fake_litellm(monkeypatch, captured, raises=None):
        litellm = pytest.importorskip("litellm")

        def fake_completion(**kwargs):
            captured.update(kwargs)
            if raises is not None:
                raise raises
            return type(
                "R",
                (),
                {"choices": [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]},
            )()

        monkeypatch.setattr(litellm, "completion", fake_completion)

    def test_complete_uses_env_timeout(self, monkeypatch):
        monkeypatch.setenv(TIMEOUT_ENV_VAR, "250")
        captured: dict = {}
        self._fake_litellm(monkeypatch, captured)
        provider.complete([{"role": "user", "content": "hi"}], model="openai/gpt-4o")
        assert captured["timeout"] == 250.0

    def test_complete_defaults_without_env(self, monkeypatch):
        captured: dict = {}
        self._fake_litellm(monkeypatch, captured)
        provider.complete([{"role": "user", "content": "hi"}], model="openai/gpt-4o")
        assert captured["timeout"] == DEFAULT_TIMEOUT

    def test_complete_with_tools_uses_its_own_default(self, monkeypatch):
        captured: dict = {}
        self._fake_litellm(monkeypatch, captured)
        provider.complete_with_tools(
            [{"role": "user", "content": "hi"}], tools=[], model="openai/gpt-4o"
        )
        assert captured["timeout"] == DEFAULT_TOOL_TIMEOUT

    def test_timeout_failure_tells_user_how_to_raise_the_limit(self, monkeypatch):
        captured: dict = {}
        self._fake_litellm(monkeypatch, captured, raises=TimeoutError("request timed out"))
        with pytest.raises(RuntimeError, match=TIMEOUT_ENV_VAR):
            provider.complete([{"role": "user", "content": "hi"}], model="openai/gpt-4o")

    def test_non_timeout_failure_is_unchanged(self, monkeypatch):
        captured: dict = {}
        self._fake_litellm(monkeypatch, captured, raises=ValueError("invalid api key"))
        with pytest.raises(RuntimeError, match="invalid api key") as excinfo:
            provider.complete([{"role": "user", "content": "hi"}], model="openai/gpt-4o")
        assert TIMEOUT_ENV_VAR not in str(excinfo.value)
