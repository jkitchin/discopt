"""
LLM provider — thin wrapper around litellm for universal model access.

Model string examples:
  - ``"anthropic/claude-sonnet-5"``
  - ``"openai/gpt-4o"``
  - ``"gemini/gemini-pro"``
  - ``"ollama/llama3"``
  - ``"bedrock/anthropic.claude-3-sonnet"``

Configuration priority:
  1. Explicit ``model=`` parameter
  2. ``DISCOPT_LLM_MODEL`` environment variable
  3. Default: ``"anthropic/claude-sonnet-5"``

Request timeouts follow the same shape via ``DISCOPT_LLM_TIMEOUT`` (seconds);
see :func:`resolve_timeout`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "anthropic/claude-sonnet-5"

#: Fallback request timeout, in seconds, for a plain completion.
DEFAULT_TIMEOUT = 30.0

#: Fallback request timeout, in seconds, for a tool-calling completion. Higher
#: than :data:`DEFAULT_TIMEOUT` because tool schemas make for longer prompts.
DEFAULT_TOOL_TIMEOUT = 60.0

#: Environment variable overriding every call site's default timeout.
TIMEOUT_ENV_VAR = "DISCOPT_LLM_TIMEOUT"


def _get_model(model: str | None = None) -> str:
    """Resolve model string from argument, env var, or default."""
    if model is not None:
        return model
    return os.environ.get("DISCOPT_LLM_MODEL", DEFAULT_MODEL)


def resolve_timeout(timeout: float | None = None, default: float = DEFAULT_TIMEOUT) -> float:
    """Resolve a request timeout in seconds.

    Priority: explicit ``timeout`` argument, then the ``DISCOPT_LLM_TIMEOUT``
    environment variable, then ``default``.

    Call sites inside :mod:`discopt.llm` pass their own ``default`` here rather
    than a bare literal, so that one environment variable raises all of them.
    That matters for local backends: the short defaults (5 s for streaming B&B
    commentary, so LLM chatter cannot stall a solve) are tuned for a hosted API
    and are not enough for a model running on the same machine.

    Parameters
    ----------
    timeout : float, optional
        Explicit timeout in seconds. Wins over the environment variable when
        given, so a caller can still ask for a *shorter* timeout than the
        environment sets.
    default : float, default :data:`DEFAULT_TIMEOUT`
        Timeout to use when neither the argument nor the environment supplies
        one.

    Returns
    -------
    float
        The resolved timeout in seconds.

    Notes
    -----
    A malformed or non-positive environment value is ignored with a warning
    rather than raising: a typo in a shell profile should not take down an
    otherwise working solve.
    """
    if timeout is not None:
        return timeout

    raw = os.environ.get(TIMEOUT_ENV_VAR)
    if raw is None:
        return default

    try:
        value = float(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a number; using the default of %gs.", TIMEOUT_ENV_VAR, raw, default
        )
        return default

    if value <= 0:
        logger.warning(
            "%s=%r is not positive; using the default of %gs.", TIMEOUT_ENV_VAR, raw, default
        )
        return default

    return value


def _is_timeout(exc: BaseException) -> bool:
    """Whether ``exc`` (or anything it wraps) is a request timeout.

    Matches on type name as well as :class:`TimeoutError` so that provider SDK
    timeout classes are recognized without importing litellm's optional
    provider dependencies just to name them.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TimeoutError) or "timeout" in type(current).__name__.lower():
            return True
        current = current.__cause__ or current.__context__
    return False


def _failure_message(kind: str, exc: Exception, timeout: float) -> str:
    """Build the error text for a failed call, with timeout guidance."""
    if not _is_timeout(exc):
        return f"{kind} failed: {exc}"
    return (
        f"{kind} timed out after {timeout:g}s: {exc}\n"
        f"Increase the limit by setting {TIMEOUT_ENV_VAR} (seconds), e.g. "
        f"`export {TIMEOUT_ENV_VAR}={max(120, int(timeout * 4))}`, or pass "
        f"timeout= to this call. Local models served through ollama typically "
        f"need considerably more than the default {DEFAULT_TIMEOUT:g}s."
    )


def complete(
    messages: list[dict[str, str]],
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    timeout: float | None = None,
    **kwargs,
) -> str:
    r"""Send a completion request via litellm.

    Parameters
    ----------
    messages : list of dict
        Chat messages in OpenAI format ``[{"role": "...", "content": "..."}]``.
    model : str, optional
        LLM model string. See module docstring for examples.
    max_tokens : int, default 2048
        Maximum tokens in response.
    temperature : float, default 0.0
        Sampling temperature (0 = deterministic).
    timeout : float, optional
        Request timeout in seconds. Defaults to ``DISCOPT_LLM_TIMEOUT`` if set,
        else :data:`DEFAULT_TIMEOUT`. See :func:`resolve_timeout`.
    \*\*kwargs
        Additional arguments forwarded to ``litellm.completion()``.

    Returns
    -------
    str
        The text content of the LLM response.

    Raises
    ------
    ImportError
        If litellm is not installed.
    RuntimeError
        If the LLM call fails. A timeout failure names ``DISCOPT_LLM_TIMEOUT``
        and the value it would take to raise the limit.
    """
    try:
        import litellm
    except ImportError:
        raise ImportError(
            "litellm is required for LLM features. Install it with: pip install discopt[llm]"
        ) from None

    resolved_model = _get_model(model)
    resolved_timeout = resolve_timeout(timeout, DEFAULT_TIMEOUT)
    logger.debug(
        "LLM request: model=%s, messages=%d, timeout=%gs",
        resolved_model,
        len(messages),
        resolved_timeout,
    )

    try:
        response = litellm.completion(
            model=resolved_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=resolved_timeout,
            **kwargs,
        )
        content = response.choices[0].message.content
        logger.debug("LLM response: %d chars", len(content) if content else 0)
        return content or ""
    except Exception as e:
        message = _failure_message("LLM call", e, resolved_timeout)
        logger.warning("%s", message)
        raise RuntimeError(message) from e


def complete_with_tools(
    messages: list[dict[str, str]],
    tools: list[dict],
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout: float | None = None,
    **kwargs,
) -> Any:
    r"""Send a completion request with tool calling via litellm.

    Parameters
    ----------
    messages : list of dict
        Chat messages in OpenAI format.
    tools : list of dict
        Tool definitions in OpenAI function-calling format.
    model : str, optional
        LLM model string.
    max_tokens : int, default 4096
        Maximum tokens in response.
    temperature : float, default 0.0
        Sampling temperature.
    timeout : float, optional
        Request timeout in seconds. Defaults to ``DISCOPT_LLM_TIMEOUT`` if set,
        else :data:`DEFAULT_TOOL_TIMEOUT`. See :func:`resolve_timeout`.
    \*\*kwargs
        Additional arguments forwarded to ``litellm.completion()``.

    Returns
    -------
    Any
        The LLM response message object (may contain tool_calls).

    Raises
    ------
    ImportError
        If litellm is not installed.
    RuntimeError
        If the LLM call fails. A timeout failure names ``DISCOPT_LLM_TIMEOUT``
        and the value it would take to raise the limit.
    """
    try:
        import litellm
    except ImportError:
        raise ImportError(
            "litellm is required for LLM features. Install it with: pip install discopt[llm]"
        ) from None

    resolved_model = _get_model(model)
    resolved_timeout = resolve_timeout(timeout, DEFAULT_TOOL_TIMEOUT)
    logger.debug(
        "LLM tool request: model=%s, tools=%d, timeout=%gs",
        resolved_model,
        len(tools),
        resolved_timeout,
    )

    try:
        response = litellm.completion(
            model=resolved_model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=resolved_timeout,
            **kwargs,
        )
        return response.choices[0].message
    except Exception as e:
        message = _failure_message("LLM tool call", e, resolved_timeout)
        logger.warning("%s", message)
        raise RuntimeError(message) from e
