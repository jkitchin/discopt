"""
LLM provider — thin wrapper around litellm for universal model access.

Model string examples:
  - ``"anthropic/claude-sonnet-4-20250514"``
  - ``"openai/gpt-4o"``
  - ``"gemini/gemini-pro"``
  - ``"ollama/llama3"``
  - ``"bedrock/anthropic.claude-3-sonnet"``

Configuration priority:
  1. Explicit ``model=`` parameter
  2. ``DISCOPT_LLM_MODEL`` environment variable
  3. Default: ``"anthropic/claude-sonnet-4-20250514"``
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "anthropic/claude-sonnet-4-20250514"


def _get_model(model: str | None = None) -> str:
    """Resolve model string from argument, env var, or default."""
    if model is not None:
        return model
    return os.environ.get("DISCOPT_LLM_MODEL", DEFAULT_MODEL)


def complete(
    messages: list[dict[str, str]],
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    timeout: float = 30.0,
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
    timeout : float, default 30.0
        Request timeout in seconds.
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
        If the LLM call fails after retries.
    """
    try:
        import litellm
    except ImportError:
        raise ImportError(
            "litellm is required for LLM features. Install it with: pip install discopt[llm]"
        ) from None

    resolved_model = _get_model(model)
    logger.debug("LLM request: model=%s, messages=%d", resolved_model, len(messages))

    try:
        response = litellm.completion(
            model=resolved_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            **kwargs,
        )
        content = response.choices[0].message.content
        logger.debug("LLM response: %d chars", len(content) if content else 0)
        return content or ""
    except Exception as e:
        logger.warning("LLM call failed: %s", e)
        raise RuntimeError(f"LLM call failed: {e}") from e


def complete_with_tools(
    messages: list[dict[str, str]],
    tools: list[dict],
    model: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    timeout: float = 60.0,
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
    timeout : float, default 60.0
        Request timeout in seconds.
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
    """
    try:
        import litellm
    except ImportError:
        raise ImportError(
            "litellm is required for LLM features. Install it with: pip install discopt[llm]"
        ) from None

    resolved_model = _get_model(model)
    logger.debug("LLM tool request: model=%s, tools=%d", resolved_model, len(tools))

    try:
        response = litellm.completion(
            model=resolved_model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            **kwargs,
        )
        return response.choices[0].message
    except Exception as e:
        logger.warning("LLM tool call failed: %s", e)
        raise RuntimeError(f"LLM tool call failed: {e}") from e
