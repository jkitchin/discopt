"""
discopt.llm -- Optional LLM integration for model explanation, formulation, and advisory.

Requires ``litellm``: install via ``pip install discopt[llm]``.

All LLM features degrade gracefully when litellm is not installed —
the solver runs identically without LLM features.

Configuration
-------------
LLM provider and model are configured via:

1. Explicit ``model=`` parameter on each function
2. ``DISCOPT_LLM_MODEL`` environment variable
3. Default: ``"anthropic/claude-sonnet-4-20250514"``

The API key is set via the provider's standard environment variable:

- Anthropic: ``ANTHROPIC_API_KEY``
- OpenAI: ``OPENAI_API_KEY``
- Google: ``GEMINI_API_KEY``
- Ollama: no key needed (local)
- See litellm docs for other providers

Submodules
----------
provider
    Thin wrapper around litellm for universal model access.
prompts
    Centralized, versionable prompt templates.
serializer
    Serialize Model/SolveResult to LLM-friendly text.
safety
    Output validation and correctness guards.
tools
    Tool definitions for from_description() structured output.
advisor
    Solver strategy advisor and pre-solve analysis.
commentary
    Streaming B&B commentary.
diagnosis
    Infeasibility diagnosis and result analysis.
chat
    Conversational model building (``discopt.chat()``).
reformulation
    Auto-reformulation engine.
"""

from __future__ import annotations


def is_available() -> bool:
    """Check if litellm is installed and LLM features are available."""
    try:
        import litellm  # noqa: F401

        return True
    except ImportError:
        return False


def get_completion(
    messages: list[dict[str, str]],
    model: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    timeout: float = 30.0,
    **kwargs,
) -> str:
    r"""Get a completion from the configured LLM provider.

    Thin convenience wrapper around :func:`discopt.llm.provider.complete`.

    Parameters
    ----------
    messages : list of dict
        Chat messages in OpenAI format (role/content dicts).
    model : str, optional
        LLM model string (e.g. ``"anthropic/claude-sonnet-4-20250514"``).
        Falls back to ``DISCOPT_LLM_MODEL`` env var, then default.
    max_tokens : int, default 2048
        Maximum tokens in response.
    temperature : float, default 0.0
        Sampling temperature.
    timeout : float, default 30.0
        Request timeout in seconds.
    \*\*kwargs
        Additional arguments passed to litellm.

    Returns
    -------
    str
        The LLM response text.

    Raises
    ------
    ImportError
        If litellm is not installed.
    """
    from discopt.llm.provider import complete

    return complete(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        **kwargs,
    )
